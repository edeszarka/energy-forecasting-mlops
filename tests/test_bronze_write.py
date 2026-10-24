"""Bronze write rules and their use in 01_ingest / 00_backfill_bronze (spec 014 T2, T3).

AC2: the temperature MERGE updates every weather column; weather-only batches
are ingested. AC3: the backfill replaces bronze over the range, NULLs included.
Notebook functions are extracted via ``ast`` as in
``tests/test_predict_timing_integration.py``.
"""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.bronze_write import (
    BRONZE_LOAD_COLUMNS,
    BRONZE_TEMPERATURE_COLUMNS,
    LOAD_MERGE_COLUMNS,
    LOAD_MERGE_CONDITION,
    TEMPERATURE_MERGE_COLUMNS,
    TEMPERATURE_MERGE_CONDITION,
    merge_set,
    monthly_row_counts,
    prepare_backfill_load,
    prepare_backfill_temperature,
    replace_where_predicate,
    should_skip_ingest,
)
from src.data_quality import DataQualityError, check_hourly_rows, check_load_quality_by_month

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
START = pd.Timestamp("2024-01-01T00:00Z")
END = pd.Timestamp("2024-01-15T00:00Z")
ZONE = "10YHU-MAVIR----U"


def _load_notebook_functions(notebook: str, *names: str, **namespace) -> dict:
    path = NOTEBOOKS / notebook
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"pd": pd, "logger": logging.getLogger("bronze-test"), **namespace}
    exec(compile(tree, str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def raw_load(start=START, end=END) -> pd.DataFrame:
    ts = pd.date_range(start, end, freq="h", inclusive="left")
    hours = np.arange(len(ts))
    return pd.DataFrame(
        {
            "timestamp": ts,
            "country": ZONE,
            "value_mwh": 5000 + 1200 * np.sin(2 * np.pi * hours / 24),
            "source": "entsoe_api",
            "fetched_at": pd.Timestamp("2026-10-25T00:00Z"),
        }
    )


def raw_temp(start=START, end=END) -> pd.DataFrame:
    ts = pd.date_range(start, end, freq="h", inclusive="left")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "temperature_c": 2.0,
            "humidity_pct": 80.0,
            "cloud_cover_pct": 50.0,
            "is_weather_imputed": False,
            "source": "openmeteo",
            "fetched_at": pd.Timestamp("2026-10-25T00:00Z"),
        }
    )


# --- T2: hourly MERGE -------------------------------------------------------


class _FakeMerge:
    def __init__(self, calls):
        self.calls = calls

    def alias(self, name):
        return self

    def merge(self, source, condition):
        self.calls.append(("merge", condition))
        return self

    def whenMatchedUpdate(self, condition, set):
        self.calls.append(("update", condition, set))
        return self

    def whenNotMatchedInsertAll(self):
        self.calls.append(("insert_all",))
        return self

    def execute(self):
        self.calls.append(("execute",))


def _merge_functions():
    return _load_notebook_functions(
        "01_ingest.py",
        "merge_bronze_load",
        "merge_bronze_temperature",
        LOAD_MERGE_COLUMNS=LOAD_MERGE_COLUMNS,
        LOAD_MERGE_CONDITION=LOAD_MERGE_CONDITION,
        TEMPERATURE_MERGE_COLUMNS=TEMPERATURE_MERGE_COLUMNS,
        TEMPERATURE_MERGE_CONDITION=TEMPERATURE_MERGE_CONDITION,
        merge_set=merge_set,
    )


def test_temperature_merge_updates_every_weather_column():
    calls = []
    _merge_functions()["merge_bronze_temperature"](_FakeMerge(calls), _FakeMerge([]))

    update = next(c for c in calls if c[0] == "update")
    assert update[1] == "source.is_weather_imputed = false"
    for col in ("temperature_c", "humidity_pct", "cloud_cover_pct"):
        assert update[2][col] == f"source.{col}"
    assert calls[-1] == ("execute",)


def test_load_merge_never_overwrites_with_null():
    calls = []
    _merge_functions()["merge_bronze_load"](_FakeMerge(calls), _FakeMerge([]))

    update = next(c for c in calls if c[0] == "update")
    assert update[1] == "source.value_mwh IS NOT NULL"
    assert update[2]["value_mwh"] == "source.value_mwh"


@pytest.mark.parametrize(
    ("load_files", "temp_files", "skip"),
    [([], [], True), ([], ["t.json"], False), (["l.json"], [], False), (["l"], ["t"], False)],
)
def test_weather_only_batches_are_ingested(load_files, temp_files, skip):
    assert should_skip_ingest(load_files, temp_files) is skip


# --- T3: backfill preparation -------------------------------------------------


def test_replace_where_predicate():
    assert replace_where_predicate(START, END, ZONE) == (
        "timestamp >= '2024-01-01 00:00:00' AND timestamp < '2024-01-15 00:00:00' "
        "AND country = '10YHU-MAVIR----U'"
    )
    assert "country" not in replace_where_predicate(START, END)


def test_prepare_backfill_load_keeps_nulls_and_latest_fetch():
    raw = raw_load()
    raw.loc[5, "value_mwh"] = np.nan
    newer = raw.iloc[[6]].assign(value_mwh=1.0, fetched_at=pd.Timestamp("2026-10-26T00:00Z"))
    outside = raw.iloc[[0]].assign(timestamp=END)
    raw = pd.concat([raw, newer, outside], ignore_index=True)

    df = prepare_backfill_load(raw, START, END, ZONE, "run-1")

    assert list(df.columns) == list(BRONZE_LOAD_COLUMNS)
    assert len(df) == 14 * 24
    assert df["timestamp"].max() < END
    assert pd.isna(df.loc[5, "value_mwh"]) and df.loc[5, "is_gap"]
    assert df.loc[6, "value_mwh"] == 1.0
    assert set(df["run_id"]) == {"run-1"}


def test_prepare_backfill_temperature_columns():
    raw = raw_temp()
    raw["is_weather_imputed"] = raw["is_weather_imputed"].astype(object)
    raw.loc[3, "is_weather_imputed"] = None

    df = prepare_backfill_temperature(raw, START, END, "run-1")

    assert list(df.columns) == list(BRONZE_TEMPERATURE_COLUMNS)
    assert df["is_weather_imputed"].dtype == bool


def test_monthly_row_counts():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-31T23:00Z", "2024-02-01T00:00Z", "2024-02-01T01:00Z"]
            ),
            "value_mwh": [1.0, None, 2.0],
        }
    )
    counts = monthly_row_counts(df, "value_mwh")
    assert counts.to_dict("records") == [
        {"month": "2024-01", "rows": 1, "non_null": 1},
        {"month": "2024-02", "rows": 2, "non_null": 1},
    ]
    assert list(monthly_row_counts(df.iloc[0:0], "value_mwh").columns) == [
        "month",
        "rows",
        "non_null",
    ]


# --- T3: 00_backfill_bronze write (AC3) ----------------------------------------


class _FakeWriter:
    def __init__(self, spark, frame):
        self.spark, self.frame, self.options = spark, frame, {}

    def format(self, fmt):
        self.options["format"] = fmt
        return self

    def mode(self, mode):
        self.options["mode"] = mode
        return self

    def option(self, key, value):
        self.options[key] = value
        return self

    def saveAsTable(self, table):
        self.spark.writes.append((table, self.frame, self.options))


class _FakeSpark:
    """Records writes; `createDataFrame` keeps the pandas frame for inspection."""

    def __init__(self):
        self.writes = []

    def createDataFrame(self, frame, schema):
        return SimpleNamespace(write=_FakeWriter(self, frame))


def _backfill_functions(bronze_before: pd.DataFrame | None = None):
    schema = lambda cols: SimpleNamespace(fields=[SimpleNamespace(name=c) for c in cols])  # noqa: E731
    return _load_notebook_functions(
        "00_backfill_bronze.py",
        "print_counts",
        "write_bronze_replace",
        "backfill_bronze",
        PATHS=SimpleNamespace(table_bronze="bronze_load", table_bronze_temp="bronze_temperature"),
        ENTSO_E_ZONE=ZONE,
        BRONZE_TABLE_SCHEMA=schema(BRONZE_LOAD_COLUMNS),
        TEMPERATURE_TABLE_SCHEMA=schema(BRONZE_TEMPERATURE_COLUMNS),
        prepare_backfill_load=prepare_backfill_load,
        prepare_backfill_temperature=prepare_backfill_temperature,
        monthly_row_counts=monthly_row_counts,
        replace_where_predicate=replace_where_predicate,
        check_load_quality_by_month=check_load_quality_by_month,
        check_hourly_rows=check_hourly_rows,
        apply_dq_mode=__import__("src.data_quality", fromlist=["apply_dq_mode"]).apply_dq_mode,
        read_bronze_range=lambda spark, table, start, end: (
            bronze_before if bronze_before is not None else pd.DataFrame(columns=["timestamp"])
        ).assign(value_mwh=1.0, temperature_c=1.0),
    )["backfill_bronze"]


def test_backfill_execute_replaces_range_including_nulls():
    raw = raw_load()
    raw.loc[10, "value_mwh"] = np.nan  # a NULL replaces whatever bronze holds
    spark = _FakeSpark()

    _backfill_functions()(spark, raw, raw_temp(), START, END, "execute", "run-1")

    (load_table, load_frame, load_opts), (temp_table, _, temp_opts) = spark.writes
    assert load_table == "bronze_load" and temp_table == "bronze_temperature"
    assert load_opts == {
        "format": "delta",
        "mode": "overwrite",
        "replaceWhere": replace_where_predicate(START, END, ZONE),
    }
    assert temp_opts["replaceWhere"] == replace_where_predicate(START, END)
    assert len(load_frame) == 14 * 24
    assert load_frame["value_mwh"].isna().sum() == 1


def test_backfill_dry_run_prints_counts_and_writes_nothing(capsys):
    before = raw_load().iloc[:100][["timestamp"]]
    spark = _FakeSpark()

    summary = _backfill_functions(before)(spark, raw_load(), raw_temp(), START, END, "dry_run", "r")

    out = capsys.readouterr().out
    assert spark.writes == []
    assert summary == {"mode": "dry_run", "load_rows": 336, "temperature_rows": 336}
    assert "bronze_load: rows per month, before vs files" in out
    assert "rows_before" in out and "rows_files" in out
    assert "Dry run: nothing written." in out


def test_backfill_dq_violation_writes_nothing():
    raw = raw_load().drop(index=range(40, 45))  # missing hours
    spark = _FakeSpark()

    with pytest.raises(DataQualityError, match="complete days without 24 hourly rows"):
        _backfill_functions()(spark, raw, raw_temp(), START, END, "execute", "r")

    assert spark.writes == []
