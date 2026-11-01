"""Missing or empty bronze_weather_forecast (spec 019b B3, AC6): 01b and 06."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
INGEST = ROOT / "notebooks" / "01b_ingest_weather_forecast.py"
TRAIN = ROOT / "notebooks" / "06_train_lgbm.py"
PATHS = SimpleNamespace(table_weather_forecast="weather", table_silver="silver")


def _extract(path: Path, *names: str, **namespace) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        n for n in tree.body if isinstance(n, ast.FunctionDef | ast.ClassDef) and n.name in names
    ]
    ns = {"pd": pd, "PATHS": PATHS, **namespace}
    exec(compile(tree, str(path), "exec"), ns)
    return {name: ns[name] for name in names}


def _top_level_call_lines(path: Path, name: str) -> list[int]:
    """Line numbers of top-level statements that call `name`."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        stmt.lineno
        for stmt in tree.body
        for node in ast.walk(stmt)
        if isinstance(node, ast.Call)
        and (getattr(node.func, "id", None) == name or getattr(node.func, "attr", None) == name)
    ]


class _Writer:
    def __init__(self, log):
        self.log = log

    def format(self, fmt):
        self.log.append(("format", fmt))
        return self

    def saveAsTable(self, name):
        self.log.append(("saveAsTable", name))


class _Spark:
    def __init__(self, tables):
        self.tables = tables
        self.log = []
        self.catalog = SimpleNamespace(tableExists=lambda name: name in self.tables)

    def createDataFrame(self, rows, schema):
        return SimpleNamespace(write=_Writer(self.log))

    @property
    def read(self):
        return SimpleNamespace(table=self.table)

    def table(self, name):
        frame = self.tables[name]
        return SimpleNamespace(
            toPandas=lambda: frame.copy(),
            select=lambda *cols: SimpleNamespace(toPandas=lambda: frame[list(cols)].copy()),
        )


# ───────────── 01b: the table exists after every run ─────────────


def test_ac6_01b_creates_the_table_when_missing():
    ensure_table = _extract(INGEST, "ensure_table", FILE_SCHEMA="schema")["ensure_table"]
    spark = _Spark({})

    assert ensure_table(spark) is True
    assert ("saveAsTable", "weather") in spark.log


def test_ac6_01b_leaves_an_existing_table_alone():
    ensure_table = _extract(INGEST, "ensure_table", FILE_SCHEMA="schema")["ensure_table"]
    spark = _Spark({"weather": pd.DataFrame()})

    assert ensure_table(spark) is False
    assert spark.log == []


def test_ac6_01b_creates_the_table_before_it_looks_for_files():
    ensure = _top_level_call_lines(INGEST, "ensure_table")
    exits = _top_level_call_lines(INGEST, "exit")

    assert ensure and exits and min(ensure) < min(exits)


# ───────────── 06: a clear error before any MLflow run ─────────────


def _read_inputs():
    fns = _extract(TRAIN, "MissingWeatherForecasts", "read_inputs")
    return fns["read_inputs"], fns["MissingWeatherForecasts"]


def test_ac6_06_missing_table_stops_with_a_clear_error():
    read_inputs, missing = _read_inputs()

    with pytest.raises(missing, match="weather does not exist. Run backfill.yml"):
        read_inputs(_Spark({"silver": pd.DataFrame({"timestamp": [], "value_mwh": []})}))


def test_ac6_06_empty_table_stops_with_a_clear_error():
    read_inputs, missing = _read_inputs()
    spark = _Spark(
        {
            "silver": pd.DataFrame({"timestamp": [], "value_mwh": []}),
            "weather": pd.DataFrame({"timestamp": []}),
        }
    )

    with pytest.raises(missing, match="weather is empty"):
        read_inputs(spark)


def test_ac6_06_reads_inputs_before_any_mlflow_run():
    reads = _top_level_call_lines(TRAIN, "read_inputs")
    runs = _top_level_call_lines(TRAIN, "start_run")

    assert reads and runs and max(reads) < min(runs)


def test_06_reads_tz_aware_inputs_when_both_tables_have_rows():
    read_inputs, _ = _read_inputs()
    ts = pd.date_range("2026-10-01", periods=3, freq="h")
    spark = _Spark(
        {
            "silver": pd.DataFrame({"timestamp": ts, "value_mwh": [1.0, 2.0, 3.0]}),
            "weather": pd.DataFrame({"timestamp": ts, "temperature_fc": [5.0, 6.0, 7.0]}),
        }
    )

    load, weather = read_inputs(spark)

    assert load["timestamp"].dt.tz is not None and weather["timestamp"].dt.tz is not None
