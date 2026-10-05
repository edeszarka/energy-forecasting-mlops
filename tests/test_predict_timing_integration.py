"""Integration tests for the corrected gold_forecasts timestamp labeling (spec 006 AC2/AC3).

The functions under test live in the Databricks notebook ``notebooks/04_predict.py``,
which cannot be imported directly (module-level ``dbutils``/``spark`` calls). We extract
the function definitions via ``ast`` and exec them with a controlled namespace, following
the mocking pattern established in ``tests/test_ingest_logic.py``.
"""

import ast
import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.config import MODEL_INPUT_FEATURES
from src.forecast_timing import resolve_target_timestamp

# pyspark is not a runtime/test dependency (see pyproject.toml); fall back to a
# permissive mock so the notebook's StructType(...) schema construction still runs.
try:  # pragma: no cover - exercised differently depending on environment
    from pyspark.sql.types import (
        BooleanType,
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )
except ImportError:  # pragma: no cover
    BooleanType = DoubleType = IntegerType = StringType = StructField = StructType = (
        TimestampType
    ) = MagicMock

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "04_predict.py"


def _load_notebook_functions(*names: str) -> dict:
    """Exec the named top-level functions from 04_predict.py in a stubbed namespace."""
    tree = ast.parse(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {
        "hashlib": hashlib,
        "np": np,
        "pd": pd,
        "datetime": datetime,
        "UTC": UTC,
        "MODEL_INPUT_FEATURES": MODEL_INPUT_FEATURES,
        "resolve_target_timestamp": resolve_target_timestamp,
        "logger": logging.getLogger("predict-test"),
        "StructType": StructType,
        "StructField": StructField,
        "StringType": StringType,
        "TimestampType": TimestampType,
        "IntegerType": IntegerType,
        "DoubleType": DoubleType,
        "BooleanType": BooleanType,
        "DeltaTable": _FakeDeltaTable,
    }
    exec(compile(tree, str(NOTEBOOK_PATH), "exec"), namespace)
    return {name: namespace[name] for name in names}


def _make_features(periods: int) -> pd.DataFrame:
    index = pd.date_range(
        start="2026-08-11T00:00:00+00:00", periods=periods, freq="h", tz="UTC", name="timestamp"
    )
    data = {col: np.arange(periods, dtype=float) for col in MODEL_INPUT_FEATURES}
    return pd.DataFrame(data, index=index)


class _FakeLgbmModel:
    """Minimal LightGBM-like regressor: no booster_, predict returns a plain array."""

    def predict(self, x):
        return np.arange(len(x), dtype=float) + 10.0


class _FakeProphetModel:
    """Minimal Prophet-like model that mirrors real Prophet's tz-`ds` rejection.

    Real Prophet raises ``ValueError`` when ``ds`` is tz-aware. Mirroring that
    here makes the Prophet-branch test evaluative: it fails against the pre-fix
    code (which forwarded a tz-aware ``ds`` straight through) and passes only
    once the branch localizes ``ds`` (spec 010 AC2a).
    """

    def predict(self, df):
        if df["ds"].dt.tz is not None:
            raise ValueError(
                "Column ds has timezone specified, which is not supported. Remove timezone."
            )
        return pd.DataFrame({"ds": df["ds"].values, "yhat": np.arange(len(df), dtype=float)})


class _FakeCatalog:
    def tableExists(self, name):
        return True


class _FakeSourceFrame:
    def __init__(self, records):
        self.records = records

    def alias(self, name):
        return self


class _FakeDeltaTable:
    """In-memory stand-in for a Delta table implementing the MERGE contract used here."""

    last_condition = None

    def __init__(self, store):
        self.store = store
        self.source = None
        self.condition = None
        self.update_matched = False
        self.insert_new = False

    @classmethod
    def forName(cls, spark, name):
        return cls(spark._store)

    def alias(self, name):
        return self

    def merge(self, source, condition):
        self.source = source
        self.condition = condition
        _FakeDeltaTable.last_condition = condition
        return self

    def whenMatchedUpdateAll(self):
        self.update_matched = True
        return self

    def whenNotMatchedInsertAll(self):
        self.insert_new = True
        return self

    def execute(self):
        for record in self.source.records:
            key = record["forecast_id"]
            if key in self.store:
                if self.update_matched:
                    self.store[key] = dict(record)
            elif self.insert_new:
                self.store[key] = dict(record)


class _FakeSpark:
    def __init__(self):
        self.catalog = _FakeCatalog()
        self._store = {}

    def createDataFrame(self, df, schema):
        return _FakeSourceFrame(df.to_dict("records"))


def _fid(model_name: str, horizon_hours: int, timestamp: pd.Timestamp) -> str:
    return hashlib.md5(f"{model_name}_{horizon_hours}_{timestamp.isoformat()}".encode()).hexdigest()


# ────────────────────────────── AC2 ──────────────────────────────


def test_prophet_timestamp_and_forecast_id_are_unchanged():
    """Regression guard: Prophet output must be byte-identical to the pre-fix output."""
    generate_forecasts = _load_notebook_functions("generate_forecasts")["generate_forecasts"]
    features = _make_features(24)

    result = generate_forecasts(
        model=_FakeProphetModel(),
        model_name="energy_prophet_24h",
        model_version="1",
        run_id="run-1",
        features_df=features,
        horizon_hours=24,
        forecast_run_at=datetime(2026, 8, 10, tzinfo=UTC),
        config={"pipeline_run_id": "test-run"},
    )

    # Timestamps are the raw anchors, unchanged.
    assert list(result["timestamp"]) == list(features.index)
    # forecast_id is the pre-fix formula, keyed on the (unchanged) anchor.
    expected_ids = [_fid("energy_prophet_24h", 24, anchor) for anchor in features.index]
    assert list(result["forecast_id"]) == expected_ids


def test_prophet_branch_strips_timezone_before_predict():
    """AC2(a): the Prophet branch must call predict() with a tz-naive `ds`.

    `_FakeProphetModel.predict()` raises exactly as real Prophet does when given
    a tz-aware `ds`, so this test fails against the pre-fix code (which passed
    the tz-aware anchor index straight through) and passes only once the branch
    localizes `ds`. Input features are deliberately tz-aware, as
    `prepare_inference_features()` always produces (spec 010 §3.1).
    """
    generate_forecasts = _load_notebook_functions("generate_forecasts")["generate_forecasts"]
    features = _make_features(24)
    assert features.index.tz is not None  # guard: the input really is tz-aware

    result = generate_forecasts(
        model=_FakeProphetModel(),
        model_name="energy_prophet_24h",
        model_version="1",
        run_id="run-prophet-tz",
        features_df=features,
        horizon_hours=24,
        forecast_run_at=datetime(2026, 8, 10, tzinfo=UTC),
        config={"pipeline_run_id": "test-run"},
    )

    assert len(result) == len(features)
    assert result["predicted_mwh"].notna().all()
    assert (result["predicted_mwh"] >= 0).all()
    # Prophet anchors its target on the anchor itself (spec 006 contract).
    assert list(result["timestamp"]) == list(features.index)


@pytest.mark.parametrize("horizon_hours", [24, 168])
def test_lgbm_timestamp_and_forecast_id_use_shifted_target(horizon_hours: int):
    generate_forecasts = _load_notebook_functions("generate_forecasts")["generate_forecasts"]
    features = _make_features(horizon_hours)

    result = generate_forecasts(
        model=_FakeLgbmModel(),
        model_name=f"energy_lgbm_{horizon_hours}h",
        model_version="1",
        run_id="run-2",
        features_df=features,
        horizon_hours=horizon_hours,
        forecast_run_at=datetime(2026, 8, 10, tzinfo=UTC),
        config={"pipeline_run_id": "test-run"},
    )

    shifted = [anchor + pd.Timedelta(hours=horizon_hours) for anchor in features.index]
    assert list(result["timestamp"]) == shifted
    # The bug is what we just fixed: stored timestamp must not be the anchor.
    assert list(result["timestamp"]) != list(features.index)
    expected_ids = [
        _fid(f"energy_lgbm_{horizon_hours}h", horizon_hours, target) for target in shifted
    ]
    assert list(result["forecast_id"]) == expected_ids


def test_lgbm_forecast_id_differs_from_anchor_based_id():
    """Proves the hash input actually changed (would fail against the pre-fix code)."""
    generate_forecasts = _load_notebook_functions("generate_forecasts")["generate_forecasts"]
    features = _make_features(24)

    result = generate_forecasts(
        model=_FakeLgbmModel(),
        model_name="energy_lgbm_24h",
        model_version="1",
        run_id="run-3",
        features_df=features,
        horizon_hours=24,
        forecast_run_at=datetime(2026, 8, 10, tzinfo=UTC),
        config={"pipeline_run_id": "test-run"},
    )

    old_ids = [_fid("energy_lgbm_24h", 24, anchor) for anchor in features.index]
    assert list(result["forecast_id"]) != old_ids


def test_lgbm_branch_unaffected_by_prophet_timezone_fix():
    """AC2(c): the Prophet-path tz strip must not alter the LGBM branch.

    Regression guard for the pre-existing LGBM behavior (shifted target
    timestamps + raw model predictions); complements the parametrized LGBM
    tests above without weakening any of their assertions.
    """
    generate_forecasts = _load_notebook_functions("generate_forecasts")["generate_forecasts"]
    features = _make_features(24)

    result = generate_forecasts(
        model=_FakeLgbmModel(),
        model_name="energy_lgbm_24h",
        model_version="1",
        run_id="run-lgbm-guard",
        features_df=features,
        horizon_hours=24,
        forecast_run_at=datetime(2026, 8, 10, tzinfo=UTC),
        config={"pipeline_run_id": "test-run"},
    )

    shifted = [anchor + pd.Timedelta(hours=24) for anchor in features.index]
    assert list(result["timestamp"]) == shifted
    assert list(result["predicted_mwh"]) == [float(v) + 10.0 for v in range(len(features))]


# ────────────────────────────── AC3 ──────────────────────────────


def _forecast_frame(horizon_hours: int = 24) -> pd.DataFrame:
    generate_forecasts = _load_notebook_functions("generate_forecasts")["generate_forecasts"]
    return generate_forecasts(
        model=_FakeLgbmModel(),
        model_name=f"energy_lgbm_{horizon_hours}h",
        model_version="1",
        run_id="run-merge",
        features_df=_make_features(horizon_hours),
        horizon_hours=horizon_hours,
        forecast_run_at=datetime(2026, 8, 10, tzinfo=UTC),
        config={"pipeline_run_id": "test-run"},
    )


def test_write_forecasts_merges_on_forecast_id_only():
    write_forecasts = _load_notebook_functions("write_forecasts")["write_forecasts"]
    spark = _FakeSpark()

    write_forecasts(_forecast_frame(), spark, {"forecast_table": "gold"}, is_backfill=False)

    assert _FakeDeltaTable.last_condition == "target.forecast_id = source.forecast_id"


def test_write_forecasts_is_idempotent_across_overlapping_runs():
    write_forecasts = _load_notebook_functions("write_forecasts")["write_forecasts"]
    frame = _forecast_frame()
    spark = _FakeSpark()

    write_forecasts(frame, spark, {"forecast_table": "gold"}, is_backfill=False)
    write_forecasts(frame, spark, {"forecast_table": "gold"}, is_backfill=False)

    assert len(spark._store) == len(frame)  # no duplicate insert


def test_write_forecasts_does_not_overwrite_actual_outside_backfill():
    write_forecasts = _load_notebook_functions("write_forecasts")["write_forecasts"]
    frame = _forecast_frame()
    spark = _FakeSpark()

    first = frame.iloc[0].to_dict()
    first["actual_mwh"] = 999.0
    spark._store[first["forecast_id"]] = first

    # Normal path: matched rows are left alone, so the backfilled actual survives.
    write_forecasts(frame, spark, {"forecast_table": "gold"}, is_backfill=False)
    assert spark._store[first["forecast_id"]]["actual_mwh"] == 999.0

    # Backfill path is the only one allowed to update matched rows.
    write_forecasts(frame, spark, {"forecast_table": "gold"}, is_backfill=True)
    assert spark._store[first["forecast_id"]]["actual_mwh"] is None
