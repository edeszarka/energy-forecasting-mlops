"""Integration tests for the corrected gold_forecasts timestamp labeling (spec 006 AC2/AC3).

The functions under test live in the Databricks notebook ``notebooks/04_predict.py``,
which cannot be imported directly (module-level ``dbutils``/``spark`` calls). We extract
the function definitions via ``ast`` and exec them with a controlled namespace, following
the mocking pattern established in ``tests/test_ingest_logic.py``.
"""

import ast
import hashlib
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.baseline import seasonal_naive_forecast
from src.config import MODEL_INPUT_FEATURES
from src.forecast_timing import (
    SERVED_MODEL,
    resolve_target_timestamp,
    serving_anchor_timestamps,
    serving_target_timestamps,
)
from src.gold_rows import combine_gold_rows
from src.promotion import pick_production_run

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


class ModelNotFoundError(Exception):
    """Stand-in for the notebook's class (only FunctionDefs are extracted)."""


def _load_notebook_functions(*names: str, **overrides) -> dict:
    """Exec the named top-level functions from 04_predict.py in a stubbed namespace.

    `overrides` replace namespace entries, e.g. a fake `mlflow` or a stub for a
    notebook function that is deliberately not extracted.
    """
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
        # Spec 013 serving path
        "timedelta": timedelta,
        "MlflowClient": object,
        "ModelNotFoundError": ModelNotFoundError,
        "pick_production_run": pick_production_run,
        "seasonal_naive_forecast": seasonal_naive_forecast,
        "serving_anchor_timestamps": serving_anchor_timestamps,
        "serving_target_timestamps": serving_target_timestamps,
        "SERVED_MODEL": SERVED_MODEL,
        "NAIVE_MODEL_NAME": "seasonal_naive",
        "NAIVE_MODEL_VERSION": "seasonal_naive_t-168h",
        "HUNGARIAN_HOLIDAYS": set(),
        "CURRENT_DATA_VERSION": "2",
        # Spec 017 shadow scoring
        "combine_gold_rows": combine_gold_rows,
        **overrides,
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
    def __init__(self, columns=("is_served",)):
        self.catalog = _FakeCatalog()
        self._store = {}
        self._columns = list(columns)
        self.statements = []

    def table(self, name):
        return SimpleNamespace(columns=self._columns)

    def sql(self, statement):
        self.statements.append(statement)

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


def test_lgbm_timestamp_and_forecast_id_use_shifted_target():
    generate_forecasts = _load_notebook_functions("generate_forecasts")["generate_forecasts"]
    features = _make_features(24)

    result = generate_forecasts(
        model=_FakeLgbmModel(),
        model_name="energy_lgbm_24h",
        model_version="1",
        run_id="run-2",
        features_df=features,
        horizon_hours=24,
        forecast_run_at=datetime(2026, 8, 10, tzinfo=UTC),
        config={"pipeline_run_id": "test-run"},
    )

    shifted = [anchor + pd.Timedelta(hours=24) for anchor in features.index]
    assert list(result["timestamp"]) == shifted
    # The bug is what we just fixed: stored timestamp must not be the anchor.
    assert list(result["timestamp"]) != list(features.index)
    expected_ids = [_fid("energy_lgbm_24h", 24, target) for target in shifted]
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


# ─────────────────── spec 013 T3: production run or seasonal naive ───────────────────

RUN_AT = datetime(2026, 10, 24, 5, 0, tzinfo=UTC)
SERVE_CONFIG = {"pipeline_run_id": "test-run"}
SERVE_FUNCTIONS = (
    "load_production_model",
    "load_shadow_model",
    "lgbm_forecasts",
    "prepare_inference_features",
    "generate_forecasts",
    "generate_naive_forecasts",
    "serve_forecasts",
)


def _run(run_id, *, production=True, promoted_at=None, mape=None):
    tags = {"model_name": "energy_lgbm_24h", "leak_affected": "false", "data_version": "2"}
    if production:
        tags["production"] = "true"
    if promoted_at:
        tags["promoted_at"] = promoted_at
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, start_time=0),
        data=SimpleNamespace(tags=tags, metrics={} if mape is None else {"mape": mape}),
    )


class _FakeMlflowClient:
    """Applies the production filter the way MLflow would and records every filter."""

    def __init__(self, runs, fail=False):
        self.runs = runs
        self.fail = fail
        self.filters = []

    def search_experiments(self):
        if self.fail:
            raise RuntimeError("MLflow unreachable")
        return [SimpleNamespace(experiment_id="1")]

    def search_runs(self, experiment_ids, filter_string, **kwargs):
        self.filters.append(filter_string)
        if "tags.production = 'true'" in filter_string:
            return [r for r in self.runs if r.data.tags.get("production") == "true"]
        return list(self.runs)


class _RaisingModel:
    def predict(self, x):
        raise ValueError("predict exploded")


def _fake_mlflow(model, loaded_uris):
    def load_model(uri):
        loaded_uris.append(uri)
        return model

    return SimpleNamespace(lightgbm=SimpleNamespace(load_model=load_model))


def _silver_history(hours=200, end=None):
    end = RUN_AT - pd.Timedelta(hours=1) if end is None else end
    index = pd.date_range(end=pd.Timestamp(end), periods=hours, freq="h", name="timestamp")
    return pd.DataFrame(
        {
            "timestamp": index,
            "value_mwh": np.arange(hours, dtype=float) + 4000.0,
            "temperature_c": 10.0,
            "humidity_pct": 60.0,
            "cloud_cover_pct": 20.0,
        }
    )


def _serve(client, model=None, history=None, loaded_uris=None):
    loaded_uris = [] if loaded_uris is None else loaded_uris
    fns = _load_notebook_functions(
        *SERVE_FUNCTIONS, mlflow=_fake_mlflow(model or _FakeLgbmModel(), loaded_uris)
    )
    history = _silver_history() if history is None else history
    return fns["serve_forecasts"](history, client, 24, RUN_AT, SERVE_CONFIG)


def _expected_targets():
    return serving_target_timestamps(RUN_AT, 24)


def test_serves_production_lgbm_run_on_lgbm_targets():
    loaded = []
    client = _FakeMlflowClient([_run("prod", promoted_at="2026-10-21T21:15:08+00:00", mape=8.39)])

    result, mode = _serve(client, loaded_uris=loaded)

    assert mode == "energy_lgbm_24h"
    assert loaded == ["runs:/prod/model"]
    served = result[result["is_served"]]
    assert set(served["model_name"]) == {"energy_lgbm_24h"}
    assert list(served["timestamp"]) == _expected_targets()  # run+25 … run+48
    # Spec 017: naive is written on the same hours, not served.
    naive = result[result["model_name"] == "seasonal_naive"]
    assert list(naive["timestamp"]) == _expected_targets()
    assert not naive["is_served"].any()


def test_ac3_unpromoted_best_mape_run_is_never_served():
    loaded = []
    client = _FakeMlflowClient([_run("best-mape", production=False, mape=0.5)])

    result, mode = _serve(client, loaded_uris=loaded)

    assert mode == "seasonal_naive"
    assert "tags.production = 'true'" in client.filters[0]  # serving looks only at production
    served = result[result["is_served"]]
    assert set(served["model_name"]) == {"seasonal_naive"}
    assert set(served["model_version"]) == {"seasonal_naive_t-168h"}
    assert set(result["horizon_hours"]) == {24}
    # Spec 017 T2: the unpromoted run is scored in shadow only.
    assert loaded == ["runs:/best-mape/model"]
    shadow = result[result["model_name"] == "energy_lgbm_24h"]
    assert len(shadow) == 24 and not shadow["is_served"].any()


def test_ac4_latest_promoted_at_wins_over_lowest_mape():
    loaded = []
    client = _FakeMlflowClient(
        [
            _run("old-low-mape", promoted_at="2026-10-01T00:00:00+00:00", mape=1.0),
            _run("new-high-mape", promoted_at="2026-10-21T21:15:08+00:00", mape=9.0),
        ]
    )

    _, mode = _serve(client, loaded_uris=loaded)

    assert mode == "energy_lgbm_24h"
    assert loaded == ["runs:/new-high-mape/model"]


def test_production_run_without_mape_metric_still_serves():
    loaded = []
    client = _FakeMlflowClient([_run("prod-no-mape", promoted_at="2026-10-21T21:15:08+00:00")])

    _, mode = _serve(client, loaded_uris=loaded)

    assert mode == "energy_lgbm_24h"
    assert loaded == ["runs:/prod-no-mape/model"]


def test_ac5_loader_error_falls_back_to_naive():
    client = _FakeMlflowClient([_run("prod")], fail=True)

    result, mode = _serve(client)

    assert mode == "seasonal_naive"
    assert list(result["timestamp"]) == _expected_targets()


def test_ac5_feature_build_error_falls_back_to_naive():
    # 30 rows are too few for LightGBM features (< 168) but cover every T-168h source hour.
    history = _silver_history(hours=30, end=RUN_AT - pd.Timedelta(hours=116))
    client = _FakeMlflowClient([_run("prod", promoted_at="2026-10-21T21:15:08+00:00")])

    result, mode = _serve(client, history=history)

    assert mode == "seasonal_naive"
    assert list(result["timestamp"]) == _expected_targets()


def test_ac5_predict_error_falls_back_to_naive():
    client = _FakeMlflowClient([_run("prod", promoted_at="2026-10-21T21:15:08+00:00")])

    result, mode = _serve(client, model=_RaisingModel())

    assert mode == "seasonal_naive"
    assert len(result) == 24


def test_ac6_naive_rows_use_t_minus_168h_and_lgbm_targets():
    history = _silver_history()
    value_by_ts = history.set_index("timestamp")["value_mwh"]
    client = _FakeMlflowClient([])

    result, mode = _serve(client, history=history)

    assert mode == "seasonal_naive"
    assert list(result["timestamp"]) == _expected_targets()
    for target, predicted in zip(result["timestamp"], result["predicted_mwh"], strict=True):
        assert predicted == value_by_ts[target - pd.Timedelta(hours=168)]
    expected_ids = [_fid("seasonal_naive", 24, t) for t in _expected_targets()]
    assert list(result["forecast_id"]) == expected_ids


def test_ac6_naive_skips_targets_without_source_value():
    history = _silver_history()
    missing = _expected_targets()[0] - pd.Timedelta(hours=168)
    history = history[history["timestamp"] != missing]

    result, mode = _serve(_FakeMlflowClient([]), history=history)

    assert mode == "seasonal_naive"
    assert len(result) == 23
    assert _expected_targets()[0] not in set(result["timestamp"])


def test_ac6_naive_raises_when_silver_too_stale():
    history = _silver_history(end=RUN_AT - pd.Timedelta(hours=200))

    with pytest.raises(ValueError, match="no T-168h value"):
        _serve(_FakeMlflowClient([]), history=history)


# ─────────────────── spec 017 T1/T2: is_served, naive always, shadow LightGBM ───────────────────


def test_ac3_naive_serving_marks_naive_rows_served():
    result, mode = _serve(_FakeMlflowClient([]))

    assert mode == "seasonal_naive"
    assert set(result["model_name"]) == {"seasonal_naive"}  # no v2 run, so no shadow
    assert result["is_served"].all()


def test_ac3_production_serving_does_not_look_up_a_shadow_run():
    loaded = []
    client = _FakeMlflowClient(
        [
            _run("prod", promoted_at="2026-10-21T21:15:08+00:00"),
            _run("newer-unpromoted", production=False),
        ]
    )

    _serve(client, loaded_uris=loaded)

    assert loaded == ["runs:/prod/model"]
    assert len(client.filters) == 1


def test_ac3_shadow_rows_carry_their_run_id_and_use_the_current_data_version():
    client = _FakeMlflowClient([_run("shadow-run", production=False)])

    result, _ = _serve(client)

    shadow = result[result["model_name"] == "energy_lgbm_24h"]
    assert set(shadow["model_version"]) == {"shadow-run"}
    assert "tags.data_version = '2'" in client.filters[-1]
    assert list(shadow["timestamp"]) == _expected_targets()


def test_ac3_shadow_failure_leaves_served_naive_rows_intact():
    client = _FakeMlflowClient([_run("shadow-run", production=False)])

    result, mode = _serve(client, model=_RaisingModel())

    assert mode == "seasonal_naive"
    assert set(result["model_name"]) == {"seasonal_naive"}
    assert len(result) == 24 and result["is_served"].all()


def test_ac3_lgbm_still_serves_when_naive_has_no_source_hours():
    # 200 rows ending 121h before the run: features build, but no T-168h source exists.
    history = _silver_history(end=RUN_AT - pd.Timedelta(hours=200))
    client = _FakeMlflowClient([_run("prod", promoted_at="2026-10-21T21:15:08+00:00")])

    result, mode = _serve(client, history=history)

    assert mode == "energy_lgbm_24h"
    assert set(result["model_name"]) == {"energy_lgbm_24h"}
    assert result["is_served"].all()


def test_write_forecasts_adds_is_served_column_to_existing_table():
    write_forecasts = _load_notebook_functions("write_forecasts")["write_forecasts"]
    spark = _FakeSpark(columns=("forecast_id", "timestamp"))

    write_forecasts(_forecast_frame(), spark, {"forecast_table": "gold"}, is_backfill=False)

    assert spark.statements == ["ALTER TABLE gold ADD COLUMNS (is_served BOOLEAN)"]


def test_write_forecasts_leaves_schema_alone_when_is_served_exists():
    write_forecasts = _load_notebook_functions("write_forecasts")["write_forecasts"]
    spark = _FakeSpark()

    write_forecasts(_forecast_frame(), spark, {"forecast_table": "gold"}, is_backfill=False)

    assert spark.statements == []
