"""Integration tests for 04_predict: serving, fallback, shadow, gold writes.

The functions under test live in the Databricks notebook ``notebooks/04_predict.py``,
which cannot be imported directly (module-level ``dbutils``/``spark`` calls). We extract
the function definitions via ``ast`` and exec them with a controlled namespace.

Spec 019: a run at issuance hour τ forecasts exactly T = τ + 24h, on the same
point-in-time features as training (src/pit_features.build_features).
"""

import ast
import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.baseline import seasonal_naive_forecast
from src.forecast_timing import SERVED_MODEL, issuance_hour, serving_target_timestamps
from src.gold_rows import combine_gold_rows
from src.pit_features import FEATURES, WEATHER_COLUMNS, build_features
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
RUN_AT = datetime(2026, 10, 24, 5, 0, tzinfo=UTC)
TARGET = pd.Timestamp("2026-10-25 05:00", tz="UTC")
SERVE_CONFIG = {"pipeline_run_id": "test-run"}
SERVE_FUNCTIONS = (
    "load_production_model",
    "load_shadow_model",
    "prepare_features",
    "lgbm_forecasts",
    "generate_forecasts",
    "generate_naive_forecasts",
    "serve_forecasts",
)


class ModelNotFoundError(Exception):
    """Stand-in for the notebook's class (only FunctionDefs are extracted)."""


def _load_notebook_functions(*names: str, **overrides) -> dict:
    """Exec the named top-level functions from 04_predict.py in a stubbed namespace."""
    tree = ast.parse(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {
        "hashlib": hashlib,
        "np": np,
        "pd": pd,
        "datetime": datetime,
        "UTC": UTC,
        "logger": logging.getLogger("predict-test"),
        "StructType": StructType,
        "StructField": StructField,
        "StringType": StringType,
        "TimestampType": TimestampType,
        "IntegerType": IntegerType,
        "DoubleType": DoubleType,
        "BooleanType": BooleanType,
        "DeltaTable": _FakeDeltaTable,
        "MlflowClient": object,
        "ModelNotFoundError": ModelNotFoundError,
        "pick_production_run": pick_production_run,
        "seasonal_naive_forecast": seasonal_naive_forecast,
        "serving_target_timestamps": serving_target_timestamps,
        "issuance_hour": issuance_hour,
        "build_features": build_features,
        "FEATURES": FEATURES,
        "REQUIRED_FEATURES": ["load_t_minus_48h", "load_t_minus_168h", "temperature_fc"],
        "SERVED_MODEL": SERVED_MODEL,
        "NAIVE_MODEL_NAME": "seasonal_naive",
        "NAIVE_MODEL_VERSION": "seasonal_naive_t-168h",
        "CURRENT_RUN_FILTER": "tags.data_version = '2' AND tags.feature_set = 'pit_24h_v1'",
        "FEATURE_SET": "pit_24h_v1",
        "combine_gold_rows": combine_gold_rows,
        **overrides,
    }
    exec(compile(tree, str(NOTEBOOK_PATH), "exec"), namespace)
    return {name: namespace[name] for name in names}


# ───────────────────────────── fakes ─────────────────────────────


class _FakeLgbmModel:
    """Minimal LightGBM-like regressor: no booster_, predicts 1000 + row index."""

    def __init__(self):
        self.inputs = []

    def predict(self, x):
        self.inputs.append(x)
        return np.arange(len(x), dtype=float) + 1000.0


class _OtherFeaturesModel(_FakeLgbmModel):
    booster_ = SimpleNamespace(feature_name=lambda: ["lag_24h", "temperature_c"])


class _RaisingModel:
    def predict(self, x):
        raise ValueError("predict exploded")


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
        self.update_matched = False
        self.insert_new = False

    @classmethod
    def forName(cls, spark, name):
        return cls(spark._store)

    def alias(self, name):
        return self

    def merge(self, source, condition):
        self.source = source
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

    def createDataFrame(self, df, schema):
        return _FakeSourceFrame(df.to_dict("records"))

    def table(self, name):
        return SimpleNamespace(columns=self._columns)

    def sql(self, statement):
        self.statements.append(statement)


def _run(run_id, *, production=True, promoted_at=None, mape=None):
    tags = {
        "model_name": "energy_lgbm_24h",
        "leak_affected": "false",
        "data_version": "2",
        "feature_set": "pit_24h_v1",
    }
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


def _fake_mlflow(model, loaded_uris):
    def load_model(uri):
        loaded_uris.append(uri)
        return model

    return SimpleNamespace(lightgbm=SimpleNamespace(load_model=load_model))


def _history(hours=400, end=None):
    end = pd.Timestamp(RUN_AT) - pd.Timedelta(hours=2) if end is None else end
    index = pd.date_range(end=pd.Timestamp(end), periods=hours, freq="h")
    return pd.DataFrame({"timestamp": index, "value_mwh": np.arange(hours, dtype=float) + 4000.0})


def _weather(hours=96):
    index = pd.date_range(pd.Timestamp(RUN_AT) - pd.Timedelta(hours=24), periods=hours, freq="h")
    return pd.DataFrame({"timestamp": index, **{c: 10.0 for c in WEATHER_COLUMNS}})


def _serve(client, model=None, history=None, weather=None, loaded_uris=None):
    loaded_uris = [] if loaded_uris is None else loaded_uris
    fns = _load_notebook_functions(
        *SERVE_FUNCTIONS, mlflow=_fake_mlflow(model or _FakeLgbmModel(), loaded_uris)
    )
    history = _history() if history is None else history
    weather = _weather() if weather is None else weather
    return fns["serve_forecasts"](history, weather, client, 24, RUN_AT, SERVE_CONFIG)


def _fid(model_name: str, timestamp: pd.Timestamp) -> str:
    return hashlib.md5(f"{model_name}_24_{timestamp.isoformat()}".encode()).hexdigest()


PROD = {"promoted_at": "2026-10-21T21:15:08+00:00"}


# ─────────────────── spec 019: one target, the training feature function ───────────────────


def test_ac4_serves_exactly_one_target_24h_after_issuance():
    result, mode = _serve(_FakeMlflowClient([_run("prod", **PROD)]))

    assert mode == "energy_lgbm_24h"
    assert set(result["timestamp"]) == {TARGET}
    assert len(result) == 2  # the served LightGBM row and the naive row
    assert set(result["forecast_id"]) == {
        _fid("energy_lgbm_24h", TARGET),
        _fid("seasonal_naive", TARGET),
    }


def test_ac4_model_gets_the_same_features_as_training():
    model = _FakeLgbmModel()
    history, weather = _history(), _weather()

    _serve(_FakeMlflowClient([_run("prod", **PROD)]), model=model, history=history, weather=weather)

    expected = build_features(history, weather, [issuance_hour(RUN_AT)], include_target=False)
    pd.testing.assert_frame_equal(model.inputs[0], expected[FEATURES])


def test_ac4_run_trained_on_other_features_is_not_served():
    result, mode = _serve(_FakeMlflowClient([_run("old", **PROD)]), model=_OtherFeaturesModel())

    assert mode == "seasonal_naive"
    assert set(result[result["is_served"]]["model_name"]) == {"seasonal_naive"}


def test_ac4_missing_weather_forecast_falls_back_to_naive():
    result, mode = _serve(
        _FakeMlflowClient([_run("prod", **PROD)]), weather=pd.DataFrame(columns=["timestamp"])
    )

    assert mode == "seasonal_naive"
    assert set(result["model_name"]) == {"seasonal_naive"}  # the shadow is skipped as well


def test_ac4_late_ingestion_within_22h_still_serves_the_model():
    # Load only up to τ - 24h: the model needs nothing newer.
    history = _history(end=pd.Timestamp(RUN_AT) - pd.Timedelta(hours=24))

    _, mode = _serve(_FakeMlflowClient([_run("prod", **PROD)]), history=history)

    assert mode == "energy_lgbm_24h"


def test_ac4_naive_row_is_t_minus_168h():
    history = _history()
    value_by_ts = history.set_index("timestamp")["value_mwh"]

    result, _ = _serve(_FakeMlflowClient([]), history=history)

    naive = result[result["model_name"] == "seasonal_naive"].iloc[0]
    source = (TARGET - pd.Timedelta(hours=168)).tz_convert("UTC")
    assert naive["predicted_mwh"] == value_by_ts[source]


# ─────────────────── spec 013/017/018 serving rules ───────────────────


def test_ac3_unpromoted_run_is_never_served_only_shadowed():
    loaded = []
    client = _FakeMlflowClient([_run("best-mape", production=False, mape=0.5)])

    result, mode = _serve(client, loaded_uris=loaded)

    assert mode == "seasonal_naive"
    assert "tags.production = 'true'" in client.filters[0]
    assert set(result[result["is_served"]]["model_name"]) == {"seasonal_naive"}
    assert loaded == ["runs:/best-mape/model"]
    shadow = result[result["model_name"] == "energy_lgbm_24h"]
    assert len(shadow) == 1 and not shadow["is_served"].any()
    assert set(shadow["model_version"]) == {"best-mape"}


def test_ac4_latest_promoted_at_wins_over_lowest_mape():
    loaded = []
    client = _FakeMlflowClient(
        [
            _run("old-low-mape", promoted_at="2026-10-01T00:00:00+00:00", mape=1.0),
            _run("new-high-mape", promoted_at="2026-10-21T21:15:08+00:00", mape=9.0),
        ]
    )

    _serve(client, loaded_uris=loaded)

    assert loaded == ["runs:/new-high-mape/model"]


def test_every_run_lookup_filters_on_feature_set_and_data_version():
    client = _FakeMlflowClient([_run("shadow", production=False)])

    _serve(client)

    assert len(client.filters) == 2
    assert all(
        "tags.feature_set = 'pit_24h_v1'" in f and "tags.data_version = '2'" in f
        for f in client.filters
    )


def test_production_rows_carry_the_run_id_and_naive_is_not_served():
    result, _ = _serve(_FakeMlflowClient([_run("prod-run", **PROD)]))

    served = result[result["is_served"]]
    assert set(served["model_version"]) == {"prod-run"}
    assert not result[result["model_name"] == "seasonal_naive"]["is_served"].any()


def test_production_serving_does_not_look_up_a_shadow_run():
    client = _FakeMlflowClient([_run("prod", **PROD), _run("newer", production=False)])

    _serve(client)

    assert len(client.filters) == 1


def test_loader_error_falls_back_to_naive():
    result, mode = _serve(_FakeMlflowClient([_run("prod")], fail=True))

    assert mode == "seasonal_naive"
    assert list(result["timestamp"]) == [TARGET]


def test_predict_error_and_shadow_failure_leave_served_naive_intact():
    result, mode = _serve(_FakeMlflowClient([_run("prod", **PROD)]), model=_RaisingModel())

    assert mode == "seasonal_naive"
    assert set(result["model_name"]) == {"seasonal_naive"}
    assert result["is_served"].all()


def test_naive_raises_when_silver_too_stale_and_no_model_serves():
    history = _history(end=pd.Timestamp(RUN_AT) - pd.Timedelta(hours=200))

    with pytest.raises(ValueError, match="no T-168h value"):
        _serve(_FakeMlflowClient([]), history=history)


# ─────────────────── gold writes ───────────────────


def _forecast_frame() -> pd.DataFrame:
    fns = _load_notebook_functions("generate_forecasts")
    feats = build_features(_history(), _weather(), [issuance_hour(RUN_AT)], include_target=False)
    return fns["generate_forecasts"](
        model=_FakeLgbmModel(),
        model_name="energy_lgbm_24h",
        model_version="run",
        run_id="run",
        features_df=feats,
        horizon_hours=24,
        forecast_run_at=RUN_AT,
        config=SERVE_CONFIG,
    )


def test_write_forecasts_merges_on_forecast_id_only():
    write_forecasts = _load_notebook_functions("write_forecasts")["write_forecasts"]

    write_forecasts(_forecast_frame(), _FakeSpark(), {"forecast_table": "gold"}, is_backfill=False)

    assert _FakeDeltaTable.last_condition == "target.forecast_id = source.forecast_id"


def test_write_forecasts_is_idempotent_and_keeps_actuals_outside_backfill():
    write_forecasts = _load_notebook_functions("write_forecasts")["write_forecasts"]
    frame = _forecast_frame()
    spark = _FakeSpark()
    first = frame.iloc[0].to_dict()
    first["actual_mwh"] = 999.0
    spark._store[first["forecast_id"]] = first

    write_forecasts(frame, spark, {"forecast_table": "gold"}, is_backfill=False)
    write_forecasts(frame, spark, {"forecast_table": "gold"}, is_backfill=False)
    assert len(spark._store) == len(frame)
    assert spark._store[first["forecast_id"]]["actual_mwh"] == 999.0

    write_forecasts(frame, spark, {"forecast_table": "gold"}, is_backfill=True)
    assert spark._store[first["forecast_id"]]["actual_mwh"] is None


def test_write_forecasts_adds_is_served_column_to_existing_table():
    write_forecasts = _load_notebook_functions("write_forecasts")["write_forecasts"]
    spark = _FakeSpark(columns=("forecast_id", "timestamp"))

    write_forecasts(_forecast_frame(), spark, {"forecast_table": "gold"}, is_backfill=False)

    assert spark.statements == ["ALTER TABLE gold ADD COLUMNS (is_served BOOLEAN)"]
