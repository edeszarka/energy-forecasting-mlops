"""Integration tests for 04_predict: catch-up serving, fallback, shadow, gold writes.

The functions under test live in the Databricks notebook ``notebooks/04_predict.py``,
which cannot be imported directly (module-level ``dbutils``/``spark`` calls). We extract
the function definitions via ``ast`` and exec them with a controlled namespace.

Target selection and the serving decision are tested in tests/test_serving_plan.py
(spec 019b D6); these tests check that 04 calls them, builds features with the
training function, and writes what they decide.
"""

import ast
import logging
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.config import DAILY_NAIVE_MODEL_NAME
from src.forecast_timing import SERVED_MODEL
from src.gold_rows import (
    DAILY_NAIVE_MODEL_VERSION,
    GOLD_COLUMNS,
    gold_forecast_id,
    gold_row,
)
from src.pit_features import DAILY_NAIVE, FEATURES, WEATHER_COLUMNS, WEEKLY_NAIVE, build_features
from src.promotion import pick_production_run
from src.serving_plan import (
    DAILY_NAIVE as DAILY_NAIVE_DECISION,
)
from src.serving_plan import (
    MODEL,
    NAIVE,
    NO_PRODUCTION_RUN,
    covered_targets,
    decide_serving,
    plan_targets,
    raise_on_model_errors,
)

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
NOW = pd.Timestamp("2026-10-24 05:00", tz="UTC")
RUN_AT = NOW.to_pydatetime()
H = pd.Timedelta(hours=1)
CONFIG = {
    "pipeline_run_id": "test-run",
    "forecast_table": "gold",
    "silver_table": "silver",
    "weather_table": "weather",
}
SERVE_FUNCTIONS = (
    "load_production_model",
    "load_shadow_model",
    "load_model",
    "predict_one",
    "serve_targets",
)
REQUIRED = ["load_t_minus_48h", "load_t_minus_168h", "temperature_fc"]


class ModelNotFoundError(Exception):
    """Stand-in for the notebook's class (only FunctionDefs are extracted)."""


class _Col:
    """Minimal stand-in for a pyspark Column: comparisons return itself."""

    def __gt__(self, other):
        return self

    __le__ = __ge__ = __eq__ = __gt__  # type: ignore[assignment]

    def cast(self, _):
        return self


FAKE_F = SimpleNamespace(col=lambda name: _Col(), lit=lambda value: _Col())


def _load_notebook_functions(*names: str, **overrides) -> dict:
    """Exec the named top-level functions from 04_predict.py in a stubbed namespace."""
    tree = ast.parse(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {
        "np": np,
        "pd": pd,
        "F": FAKE_F,
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
        "build_features": build_features,
        "FEATURES": FEATURES,
        "WEEKLY_NAIVE": WEEKLY_NAIVE,
        "DAILY_NAIVE": DAILY_NAIVE,
        "DAILY_NAIVE_DECISION": DAILY_NAIVE_DECISION,
        "DAILY_NAIVE_MODEL_NAME": DAILY_NAIVE_MODEL_NAME,
        "DAILY_NAIVE_MODEL_VERSION": DAILY_NAIVE_MODEL_VERSION,
        "REQUIRED_FEATURES": REQUIRED,
        "HISTORY_HOURS": 400,
        "HORIZON_HOURS": 24,
        "ADDED_GOLD_COLUMNS": {
            "is_served": "BOOLEAN",
            "issued_at": "TIMESTAMP",
            "is_catch_up": "BOOLEAN",
            "fallback_reason": "STRING",
        },
        "GOLD_COLUMNS": GOLD_COLUMNS,
        "SERVED_MODEL": SERVED_MODEL,
        "NAIVE_MODEL_NAME": "seasonal_naive",
        "NAIVE_MODEL_VERSION": "seasonal_naive_t-168h",
        "CURRENT_RUN_FILTER": "tags.data_version = '2' AND tags.feature_set = 'pit_24h_v1'",
        "FEATURE_SET": "pit_24h_v1",
        "gold_row": gold_row,
        "covered_targets": covered_targets,
        "decide_serving": decide_serving,
        "MODEL": MODEL,
        "NAIVE": NAIVE,
        "NO_PRODUCTION_RUN": NO_PRODUCTION_RUN,
        **overrides,
    }
    exec(compile(tree, str(NOTEBOOK_PATH), "exec"), namespace)
    return {name: namespace[name] for name in names}


# ───────────────────────────── fakes ─────────────────────────────


class _FeatureSumModel:
    """Predicts a deterministic function of its inputs and records them."""

    def __init__(self):
        self.inputs = []

    def predict(self, x):
        self.inputs.append(x.copy())
        return x.fillna(0).sum(axis=1).to_numpy() / 10.0


class _OtherFeaturesModel(_FeatureSumModel):
    booster_ = SimpleNamespace(feature_name=lambda: ["lag_24h", "temperature_c"])


class _RaisingModel:
    def predict(self, x):
        raise ValueError("predict exploded")


class _FakeDeltaTable:
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


class _FakeSourceFrame:
    def __init__(self, records):
        self.records = records

    def alias(self, name):
        return self


class _FakeTable:
    def __init__(self, frame):
        self.frame = frame
        self.columns = list(frame.columns)

    def filter(self, _):
        return self

    def select(self, *_):
        return self

    def withColumn(self, name, _):
        return _FakeTable(self.frame.assign(**{name: None}))

    def toPandas(self):
        return self.frame.copy()


class _FakeSpark:
    def __init__(self, tables=None, gold_columns=("is_served",)):
        self.tables = tables or {}
        self._store = {}
        self.statements = []
        self.gold_columns = list(gold_columns)
        self.catalog = SimpleNamespace(
            tableExists=lambda name: name in self.tables or name == "gold"
        )

    def table(self, name):
        if name == "gold" and name not in self.tables:
            return SimpleNamespace(columns=self.gold_columns)
        return _FakeTable(self.tables[name])

    def createDataFrame(self, df, schema):
        return _FakeSourceFrame(df.to_dict("records"))

    def sql(self, statement):
        self.statements.append(statement)


def _run(run_id, *, production=True, promoted_at="2026-10-21T21:15:08+00:00"):
    tags = {
        "model_name": "energy_lgbm_24h",
        "leak_affected": "false",
        "data_version": "2",
        "feature_set": "pit_24h_v1",
    }
    if production:
        tags["production"] = "true"
        tags["promoted_at"] = promoted_at
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, start_time=0),
        data=SimpleNamespace(tags=tags, metrics={}),
    )


class _FakeMlflowClient:
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


def _fake_mlflow(model, loaded):
    def load_model(uri):
        loaded.append(uri)
        return model

    return SimpleNamespace(lightgbm=SimpleNamespace(load_model=load_model))


def _history(hours=420, end=None):
    end = NOW - 2 * H if end is None else end
    index = pd.date_range(end=end, periods=hours, freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"timestamp": index, "value_mwh": 4000.0 + rng.normal(0, 200, hours)})


def _weather(hours=120):
    index = pd.date_range(NOW - 48 * H, periods=hours, freq="h")
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {"timestamp": index, **{c: rng.normal(10, 3, hours) for c in WEATHER_COLUMNS}}
    )


def _serve(
    client,
    plan,
    model=None,
    history=None,
    weather=None,
    weather_present=True,
    loaded=None,
    run_at=RUN_AT,
):
    loaded = [] if loaded is None else loaded
    client = client if isinstance(client, _FakeMlflowClient) else _FakeMlflowClient(client)
    fns = _load_notebook_functions(
        *SERVE_FUNCTIONS, mlflow=_fake_mlflow(model or _FeatureSumModel(), loaded)
    )
    history = _history() if history is None else history
    weather = _weather() if weather is None else weather
    rows, decisions = fns["serve_targets"](
        plan, history, weather, weather_present, client, run_at, CONFIG
    )
    return pd.DataFrame(rows, columns=GOLD_COLUMNS), decisions


def _only(target, run_at=NOW):
    """Plan with just `target` uncovered."""
    covered = [run_at + k * H for k in range(1, 25) if run_at + k * H != target]
    return plan_targets(run_at, covered)


PROD = [_run("prod")]


# ─────────────────── AC2: catch-up equals in-time ───────────────────


def test_ac2_catch_up_equals_in_time_for_the_same_issuance():
    issued = NOW - 3 * H
    target = issued + 24 * H
    history, weather = _history(), _weather()

    late_model, timely_model = _FeatureSumModel(), _FeatureSumModel()
    late_rows, _ = _serve(PROD, _only(target), late_model, history, weather)
    timely_rows, _ = _serve(
        PROD,
        _only(target, run_at=issued),
        timely_model,
        history,
        weather,
        run_at=issued.to_pydatetime(),
    )

    late, timely = (
        r[r["model_name"] == "energy_lgbm_24h"].iloc[0] for r in (late_rows, timely_rows)
    )
    pd.testing.assert_frame_equal(late_model.inputs[0], timely_model.inputs[0])
    assert late["predicted_mwh"] == timely["predicted_mwh"]
    assert (late["is_catch_up"], timely["is_catch_up"]) == (True, False)
    assert late["issued_at"] == timely["issued_at"] == issued
    expected = build_features(history, weather, [issued], include_target=False)[FEATURES]
    pd.testing.assert_frame_equal(late_model.inputs[0], expected)


def test_catch_up_fills_every_planned_target_in_one_run():
    plan = plan_targets(NOW, covered=[])  # a whole day of missed runs

    rows, decisions = _serve(PROD, plan)

    served = rows[rows["is_served"]]
    assert sorted(served["timestamp"]) == [NOW + k * H for k in range(1, 25)]
    assert served["is_catch_up"].sum() == 23
    assert {d.kind for _, d in decisions} == {MODEL}


# ─────────────────── AC4: gold columns ───────────────────


def test_ac4_rows_carry_issued_at_is_catch_up_and_fallback_reason():
    target = NOW + 24 * H
    rows, _ = _serve([], _only(target))  # no production run → naive

    naive = rows[rows["model_name"] == "seasonal_naive"].iloc[0]
    assert naive["forecast_id"] == gold_forecast_id("seasonal_naive", 24, target)
    assert naive["issued_at"] == NOW and not naive["is_catch_up"]
    assert naive["fallback_reason"] == "no_production_run" and naive["is_served"]
    assert list(rows.columns) == GOLD_COLUMNS and "written_at" not in rows


def test_ac4_model_rows_have_no_fallback_reason_and_naive_is_not_served():
    rows, _ = _serve(PROD, _only(NOW + 24 * H))

    model = rows[rows["model_name"] == "energy_lgbm_24h"].iloc[0]
    naive = rows[rows["model_name"] == "seasonal_naive"].iloc[0]
    assert (
        model["is_served"] and model["fallback_reason"] is None and model["model_version"] == "prod"
    )
    assert not naive["is_served"] and naive["fallback_reason"] is None


# ─────────────────── AC5 / AC6: fallbacks ───────────────────


def test_fallback_model_inputs_missing_serves_naive_quietly():
    target = NOW + 24 * H
    weather = _weather()
    weather = weather[weather["timestamp"] != target]

    rows, decisions = _serve(PROD, _only(target), weather=weather)

    assert decisions[0][1].fallback_reason == "model_inputs_missing:temperature_fc"
    assert set(rows[rows["is_served"]]["model_name"]) == {"seasonal_naive"}
    raise_on_model_errors(decisions)  # quiet: no exception


def test_fallback_no_naive_input_writes_nothing_and_does_not_raise():
    """No production run, and neither T - 168h nor T - 48h: skip (O1)."""
    target = NOW + 24 * H
    history = _history()
    history = history[~history["timestamp"].isin([target - 168 * H, target - 48 * H])]
    weather = _weather()
    weather = weather[weather["timestamp"] != target]

    rows, decisions = _serve([], _only(target), history=history, weather=weather)

    assert rows.empty
    assert decisions[0][1].kind == "skip"
    raise_on_model_errors(decisions)


def test_d8_weekly_naive_missing_serves_daily_naive():
    target = NOW + 24 * H
    history = _history()
    history = history[history["timestamp"] != target - 168 * H]
    value_by_ts = history.set_index("timestamp")["value_mwh"]

    rows, decisions = _serve(PROD, _only(target), history=history)

    assert decisions[0][1].kind == DAILY_NAIVE_DECISION
    served = rows[rows["is_served"]]
    assert len(rows) == 1 and len(served) == 1  # no seasonal-naive or shadow row
    row = served.iloc[0]
    assert row["model_name"] == "daily_naive" and row["model_version"] == "daily_naive_t-48h"
    assert row["fallback_reason"] == "model_inputs_missing:load_t_minus_168h"
    assert row["predicted_mwh"] == value_by_ts[target - 48 * H]
    assert row["forecast_id"] == gold_forecast_id("daily_naive", 24, target)
    raise_on_model_errors(decisions)  # quiet


def test_d8_model_error_with_missing_t_minus_168h_serves_daily_naive_then_fails():
    target = NOW + 24 * H
    history = _history()
    history = history[history["timestamp"] != target - 168 * H]

    rows, decisions = _serve(
        _FakeMlflowClient([_run("prod")], fail=True), _only(target), history=history
    )

    served = rows[rows["is_served"]].iloc[0]
    assert served["model_name"] == "daily_naive"
    assert served["fallback_reason"] == "model_error:RuntimeError"
    with pytest.raises(RuntimeError, match="model_error:RuntimeError"):
        raise_on_model_errors(decisions)  # 04 calls this after write_forecasts


def test_d8_weekly_and_daily_naive_missing_skips_without_exception():
    target = NOW + 24 * H
    history = _history()
    history = history[~history["timestamp"].isin([target - 168 * H, target - 48 * H])]

    rows, decisions = _serve(PROD, _only(target), history=history)

    assert rows.empty
    assert decisions[0][1].kind == "skip"
    raise_on_model_errors(decisions)


def test_ac6_weather_table_missing_serves_naive_without_shadow():
    rows, decisions = _serve(
        [_run("shadow", production=False)],
        _only(NOW + 24 * H),
        weather=pd.DataFrame(columns=["timestamp"]),
        weather_present=False,
    )

    assert decisions[0][1].fallback_reason == "no_production_run"  # first match (B2 order)
    assert set(rows["model_name"]) == {"seasonal_naive"}

    rows, decisions = _serve(
        PROD,
        _only(NOW + 24 * H),
        weather=pd.DataFrame(columns=["timestamp"]),
        weather_present=False,
    )
    assert decisions[0][1].fallback_reason == "weather_table_missing"
    assert set(rows["model_name"]) == {"seasonal_naive"}


# ─────────────────── AC5b: model errors ───────────────────


def test_model_error_mlflow_unreachable_writes_fallback_then_raises():
    rows, decisions = _serve(_FakeMlflowClient([_run("prod")], fail=True), _only(NOW + 24 * H))

    assert rows[rows["is_served"]]["fallback_reason"].tolist() == ["model_error:RuntimeError"]
    with pytest.raises(RuntimeError, match="model_error:RuntimeError"):
        raise_on_model_errors(decisions)


def test_model_error_predict_raises():
    _, decisions = _serve(PROD, _only(NOW + 24 * H), model=_RaisingModel())

    assert decisions[0][1].fallback_reason == "model_error:ValueError"


def test_model_error_feature_mismatch_is_an_error_not_a_quiet_fallback():
    _, decisions = _serve(PROD, _only(NOW + 24 * H), model=_OtherFeaturesModel())

    assert decisions[0][1].fallback_reason == "model_error:FeatureMismatch"


def test_model_error_main_writes_before_it_raises():
    tree = ast.parse(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    calls = {
        n.value.func.id: n.lineno
        for n in tree.body
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
    }

    assert calls["write_forecasts"] < calls["raise_on_model_errors"]


# ─────────────────── spec 013/017/018 rules still hold ───────────────────


def test_unpromoted_run_is_never_served_only_shadowed():
    loaded = []
    client = _FakeMlflowClient([_run("best-mape", production=False)])

    rows, decisions = _serve(client, _only(NOW + 24 * H), loaded=loaded)

    assert decisions[0][1].fallback_reason == "no_production_run"
    assert "tags.production = 'true'" in client.filters[0]
    assert loaded == ["runs:/best-mape/model"]
    shadow = rows[rows["model_name"] == "energy_lgbm_24h"]
    assert len(shadow) == 1 and not shadow["is_served"].any()
    assert set(shadow["model_version"]) == {"best-mape"}


def test_latest_promoted_at_wins():
    loaded = []
    client = _FakeMlflowClient(
        [
            _run("old", promoted_at="2026-10-01T00:00:00+00:00"),
            _run("new", promoted_at="2026-10-21T21:15:08+00:00"),
        ]
    )

    _serve(client, _only(NOW + 24 * H), loaded=loaded)

    assert loaded == ["runs:/new/model"]


def test_every_run_lookup_filters_on_feature_set_and_data_version():
    client = _FakeMlflowClient([_run("shadow", production=False)])

    _serve(client, _only(NOW + 24 * H))

    assert len(client.filters) == 2
    assert all(
        "tags.feature_set = 'pit_24h_v1'" in f and "tags.data_version = '2'" in f
        for f in client.filters
    )


def test_the_model_is_loaded_once_per_run():
    loaded = []

    _serve(_FakeMlflowClient(PROD), plan_targets(NOW, covered=[]), loaded=loaded)

    assert loaded == ["runs:/prod/model"]


# ─────────────────── inputs and gold writes ───────────────────


def test_load_serving_inputs_reports_missing_and_empty_weather_table():
    load = _load_notebook_functions("load_serving_inputs")["load_serving_inputs"]
    silver = _history()

    _, weather, present = load(_FakeSpark({"silver": silver}), CONFIG, NOW)
    assert present is False and weather.empty

    _, weather, present = load(
        _FakeSpark({"silver": silver, "weather": _weather().iloc[0:0]}), CONFIG, NOW
    )
    assert present is False

    _, weather, present = load(_FakeSpark({"silver": silver, "weather": _weather()}), CONFIG, NOW)
    assert present is True and len(weather) == len(_weather())


def test_read_covered_counts_served_rows_and_handles_a_gold_without_is_served():
    read = _load_notebook_functions("read_covered")["read_covered"]
    gold = pd.DataFrame(
        {"timestamp": [NOW + H, NOW + 2 * H], "is_served": [True, False], "horizon_hours": 24}
    )

    assert read(_FakeSpark({"gold": gold}), CONFIG, NOW) == {NOW + H}
    legacy = gold.drop(columns="is_served")
    assert read(_FakeSpark({"gold": legacy}), CONFIG, NOW) == {NOW + H, NOW + 2 * H}


def _rows_frame():
    rows, _ = _serve(PROD, _only(NOW + 24 * H))
    return rows


def test_write_forecasts_adds_every_missing_gold_column():
    write = _load_notebook_functions("write_forecasts")["write_forecasts"]
    spark = _FakeSpark(gold_columns=("forecast_id", "timestamp", "is_served"))

    write(_rows_frame(), spark, CONFIG, is_backfill=False)

    assert spark.statements == [
        "ALTER TABLE gold ADD COLUMNS (issued_at TIMESTAMP, is_catch_up BOOLEAN, fallback_reason STRING)"
    ]


def test_write_forecasts_is_insert_only_and_skips_an_empty_frame():
    write = _load_notebook_functions("write_forecasts")["write_forecasts"]
    spark = _FakeSpark(gold_columns=GOLD_COLUMNS)
    frame = _rows_frame()
    first = frame.iloc[0].to_dict()
    spark._store[first["forecast_id"]] = {**first, "predicted_mwh": -1.0}

    write(frame, spark, CONFIG, is_backfill=False)
    assert spark._store[first["forecast_id"]]["predicted_mwh"] == -1.0  # covered rows stay
    assert _FakeDeltaTable.last_condition == "target.forecast_id = source.forecast_id"

    before = dict(spark._store)
    write(pd.DataFrame(columns=GOLD_COLUMNS), spark, CONFIG, is_backfill=False)
    assert spark._store == before
