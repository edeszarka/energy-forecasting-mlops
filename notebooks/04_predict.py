# Databricks notebook source
# COMMAND ----------

from pathlib import Path

nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
req_path = str(Path("/Workspace" + nb_path).parent.parent / "requirements.txt")

# COMMAND ----------

# MAGIC %pip install -r {req_path}

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
from pathlib import Path

root_path = str(Path(os.getcwd()).parent)
if root_path not in sys.path:
    sys.path.append(root_path)

# COMMAND ----------

# %% [markdown]
# # 04_predict
# **Purpose:** Forecast every uncovered target hour T in (τ_now, τ_now + 24h], each as if
# issued at τ' = T − 24h (spec 019, ADR-0008; catch-up for missed runs, spec 019b), with the
# production LightGBM run on point-in-time features built by the same function as training
# (`src/pit_features.build_features`). Target selection and the serving decision are pure
# functions in `src/serving_plan.py`. Seasonal naive (T − 168h) is written for every target
# and served, with a `fallback_reason`, when the model cannot be used (daily naive, T − 48h,
# when the model's inputs and T − 168h are both missing, D8); while it serves, the
# latest current-feature-set run is scored in shadow (spec 017). A model error writes the
# fallback and then fails the task; missing data never does. Also backfills actuals.
# **Inputs:** `silver_features` (actual load), `bronze_weather_forecast`, MLflow runs
# **Outputs:** `workspace.energy_forecasting.gold_forecasts`

# COMMAND ----------

import json
import logging
from datetime import UTC, datetime

import mlflow
import numpy as np
import pandas as pd
from delta.tables import DeltaTable
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from src.config import (
    CATALOG,
    CURRENT_RUN_FILTER,
    DAILY_NAIVE_MODEL_NAME,
    FEATURE_SET,
    NAIVE_MODEL_NAME,
    PATHS,
)
from src.forecast_timing import SERVED_MODEL, issuance_hour
from src.gold_rows import (
    DAILY_NAIVE_MODEL_VERSION,
    GOLD_COLUMNS,
    NAIVE_MODEL_VERSION,
    gold_row,
)
from src.pit_features import (
    DAILY_NAIVE,
    FEATURES,
    FORECAST_HORIZON_HOURS,
    WEEKLY_NAIVE,
    build_features,
)
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
    summarize_decisions,
)

# COMMAND ----------

# SECTION 1 — SETUP AND CONFIG
# ─────────────────────────────

dbutils.widgets.text("force_backfill", "false")

HORIZON_HOURS = 24  # ADR-0001: 24h only

try:
    pipeline_run_id = (
        dbutils.notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .currentRunId()
        .getOrElse(lambda: "manual")
    )
except Exception:
    pipeline_run_id = "manual"

CONFIG = {
    "silver_table": PATHS.table_silver,
    "weather_table": PATHS.table_weather_forecast,
    "forecast_table": PATHS.table_gold,
    "force_backfill": dbutils.widgets.get("force_backfill").lower() == "true",
    "pipeline_run_id": pipeline_run_id,
}

# Without these the model is not served for a target: naive serves instead.
REQUIRED_FEATURES = ["load_t_minus_48h", "load_t_minus_168h", "temperature_fc"]
# New gold columns (spec 017, 019b), added to an existing table when missing.
ADDED_GOLD_COLUMNS = {
    "is_served": "BOOLEAN",
    "issued_at": "TIMESTAMP",
    "is_catch_up": "BOOLEAN",
    "fallback_reason": "STRING",
}
# Load history the features reach back to: T - 336h = τ - 312h, plus margin.
HISTORY_HOURS = 400

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("predict")


class ModelNotFoundError(Exception):
    """Raised when a Production model is missing from Registry or Runs."""

    pass


# COMMAND ----------

# SECTION 2 — MODEL LOADING FROM RUNS (Workaround for IAM Restrictions)
# ───────────────────────────────────────────────────────────────────────


def load_production_model(model_name: str, mlflow_client: MlflowClient):
    """
    Loads the production-tagged run of `model_name`, chosen with
    pick_production_run (latest promoted_at) — the same rule 07/08/03 use.
    Never falls back to an unpromoted run (ADR-0002); the caller serves
    seasonal naive instead.

    Raises:
        ModelNotFoundError: If no production run exists.
    """
    exp_ids = [e.experiment_id for e in mlflow_client.search_experiments()]
    runs = mlflow_client.search_runs(
        experiment_ids=exp_ids,
        filter_string=(
            f"tags.model_name = '{model_name}' AND tags.production = 'true' "
            "AND tags.leak_affected = 'false' "
            f"AND {CURRENT_RUN_FILTER}"
        ),
        max_results=50,
    )
    run = pick_production_run(runs)
    if run is None:
        raise ModelNotFoundError(f"No production run found for {model_name}.")

    run_id = run.info.run_id
    model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")

    mape = run.data.metrics.get("mape")
    mape_str = f"{mape:.4f}" if mape is not None else "n/a"
    logger.info(f"Loaded {model_name} from Run ID {run_id} (MAPE: {mape_str})")
    return model, run_id, run_id  # model_version = run id (spec 018 T1)


def load_shadow_model(model_name: str, mlflow_client: MlflowClient):
    """
    Loads the latest current-data-version run of `model_name` for shadow scoring
    (spec 017 T2): its forecasts are written to gold with is_served = false and
    are never served. Used only while no production run exists.

    Raises:
        ModelNotFoundError: If no run of the current data version exists.
    """
    exp_ids = [e.experiment_id for e in mlflow_client.search_experiments()]
    runs = mlflow_client.search_runs(
        experiment_ids=exp_ids,
        filter_string=(
            f"tags.model_name = '{model_name}' AND tags.leak_affected = 'false' "
            f"AND {CURRENT_RUN_FILTER}"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ModelNotFoundError(f"No run of feature set {FEATURE_SET} for {model_name}.")

    run_id = runs[0].info.run_id
    model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
    logger.info(f"Shadow scoring {model_name} from Run ID {run_id}")
    return model, run_id, run_id


# COMMAND ----------

# SECTION 3 — INPUTS
# ──────────────────


def read_covered(spark, config: dict, now: pd.Timestamp) -> set:
    """Targets in (now, now + 24h] that already have a served gold row."""
    if not spark.catalog.tableExists(config["forecast_table"]):
        return set()
    table = spark.table(config["forecast_table"])
    if "is_served" not in table.columns:
        table = table.withColumn("is_served", F.lit(None).cast("boolean"))
    rows = (
        table.filter(F.col("horizon_hours") == HORIZON_HOURS)
        .filter(F.col("timestamp") > F.lit(now.to_pydatetime()))
        .filter(F.col("timestamp") <= F.lit((now + pd.Timedelta(hours=24)).to_pydatetime()))
        .select("timestamp", "is_served")
        .toPandas()
    )
    return covered_targets(rows)


def load_serving_inputs(
    spark, config: dict, first_issued: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Load history, weather forecasts, and whether the weather table exists and has rows."""
    since = (first_issued - pd.Timedelta(hours=HISTORY_HOURS)).to_pydatetime()
    history = (
        spark.table(config["silver_table"])
        .filter(F.col("timestamp") >= F.lit(since))
        .select("timestamp", "value_mwh")
        .toPandas()
        .sort_values("timestamp")
    )
    if not spark.catalog.tableExists(config["weather_table"]):
        logger.warning(f"{config['weather_table']} does not exist: weather_table_missing.")
        return history, pd.DataFrame(columns=["timestamp"]), False
    weather = (
        spark.table(config["weather_table"])
        .filter(F.col("timestamp") > F.lit(first_issued.to_pydatetime()))
        .toPandas()
    )
    if weather.empty:
        logger.warning(f"{config['weather_table']} has no forecasts: weather_table_missing.")
    return history, weather, not weather.empty


def load_model(loader, mlflow_client: MlflowClient) -> tuple[object | None, str | None, str]:
    """(model, run_id, state): state is "ok", NO_PRODUCTION_RUN, or the error's type name.

    A model trained on other features than FEATURES is an error, not a quiet fallback.
    """
    try:
        model, _, run_id = loader(SERVED_MODEL, mlflow_client)
    except ModelNotFoundError:
        return None, None, NO_PRODUCTION_RUN
    except Exception as e:
        logger.error(f"{loader.__name__} failed: {type(e).__name__}: {e}")
        return None, None, type(e).__name__
    trained = model.booster_.feature_name() if hasattr(model, "booster_") else FEATURES
    if list(trained) != list(FEATURES):
        logger.error(f"Run {run_id} was trained on other features: {list(trained)}")
        return None, None, "FeatureMismatch"
    return model, run_id, "ok"


def predict_one(model, features: pd.DataFrame) -> tuple[float | None, str | None]:
    """(prediction, error type name) for one feature row."""
    try:
        return float(np.clip(model.predict(features[FEATURES])[0], 0, None)), None
    except Exception as e:
        logger.error(f"Prediction failed: {type(e).__name__}: {e}")
        return None, type(e).__name__


# COMMAND ----------

# SECTION 4 — SERVE EVERY PLANNED TARGET
# ────────────────────────────────────────


def serve_targets(
    plan: list,
    history: pd.DataFrame,
    weather: pd.DataFrame,
    weather_present: bool,
    mlflow_client: MlflowClient,
    forecast_run_at: datetime,
    config: dict,
) -> tuple[list[dict], list]:
    """Gold rows and (planned, decision) pairs for every planned target (spec 019b).

    One build_features call for all nominal issuance hours, the same call as
    training. decide_serving() picks model / naive + reason / skip per target.
    """
    if not plan:
        return [], []
    feats = build_features(history, weather, [p.issued_at for p in plan], include_target=False)
    model, run_id, model_state = load_model(load_production_model, mlflow_client)
    shadow: dict = {}
    created_at = datetime.now(UTC)
    rows, decisions = [], []

    def row(model_name, version, planned, predicted, served, reason=None):
        return gold_row(
            model_name=model_name,
            model_version=version,
            target=planned.target,
            issued_at=planned.issued_at,
            is_catch_up=planned.is_catch_up,
            predicted_mwh=predicted,
            is_served=served,
            forecast_run_at=forecast_run_at,
            pipeline_run_id=config["pipeline_run_id"],
            created_at=created_at,
            fallback_reason=reason,
            horizon_hours=HORIZON_HOURS,
        )

    for i, planned in enumerate(plan):
        x = feats.iloc[[i]]
        missing = [c for c in REQUIRED_FEATURES if pd.isna(x[c].iloc[0])]
        naive_value = x[WEEKLY_NAIVE].iloc[0]
        daily_value = x[DAILY_NAIVE].iloc[0]
        prediction, prediction_error = None, None
        if model_state == "ok" and weather_present and not missing:
            prediction, prediction_error = predict_one(model, x)
        decision = decide_serving(
            model_state=model_state,
            weather_table_present=weather_present,
            missing_model_inputs=missing,
            prediction_error=prediction_error,
            naive_available=pd.notna(naive_value),
            daily_naive_available=pd.notna(daily_value),
        )
        decisions.append((planned, decision))

        if decision.kind == MODEL:
            rows.append(row(SERVED_MODEL, run_id, planned, prediction, True))
            if pd.notna(naive_value):
                rows.append(row(NAIVE_MODEL_NAME, NAIVE_MODEL_VERSION, planned, naive_value, False))
        elif decision.kind == NAIVE:
            rows.append(
                row(
                    NAIVE_MODEL_NAME,
                    NAIVE_MODEL_VERSION,
                    planned,
                    naive_value,
                    True,
                    decision.fallback_reason,
                )
            )
            if weather_present and not missing:
                if not shadow:
                    shadow_model, shadow_run, shadow_state = load_model(
                        load_shadow_model, mlflow_client
                    )
                    shadow.update(model=shadow_model, run=shadow_run, state=shadow_state)
                if shadow["state"] == "ok":
                    value, error = predict_one(shadow["model"], x)
                    if error is None:
                        rows.append(row(SERVED_MODEL, shadow["run"], planned, value, False))
        elif decision.kind == DAILY_NAIVE_DECISION:
            rows.append(
                row(
                    DAILY_NAIVE_MODEL_NAME,
                    DAILY_NAIVE_MODEL_VERSION,
                    planned,
                    daily_value,
                    True,
                    decision.fallback_reason,
                )
            )
        else:
            logger.warning(
                f"No forecast for {planned.target.isoformat()}: {decision.fallback_reason} "
                "and no T-168h or T-48h value for a naive fallback."
            )
    return rows, decisions


# COMMAND ----------

# SECTION 5 — WRITE FORECASTS TO DELTA
# ───────────────────────────────────────


def write_forecasts(forecasts_df: pd.DataFrame, spark, config: dict, is_backfill: bool = False):
    """Idempotent write using MERGE."""
    schema = StructType(
        [
            StructField("forecast_id", StringType(), False),
            StructField("timestamp", TimestampType(), False),
            StructField("forecast_run_at", TimestampType(), False),
            StructField("model_name", StringType(), False),
            StructField("model_version", StringType(), False),
            StructField("horizon_hours", IntegerType(), False),
            StructField("predicted_mwh", DoubleType(), False),
            StructField("actual_mwh", DoubleType(), True),
            StructField("is_backfilled", BooleanType(), False),
            StructField("pipeline_run_id", StringType(), False),
            StructField("created_at", TimestampType(), False),
            StructField("is_served", BooleanType(), True),  # spec 017; NULL before it
            # spec 019b; NULL on older rows (legacy)
            StructField("issued_at", TimestampType(), True),
            StructField("is_catch_up", BooleanType(), True),
            StructField("fallback_reason", StringType(), True),
        ]
    )

    if not spark.catalog.tableExists(config["forecast_table"]):
        spark.createDataFrame([], schema).write.format("delta").saveAsTable(
            config["forecast_table"]
        )
    else:
        existing = set(spark.table(config["forecast_table"]).columns)
        missing = [f"{c} {t}" for c, t in ADDED_GOLD_COLUMNS.items() if c not in existing]
        if missing:
            spark.sql(f"ALTER TABLE {config['forecast_table']} ADD COLUMNS ({', '.join(missing)})")

    if forecasts_df.empty:
        return
    sdf = spark.createDataFrame(forecasts_df[GOLD_COLUMNS], schema)
    target = DeltaTable.forName(spark, config["forecast_table"])
    condition = "target.forecast_id = source.forecast_id"

    merge_builder = target.alias("target").merge(sdf.alias("source"), condition)
    if is_backfill:
        merge_builder = merge_builder.whenMatchedUpdateAll()
    merge_builder.whenNotMatchedInsertAll().execute()


# COMMAND ----------

# SECTION 6 — RETROACTIVE ACTUAL FILL
# ──────────────────────────────────────


def backfill_actuals(spark, config: dict) -> int:
    """Updates gold_forecasts with actuals from silver_features."""
    merge_sql = f"""
    MERGE INTO {config["forecast_table"]} AS target
    USING {config["silver_table"]} AS source
    ON target.timestamp = source.timestamp
    AND target.actual_mwh IS NULL
    AND source.value_mwh IS NOT NULL
    WHEN MATCHED THEN UPDATE SET target.actual_mwh = source.value_mwh
    """
    spark.sql(merge_sql)
    try:
        history = spark.sql(f"DESCRIBE HISTORY {config['forecast_table']} LIMIT 1").collect()[0]
        return int(history["operationMetrics"].get("numTargetRowsUpdated", 0))
    except Exception:
        return 0


# COMMAND ----------

# SECTION 7 — MAIN ORCHESTRATION
# ────────────────────────────────────────────────

spark.sql(f"USE CATALOG {CATALOG}")
mlflow_client = MlflowClient()
forecast_run_at = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
now = issuance_hour(forecast_run_at)

covered = read_covered(spark, CONFIG, now)
plan = plan_targets(now, covered, FORECAST_HORIZON_HOURS)
rows, decisions = [], []
if plan:
    history_pd, weather_pd, weather_present = load_serving_inputs(spark, CONFIG, plan[0].issued_at)
    rows, decisions = serve_targets(
        plan, history_pd, weather_pd, weather_present, mlflow_client, forecast_run_at, CONFIG
    )
write_forecasts(
    pd.DataFrame(rows, columns=GOLD_COLUMNS), spark, CONFIG, is_backfill=CONFIG["force_backfill"]
)
summary = summarize_decisions(decisions, len(covered))
logger.info(f"Targets ({now.isoformat()}, +24h]: {summary}; wrote {len(rows)} rows")

backfilled_count = backfill_actuals(spark, CONFIG)
# A model error wrote its fallback above; now fail the task so the alert goes out (D4).
raise_on_model_errors(decisions)
dbutils.notebook.exit(json.dumps({"status": "SUCCESS", **summary}))
