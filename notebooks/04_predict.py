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
# **Purpose:** Forecast the hour T = τ + 24h, τ being this run's issuance hour (spec 019,
# ADR-0008), with the production LightGBM run, on point-in-time features built by the same
# function as training (`src/pit_features.build_features`). Seasonal naive (T − 168h) is
# written every hour and served when no production run can be used (ADR-0002); while it
# serves, the latest current-feature-set run is scored in shadow (spec 017). Also backfills
# actuals into gold.
# **Inputs:** `silver_features` (actual load), `bronze_weather_forecast`, MLflow runs
# **Outputs:** `workspace.energy_forecasting.gold_forecasts`

# COMMAND ----------

import hashlib
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

from src.baseline import seasonal_naive_forecast
from src.config import (
    CATALOG,
    CURRENT_RUN_FILTER,
    FEATURE_SET,
    NAIVE_MODEL_NAME,
    PATHS,
)
from src.forecast_timing import SERVED_MODEL, issuance_hour, serving_target_timestamps
from src.gold_rows import NAIVE_MODEL_VERSION, combine_gold_rows
from src.pit_features import FEATURES, build_features
from src.promotion import pick_production_run

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

# Without these the model is not served for the hour: naive serves instead.
REQUIRED_FEATURES = ["load_t_minus_48h", "load_t_minus_168h", "temperature_fc"]
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

# SECTION 3 — FEATURE PREPARATION FOR INFERENCE
# ────────────────────────────────────────────────


def load_serving_inputs(
    spark, config: dict, issued: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load history (for features and the naive fallback) and the weather forecasts for T."""
    since = (issued - pd.Timedelta(hours=HISTORY_HOURS)).to_pydatetime()
    history = (
        spark.table(config["silver_table"])
        .filter(F.col("timestamp") >= F.lit(since))
        .select("timestamp", "value_mwh")
        .toPandas()
        .sort_values("timestamp")
    )
    weather = (
        spark.table(config["weather_table"])
        .filter(F.col("timestamp") > F.lit(issued.to_pydatetime()))
        .toPandas()
    )
    return history, weather


def prepare_features(history: pd.DataFrame, weather: pd.DataFrame, forecast_run_at) -> pd.DataFrame:
    """The serving row for issuance τ: the same build_features call as training (spec 019).

    Raises:
        ValueError: If an input the model needs is missing for this hour.
    """
    feats = build_features(history, weather, [issuance_hour(forecast_run_at)], include_target=False)
    missing = [c for c in REQUIRED_FEATURES if pd.isna(feats[c].iloc[0])]
    if missing:
        raise ValueError(f"Missing point-in-time inputs for this hour: {missing}")
    return feats


# COMMAND ----------

# SECTION 4 — GENERATE FORECASTS
# ─────────────────────────────────


def generate_forecasts(
    model,
    model_name: str,
    model_version: str,
    run_id: str,
    features_df: pd.DataFrame,
    horizon_hours: int,
    forecast_run_at: datetime,
    config: dict,
) -> pd.DataFrame:
    """Gold rows for the target hours in `features_df` (one per issuance)."""
    trained = model.booster_.feature_name() if hasattr(model, "booster_") else FEATURES
    if list(trained) != list(FEATURES):
        raise ValueError(f"Run {run_id} was trained on other features: {list(trained)}")
    preds = np.clip(model.predict(features_df[FEATURES]), a_min=0, a_max=None)

    output_rows = []
    for target, pred in zip(features_df["target_timestamp"], preds, strict=True):
        target_ts = pd.Timestamp(target).tz_localize("UTC")
        # IDEMPOTENCY: MD5(model_name + horizon + target timestamp).
        f_id = hashlib.md5(
            f"{model_name}_{horizon_hours}_{target_ts.isoformat()}".encode()
        ).hexdigest()
        output_rows.append(
            {
                "forecast_id": f_id,
                "timestamp": target_ts,
                "forecast_run_at": forecast_run_at,
                "model_name": model_name,
                "model_version": str(model_version),
                "horizon_hours": horizon_hours,
                "predicted_mwh": float(pred),
                "actual_mwh": None,
                "is_backfilled": False,
                "pipeline_run_id": config["pipeline_run_id"],
                "created_at": datetime.now(UTC),
            }
        )

    return pd.DataFrame(output_rows)


def generate_naive_forecasts(
    history_pd: pd.DataFrame,
    target_timestamps: list,
    horizon_hours: int,
    forecast_run_at: datetime,
    config: dict,
) -> pd.DataFrame:
    """Seasonal-naive gold rows (ADR-0002) for exactly the given target hours."""
    naive = seasonal_naive_forecast(history_pd, target_timestamps)
    skipped = len(target_timestamps) - len(naive)
    if skipped:
        logger.warning(f"Seasonal naive: skipped {skipped} targets with no T-168h value in silver.")

    output_rows = []
    for target_ts, pred in zip(naive["timestamp"], naive["predicted_mwh"], strict=True):
        f_id = hashlib.md5(
            f"{NAIVE_MODEL_NAME}_{horizon_hours}_{target_ts.isoformat()}".encode()
        ).hexdigest()
        output_rows.append(
            {
                "forecast_id": f_id,
                "timestamp": target_ts,
                "forecast_run_at": forecast_run_at,
                "model_name": NAIVE_MODEL_NAME,
                "model_version": NAIVE_MODEL_VERSION,
                "horizon_hours": horizon_hours,
                "predicted_mwh": float(pred),
                "actual_mwh": None,
                "is_backfilled": False,
                "pipeline_run_id": config["pipeline_run_id"],
                "created_at": datetime.now(UTC),
            }
        )

    return pd.DataFrame(output_rows)


def lgbm_forecasts(
    model_loader,
    history_pd: pd.DataFrame,
    weather_pd: pd.DataFrame,
    mlflow_client: MlflowClient,
    horizon_hours: int,
    forecast_run_at: datetime,
    config: dict,
) -> pd.DataFrame:
    """LightGBM rows from the run `model_loader` picks (production or shadow)."""
    model, model_version, run_id = model_loader(SERVED_MODEL, mlflow_client)
    feats = prepare_features(history_pd, weather_pd, forecast_run_at)
    return generate_forecasts(
        model=model,
        model_name=SERVED_MODEL,
        model_version=model_version,
        run_id=run_id,
        features_df=feats,
        horizon_hours=horizon_hours,
        forecast_run_at=forecast_run_at,
        config=config,
    )


def serve_forecasts(
    history_pd: pd.DataFrame,
    weather_pd: pd.DataFrame,
    mlflow_client: MlflowClient,
    horizon_hours: int,
    forecast_run_at: datetime,
    config: dict,
) -> tuple[pd.DataFrame, str]:
    """Production LightGBM forecasts, or seasonal naive if any model-side step fails.

    Returns (forecasts_df, serving_mode). Seasonal-naive rows are written every
    hour (spec 017 T2); `is_served` marks the served rows. While naive serves, the
    latest current-feature-set LightGBM run is scored in shadow; a shadow failure
    only logs. All rows use the same target hour, τ + 24h (serving_target_timestamps).
    Errors that break the naive path while naive serves (no usable silver
    history) still raise.
    """
    targets = serving_target_timestamps(forecast_run_at, horizon_hours)
    naive_error = None
    try:
        naive_df = generate_naive_forecasts(
            history_pd, targets, horizon_hours, forecast_run_at, config
        )
    except ValueError as e:
        naive_error, naive_df = e, pd.DataFrame()

    try:
        model_df = lgbm_forecasts(
            load_production_model,
            history_pd,
            weather_pd,
            mlflow_client,
            horizon_hours,
            forecast_run_at,
            config,
        )
        if naive_error is not None:
            logger.warning(f"No seasonal-naive rows this hour: {naive_error}")
        return combine_gold_rows(naive_df, model_df, model_served=True), SERVED_MODEL
    except Exception as e:
        logger.warning(f"Serving seasonal naive instead of {SERVED_MODEL}: {type(e).__name__}: {e}")

    if naive_error is not None:
        raise naive_error

    try:
        shadow_df = lgbm_forecasts(
            load_shadow_model,
            history_pd,
            weather_pd,
            mlflow_client,
            horizon_hours,
            forecast_run_at,
            config,
        )
    except Exception as e:
        logger.warning(f"Shadow scoring skipped: {type(e).__name__}: {e}")
        shadow_df = pd.DataFrame()
    return combine_gold_rows(naive_df, shadow_df, model_served=False), NAIVE_MODEL_NAME


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
        ]
    )

    if not spark.catalog.tableExists(config["forecast_table"]):
        spark.createDataFrame([], schema).write.format("delta").saveAsTable(
            config["forecast_table"]
        )
    elif "is_served" not in spark.table(config["forecast_table"]).columns:
        spark.sql(f"ALTER TABLE {config['forecast_table']} ADD COLUMNS (is_served BOOLEAN)")

    sdf = spark.createDataFrame(forecasts_df, schema)
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

history_pd, weather_pd = load_serving_inputs(spark, CONFIG, issuance_hour(forecast_run_at))
forecasts_df, serving_mode = serve_forecasts(
    history_pd, weather_pd, mlflow_client, HORIZON_HOURS, forecast_run_at, CONFIG
)
write_forecasts(forecasts_df, spark, CONFIG, is_backfill=CONFIG["force_backfill"])
logger.info(
    f"Horizon {HORIZON_HOURS}h: wrote {len(forecasts_df)} forecast rows (serving_mode={serving_mode})"
)

backfilled_count = backfill_actuals(spark, CONFIG)
dbutils.notebook.exit("SUCCESS")
