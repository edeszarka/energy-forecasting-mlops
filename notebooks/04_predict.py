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
# **Purpose:** Load the production LightGBM 24h run from MLflow, generate 24h forecasts, and backfill
# past actuals. If no production run can be used, serve a seasonal-naive forecast (ADR-0002).
# **Inputs:** `workspace.energy_forecasting.silver_features`, MLflow Model Registry
# **Outputs:** `workspace.energy_forecasting.gold_forecasts`
# **Last Updated:** 2024-05-21
#
# **Required:** mlflow>=2.12.0, lightgbm>=4.3.0, prophet>=1.1.5

# COMMAND ----------

import hashlib
import logging
from datetime import UTC, datetime, timedelta

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
from src.config import CATALOG, CURRENT_DATA_VERSION, MODEL_INPUT_FEATURES, PATHS
from src.forecast_timing import (
    SERVED_MODEL,
    resolve_target_timestamp,
    serving_anchor_timestamps,
    serving_target_timestamps,
)
from src.promotion import pick_production_run

# COMMAND ----------

# SECTION 1 — SETUP AND CONFIG
# ─────────────────────────────

dbutils.widgets.text("force_backfill", "false")

HORIZON_HOURS = 24  # ADR-0001: 24h only
NAIVE_MODEL_NAME = "seasonal_naive"
NAIVE_MODEL_VERSION = "seasonal_naive_t-168h"

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
    "forecast_table": PATHS.table_gold,
    "force_backfill": dbutils.widgets.get("force_backfill").lower() == "true",
    "pipeline_run_id": pipeline_run_id,
}

# Hungarian Public Holidays (Static placeholder)
HUNGARIAN_HOLIDAYS = {
    (1, 1),
    (3, 15),
    (4, 21),
    (5, 1),
    (5, 19),
    (8, 20),
    (10, 23),
    (11, 1),
    (12, 25),
    (12, 26),
}

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
            f"AND tags.data_version = '{CURRENT_DATA_VERSION}'"
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
    return model, "run_latest", run_id


# COMMAND ----------

# SECTION 3 — FEATURE PREPARATION FOR INFERENCE
# ────────────────────────────────────────────────


def load_silver_history(spark, config: dict, horizon_hours: int) -> pd.DataFrame:
    """Latest silver rows, shared by the LightGBM features and the seasonal-naive fallback."""
    history_limit = max(horizon_hours + 7 * 24, 200)
    return (
        spark.table(config["silver_table"])
        .orderBy(F.col("timestamp").desc())
        .limit(history_limit)
        .toPandas()
        .sort_values("timestamp")
    )


def prepare_inference_features(
    history_pd: pd.DataFrame, horizon_hours: int, forecast_run_at: datetime
) -> pd.DataFrame:
    """Builds future features."""
    if len(history_pd) < 168:
        raise ValueError(f"Insufficient history: {len(history_pd)} rows. Need at least 168.")

    last_actual = history_pd["value_mwh"].dropna().iloc[-1]
    future_ts = serving_anchor_timestamps(forecast_run_at, horizon_hours)

    future_rows = []
    for t in future_ts:
        is_holiday = 1 if (t.month, t.day) in HUNGARIAN_HOLIDAYS else 0
        proxy_time = t - timedelta(days=7)
        temp_matches = history_pd[history_pd["timestamp"] == proxy_time]["temperature_c"]
        temp_c = (
            temp_matches.iloc[0] if not temp_matches.empty else history_pd["temperature_c"].iloc[-1]
        )
        hum_matches = history_pd[history_pd["timestamp"] == proxy_time]["humidity_pct"]
        humidity_pct = (
            hum_matches.iloc[0] if not hum_matches.empty else history_pd["humidity_pct"].iloc[-1]
        )
        cc_matches = history_pd[history_pd["timestamp"] == proxy_time]["cloud_cover_pct"]
        cloud_cover_pct = (
            cc_matches.iloc[0] if not cc_matches.empty else history_pd["cloud_cover_pct"].iloc[-1]
        )

        def get_lag(target_t):
            match = history_pd[history_pd["timestamp"] == target_t]["value_mwh"]
            return match.iloc[0] if not match.empty else last_actual

        row = {
            "timestamp": t,
            "temperature_c": temp_c,
            "humidity_pct": humidity_pct,
            "cloud_cover_pct": cloud_cover_pct,
            "lag_24h": get_lag(t - timedelta(hours=24)),
            "lag_48h": get_lag(t - timedelta(hours=48)),
            "lag_168h": get_lag(t - timedelta(hours=168)),
            "rolling_7d_mean": history_pd["value_mwh"].tail(168).mean(),
            "rolling_7d_std": history_pd["value_mwh"].tail(168).std() or 0.0,
            "rolling_24h_mean": history_pd["value_mwh"].tail(24).mean(),
            "hour_of_day": t.hour,
            "day_of_week": t.weekday(),
            "month": t.month,
            "is_weekend": 1 if t.weekday() >= 5 else 0,
            "is_holiday": is_holiday,
        }
        future_rows.append(row)

    return pd.DataFrame(future_rows).set_index("timestamp")


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
    """Inference loop."""
    # Sourced from src/config.py — keep in sync with 06_train_lgbm.py
    MODEL_FEATURES = MODEL_INPUT_FEATURES

    # Force alignment and log for debugging
    X = features_df[MODEL_FEATURES].copy()

    # Subset to features the loaded model actually expects (handles
    # old models trained before humidity_pct/cloud_cover_pct were added)
    if "lgbm" in model_name and hasattr(model, "booster_"):
        trained_features = model.booster_.feature_name()
        missing = [c for c in trained_features if c not in X.columns]
        extra = [c for c in X.columns if c not in trained_features]
        if missing:
            logger.warning(f"Model expects features missing from input: {missing}")
        if extra:
            logger.info(f"Dropping {len(extra)} features not in training set: {extra}")
        X = X[[c for c in trained_features if c in X.columns]]

    logger.info(f"Model Input: {X.shape[1]} features. Columns: {list(X.columns)}")

    if "lgbm" in model_name:
        preds = np.clip(model.predict(X), a_min=0, a_max=None)
    else:  # Prophet
        p_df = X.reset_index().rename(columns={"timestamp": "ds"})
        # Prophet rejects tz-aware ds outright; 05_train_prophet.py strips tz
        # at training time (tz_localize(None)) — this path never had the
        # equivalent, and was never exercised against it until spec 009's
        # remediation first forced LGBM-fallback traffic onto Prophet
        # (spec 010 §1.1).
        p_df["ds"] = pd.to_datetime(p_df["ds"]).dt.tz_localize(None)
        forecast = model.predict(p_df)
        preds = forecast["yhat"].clip(lower=0).values

    output_rows = []
    for _i, (anchor_ts, pred) in enumerate(zip(features_df.index, preds, strict=False)):
        target_ts = resolve_target_timestamp(anchor_ts, model_name, horizon_hours)
        # IDEMPOTENCY: Deterministic Hash — keyed on the row's own (target)
        # timestamp, per GEMINI.md's MD5(model_name + horizon + timestamp)
        # convention. Using anchor_ts here would silently reintroduce the bug.
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


def serve_forecasts(
    history_pd: pd.DataFrame,
    mlflow_client: MlflowClient,
    horizon_hours: int,
    forecast_run_at: datetime,
    config: dict,
) -> tuple[pd.DataFrame, str]:
    """Production LightGBM forecasts, or seasonal naive if any model-side step fails.

    Returns (forecasts_df, serving_mode). Both paths write the same target hours
    (serving_target_timestamps). Errors that also break the naive path (no usable
    silver history) still raise.
    """
    try:
        model, ver, run_id = load_production_model(SERVED_MODEL, mlflow_client)
        feats = prepare_inference_features(history_pd, horizon_hours, forecast_run_at)
        forecasts_df = generate_forecasts(
            model=model,
            model_name=SERVED_MODEL,
            model_version=ver,
            run_id=run_id,
            features_df=feats,
            horizon_hours=horizon_hours,
            forecast_run_at=forecast_run_at,
            config=config,
        )
        return forecasts_df, SERVED_MODEL
    except Exception as e:
        logger.warning(f"Serving seasonal naive instead of {SERVED_MODEL}: {type(e).__name__}: {e}")

    targets = serving_target_timestamps(forecast_run_at, horizon_hours)
    forecasts_df = generate_naive_forecasts(
        history_pd, targets, horizon_hours, forecast_run_at, config
    )
    return forecasts_df, NAIVE_MODEL_NAME


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
        ]
    )

    if not spark.catalog.tableExists(config["forecast_table"]):
        spark.createDataFrame([], schema).write.format("delta").saveAsTable(
            config["forecast_table"]
        )

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

history_pd = load_silver_history(spark, CONFIG, HORIZON_HOURS)
forecasts_df, serving_mode = serve_forecasts(
    history_pd, mlflow_client, HORIZON_HOURS, forecast_run_at, CONFIG
)
write_forecasts(forecasts_df, spark, CONFIG, is_backfill=CONFIG["force_backfill"])
logger.info(
    f"Horizon {HORIZON_HOURS}h: wrote {len(forecasts_df)} forecast rows (serving_mode={serving_mode})"
)

backfilled_count = backfill_actuals(spark, CONFIG)
dbutils.notebook.exit("SUCCESS")
