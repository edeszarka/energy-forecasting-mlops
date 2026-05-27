# Databricks notebook source
# %% [markdown]
# # 04_predict
# **Purpose:** Load Production models from MLflow, generate 24h and 168h forecasts, and backfill past actuals.
# **Inputs:** `workspace.energy_forecasting.silver_features`, MLflow Model Registry
# **Outputs:** `workspace.energy_forecasting.gold_forecasts`
# **Last Updated:** 2024-05-21
#
# **Required:** mlflow>=2.12.0, lightgbm>=4.3.0, prophet>=1.1.5

# COMMAND ----------

import logging
import json
import hashlib
import uuid
from datetime import datetime, timezone, timedelta
from typing import Tuple, Dict, Any, List, Optional

import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import *
from delta.tables import DeltaTable

# COMMAND ----------

# SECTION 1 — SETUP AND CONFIG
# ─────────────────────────────

dbutils.widgets.text("force_backfill", "false")
dbutils.widgets.text("horizon_hours", "both")

# Unity Catalog Paths
CATALOG = "workspace"
SCHEMA = "energy_forecasting"

CONFIG = {
    "silver_table": f"{CATALOG}.{SCHEMA}.silver_features",
    "forecast_table": f"{CATALOG}.{SCHEMA}.gold_forecasts",
    "feature_columns": [
        'temperature_c', 'lag_24h', 'lag_48h', 'lag_168h',
        'rolling_7d_mean', 'rolling_7d_std', 'rolling_24h_mean',
        'hour_of_day', 'day_of_week', 'month',
        'is_weekend', 'is_holiday'
    ],
    "force_backfill": dbutils.widgets.get("force_backfill").lower() == "true",
    "horizon_hours": dbutils.widgets.get("horizon_hours"),
    "pipeline_run_id": dbutils.notebook.entry_point.getDbutils().notebook().getContext().currentRunId().getOrElse(lambda: "manual")
}

# Hungarian Public Holidays (Static placeholder)
HUNGARIAN_HOLIDAYS = {
    (1, 1), (3, 15), (4, 21), (5, 1), (5, 19), (8, 20), (10, 23), (11, 1), (12, 25), (12, 26),
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("predict")

class ModelNotFoundError(Exception):
    """Raised when a Production model is missing from Registry."""
    pass

# COMMAND ----------

# SECTION 2 — MODEL LOADING WITH FALLBACK
# ─────────────────────────────────────────

def load_production_model(model_name: str, mlflow_client: MlflowClient) -> Tuple[Any, str, str]:
    """Loads the Production version of a model."""
    try:
        version_info = mlflow_client.get_latest_versions(model_name, stages=["Production"])
        if not version_info:
            raise ModelNotFoundError(f"No Production version found for {model_name}.")
        
        version = version_info[0].version
        run_id = version_info[0].run_id
        model_uri = f"models:/{model_name}/Production"
        
        if "lgbm" in model_name:
            model = mlflow.lightgbm.load_model(model_uri)
        elif "prophet" in model_name:
            model = mlflow.prophet.load_model(model_uri)
        else:
            model = mlflow.pyfunc.load_model(model_uri)
            
        logger.info(f"Loaded {model_name} version {version} from Production.")
        return model, version, run_id
    except Exception as e:
        if isinstance(e, ModelNotFoundError): raise
        raise RuntimeError(f"Error loading {model_name}: {e}")

def load_models_with_fallback(config: dict, mlflow_client: MlflowClient) -> Dict[int, Tuple[Any, str, str]]:
    """Tries LGBM, falls back to Prophet if missing."""
    loaded = {}
    for h in [24, 168]:
        primary = f"energy_lgbm_{h}h"
        fallback = f"energy_prophet_{h}h"
        try:
            loaded[h] = load_production_model(primary, mlflow_client)
        except ModelNotFoundError:
            logger.warning(f"{primary} not found, trying fallback {fallback}")
            try:
                loaded[h] = load_production_model(fallback, mlflow_client)
            except ModelNotFoundError:
                raise RuntimeError(f"No Production model available for horizon {h}h.")
    return loaded

# COMMAND ----------

# SECTION 3 — FEATURE PREPARATION FOR INFERENCE
# ────────────────────────────────────────────────

def prepare_inference_features(
    spark: SparkSession,
    config: dict,
    horizon_hours: int,
    forecast_run_at: datetime
) -> pd.DataFrame:
    """Builds future features."""
    history_limit = max(horizon_hours + 7*24, 200)
    history_pd = spark.table(config["silver_table"]) \
        .orderBy(F.col("timestamp").desc()) \
        .limit(history_limit) \
        .toPandas() \
        .sort_values("timestamp")
    
    if len(history_pd) < 168:
        raise ValueError(f"Insufficient history: {len(history_pd)} rows. Need at least 168.")
        
    last_actual = history_pd["value_mwh"].dropna().iloc[-1]
    start_ts = forecast_run_at.replace(minute=0, second=0, microsecond=0)
    future_ts = [start_ts + timedelta(hours=i) for i in range(1, horizon_hours + 1)]
    
    future_rows = []
    for t in future_ts:
        is_holiday = 1 if (t.month, t.day) in HUNGARIAN_HOLIDAYS else 0
        proxy_time = t - timedelta(days=7)
        temp_matches = history_pd[history_pd["timestamp"] == proxy_time]["temperature_c"]
        temp_c = temp_matches.iloc[0] if not temp_matches.empty else history_pd["temperature_c"].iloc[-1]
        
        def get_lag(target_t):
            match = history_pd[history_pd["timestamp"] == target_t]["value_mwh"]
            return match.iloc[0] if not match.empty else last_actual
            
        row = {
            "timestamp": t,
            "hour_of_day": t.hour,
            "day_of_week": t.weekday(),
            "month": t.month,
            "is_weekend": 1 if t.weekday() >= 5 else 0,
            "is_holiday": is_holiday,
            "temperature_c": temp_c,
            "lag_24h": get_lag(t - timedelta(hours=24)),
            "lag_48h": get_lag(t - timedelta(hours=48)),
            "lag_168h": get_lag(t - timedelta(hours=168)),
            "rolling_7d_mean": history_pd["value_mwh"].tail(168).mean(),
            "rolling_7d_std": history_pd["value_mwh"].tail(168).std() or 0.0,
            "rolling_24h_mean": history_pd["value_mwh"].tail(24).mean()
        }
        future_rows.append(row)
        
    return pd.DataFrame(future_rows).set_index("timestamp")

# COMMAND ----------

# SECTION 4 — GENERATE FORECASTS
# ─────────────────────────────────

def generate_forecasts(
    model: Any,
    model_name: str,
    model_version: str,
    run_id: str,
    features_df: pd.DataFrame,
    horizon_hours: int,
    forecast_run_at: datetime,
    config: dict
) -> pd.DataFrame:
    """Inference loop."""
    if "lgbm" in model_name:
        X = features_df[config["feature_columns"]]
        preds = np.clip(model.predict(X), a_min=0, a_max=None)
    else: # Prophet
        p_df = features_df.reset_index().rename(columns={"timestamp": "ds", "temperature_c": "temperature_c"})
        forecast = model.predict(p_df)
        preds = forecast["yhat"].clip(lower=0).values
        
    output_rows = []
    for i, (ts, pred) in enumerate(zip(features_df.index, preds)):
        # IDEMPOTENCY: Deterministic Hash
        f_id = hashlib.md5(f"{model_name}_{horizon_hours}_{ts.isoformat()}".encode()).hexdigest()
        
        output_rows.append({
            "forecast_id": f_id,
            "timestamp": ts,
            "forecast_run_at": forecast_run_at,
            "model_name": model_name,
            "model_version": str(model_version),
            "horizon_hours": horizon_hours,
            "predicted_mwh": float(pred),
            "actual_mwh": None,
            "is_backfilled": False,
            "pipeline_run_id": config["pipeline_run_id"],
            "created_at": datetime.now(timezone.utc)
        })
        
    return pd.DataFrame(output_rows)

# COMMAND ----------

# SECTION 5 — WRITE FORECASTS TO DELTA
# ───────────────────────────────────────

def write_forecasts(forecasts_df: pd.DataFrame, spark: SparkSession, config: dict, is_backfill: bool = False):
    """Idempotent write using MERGE."""
    schema = StructType([
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
        StructField("created_at", TimestampType(), False)
    ])
    
    if not spark.catalog.tableExists(config["forecast_table"]):
        spark.createDataFrame([], schema).write.format("delta").saveAsTable(config["forecast_table"])
        
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

def backfill_actuals(spark: SparkSession, config: dict) -> int:
    """Updates gold_forecasts with actuals from silver_features."""
    merge_sql = f"""
    MERGE INTO {config['forecast_table']} AS target
    USING {config['silver_table']} AS source
    ON target.timestamp = source.timestamp
    AND target.actual_mwh IS NULL
    AND source.value_mwh IS NOT NULL
    AND target.forecast_run_at >= (current_timestamp() - INTERVAL 30 DAYS)
    WHEN MATCHED THEN UPDATE SET target.actual_mwh = source.value_mwh
    """
    spark.sql(merge_sql)
    try:
        history = spark.sql(f"DESCRIBE HISTORY {config['forecast_table']} LIMIT 1").collect()[0]
        return int(history['operationMetrics'].get('numTargetRowsUpdated', 0))
    except: return 0

# COMMAND ----------

# SECTION 7 — MAIN ORCHESTRATION
# ────────────────────────────────────────────────

spark.sql(f"USE CATALOG {CATALOG}")
mlflow_client = MlflowClient()
forecast_run_at = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

h_widget = CONFIG["horizon_hours"]
horizons = [24, 168] if h_widget == "both" else [int(h_widget)]

models_dict = load_models_with_fallback(CONFIG, mlflow_client)

for h in horizons:
    logger.info(f"Starting forecast for {h}h horizon...")
    model, ver, r_id = models_dict[h]
    
    # Determine actual model name from the loaded model metadata
    lgbm_name = f"energy_lgbm_{h}h"
    prophet_name = f"energy_prophet_{h}h"
    
    try:
        lgbm_versions = mlflow_client.get_latest_versions(lgbm_name, stages=["Production"])
        actual_model_name = lgbm_name if lgbm_versions and lgbm_versions[0].run_id == r_id else prophet_name
    except Exception:
        actual_model_name = prophet_name
    
    feats = prepare_inference_features(spark, CONFIG, h, forecast_run_at)
    forecasts_df = generate_forecasts(
        model=model,
        model_name=actual_model_name,
        model_version=ver,
        run_id=r_id,
        features_df=feats,
        horizon_hours=h,
        forecast_run_at=forecast_run_at,
        config=CONFIG
    )
    write_forecasts(forecasts_df, spark, CONFIG, is_backfill=CONFIG["force_backfill"])
    logger.info(f"Horizon {h}h: wrote {len(forecasts_df)} forecast rows using {actual_model_name} v{ver}")

backfilled_count = backfill_actuals(spark, CONFIG)
dbutils.notebook.exit("SUCCESS")

# FIX APPLIED: Corrected model name resolution logic to distinguish between primary (LGBM) and fallback (Prophet) loaded models using run_id comparison.
