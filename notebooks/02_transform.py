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

# MAGIC %md
# MAGIC # 02_transform
# MAGIC
# MAGIC This notebook reads raw data from bronze Delta tables, applies feature engineering using the shared `src/features.py` library, and writes the results to the silver features table.
# MAGIC
# MAGIC **Inputs:**
# MAGIC - `workspace.energy_forecasting.bronze_load`
# MAGIC - `workspace.energy_forecasting.bronze_temperature`
# MAGIC
# MAGIC **Outputs:**
# MAGIC - `workspace.energy_forecasting.silver_features`
# MAGIC
# MAGIC **Schedule:** Hourly, following `01_ingest`.
# MAGIC
# MAGIC **Rebuild mode** (manual, spec 014 T4): set `rebuild_start` and `rebuild_end` (UTC, end exclusive) to rebuild silver from bronze in 30-day chunks with `replaceWhere`; afterwards the table's `pipeline.data_version` is set to `CURRENT_DATA_VERSION`.
# MAGIC
# MAGIC **`dq_mode`:** `warn` logs data-quality violations; `enforce` fails before anything is written (ADR-0004).

# COMMAND ----------

# Cell 1: Environment Setup
# Adds the project root to sys.path to allow importing from the 'src' directory.

import os
import sys
from pathlib import Path

# In Databricks, the working directory is the folder containing the notebook.
# We add the parent directory (project root) to sys.path.
root_path = str(Path(os.getcwd()).parent)
if root_path not in sys.path:
    sys.path.append(root_path)

# COMMAND ----------

# Cell 2: Widgets
# Handles runtime parameters for the transformation process.

import logging
from datetime import UTC, datetime, timedelta

try:
    dbutils.widgets.text("lookback_hours", "720")  # must be >= MIN_TRAINING_ROWS (src/config.py)
    dbutils.widgets.text("run_date", "")
    dbutils.widgets.text("dry_run", "false")
    dbutils.widgets.text("rebuild_start", "")  # empty = hourly path
    dbutils.widgets.text("rebuild_end", "")  # exclusive; empty = current UTC hour
    dbutils.widgets.text("dq_mode", "warn")  # warn | enforce (spec 014 T5)
except NameError:
    pass

lookback_hours = int(dbutils.widgets.get("lookback_hours"))
run_date_raw = dbutils.widgets.get("run_date")
dry_run = dbutils.widgets.get("dry_run").lower() == "true"
rebuild_start_raw = dbutils.widgets.get("rebuild_start")
rebuild_end_raw = dbutils.widgets.get("rebuild_end")
dq_mode = dbutils.widgets.get("dq_mode")

if not run_date_raw:
    run_date = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
else:
    run_date = datetime.fromisoformat(run_date_raw.replace("Z", "+00:00"))

print(f"Resolved run_date: {run_date}")
print(f"Lookback hours: {lookback_hours}")
print(f"Dry run: {dry_run}")
print(f"Rebuild: {rebuild_start_raw or '-'} -> {rebuild_end_raw or '-'}")
print(f"DQ mode: {dq_mode}")

# COMMAND ----------

# Cell 2: Imports and config
# Standard imports and initialization of the Spark session context.

import json

import pandas as pd
from delta.tables import DeltaTable
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

from src.bronze_write import replace_where_predicate
from src.config import (
    CATALOG,
    CURRENT_DATA_VERSION,
    ENTSO_E_ZONE,
    LAG_HOURS,
    MIN_TRAINING_ROWS,
    PATHS,
)
from src.data_quality import apply_dq_mode, check_load_quality
from src.data_version import DATA_VERSION_PROPERTY
from src.features import get_feature_columns
from src.silver_build import build_silver_frame, plan_rebuild_chunks

# Ensure UTC consistency
spark.conf.set("spark.sql.session.timeZone", "UTC")
spark.sql(f"USE CATALOG {CATALOG}")

logger = logging.getLogger("02_transform")
logging.basicConfig(level=logging.INFO)

# COMMAND ----------

# Cell 3: Silver schema definition
# Defines the structural contract for the silver feature table.

SILVER_SCHEMA = StructType(
    [
        StructField("timestamp", TimestampType(), False),
        StructField("country", StringType(), False),
        StructField("value_mwh", DoubleType(), True),
        StructField("is_gap", BooleanType(), False),
        StructField("hour_of_day", IntegerType(), False),
        StructField("day_of_week", IntegerType(), False),
        StructField("month", IntegerType(), False),
        StructField("quarter", IntegerType(), False),
        StructField("is_weekend", BooleanType(), False),
        StructField("is_holiday", BooleanType(), False),
        StructField("is_holiday_eve", BooleanType(), False),
        StructField("days_since_epoch", IntegerType(), False),
        StructField("lag_24h", DoubleType(), True),
        StructField("lag_48h", DoubleType(), True),
        StructField("lag_168h", DoubleType(), True),
        StructField("has_lag_gap", BooleanType(), False),
        StructField("rolling_7d_mean", DoubleType(), True),
        StructField("rolling_7d_std", DoubleType(), True),
        StructField("rolling_24h_mean", DoubleType(), True),
        StructField("temperature_c", DoubleType(), True),
        StructField("humidity_pct", DoubleType(), True),
        StructField("cloud_cover_pct", DoubleType(), True),
        StructField("temperature_lag_24h", DoubleType(), True),
        StructField("is_weather_imputed", BooleanType(), False),
        StructField("temp_missing", BooleanType(), False),
        StructField("source", StringType(), True),
        StructField("fetched_at", TimestampType(), True),
        StructField("feature_built_at", TimestampType(), False),
        StructField("run_id", StringType(), False),
    ]
)

# COMMAND ----------

# Cell 4: Resolve run context
# Determines the time window and job identity.

try:
    context = json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson())
    run_id = str(context.get("tags", {}).get("runId", "manual"))
except Exception:
    run_id = "manual"

# We extend the window by max(LAG_HOURS) to ensure we have history for the lags of the first row
max_lag = max(LAG_HOURS)
window_start = run_date - timedelta(hours=lookback_hours + max_lag)
window_end = run_date

logger.info(f"Processing data window: {window_start} to {window_end}")

# COMMAND ----------

# Cell 5: Silver write helpers, shared by the hourly path and the rebuild


def ensure_silver_table(spark) -> None:
    DeltaTable.createIfNotExists(spark).tableName(PATHS.table_silver).addColumns(
        SILVER_SCHEMA
    ).partitionedBy("country").property("delta.autoOptimize.optimizeWrite", "true").property(
        "delta.autoOptimize.autoCompact", "true"
    ).property("delta.enableChangeDataFeed", "true").property(
        "delta.dataSkippingNumIndexedCols", "4"
    ).execute()


def to_silver_spark(spark, frame: pd.DataFrame, run_id: str):
    frame = frame.copy()
    frame["run_id"] = run_id
    frame["country"] = ENTSO_E_ZONE
    # Reorder columns to match schema exactly (prevents Arrow positional errors)
    schema_cols = [f.name for f in SILVER_SCHEMA.fields]
    return spark.createDataFrame(frame[schema_cols], schema=SILVER_SCHEMA)


def read_bronze_window(spark, start, end) -> tuple[pd.DataFrame, pd.DataFrame]:
    """bronze_load and bronze_temperature rows with timestamp in [start, end]."""
    load_pd = (
        spark.read.table(PATHS.table_bronze)
        .filter(F.col("timestamp").between(start, end))
        .filter(F.col("country") == ENTSO_E_ZONE)
        .toPandas()
    )
    temp_pd = (
        spark.read.table(PATHS.table_bronze_temp)
        .filter(F.col("timestamp").between(start, end))
        .toPandas()
    )
    load_pd["timestamp"] = pd.to_datetime(load_pd["timestamp"], utc=True)
    temp_pd["timestamp"] = pd.to_datetime(temp_pd["timestamp"], utc=True)
    return load_pd, temp_pd


def write_silver_replace(spark, frame: pd.DataFrame, start, end, run_id: str) -> None:
    """Replaces silver rows in [start, end) with `frame`."""
    to_silver_spark(spark, frame, run_id).write.format("delta").mode("overwrite").option(
        "replaceWhere", replace_where_predicate(start, end, ENTSO_E_ZONE)
    ).saveAsTable(PATHS.table_silver)


def set_silver_data_version(spark, version: str) -> None:
    spark.sql(
        f"ALTER TABLE {PATHS.table_silver} "
        f"SET TBLPROPERTIES ('{DATA_VERSION_PROPERTY}' = '{version}')"
    )


def rebuild_silver(spark, start, end, dq_mode: str, dry_run: bool, run_id: str) -> list[dict]:
    """Rebuilds silver on [start, end) in chunks (spec 014 T4).

    Each chunk is read with a 168h look-back and a 72h look-ahead margin, and
    only its core range is written. Every chunk is built and checked before the
    first write, so `enforce` cannot leave a half-rebuilt table.
    """
    built = []
    violations = []
    for chunk in plan_rebuild_chunks(start, end):
        load_pd, temp_pd = read_bronze_window(spark, chunk.read_start, chunk.read_end)
        frame = build_silver_frame(load_pd, temp_pd, chunk.core_start, chunk.core_end)
        chunk_violations = check_load_quality(frame, chunk.core_start, chunk.core_end)
        violations += [f"{chunk.core_start:%Y-%m-%d}: {v}" for v in chunk_violations]
        built.append((chunk, frame))
        print(
            f"Chunk {chunk.core_start} -> {chunk.core_end}: {len(frame)} rows, "
            f"{len(chunk_violations)} DQ violations"
        )

    apply_dq_mode(violations, dq_mode)

    summary = [
        {"core_start": c.core_start.isoformat(), "core_end": c.core_end.isoformat(), "rows": len(f)}
        for c, f in built
    ]
    if dry_run:
        return summary

    ensure_silver_table(spark)
    for chunk, frame in built:
        write_silver_replace(spark, frame, chunk.core_start, chunk.core_end, run_id)
    # Only a completed rebuild from verified bronze bumps the version (ADR-0003).
    set_silver_data_version(spark, CURRENT_DATA_VERSION)
    return summary


# COMMAND ----------

# Cell 6: Rebuild mode
# Runs instead of the hourly path when rebuild_start is set.

if rebuild_start_raw:
    rebuild_start = pd.Timestamp(rebuild_start_raw, tz="UTC")
    rebuild_end = (
        pd.Timestamp(rebuild_end_raw, tz="UTC") if rebuild_end_raw else pd.Timestamp(run_date)
    )
    chunks = rebuild_silver(spark, rebuild_start, rebuild_end, dq_mode, dry_run, run_id)
    dbutils.notebook.exit(
        json.dumps(
            {
                "status": "SUCCESS",
                "mode": "rebuild",
                "dry_run": dry_run,
                "data_version": None if dry_run else CURRENT_DATA_VERSION,
                "chunks": chunks,
            }
        )
    )

# COMMAND ----------

# Cell 7: Load bronze data
# Reads raw data from Delta tables filtered by the resolved window.

bronze_load_df = (
    spark.read.table(PATHS.table_bronze)
    .filter(F.col("timestamp").between(window_start, window_end))
    .filter(F.col("country") == ENTSO_E_ZONE)
    .orderBy("timestamp")
)

bronze_temp_df = (
    spark.read.table(PATHS.table_bronze_temp)
    .filter(F.col("timestamp").between(window_start, window_end))
    .orderBy("timestamp")
)

load_count = bronze_load_df.count()
temp_count = bronze_temp_df.count()

logger.info(f"Loaded {load_count} load rows and {temp_count} temperature rows.")

if load_count == 0:
    dbutils.notebook.exit(json.dumps({"status": "skipped", "reason": "no_bronze_data"}))

# COMMAND ----------

# Cell 8: Validate bronze data quality
# Performs sanity checks on the input data before feature engineering.

# Check for gaps
null_load = bronze_load_df.filter(F.col("value_mwh").isNull()).count()
if load_count > 0 and (null_load / load_count) > 0.1:
    logger.warning(f"High gap rate detected: {null_load / load_count:.2%}")

# Check for duplicates
duplicates = bronze_load_df.groupBy("timestamp").count().filter("count > 1").count()
if duplicates > 0:
    logger.warning(f"Found {duplicates} duplicate timestamps. Deduplicating...")

# Deduplicate
bronze_load_df = bronze_load_df.dropDuplicates(["timestamp"])

# COMMAND ----------

# Cell 9: Convert to pandas and build feature matrix
# Bridges Spark and Pandas to utilize the shared feature engineering library.

if load_count < MIN_TRAINING_ROWS:
    msg = f"Insufficient data for feature engineering. Required: {MIN_TRAINING_ROWS}, Actual: {load_count}"
    logger.warning(msg)
    dbutils.notebook.exit(
        json.dumps(
            {
                "status": "SUCCESS",
                "message": msg,
                "rows_written": 0,
                "insufficient_data": True,
            }
        )
    )

load_pd = bronze_load_df.toPandas()
temp_pd = bronze_temp_df.toPandas()

# Ensure TZ awareness
load_pd["timestamp"] = pd.to_datetime(load_pd["timestamp"], utc=True)
temp_pd["timestamp"] = pd.to_datetime(temp_pd["timestamp"], utc=True)

# Build features and keep only the core window. The leading max_lag-hour margin
# lacks full backward context for lag/rolling features within THIS run's fetch
# window. Writing it would silently degrade any previously-good value for the
# same timestamp once a future run's MERGE (Cell 12, whenMatchedUpdateAll)
# overwrites it — every row is guaranteed to age past this margin exactly
# once, so an unfiltered write here permanently corrupts it (see spec 009
# §1.1). The rebuild uses the same function (spec 014 T4).
core_start = pd.Timestamp(window_start) + pd.Timedelta(hours=max_lag)
feature_pd = build_silver_frame(load_pd, temp_pd, core_start)
logger.info(f"Feature matrix (core window): {feature_pd.shape}")

# COMMAND ----------

# Cell 10: Post-feature validation
# Ensures the output of feature engineering meets quality standards.
# build_silver_frame already fails on missing feature columns.

# Data-quality checks on the core window (ADR-0004): `enforce` raises here,
# before the MERGE, so nothing is written.
dq_violations = check_load_quality(
    feature_pd, core_start, pd.Timestamp(window_end) + pd.Timedelta(hours=1)
)
apply_dq_mode(dq_violations, dq_mode)

# Check for excessive NaNs
nan_report = feature_pd[get_feature_columns()].isna().mean()
high_nan = nan_report[nan_report > 0.5]
if not high_nan.empty:
    logger.warning(f"Columns with >50% NaN: {high_nan.to_dict()}")

# COMMAND ----------

# Cell 11: Create silver Delta table if not exists, convert to Spark

if not dry_run:
    ensure_silver_table(spark)

silver_spark_df = to_silver_spark(spark, feature_pd, run_id)

# COMMAND ----------

# Cell 12: MERGE INTO silver
# Idempotently updates the silver table.

if not dry_run:
    delta_silver = DeltaTable.forName(spark, PATHS.table_silver)

    delta_silver.alias("target").merge(
        silver_spark_df.alias("source"),
        "target.timestamp = source.timestamp AND target.country = source.country",
    ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()

    history = delta_silver.history(1).collect()[0]
    history_dict = history.asDict()
    metrics = history_dict.get("operationMetrics", {})
    logger.info(
        f"Silver Merge: {metrics.get('numTargetRowsInserted', 0)} inserted, {metrics.get('numTargetRowsUpdated', 0)} updated."
    )

# COMMAND ----------

# Cell 13: Notebook exit
# Returns metadata about the transformation run.

exit_info = {
    "status": "SUCCESS",
    "message": "Transform complete",
    "rows_written": len(feature_pd),
    "insufficient_data": False,
    "feature_columns": get_feature_columns(),
    "window_start": window_start.isoformat(),
    "window_end": window_end.isoformat(),
    "run_id": run_id,
    "null_rate_summary": feature_pd[get_feature_columns()].isna().mean().to_dict(),
    "dq_mode": dq_mode,
    "dq_violations": dq_violations,
}

dbutils.notebook.exit(json.dumps(exit_info))
