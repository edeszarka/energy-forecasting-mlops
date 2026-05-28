# Databricks notebook source
# MAGIC %md
# MAGIC # 01_ingest
# MAGIC 
# MAGIC **Architecture Note:** Due to outbound internet restrictions in Databricks Free Edition, data is fetched externally by GitHub Actions and deposited into Unity Catalog Volumes as JSON files. This notebook validates and ingests that data into bronze Delta tables.
# MAGIC 
# MAGIC **Inputs:**
# MAGIC - JSON files in `/Volumes/workspace/energy_forecasting/raw_ingestion/`
# MAGIC - `run_date`: ISO 8601 timestamp (UTC) for the run.
# MAGIC - `lookback_files`: Number of hourly files to look back.
# MAGIC - `dry_run`: If true, no data is written to Delta tables.
# MAGIC 
# MAGIC **Outputs:**
# MAGIC - `workspace.energy_forecasting.bronze_load`
# MAGIC - `workspace.energy_forecasting.bronze_temperature`
# MAGIC - `workspace.energy_forecasting.ingestion_log`
# MAGIC 
# MAGIC **Schedule:** Hourly, triggered after GitHub Actions upload.

# COMMAND ----------

# Cell 1: Widgets
# Handles runtime parameters and resolves the execution date.

from datetime import datetime, timezone, timedelta
import logging

try:
    dbutils.widgets.text("run_date", "")
    dbutils.widgets.text("dry_run", "false")
    dbutils.widgets.text("lookback_files", "2")
except NameError:
    pass

run_date_raw = dbutils.widgets.get("run_date")
dry_run = dbutils.widgets.get("dry_run").lower() == "true"
lookback_files = int(dbutils.widgets.get("lookback_files"))

if not run_date_raw:
    # Floor to current UTC hour
    run_date = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
else:
    run_date = datetime.fromisoformat(run_date_raw.replace("Z", "+00:00"))

print(f"Resolved run_date: {run_date}")
print(f"Dry run: {dry_run}")
print(f"Lookback files: {lookback_files}")

# COMMAND ----------

# Cell 2: Imports and catalog setup
# Initializes required libraries and sets the Spark environment context.

import json
from pathlib import PurePosixPath
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, TimestampType, DoubleType, StringType, BooleanType
)
from pyspark.sql.utils import AnalysisException
from delta.tables import DeltaTable

from src.config import (
    PATHS, ENTSO_E_ZONE,
    ENV_DATABRICKS_HOST, ENV_DATABRICKS_TOKEN
)

# Setup catalog and schema
spark.sql("USE CATALOG workspace")
spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.energy_forecasting")
spark.conf.set("spark.sql.session.timeZone", "UTC")

logger = logging.getLogger("01_ingest")
logging.basicConfig(level=logging.INFO)

# COMMAND ----------

# Cell 3: Schema definitions
# Defines the expected structure for the JSON files and Delta tables.

BRONZE_SCHEMA = StructType([
    StructField("timestamp", TimestampType(), False),
    StructField("country", StringType(), False),
    StructField("value_mwh", DoubleType(), True),
    StructField("source", StringType(), False),
    StructField("fetched_at", TimestampType(), False)
])

# Final table schema includes metadata added in this notebook
BRONZE_TABLE_SCHEMA = StructType(BRONZE_SCHEMA.fields + [
    StructField("run_id", StringType(), False),
    StructField("is_gap", BooleanType(), False)
])

TEMPERATURE_BRONZE_SCHEMA = StructType([
    StructField("timestamp", TimestampType(), False),
    StructField("temperature_c", DoubleType(), True),
    StructField("is_temp_imputed", BooleanType(), False),
    StructField("source", StringType(), False),
    StructField("fetched_at", TimestampType(), False)
])

# Final table schema includes metadata
TEMPERATURE_TABLE_SCHEMA = StructType(TEMPERATURE_BRONZE_SCHEMA.fields + [
    StructField("run_id", StringType(), False)
])

# COMMAND ----------

# Cell 4: Resolve file paths to ingest
# Identifies which JSON files exist in the Volume for the given lookback window.

VOLUME_LOAD_PATH = "/Volumes/workspace/energy_forecasting/raw_ingestion/load"
VOLUME_TEMP_PATH = "/Volumes/workspace/energy_forecasting/raw_ingestion/temperature"

found_load_files = []
missing_load_files = []
found_temp_files = []
missing_temp_files = []

for i in range(lookback_files):
    target_hour = run_date - timedelta(hours=i)
    filename = target_hour.strftime("%Y-%m-%dT%H-00-00Z") + ".json"
    
    load_path = f"{VOLUME_LOAD_PATH}/{filename}"
    temp_path = f"{VOLUME_TEMP_PATH}/{filename}"
    
    # Check Load file
    try:
        dbutils.fs.ls(load_path)
        found_load_files.append(load_path)
    except:
        missing_load_files.append(load_path)
        logger.warning(f"Load file missing: {load_path}")
        
    # Check Temp file
    try:
        dbutils.fs.ls(temp_path)
        found_temp_files.append(temp_path)
    except:
        missing_temp_files.append(temp_path)
        logger.warning(f"Temperature file missing: {temp_path}")

print(f"Found {len(found_load_files)} of {lookback_files} expected load files.")
print(f"Found {len(found_temp_files)} of {lookback_files} expected temp files.")

if not found_load_files:
    dbutils.notebook.exit(json.dumps({
        "status": "skipped",
        "reason": "no_files_found",
        "expected_paths": [f"{VOLUME_LOAD_PATH}/{run_date.strftime('%Y-%m-%dT%H-00-00Z')}.json"]
    }))

# COMMAND ----------

# Cell 5: Read JSON files into Spark DataFrames
# Loads the raw data from the Volume, enforcing the defined schema.

try:
    load_raw_df = spark.read.option("multiline", "true").schema(BRONZE_SCHEMA).json(found_load_files)
    
    if found_temp_files:
        temp_raw_df = spark.read.option("multiline", "true").schema(TEMPERATURE_BRONZE_SCHEMA).json(found_temp_files)
    else:
        temp_raw_df = spark.createDataFrame([], TEMPERATURE_BRONZE_SCHEMA)
        
except AnalysisException as e:
    logger.error(f"Schema mismatch or JSON corruption detected: {e}")
    dbutils.notebook.exit(json.dumps({"status": "schema_error", "message": str(e)}))

print(f"Loaded {load_raw_df.count()} load rows and {temp_raw_df.count()} temperature rows.")

# COMMAND ----------

# Cell 6: Validate ingested data
# Performs quality checks and adds operational metadata.

# Get run_id
try:
    context = json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson())
    run_id = str(context.get("tags", {}).get("runId", "manual"))
except:
    run_id = "manual"

# Validate and augment Load
load_count = load_raw_df.count()
null_load = load_raw_df.filter(F.col("value_mwh").isNull()).count()
if load_count > 0 and (null_load / load_count) > 0.1:
    logger.warning(f"High gap rate detected: {null_load/load_count:.2%}")

load_processed_df = load_raw_df.dropDuplicates(["timestamp"]) \
    .withColumn("run_id", F.lit(run_id)) \
    .withColumn("is_gap", F.col("value_mwh").isNull())

# Augment Temp
temp_processed_df = temp_raw_df.dropDuplicates(["timestamp"]) \
    .withColumn("run_id", F.lit(run_id))

# COMMAND ----------

# Cell 7: Create Delta tables if not exist
# Initializes the 3-level Unity Catalog tables.

if not dry_run:
    # Load Table
    DeltaTable.createIfNotExists(spark) \
        .tableName("workspace.energy_forecasting.bronze_load") \
        .addColumns(BRONZE_TABLE_SCHEMA) \
        .partitionedBy("country") \
        .property("delta.autoOptimize.optimizeWrite", "true") \
        .property("delta.autoOptimize.autoCompact", "true") \
        .property("delta.enableChangeDataFeed", "true") \
        .execute()
    
    # Temperature Table
    DeltaTable.createIfNotExists(spark) \
        .tableName("workspace.energy_forecasting.bronze_temperature") \
        .addColumns(TEMPERATURE_TABLE_SCHEMA) \
        .property("delta.autoOptimize.optimizeWrite", "true") \
        .property("delta.autoOptimize.autoCompact", "true") \
        .property("delta.enableChangeDataFeed", "true") \
        .execute()

# COMMAND ----------

# Cell 8: MERGE INTO bronze_load
# Idempotently updates the bronze load table.

if not dry_run:
    delta_load = DeltaTable.forName(spark, "workspace.energy_forecasting.bronze_load")
    
    delta_load.alias("target").merge(
        load_processed_df.alias("source"),
        "target.timestamp = source.timestamp AND target.country = source.country"
    ).whenMatchedUpdate(
        condition="source.value_mwh IS NOT NULL",
        set={
            "value_mwh": "source.value_mwh",
            "source": "source.source",
            "fetched_at": "source.fetched_at",
            "run_id": "source.run_id",
            "is_gap": "source.is_gap"
        }
    ).whenNotMatchedInsertAll().execute()
    
    history = delta_load.history(1).collect()[0]
    logger.info(f"Load Merge: {history['operationMetrics'].get('numTargetRowsInserted')} inserted, {history['operationMetrics'].get('numTargetRowsUpdated')} updated.")

# COMMAND ----------

# Cell 9: MERGE INTO bronze_temperature
# Idempotently updates the bronze temperature table.

if not dry_run and not temp_processed_df.isEmpty():
    delta_temp = DeltaTable.forName(spark, "workspace.energy_forecasting.bronze_temperature")
    
    delta_temp.alias("target").merge(
        temp_processed_df.alias("source"),
        "target.timestamp = source.timestamp"
    ).whenMatchedUpdate(
        condition="source.is_temp_imputed = false",
        set={
            "temperature_c": "source.temperature_c",
            "is_temp_imputed": "source.is_temp_imputed",
            "source": "source.source",
            "fetched_at": "source.fetched_at",
            "run_id": "source.run_id"
        }
    ).whenNotMatchedInsertAll().execute()

# COMMAND ----------

# Cell 10: Write ingestion log
# Records the audit trail of the ingestion run.

if not dry_run:
    log_schema = StructType([
        StructField("run_id", StringType(), False),
        StructField("run_date", TimestampType(), False),
        StructField("files_found", IntegerType(), False),
        StructField("files_missing", IntegerType(), False),
        StructField("rows_ingested", IntegerType(), False),
        StructField("null_count", IntegerType(), False),
        StructField("schema_errors", IntegerType(), False),
        StructField("dry_run", BooleanType(), False),
        StructField("written_at", TimestampType(), False)
    ])
    
    log_data = [(
        run_id, run_date, len(found_load_files), len(missing_load_files),
        int(load_count), int(null_load), 0, dry_run, datetime.now(timezone.utc)
    )]
    
    spark.createDataFrame(log_data, log_schema) \
        .withColumn("date", F.to_date("run_date")) \
        .write.format("delta") \
        .mode("append") \
        .partitionBy("date") \
        .saveAsTable("workspace.energy_forecasting.ingestion_log")

# COMMAND ----------

# Cell 11: Archive processed files
# Moves successfully ingested files to an archive folder to prevent reprocessing.

if not dry_run:
    ARCHIVE_LOAD_PATH = "/Volumes/workspace/energy_forecasting/data/raw_ingestion/archive/load"
    ARCHIVE_TEMP_PATH = "/Volumes/workspace/energy_forecasting/data/raw_ingestion/archive/temperature"
    
    dbutils.fs.mkdirs(ARCHIVE_LOAD_PATH)
    dbutils.fs.mkdirs(ARCHIVE_TEMP_PATH)
    
    for f in found_load_files:
        try:
            dbutils.fs.mv(f, f"{ARCHIVE_LOAD_PATH}/{PurePosixPath(f).name}")
        except:
            logger.warning(f"Failed to archive load file: {f}")
            
    for f in found_temp_files:
        try:
            dbutils.fs.mv(f, f"{ARCHIVE_TEMP_PATH}/{PurePosixPath(f).name}")
        except:
            logger.warning(f"Failed to archive temperature file: {f}")

# COMMAND ----------

# Cell 12: Notebook exit
# Terminates with a summary status.

dbutils.notebook.exit(json.dumps({
    "status": "success",
    "files_found": len(found_load_files),
    "files_missing": len(missing_load_files),
    "rows_ingested": int(load_count),
    "dry_run": dry_run,
    "run_id": run_id,
    "run_date": run_date.isoformat(),
}))
