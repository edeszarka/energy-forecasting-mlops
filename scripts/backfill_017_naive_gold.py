# Databricks notebook source
# MAGIC %md
# MAGIC # Seasonal-naive history in gold (spec 017 T7, ADR-0006)
# MAGIC
# MAGIC One-time. Writes `seasonal_naive` gold rows reconstructed from silver,
# MAGIC `predicted(T) = actual(T − 168h)`, for target hours from `start` (default: one week after
# MAGIC the first silver hour) up to the first live gold row. Rows are `is_backfilled = true`,
# MAGIC `is_served = NULL`, `forecast_run_at = T − 48h`; `09_error_monitor` never counts them as
# MAGIC served. Actuals are filled by `04_predict`'s hourly actual backfill.
# MAGIC
# MAGIC `dry_run` (default `true`) prints row counts per month and writes nothing. The write is
# MAGIC insert-only on `forecast_id`, so re-running is safe and live rows are never touched.

# COMMAND ----------

import os
import sys
from pathlib import Path

# The script lives in scripts/; add the project root so `src` imports work.
root_path = str(Path(os.getcwd()).parent)
if root_path not in sys.path:
    sys.path.append(root_path)

# COMMAND ----------

from datetime import UTC, datetime

import pandas as pd
from delta.tables import DeltaTable
from pyspark.sql import functions as F

from src.config import NAIVE_MODEL_NAME, PATHS
from src.gold_rows import NAIVE_LAG_HOURS, reconstruct_naive_gold

HORIZON_HOURS = 24

# COMMAND ----------

dbutils.widgets.text("dry_run", "true")
dbutils.widgets.text("start", "")  # empty = first silver hour + 168h
dry_run = dbutils.widgets.get("dry_run").lower() != "false"

silver = (
    spark.table(PATHS.table_silver)
    .select("timestamp", "value_mwh")
    .filter(F.col("value_mwh").isNotNull())
    .toPandas()
)
silver["timestamp"] = pd.to_datetime(silver["timestamp"], utc=True)
first_silver, last_silver = silver["timestamp"].min(), silver["timestamp"].max()

start_raw = dbutils.widgets.get("start")
start = (
    pd.Timestamp(start_raw, tz="UTC")
    if start_raw
    else first_silver + pd.Timedelta(hours=NAIVE_LAG_HOURS)
)

gold = spark.table(PATHS.table_gold)
if "is_served" not in gold.columns:
    raise RuntimeError("gold_forecasts has no is_served column: let 04_predict run once first.")
first_live = (
    gold.filter(~F.col("is_backfilled"))
    .filter(F.col("model_name") == NAIVE_MODEL_NAME)
    .agg(F.min("timestamp"))
    .collect()[0][0]
)
# Only hours that already have an actual, and never past the first live naive row.
end = last_silver + pd.Timedelta(hours=1)
if first_live is not None:
    end = min(end, pd.Timestamp(first_live, tz="UTC"))

rows = reconstruct_naive_gold(
    silver, start, end, HORIZON_HOURS, pipeline_run_id="backfill_017_naive_gold"
)
print(f"{'DRY RUN — no changes' if dry_run else 'EXECUTE'}: target hours [{start}, {end})")
print(f"{len(rows)} seasonal-naive rows")
if not rows.empty:
    print(rows.groupby(rows["timestamp"].dt.strftime("%Y-%m")).size().to_string())

# COMMAND ----------

if not dry_run and not rows.empty:
    # Gold's own schema: actual_mwh and is_served are all NULL, so nothing can be inferred.
    rows = rows.assign(created_at=datetime.now(UTC))[gold.columns]
    sdf = spark.createDataFrame(rows, schema=gold.schema)
    (
        DeltaTable.forName(spark, PATHS.table_gold)
        .alias("target")
        .merge(sdf.alias("source"), "target.forecast_id = source.forecast_id")
        .whenNotMatchedInsertAll()
        .execute()
    )
    written = gold.filter(F.col("pipeline_run_id") == "backfill_017_naive_gold").count()
    print(f"Backfilled naive rows now in gold: {written}")
