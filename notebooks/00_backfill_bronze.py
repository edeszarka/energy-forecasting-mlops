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
# MAGIC # 00_backfill_bronze
# MAGIC
# MAGIC Replaces `bronze_load` and `bronze_temperature` over `[start, end)` with the chunk files uploaded by the `backfill.yml` workflow (spec 014 T3, ADR-0005). Manual run only; not part of any job.
# MAGIC
# MAGIC **Inputs:**
# MAGIC - JSON files in `/Volumes/workspace/energy_forecasting/data/raw_backfill/{load,temperature}/`
# MAGIC - `start`, `end`: UTC dates (YYYY-MM-DD), `end` exclusive — the same range as the workflow run.
# MAGIC - `mode`: `dry_run` (default) prints counts and data-quality results and writes nothing; `execute` writes.
# MAGIC
# MAGIC Unlike the hourly MERGE, NULL load values replace stored ones here. Data-quality checks are always enforced: any violation fails the notebook before anything is written (ADR-0004).

# COMMAND ----------

# Cell 1: Environment Setup
# Adds the project root to sys.path to allow importing from the 'src' directory.

import os
import sys
from pathlib import Path

root_path = str(Path(os.getcwd()).parent)
if root_path not in sys.path:
    sys.path.append(root_path)

# COMMAND ----------

# Cell 2: Widgets

import json
import logging
from datetime import UTC, datetime
from pathlib import PurePosixPath

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from src.backfill import chunk_within
from src.bronze_write import (
    monthly_row_counts,
    prepare_backfill_load,
    prepare_backfill_temperature,
    replace_where_predicate,
)
from src.config import CATALOG, ENTSO_E_ZONE, PATHS
from src.data_quality import apply_dq_mode, check_hourly_rows, check_load_quality_by_month

try:
    dbutils.widgets.text("start", "")
    dbutils.widgets.text("end", "")
    dbutils.widgets.dropdown("mode", "dry_run", ["dry_run", "execute"])
except NameError:
    pass

start = pd.Timestamp(dbutils.widgets.get("start"), tz="UTC")
end = pd.Timestamp(dbutils.widgets.get("end"), tz="UTC")
mode = dbutils.widgets.get("mode")
if end <= start:
    raise ValueError(f"end {end} must be after start {start}.")

spark.sql(f"USE CATALOG {CATALOG}")
spark.conf.set("spark.sql.session.timeZone", "UTC")

logger = logging.getLogger("00_backfill_bronze")
logging.basicConfig(level=logging.INFO)

print(f"Range: [{start}, {end}), mode: {mode}")

# COMMAND ----------

# Cell 3: Schemas (file layout as in 01_ingest; table layout adds run_id/is_gap)

LOAD_FILE_SCHEMA = StructType(
    [
        StructField("timestamp", TimestampType(), False),
        StructField("country", StringType(), False),
        StructField("value_mwh", DoubleType(), True),
        StructField("source", StringType(), False),
        StructField("fetched_at", TimestampType(), False),
    ]
)
BRONZE_TABLE_SCHEMA = StructType(
    LOAD_FILE_SCHEMA.fields
    + [StructField("run_id", StringType(), False), StructField("is_gap", BooleanType(), False)]
)

TEMPERATURE_FILE_SCHEMA = StructType(
    [
        StructField("timestamp", TimestampType(), False),
        StructField("temperature_c", DoubleType(), True),
        StructField("humidity_pct", DoubleType(), True),
        StructField("cloud_cover_pct", DoubleType(), True),
        StructField("is_weather_imputed", BooleanType(), False),
        StructField("source", StringType(), False),
        StructField("fetched_at", TimestampType(), False),
    ]
)
TEMPERATURE_TABLE_SCHEMA = StructType(
    TEMPERATURE_FILE_SCHEMA.fields + [StructField("run_id", StringType(), False)]
)

# COMMAND ----------

# Cell 4: Functions


def list_json_files(path: str) -> list[str]:
    try:
        return sorted(f.path for f in dbutils.fs.ls(path) if f.path.endswith(".json"))
    except Exception:
        return []


def read_files(spark, files: list[str], schema) -> pd.DataFrame:
    if not files:
        return pd.DataFrame(columns=[f.name for f in schema.fields])
    return spark.read.option("multiline", "true").schema(schema).json(files).toPandas()


def read_bronze_range(spark, table: str, start, end) -> pd.DataFrame:
    return (
        spark.read.table(table)
        .filter(
            (F.col("timestamp") >= start.to_pydatetime())
            & (F.col("timestamp") < end.to_pydatetime())
        )
        .toPandas()
    )


def print_counts(table: str, before: pd.DataFrame, after: pd.DataFrame, label: str) -> None:
    merged = before.merge(after, on="month", how="outer", suffixes=("_before", f"_{label}"))
    for col in merged.columns.drop("month"):
        merged[col] = pd.to_numeric(merged[col]).fillna(0).astype(int)
    print(f"\n{table}: rows per month, before vs {label}")
    print(merged.sort_values("month").to_string(index=False))


def write_bronze_replace(spark, frame: pd.DataFrame, schema, table: str, predicate: str) -> None:
    """Replaces every row matching `predicate` with `frame` (NULL values included)."""
    columns = [f.name for f in schema.fields]
    (
        spark.createDataFrame(frame[columns], schema=schema)
        .write.format("delta")
        .mode("overwrite")
        .option("replaceWhere", predicate)
        .saveAsTable(table)
    )


def backfill_bronze(spark, raw_load, raw_temp, start, end, mode: str, run_id: str) -> dict:
    """Validates the backfill files and, in `execute`, replaces bronze over [start, end)."""
    load_pd = prepare_backfill_load(raw_load, start, end, ENTSO_E_ZONE, run_id)
    temp_pd = prepare_backfill_temperature(raw_temp, start, end, run_id)

    print_counts(
        "bronze_load",
        monthly_row_counts(read_bronze_range(spark, PATHS.table_bronze, start, end), "value_mwh"),
        monthly_row_counts(load_pd, "value_mwh"),
        "files",
    )
    print_counts(
        "bronze_temperature",
        monthly_row_counts(
            read_bronze_range(spark, PATHS.table_bronze_temp, start, end), "temperature_c"
        ),
        monthly_row_counts(temp_pd, "temperature_c"),
        "files",
    )

    violations = check_load_quality_by_month(load_pd, start, end) + [
        f"weather: {v}" for v in check_hourly_rows(temp_pd, start, end)
    ]
    print(f"\nData-quality violations: {len(violations)}")
    for v in violations:
        print(f"  {v}")
    # Always enforced on this path: raises before anything is written.
    apply_dq_mode(violations, "enforce")

    summary = {"mode": mode, "load_rows": len(load_pd), "temperature_rows": len(temp_pd)}
    if mode != "execute":
        print("\nDry run: nothing written.")
        return summary

    write_bronze_replace(
        spark,
        load_pd,
        BRONZE_TABLE_SCHEMA,
        PATHS.table_bronze,
        replace_where_predicate(start, end, ENTSO_E_ZONE),
    )
    write_bronze_replace(
        spark,
        temp_pd,
        TEMPERATURE_TABLE_SCHEMA,
        PATHS.table_bronze_temp,
        replace_where_predicate(start, end),
    )

    print_counts(
        "bronze_load",
        monthly_row_counts(load_pd, "value_mwh"),
        monthly_row_counts(read_bronze_range(spark, PATHS.table_bronze, start, end), "value_mwh"),
        "after",
    )
    return summary


# COMMAND ----------

# Cell 5: Run

try:
    context = json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson())
    run_id = str(context.get("tags", {}).get("runId", "manual"))
except Exception:
    run_id = "manual"

load_files = list_json_files(PATHS.volume_backfill_load)
temp_files = list_json_files(PATHS.volume_backfill_temp)
print(f"Found {len(load_files)} load files and {len(temp_files)} temperature files.")
if not load_files:
    dbutils.notebook.exit(json.dumps({"status": "skipped", "reason": "no_backfill_files"}))

summary = backfill_bronze(
    spark,
    read_files(spark, load_files, LOAD_FILE_SCHEMA),
    read_files(spark, temp_files, TEMPERATURE_FILE_SCHEMA),
    start,
    end,
    mode,
    run_id,
)

# COMMAND ----------

# Cell 6: Archive processed files (execute only)
# One folder per run, so a later backfill of the same range never collides.
# Only chunks entirely inside [start, end) were fully loaded; others stay.

if mode == "execute":
    archive_root = f"{PATHS.volume_backfill_archive}/{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{run_id}"
    archived = 0
    for kind, files in (("load", load_files), ("temperature", temp_files)):
        dbutils.fs.mkdirs(f"{archive_root}/{kind}")
        for f in files:
            name = PurePosixPath(f).name
            if chunk_within(name, start, end):
                dbutils.fs.mv(f, f"{archive_root}/{kind}/{name}")
                archived += 1
    print(f"Archived {archived} files to {archive_root}")

dbutils.notebook.exit(json.dumps({"status": "success", **summary}))
