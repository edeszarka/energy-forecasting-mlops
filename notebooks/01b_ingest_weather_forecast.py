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
# MAGIC # 01b_ingest_weather_forecast
# MAGIC
# MAGIC Merges as-issued weather forecasts (spec 019, ADR-0008) into `bronze_weather_forecast`.
# MAGIC Files come from the hourly GitHub ingestion (`raw_ingestion/weather_forecast/`) and the
# MAGIC backfill (`raw_backfill/weather_forecast/`), both written by `src/weather_forecast.py`.
# MAGIC
# MAGIC One row per valid hour; the most recent fetch wins. Processed files are archived.
# MAGIC The table is created at the start of every run if missing, also when there are no
# MAGIC files (spec 019b B3).
# MAGIC
# MAGIC **Inputs:** JSON files in the two folders above
# MAGIC
# MAGIC **Outputs:** `workspace.energy_forecasting.bronze_weather_forecast`

# COMMAND ----------

import os
import sys
from pathlib import Path  # re-import: restartPython() cleared the first cell

root_path = str(Path(os.getcwd()).parent)
if root_path not in sys.path:
    sys.path.append(root_path)

# COMMAND ----------

import json
from datetime import UTC, datetime
from pathlib import PurePosixPath

from delta.tables import DeltaTable
from pyspark.sql.types import DoubleType, StringType, StructField, StructType, TimestampType

from src.config import PATHS
from src.pit_features import WEATHER_COLUMNS
from src.weather_forecast import latest_per_hour

FILE_SCHEMA = StructType(
    [StructField("timestamp", TimestampType(), False)]
    + [StructField(c, DoubleType(), True) for c in WEATHER_COLUMNS]
    + [
        StructField("model", StringType(), True),
        StructField("fetched_at", TimestampType(), False),
    ]
)
SOURCES = [
    (PATHS.volume_raw_weather_forecast, PATHS.volume_archive_weather_forecast),
    (PATHS.volume_backfill_weather_forecast, f"{PATHS.volume_backfill_archive}/weather_forecast"),
]

# COMMAND ----------


def list_json_files(path: str) -> list[str]:
    try:
        return sorted(f.path for f in dbutils.fs.ls(path) if f.path.endswith(".json"))
    except Exception:
        return []


def ensure_table(spark) -> bool:
    """Creates bronze_weather_forecast if missing (spec 019b B3); True if it was created.

    Runs before anything else, also when there are no files, so readers
    (04, 06, 07) always find the table.
    """
    if spark.catalog.tableExists(PATHS.table_weather_forecast):
        return False
    spark.createDataFrame([], FILE_SCHEMA).write.format("delta").saveAsTable(
        PATHS.table_weather_forecast
    )
    return True


def merge_forecasts(spark, rows) -> None:
    """Upsert by valid hour; a row only replaces an older or equal fetch."""
    sdf = spark.createDataFrame(rows, FILE_SCHEMA)
    (
        DeltaTable.forName(spark, PATHS.table_weather_forecast)
        .alias("t")
        .merge(sdf.alias("s"), "t.timestamp = s.timestamp")
        .whenMatchedUpdateAll(condition="s.fetched_at >= t.fetched_at")
        .whenNotMatchedInsertAll()
        .execute()
    )


# COMMAND ----------

created = ensure_table(spark)
files = [(f, archive) for folder, archive in SOURCES for f in list_json_files(folder)]
if not files:
    dbutils.notebook.exit(
        json.dumps({"status": "skipped", "reason": "no_files_found", "table_created": created})
    )

frame = (
    spark.read.option("multiline", "true")
    .schema(FILE_SCHEMA)
    .json([f for f, _ in files])
    .toPandas()
)
rows = latest_per_hour(frame.dropna(subset=["timestamp", "fetched_at"]))
merge_forecasts(spark, rows)

stamp = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
for f, archive in files:
    dbutils.fs.mkdirs(f"{archive}/{stamp}")
    dbutils.fs.mv(f, f"{archive}/{stamp}/{PurePosixPath(f).name}")

dbutils.notebook.exit(
    json.dumps(
        {
            "status": "success",
            "files": len(files),
            "hours": len(rows),
            "first_hour": str(rows["timestamp"].min()),
            "last_hour": str(rows["timestamp"].max()),
        }
    )
)
