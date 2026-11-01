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
# MAGIC # 09_error_monitor
# MAGIC
# MAGIC Live forecast error monitoring (spec 017, ADR-0006). Runs after `04_predict`, which
# MAGIC writes the hour's forecasts and backfills actuals.
# MAGIC
# MAGIC For the 24h, 7d and 30d windows (by target hour) it scores the **served** forecast,
# MAGIC **seasonal naive** and the **shadow LightGBM** run, compares each model with naive on the
# MAGIC same hours, and writes one row per window and stream to `forecast_error_monitor`.
# MAGIC
# MAGIC Alerts are evaluated on the 7d window. When a failing alert **starts** (it was not active
# MAGIC at the previous check), the task fails after writing its rows, so the job's failure email
# MAGIC fires once per episode. `SHADOW_BEATS_NAIVE` is informational and never fails the task.
# MAGIC
# MAGIC **Inputs:** `workspace.energy_forecasting.gold_forecasts`
# MAGIC
# MAGIC **Outputs:** `workspace.energy_forecasting.forecast_error_monitor`

# COMMAND ----------

import os
import sys
from pathlib import Path  # re-import: restartPython() cleared the first cell

root_path = str(Path(os.getcwd()).parent)
if root_path not in sys.path:
    sys.path.append(root_path)

# COMMAND ----------

import json
import logging
from datetime import UTC, datetime, timedelta

import pandas as pd
from delta.tables import DeltaTable
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from src.config import ERROR_ALERT_WINDOW_HOURS, ERROR_WINDOWS_HOURS, PATHS
from src.error_monitoring import SERVED_STREAM, build_monitor_rows, new_failing_alerts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("error_monitor")

GOLD_COLUMNS = [
    "timestamp",
    "model_name",
    "predicted_mwh",
    "actual_mwh",
    "is_served",
    "is_backfilled",
]

MONITOR_SCHEMA = StructType(
    [
        StructField("check_timestamp", TimestampType(), False),
        StructField("window_hours", IntegerType(), False),
        StructField("stream", StringType(), False),
        StructField("model_name", StringType(), True),
        StructField("n", IntegerType(), False),
        StructField("coverage", DoubleType(), False),
        StructField("mape", DoubleType(), True),
        StructField("mae", DoubleType(), True),
        StructField("bias", DoubleType(), True),
        StructField("naive_mape_paired", DoubleType(), True),
        StructField("relative_skill", DoubleType(), True),
        StructField("alerts", ArrayType(StringType()), False),
        StructField("created_at", TimestampType(), False),
    ]
)


class ForecastErrorAlert(Exception):
    """Raised after the monitor rows are written, when a failing alert starts."""


# COMMAND ----------


def read_gold(spark, check_time: datetime) -> pd.DataFrame:
    """Gold rows whose target hour falls in the longest window."""
    since = check_time - timedelta(hours=max(ERROR_WINDOWS_HOURS))
    return (
        spark.table(PATHS.table_gold)
        .filter(F.col("horizon_hours") == 24)
        .filter(F.col("timestamp") > F.lit(since))
        .filter(F.col("timestamp") <= F.lit(check_time))
        .select(*GOLD_COLUMNS)
        .toPandas()
    )


def previous_alerts(spark, check_time: datetime) -> list[str] | None:
    """Alerts of the latest earlier check (served row of the alert window), if any."""
    if not spark.catalog.tableExists(PATHS.table_error_monitor):
        return None
    rows = (
        spark.table(PATHS.table_error_monitor)
        .filter(F.col("check_timestamp") < F.lit(check_time))
        .filter(F.col("stream") == SERVED_STREAM)
        .filter(F.col("window_hours") == ERROR_ALERT_WINDOW_HOURS)
        .orderBy(F.col("check_timestamp").desc())
        .limit(1)
        .select("alerts")
        .collect()
    )
    return list(rows[0]["alerts"]) if rows else None


def write_monitor_rows(spark, rows: list[dict]) -> None:
    """MERGE on (check_timestamp, window_hours, stream), so a retried hour replaces its rows."""
    now = datetime.now(UTC)
    sdf = spark.createDataFrame([{**r, "created_at": now} for r in rows], MONITOR_SCHEMA)
    if not spark.catalog.tableExists(PATHS.table_error_monitor):
        spark.createDataFrame([], MONITOR_SCHEMA).write.format("delta").saveAsTable(
            PATHS.table_error_monitor
        )
    (
        DeltaTable.forName(spark, PATHS.table_error_monitor)
        .alias("t")
        .merge(
            sdf.alias("s"),
            "t.check_timestamp = s.check_timestamp AND t.window_hours = s.window_hours "
            "AND t.stream = s.stream",
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )


# COMMAND ----------

check_time = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
gold_pd = read_gold(spark, check_time)
rows, alerts = build_monitor_rows(gold_pd, pd.Timestamp(check_time))
before = previous_alerts(spark, check_time)
write_monitor_rows(spark, rows)

for r in rows:
    mape = f"{r['mape']:.2f}%" if r["mape"] is not None else "n/a"
    skill = f"{r['relative_skill']:+.3f}" if r["relative_skill"] is not None else "n/a"
    logger.info(
        f"{r['window_hours']:>4}h {r['stream']:<12} {r['model_name'] or '-':<16} "
        f"n={r['n']:<4} coverage={r['coverage']:.2f} MAPE={mape} skill={skill}"
    )
logger.info(f"Alerts ({ERROR_ALERT_WINDOW_HOURS}h): {alerts or 'none'} (previous: {before})")

failing = new_failing_alerts(alerts, before)
if failing:
    raise ForecastErrorAlert(
        f"Forecast error alert started: {', '.join(failing)}. "
        f"See {PATHS.table_error_monitor} at {check_time.isoformat()}."
    )

dbutils.notebook.exit(json.dumps({"status": "SUCCESS", "alerts": alerts}))
