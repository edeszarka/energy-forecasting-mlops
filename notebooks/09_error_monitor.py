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
# MAGIC **Retrain trigger** (spec 018, ADR-0007): the served LightGBM run is judged on the hours it
# MAGIC served itself. When it is worse than naive on the same hours, or above the MAPE ceiling, for
# MAGIC 24 consecutive hourly checks (with at least 72 own hours, a 72h cooldown and no pending
# MAGIC request), `energy_retraining_pipeline` is started fire-and-forget. Data drift never triggers.
# MAGIC
# MAGIC **Inputs:** `workspace.energy_forecasting.gold_forecasts`
# MAGIC
# MAGIC **Outputs:** `workspace.energy_forecasting.forecast_error_monitor`, the retrain flag

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
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from src.config import (
    ERROR_ALERT_WINDOW_HOURS,
    ERROR_WINDOWS_HOURS,
    PATHS,
    RETRAIN_FLAG_PATH,
)
from src.error_monitoring import SERVED_STREAM, build_monitor_rows, new_failing_alerts
from src.error_trigger import assess_degradation, decide_error_retrain, start_retrain
from src.retrain_trigger import read_pending_since, trigger_retraining_job, write_retrain_flag

# Job id of energy_retraining_pipeline; injected by databricks.yml. Empty = flag only.
dbutils.widgets.text("retraining_job_id", "")
RETRAINING_JOB_ID = dbutils.widgets.get("retraining_job_id").strip()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("error_monitor")

GOLD_COLUMNS = [
    "timestamp",
    "model_name",
    "model_version",
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
        # Spec 018: set on the served row of the alert window only.
        StructField("degraded", BooleanType(), True),
        StructField("consecutive_degraded_hours", IntegerType(), True),
        StructField("retrain_triggered", BooleanType(), True),
        StructField("retrain_reason", StringType(), True),
    ]
)
TRIGGER_COLUMNS = {
    "degraded": "BOOLEAN",
    "consecutive_degraded_hours": "INT",
    "retrain_triggered": "BOOLEAN",
    "retrain_reason": "STRING",
}


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


def ensure_monitor_table(spark) -> None:
    """Creates forecast_error_monitor, or adds the spec 018 columns to it."""
    if not spark.catalog.tableExists(PATHS.table_error_monitor):
        spark.createDataFrame([], MONITOR_SCHEMA).write.format("delta").saveAsTable(
            PATHS.table_error_monitor
        )
        return
    existing = set(spark.table(PATHS.table_error_monitor).columns)
    missing = [f"{c} {t}" for c, t in TRIGGER_COLUMNS.items() if c not in existing]
    if missing:
        spark.sql(f"ALTER TABLE {PATHS.table_error_monitor} ADD COLUMNS ({', '.join(missing)})")


def read_history(spark, check_time: datetime) -> tuple[list[str] | None, list, datetime | None]:
    """From earlier served rows of the alert window: the latest check's alerts,
    (check_timestamp, degraded) pairs of the last 72 checks, and the last retrain time."""
    served = (
        spark.table(PATHS.table_error_monitor)
        .filter(F.col("check_timestamp") < F.lit(check_time))
        .filter(F.col("stream") == SERVED_STREAM)
        .filter(F.col("window_hours") == ERROR_ALERT_WINDOW_HOURS)
    )
    recent = (
        served.orderBy(F.col("check_timestamp").desc())
        .limit(72)
        .select("check_timestamp", "alerts", "degraded")
        .collect()
    )
    last_retrain = served.filter(F.col("retrain_triggered")).agg(F.max("check_timestamp")).collect()
    alerts = list(recent[0]["alerts"]) if recent else None
    history = [(r["check_timestamp"], bool(r["degraded"])) for r in recent]
    return alerts, history, last_retrain[0][0]


def write_monitor_rows(spark, rows: list[dict]) -> None:
    """MERGE on (check_timestamp, window_hours, stream), so a retried hour replaces its rows."""
    now = datetime.now(UTC)
    empty = dict.fromkeys(TRIGGER_COLUMNS)
    sdf = spark.createDataFrame([{**empty, **r, "created_at": now} for r in rows], MONITOR_SCHEMA)
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
ensure_monitor_table(spark)
gold_pd = read_gold(spark, check_time)
rows, alerts = build_monitor_rows(gold_pd, pd.Timestamp(check_time))
before, degraded_history, last_retrain_at = read_history(spark, check_time)

# Retrain trigger (spec 018): degradation of the served run on its own hours.
degradation = assess_degradation(gold_pd, pd.Timestamp(check_time))
consecutive, should_retrain, retrain_reason = decide_error_retrain(
    degradation,
    degraded_history,
    pd.Timestamp(check_time),
    last_retrain_at=last_retrain_at,
    pending_since=read_pending_since(RETRAIN_FLAG_PATH),
)
retrain_run_id = None
if should_retrain:
    should_retrain, retrain_reason, retrain_run_id = start_retrain(
        retrain_reason,
        RETRAINING_JOB_ID,
        trigger_retraining_job,
        lambda reason, run_id: write_retrain_flag(RETRAIN_FLAG_PATH, reason, consecutive, run_id),
    )
for r in rows:
    if r["stream"] == SERVED_STREAM and r["window_hours"] == ERROR_ALERT_WINDOW_HOURS:
        r.update(
            degraded=degradation.degraded,
            consecutive_degraded_hours=consecutive,
            retrain_triggered=should_retrain,
            retrain_reason=retrain_reason,
        )
write_monitor_rows(spark, rows)

for r in rows:
    mape = f"{r['mape']:.2f}%" if r["mape"] is not None else "n/a"
    skill = f"{r['relative_skill']:+.3f}" if r["relative_skill"] is not None else "n/a"
    logger.info(
        f"{r['window_hours']:>4}h {r['stream']:<12} {r['model_name'] or '-':<16} "
        f"n={r['n']:<4} coverage={r['coverage']:.2f} MAPE={mape} skill={skill}"
    )
logger.info(f"Alerts ({ERROR_ALERT_WINDOW_HOURS}h): {alerts or 'none'} (previous: {before})")
logger.info(
    f"Retrain: {'TRIGGERED' if should_retrain else 'no'} "
    f"(degraded {consecutive}h, job run {retrain_run_id or 'n/a'}): {retrain_reason}"
)

failing = new_failing_alerts(alerts, before)
if failing:
    raise ForecastErrorAlert(
        f"Forecast error alert started: {', '.join(failing)}. "
        f"See {PATHS.table_error_monitor} at {check_time.isoformat()}."
    )

dbutils.notebook.exit(
    json.dumps({"status": "SUCCESS", "alerts": alerts, "retrain_triggered": should_retrain})
)
