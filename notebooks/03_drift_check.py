# Databricks notebook source
# COMMAND ----------

from pathlib import Path

nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
req_path = str(Path("/Workspace" + nb_path).parent.parent / "requirements.txt")

# COMMAND ----------

# MAGIC %pip install -r {req_path}

# COMMAND ----------

# Cell: Environment Setup
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

import subprocess
import sys

check = subprocess.run(
    [sys.executable, "-m", "pip", "show", "evidently"], capture_output=True, text=True
)
print(check.stdout)
verify = subprocess.run(
    [sys.executable, "-c", "from evidently.metric_preset import DataDriftPreset; print('OK')"],
    capture_output=True,
    text=True,
)
if verify.returncode != 0:
    raise RuntimeError(
        f"evidently installed but metric_preset missing. stderr: {verify.stderr}\n"
        "Fix: pin a different evidently version in requirements.txt"
    )
print("evidently verify:", verify.stdout)

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import subprocess
import sys

result = subprocess.run(
    [
        sys.executable,
        "-c",
        "import os, evidently; root = os.path.dirname(evidently.__file__); "
        "print([d for d in os.listdir(root) if 'preset' in d or 'metric' in d])",
    ],
    capture_output=True,
    text=True,
)
print("evidently submodules found:", result.stdout)
print("version:", __import__("evidently").__version__)

# COMMAND ----------

# %% [markdown]
# # 03_drift_check
# **Purpose:** Monitor data drift and prediction drift using Evidently AI.
# **Inputs:** `workspace.energy_forecasting.silver_features`, `workspace.energy_forecasting.gold_forecasts`
# **Outputs:** `workspace.energy_forecasting.drift_control`, HTML reports in Volumes.
# **Last Updated:** 2024-05-21
#
# **Required:** evidently>=0.4.33, mlflow>=2.12.0

# COMMAND ----------

import json
import logging
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import ColumnDriftMetric
from evidently.report import Report
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
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

# Add project root to sys.path for src package imports (after kernel restart)
root_path = str(Path(os.getcwd()).parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from src.config import (
    CURRENT_DATA_VERSION,
    MODEL_INPUT_FEATURES,
    PATHS,
    RETRAIN_FLAG_PATH,
    SCHEMA,
)
from src.drift import count_consecutive_drift_hours, decide_retrain, select_drift_columns
from src.promotion import pick_production_run
from src.retrain_trigger import trigger_retraining_job

# COMMAND ----------

# SECTION 1 — SETUP AND CONFIG
# ─────────────────────────────

# Widgets
dbutils.widgets.text("drift_threshold", "0.15")
dbutils.widgets.text("consecutive_hours_threshold", "3")
dbutils.widgets.text("current_window_days", "7")
dbutils.widgets.text("min_rows_for_drift", "100")
# Job id of energy_retraining_pipeline; injected by databricks.yml. Empty = flag only.
dbutils.widgets.text("retraining_job_id", "")
dbutils.widgets.text("retrain_cooldown_hours", "24")
dbutils.widgets.text("pending_timeout_hours", "6")

CONFIG = {
    "silver_table": PATHS.table_silver,
    "forecast_table": PATHS.table_gold,
    "drift_table": PATHS.table_drift,
    "report_base_path": PATHS.volume_reports,
    "flag_path": RETRAIN_FLAG_PATH,
    "drift_threshold": float(dbutils.widgets.get("drift_threshold")),
    "consecutive_hours_threshold": int(dbutils.widgets.get("consecutive_hours_threshold")),
    "current_window_days": int(dbutils.widgets.get("current_window_days")),
    "min_rows_for_drift": int(dbutils.widgets.get("min_rows_for_drift")),
    "retraining_job_id": dbutils.widgets.get("retraining_job_id").strip(),
    "retrain_cooldown_hours": float(dbutils.widgets.get("retrain_cooldown_hours")),
    "pending_timeout_hours": float(dbutils.widgets.get("pending_timeout_hours")),
    "primary_model_name": "energy_lgbm_24h",
    "feature_columns": MODEL_INPUT_FEATURES,  # sourced from src/config.py
    # Calendar features are excluded from the drift test (spec 011 §1.4)
    "drift_columns": select_drift_columns(MODEL_INPUT_FEATURES),
    "target_column": "value_mwh",
}

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("drift_check")

# COMMAND ----------

# Idempotency Check
# ────────────────
check_time = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)


def check_already_ran(spark, config: dict, check_time: datetime) -> bool:
    """
    Prevents duplicate runs within the same hour to handle job retries gracefully.
    """
    try:
        return (
            spark.table(config["drift_table"])
            .filter(F.col("check_timestamp") == check_time)
            .count()
            > 0
        )
    except Exception:
        return False


if check_already_ran(spark, CONFIG, check_time):
    logger.warning(f"Drift check for {check_time} already exists. Skipping.")
    dbutils.notebook.exit("ALREADY_RAN")

# COMMAND ----------

# SECTION 2 — LOAD REFERENCE WINDOW
# ────────────────────────────────────


def get_reference_window(
    spark: SparkSession, mlflow_client: MlflowClient, config: dict
) -> tuple[pd.DataFrame, str, datetime, datetime]:
    """
    Returns (reference_df, source_description, start_date, end_date).
    """
    try:
        # Promotion is tag-based (no Model Registry on Free Edition), so look up the
        # production run the same way 04_predict does (spec 011 §1.7).
        runs = mlflow_client.search_runs(
            experiment_ids=[e.experiment_id for e in mlflow_client.search_experiments()],
            filter_string=(
                f"tags.model_name = '{config['primary_model_name']}' "
                "AND tags.production = 'true' AND tags.leak_affected = 'false' "
                f"AND tags.data_version = '{CURRENT_DATA_VERSION}'"
            ),
            max_results=50,
        )
        run = pick_production_run(runs)
        if run is None:
            raise LookupError(f"No production run for {config['primary_model_name']}")

        # Notebooks 05/06 log the training window as tags
        training_end = datetime.fromisoformat(run.data.tags["training_data_end"]).replace(
            tzinfo=UTC
        )
        training_start = datetime.fromisoformat(run.data.tags["training_data_start"]).replace(
            tzinfo=UTC
        )

        source = f"mlflow_training_window (run {run.info.run_id})"
    except Exception as e:
        logger.warning(f"Could not retrieve MLflow Production window: {e}. Using fallback.")
        training_end = datetime.now(UTC) - timedelta(days=7)
        training_start = training_end - timedelta(days=30)
        source = "fallback_30d"

    # Load data
    ref_df = (
        spark.table(config["silver_table"])
        .filter(F.col("timestamp").between(training_start, training_end))
        .select(["timestamp", config["target_column"]] + config["feature_columns"])
        .toPandas()
    )

    return ref_df, source, training_start, training_end


# COMMAND ----------

# SECTION 3 — LOAD CURRENT WINDOW
# ──────────────────────────────────


def get_current_window(
    spark: SparkSession, config: dict
) -> tuple[pd.DataFrame, datetime, datetime]:
    """
    Returns (current_df, window_start, window_end).
    """
    window_end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    window_start = window_end - timedelta(days=config["current_window_days"])

    current_df = (
        spark.table(config["silver_table"])
        .filter(F.col("timestamp").between(window_start, window_end))
        .select(["timestamp", config["target_column"]] + config["feature_columns"])
        .toPandas()
    )

    if len(current_df) < config["min_rows_for_drift"]:
        raise ValueError(
            f"Current window has only {len(current_df)} rows, minimum is {config['min_rows_for_drift']}."
        )

    return current_df, window_start, window_end


# COMMAND ----------

# SECTION 4 — DATA DRIFT DETECTION
# ───────────────────────────────────


def run_data_drift(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, config: dict
) -> dict[str, Any]:
    """
    Runs Evidently DataDriftPreset and returns results.
    """
    report = Report(
        metrics=[
            DataDriftPreset(
                columns=[config["target_column"]] + config["drift_columns"],
                drift_share=config["drift_threshold"],
            ),
            ColumnDriftMetric(column_name=config["target_column"]),
            ColumnDriftMetric(column_name="temperature_c"),
        ]
    )

    report.run(reference_data=reference_df, current_data=current_df)
    report_dict = report.as_dict()

    try:
        # Evidently v0.4.x / v0.5.x path compatibility
        metrics = report_dict["metrics"]

        # DatasetDriftMetric (index 0) — aggregate drift summary
        dataset_drift = metrics[0]["result"]["dataset_drift"]
        n_drifted_features = metrics[0]["result"]["number_of_drifted_columns"]

        # DataDriftTable (index 1) — per-column drift details
        drift_by_columns = metrics[1]["result"]["drift_by_columns"]
        drifted_features = [col for col, info in drift_by_columns.items() if info["drift_detected"]]

        # ColumnDriftMetric (index 2) — target column drift score
        drift_score_target = metrics[2]["result"]["drift_score"]

        # ColumnDriftMetric (index 3) — temperature drift score
        drift_score_temp = metrics[3]["result"]["drift_score"]

    except (KeyError, IndexError) as e:
        logger.error(f"Error parsing Evidently results: {e}")
        # Generic fallback extraction logic could be added here
        raise

    return {
        "report": report,
        "dataset_drift": dataset_drift,
        "n_drifted_features": n_drifted_features,
        "drifted_features": drifted_features,
        "drift_score_value_mwh": drift_score_target,
        "drift_score_temp": drift_score_temp,
    }


# COMMAND ----------

# SECTION 5 — PREDICTION DRIFT DETECTION
# ─────────────────────────────────────────


def run_prediction_drift(
    spark: SparkSession,
    config: dict,
    window_start: datetime,
    window_end: datetime,
    reference_start: datetime,
    reference_end: datetime,
) -> dict[str, Any]:
    """
    Computes MAE degradation.
    """

    def get_mae(start, end):
        df = (
            spark.table(config["forecast_table"])
            .filter(F.col("model_name") == config["primary_model_name"])
            .filter(F.col("horizon_hours") == 24)
            .filter(F.col("actual_mwh").isNotNull())
            .filter(F.col("forecast_run_at").between(start, end))
        )

        count = df.count()
        if count < 24:
            logger.warning(f"Insufficient actuals for window {start}-{end}: {count} rows.")
            return None

        return df.select(F.mean(F.abs(F.col("predicted_mwh") - F.col("actual_mwh")))).collect()[0][
            0
        ]

    mae_current = get_mae(window_start, window_end)
    mae_reference = get_mae(reference_start, reference_end)

    drift_detected = False
    if mae_current and mae_reference:
        # 20% degradation threshold
        drift_detected = mae_current > (mae_reference * 1.20)

    return {
        "prediction_drift_detected": drift_detected,
        "mae_current": mae_current,
        "mae_reference": mae_reference,
    }


# COMMAND ----------

# SECTION 6 — CONSECUTIVE DRIFT COUNTER
# ────────────────────────────────────────


def get_consecutive_drift_hours(
    spark: SparkSession, config: dict, check_time: datetime, current_any_drift: bool
) -> int:
    """
    Counts consecutive HOURLY drift checks; a missing hour breaks the streak.
    """
    try:
        history = (
            spark.table(config["drift_table"])
            .orderBy(F.col("check_timestamp").desc())
            .limit(72)
            .select("check_timestamp", "any_drift_detected")
            .collect()
        )
        pairs = [(row[0], row[1]) for row in history]
    except Exception as e:
        logger.warning(f"Could not read drift history: {e}")
        pairs = []

    return count_consecutive_drift_hours(pairs, check_time, current_any_drift)


# COMMAND ----------

# SECTION 7 — RETRAINING TRIGGER
# ─────────────────────────────────


def read_pending_since(config: dict) -> datetime | None:
    """Returns when the outstanding retrain request was written, if any."""
    path = Path(config["flag_path"])
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return datetime.fromisoformat(json.load(f)["triggered_at"])
    except Exception as e:
        # An unreadable flag must not block triggering forever.
        logger.warning(f"Unreadable retrain flag {path}: {e}. Treating as stale.")
        return None


def should_trigger_retrain(
    spark: SparkSession,
    config: dict,
    consecutive_hours: int,
    any_drift_detected: bool,
    drifted_features: list[str],
) -> tuple[bool, str]:
    """
    Enforces trigger conditions, cooldown and pending-request timeout.
    """
    last_retrain_at = None
    try:
        last_retrain = (
            spark.table(config["drift_table"])
            .filter(F.col("retrain_triggered"))
            .orderBy(F.col("check_timestamp").desc())
            .limit(1)
            .select("check_timestamp")
            .collect()
        )
        if last_retrain:
            last_retrain_at = last_retrain[0][0]
    except Exception as e:
        logger.warning(f"Could not read last retrain time (cooldown not applied): {e}")

    should, reason = decide_retrain(
        any_drift=any_drift_detected,
        consecutive_hours=consecutive_hours,
        consecutive_hours_threshold=config["consecutive_hours_threshold"],
        now=datetime.now(UTC),
        last_retrain_at=last_retrain_at,
        cooldown_hours=config["retrain_cooldown_hours"],
        pending_since=read_pending_since(config),
        pending_timeout_hours=config["pending_timeout_hours"],
    )
    if should and drifted_features:
        reason = f"{reason} Drifted features: {', '.join(drifted_features)}."
    return should, reason


# COMMAND ----------

# SECTION 8 — WRITE FLAG FILE
# ──────────────────────────────


def write_retrain_flag(
    config: dict,
    reason: str,
    consecutive_hours: int,
    drifted_features: list,
    job_run_id: int | None = None,
):
    """
    Writes JSON flag file to Volume. It marks the request as pending until
    08_promote_model deletes it (or it ages past pending_timeout_hours).
    """
    flag_data = {
        "triggered_at": datetime.now(UTC).isoformat(),
        "reason": reason,
        "consecutive_drift_hours": consecutive_hours,
        "drifted_features": drifted_features,
        "job_run_id": job_run_id,
        "pipeline_version": "2.0",
    }

    path = Path(config["flag_path"])
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(flag_data, f, indent=2)

    return str(path)


# COMMAND ----------

# SECTION 9 — SAVE EVIDENTLY HTML REPORT
# ─────────────────────────────────────────


def save_html_report(report: Report, config: dict, check_time: datetime) -> str:
    """
    Saves HTML report and cleans up old ones (>30 days).
    """
    report_dir = Path(config["report_base_path"])
    report_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"drift_{check_time.strftime('%Y%m%d_%H')}.html"
    file_path = report_dir / file_name

    report.save_html(str(file_path))

    # Cleanup
    for old_file in report_dir.glob("*.html"):
        if (datetime.now() - datetime.fromtimestamp(old_file.stat().st_mtime)).days > 30:
            old_file.unlink()

    return str(file_path)


# COMMAND ----------

# SECTION 10 — MAIN ORCHESTRATION
# ──────────────────────────────────

mlflow_client = MlflowClient()
spark.sql(f"CREATE DATABASE IF NOT EXISTS {SCHEMA}")

# Create table with explicit schema
drift_control_schema = StructType(
    [
        StructField("check_timestamp", TimestampType(), False),
        StructField("window_start", TimestampType(), False),
        StructField("window_end", TimestampType(), False),
        StructField("data_drift_detected", BooleanType(), False),
        StructField("prediction_drift_detected", BooleanType(), False),
        StructField("any_drift_detected", BooleanType(), False),
        StructField("n_drifted_features", IntegerType(), False),
        StructField("drifted_features", StringType(), True),
        StructField("drift_score_value_mwh", DoubleType(), True),
        StructField("drift_score_temp", DoubleType(), True),
        StructField("prediction_mae_current", DoubleType(), True),
        StructField("prediction_mae_reference", DoubleType(), True),
        StructField("consecutive_drift_hours", IntegerType(), False),
        StructField("retrain_triggered", BooleanType(), False),
        StructField("report_path", StringType(), True),
        StructField("created_at", TimestampType(), False),
    ]
)

# Load windows
ref_df, source_desc, ref_start, ref_end = get_reference_window(spark, mlflow_client, CONFIG)
logger.info(f"Loaded reference window from {source_desc}: {ref_start} to {ref_end}")

curr_df, window_start, window_end = get_current_window(spark, CONFIG)
logger.info(f"Loaded current window: {window_start} to {window_end}")

# Run monitoring
data_drift = run_data_drift(ref_df, curr_df, CONFIG)
pred_drift = run_prediction_drift(spark, CONFIG, window_start, window_end, ref_start, ref_end)

any_drift = data_drift["dataset_drift"] or pred_drift["prediction_drift_detected"]
consecutive_hours = get_consecutive_drift_hours(spark, CONFIG, check_time, any_drift)

# Trigger Retraining
should_retrain, retrain_reason = should_trigger_retrain(
    spark, CONFIG, consecutive_hours, any_drift, data_drift["drifted_features"]
)

retrain_run_id = None
if should_retrain:
    if CONFIG["retraining_job_id"]:
        try:
            retrain_run_id, msg = trigger_retraining_job(int(CONFIG["retraining_job_id"]))
            logger.info(msg)
        except Exception as e:
            # Not raising: predict depends on this task. With retrain_triggered=False
            # and no flag written, the next hourly check simply retries.
            logger.error(f"Failed to start retraining job: {e}")
            should_retrain = False
            retrain_reason = f"Trigger failed: {e}"
    else:
        logger.warning(
            "retraining_job_id not set — writing flag only (weekly retrain picks it up)."
        )

if should_retrain:
    write_retrain_flag(
        CONFIG, retrain_reason, consecutive_hours, data_drift["drifted_features"], retrain_run_id
    )
    logger.info(f"RETRAINING TRIGGERED: {retrain_reason}")
else:
    logger.info(f"No retraining triggered. Reason: {retrain_reason}")

# Report
report_path = save_html_report(data_drift["report"], CONFIG, check_time)

# Log Result to Delta
result_row = {
    "check_timestamp": check_time,
    "window_start": window_start,
    "window_end": window_end,
    "data_drift_detected": bool(data_drift["dataset_drift"]),
    "prediction_drift_detected": bool(pred_drift["prediction_drift_detected"]),
    "any_drift_detected": bool(any_drift),
    "n_drifted_features": int(data_drift["n_drifted_features"]),
    "drifted_features": ",".join(data_drift["drifted_features"]),
    "drift_score_value_mwh": float(data_drift["drift_score_value_mwh"]),
    "drift_score_temp": float(data_drift["drift_score_temp"]),
    "prediction_mae_current": float(pred_drift["mae_current"])
    if pred_drift["mae_current"]
    else None,
    "prediction_mae_reference": float(pred_drift["mae_reference"])
    if pred_drift["mae_reference"]
    else None,
    "consecutive_drift_hours": int(consecutive_hours),
    "retrain_triggered": bool(should_retrain),
    "report_path": report_path,
    "created_at": datetime.now(UTC),
}

# Create table if not exists
spark.createDataFrame([], drift_control_schema).write.format("delta").mode("ignore").saveAsTable(
    CONFIG["drift_table"]
)

# Append result
spark.createDataFrame([result_row], drift_control_schema).write.format("delta").mode(
    "append"
).saveAsTable(CONFIG["drift_table"])

# Final Summary
print(f"""
┌─────────────────────────────────────────────────┐
│ DRIFT CHECK SUMMARY — {check_time.strftime("%Y-%m-%d %H:%M")} UTC │
├─────────────────────────────────────────────────┤
│ Data drift detected:       {data_drift["dataset_drift"]}
│ Drifted features (N):      {data_drift["n_drifted_features"]} ({result_row["drifted_features"]})
│ value_mwh drift score:     {data_drift["drift_score_value_mwh"]:.4f}
│ temperature_c drift score:  {data_drift["drift_score_temp"]:.4f}
│ Prediction drift:          {pred_drift["prediction_drift_detected"]}
│ MAE current window:        {result_row["prediction_mae_current"] if result_row["prediction_mae_current"] else "N/A"} MWh
│ MAE reference window:      {result_row["prediction_mae_reference"] if result_row["prediction_mae_reference"] else "N/A"} MWh
│ Consecutive drift hours:   {consecutive_hours}
│ Retrain triggered:         {should_retrain} (job run: {retrain_run_id or "N/A"})
│ Retrain decision:          {retrain_reason}
│ HTML report:               {report_path}
└─────────────────────────────────────────────────┘
""")

dbutils.notebook.exit("SUCCESS")
