# Databricks notebook source
# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

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
# # 08_promote_model
# **Purpose:** Automate model promotion from Staging/None to Production using Champion/Challenger logic.
# **Inputs:**
# - `workspace.energy_forecasting.model_evaluation`
# - `workspace.energy_forecasting.drift_control`
# - `/Volumes/workspace/energy_forecasting/data/flags/retrain_requested.flag`
# **Outputs:**
# - MLflow Model Registry stage transitions
# - `workspace.energy_forecasting.promotion_log` (Audit trail)
# **Last Updated:** 2024-05-21
#
# **Required:** mlflow>=2.12.0, pandas, pyspark

# COMMAND ----------

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from pyspark.sql.window import Window

from src.config import CATALOG, PATHS, RETRAIN_FLAG_PATH

# COMMAND ----------

# SECTION 1 — SETUP AND CONFIG
# ─────────────────────────────

dbutils.widgets.text("mape_improvement_threshold", "0.01")
dbutils.widgets.text("dry_run", "false")
dbutils.widgets.text("force_promote", "false")

CONFIG = {
    "eval_table": PATHS.table_eval,
    "promotion_log_table": PATHS.table_promotion,
    "flag_path": RETRAIN_FLAG_PATH,
    "model_names": [
        "energy_lgbm_24h",
        "energy_lgbm_168h",
        "energy_prophet_24h",
        "energy_prophet_168h",
    ],
    "mape_threshold": float(dbutils.widgets.get("mape_improvement_threshold")),
    "dry_run": dbutils.widgets.get("dry_run").lower() == "true",
    "force_promote": dbutils.widgets.get("force_promote").lower() == "true",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("promote_model")

# COMMAND ----------

# SECTION 2 — LOAD EVALUATION RESULTS
# ──────────────────────────────────────


def load_latest_evaluation(spark: SparkSession, config: dict) -> pd.DataFrame:
    """
    Loads the most recent evaluation results for each model from
    model_evaluation table where promoted = False.
    """
    if not spark.catalog.tableExists(config["eval_table"]):
        logger.warning(f"Evaluation table {config['eval_table']} does not exist yet. Skipping.")
        return pd.DataFrame()

    # Use window function to get latest evaluated_at per model_name
    window = Window.partitionBy("model_name").orderBy(F.col("evaluated_at").desc())

    eval_df = (
        spark.table(config["eval_table"])
        .filter(~F.col("promoted"))
        .withColumn("rn", F.row_number().over(window))
        .filter(F.col("rn") == 1)
        .drop("rn")
        .toPandas()
    )

    if eval_df.empty:
        logger.info("No unpromoted challengers found in model_evaluation table.")

    return eval_df


# COMMAND ----------

# SECTION 3 — PROMOTION DECISION LOGIC
# ───────────────────────────────────────


def decide_promotions(
    eval_df: pd.DataFrame, mlflow_client: MlflowClient, config: dict
) -> list[dict]:
    """
    Applies Champion/Challenger rules using MLflow Run tags (production=true).
    """
    decisions = []

    for _, row in eval_df.iterrows():
        model_name = row["model_name"]
        challenger_mape = row["challenger_mape"]
        challenger_run_id = row["challenger_run_id"]

        # Get current 'Production' run by searching tags
        try:
            prod_runs = mlflow_client.search_runs(
                experiment_ids=[r.experiment_id for r in mlflow_client.search_experiments()],
                filter_string=f"tags.model_name = '{model_name}' AND tags.production = 'true'",
                max_results=1,
            )
            champion = prod_runs[0] if prod_runs else None

            if champion:
                champion_mape = champion.data.metrics.get("mape")
                champion_run_id = champion.info.run_id
            else:
                champion_mape = None
                champion_run_id = None

        except Exception as e:
            logger.warning(f"Error querying production run for {model_name}: {e}")
            continue

        # Decision Logic
        first_run = champion_mape is None or champion_mape == 0.0
        should_promote = False
        reason = ""

        if config["force_promote"]:
            should_promote = True
            reason = "Force promotion requested via widget"
        elif first_run:
            should_promote = True
            reason = "No Production version exists — promoting unconditionally"
        elif challenger_mape is None or pd.isna(challenger_mape):
            should_promote = False
            reason = "Invalid challenger metrics — promotion skipped"
        else:
            # Relative improvement comparison
            threshold = config["mape_threshold"]
            improvement = (champion_mape - challenger_mape) / champion_mape

            if improvement > threshold:
                should_promote = True
                reason = f"MAPE improved by {improvement:.2%} ({champion_mape:.3f}% -> {challenger_mape:.3f}%)"
            else:
                should_promote = False
                reason = f"Insufficient improvement ({improvement:.2%} < {threshold:.2%}). Keeping champion."

        decisions.append(
            {
                "model_name": model_name,
                "challenger_run_id": challenger_run_id,
                "challenger_mape": challenger_mape,
                "champion_run_id": champion_run_id,
                "champion_mape": champion_mape,
                "should_promote": should_promote,
                "first_run": first_run,
                "promotion_reason": reason,
            }
        )

    return decisions


# COMMAND ----------

# SECTION 4 — EXECUTE PROMOTIONS (Tag-based Workaround)
# ──────────────────────────────────────────────────────


def execute_promotions(decisions: list, mlflow_client: MlflowClient, config: dict) -> list[dict]:
    """
    Simulates promotion by tagging winning runs with production=true and
    removing that tag from former champions.
    """
    if config["dry_run"]:
        logger.info("DRY RUN: Skipping actual MLflow tag updates.")
        return decisions

    for d in decisions:
        if not d["should_promote"]:
            continue

        try:
            # 1. Remove production tag from old champion
            if d["champion_run_id"]:
                mlflow_client.set_tag(d["champion_run_id"], "production", "false")
                logger.info(f"Former champion {d['champion_run_id']} tagged production=false")

            # 2. Tag new challenger as production
            mlflow_client.set_tag(d["challenger_run_id"], "production", "true")

            # Add metadata tags for prediction/drift logic
            mlflow_client.set_tag(
                d["challenger_run_id"], "promoted_at", datetime.now(UTC).isoformat()
            )

            logger.info(
                f"Successfully promoted {d['model_name']} (Run {d['challenger_run_id']}) via MLflow tags."
            )

        except Exception as e:
            logger.error(f"Failed to tag {d['model_name']}: {e}")
            d["should_promote"] = False

    return decisions


# COMMAND ----------

# SECTION 5 — UPDATE EVALUATION TABLE
# ──────────────────────────────────────


def mark_promoted_in_eval_table(decisions: list, spark: SparkSession, config: dict) -> None:
    """Updates the model_evaluation table to mark records as processed."""
    if config["dry_run"]:
        return

    for d in decisions:
        try:
            # We mark the run as processed regardless of whether it was promoted or skipped
            spark.sql(f"""
                UPDATE {config["eval_table"]}
                SET promoted = true
                WHERE run_id = '{d["challenger_run_id"]}'
            """)
            logger.info(f"Marked run {d['challenger_run_id']} as processed in eval table.")
        except Exception as e:
            logger.error(f"Failed to update eval table for run {d['challenger_run_id']}: {e}")


# COMMAND ----------

# SECTION 6 — WRITE PROMOTION AUDIT LOG
# ────────────────────────────────────────


def write_promotion_log(
    decisions: list, spark: SparkSession, config: dict, drift_meta: dict
) -> None:
    """Appends decisions to the permanent audit log."""
    if config["dry_run"] or not decisions:
        return

    log_rows = []
    now = datetime.now(UTC)

    for d in decisions:
        # Deterministic ID for idempotency
        p_id = hashlib.md5(
            f"{d['model_name']}_{d['challenger_run_id']}_{now.isoformat()}".encode()
        ).hexdigest()

        log_rows.append(
            {
                "promotion_id": p_id,
                "promoted_at": now,
                "model_name": d["model_name"],
                "challenger_run_id": d["challenger_run_id"],
                "challenger_version": d.get("challenger_version"),
                "challenger_mape": float(d["challenger_mape"]) if d["challenger_mape"] else None,
                "champion_run_id": d.get("champion_run_id"),
                "champion_version": d.get("champion_version"),
                "champion_mape": float(d["champion_mape"]) if d.get("champion_mape") else None,
                "promotion_reason": d["promotion_reason"],
                "first_run": bool(d["first_run"]),
                "drift_triggered": bool(drift_meta["triggered"]),
                "drifted_features": str(drift_meta["features"]),
                "promoted_by": "automated_pipeline",
                "created_at": now,
            }
        )

    # Define schema explicitly to avoid inference errors with nulls
    LOG_SCHEMA = StructType(
        [
            StructField("promotion_id", StringType(), False),
            StructField("promoted_at", TimestampType(), False),
            StructField("model_name", StringType(), False),
            StructField("challenger_run_id", StringType(), False),
            StructField("challenger_version", StringType(), True),
            StructField("challenger_mape", DoubleType(), True),
            StructField("champion_run_id", StringType(), True),
            StructField("champion_version", StringType(), True),
            StructField("champion_mape", DoubleType(), True),
            StructField("promotion_reason", StringType(), True),
            StructField("first_run", BooleanType(), False),
            StructField("drift_triggered", BooleanType(), False),
            StructField("drifted_features", StringType(), True),
            StructField("promoted_by", StringType(), False),
            StructField("created_at", TimestampType(), False),
        ]
    )

    # DDL for log table
    spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {config["promotion_log_table"]} (
        promotion_id STRING,
        promoted_at TIMESTAMP,
        model_name STRING,
        challenger_run_id STRING,
        challenger_version STRING,
        challenger_mape DOUBLE,
        champion_run_id STRING,
        champion_version STRING,
        champion_mape DOUBLE,
        promotion_reason STRING,
        first_run BOOLEAN,
        drift_triggered BOOLEAN,
        drifted_features STRING,
        promoted_by STRING,
        created_at TIMESTAMP
    ) USING DELTA
    """)

    spark.createDataFrame(log_rows, schema=LOG_SCHEMA).write.format("delta").mode(
        "append"
    ).saveAsTable(config["promotion_log_table"])
    logger.info(f"Written {len(log_rows)} rows to promotion audit log.")


# COMMAND ----------

# SECTION 7 — CLEANUP FLAG FILE
# ────────────────────────────────


def cleanup_flag_file(config: dict) -> dict:
    """Reads and deletes the retrain flag file."""
    p = Path(config["flag_path"])
    meta = {"triggered": False, "features": ""}

    if p.exists():
        try:
            with open(p) as f:
                content = json.load(f)
                meta = {"triggered": True, "features": content.get("drifted_features", [])}

            if not config["dry_run"]:
                p.unlink()
                logger.info("Retrain flag file consumed and deleted.")
        except Exception as e:
            logger.error(f"Failed to process/delete flag file: {e}")

    return meta


# COMMAND ----------

# SECTION 8 — MAIN ORCHESTRATION
# ────────────────────────────────────────────────

spark.sql(f"USE CATALOG {CATALOG}")
client = MlflowClient()

if CONFIG["dry_run"]:
    logger.info("DRY RUN MODE — no MLflow transitions or Delta writes will occur.")

# Load candidates
eval_pdf = load_latest_evaluation(spark, CONFIG)
drift_metadata = cleanup_flag_file(CONFIG)

if not eval_pdf.empty:
    # Logic
    promo_decisions = decide_promotions(eval_pdf, client, CONFIG)

    # MLflow
    final_decisions = execute_promotions(promo_decisions, client, CONFIG)

    # Persistence
    mark_promoted_in_eval_table(final_decisions, spark, CONFIG)
    write_promotion_log(final_decisions, spark, CONFIG, drift_metadata)

    # Print Summary Table for Job Output
    print("\n" + "=" * 64)
    print(" MODEL PROMOTION SUMMARY")
    print("=" * 64)
    print(
        pd.DataFrame(final_decisions)[
            ["model_name", "should_promote", "promotion_reason"]
        ].to_string(index=False)
    )
    print("=" * 64 + "\n")
else:
    logger.info("No unpromoted models to process.")

if CONFIG["dry_run"]:
    dbutils.notebook.exit("DRY_RUN_COMPLETE")
else:
    dbutils.notebook.exit("SUCCESS")
