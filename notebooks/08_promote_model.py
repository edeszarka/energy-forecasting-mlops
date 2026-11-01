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
# **Purpose:** Apply 07_evaluate's Champion/Challenger decision via MLflow run tags (spec 011).
# **Inputs:**
# - `workspace.energy_forecasting.model_evaluation`
# - `workspace.energy_forecasting.drift_control`
# - `/Volumes/workspace/energy_forecasting/data/flags/retrain_requested.flag`
# **Outputs:**
# - MLflow Model Registry stage transitions
# - `workspace.energy_forecasting.promotion_log` (Audit trail)
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

from src.config import CATALOG, CURRENT_RUN_FILTER, PATHS, RETRAIN_FLAG_PATH
from src.promotion import BENCHMARK_ONLY_REASON, is_promotable, pick_production_run

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

    eval_sdf = spark.table(config["eval_table"])
    if "comparison_method" not in eval_sdf.columns:
        logger.info("Evaluation table has no spec-011 rows yet. Skipping.")
        return pd.DataFrame()

    # Pre-spec-011 rows were never marked processed (UPDATE used a non-existent
    # column) and their challenger_wins came from a biased comparison — ignore them.
    eval_df = (
        eval_sdf.filter(~F.col("promoted"))
        .filter(F.col("comparison_method").isNotNull())
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


def get_production_runs(model_name: str, mlflow_client: MlflowClient) -> list:
    """All runs currently tagged production=true for a model (normally exactly one)."""
    return mlflow_client.search_runs(
        experiment_ids=[r.experiment_id for r in mlflow_client.search_experiments()],
        filter_string=(
            f"tags.model_name = '{model_name}' AND tags.production = 'true' "
            "AND tags.leak_affected = 'false' "
            f"AND {CURRENT_RUN_FILTER}"
        ),
        max_results=50,
    )


def decide_promotions(
    eval_df: pd.DataFrame, mlflow_client: MlflowClient, config: dict
) -> list[dict]:
    """
    Acts on 07_evaluate's same-window champion/challenger decision. If 07 found the
    champion better, nothing changes in production.
    """
    decisions = []

    for _, row in eval_df.iterrows():
        model_name = row["model_name"]
        eval_champion_id = row["champion_run_id"] if pd.notna(row["champion_run_id"]) else None

        try:
            current = pick_production_run(get_production_runs(model_name, mlflow_client))
        except Exception as e:
            logger.warning(f"Error querying production run for {model_name}: {e}")
            continue
        current_id = current.info.run_id if current else None

        reason = row["decision_reason"] if pd.notna(row["decision_reason"]) else ""
        if not is_promotable(model_name):
            # Checked before force_promote: a benchmark can never be tagged (ADR-0002).
            should_promote, reason = False, BENCHMARK_ONLY_REASON
        elif config["force_promote"]:
            should_promote, reason = True, "Force promotion requested via widget"
        elif current_id == row["challenger_run_id"]:
            should_promote, reason = False, "Challenger is already production — nothing to do"
        elif current_id != eval_champion_id:
            # Production changed after 07 ran; its comparison is against a stale champion.
            should_promote = False
            reason = (
                f"Production run changed since evaluation ({eval_champion_id} -> {current_id})"
                " — skipping, re-evaluate on the next retrain"
            )
        else:
            should_promote = bool(row["challenger_wins"])

        decisions.append(
            {
                "model_name": model_name,
                "challenger_run_id": row["challenger_run_id"],
                "challenger_mape": row["challenger_mape"],
                "champion_run_id": current_id,
                "champion_mape": row["champion_mape"],
                "should_promote": should_promote,
                "first_run": bool(row["first_run"]),
                "promotion_reason": reason,
            }
        )

    return decisions


# COMMAND ----------

# SECTION 4 — EXECUTE PROMOTIONS (Tag-based Workaround)
# ──────────────────────────────────────────────────────


def execute_promotions(decisions: list, mlflow_client: MlflowClient, config: dict) -> list[dict]:
    """
    Tags the winning run production=true, then demotes every other production run
    of that model. Tagging the new run first means a crash in between leaves two
    production runs (resolved by pick_production_run's promoted_at ordering)
    rather than none.
    """
    if config["dry_run"]:
        logger.info("DRY RUN: Skipping actual MLflow tag updates.")
        return decisions

    for d in decisions:
        if not d["should_promote"]:
            continue
        if not is_promotable(d["model_name"]):
            # Last line of defence: decide_promotions should never let this through.
            logger.error(f"Refusing to tag non-promotable model {d['model_name']} (ADR-0002).")
            d["should_promote"] = False
            continue

        try:
            mlflow_client.set_tag(
                d["challenger_run_id"], "promoted_at", datetime.now(UTC).isoformat()
            )
            mlflow_client.set_tag(d["challenger_run_id"], "production", "true")

            for run in get_production_runs(d["model_name"], mlflow_client):
                if run.info.run_id != d["challenger_run_id"]:
                    mlflow_client.set_tag(run.info.run_id, "production", "false")
                    logger.info(f"Former champion {run.info.run_id} tagged production=false")

            logger.info(
                f"Successfully promoted {d['model_name']} (Run {d['challenger_run_id']}) via MLflow tags."
            )

        except Exception as e:
            logger.error(f"Failed to tag {d['model_name']}: {e}")
            d["should_promote"] = False
            d["promotion_failed"] = True

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
                WHERE challenger_run_id = '{d["challenger_run_id"]}'
            """)
            logger.info(f"Marked run {d['challenger_run_id']} as processed in eval table.")
        except Exception as e:
            # Unmarked rows get re-processed on every run — fail loudly instead.
            raise RuntimeError(
                f"Failed to update eval table for run {d['challenger_run_id']}: {e}"
            ) from e


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


def read_flag_file(config: dict) -> dict:
    """Reads the retrain request written by 09_error_monitor (if any; spec 018)."""
    p = Path(config["flag_path"])
    meta = {"triggered": False, "features": ""}

    if p.exists():
        try:
            with open(p) as f:
                content = json.load(f)
            meta = {"triggered": True, "features": content.get("drifted_features", [])}
        except Exception as e:
            logger.error(f"Failed to read flag file: {e}")

    return meta


def cleanup_flag_file(config: dict) -> None:
    """Deletes the retrain request. Called only at the very end of a successful
    retrain, so a failed retrain leaves it for 03's pending-timeout to expire."""
    p = Path(config["flag_path"])
    if p.exists() and not config["dry_run"]:
        p.unlink()
        logger.info("Retrain flag file consumed and deleted.")


# COMMAND ----------

# SECTION 8 — MAIN ORCHESTRATION
# ────────────────────────────────────────────────

spark.sql(f"USE CATALOG {CATALOG}")
client = MlflowClient()

if CONFIG["dry_run"]:
    logger.info("DRY RUN MODE — no MLflow transitions or Delta writes will occur.")

# Load candidates
eval_pdf = load_latest_evaluation(spark, CONFIG)
drift_metadata = read_flag_file(CONFIG)

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
    final_decisions = []
    logger.info("No unpromoted models to process.")

cleanup_flag_file(CONFIG)

if any(d.get("promotion_failed") for d in final_decisions):
    raise RuntimeError("One or more promotions failed to apply MLflow tags — see log above.")

if CONFIG["dry_run"]:
    dbutils.notebook.exit("DRY_RUN_COMPLETE")
else:
    dbutils.notebook.exit("SUCCESS")
