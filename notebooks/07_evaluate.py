# Databricks notebook source
# %% [markdown]
# # 07_evaluate
# **Purpose:** Compare new Challenger models against current Production versions.
# **Inputs:** MLflow Model Registry, `workspace.energy_forecasting.silver_features`
# **Outputs:** `workspace.energy_forecasting.model_evaluation` Delta table
# **Last Updated:** 2024-05-21

# COMMAND ----------

import logging
import json
from datetime import datetime, timezone
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F
from pyspark.sql.types import *

# COMMAND ----------

dbutils.widgets.text("mape_improvement_threshold", "0.01")
THRESHOLD = float(dbutils.widgets.get("mape_improvement_threshold"))

CONFIG = {
    "eval_table": "workspace.energy_forecasting.model_evaluation"
}

# COMMAND ----------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("07_evaluate")
client = MlflowClient()

# COMMAND ----------

def get_model_metrics(model_name: str, stage: str):
    """Retrieves metrics for a specific model version from MLflow."""
    try:
        latest_versions = client.get_latest_versions(model_name, stages=[stage])
        if not latest_versions:
            # Check for "None" if Staging/Production is empty (useful for first runs)
            latest_versions = client.get_latest_versions(model_name, stages=["None"])
            if not latest_versions:
                return None
            
        version = latest_versions[0]
        run = client.get_run(version.run_id)
        metrics = run.data.metrics
        return {
            "run_id": version.run_id,
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "mape": metrics.get("mape"),
            "n_train": int(metrics.get("n_train", 0)),
            "n_test": int(metrics.get("n_test", 0))
        }
    except Exception as e:
        logger.warning(f"Error fetching {stage} version for {model_name}: {e}")
        return None

# COMMAND ----------

# Main execution
spark.sql("USE CATALOG workspace")
spark.sql("CREATE DATABASE IF NOT EXISTS energy_forecasting")

model_names = ["energy_prophet_24h", "energy_prophet_168h", "energy_lgbm_24h", "energy_lgbm_168h"]
eval_rows = []

for name in model_names:
    horizon = 24 if "24h" in name else 168
    
    # Challenger is the latest version (usually in "None" or "Staging")
    challenger = get_model_metrics(name, "None")
    # Champion is in "Production"
    champion = get_model_metrics(name, "Production")
    
    first_run = champion is None
    challenger_wins = False
    
    if first_run:
        challenger_wins = True
        logger.info(f"First run detected for {name}. Challenger wins by default.")
    elif challenger and champion:
        # Check if challenger is significantly better
        improvement = (champion["mape"] - challenger["mape"]) / champion["mape"]
        if improvement > THRESHOLD:
            challenger_wins = True
            logger.info(f"{name}: Challenger wins (Improvement: {improvement:.2%})")
        else:
            logger.info(f"{name}: Champion stays (Improvement: {improvement:.2%}, Threshold: {THRESHOLD:.2%})")

    if challenger:
        eval_rows.append({
            "model_name": name,
            "horizon_hours": horizon,
            "challenger_run_id": challenger["run_id"],
            "challenger_mae": challenger["mae"],
            "challenger_rmse": challenger["rmse"],
            "challenger_mape": challenger["mape"],
            "champion_run_id": champion["run_id"] if champion else None,
            "champion_mae": champion["mae"] if champion else None,
            "champion_rmse": champion["rmse"] if champion else None,
            "champion_mape": champion["mape"] if champion else None,
            "challenger_wins": challenger_wins,
            "first_run": first_run,
            "evaluated_at": datetime.now(timezone.utc),
            "promoted": False # Placeholder for notebook 08
        })

if not eval_rows:
    raise ValueError("No challenger models found to evaluate.")

# Write results to Delta
eval_df = spark.createDataFrame(eval_rows)
eval_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(CONFIG["eval_table"])

# Print recommendation summary
print("\nModel Evaluation Summary:")
pdf = eval_df.toPandas()
print(pdf[["model_name", "challenger_mape", "champion_mape", "challenger_wins"]].to_string(index=False))

recommendations = []
for _, row in pdf.iterrows():
    status = "PROMOTE" if row['challenger_wins'] else "SKIP"
    msg = f"{status}: {row['model_name']} (MAPE: {row['challenger_mape']:.2f}% vs {row['champion_mape'] if row['champion_mape'] else 'N/A'})"
    recommendations.append(msg)

print("\nFinal Recommendations:")
for rec in recommendations:
    print(rec)

dbutils.notebook.exit("SUCCESS")
