# Databricks notebook source
# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

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

def get_run_metrics(model_name: str, type_filter: str):
    """
    Retrieves metrics for a model from MLflow Runs.
    - If type_filter is 'challenger', returns the latest successful run.
    - If type_filter is 'champion', returns the historic run with lowest MAPE.
    """
    try:
        # Search for runs with this model_name
        # Note: We filter by model_name tag we added in notebooks 05/06
        runs = client.search_runs(
            experiment_ids=[r.experiment_id for r in client.search_experiments()],
            filter_string=f"tags.model_name = '{model_name}'",
            order_by=["attributes.start_time DESC"] if type_filter == 'challenger' else ["metrics.mape ASC"],
            max_results=5
        )
        
        if not runs:
            return None
            
        # For challenger, just take the absolute latest
        # For champion, take the best one that isn't the current challenger
        if type_filter == 'challenger':
            run = runs[0]
        else:
            # The champion is the best performing run that is NOT the one we just trained
            # (Assuming the challenger is the very latest run)
            latest_run_id = runs[0].info.run_id
            candidates = [r for r in runs if r.info.run_id != latest_run_id]
            if not candidates:
                return None
            run = candidates[0]

        metrics = run.data.metrics
        return {
            "run_id": run.info.run_id,
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "mape": metrics.get("mape"),
            "n_train": int(metrics.get("n_train", 0)),
            "n_test": int(metrics.get("n_test", 0))
        }
    except Exception as e:
        logger.warning(f"Error fetching {type_filter} run for {model_name}: {e}")
        return None

# COMMAND ----------

# Main execution
spark.sql("USE CATALOG workspace")
spark.sql("CREATE DATABASE IF NOT EXISTS energy_forecasting")

model_names = ["energy_prophet_24h", "energy_prophet_168h", "energy_lgbm_24h", "energy_lgbm_168h"]
eval_rows = []

for name in model_names:
    horizon = 24 if "24h" in name else 168
    
    # Challenger is the latest run
    challenger = get_run_metrics(name, "challenger")
    # Champion is the best historic run
    champion = get_run_metrics(name, "champion")
    
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
EVAL_SCHEMA = StructType([
    StructField("model_name", StringType(), False),
    StructField("horizon_hours", IntegerType(), False),
    StructField("challenger_run_id", StringType(), False),
    StructField("challenger_mae", DoubleType(), True),
    StructField("challenger_rmse", DoubleType(), True),
    StructField("challenger_mape", DoubleType(), True),
    StructField("champion_run_id", StringType(), True),
    StructField("champion_mae", DoubleType(), True),
    StructField("champion_rmse", DoubleType(), True),
    StructField("champion_mape", DoubleType(), True),
    StructField("challenger_wins", BooleanType(), False),
    StructField("first_run", BooleanType(), False),
    StructField("evaluated_at", TimestampType(), False),
    StructField("promoted", BooleanType(), False)
])

eval_df = spark.createDataFrame(eval_rows, schema=EVAL_SCHEMA)
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
