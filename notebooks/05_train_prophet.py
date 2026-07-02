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
# # 05_train_prophet
# **Purpose:** Train two Prophet models (24h and 168h horizons) using Hungarian energy consumption data.
# **Inputs:** `workspace.energy_forecasting.silver_features`
# **Outputs:** Registered MLflow models: `energy_prophet_24h`, `energy_prophet_168h`
# **Last Updated:** 2024-05-21
#
# **Required Libraries:** prophet==1.1.5, pandas, numpy, mlflow

# COMMAND ----------

import logging
import os
from datetime import UTC, datetime

import cmdstanpy
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from prophet import Prophet

# Fix for MLflow model registration in Databricks Unity Catalog
os.environ['MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC'] = 'True'

if tuple(int(x) for x in cmdstanpy.__version__.split(".")[:2]) >= (1, 2):
    raise ImportError(
        f"cmdstanpy {cmdstanpy.__version__} is incompatible with prophet 1.1.5. "
        "Pin cmdstanpy>=1.1,<1.2 in requirements.txt."
    )
from pyspark.sql import functions as F

from src.config import CATALOG, PATHS

# COMMAND ----------

# Widgets for configuration
dbutils.widgets.text("test_days", "5")
dbutils.widgets.text("min_train_rows", "200")

CONFIG = {
    "silver_table": PATHS.table_silver,
    "test_days": int(dbutils.widgets.get("test_days")),
    "min_train_rows": int(dbutils.widgets.get("min_train_rows")),
}

# COMMAND ----------

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("05_train_prophet")

# COMMAND ----------

def calculate_mape(actual: pd.Series, predicted: pd.Series) -> float:
    """Computes MAPE guarding against division by zero."""
    mask = actual != 0
    if not mask.any():
        return 0.0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

def train_prophet_model(df: pd.DataFrame, horizon_hours: int, model_name: str):
    """Trains a Prophet model for a specific horizon and logs to MLflow."""
    
    # Split data
    split_date = df['ds'].max() - pd.Timedelta(days=CONFIG["test_days"])
    train_df = df[df['ds'] <= split_date].copy()
    test_df = df[df['ds'] > split_date].copy()
    
    if len(train_df) < CONFIG["min_train_rows"]:
        raise ValueError(f"Insufficient training data for {model_name}. Need {CONFIG['min_train_rows']}, got {len(train_df)}")
    
    with mlflow.start_run(run_name=f"prophet_{horizon_hours}h", nested=True) as run:
        # Define model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_mode='multiplicative'
        )
        model.add_regressor('temperature_c')
        
        # Fit
        logger.info(f"Fitting Prophet model for {horizon_hours}h horizon...")
        model.fit(train_df)
        
        # Evaluate on test set
        forecast = model.predict(test_df[['ds', 'temperature_c']])
        
        # Metrics
        y_true = test_df['y'].values
        y_pred = forecast['yhat'].values
        
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        mape = calculate_mape(pd.Series(y_true), pd.Series(y_pred))
        
        # Log params and metrics
        mlflow.log_params({
            "changepoint_prior_scale": 0.05,
            "seasonality_mode": "multiplicative",
            "horizon_hours": horizon_hours,
            "n_train": len(train_df),
            "n_test": len(test_df)
        })
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "mape": mape})
        
        # Reference Window Metadata
        training_data_end = train_df['ds'].max()
        training_data_start = train_df['ds'].min()
        mlflow.set_tag("model_name", model_name) # Added for discovery
        mlflow.set_tag("training_data_end", training_data_end.isoformat())
        mlflow.set_tag("training_data_start", training_data_start.isoformat())

        # Log model (This works as it logs to the tracking server)
        signature = infer_signature(
            model_input=test_df[['ds', 'temperature_c']],
            model_output=forecast[['yhat']],
        )
        mlflow.prophet.log_model(model, artifact_path="model", signature=signature)
        
        logger.info(f"Model {model_name} trained and logged to run {run.info.run_id}")
        
        return {
            "run_id": run.info.run_id,  # Return run_id so we can find it later
            "model_name": model_name,
            "mae": mae, "rmse": rmse, "mape": mape,
            "n_train": len(train_df), "n_test": len(test_df)
        }

# COMMAND ----------

# Main execution
spark.sql(f"USE CATALOG {CATALOG}")
pdf = spark.read.table(CONFIG["silver_table"]).filter(F.col("value_mwh").isNotNull()).toPandas()
pdf = pdf.rename(columns={"timestamp": "ds", "value_mwh": "y"})
pdf['ds'] = pd.to_datetime(pdf['ds']).dt.tz_localize(None)

results = []
parent_run_name = f"prophet_training_{datetime.now(UTC).strftime('%Y%m%d_%H%M')}"

with mlflow.start_run(run_name=parent_run_name):
    res_24 = train_prophet_model(pdf, 24, "energy_prophet_24h")
    res_168 = train_prophet_model(pdf, 168, "energy_prophet_168h")
    results.extend([res_24, res_168])

print("\nProphet Training Summary:")
print(pd.DataFrame(results).to_string(index=False))
dbutils.notebook.exit("SUCCESS")
