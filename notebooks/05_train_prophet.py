# Databricks notebook source
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
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import mlflow
from prophet import Prophet
from pyspark.sql import functions as F

# COMMAND ----------

# Widgets for configuration
dbutils.widgets.text("test_days", "30")
dbutils.widgets.text("min_train_rows", "2000")

CONFIG = {
    "silver_table": "workspace.energy_forecasting.silver_features",
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
        # For evaluation, we use the actual temperature in the test set
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
        
        mlflow.log_metrics({
            "mae": mae,
            "rmse": rmse,
            "mape": mape
        })
        
        # Log model
        mlflow.prophet.log_model(model, artifact_path="model")
        
        # Register model
        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name=model_name,
            tags={"horizon": f"{horizon_hours}h", "model_type": "prophet"}
        )
        
        return {
            "model_name": model_name,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "n_train": len(train_df),
            "n_test": len(test_df)
        }

# COMMAND ----------

# Main execution
spark.sql("USE CATALOG workspace")

# Load data
logger.info("Loading silver features...")
pdf = spark.read.table(CONFIG["silver_table"]).filter(F.col("value_mwh").isNotNull()).toPandas()

# Prep for Prophet
# Rename columns: timestamp -> ds, value_mwh -> y
pdf = pdf.rename(columns={"timestamp": "ds", "value_mwh": "y"})
pdf['ds'] = pd.to_datetime(pdf['ds']).dt.tz_localize(None) # Prophet prefers naive local or UTC

# Check convergence/future temperature notes
# Naive forecast for future temperature: in a real production 04_predict notebook, 
# we would join future weather. Here we evaluate on test sets with known weather.

results = []
parent_run_name = f"prophet_training_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"

with mlflow.start_run(run_name=parent_run_name):
    res_24 = train_prophet_model(pdf, 24, "energy_prophet_24h")
    res_168 = train_prophet_model(pdf, 168, "energy_prophet_168h")
    results.extend([res_24, res_168])

# Print summary table
summary_df = pd.DataFrame(results)
print("\nProphet Training Summary:")
print(summary_df.to_string(index=False))

dbutils.notebook.exit("SUCCESS")
