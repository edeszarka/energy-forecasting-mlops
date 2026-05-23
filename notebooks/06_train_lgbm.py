# Databricks notebook source
# %% [markdown]
# # 06_train_lgbm
# **Purpose:** Train two LightGBM models (24h and 168h horizons) using a direct multi-step approach.
# **Inputs:** `workspace.energy_forecasting.silver_features`
# **Outputs:** Registered MLflow models: `energy_lgbm_24h`, `energy_lgbm_168h`
# **Last Updated:** 2024-05-21
#
# **Required Libraries:** lightgbm==4.3.0, shap==0.45.0, pandas, numpy, mlflow, matplotlib

# COMMAND ----------

import logging
import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import mlflow
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

FEATURE_COLS = [
    'temperature_c', 'lag_24h', 'lag_48h', 'lag_168h', 
    'rolling_7d_mean', 'rolling_7d_std', 'hour_of_day', 
    'day_of_week', 'month', 'is_weekend', 'is_holiday'
]

# COMMAND ----------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("06_train_lgbm")

# COMMAND ----------

def calculate_mape(actual: pd.Series, predicted: pd.Series) -> float:
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

def train_lgbm_model(df: pd.DataFrame, horizon_hours: int, model_name: str):
    """Trains a LightGBM model using direct forecasting strategy."""
    
    # Direct forecasting: Shift the target column by -horizon
    # Leakage Prevention: We shift the target, effectively saying "at time T, predict value at T+H"
    df_model = df.copy()
    df_model['target'] = df_model['value_mwh'].shift(-horizon_hours)
    df_model = df_model.dropna(subset=['target'] + FEATURE_COLS)
    
    # Time-based split
    split_date = df_model['timestamp'].max() - pd.Timedelta(days=CONFIG["test_days"])
    train_df = df_model[df_model['timestamp'] <= split_date]
    test_df = df_model[df_model['timestamp'] > split_date]
    
    if len(train_df) < CONFIG["min_train_rows"]:
        raise ValueError(f"Insufficient data for {model_name}. Need {CONFIG['min_train_rows']}, got {len(train_df)}")
    
    X_train, y_train = train_df[FEATURE_COLS], train_df['target']
    X_test, y_test = test_df[FEATURE_COLS], test_df['target']
    
    with mlflow.start_run(run_name=f"lgbm_{horizon_hours}h", nested=True) as run:
        params = {
            "objective": "regression",
            "metric": "mae",
            "num_leaves": 64,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1
        }
        
        model = lgb.LGBMRegressor(**params)
        
        # Fit with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50)]
        )
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = calculate_mape(y_test, y_pred)
        
        # Log params and metrics
        mlflow.log_params(params)
        mlflow.log_param("best_iteration", model.best_iteration_)
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "mape": mape})
        
        if model.best_iteration_ < 50:
            mlflow.set_tag("early_stop", True)

        # Artifacts: Feature Importance
        importance = pd.DataFrame({'feature': FEATURE_COLS, 'importance': model.feature_importances_})
        importance.to_json("importance.json")
        mlflow.log_artifact("importance.json")
        
        # Artifacts: SHAP
        try:
            sample_idx = np.random.choice(X_test.index, min(500, len(X_test)), replace=False)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test.loc[sample_idx])
            
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test.loc[sample_idx], show=False)
            plt.tight_layout()
            plt.savefig("shap_summary.png")
            mlflow.log_artifact("shap_summary.png")
            plt.close()
        except Exception as e:
            logger.warning(f"SHAP failed: {e}")
            mlflow.set_tag("shap_failed", True)

        # Log Model
        mlflow.lightgbm.log_model(model, artifact_path="model")
        
        # Register Model
        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name=model_name,
            tags={"horizon": f"{horizon_hours}h", "model_type": "lgbm"}
        )
        
        return {
            "model_name": model_name,
            "mae": mae, "rmse": rmse, "mape": mape,
            "n_train": len(train_df), "n_test": len(test_df)
        }

# COMMAND ----------

# Main execution
spark.sql("USE CATALOG workspace")

logger.info("Loading silver features...")
pdf = spark.read.table(CONFIG["silver_table"]).toPandas()
pdf['timestamp'] = pd.to_datetime(pdf['timestamp'])

results = []
parent_run_name = f"lgbm_training_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"

with mlflow.start_run(run_name=parent_run_name):
    res_24 = train_lgbm_model(pdf, 24, "energy_lgbm_24h")
    res_168 = train_lgbm_model(pdf, 168, "energy_lgbm_168h")
    results.extend([res_24, res_168])

summary_df = pd.DataFrame(results)
print("\nLightGBM Training Summary:")
print(summary_df.to_string(index=False))

dbutils.notebook.exit("SUCCESS")
