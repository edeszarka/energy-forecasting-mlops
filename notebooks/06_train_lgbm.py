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
# # 06_train_lgbm
# **Purpose:** Train two LightGBM models (24h and 168h horizons) using a direct multi-step approach.
# **Inputs:** `workspace.energy_forecasting.silver_features`
# **Outputs:** Registered MLflow models: `energy_lgbm_24h`, `energy_lgbm_168h`
# **Last Updated:** 2024-05-21
#
# **Required Libraries:** lightgbm==4.3.0, shap==0.45.0, pandas, numpy, mlflow, matplotlib

# COMMAND ----------

import logging
import os
from datetime import UTC, datetime

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.baseline import compute_naive_baseline_metrics
from src.config import (
    CATALOG,
    LGBM_PARAMS,
    MIN_TRAINING_ROWS,
    MODEL_INPUT_FEATURES,
    OPTUNA_N_TRIALS,
    PATHS,
)
from src.splits import make_holdout_splits
from src.tuning import run_lgbm_tuning

# Fix for MLflow model registration in Databricks Unity Catalog
os.environ["MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC"] = "True"

# COMMAND ----------

dbutils.widgets.text("val_days", "5")
dbutils.widgets.text("test_days", "5")
dbutils.widgets.text("min_train_rows", "200")
dbutils.widgets.text("n_trials", str(OPTUNA_N_TRIALS))

CONFIG = {
    "silver_table": PATHS.table_silver,
    "val_days": int(dbutils.widgets.get("val_days")),
    "test_days": int(dbutils.widgets.get("test_days")),
    "min_train_rows": int(dbutils.widgets.get("min_train_rows")),
    "n_trials": int(dbutils.widgets.get("n_trials")),
}

FEATURE_COLS = MODEL_INPUT_FEATURES  # sourced from src/config.py

# COMMAND ----------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("06_train_lgbm")

# COMMAND ----------


def calculate_mape(actual: pd.Series, predicted: pd.Series) -> float:
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def train_lgbm_model(df: pd.DataFrame, horizon_hours: int, model_name: str):
    """Trains a LightGBM model using direct forecasting strategy."""
    df_model = df.copy()
    df_model["target"] = df_model["value_mwh"].shift(-horizon_hours)
    df_model = df_model.dropna(subset=["target"] + FEATURE_COLS)

    train_mask, val_mask, test_mask = make_holdout_splits(
        df_model["timestamp"], CONFIG["val_days"], CONFIG["test_days"]
    )
    train_df = df_model[train_mask]
    val_df = df_model[val_mask]
    test_df = df_model[test_mask]

    if len(train_df) < max(MIN_TRAINING_ROWS, CONFIG["min_train_rows"]):
        raise ValueError(
            f"Insufficient data for {model_name}. Need {max(MIN_TRAINING_ROWS, CONFIG['min_train_rows'])}, got {len(train_df)}"
        )

    X_train, y_train = train_df[FEATURE_COLS], train_df["target"]
    X_val, y_val = val_df[FEATURE_COLS], val_df["target"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["target"]

    with mlflow.start_run(run_name=f"lgbm_{horizon_hours}h", nested=True) as run:
        # Optuna tuning (skip if n_trials == 0)
        tuned_params = {}
        if CONFIG["n_trials"] > 0:
            logger.info(f"Tuning {model_name} with {CONFIG['n_trials']} Optuna trials...")
            tuned_params = run_lgbm_tuning(
                X_train, y_train, horizon_hours, n_trials=CONFIG["n_trials"]
            )
            mlflow.log_params({f"tuned_{k}": v for k, v in tuned_params.items()})
            mlflow.set_tag("optuna_tuned", "true")
            mlflow.set_tag("optuna_n_trials", str(CONFIG["n_trials"]))

        # Merge defaults with tuned overrides
        params = {**LGBM_PARAMS, **tuned_params}
        params["n_estimators"] = 500  # fixed cap, early stopping handles actual count

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)])

        y_pred = model.predict(X_test)
        mae, rmse, mape = (
            mean_absolute_error(y_test, y_pred),
            np.sqrt(mean_squared_error(y_test, y_pred)),
            calculate_mape(y_test, y_pred),
        )
        naive_metrics = compute_naive_baseline_metrics(test_df["target"], test_df["value_mwh"])

        mlflow.log_params(params)
        mlflow.log_param("best_iteration", model.best_iteration_)
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "mape": mape})
        mlflow.log_metrics(naive_metrics)

        # Reference Window Metadata
        training_data_end = train_df["timestamp"].max()
        training_data_start = train_df["timestamp"].min()
        mlflow.set_tag("model_name", model_name)  # Added for discovery
        mlflow.set_tag("training_data_end", training_data_end.isoformat())
        mlflow.set_tag("training_data_start", training_data_start.isoformat())

        # Artifacts
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
        # Log model
        mlflow.lightgbm.log_model(model, artifact_path="model")

        logger.info(f"Model {model_name} trained and logged to run {run.info.run_id}")

        return {
            "run_id": run.info.run_id,
            "model_name": model_name,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            **naive_metrics,
            "n_train": len(train_df),
            "n_test": len(test_df),
        }


# COMMAND ----------

# Main execution
spark.sql(f"USE CATALOG {CATALOG}")
pdf = spark.read.table(CONFIG["silver_table"]).toPandas()
pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])

results = []
parent_run_name = f"lgbm_training_{datetime.now(UTC).strftime('%Y%m%d_%H%M')}"

with mlflow.start_run(run_name=parent_run_name):
    res_24 = train_lgbm_model(pdf, 24, "energy_lgbm_24h")
    res_168 = train_lgbm_model(pdf, 168, "energy_lgbm_168h")
    results.extend([res_24, res_168])

print("\nLightGBM Training Summary:")
print(pd.DataFrame(results).to_string(index=False))
dbutils.notebook.exit("SUCCESS")
