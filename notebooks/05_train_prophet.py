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
os.environ["MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC"] = "True"

if tuple(int(x) for x in cmdstanpy.__version__.split(".")[:2]) >= (1, 2):
    raise ImportError(
        f"cmdstanpy {cmdstanpy.__version__} is incompatible with prophet 1.1.5. "
        "Pin cmdstanpy>=1.1,<1.2 in requirements.txt."
    )
from pyspark.sql import functions as F

from src.baseline import compute_naive_baseline_metrics
from src.config import CATALOG, PATHS
from src.splits import make_prophet_rolling_origins

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


def train_prophet_model(df: pd.DataFrame, horizon_hours: int, model_name: str) -> dict:
    """Trains a Prophet model for a specific horizon and logs to MLflow."""

    # Rolling-origin backtest: K folds, each fit independently and scored at horizon_hours ahead
    n_folds = 5
    folds = make_prophet_rolling_origins(
        df["ds"], horizon_hours, int(CONFIG["test_days"]), n_folds=n_folds
    )

    all_actual = []
    all_pred = []
    all_baseline = []
    fold_cutoffs = []

    for fold_idx, (train_mask, target_mask) in enumerate(folds):
        fold_train = df[train_mask].copy()
        fold_target = df[target_mask].copy()

        if fold_target.empty:
            # Target row didn't land exactly on an hourly timestamp; skip fold
            logger.warning(f"Fold {fold_idx}: target window empty, skipping")
            continue

        if len(fold_train) < CONFIG["min_train_rows"]:
            logger.warning(
                f"Fold {fold_idx}: insufficient training data ({len(fold_train)} < {CONFIG['min_train_rows']}), skipping"
            )
            continue

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_mode="multiplicative",
        )
        model.add_regressor("temperature_c")

        logger.info(
            f"Fitting Prophet model for {horizon_hours}h horizon, fold {fold_idx + 1}/{n_folds}..."
        )
        model.fit(fold_train)

        forecast = model.predict(fold_target[["ds", "temperature_c"]])

        all_actual.extend(fold_target["y"].values)
        all_pred.extend(forecast["yhat"].values)
        baseline_column = f"lag_{horizon_hours}h"
        all_baseline.extend(fold_target[baseline_column].values)

        # Track fold cutoff for MLflow logging
        fold_cutoffs.append(fold_train["ds"].max().isoformat())

    if not all_actual:
        raise ValueError(
            f"No valid folds completed for {model_name} (horizon={horizon_hours}h). "
            f"Check data alignment and tolerance window."
        )

    y_true = np.array(all_actual)
    y_pred = np.array(all_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = calculate_mape(pd.Series(y_true), pd.Series(y_pred))
    naive_metrics = compute_naive_baseline_metrics(pd.Series(all_actual), pd.Series(all_baseline))

    with mlflow.start_run(run_name=f"prophet_{horizon_hours}h", nested=True) as run:
        # Log params and metrics
        mlflow.log_params(
            {
                "changepoint_prior_scale": 0.05,
                "seasonality_mode": "multiplicative",
                "horizon_hours": horizon_hours,
                "n_train": len(y_true),  # Pooled test rows (one per fold)
                "n_folds": n_folds,
                "fold_cutoffs": ",".join(fold_cutoffs),
            }
        )
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "mape": mape})
        mlflow.log_metrics(naive_metrics)

        # Reference Window Metadata - use the full training data range
        training_data_end = df["ds"].max()
        training_data_start = df["ds"].min()
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("training_data_end", training_data_end.isoformat())
        mlflow.set_tag("training_data_start", training_data_start.isoformat())
        mlflow.set_tag("evaluation_method", "rolling_origin")
        mlflow.set_tag("n_folds", str(n_folds))
        mlflow.set_tag("fold_cutoffs", ",".join(fold_cutoffs))

        # Log model - use the last fold's model for registration (or retrain on all data)
        # For now, retrain on all available data up to the last cutoff for the registered model
        last_cutoff = pd.Timestamp(fold_cutoffs[-1])
        full_train_mask = df["ds"] <= last_cutoff
        full_train_df = df[full_train_mask].copy()

        final_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_mode="multiplicative",
        )
        final_model.add_regressor("temperature_c")
        final_model.fit(full_train_df)

        # Log model using the last fold's target for signature
        signature = infer_signature(
            model_input=fold_target[["ds", "temperature_c"]],
            model_output=forecast[["yhat"]],
        )
        mlflow.prophet.log_model(final_model, artifact_path="model", signature=signature)

        logger.info(f"Model {model_name} trained and logged to run {run.info.run_id}")

        return {
            "run_id": run.info.run_id,
            "model_name": model_name,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            **naive_metrics,
            "n_train": len(full_train_df),
            "n_test": len(y_true),
            "n_folds": n_folds,
        }


# COMMAND ----------

# Main execution
spark.sql(f"USE CATALOG {CATALOG}")
pdf = spark.read.table(CONFIG["silver_table"]).filter(F.col("value_mwh").isNotNull()).toPandas()
pdf = pdf.rename(columns={"timestamp": "ds", "value_mwh": "y"})
pdf["ds"] = pd.to_datetime(pdf["ds"]).dt.tz_localize(None)
pdf = pdf.dropna(subset=["temperature_c"])

results = []
parent_run_name = f"prophet_training_{datetime.now(UTC).strftime('%Y%m%d_%H%M')}"

with mlflow.start_run(run_name=parent_run_name):
    res_24 = train_prophet_model(pdf, 24, "energy_prophet_24h")
    res_168 = train_prophet_model(pdf, 168, "energy_prophet_168h")
    results.extend([res_24, res_168])

print("\nProphet Training Summary:")
print(pd.DataFrame(results).to_string(index=False))
dbutils.notebook.exit("SUCCESS")
