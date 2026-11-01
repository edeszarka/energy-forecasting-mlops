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

# MAGIC %md
# MAGIC # 06_train_lgbm
# MAGIC
# MAGIC Trains the 24h-ahead LightGBM on point-in-time features (spec 019, ADR-0008): a forecast
# MAGIC issued at hour τ for T = τ + 24h sees load up to τ − 24h, the as-issued weather forecast for
# MAGIC hours ≤ T, and T's calendar (`src/pit_features.py`, shared with serving).
# MAGIC
# MAGIC 1. **Rolling-origin backtest** over the last `backtest_months` months: each month is predicted
# MAGIC    by a model trained only on issuances whose target ends before the month starts. The weekly
# MAGIC    naive (T − 168h, the served fallback) and the daily naive (T − 48h) are scored on the same
# MAGIC    rows. The logged `mape` and `naive_mape` come from this backtest.
# MAGIC 2. **Final model** trained on all rows (early stopping on the last 28 days) and logged.
# MAGIC
# MAGIC Optuna tuning is off by default (`n_trials = 0`): the default parameters beat the weekly
# MAGIC naive by 55% in the spec 019 backtest. With `n_trials > 0` it tunes only on rows before the
# MAGIC backtest window.
# MAGIC
# MAGIC **Inputs:** `silver_features` (actual load, DQ-checked), `bronze_weather_forecast`
# MAGIC
# MAGIC **Outputs:** MLflow run tagged `model_name=energy_lgbm_24h`, `feature_set`, `data_version`

# COMMAND ----------

import logging
from datetime import UTC, datetime

import mlflow
import pandas as pd

from src.backtest import (
    PURGE,
    fit_lgbm,
    monthly_folds,
    monthly_table,
    rolling_origin_backtest,
    summarize,
)
from src.config import CATALOG, FEATURE_SET, LGBM_PARAMS, PATHS
from src.data_version import read_data_version, table_properties
from src.pit_features import FEATURES, FORECAST_HORIZON_HOURS, build_features
from src.promotion import BACKTEST_ARTIFACT
from src.tuning import run_lgbm_tuning

dbutils.widgets.text("backtest_months", "12")
dbutils.widgets.text("n_trials", "0")
CONFIG = {
    "backtest_months": int(dbutils.widgets.get("backtest_months")),
    "n_trials": int(dbutils.widgets.get("n_trials")),
}
MODEL_NAME = "energy_lgbm_24h"
BASE_PARAMS = {k: v for k, v in LGBM_PARAMS.items() if k not in ("n_estimators", "verbose")}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("06_train_lgbm")

# COMMAND ----------


def read_inputs(spark) -> tuple[pd.DataFrame, pd.DataFrame]:
    load = spark.read.table(PATHS.table_silver).select("timestamp", "value_mwh").toPandas()
    weather = spark.read.table(PATHS.table_weather_forecast).toPandas()
    for frame in (load, weather):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return load, weather


def issuance_hours(load: pd.DataFrame, weather: pd.DataFrame) -> pd.DatetimeIndex:
    """Every hour with a possible actual: from when both sources exist to the last target."""
    first = max(load["timestamp"].min(), weather["timestamp"].min()).tz_localize(None)
    last = load["timestamp"].max().tz_localize(None) - pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
    return pd.date_range(first.ceil("h"), last, freq="h")


def tune(data: pd.DataFrame, n_trials: int) -> dict:
    """Optuna on rows whose targets end before the backtest window (no peeking)."""
    folds = monthly_folds(data.dropna(subset=["target_mwh"]).index, CONFIG["backtest_months"])
    train = data.dropna(subset=["target_mwh"])
    train = train[train.index < folds[0][0] - PURGE]
    return run_lgbm_tuning(
        train[FEATURES], train["target_mwh"], FORECAST_HORIZON_HOURS, n_trials=n_trials
    )


# COMMAND ----------

spark.sql(f"USE CATALOG {CATALOG}")
# Read the version before the data: a rebuild finishing in between leaves the
# run tagged with the older version, which is never served (ADR-0003).
data_version = read_data_version(
    table_properties(spark.sql(f"SHOW TBLPROPERTIES {PATHS.table_silver}").collect())
)
load_pd, weather_pd = read_inputs(spark)
data = build_features(load_pd, weather_pd, issuance_hours(load_pd, weather_pd))
logger.info(f"{len(data)} issuance rows, {data['target_mwh'].notna().sum()} with an actual")

with (
    mlflow.start_run(run_name=f"lgbm_training_{datetime.now(UTC):%Y%m%d_%H%M}"),
    mlflow.start_run(run_name="lgbm_24h", nested=True) as run,
):
    params = dict(BASE_PARAMS)
    if CONFIG["n_trials"] > 0:
        params.update(tune(data, CONFIG["n_trials"]))
        mlflow.set_tag("optuna_n_trials", str(CONFIG["n_trials"]))
    mlflow.set_tag("optuna_tuned", str(CONFIG["n_trials"] > 0).lower())

    rows = rolling_origin_backtest(data, params, CONFIG["backtest_months"])
    summary = summarize(rows)
    table = monthly_table(rows)
    logger.info(f"Backtest: {summary}")
    print(table.round(2).to_string(index=False))

    train = data.dropna(subset=["target_mwh"])
    model = fit_lgbm(train, params)

    mlflow.log_params({**params, "best_iteration": model.best_iteration_})
    mlflow.log_param("backtest_months", CONFIG["backtest_months"])
    mlflow.log_metrics(
        {
            # `mape` and `naive_mape` are what 07/08 compare (first run: model vs naive).
            "mape": summary["lgbm_mape"],
            "mae": summary["lgbm_mae"],
            "rmse": summary["lgbm_rmse"],
            "naive_mape": summary["weekly_naive_mape"],
            "naive_mae": summary["weekly_naive_mae"],
            "naive_rmse": summary["weekly_naive_rmse"],
            "daily_naive_mape": summary["daily_naive_mape"],
            "skill_vs_weekly_naive": summary["skill_vs_weekly_naive"],
            "backtest_rows": summary["n_rows"],
        }
    )
    mlflow.log_dict(table.to_dict(orient="list"), "backtest_monthly.json")
    mlflow.log_dict(
        dict(
            zip(
                FEATURES,
                model.booster_.feature_importance(importance_type="gain").tolist(),
                strict=True,
            )
        ),
        "feature_importance_gain.json",
    )
    # Per-row backtest: 07_evaluate scores the champion on the same rows
    # (spec 011 §1.3). `timestamp` is the issuance hour.
    mlflow.log_dict(
        {
            "timestamp": [pd.Timestamp(t).isoformat() for t in rows["issued_at"]],
            "target_timestamp": [pd.Timestamp(t).isoformat() for t in rows["target_timestamp"]],
            "actual": rows["actual"].astype(float).tolist(),
            "predicted": rows["predicted"].astype(float).tolist(),
            "naive": rows["weekly_naive"].astype(float).tolist(),
        },
        BACKTEST_ARTIFACT,
    )

    mlflow.set_tags(
        {
            "model_name": MODEL_NAME,
            "leak_affected": "false",
            "data_version": data_version,
            "feature_set": FEATURE_SET,
            "training_data_start": train.index.min().isoformat(),
            "training_data_end": train.index.max().isoformat(),
            # The final model's early stopping saw the last targets (spec 011 §1.6).
            "served_data_end": train["target_timestamp"].max().isoformat(),
        }
    )
    mlflow.lightgbm.log_model(model, artifact_path="model")
    logger.info(f"{MODEL_NAME} logged to run {run.info.run_id}")

dbutils.notebook.exit(
    "SUCCESS: "
    f"MAPE {summary['lgbm_mape']:.3f}% vs weekly naive {summary['weekly_naive_mape']:.3f}% "
    f"(skill {summary['skill_vs_weekly_naive']:.3f}, {int(summary['n_rows'])} rows)"
)
