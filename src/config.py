"""
Configuration management for the energy forecasting system.

This module serves as the single source of truth for all constants, thresholds,
zone codes, file paths, and environment variable names.

Windows Development Note:
Prophet requires pystan, which needs Microsoft C++ Build Tools on Windows.
See: https://facebook.github.io/prophet/docs/installing_in_windows.html
Note that Databricks runs on Linux, so these tools are only needed for local dev.
"""

from dataclasses import dataclass
from typing import Final

# API CONFIGURATION
ENTSO_E_BASE_URL: Final[str] = "https://web-api.tp.entsoe.eu/api"
ENTSO_E_ZONE: Final[str] = "10YHU-MAVIR----U"
ENTSO_E_DOC_TYPE: Final[str] = "A65"  # Actual Total Load
ENTSO_E_PROCESS_TYPE: Final[str] = "A16"  # Realised
ENTSO_E_MAX_RANGE_DAYS: Final[int] = 7  # Max days per single API request

OPENMETEO_BASE_URL: Final[str] = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_LAT: Final[float] = 47.4979
OPENMETEO_LON: Final[float] = 19.0402
# Request UTC: local Budapest times repeat/skip an hour at DST changes, which
# tz_localize cannot map back to UTC (spec 012). start_date/end_date are also
# interpreted in this timezone, so UTC keeps them aligned with UTC fetch windows.
OPENMETEO_TIMEZONE: Final[str] = "UTC"
# Same models and parameters as /v1/forecast, first hours of each run stitched;
# used only by the backfill path (spec 014 T3, ADR-0005).
OPENMETEO_HISTORICAL_FORECAST_URL: Final[str] = (
    "https://historical-forecast-api.open-meteo.com/v1/forecast"
)

HTTP_TIMEOUT_SECONDS: Final[int] = 30
HTTP_MAX_RETRIES: Final[int] = 3
HTTP_BACKOFF_FACTOR: Final[int] = 2

# ENVIRONMENT VARIABLE NAMES
ENV_ENTSO_E_API_KEY: Final[str] = "ENTSO_E_API_KEY"
ENV_DATABRICKS_HOST: Final[str] = "DATABRICKS_HOST"
ENV_DATABRICKS_TOKEN: Final[str] = "DATABRICKS_TOKEN"

# UNITY CATALOG & PATH CONFIGURATION (Databricks Free Edition)
CATALOG: Final[str] = "workspace"
SCHEMA: Final[str] = "energy_forecasting"
VOLUME_PATH: Final[str] = f"/Volumes/{CATALOG}/{SCHEMA}/data"


@dataclass(frozen=True)
class DataPaths:
    """Grouped paths for Delta tables and Volumes."""

    # Table names (3-level)
    table_bronze: str = f"{CATALOG}.{SCHEMA}.bronze_load"
    table_bronze_temp: str = f"{CATALOG}.{SCHEMA}.bronze_temperature"
    table_silver: str = f"{CATALOG}.{SCHEMA}.silver_features"
    table_drift: str = f"{CATALOG}.{SCHEMA}.drift_control"
    table_gold: str = f"{CATALOG}.{SCHEMA}.gold_forecasts"
    table_eval: str = f"{CATALOG}.{SCHEMA}.model_evaluation"
    table_promotion: str = f"{CATALOG}.{SCHEMA}.promotion_log"
    table_ingestion_log: str = f"{CATALOG}.{SCHEMA}.ingestion_log"

    # Volume paths for files/artifacts
    volume_raw_load: str = f"{VOLUME_PATH}/raw_ingestion/load"
    volume_raw_temp: str = f"{VOLUME_PATH}/raw_ingestion/temperature"
    volume_archive_load: str = f"{VOLUME_PATH}/raw_ingestion/archive/load"
    volume_archive_temp: str = f"{VOLUME_PATH}/raw_ingestion/archive/temperature"
    # Backfill path (spec 014 T3): one file per chunk, archived per run
    volume_backfill_load: str = f"{VOLUME_PATH}/raw_backfill/load"
    volume_backfill_temp: str = f"{VOLUME_PATH}/raw_backfill/temperature"
    volume_backfill_archive: str = f"{VOLUME_PATH}/raw_backfill/archive"
    volume_reports: str = f"{VOLUME_PATH}/reports"
    volume_flags: str = f"{VOLUME_PATH}/flags"


PATHS: Final[DataPaths] = DataPaths()

# RETRAINING FLAG PATH
RETRAIN_FLAG_PATH: Final[str] = f"{PATHS.volume_flags}/retrain_requested.flag"

# MLFLOW CONFIGURATION
MLFLOW_EXPERIMENT_NAME: Final[str] = f"/{SCHEMA}/experiments/main"
MLFLOW_MODEL_NAME_PROPHET: Final[str] = "energy_forecast_prophet"
MLFLOW_MODEL_NAME_LGBM: Final[str] = "energy_forecast_lgbm"
MLFLOW_STAGING_ALIAS: Final[str] = "Staging"
MLFLOW_PROD_ALIAS: Final[str] = "Production"

# MODEL ROLES (ADR-0002): only PROMOTABLE_MODELS can be production-tagged and
# served; BENCHMARK_MODELS are trained and evaluated for reporting only.
PROMOTABLE_MODELS: Final[tuple[str, ...]] = ("energy_lgbm_24h",)
BENCHMARK_MODELS: Final[tuple[str, ...]] = ("energy_prophet_24h",)
EVALUATED_MODELS: Final[tuple[str, ...]] = PROMOTABLE_MODELS + BENCHMARK_MODELS

# DATA VERSIONING (ADR-0003): runs tag the silver `pipeline.data_version` they
# read; serving, evaluation, promotion and the drift reference only consider
# runs tagged with CURRENT_DATA_VERSION.
CURRENT_DATA_VERSION: Final[str] = "2"

# FEATURE ENGINEERING CONSTANTS
LAG_HOURS: Final[list[int]] = [24, 48, 168]  # t-24h, t-48h, t-168h
ROLLING_WINDOW_DAYS: Final[int] = 7

# Single source of truth for model input columns
# Used by: 06_train_lgbm.py, 04_predict.py, 03_drift_check.py
# If you add a feature here, update src/features.py:get_feature_columns() too
MODEL_INPUT_FEATURES: Final[list[str]] = [
    "temperature_c",
    "lag_24h",
    "lag_48h",
    "lag_168h",
    "rolling_7d_mean",
    "rolling_7d_std",
    "rolling_24h_mean",
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "humidity_pct",
    "cloud_cover_pct",
]

# DRIFT MONITORING THRESHOLDS
DRIFT_SCORE_THRESHOLD: Final[float] = 0.15  # Jensen-Shannon divergence
DRIFT_CONSECUTIVE_HOURS: Final[int] = 3  # Trigger retraining
DRIFT_REFERENCE_WINDOW_DAYS: Final[int] = 30
RETRAINING_FLAG_TABLE: Final[str] = f"{CATALOG}.{SCHEMA}.retraining_flags"

# DATA-QUALITY THRESHOLDS (ADR-0004, spec 014 T5)
# PROVISIONAL: set from the T0 raw ENTSO-E responses (reports/partB1_spec014_2026-10-24.md),
# to be calibrated on the rebuilt 2024-2025 silver (runbook step 6) in the
# follow-up PR that switches dq_mode to enforce.
# Observed hourly means 3,488-7,073 MW (Jan 2024 weekday peak to Oct 2026 night).
# Backfill dry run: lowest values 2211-2487 MW at midday on Easter 2025 and a
# Sunday in May 2026 (holiday load minus rooftop solar), so 2500 was too tight.
LOAD_MIN_MW: Final[float] = 2000.0  # calibrated on the 2024-01 → 2026-10 backfill (min 2211 MW)
LOAD_MAX_MW: Final[float] = 9000.0  # PROVISIONAL
# Observed lag-24h correlation 0.70-0.89 on healthy samples; the same samples
# stretched 4x (pre-013 parser) give -0.36 to -0.14, the owner's corrupted
# months 0.02-0.22.
DQ_MIN_CORR_24H: Final[float] = 0.5  # PROVISIONAL

# HUNGARIAN PUBLIC HOLIDAYS (Fixed dates)
FIXED_HOLIDAYS: Final[dict[int, list[int]]] = {
    1: [1],  # New Year
    3: [15],  # Revolution Day 1848
    5: [1],  # Labour Day
    8: [20],  # State Foundation
    10: [23],  # Republic Day / 1956 Revolution
    11: [1],  # All Saints Day
    12: [25, 26],  # Christmas
}

# MODEL TRAINING CONFIGURATION
TRAIN_TEST_SPLIT_DAYS: Final[int] = 30
MIN_TRAINING_ROWS: Final[int] = 720
PROPHET_SEASONALITY_MODE: Final[str] = "multiplicative"

# Default LightGBM hyperparameters (sklearn API names for LGBMRegressor)
# Used by 06_train_lgbm.py when Optuna tuning is disabled (n_trials=0)
# Tuned params override individual keys when tuning is active
LGBM_PARAMS: Final[dict] = {
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 64,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "verbose": -1,
}

# OPTUNA HYPERPARAMETER OPTIMIZATION
OPTUNA_N_TRIALS: Final[int] = 25
OPTUNA_TIMEOUT_SECONDS: Final[int] = 600
OPTUNA_N_SPLITS: Final[int] = 3  # TimeSeriesSplit folds
OPTUNA_GAP_HOURS: Final[int] = 168  # 1-week gap between train/test splits

# FIX APPLIED: Corrected table_silver name to silver_features and added missing table constants (drift, gold, eval, promotion, ingestion_log).
