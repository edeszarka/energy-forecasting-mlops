"""
Configuration management for the energy forecasting system.

This module serves as the single source of truth for all constants, thresholds,
zone codes, file paths, and environment variable names.

Windows Development Note:
Prophet requires pystan, which needs Microsoft C++ Build Tools on Windows.
See: https://facebook.github.io/prophet/docs/installing_in_windows.html
Note that Databricks runs on Linux, so these tools are only needed for local dev.
"""

import os
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Final

# API CONFIGURATION
ENTSO_E_BASE_URL: Final[str] = "https://web-api.transparency.entsoe.eu/api"
ENTSO_E_ZONE: Final[str] = "10YHU-MAVIR----U"
ENTSO_E_DOC_TYPE: Final[str] = "A65"   # Actual Total Load
ENTSO_E_PROCESS_TYPE: Final[str] = "A16"   # Realised
ENTSO_E_MAX_RANGE_DAYS: Final[int] = 7      # Max days per single API request

OPENMETEO_BASE_URL: Final[str] = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_LAT: Final[float] = 47.4979
OPENMETEO_LON: Final[float] = 19.0402
OPENMETEO_TIMEZONE: Final[str] = "Europe/Budapest"

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
    table_silver: str = f"{CATALOG}.{SCHEMA}.silver_load"
    table_gold: str = f"{CATALOG}.{SCHEMA}.gold_forecast"
    
    # Volume paths for files/artifacts
    bronze: str = f"{VOLUME_PATH}/bronze"
    silver: str = f"{VOLUME_PATH}/silver"
    gold: str = f"{VOLUME_PATH}/gold"
    drift: str = f"{VOLUME_PATH}/drift"
    reports: str = f"{VOLUME_PATH}/reports"

PATHS: Final[DataPaths] = DataPaths()

# MLFLOW CONFIGURATION
MLFLOW_EXPERIMENT_NAME: Final[str] = f"/{SCHEMA}/experiments/main"
MLFLOW_MODEL_NAME_PROPHET: Final[str] = "energy_forecast_prophet"
MLFLOW_MODEL_NAME_LGBM: Final[str] = "energy_forecast_lgbm"
MLFLOW_STAGING_ALIAS: Final[str] = "Staging"
MLFLOW_PROD_ALIAS: Final[str] = "Production"

# FEATURE ENGINEERING CONSTANTS
LAG_HOURS: Final[list[int]] = [24, 48, 168]      # t-24h, t-48h, t-168h
ROLLING_WINDOW_DAYS: Final[int] = 7
FORECAST_HORIZON_24H: Final[int] = 24
FORECAST_HORIZON_7D: Final[int] = 168

# DRIFT MONITORING THRESHOLDS
DRIFT_SCORE_THRESHOLD: Final[float] = 0.15   # Jensen-Shannon divergence
DRIFT_CONSECUTIVE_HOURS: Final[int] = 3      # Trigger retraining
DRIFT_REFERENCE_WINDOW_DAYS: Final[int] = 30
RETRAINING_FLAG_TABLE: Final[str] = f"{CATALOG}.{SCHEMA}.retraining_flags"

# HUNGARIAN PUBLIC HOLIDAYS (Fixed dates)
# Moveable feasts (Easter, Pentecost) are handled via 'holidays' package in features.py
FIXED_HOLIDAYS: Final[dict[int, list[int]]] = {
    1: [1],      # New Year
    3: [15],     # Revolution Day 1848
    5: [1],      # Labour Day
    8: [20],     # State Foundation
    10: [23],    # Republic Day / 1956 Revolution
    11: [1],     # All Saints Day
    12: [25, 26] # Christmas
}

# MODEL TRAINING CONFIGURATION
TRAIN_TEST_SPLIT_DAYS: Final[int] = 30
MIN_TRAINING_ROWS: Final[int] = 720
PROPHET_SEASONALITY_MODE: Final[str] = "multiplicative"

LGBM_PARAMS: Final[dict] = {
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "verbose": -1,
}
