"""
Feature engineering logic for the energy forecasting system.

This module provides pure Python + pandas functions to transform raw electricity
load and weather data into a feature matrix ready for model training or inference.

Training-Serving Skew Mitigation:
By centralizing all feature logic here and reusing this module in both the
daily inference pipeline (02_transform.py) and the training notebooks (05, 06),
we ensure that the exact same mathematical operations are applied to data
regardless of the environment.

Features defined:
- Calendar: hour_of_day, day_of_week, month, quarter, is_weekend, is_holiday, is_holiday_eve.
- Trend: days_since_epoch.
- Lags: lag_24h, lag_48h, lag_168h.
- Rolling: 7d mean/std, 24h mean.
- Weather: temperature_c, temperature_lag_24h.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Final

import holidays
import numpy as np
import pandas as pd

from src.config import FIXED_HOLIDAYS, LAG_HOURS, MIN_TRAINING_ROWS, ROLLING_WINDOW_DAYS

logger = logging.getLogger(__name__)

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds calendar-based features derived from the timestamp.
    
    Args:
        df: Input DataFrame with 'timestamp' column (tz-aware UTC).
        
    Returns:
        DataFrame with new calendar columns.
    
    Raises:
        ValueError: If timestamp is timezone-naive.
    """
    if df.empty:
        cols = [
            "hour_of_day", "day_of_week", "month", "quarter", 
            "is_weekend", "is_holiday", "is_holiday_eve", "days_since_epoch"
        ]
        return df.assign(**{col: [] for col in cols})

    if df["timestamp"].dt.tz is None:
        raise ValueError("The 'timestamp' column must be timezone-aware (UTC).")

    df = df.copy()
    
    # Convert to local Budapest time for accurate hour/holiday features
    local_ts = df["timestamp"].dt.tz_convert("Europe/Budapest")
    
    df["hour_of_day"] = local_ts.dt.hour
    df["day_of_week"] = local_ts.dt.dayofweek
    df["month"] = local_ts.dt.month
    df["quarter"] = local_ts.dt.quarter
    df["is_weekend"] = df["day_of_week"] >= 5
    
    # Hungarian Holidays (Moveable)
    min_year = local_ts.dt.year.min()
    max_year = local_ts.dt.year.max()
    hu_holidays = holidays.Hungary(years=range(min_year, max_year + 1))
    
    # Fixed Holidays from config
    def check_is_holiday(row: pd.Timestamp) -> bool:
        # Check moveable
        if row.date() in hu_holidays:
            return True
        # Check fixed
        fixed = FIXED_HOLIDAYS.get(row.month, [])
        return row.day in fixed

    df["is_holiday"] = local_ts.apply(check_is_holiday).astype(bool)
    
    # Holiday Eve (Day before holiday or weekend)
    # We use shift(-1) on the sorted local timestamps to see if tomorrow is a holiday
    df_sorted = df.sort_values("timestamp")
    tomorrow_is_holiday = df_sorted["is_holiday"].shift(-1).fillna(False)
    tomorrow_is_weekend = (df_sorted["day_of_week"].shift(-1) >= 5).fillna(False)
    df["is_holiday_eve"] = tomorrow_is_holiday | tomorrow_is_weekend
    
    # Trend feature: Days since a fixed baseline
    baseline = pd.Timestamp("2015-01-01", tz="UTC")
    df["days_since_epoch"] = (df["timestamp"] - baseline).dt.days
    
    return df

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds historical lag features for the target variable 'value_mwh'.
    
    Args:
        df: Input DataFrame sorted by timestamp.
        
    Returns:
        DataFrame with lag columns and gap flags.
    """
    if df.empty:
        cols = [f"lag_{h}h" for h in LAG_HOURS] + ["has_lag_gap"]
        return df.assign(**{col: [] for col in cols})

    df = df.sort_values("timestamp").copy()
    
    # Ensure complete hourly index to prevent shift() misalignment
    full_range = pd.date_range(
        start=df["timestamp"].min(),
        end=df["timestamp"].max(),
        freq="H",
        tz="UTC"
    )
    
    # Identify original timestamps to drop reindex-only rows later
    original_timestamps = set(df["timestamp"])
    
    df = df.set_index("timestamp").reindex(full_range)
    
    for lag_h in LAG_HOURS:
        df[f"lag_{lag_h}h"] = df["value_mwh"].shift(lag_h)
        
    lag_cols = [f"lag_{h}h" for h in LAG_HOURS]
    df["has_lag_gap"] = df[lag_cols].isna().any(axis=1)
    
    # Restore original rows only
    df = df.reset_index().rename(columns={"index": "timestamp"})
    df = df[df["timestamp"].isin(original_timestamps)]
    
    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rolling window statistics for the target variable.
    
    Args:
        df: Input DataFrame sorted by timestamp.
        
    Returns:
        DataFrame with rolling columns.
    """
    if df.empty:
        cols = ["rolling_7d_mean", "rolling_7d_std", "rolling_24h_mean"]
        return df.assign(**{col: [] for col in cols})

    df = df.sort_values("timestamp").copy()
    
    # 7-day (168h) rolling statistics
    df["rolling_7d_mean"] = df["value_mwh"].rolling(window=168, min_periods=72).mean()
    df["rolling_7d_std"] = df["value_mwh"].rolling(window=168, min_periods=72).std()
    
    # 24-hour rolling mean
    df["rolling_24h_mean"] = df["value_mwh"].rolling(window=24, min_periods=12).mean()
    
    return df

def _fill_missing_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Private helper to impute missing weather data.
    """
    df = df.copy()
    
    # Track which rows were missing before filling
    df["temp_missing"] = df["temperature_c"].isna()
    
    # Step 1: Forward-fill short gaps (up to 3h)
    df["temperature_c"] = df["temperature_c"].ffill(limit=3)
    
    # Step 2: Backward-fill short gaps (up to 3h)
    df["temperature_c"] = df["temperature_c"].bfill(limit=3)
    
    # Step 3: 72h rolling mean for remaining NaN
    df["temperature_c"] = df["temperature_c"].fillna(
        df["temperature_c"].rolling(window=72, min_periods=1, center=True).mean()
    )
    
    # Update is_temp_imputed flag
    df["is_temp_imputed"] = df["is_temp_imputed"] | (df["temp_missing"] & df["temperature_c"].notna())
    
    return df

def add_temperature_features(
    df: pd.DataFrame,
    temp_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges and processes temperature features.
    """
    if temp_df.empty:
        logger.warning("Temperature DataFrame is empty. Features will be null.")
        df["temperature_c"] = np.nan
        df["temperature_lag_24h"] = np.nan
        df["is_temp_imputed"] = False
        df["temp_missing"] = True
        return df

    # Merge temperature data
    df = pd.merge(
        df, 
        temp_df[["timestamp", "temperature_c", "is_temp_imputed"]], 
        on="timestamp", 
        how="left"
    )
    
    # Impute missing temperature values
    df = _fill_missing_temperature(df)
    
    # Add temperature lag
    df = df.sort_values("timestamp")
    df["temperature_lag_24h"] = df["temperature_c"].shift(24)
    
    return df

def build_feature_matrix(
    load_df: pd.DataFrame,
    temp_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Orchestrates the creation of the full feature matrix.
    
    Args:
        load_df: Raw load data.
        temp_df: Raw temperature data.
        
    Returns:
        Complete feature matrix.
        
    Raises:
        ValueError: If load_df has insufficient data.
    """
    if len(load_df) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"Insufficient data for feature engineering. "
            f"Required: {MIN_TRAINING_ROWS}, Actual: {len(load_df)}"
        )
    
    # Validate required columns
    required_cols = {"timestamp", "value_mwh"}
    if not required_cols.issubset(load_df.columns):
        raise ValueError(f"load_df missing required columns: {required_cols - set(load_df.columns)}")

    # Process features in order
    df = add_calendar_features(load_df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_temperature_features(df, temp_df)
    
    # Final cleanup
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["feature_built_at"] = pd.Timestamp.now(tz="UTC")
    
    return df

def get_feature_columns() -> list[str]:
    """
    Returns the canonical list of feature columns used as model input.
    """
    return [
        "hour_of_day", "day_of_week", "month", "quarter",
        "is_weekend", "is_holiday", "is_holiday_eve", "days_since_epoch",
        "lag_24h", "lag_48h", "lag_168h",
        "rolling_7d_mean", "rolling_7d_std", "rolling_24h_mean",
        "temperature_c", "temperature_lag_24h",
    ]
