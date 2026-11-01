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

import holidays
import numpy as np
import pandas as pd

from src.config import FIXED_HOLIDAYS, LAG_HOURS, MIN_TRAINING_ROWS

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
            "hour_of_day",
            "day_of_week",
            "month",
            "quarter",
            "is_weekend",
            "is_holiday",
            "is_holiday_eve",
            "days_since_epoch",
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
    hu_holidays = holidays.country_holidays("HU", years=range(min_year, max_year + 1))

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

    # Fix FutureWarning: set option to opt-in to future behavior
    pd.set_option("future.no_silent_downcasting", True)

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
        start=df["timestamp"].min(), end=df["timestamp"].max(), freq="h", tz="UTC"
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

    # Windows are computed on the full hourly grid (spec 014 T4): over rows, a
    # gap would pull hours older than the window into it. On the grid a window
    # of N rows is exactly N hours, and min_periods counts hours with a value.
    values = df.set_index("timestamp")["value_mwh"]
    grid = values.reindex(pd.date_range(values.index.min(), values.index.max(), freq="h"))

    # 7-day (168h) rolling statistics
    rolling_7d = grid.rolling(window=168, min_periods=72)
    # 24-hour rolling mean
    rolling_24h = grid.rolling(window=24, min_periods=12)

    stats = pd.DataFrame(
        {
            "rolling_7d_mean": rolling_7d.mean(),
            "rolling_7d_std": rolling_7d.std(),
            "rolling_24h_mean": rolling_24h.mean(),
        }
    ).reindex(df["timestamp"])
    for col in stats.columns:
        df[col] = stats[col].to_numpy()

    return df


def _fill_missing_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Private helper to impute missing weather data for all columns.
    """
    df = df.copy()

    weather_cols = ["temperature_c", "humidity_pct", "cloud_cover_pct"]

    # Track which rows were missing before filling
    missing_before = df[weather_cols].isna()
    df["temp_missing"] = df["temperature_c"].isna()

    for col in weather_cols:
        if col not in df.columns:
            continue
        # Only earlier hours fill a gap (spec 019): forward-fill up to 3h, then
        # the trailing 72h mean. No backward fill, no centred window.
        df[col] = df[col].ffill(limit=3)
        df[col] = df[col].fillna(df[col].rolling(window=72, min_periods=1).mean())

    # Update is_weather_imputed flag
    if "is_weather_imputed" not in df.columns:
        df["is_weather_imputed"] = False

    # Mark as imputed if any weather column was missing but is now filled
    was_filled = (missing_before & df[weather_cols].notna()).any(axis=1)
    df["is_weather_imputed"] = df["is_weather_imputed"] | was_filled

    return df


def add_weather_features(
    df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges and processes weather features (temperature, humidity, cloud cover).
    """
    if weather_df.empty:
        logger.warning("Weather DataFrame is empty. Features will be null.")
        df["temperature_c"] = np.nan
        df["humidity_pct"] = np.nan
        df["cloud_cover_pct"] = np.nan
        df["temperature_lag_24h"] = np.nan
        df["is_weather_imputed"] = False
        df["temp_missing"] = True
        return df

    # Select only the columns that exist in weather_df
    weather_cols = [
        "timestamp",
        "temperature_c",
        "humidity_pct",
        "cloud_cover_pct",
        "is_weather_imputed",
    ]
    available = [c for c in weather_cols if c in weather_df.columns]

    # Weather is filled and lagged on the full hourly grid (spec 014 T4), so a
    # missing load hour can't shift temperature_lag_24h or the fill windows.
    grid = pd.DataFrame(
        {"timestamp": pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")}
    )
    weather = pd.merge(grid, weather_df[available], on="timestamp", how="left")

    # Create any missing weather columns as NaN
    for col in weather_cols:
        if col not in weather.columns:
            weather[col] = np.nan if col != "is_weather_imputed" else False

    # Impute missing values
    weather = _fill_missing_weather(weather)

    # Add temperature lag
    weather["temperature_lag_24h"] = weather["temperature_c"].shift(24)

    df = df.drop(columns=[c for c in weather.columns if c != "timestamp"], errors="ignore")
    return pd.merge(df.sort_values("timestamp"), weather, on="timestamp", how="left")


def build_feature_matrix(
    load_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Orchestrates the creation of the full feature matrix.

    Args:
        load_df: Raw load data.
        weather_df: Raw weather data.

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
        raise ValueError(
            f"load_df missing required columns: {required_cols - set(load_df.columns)}"
        )

    # Process features in order
    df = add_calendar_features(load_df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_weather_features(df, weather_df)

    # Final cleanup
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["feature_built_at"] = pd.Timestamp.now(tz="UTC")

    return df


def get_feature_columns() -> list[str]:
    """
    Returns the canonical list of feature columns used as model input.
    """
    return [
        "hour_of_day",
        "day_of_week",
        "month",
        "quarter",
        "is_weekend",
        "is_holiday",
        "is_holiday_eve",
        "days_since_epoch",
        "lag_24h",
        "lag_48h",
        "lag_168h",
        "rolling_7d_mean",
        "rolling_7d_std",
        "rolling_24h_mean",
        "temperature_c",
        "temperature_lag_24h",
        "humidity_pct",
        "cloud_cover_pct",
    ]
