"""
Unit tests for the feature engineering logic in src/features.py.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from freezegun import freeze_time

from src.features import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    add_temperature_features,
    build_feature_matrix,
    get_feature_columns,
)
from src.config import MIN_TRAINING_ROWS

@pytest.fixture
def make_hourly_df():
    """Fixture to create a mock hourly load DataFrame."""
    def _make(n_hours=200, start="2024-01-01", base_load=4000.0):
        ts = pd.date_range(start=start, periods=n_hours, freq="H", tz="UTC")
        # Add sinusoidal daily pattern
        hours = np.arange(n_hours)
        load = base_load + 500 * np.sin(2 * np.pi * hours / 24)
        return pd.DataFrame({"timestamp": ts, "value_mwh": load})
    return _make

@pytest.fixture
def make_temp_df():
    """Fixture to create a mock hourly temperature DataFrame."""
    def _make(n_hours=200, start="2024-01-01"):
        ts = pd.date_range(start=start, periods=n_hours, freq="H", tz="UTC")
        temp = 10 + 5 * np.cos(2 * np.pi * np.arange(n_hours) / 24)
        return pd.DataFrame({
            "timestamp": ts, 
            "temperature_c": temp, 
            "is_temp_imputed": False
        })
    return _make

def test_calendar_features_columns_present(make_hourly_df):
    df = make_hourly_df(n_hours=10)
    feat_df = add_calendar_features(df)
    
    expected_cols = [
        "hour_of_day", "day_of_week", "month", "quarter", 
        "is_weekend", "is_holiday", "is_holiday_eve", "days_since_epoch"
    ]
    for col in expected_cols:
        assert col in feat_df.columns

def test_calendar_features_hour_is_local_time():
    # 2024-01-01 00:00 UTC is 01:00 Budapest time
    df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-01 00:00", tz="UTC")],
        "value_mwh": [4000.0]
    })
    feat_df = add_calendar_features(df)
    assert feat_df.iloc[0]["hour_of_day"] == 1

def test_calendar_features_is_holiday_christmas():
    # 2024-12-25 is Christmas
    df = pd.DataFrame({
        "timestamp": [
            pd.Timestamp("2024-12-25 10:00", tz="UTC"),
            pd.Timestamp("2024-12-26 10:00", tz="UTC")
        ],
        "value_mwh": [4000.0, 4100.0]
    })
    feat_df = add_calendar_features(df)
    assert feat_df["is_holiday"].all()

def test_calendar_features_empty_df():
    df = pd.DataFrame(columns=["timestamp", "value_mwh"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    feat_df = add_calendar_features(df)
    assert feat_df.empty
    assert "hour_of_day" in feat_df.columns

def test_lag_features_correct_shift(make_hourly_df):
    df = make_hourly_df(n_hours=200)
    feat_df = add_lag_features(df)
    
    # Row 100 lag_24h should be value_mwh from row 76
    assert feat_df.iloc[100]["lag_24h"] == df.iloc[76]["value_mwh"]
    assert feat_df.iloc[100]["lag_168h"] == df.iloc[100-168]["value_mwh"]

def test_lag_features_gap_detection(make_hourly_df):
    df = make_hourly_df(n_hours=200)
    # Drop row 50
    df = df.drop(index=50).reset_index(drop=True)
    
    feat_df = add_lag_features(df)
    # The reindexing logic in add_lag_features should handle the gap
    # and has_lag_gap should flag rows that shifted the NaN
    assert feat_df["has_lag_gap"].any()

def test_rolling_features_min_periods(make_hourly_df):
    # With only 80 rows, rolling_7d (168h) needs min_periods=72
    df = make_hourly_df(n_hours=80)
    feat_df = add_rolling_features(df)
    
    # First 71 rows should be NaN for 7d rolling
    assert feat_df.iloc[70]["rolling_7d_mean"] is np.nan or np.isnan(feat_df.iloc[70]["rolling_7d_mean"])
    # Row 72 should have a value (index 71 is the 72nd row)
    assert not np.isnan(feat_df.iloc[71]["rolling_7d_mean"])

def test_temperature_merge_empty_temp(make_hourly_df):
    df = make_hourly_df(n_hours=10)
    temp_df = pd.DataFrame(columns=["timestamp", "temperature_c", "is_temp_imputed"])
    
    feat_df = add_temperature_features(df, temp_df)
    assert feat_df["temp_missing"].all()
    assert feat_df["temperature_c"].isna().all()

def test_temperature_fill_forward(make_hourly_df):
    df = make_hourly_df(n_hours=20)
    temp_df = pd.DataFrame({
        "timestamp": df["timestamp"],
        "temperature_c": [10.0] * 5 + [np.nan] * 3 + [10.0] * 12,
        "is_temp_imputed": [False] * 20
    })
    
    feat_df = add_temperature_features(df, temp_df)
    # 3 hour gap should be filled
    assert not feat_df["temperature_c"].isna().any()
    assert feat_df.iloc[6]["is_temp_imputed"] == True

def test_build_feature_matrix_column_contract(make_hourly_df, make_temp_df):
    # Need enough rows to pass MIN_TRAINING_ROWS
    n = MIN_TRAINING_ROWS + 200
    load_df = make_hourly_df(n_hours=n)
    temp_df = make_temp_df(n_hours=n)
    
    feat_df = build_feature_matrix(load_df, temp_df)
    
    for col in get_feature_columns():
        assert col in feat_df.columns

def test_build_feature_matrix_raises_on_insufficient_data():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="H", tz="UTC"),
        "value_mwh": [4000.0] * 10
    })
    with pytest.raises(ValueError, match="Insufficient data"):
        build_feature_matrix(df, pd.DataFrame())

def test_training_serving_skew_guard(make_hourly_df, make_temp_df):
    n = MIN_TRAINING_ROWS + 100
    load_df = make_hourly_df(n_hours=n)
    temp_df = make_temp_df(n_hours=n)
    
    df1 = build_feature_matrix(load_df, temp_df)
    df2 = build_feature_matrix(load_df, temp_df)
    
    pd.testing.assert_frame_equal(df1, df2)
