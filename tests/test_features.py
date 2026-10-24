"""
Unit tests for the feature engineering logic in src/features.py.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import MIN_TRAINING_ROWS, MODEL_INPUT_FEATURES
from src.features import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    add_weather_features,
    build_feature_matrix,
    get_feature_columns,
)


@pytest.fixture
def make_hourly_df():
    """Fixture to create a mock hourly load DataFrame."""

    def _make(n_hours=200, start="2024-01-01", base_load=4000.0):
        ts = pd.date_range(start=start, periods=n_hours, freq="h", tz="UTC")
        # Add sinusoidal daily pattern
        hours = np.arange(n_hours)
        load = base_load + 500 * np.sin(2 * np.pi * hours / 24)
        return pd.DataFrame({"timestamp": ts, "value_mwh": load})

    return _make


@pytest.fixture
def make_weather_df():
    """Fixture to create a mock hourly weather DataFrame."""

    def _make(n_hours=200, start="2024-01-01"):
        ts = pd.date_range(start=start, periods=n_hours, freq="h", tz="UTC")
        temp = 10 + 5 * np.cos(2 * np.pi * np.arange(n_hours) / 24)
        return pd.DataFrame(
            {
                "timestamp": ts,
                "temperature_c": temp,
                "humidity_pct": [60.0] * n_hours,
                "cloud_cover_pct": [20.0] * n_hours,
                "is_weather_imputed": False,
            }
        )

    return _make


def test_calendar_features_columns_present(make_hourly_df):
    df = make_hourly_df(n_hours=10)
    feat_df = add_calendar_features(df)

    expected_cols = [
        "hour_of_day",
        "day_of_week",
        "month",
        "quarter",
        "is_weekend",
        "is_holiday",
        "is_holiday_eve",
        "days_since_epoch",
    ]
    for col in expected_cols:
        assert col in feat_df.columns


def test_calendar_features_hour_is_local_time():
    # 2024-01-01 00:00 UTC is 01:00 Budapest time
    df = pd.DataFrame(
        {"timestamp": [pd.Timestamp("2024-01-01 00:00", tz="UTC")], "value_mwh": [4000.0]}
    )
    feat_df = add_calendar_features(df)
    assert feat_df.iloc[0]["hour_of_day"] == 1


def test_calendar_features_is_holiday_christmas():
    # 2024-12-25 is Christmas
    df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-12-25 10:00", tz="UTC"),
                pd.Timestamp("2024-12-26 10:00", tz="UTC"),
            ],
            "value_mwh": [4000.0, 4100.0],
        }
    )
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
    # Row 168 is the first row with a valid lag_168h
    assert feat_df.iloc[168]["lag_168h"] == df.iloc[0]["value_mwh"]


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
    assert feat_df.iloc[70]["rolling_7d_mean"] is np.nan or np.isnan(
        feat_df.iloc[70]["rolling_7d_mean"]
    )
    # Row 72 should have a value (index 71 is the 72nd row)
    assert not np.isnan(feat_df.iloc[71]["rolling_7d_mean"])


def test_rolling_features_are_hour_aligned_across_a_gap(make_hourly_df):
    """AC4 (spec 014 T4): a gap of N hours does not shift the rolling windows."""
    df = make_hourly_df(n_hours=300)
    gap = range(200, 210)  # 10 missing hours
    gapped = df.drop(index=gap)

    feat = add_rolling_features(gapped).set_index("timestamp")

    values = df.set_index("timestamp")["value_mwh"].drop(df["timestamp"].iloc[list(gap)])
    for t in df["timestamp"].iloc[[215, 230, 299]]:
        last_24h = values[(values.index > t - pd.Timedelta(hours=24)) & (values.index <= t)]
        last_7d = values[(values.index > t - pd.Timedelta(hours=168)) & (values.index <= t)]
        assert feat.loc[t, "rolling_24h_mean"] == pytest.approx(last_24h.mean())
        assert feat.loc[t, "rolling_7d_mean"] == pytest.approx(last_7d.mean())
        assert feat.loc[t, "rolling_7d_std"] == pytest.approx(last_7d.std())


def test_rolling_min_periods_count_hours_with_values(make_hourly_df):
    df = make_hourly_df(n_hours=40)
    # 12 hours, then a 20-hour gap: at the next row only 11 of the last 24 hours
    # have a value, so min_periods=12 is not met (row-based rolling would see 12).
    gapped = df.drop(index=range(12, 32))

    feat = add_rolling_features(gapped).set_index("timestamp")

    assert pd.isna(feat.loc[df["timestamp"].iloc[32], "rolling_24h_mean"])


def test_temperature_lag_is_hour_aligned_across_a_load_gap(make_hourly_df, make_weather_df):
    """AC4: missing load hours do not shift temperature_lag_24h."""
    load = make_hourly_df(n_hours=100).drop(index=range(50, 55))
    weather = make_weather_df(n_hours=100)
    weather["temperature_c"] = np.arange(100, dtype=float)  # temperature == hour index

    feat = add_weather_features(load, weather).set_index("timestamp")

    for i in (60, 74, 99):
        t = weather["timestamp"].iloc[i]
        assert feat.loc[t, "temperature_lag_24h"] == i - 24


def test_weather_merge_empty_df(make_hourly_df):
    df = make_hourly_df(n_hours=10)
    weather_df = pd.DataFrame(
        columns=[
            "timestamp",
            "temperature_c",
            "humidity_pct",
            "cloud_cover_pct",
            "is_weather_imputed",
        ]
    )

    feat_df = add_weather_features(df, weather_df)
    assert feat_df["temp_missing"].all()
    assert feat_df["temperature_c"].isna().all()


def test_weather_fill_forward(make_hourly_df):
    df = make_hourly_df(n_hours=20)
    weather_df = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "temperature_c": [10.0] * 5 + [np.nan] * 3 + [10.0] * 12,
            "humidity_pct": [50.0] * 20,
            "cloud_cover_pct": [10.0] * 20,
            "is_weather_imputed": [False] * 20,
        }
    )

    feat_df = add_weather_features(df, weather_df)
    # 3 hour gap should be filled
    assert not feat_df["temperature_c"].isna().any()
    assert feat_df.iloc[6]["is_weather_imputed"]


def test_build_feature_matrix_column_contract(make_hourly_df, make_weather_df):
    # Need enough rows to pass MIN_TRAINING_ROWS
    n = MIN_TRAINING_ROWS + 200
    load_df = make_hourly_df(n_hours=n)
    weather_df = make_weather_df(n_hours=n)

    feat_df = build_feature_matrix(load_df, weather_df)

    for col in get_feature_columns():
        assert col in feat_df.columns


def test_build_feature_matrix_raises_on_insufficient_data():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC"),
            "value_mwh": [4000.0] * 10,
        }
    )
    with pytest.raises(ValueError, match="Insufficient data"):
        build_feature_matrix(df, pd.DataFrame())


def test_training_serving_skew_guard(make_hourly_df, make_weather_df):
    n = MIN_TRAINING_ROWS + 100
    load_df = make_hourly_df(n_hours=n)
    weather_df = make_weather_df(n_hours=n)

    df1 = build_feature_matrix(load_df, weather_df)
    df2 = build_feature_matrix(load_df, weather_df)

    # Exclude feature_built_at as it will always differ
    pd.testing.assert_frame_equal(
        df1.drop(columns=["feature_built_at"]), df2.drop(columns=["feature_built_at"])
    )


def test_silver_features_include_humidity_and_cloud_cover(make_hourly_df, make_weather_df):
    """Verifies the silver feature matrix includes humidity_pct and cloud_cover_pct
    for model consumption. These columns are already in get_feature_columns()."""
    n = MIN_TRAINING_ROWS + 200
    load_df = make_hourly_df(n_hours=n)
    weather_df = make_weather_df(n_hours=n)
    feat_df = build_feature_matrix(load_df, weather_df)

    assert "humidity_pct" in feat_df.columns
    assert "cloud_cover_pct" in feat_df.columns
    assert feat_df["humidity_pct"].notna().any()
    assert feat_df["cloud_cover_pct"].notna().any()


def test_model_input_includes_humidity_and_cloud_cover():
    """Asserts the model input feature list includes humidity_pct and cloud_cover_pct.

    Currently FAILS because these columns exist in get_feature_columns() (silver output)
    but are not in the training notebook's FEATURE_COLS or predict notebook's MODEL_FEATURES.
    Passes once both are added — tests the cross-file contract between feature generation
    and model consumption.
    """
    missing = [c for c in ["humidity_pct", "cloud_cover_pct"] if c not in MODEL_INPUT_FEATURES]
    assert not missing, f"Model input feature list missing columns: {missing}"
