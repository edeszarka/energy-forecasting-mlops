"""
Unit tests for the monitoring logic in src/drift.py.
"""

import pandas as pd

from src.drift import compute_prediction_mae, extract_drift_results


def test_extract_drift_results_v4_format():
    """Tests extraction from a mock evidently report dict."""
    mock_report = {
        "metrics": [
            {
                "metric": "DatasetDriftMetric",
                "result": {
                    "dataset_drift": True,
                    "number_of_drifted_columns": 2,
                    "drifted_columns": ["temp_c", "lag_24h"],
                },
            },
            {
                "metric": "ColumnDriftMetric",
                "parameters": {"column_name": "value_mwh"},
                "result": {"drift_score": 0.12},
            },
            {
                "metric": "ColumnDriftMetric",
                "parameters": {"column_name": "temperature_c"},
                "result": {"drift_score": 0.25},
            },
        ]
    }

    res = extract_drift_results(mock_report)

    assert res["dataset_drift"] is True
    assert res["n_drifted_features"] == 2
    assert "temp_c" in res["drifted_features"]
    assert res["drift_score_target"] == 0.12
    assert res["drift_score_temp"] == 0.25


def test_compute_prediction_mae_happy_path():
    """Tests MAE calculation with sufficient data."""
    df = pd.DataFrame(
        {
            "model_name": ["energy_lgbm_24h"] * 30,
            "horizon_hours": [24] * 30,
            "predicted_mwh": [100.0] * 30,
            "actual_mwh": [110.0] * 30,
        }
    )

    mae = compute_prediction_mae(df, "energy_lgbm_24h", 24)
    assert mae == 10.0


def test_compute_prediction_mae_insufficient_data():
    """Tests that None is returned when fewer than 24 actuals exist."""
    df = pd.DataFrame(
        {
            "model_name": ["energy_lgbm_24h"] * 10,
            "horizon_hours": [24] * 10,
            "predicted_mwh": [100.0] * 10,
            "actual_mwh": [110.0] * 10,
        }
    )

    mae = compute_prediction_mae(df, "energy_lgbm_24h", 24)
    assert mae is None


# FIX APPLIED: Added unit tests for drift result extraction and prediction MAE calculation.


# ── Retrain trigger logic (spec 011) ─────────────────────────────────────────

from datetime import UTC, datetime, timedelta  # noqa: E402

import pytest  # noqa: E402

from src.drift import (  # noqa: E402
    count_consecutive_hours,
    decide_retrain,
    seasonal_reference_windows,
    select_drift_columns,
)

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)


def hours_ago(n):
    return (NOW - timedelta(hours=n)).replace(tzinfo=None)


class TestSeasonalReferenceWindows:
    START = datetime(2026, 11, 10, 3, 0)
    END = datetime(2026, 11, 17, 3, 0)

    def test_same_weeks_previous_years_with_padding(self):
        windows = seasonal_reference_windows(self.START, self.END, years=2, pad_days=7)
        assert windows == [
            (pd.Timestamp("2025-11-04 03:00"), pd.Timestamp("2025-11-25 03:00")),
            (pd.Timestamp("2024-11-05 03:00"), pd.Timestamp("2024-11-26 03:00")),
        ]

    def test_weekdays_stay_aligned(self):
        for ref_start, _ in seasonal_reference_windows(self.START, self.END, pad_days=0):
            assert ref_start.dayofweek == pd.Timestamp(self.START).dayofweek

    def test_accepts_aware_timestamps(self):
        aware = seasonal_reference_windows(
            self.START.replace(tzinfo=UTC), self.END.replace(tzinfo=UTC), years=1
        )
        assert aware == seasonal_reference_windows(self.START, self.END, years=1)


class TestSelectDriftColumns:
    def test_excludes_calendar_features(self):
        cols = select_drift_columns(["temperature_c", "month", "hour_of_day", "lag_24h"])
        assert cols == ["temperature_c", "lag_24h"]


class TestCountConsecutiveDriftHours:
    def test_zero_when_current_check_has_no_drift(self):
        assert count_consecutive_hours([(hours_ago(1), True)], NOW, False) == 0

    def test_counts_contiguous_hourly_streak(self):
        history = [(hours_ago(1), True), (hours_ago(2), True), (hours_ago(3), False)]
        assert count_consecutive_hours(history, NOW, True) == 3

    def test_gap_in_hours_breaks_streak(self):
        # Rows exist but are spread over days (job outage) — not consecutive hours.
        history = [(hours_ago(1), True), (hours_ago(30), True), (hours_ago(60), True)]
        assert count_consecutive_hours(history, NOW, True) == 2

    def test_history_order_does_not_matter(self):
        history = [(hours_ago(2), True), (hours_ago(1), True)]
        assert count_consecutive_hours(history, NOW, True) == 3

    def test_empty_history(self):
        assert count_consecutive_hours([], NOW, True) == 1

    def test_daily_step_counts_contiguous_days(self):
        history = [(hours_ago(24), True), (hours_ago(48), True), (hours_ago(72), False)]
        assert count_consecutive_hours(history, NOW, True, step="D") == 3

    def test_daily_step_missing_day_breaks_streak(self):
        history = [(hours_ago(24), True), (hours_ago(72), True)]
        assert count_consecutive_hours(history, NOW, True, step="D") == 2

    def test_daily_step_with_older_hourly_rows(self):
        # Hourly rows from before the daily schedule: the newest in each day counts.
        yesterday = NOW - timedelta(days=1)
        # Newest first, as 03_drift_check reads it.
        history = [
            (yesterday.replace(hour=23, tzinfo=None), True),
            (yesterday.replace(hour=1, tzinfo=None), False),
        ]
        assert count_consecutive_hours(history, NOW, True, step="D") == 2


class TestDecideRetrain:
    def test_no_drift(self):
        assert decide_retrain(False, 0, 3, NOW)[0] is False

    def test_below_consecutive_threshold(self):
        assert decide_retrain(True, 2, 3, NOW)[0] is False

    def test_triggers_on_sustained_drift(self):
        should, reason = decide_retrain(True, 3, 3, NOW)
        assert should is True
        assert "3 consecutive hours" in reason

    def test_cooldown_blocks(self):
        should, reason = decide_retrain(True, 5, 3, NOW, last_retrain_at=hours_ago(5))
        assert should is False
        assert "Cooldown" in reason

    def test_cooldown_expired(self):
        assert decide_retrain(True, 5, 3, NOW, last_retrain_at=hours_ago(25))[0] is True

    def test_recent_pending_request_blocks(self):
        should, reason = decide_retrain(True, 5, 3, NOW, pending_since=NOW - timedelta(hours=1))
        assert should is False
        assert "pending" in reason

    @pytest.mark.parametrize("age_hours", [6, 48])
    def test_stale_pending_request_no_longer_blocks(self, age_hours):
        # A failed retrain leaves its flag behind; it must not disable triggering forever.
        pending = NOW - timedelta(hours=age_hours)
        assert decide_retrain(True, 5, 3, NOW, pending_since=pending)[0] is True
