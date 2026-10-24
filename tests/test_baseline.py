"""Unit tests for naive baseline metrics."""

import numpy as np
import pandas as pd
import pytest

from src.baseline import compute_naive_baseline_metrics, seasonal_naive_forecast


def test_compute_naive_baseline_metrics_happy_path():
    metrics = compute_naive_baseline_metrics(pd.Series([100.0, 200.0]), pd.Series([90.0, 220.0]))

    assert metrics == {
        "naive_mae": 15.0,
        "naive_rmse": pytest.approx(15.811388300841896),
        "naive_mape": 10.0,
    }


def test_compute_naive_baseline_metrics_drops_null_pairs():
    metrics = compute_naive_baseline_metrics(
        pd.Series([100.0, None, 300.0]), pd.Series([90.0, 500.0, None])
    )

    assert metrics == {"naive_mae": 10.0, "naive_rmse": 10.0, "naive_mape": 10.0}


def test_compute_naive_baseline_metrics_handles_zero_actuals():
    metrics = compute_naive_baseline_metrics(pd.Series([0.0, 0.0]), pd.Series([10.0, -10.0]))

    assert metrics["naive_mae"] == 10.0
    assert metrics["naive_rmse"] == 10.0
    assert metrics["naive_mape"] == 0.0


# ── Seasonal-naive serving fallback (spec 013 T3, AC6) ──────────────────────

RUN_AT = pd.Timestamp("2026-10-24T05:00:00+00:00")
TARGETS = [RUN_AT + pd.Timedelta(hours=h) for h in range(25, 49)]


def _history(start: pd.Timestamp, hours: int, tz_aware: bool = True) -> pd.DataFrame:
    index = pd.date_range(start=start, periods=hours, freq="h")
    if not tz_aware:
        index = index.tz_convert("UTC").tz_localize(None)
    return pd.DataFrame({"timestamp": index, "value_mwh": np.arange(hours, dtype=float) + 4000.0})


def test_seasonal_naive_predicts_value_one_week_earlier():
    history = _history(RUN_AT - pd.Timedelta(hours=200), 200)
    value_by_ts = history.set_index("timestamp")["value_mwh"]

    result = seasonal_naive_forecast(history, TARGETS)

    assert list(result["timestamp"]) == TARGETS
    for target, predicted in zip(result["timestamp"], result["predicted_mwh"], strict=True):
        assert predicted == value_by_ts[target - pd.Timedelta(hours=168)]


def test_seasonal_naive_matches_naive_utc_history_timestamps():
    # Spark may hand back naive UTC timestamps; targets are tz-aware.
    history = _history(RUN_AT - pd.Timedelta(hours=200), 200, tz_aware=False)

    result = seasonal_naive_forecast(history, TARGETS)

    assert len(result) == len(TARGETS)
    assert result["timestamp"].iloc[0] == TARGETS[0]


def test_seasonal_naive_skips_target_with_missing_source_hour():
    history = _history(RUN_AT - pd.Timedelta(hours=200), 200)
    missing_source = TARGETS[3] - pd.Timedelta(hours=168)
    history = history[history["timestamp"] != missing_source].copy()
    nulled_source = TARGETS[5] - pd.Timedelta(hours=168)
    history.loc[history["timestamp"] == nulled_source, "value_mwh"] = np.nan

    result = seasonal_naive_forecast(history, TARGETS)

    assert len(result) == len(TARGETS) - 2
    assert TARGETS[3] not in set(result["timestamp"])
    assert TARGETS[5] not in set(result["timestamp"])


def test_seasonal_naive_raises_when_no_target_has_a_source_value():
    # Silver stale by more than 120h: every T-168h source hour is missing.
    history = _history(RUN_AT - pd.Timedelta(hours=400), 200)

    with pytest.raises(ValueError, match="no T-168h value"):
        seasonal_naive_forecast(history, TARGETS)
