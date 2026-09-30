"""Evaluative tests for src.silver_window.filter_to_core_window (spec 009 AC2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import LAG_HOURS, MIN_TRAINING_ROWS, TRAIN_TEST_SPLIT_DAYS
from src.features import build_feature_matrix
from src.silver_window import filter_to_core_window

MAX_LAG = max(LAG_HOURS)
# Mirrors 02_transform.py's default lookback_hours widget (720 = 30 days).
LOOKBACK_HOURS = TRAIN_TEST_SPLIT_DAYS * 24
WINDOW_HOURS = LOOKBACK_HOURS + MAX_LAG
CHECKED_COLUMNS = ["lag_168h", "rolling_7d_mean", "rolling_7d_std", "rolling_24h_mean"]


def _synthetic(
    periods: int, start: str = "2026-01-01T00:00:00+00:00"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range(start=start, periods=periods, freq="h", tz="UTC")
    steps = np.arange(periods, dtype=float)
    load = pd.DataFrame(
        {
            "timestamp": index,
            "value_mwh": 1000.0 + steps + 50.0 * np.sin(steps / 24.0),
        }
    )
    temp = pd.DataFrame(
        {
            "timestamp": index,
            "temperature_c": 10.0 + 5.0 * np.sin(steps / 24.0),
            "humidity_pct": 60.0,
            "cloud_cover_pct": 40.0,
        }
    )
    return load, temp


def _simulate_run(
    load: pd.DataFrame, temp: pd.DataFrame, run_date: pd.Timestamp
) -> tuple[pd.Timestamp, pd.DataFrame]:
    window_start = run_date - pd.Timedelta(hours=WINDOW_HOURS)
    load_slice = load[(load["timestamp"] >= window_start) & (load["timestamp"] <= run_date)]
    temp_slice = temp[(temp["timestamp"] >= window_start) & (temp["timestamp"] <= run_date)]
    assert len(load_slice) >= MIN_TRAINING_ROWS
    feature_pd = build_feature_matrix(
        load_slice.reset_index(drop=True), temp_slice.reset_index(drop=True)
    )
    return window_start, feature_pd


# ── (a) boundary filter ──────────────────────────────────────────────────────


def test_filters_rows_before_cutoff():
    window_start = pd.Timestamp("2026-03-01T00:00:00+00:00")
    cutoff = window_start + pd.Timedelta(hours=MAX_LAG)
    feature_pd = pd.DataFrame(
        {
            "timestamp": [
                cutoff - pd.Timedelta(hours=1),
                cutoff,
                cutoff + pd.Timedelta(hours=1),
            ],
            "value_mwh": [1.0, 2.0, 3.0],
        }
    )

    result = filter_to_core_window(feature_pd, window_start, MAX_LAG)

    assert list(result["timestamp"]) == [cutoff, cutoff + pd.Timedelta(hours=1)]
    assert list(result.index) == [0, 1]
    assert (result["timestamp"] >= cutoff).all()
    # The row exactly AT the cutoff is KEPT, not excluded.
    assert cutoff in set(result["timestamp"])


# ── (b) defensive empty case ─────────────────────────────────────────────────


def test_empty_result_when_all_rows_are_margin():
    window_start = pd.Timestamp("2026-03-01T00:00:00+00:00")
    cutoff = window_start + pd.Timedelta(hours=MAX_LAG)
    feature_pd = pd.DataFrame(
        {
            "timestamp": [window_start, cutoff - pd.Timedelta(hours=1)],
            "value_mwh": [1.0, 2.0],
        }
    )

    result = filter_to_core_window(feature_pd, window_start, MAX_LAG)

    assert result.empty
    assert list(result.columns) == ["timestamp", "value_mwh"]


# ── (c) THE evaluative test: overlapping runs never degrade written rows ─────


def test_overlapping_runs_never_degrade_previously_written_rows():
    load, temp = _synthetic(40 * 24)
    start = pd.Timestamp("2026-01-01T00:00:00+00:00")
    run_date_1 = start + pd.Timedelta(hours=40 * 24 - 10)
    run_date_2 = run_date_1 + pd.Timedelta(hours=1)

    window_start_1, features_1 = _simulate_run(load, temp, run_date_1)
    window_start_2, features_2 = _simulate_run(load, temp, run_date_2)

    filtered_1 = filter_to_core_window(features_1, window_start_1, MAX_LAG).set_index("timestamp")
    filtered_2 = filter_to_core_window(features_2, window_start_2, MAX_LAG).set_index("timestamp")

    common = filtered_1.index.intersection(filtered_2.index)
    assert len(common) > 0

    # Every row with a full-context value in run 1 keeps that value in run 2 —
    # no previously-written row is ever downgraded to NaN by a later MERGE.
    for col in CHECKED_COLUMNS:
        previously_present = filtered_1.loc[common, col].notna()
        assert filtered_2.loc[common, col][previously_present].notna().all()
        pd.testing.assert_series_equal(
            filtered_2.loc[common, col], filtered_1.loc[common, col], check_names=False
        )

    # Prove the fix matters: without the filter, the raw feature matrix of the
    # later run contains a row that the earlier run wrote with a valid lag_168h
    # but which the later run would overwrite with a margin-induced NaN.
    raw_1 = features_1.set_index("timestamp")["lag_168h"]
    raw_2 = features_2.set_index("timestamp")["lag_168h"]
    raw_common = raw_1.index.intersection(raw_2.index)

    boundary = window_start_1 + pd.Timedelta(hours=MAX_LAG)
    assert boundary in raw_common
    assert pd.notna(raw_1.loc[boundary])
    assert pd.isna(raw_2.loc[boundary])

    regressed = [ts for ts in raw_common if pd.notna(raw_1.loc[ts]) and pd.isna(raw_2.loc[ts])]
    assert regressed
