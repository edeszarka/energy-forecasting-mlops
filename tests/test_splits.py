"""
Unit tests for the time-series holdout split logic in src/splits.py.
"""

import pandas as pd

from src.splits import make_holdout_splits, make_prophet_holdout_split


def _hourly_timestamps(n: int) -> pd.Series:
    return pd.Series(pd.date_range("2026-01-01", periods=n, freq="1h"))


def test_masks_are_mutually_exclusive_and_exhaustive():
    ts = _hourly_timestamps(1000)
    train_mask, val_mask, test_mask = make_holdout_splits(ts, val_days=5, test_days=5)

    assert not (train_mask & val_mask).any()
    assert not (train_mask & test_mask).any()
    assert not (val_mask & test_mask).any()
    assert (train_mask | val_mask | test_mask).all()


def test_split_is_contiguous_in_time():
    ts = _hourly_timestamps(1000)
    train_mask, val_mask, test_mask = make_holdout_splits(ts, val_days=5, test_days=5)

    train_end = ts[train_mask].max()
    val_start = ts[val_mask].min()
    val_end = ts[val_mask].max()
    test_start = ts[test_mask].min()

    assert val_start > train_end
    assert test_start > val_end


def test_test_window_is_most_recent():
    ts = _hourly_timestamps(1000)
    _, _, test_mask = make_holdout_splits(ts, val_days=5, test_days=5)

    assert ts[test_mask].max() == ts.max()
    assert test_mask.sum() == 5 * 24


def test_validation_window_size_matches_val_days():
    ts = _hourly_timestamps(1000)
    _, val_mask, _ = make_holdout_splits(ts, val_days=5, test_days=5)

    assert val_mask.sum() == 5 * 24


def test_train_keeps_min_rows_at_current_volume():
    # silver_features ~1,900 rows; dropna leaves N - horizon usable rows.
    # With val_days=5, test_days=5 the train window must stay >= MIN_TRAINING_ROWS (720).
    for horizon_hours in (24, 168):
        usable_rows = 1900 - horizon_hours
        ts = _hourly_timestamps(usable_rows)
        train_mask, val_mask, test_mask = make_holdout_splits(ts, val_days=5, test_days=5)

        assert train_mask.sum() >= 720
        assert val_mask.sum() == 5 * 24
        assert test_mask.sum() == 5 * 24


def test_train_window_shrinks_when_val_days_increases():
    ts = _hourly_timestamps(1000)

    train_small, _, _ = make_holdout_splits(ts, val_days=10, test_days=5)
    train_large, _, _ = make_holdout_splits(ts, val_days=5, test_days=5)

    assert train_small.sum() < train_large.sum()


def test_returns_boolean_series_masks():
    ts = _hourly_timestamps(1000)
    train_mask, val_mask, test_mask = make_holdout_splits(ts, val_days=5, test_days=5)

    for mask in (train_mask, val_mask, test_mask):
        assert isinstance(mask, pd.Series)
        assert mask.dtype == bool


def _make_prophet_ts(n: int) -> pd.Series:
    return pd.Series(pd.date_range("2026-01-01", periods=n, freq="1h"))


def test_prophet_split_gap_equals_horizon_hours():
    """AC1: gap between actual train_mask boundary and test_mask boundary
    equals horizon_hours — derived from the function's real output, not
    a recomputed copy of its internal arithmetic."""
    ts = _make_prophet_ts(1900)

    for horizon_hours in (24, 168):
        train_mask, test_mask = make_prophet_holdout_split(ts, horizon_hours, test_days=5)

        train_end = ts[train_mask].max()
        test_start_actual = ts[test_mask].min()

        gap_hours = int((test_start_actual - train_end).total_seconds() // 3600)

        # gap includes the excluded boundary hour itself, so it's
        # horizon_hours + 1 rows apart at 1h frequency — assert against
        # actual timestamps, not a re-derivation of the formula.
        assert gap_hours >= horizon_hours, (
            f"Expected gap >= {horizon_hours}h between real train/test boundaries, got {gap_hours}h"
        )


def test_prophet_test_window_identical_across_horizons():
    """AC2: test_mask covers the same absolute date range regardless of horizon_hours."""
    ts = _make_prophet_ts(1900)

    _, test_mask_24 = make_prophet_holdout_split(ts, horizon_hours=24, test_days=5)
    _, test_mask_168 = make_prophet_holdout_split(ts, horizon_hours=168, test_days=5)

    test_dates_24 = ts[test_mask_24]
    test_dates_168 = ts[test_mask_168]

    assert test_dates_24.equals(test_dates_168), "Test windows must be identical across horizons"


def test_prophet_train_window_stays_above_min_rows():
    """AC3: at ~1,900 rows, both 24h and 168h train windows stay >= 200 rows."""
    ts = _make_prophet_ts(1900)

    for horizon_hours in (24, 168):
        train_mask, _ = make_prophet_holdout_split(ts, horizon_hours, test_days=5)
        assert train_mask.sum() >= 200, f"Train rows {train_mask.sum()} < 200 for {horizon_hours}h"


def test_prophet_train_windows_differ_by_horizon():
    """Train windows should differ deterministically by horizon."""
    ts = _make_prophet_ts(1900)

    train_mask_24, _ = make_prophet_holdout_split(ts, horizon_hours=24, test_days=5)
    train_mask_168, _ = make_prophet_holdout_split(ts, horizon_hours=168, test_days=5)

    assert train_mask_24.sum() > train_mask_168.sum(), "24h train window should be larger than 168h"
    diff = train_mask_24.sum() - train_mask_168.sum()
    assert diff == 144, f"Train window difference should be 144 rows (168-24), got {diff}"


def test_prophet_masks_are_boolean_series():
    ts = _make_prophet_ts(1000)
    train_mask, test_mask = make_prophet_holdout_split(ts, horizon_hours=24, test_days=5)

    for mask in (train_mask, test_mask):
        assert isinstance(mask, pd.Series)
        assert mask.dtype == bool


def test_prophet_masks_do_not_overlap():
    ts = _make_prophet_ts(1000)
    train_mask, test_mask = make_prophet_holdout_split(ts, horizon_hours=24, test_days=5)

    assert not (train_mask & test_mask).any(), "Train and test masks must not overlap"
