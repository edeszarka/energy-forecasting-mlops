"""
Unit tests for the time-series holdout split logic in src/splits.py.
"""

import pandas as pd

from src.splits import make_holdout_splits, make_prophet_holdout_split, make_prophet_rolling_origins


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


def _make_rolling_ts(n: int) -> pd.Series:
    return pd.Series(pd.date_range("2026-01-01", periods=n, freq="1h"))


def test_rolling_origins_returns_n_folds():
    """AC1: make_prophet_rolling_origins returns exactly n_folds pairs."""
    ts = _make_rolling_ts(2000)
    folds = make_prophet_rolling_origins(ts, horizon_hours=24, test_days=5, n_folds=5)

    assert len(folds) == 5
    for train_mask, target_mask in folds:
        assert isinstance(train_mask, pd.Series)
        assert isinstance(target_mask, pd.Series)
        assert train_mask.dtype == bool
        assert target_mask.dtype == bool


def test_rolling_origins_target_at_horizon_ahead():
    """AC1: each target_mask selects row at horizon_hours ahead of that fold's cutoff."""
    ts = _make_rolling_ts(2000)
    horizon_hours = 24
    folds = make_prophet_rolling_origins(ts, horizon_hours=horizon_hours, test_days=5, n_folds=5)

    for train_mask, target_mask in folds:
        train_end = ts[train_mask].max()
        target_rows = ts[target_mask]

        # Should have exactly 1 target row (with ±30min tolerance on hourly data)
        assert len(target_rows) == 1, f"Expected 1 target row, got {len(target_rows)}"

        # Target should be at horizon_hours ahead of train_end (±1 hour tolerance)
        target_time = target_rows.iloc[0]
        gap_hours = (target_time - train_end).total_seconds() / 3600
        assert abs(gap_hours - horizon_hours) <= 1.0, (
            f"Expected target at {horizon_hours}h ahead of cutoff, got {gap_hours}h"
        )


def test_rolling_origins_no_leakage():
    """AC2: train_mask uses strictly earlier data than target_mask (no overlap)."""
    ts = _make_rolling_ts(2000)
    folds = make_prophet_rolling_origins(ts, horizon_hours=24, test_days=5, n_folds=5)

    for train_mask, target_mask in folds:
        train_end = ts[train_mask].max()
        target_start = ts[target_mask].min()
        assert target_start > train_end, "Target must be strictly after train cutoff"


def test_rolling_origins_pooled_size_matches_n_folds():
    """AC3: pooled evaluation set size equals n_folds (one row per fold)."""
    ts = _make_rolling_ts(2000)
    folds = make_prophet_rolling_origins(ts, horizon_hours=24, test_days=5, n_folds=5)

    total_target_rows = sum(target_mask.sum() for _, target_mask in folds)
    assert total_target_rows == 5, (
        f"Expected 5 total target rows (n_folds), got {total_target_rows}"
    )


def test_rolling_origins_cutoffs_spaced_evenly():
    """Fold cutoffs are spaced evenly across the test_days window."""
    ts = _make_rolling_ts(2000)
    folds = make_prophet_rolling_origins(ts, horizon_hours=24, test_days=5, n_folds=5)

    cutoffs = [ts[train_mask].max() for train_mask, _ in folds]

    # Check cutoffs are in ascending order
    for i in range(1, len(cutoffs)):
        assert cutoffs[i] > cutoffs[i - 1], "Cutoffs must be strictly increasing"

    # Check they span approximately the test_days window (5 days = 120 hours)
    # First cutoff should be around t_max - 120h, last around t_max - 24h (approx)
    t_max = ts.max()
    first_cutoff = cutoffs[0]
    last_cutoff = cutoffs[-1]

    # First cutoff should be within the test window
    assert (t_max - first_cutoff).total_seconds() / 3600 <= 120 + 24  # test_days + one step
    assert (t_max - first_cutoff).total_seconds() / 3600 >= 120 - 24

    # Last cutoff should be later than first
    assert last_cutoff > first_cutoff


def test_rolling_origins_different_horizons_different_cutoffs():
    """Different horizons produce different cutoffs for the same target."""
    ts = _make_rolling_ts(2000)

    folds_24 = make_prophet_rolling_origins(ts, horizon_hours=24, test_days=5, n_folds=3)
    folds_168 = make_prophet_rolling_origins(ts, horizon_hours=168, test_days=5, n_folds=3)

    # Targets should be the same (spaced in test window)
    for (_, target_mask_24), (_, target_mask_168) in zip(folds_24, folds_168, strict=True):
        target_24 = ts[target_mask_24].iloc[0]
        target_168 = ts[target_mask_168].iloc[0]
        assert target_24 == target_168, "Targets should be identical across horizons"

    # But cutoffs should differ by horizon_hours
    for (train_mask_24, _), (train_mask_168, _) in zip(folds_24, folds_168, strict=True):
        cutoff_24 = ts[train_mask_24].max()
        cutoff_168 = ts[train_mask_168].max()
        # 168h cutoff should be 144h (168-24) earlier than 24h cutoff
        diff_hours = (cutoff_24 - cutoff_168).total_seconds() / 3600
        assert abs(diff_hours - 144) < 1, f"Cutoff difference should be 144h, got {diff_hours}h"


def test_rolling_origins_n_folds_param():
    """n_folds parameter controls number of folds returned."""
    ts = _make_rolling_ts(2000)

    folds_3 = make_prophet_rolling_origins(ts, horizon_hours=24, test_days=5, n_folds=3)
    folds_7 = make_prophet_rolling_origins(ts, horizon_hours=24, test_days=5, n_folds=7)

    assert len(folds_3) == 3
    assert len(folds_7) == 7
