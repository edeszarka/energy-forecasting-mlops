"""
Unit tests for the time-series holdout split logic in src/splits.py.
"""

import pandas as pd

from src.splits import make_holdout_splits


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
