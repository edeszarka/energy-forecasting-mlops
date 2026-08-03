"""Time-series holdout split helpers for training notebook evaluation."""

from __future__ import annotations

import pandas as pd


def make_holdout_splits(
    timestamps: pd.Series,
    val_days: int,
    test_days: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (train_mask, val_mask, test_mask) for a contiguous three-way
    time split of `timestamps` (sorted, ascending).

    train  : [t_min, t_max - (val_days + test_days)]
    val    : (t_max - (val_days + test_days), t_max - test_days]
    test   : (t_max - test_days, t_max]

    The three masks are mutually exclusive and jointly exhaustive (no gaps,
    no overlap), and test is always the most recent window.
    """
    t_max = timestamps.max()
    val_split_date = t_max - pd.Timedelta(days=val_days + test_days)
    test_split_date = t_max - pd.Timedelta(days=test_days)

    train_mask = timestamps <= val_split_date
    val_mask = (timestamps > val_split_date) & (timestamps <= test_split_date)
    test_mask = timestamps > test_split_date

    return train_mask, val_mask, test_mask
