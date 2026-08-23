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


def make_prophet_holdout_split(
    timestamps: pd.Series,
    horizon_hours: int,
    test_days: int,
) -> tuple[pd.Series, pd.Series]:
    """Returns (train_mask, test_mask) for Prophet's horizon-aware split.

    test  : (t_max - test_days, t_max]                    — fixed test window
    train : [t_min, test_start - horizon_hours]            — cutoff recedes with horizon

    Unlike make_holdout_splits (LightGBM's 3-way split), this leaves a gap
    of exactly `horizon_hours` between train and test, rather than carving
    out a validation window — Prophet has no early-stopping mechanism to
    protect, so the gap exists solely to prevent the model from training on
    data inside its own forecast lead time.

    Note this deliberately does **not** guarantee `train_mask | test_mask` covers
    every row (there's an intentional gap of unused rows between them, unlike
    `make_holdout_splits`'s exhaustive 3-way partition) — that gap is the point.
    """
    t_max = timestamps.max()
    test_start = t_max - pd.Timedelta(days=test_days)
    train_cutoff = test_start - pd.Timedelta(hours=horizon_hours)

    train_mask = timestamps <= train_cutoff
    test_mask = timestamps > test_start

    return train_mask, test_mask
