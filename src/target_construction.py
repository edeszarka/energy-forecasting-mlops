"""Time-aligned target construction for LightGBM's direct multi-step forecasting."""

from __future__ import annotations

import pandas as pd


def build_horizon_target(df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    """Adds a `target` column equal to value_mwh exactly `horizon_hours` HOURS
    ahead of each row — not `horizon_hours` ROWS ahead.

    pandas .shift(-n) is positional: it is only equivalent to an n-hour
    lookahead if the frame is ascending-sorted by timestamp and has no
    missing hourly rows in the shift window. Neither is guaranteed by a raw
    Spark table read. This mirrors src/features.py::add_lag_features()'s
    reindex-before-shift pattern: sort, reindex to a complete hourly grid,
    shift, then restore only the original rows. A missing hour anywhere in
    the shift window now correctly produces target = NaN (caught by the
    existing dropna(subset=["target"] + FEATURE_COLS)) instead of being
    silently paired with the wrong hour's value.
    """
    df = df.sort_values("timestamp").copy()

    tz = df["timestamp"].dt.tz
    full_range = pd.date_range(
        start=df["timestamp"].min(), end=df["timestamp"].max(), freq="h", tz=tz
    )

    # Reindex and shift ONLY the value series needed for the target.
    # Reindexing the WHOLE frame (the prior implementation) inserts an
    # all-NaN row for every gap hour across every column — including typed
    # columns like bool, which a plain NumPy array can't hold NaN in, so
    # they upcast to object. Filtering the gap rows back out afterward
    # does not revert that upcast (spec 010 §1.2). Mapping the target back
    # onto the original, never-reindexed rows touches no column but the
    # new `target` itself.
    value_by_ts = df.set_index("timestamp")["value_mwh"]
    target_by_ts = value_by_ts.reindex(full_range).shift(-horizon_hours)
    df["target"] = df["timestamp"].map(target_by_ts)

    # Deterministic RangeIndex (timestamp-sorted) so output is independent of
    # the input row order — the prior reindex path reset to a fresh index too,
    # and spec 007's order-independence guard asserts it (spec 010 §3.2).
    return df.reset_index(drop=True)
