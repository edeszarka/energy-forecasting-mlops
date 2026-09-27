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
    original_timestamps = set(df["timestamp"])

    df = df.set_index("timestamp").reindex(full_range)
    df["target"] = df["value_mwh"].shift(-horizon_hours)

    df = df.reset_index().rename(columns={"index": "timestamp"})
    df = df[df["timestamp"].isin(original_timestamps)]

    return df
