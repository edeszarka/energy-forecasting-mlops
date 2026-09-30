"""Silver window integrity helpers (spec 009).

Keeps feature rows written to ``silver_features`` inside the sub-window where
every row has full backward context for its lag/rolling features, so an
unconditional MERGE can never overwrite a previously-good value with a
context-starved NaN.
"""

from __future__ import annotations

import pandas as pd


def filter_to_core_window(
    feature_pd: pd.DataFrame,
    window_start: pd.Timestamp,
    max_lag_hours: int,
) -> pd.DataFrame:
    """Keeps only rows with timestamp >= window_start + max_lag_hours —
    the sub-window where every row has full backward context for its
    lag/rolling features within THIS window. Rows before that cutoff
    are margin, kept only so LATER rows in this same window have
    enough history; writing margin rows to silver_features would let
    a future run's MERGE (whenMatchedUpdateAll) silently overwrite a
    previously-good value with a context-starved NaN (spec 009 §1.1).
    """
    cutoff = window_start + pd.Timedelta(hours=max_lag_hours)
    return feature_pd[feature_pd["timestamp"] >= cutoff].reset_index(drop=True)
