"""Silver feature build shared by the hourly path and the rebuild (spec 014 T4).

`build_silver_frame` turns bronze load and weather frames into the silver rows
of one core window. Rows outside the core window are margin: they give the core
rows backward context (lags, rolling windows) and forward context (centred
weather fill, `is_holiday_eve`) but are never written (spec 009 §1.1).
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from src.config import LAG_HOURS, MIN_TRAINING_ROWS
from src.features import build_feature_matrix, get_feature_columns

REBUILD_CHUNK_DAYS: int = 30
REBUILD_LOOKBACK_HOURS: int = max(LAG_HOURS)
REBUILD_LOOKAHEAD_HOURS: int = 72


class RebuildChunk(NamedTuple):
    """Bronze is read on [read_start, read_end]; silver is written on [core_start, core_end)."""

    read_start: pd.Timestamp
    read_end: pd.Timestamp
    core_start: pd.Timestamp
    core_end: pd.Timestamp


def plan_rebuild_chunks(
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_days: int = REBUILD_CHUNK_DAYS,
    lookback_hours: int = REBUILD_LOOKBACK_HOURS,
    lookahead_hours: int = REBUILD_LOOKAHEAD_HOURS,
) -> list[RebuildChunk]:
    """Splits [start, end) into core chunks of `chunk_days` with read margins.

    The look-back is extended for a short last chunk so the read window still
    spans MIN_TRAINING_ROWS hours (build_feature_matrix's minimum). Extra
    look-back cannot change core rows: no feature looks further back than
    `lookback_hours`.
    """
    if end <= start:
        raise ValueError(f"Rebuild end {end} must be after start {start}.")

    chunks = []
    core_start = start
    while core_start < end:
        core_end = min(core_start + pd.Timedelta(days=chunk_days), end)
        read_start = min(
            core_start - pd.Timedelta(hours=lookback_hours),
            core_end - pd.Timedelta(hours=MIN_TRAINING_ROWS),
        )
        read_end = core_end + pd.Timedelta(hours=lookahead_hours)
        chunks.append(RebuildChunk(read_start, read_end, core_start, core_end))
        core_start = core_end
    return chunks


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")


def build_silver_frame(
    load_pd: pd.DataFrame,
    temp_pd: pd.DataFrame,
    core_start: pd.Timestamp,
    core_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Silver rows for timestamps in [core_start, core_end) (no upper bound if None).

    Args:
        load_pd: bronze_load rows covering the core window plus its margins.
        temp_pd: bronze_temperature rows for the same range.
        core_start: First timestamp that gets a silver row.
        core_end: Exclusive end of the core window.

    Raises:
        ValueError: Too few load rows, or a feature column is missing.
    """
    load = _prepare(load_pd)
    temp = _prepare(temp_pd) if not temp_pd.empty else temp_pd

    features = build_feature_matrix(load, temp)

    keep = features["timestamp"] >= core_start
    if core_end is not None:
        keep &= features["timestamp"] < core_end
    features = features[keep].reset_index(drop=True)

    missing = set(get_feature_columns()) - set(features.columns)
    if missing:
        raise ValueError(f"Missing feature columns in output: {missing}")
    return features
