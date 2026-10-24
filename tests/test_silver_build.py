"""Silver build shared by the hourly path and the rebuild (spec 014 T4, AC5)."""

import numpy as np
import pandas as pd
import pytest

from src.config import LAG_HOURS, MIN_TRAINING_ROWS
from src.silver_build import RebuildChunk, build_silver_frame, plan_rebuild_chunks

START = pd.Timestamp("2026-03-01T00:00Z")


def bronze(days: int = 80) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.date_range(START, periods=days * 24, freq="h")
    hours = np.arange(len(ts))
    load = pd.DataFrame(
        {
            "timestamp": ts,
            "value_mwh": 5000 + 1000 * np.sin(2 * np.pi * hours / 24) + hours % 7,
            "is_gap": False,
            "source": "entsoe_api",
            "fetched_at": ts,
        }
    )
    temp = pd.DataFrame(
        {
            "timestamp": ts,
            "temperature_c": 10 + 5 * np.cos(2 * np.pi * hours / 24),
            "humidity_pct": 60.0,
            "cloud_cover_pct": 20.0,
            "is_weather_imputed": False,
        }
    )
    return load, temp


def between(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """The notebook's bronze read: timestamp BETWEEN start AND end (inclusive)."""
    return df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]


def test_rebuild_equals_hourly_build_on_overlapping_hours():
    """AC5: the rebuild chunks and the hourly path produce identical silver rows."""
    load, temp = bronze()

    # Hourly path (02_transform Cells 4-9): 720h core plus a max_lag margin.
    run_date = START + pd.Timedelta(days=70)
    window_start = run_date - pd.Timedelta(hours=720 + max(LAG_HOURS))
    core_start = window_start + pd.Timedelta(hours=max(LAG_HOURS))
    hourly = build_silver_frame(
        between(load, window_start, run_date), between(temp, window_start, run_date), core_start
    )

    # Rebuild path, with chunk boundaries inside the hourly core window.
    chunks = plan_rebuild_chunks(START + pd.Timedelta(days=10), START + pd.Timedelta(days=78))
    rebuilt = pd.concat(
        [
            build_silver_frame(
                between(load, c.read_start, c.read_end),
                between(temp, c.read_start, c.read_end),
                c.core_start,
                c.core_end,
            )
            for c in chunks
        ],
        ignore_index=True,
    )

    # The hourly path has no look-ahead: its last hour cannot see the next day
    # (is_holiday_eve). Everything else must match exactly.
    overlap = hourly[hourly["timestamp"] < run_date]["timestamp"]
    assert len(overlap) == 720
    cols = [c for c in hourly.columns if c != "feature_built_at"]
    left = hourly.set_index("timestamp").loc[overlap, [c for c in cols if c != "timestamp"]]
    right = rebuilt.set_index("timestamp").loc[overlap, [c for c in cols if c != "timestamp"]]
    pd.testing.assert_frame_equal(left, right, check_dtype=False)


def test_build_silver_frame_keeps_only_the_core_window():
    load, temp = bronze(days=40)
    core_start = START + pd.Timedelta(days=10)
    core_end = START + pd.Timedelta(days=12)

    frame = build_silver_frame(load, temp, core_start, core_end)

    assert frame["timestamp"].min() == core_start
    assert frame["timestamp"].max() == core_end - pd.Timedelta(hours=1)
    assert len(frame) == 48


def test_build_silver_frame_deduplicates_bronze_rows():
    load, temp = bronze(days=40)
    duplicated = pd.concat([load, load.iloc[[500]]], ignore_index=True)

    frame = build_silver_frame(duplicated, temp, START + pd.Timedelta(days=10))

    assert frame["timestamp"].is_unique


def test_build_silver_frame_rejects_too_few_rows():
    load, temp = bronze(days=10)
    with pytest.raises(ValueError, match="Insufficient data"):
        build_silver_frame(load, temp, START)


def test_plan_rebuild_chunks_margins():
    chunks = plan_rebuild_chunks(
        pd.Timestamp("2024-01-01T00:00Z"), pd.Timestamp("2024-03-01T00:00Z")
    )

    assert [c.core_start for c in chunks] == [
        pd.Timestamp("2024-01-01T00:00Z"),
        pd.Timestamp("2024-01-31T00:00Z"),
    ]
    assert chunks[0] == RebuildChunk(
        read_start=pd.Timestamp("2023-12-25T00:00Z"),  # 168h look-back
        read_end=pd.Timestamp("2024-01-31T00:00Z") + pd.Timedelta(hours=72),
        core_start=pd.Timestamp("2024-01-01T00:00Z"),
        core_end=pd.Timestamp("2024-01-31T00:00Z"),
    )
    assert chunks[-1].core_end == pd.Timestamp("2024-03-01T00:00Z")


def test_plan_rebuild_chunks_extends_lookback_for_a_short_last_chunk():
    chunks = plan_rebuild_chunks(
        pd.Timestamp("2024-01-01T00:00Z"), pd.Timestamp("2024-02-03T00:00Z")
    )

    last = chunks[-1]
    assert last.core_end - last.core_start == pd.Timedelta(days=3)
    assert last.core_end - last.read_start == pd.Timedelta(hours=MIN_TRAINING_ROWS)


def test_plan_rebuild_chunks_rejects_empty_range():
    with pytest.raises(ValueError, match="must be after"):
        plan_rebuild_chunks(START, START)
