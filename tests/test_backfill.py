"""Backfill fetch: chunking, file naming and file contents (spec 014 T3)."""

import json
from datetime import UTC, date, datetime

import pandas as pd
import pytest

from src.backfill import (
    chunk_filename,
    chunk_range,
    chunk_within,
    load_chunks,
    run_backfill,
    to_records_json,
    weather_chunks,
)


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


def test_load_chunks_are_seven_days_and_end_exclusive():
    chunks = load_chunks(date(2024, 1, 1), date(2024, 1, 20))

    assert chunks == [
        (utc(2024, 1, 1), utc(2024, 1, 8)),
        (utc(2024, 1, 8), utc(2024, 1, 15)),
        (utc(2024, 1, 15), utc(2024, 1, 20)),
    ]


def test_full_range_needs_147_entsoe_requests():
    # Part A estimate for 2024-01-01 -> 2026-10-24.
    assert len(load_chunks(date(2024, 1, 1), date(2026, 10, 24))) == 147


def test_weather_chunks_split_at_months_with_inclusive_end_dates():
    chunks = weather_chunks(date(2024, 11, 15), date(2025, 2, 1))

    assert chunks == [
        (date(2024, 11, 15), date(2024, 11, 30)),
        (date(2024, 12, 1), date(2024, 12, 31)),
        (date(2025, 1, 1), date(2025, 1, 31)),
    ]


def test_full_range_weather_is_34_monthly_files():
    assert len(weather_chunks(date(2024, 1, 1), date(2026, 10, 24))) == 34


@pytest.mark.parametrize("chunker", [load_chunks, weather_chunks])
def test_chunkers_reject_empty_range(chunker):
    with pytest.raises(ValueError, match="must be after"):
        chunker(date(2024, 1, 2), date(2024, 1, 2))


def test_chunk_filename_is_unique_and_sortable():
    name = chunk_filename(utc(2024, 1, 1), utc(2024, 1, 8))

    assert name == "2024-01-01T00-00-00Z_2024-01-08T00-00-00Z.json"
    assert name < chunk_filename(utc(2024, 1, 8), utc(2024, 1, 15))


def test_chunk_range_inverts_chunk_filename():
    name = chunk_filename(utc(2024, 1, 29), utc(2024, 2, 3))

    assert chunk_range(name) == (
        pd.Timestamp("2024-01-29T00:00Z"),
        pd.Timestamp("2024-02-03T00:00Z"),
    )


def test_only_chunks_inside_the_range_count_as_loaded():
    start, end = pd.Timestamp("2024-01-01T00:00Z"), pd.Timestamp("2024-02-01T00:00Z")

    assert chunk_within(chunk_filename(utc(2024, 1, 1), utc(2024, 1, 8)), start, end)
    assert chunk_within(chunk_filename(utc(2024, 1, 29), utc(2024, 2, 1)), start, end)
    assert not chunk_within(chunk_filename(utc(2024, 1, 29), utc(2024, 2, 5)), start, end)
    assert not chunk_within(chunk_filename(utc(2023, 12, 25), utc(2024, 1, 1)), start, end)


def test_records_json_uses_the_hourly_file_layout():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01T00:00Z"]),
            "value_mwh": [None],
            "fetched_at": pd.to_datetime(["2026-10-25T08:30:00Z"]),
        }
    )

    assert json.loads(to_records_json(df)) == [
        {
            "timestamp": "2024-01-01T00:00:00+00:00",
            "value_mwh": None,
            "fetched_at": "2026-10-25T08:30:00+00:00",
        }
    ]


class _FakeLoadClient:
    def __init__(self, empty_from=None):
        self.calls = []
        self.empty_from = empty_from

    def fetch_actual_load(self, start, end):
        self.calls.append((start, end))
        if self.empty_from is not None and start >= self.empty_from:
            return pd.DataFrame(columns=["timestamp", "value_mwh"])
        # One row past the chunk end, as an over-long Period would return.
        ts = pd.date_range(start, end, freq="h")
        return pd.DataFrame(
            {
                "timestamp": ts,
                "value_mwh": 5000.0,
                "country": "10YHU-MAVIR----U",
                "source": "entsoe_api",
                "fetched_at": pd.Timestamp("2026-10-25T00:00Z"),
            }
        )


class _FakeWeatherClient:
    def __init__(self):
        self.calls = []

    def fetch_weather(self, start_date, end_date):
        self.calls.append((start_date, end_date))
        ts = pd.date_range(pd.Timestamp(start_date, tz="UTC"), periods=24, freq="h")
        return pd.DataFrame(
            {"timestamp": ts, "temperature_c": 1.0, "fetched_at": pd.Timestamp("2026-10-25T00:00Z")}
        )


def test_run_backfill_writes_one_file_per_chunk(tmp_path):
    load, weather = _FakeLoadClient(), _FakeWeatherClient()

    counts = run_backfill(date(2024, 1, 1), date(2024, 2, 3), tmp_path, load, weather)

    assert counts == {"load_files": 5, "temperature_files": 2}
    assert weather.calls == [
        (date(2024, 1, 1), date(2024, 1, 31)),
        (date(2024, 2, 1), date(2024, 2, 2)),
    ]
    load_files = sorted(p.name for p in (tmp_path / "load").iterdir())
    assert load_files[0] == "2024-01-01T00-00-00Z_2024-01-08T00-00-00Z.json"
    assert load_files[-1] == "2024-01-29T00-00-00Z_2024-02-03T00-00-00Z.json"
    assert sorted(p.name for p in (tmp_path / "temperature").iterdir()) == [
        "2024-01-01T00-00-00Z_2024-02-01T00-00-00Z.json",
        "2024-02-01T00-00-00Z_2024-02-03T00-00-00Z.json",
    ]
    first = json.loads((tmp_path / "load" / load_files[0]).read_text(encoding="utf-8"))
    # Rows outside the chunk are dropped: exactly 7 x 24 hours.
    assert len(first) == 168
    assert first[-1]["timestamp"] == "2024-01-07T23:00:00+00:00"


def test_run_backfill_fails_on_an_empty_chunk(tmp_path):
    load = _FakeLoadClient(empty_from=utc(2024, 1, 8))

    with pytest.raises(ValueError, match="no load for 2024-01-08"):
        run_backfill(date(2024, 1, 1), date(2024, 1, 15), tmp_path, load, _FakeWeatherClient())
