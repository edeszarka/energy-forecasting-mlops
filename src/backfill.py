"""Backfill fetch: historical load and weather in large chunks (spec 014 T3, ADR-0005).

Run by `.github/workflows/backfill.yml` (Databricks has no outbound internet):

    python -m src.backfill --start 2024-01-01 --end 2026-10-24 --out /tmp/backfill

`--end` is exclusive. Writes one JSON file per chunk to `<out>/load/` (ENTSO-E,
7-day chunks) and `<out>/temperature/` (Open-Meteo Historical Forecast API,
calendar-month chunks), in the same record layout as the hourly files, for
`notebooks/00_backfill_bronze.py`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Protocol

import pandas as pd

from src.config import ENTSO_E_MAX_RANGE_DAYS, OPENMETEO_HISTORICAL_FORECAST_URL

logger = logging.getLogger(__name__)

_ISO_FORMAT = "%Y-%m-%dT%H:%M:%S+00:00"


class LoadClient(Protocol):
    def fetch_actual_load(self, start: datetime, end: datetime) -> pd.DataFrame: ...


class WeatherClient(Protocol):
    def fetch_weather(self, start_date: date, end_date: date) -> pd.DataFrame: ...


def _utc_midnight(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=UTC)


def load_chunks(
    start: date, end: date, max_days: int = ENTSO_E_MAX_RANGE_DAYS
) -> list[tuple[datetime, datetime]]:
    """[start, end) as UTC-midnight windows of at most `max_days` (ENTSO-E limit)."""
    if end <= start:
        raise ValueError(f"end {end} must be after start {start}.")
    chunks = []
    current = _utc_midnight(start)
    stop = _utc_midnight(end)
    while current < stop:
        chunk_end = min(current + timedelta(days=max_days), stop)
        chunks.append((current, chunk_end))
        current = chunk_end
    return chunks


def weather_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """[start, end) split at calendar months, as inclusive (first_day, last_day) pairs
    because Open-Meteo's start_date/end_date are inclusive."""
    if end <= start:
        raise ValueError(f"end {end} must be after start {start}.")
    chunks = []
    current = start
    while current < end:
        next_month = date(current.year + current.month // 12, current.month % 12 + 1, 1)
        chunk_end = min(next_month, end)
        chunks.append((current, chunk_end - timedelta(days=1)))
        current = chunk_end
    return chunks


def chunk_filename(start: datetime, end: datetime) -> str:
    """`<start>_<end>.json`, UTC, end exclusive; unique per chunk and sortable."""
    return f"{start:%Y-%m-%dT%H-%M-%SZ}_{end:%Y-%m-%dT%H-%M-%SZ}.json"


def chunk_range(filename: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Inverse of chunk_filename: the (start, end) a chunk file covers."""
    start, end = filename.removesuffix(".json").split("_")
    fmt = "%Y-%m-%dT%H-%M-%SZ"
    return (
        pd.Timestamp(datetime.strptime(start, fmt), tz="UTC"),
        pd.Timestamp(datetime.strptime(end, fmt), tz="UTC"),
    )


def chunk_within(filename: str, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """True if the chunk lies entirely in [start, end), i.e. it was fully loaded."""
    chunk_start, chunk_end = chunk_range(filename)
    return start <= chunk_start and chunk_end <= end


def to_records_json(df: pd.DataFrame) -> str:
    """Records JSON with ISO UTC timestamps, as the hourly workflow writes them."""
    out = df.copy()
    for col in ("timestamp", "fetched_at"):
        out[col] = pd.to_datetime(out[col], utc=True).dt.strftime(_ISO_FORMAT)
    return str(out.to_json(orient="records"))


def _within(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return df[(ts >= pd.Timestamp(start)) & (ts < pd.Timestamp(end))]


def run_backfill(
    start: date,
    end: date,
    out_dir: Path,
    load_client: LoadClient,
    weather_client: WeatherClient,
) -> dict[str, int]:
    """Fetches [start, end) and writes one file per chunk; returns files written.

    An empty chunk raises: a missing file would make 00_backfill_bronze replace
    that range with nothing.
    """
    load_dir = out_dir / "load"
    temp_dir = out_dir / "temperature"
    load_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    load_files = 0
    for chunk_start, chunk_end in load_chunks(start, end):
        logger.info("Fetching load %s -> %s", chunk_start, chunk_end)
        df = _within(load_client.fetch_actual_load(chunk_start, chunk_end), chunk_start, chunk_end)
        if df.empty:
            raise ValueError(f"ENTSO-E returned no load for {chunk_start} -> {chunk_end}.")
        (load_dir / chunk_filename(chunk_start, chunk_end)).write_text(
            to_records_json(df), encoding="utf-8"
        )
        load_files += 1

    temp_files = 0
    for first_day, last_day in weather_chunks(start, end):
        chunk_start = _utc_midnight(first_day)
        chunk_end = _utc_midnight(last_day + timedelta(days=1))
        logger.info("Fetching weather %s -> %s", first_day, last_day)
        df = _within(weather_client.fetch_weather(first_day, last_day), chunk_start, chunk_end)
        if df.empty:
            raise ValueError(f"Open-Meteo returned no weather for {first_day} -> {last_day}.")
        (temp_dir / chunk_filename(chunk_start, chunk_end)).write_text(
            to_records_json(df), encoding="utf-8"
        )
        temp_files += 1

    return {"load_files": load_files, "temperature_files": temp_files}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper
    from src.api_client import EntsoEClient, OpenMeteoClient

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat, help="exclusive")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    counts = run_backfill(
        args.start,
        args.end,
        args.out,
        EntsoEClient(),
        OpenMeteoClient(base_url=OPENMETEO_HISTORICAL_FORECAST_URL),
    )
    print(f"Backfill files written: {counts}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
