"""As-issued weather forecasts from the Open-Meteo Previous Runs API (spec 019 T2, ADR-0008).

Fetches the `*_previous_day2` variables (see src/pit_features.py) and writes one
JSON records file per fetch, which `01b_ingest_weather_forecast` merges into
`bronze_weather_forecast`. The hourly ingestion fetches a few days around now;
the backfill fetches history in yearly chunks.

    python -m src.weather_forecast --days-back 3 --days-ahead 2 --out /tmp/weather_forecast
    python -m src.weather_forecast --start 2024-01-01 --end 2026-11-02 --out /tmp/backfill/weather_forecast
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    HTTP_TIMEOUT_SECONDS,
    OPENMETEO_FORECAST_MODEL,
    OPENMETEO_LAT,
    OPENMETEO_LON,
    OPENMETEO_PREVIOUS_RUNS_URL,
)
from src.pit_features import WEATHER_FORECAST_VARIABLES, weather_frame_from_api

logger = logging.getLogger(__name__)


def fetch_weather_forecast(
    start: date, end: date, session: Any, fetched_at: datetime | None = None
) -> pd.DataFrame:
    """`*_previous_day2` forecasts for valid hours on [start, end] (inclusive days, UTC)."""
    params: dict[str, str | float] = {
        "latitude": OPENMETEO_LAT,
        "longitude": OPENMETEO_LON,
        "timezone": "UTC",
        "models": OPENMETEO_FORECAST_MODEL,
        "hourly": ",".join(WEATHER_FORECAST_VARIABLES),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    response = session.get(OPENMETEO_PREVIOUS_RUNS_URL, params=params, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    if "hourly" not in payload or "time" not in payload["hourly"]:
        raise ValueError(f"Unexpected Previous Runs API response: {str(payload)[:200]}")
    frame = weather_frame_from_api(payload["hourly"])
    frame["model"] = OPENMETEO_FORECAST_MODEL
    frame["fetched_at"] = pd.Timestamp(fetched_at or datetime.now(UTC))
    return frame


def latest_per_hour(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per valid hour: the most recent fetch wins.

    A forecast for an hour more than 48h ahead is provisional (its run is not
    initialised yet) and is replaced by later fetches; for any issuance 24h
    ahead the value is final, and every fetch carries only information that
    existed when it was made.
    """
    ordered = frame.sort_values(["timestamp", "fetched_at"])
    return ordered.drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def yearly_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Inclusive (first, last) day pairs covering [start, end), at most one calendar year each."""
    chunks = []
    first = start
    while first < end:
        last = min(date(first.year, 12, 31), end - timedelta(days=1))
        chunks.append((first, last))
        first = last + timedelta(days=1)
    return chunks


def to_records_json(frame: pd.DataFrame) -> str:
    return frame.to_json(orient="records", date_format="iso", date_unit="s")


def write_fetch(frame: pd.DataFrame, out_dir: Path, first: date, last: date) -> Path:
    """One file per fetch; the fetch time keeps hourly files unique."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp(frame["fetched_at"].iloc[0]).strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"weather_forecast_{first:%Y%m%d}_{last:%Y%m%d}_{stamp}.json"
    path.write_text(to_records_json(frame), encoding="utf-8")
    return path


def run(
    out_dir: Path,
    session: Any,
    start: date | None = None,
    end: date | None = None,
    days_back: int = 3,
    days_ahead: int = 2,
    today: date | None = None,
) -> list[Path]:
    """Backfill [start, end) in yearly chunks, or the window around `today`."""
    if start and end:
        windows = yearly_chunks(start, end)
    else:
        today = today or datetime.now(UTC).date()
        windows = [(today - timedelta(days=days_back), today + timedelta(days=days_ahead))]
    paths = []
    for first, last in windows:
        logger.info("Fetching weather forecasts %s -> %s", first, last)
        frame = fetch_weather_forecast(first, last, session)
        if frame.drop(columns=["timestamp", "model", "fetched_at"]).isna().all().all():
            raise ValueError(f"No weather forecast values for {first} -> {last}.")
        paths.append(write_fetch(frame, out_dir, first, last))
    return paths


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper
    from src.api_client import build_retry_session

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat, help="exclusive")
    parser.add_argument("--days-back", type=int, default=3)
    parser.add_argument("--days-ahead", type=int, default=2)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    paths = run(
        args.out, build_retry_session(), args.start, args.end, args.days_back, args.days_ahead
    )
    print(f"Weather forecast files written: {len(paths)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
