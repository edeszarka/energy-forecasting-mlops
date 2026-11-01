"""Previous Runs API fetch and files (spec 019 T2)."""

import json
from datetime import UTC, date, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from src.pit_features import WEATHER_COLUMNS, WEATHER_FORECAST_VARIABLES
from src.weather_forecast import fetch_weather_forecast, latest_per_hour, run, yearly_chunks

FETCHED = datetime(2026, 11, 1, 12, 5, tzinfo=UTC)


class FakeSession:
    def __init__(self, hourly=None, status=200):
        self.calls = []
        self.hourly = hourly
        self.status = status

    def get(self, url, params, timeout):
        self.calls.append((url, params))
        hourly = self.hourly or {
            "time": ["2026-11-02T00:00", "2026-11-02T01:00"],
            **{name: [1.0, 2.0] for name in WEATHER_FORECAST_VARIABLES},
        }

        def raise_for_status():
            if self.status != 200:
                raise RuntimeError(f"HTTP {self.status}")

        return SimpleNamespace(raise_for_status=raise_for_status, json=lambda: {"hourly": hourly})


def test_requests_previous_day2_from_the_fixed_model():
    session = FakeSession()

    frame = fetch_weather_forecast(date(2026, 11, 2), date(2026, 11, 3), session, FETCHED)

    url, params = session.calls[0]
    assert "previous-runs-api" in url
    assert params["models"] == "icon_seamless"
    assert all(v.endswith("_previous_day2") for v in params["hourly"].split(","))
    assert (params["start_date"], params["end_date"], params["timezone"]) == (
        "2026-11-02",
        "2026-11-03",
        "UTC",
    )
    assert list(frame.columns) == ["timestamp", *WEATHER_COLUMNS, "model", "fetched_at"]
    assert frame["fetched_at"].iloc[0] == pd.Timestamp(FETCHED)


def test_unexpected_response_raises():
    session = FakeSession()
    session.get = lambda url, params, timeout: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: {"error": True}
    )

    with pytest.raises(ValueError, match="Unexpected"):
        fetch_weather_forecast(date(2026, 11, 2), date(2026, 11, 2), session)


def test_yearly_chunks_cover_the_range_exactly():
    assert yearly_chunks(date(2024, 1, 1), date(2026, 3, 1)) == [
        (date(2024, 1, 1), date(2024, 12, 31)),
        (date(2025, 1, 1), date(2025, 12, 31)),
        (date(2026, 1, 1), date(2026, 2, 28)),
    ]
    assert yearly_chunks(date(2026, 1, 1), date(2026, 1, 1)) == []


def test_hourly_run_fetches_the_window_around_today(tmp_path):
    session = FakeSession()

    paths = run(tmp_path, session, today=date(2026, 11, 1), days_back=3, days_ahead=2)

    _, params = session.calls[0]
    assert (params["start_date"], params["end_date"]) == ("2026-10-29", "2026-11-03")
    records = json.loads(paths[0].read_text())
    assert records[0]["temperature_fc"] == 1.0
    assert paths[0].name.startswith("weather_forecast_20261029_20261103_")


def test_backfill_writes_one_file_per_year(tmp_path):
    paths = run(tmp_path, FakeSession(), start=date(2024, 6, 1), end=date(2025, 2, 1))

    assert len(paths) == 2


def test_all_empty_values_fail_loudly(tmp_path):
    hourly = {"time": ["2026-11-02T00:00"], **{name: [None] for name in WEATHER_FORECAST_VARIABLES}}

    with pytest.raises(ValueError, match="No weather forecast values"):
        run(tmp_path, FakeSession(hourly=hourly), today=date(2026, 11, 1))


def test_latest_fetch_wins_per_hour():
    hour = pd.Timestamp("2026-11-03 10:00", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": [hour, hour, hour + pd.Timedelta(hours=1)],
            "temperature_fc": [5.0, 6.0, 7.0],
            "fetched_at": pd.to_datetime(
                ["2026-11-01 08:00", "2026-11-01 09:00", "2026-11-01 08:00"], utc=True
            ),
        }
    )

    out = latest_per_hour(frame.iloc[[1, 0, 2]])

    assert list(out["temperature_fc"]) == [6.0, 7.0]
