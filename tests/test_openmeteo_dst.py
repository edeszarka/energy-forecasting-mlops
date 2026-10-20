"""
Regression tests for OpenMeteo timestamps across DST transitions (spec 012).

Requesting Europe/Budapest local times made tz_localize raise AmbiguousTimeError
on the autumn fall-back hour (02:00 occurs twice), failing every weather fetch
whose window touched that day.
"""

from datetime import date
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest
import responses

from src.api_client import OpenMeteoClient
from src.config import OPENMETEO_BASE_URL


def _hourly_payload(start_utc: str, hours: int) -> dict:
    times = pd.date_range(start_utc, periods=hours, freq="h")
    return {
        "hourly": {
            # OpenMeteo returns naive ISO strings in the requested timezone (UTC)
            "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
            "temperature_2m": [10.0 + i for i in range(hours)],
            "relative_humidity_2m": [80.0] * hours,
            "cloud_cover": [50.0] * hours,
        }
    }


@pytest.mark.parametrize(
    ("label", "start_utc"),
    [
        # 2026-10-25 01:00 UTC is the second local 02:00 (fall-back)
        ("autumn_fall_back", "2026-10-24T22:00"),
        # 2026-03-29 01:00 UTC: local 02:00 does not exist (spring-forward)
        ("spring_forward", "2026-03-28T22:00"),
    ],
)
@responses.activate
def test_fetch_weather_survives_dst_transition(label, start_utc):
    responses.add(responses.GET, OPENMETEO_BASE_URL, json=_hourly_payload(start_utc, 6), status=200)

    df = OpenMeteoClient().fetch_weather(date(2026, 1, 1), date(2026, 1, 1))

    expected = pd.date_range(start_utc, periods=6, freq="h", tz="UTC")
    assert list(df["timestamp"]) == list(expected), label
    assert df["timestamp"].is_unique
    assert df["temperature_c"].tolist() == [10.0 + i for i in range(6)]


@responses.activate
def test_fetch_weather_requests_utc():
    responses.add(
        responses.GET, OPENMETEO_BASE_URL, json=_hourly_payload("2026-10-20T00:00", 2), status=200
    )

    OpenMeteoClient().fetch_weather(date(2026, 10, 20), date(2026, 10, 20))

    query = parse_qs(urlparse(responses.calls[0].request.url).query)
    assert query["timezone"] == ["UTC"]
