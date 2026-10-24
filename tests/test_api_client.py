"""
Unit tests for API clients.

Uses pytest, responses for HTTP mocking, and freezegun for time mocking.
"""

import os
from datetime import date, datetime, timedelta

import pandas as pd
import pytest
import responses
from freezegun import freeze_time

from src.api_client import EntsoEClient, EntsoEParseError, OpenMeteoClient, fetch_all
from src.config import ENTSO_E_BASE_URL, ENV_ENTSO_E_API_KEY, OPENMETEO_BASE_URL

# MOCK DATA
VALID_ENTSOE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Acknowledgement_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-1:acknowledgementdocument:7:0">
    <TimeSeries>
        <quantity_Measure_Unit.name>MAW</quantity_Measure_Unit.name>
        <curveType>A01</curveType>
        <Period>
            <timeInterval>
                <start>2024-01-01T00:00Z</start>
                <end>2024-01-01T04:00Z</end>
            </timeInterval>
            <resolution>PT60M</resolution>
            <Point>
                <position>1</position>
                <quantity>5000</quantity>
            </Point>
            <Point>
                <position>2</position>
                <quantity>5100</quantity>
            </Point>
            <Point>
                <position>4</position>
                <quantity>5300</quantity>
            </Point>
        </Period>
    </TimeSeries>
</Acknowledgement_MarketDocument>
"""


def _entsoe_xml(resolution: str, start: str, end: str, points: dict[int, float]) -> str:
    point_xml = "".join(
        f"<Point><position>{pos}</position><quantity>{qty}</quantity></Point>"
        for pos, qty in points.items()
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<GL_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0">
    <TimeSeries>
        <quantity_Measure_Unit.name>MAW</quantity_Measure_Unit.name>
        <curveType>A01</curveType>
        <Period>
            <timeInterval>
                <start>{start}</start>
                <end>{end}</end>
            </timeInterval>
            <resolution>{resolution}</resolution>
            {point_xml}
        </Period>
    </TimeSeries>
</GL_MarketDocument>
"""


# 00:00 has all four quarters (mean 4150); 01:00 is missing position 8 (01:45).
PT15M_ENTSOE_XML = _entsoe_xml(
    "PT15M",
    "2024-01-01T00:00Z",
    "2024-01-01T02:00Z",
    {1: 4000, 2: 4100, 3: 4200, 4: 4300, 5: 5000, 6: 5000, 7: 5000},
)

PT30M_ENTSOE_XML = _entsoe_xml(
    "PT30M", "2024-01-01T00:00Z", "2024-01-01T01:00Z", {1: 4000, 2: 4100}
)

# Spec 014 T1 fixtures, trimmed from the T0 raw responses (reports/entsoe_raw/,
# fetched 2026-10-24). Header and series elements are copied verbatim; only the
# points are cut down. The raw responses never contained more than one Period,
# so the multi-period fixture splits a real Period in two.
_RAW_HEADER = """<?xml version="1.0" encoding="utf-8"?>
<GL_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0">
  <mRID>16ad6d16467540e4b6e74a8ed0dcab53</mRID>
  <revisionNumber>1</revisionNumber>
  <type>A65</type>
  <process.processType>A16</process.processType>
  <createdDateTime>2026-10-24T18:36:51Z</createdDateTime>
"""


def _raw_period(start: str, end: str, points: dict[int, str]) -> str:
    point_xml = "".join(
        f"<Point><position>{pos}</position><quantity>{qty}</quantity></Point>"
        for pos, qty in points.items()
    )
    return (
        f"<Period><timeInterval><start>{start}</start><end>{end}</end></timeInterval>"
        f"<resolution>PT15M</resolution>{point_xml}</Period>"
    )


def _raw_series(curve_type: str, *periods: str) -> str:
    return (
        "<TimeSeries><mRID>1</mRID><businessType>A04</businessType>"
        "<objectAggregation>A01</objectAggregation>"
        '<outBiddingZone_Domain.mRID codingScheme="A01">10YHU-MAVIR----U</outBiddingZone_Domain.mRID>'
        "<quantity_Measure_Unit.name>MAW</quantity_Measure_Unit.name>"
        f"<curveType>{curve_type}</curveType>{''.join(periods)}</TimeSeries>"
    )


def _raw_document(*series: str) -> str:
    return _RAW_HEADER + "".join(series) + "</GL_MarketDocument>"


# a_24h_utc_midnight.xml positions 1-8 (2026-10-20 12:00-14:00Z), split into two Periods.
RAW_A_POINTS = {
    1: "4667.031", 2: "4699.888", 3: "4724.869", 4: "4748.155",
    5: "4755.556", 6: "4811.226", 7: "4959.473", 8: "5044.017",
}  # fmt: skip
MULTI_PERIOD_XML = _raw_document(
    _raw_series(
        "A03",
        _raw_period(
            "2026-10-20T12:00Z", "2026-10-20T13:00Z", {p: RAW_A_POINTS[p] for p in range(1, 5)}
        ),
        _raw_period(
            "2026-10-20T13:00Z", "2026-10-20T14:00Z", {p - 4: RAW_A_POINTS[p] for p in range(5, 9)}
        ),
    )
)

# b_7day.xml positions 1-8 (2026-10-13 00:00-02:00Z). Positions 2 and 6-8 are
# dropped, as ENTSO-E does under A03 when a value repeats the previous one.
RAW_B_POINTS = {1: "4273.79", 3: "4249.994", 4: "4299.423", 5: "4243.186"}
A03_GAPS_XML = _raw_document(
    _raw_series("A03", _raw_period("2026-10-13T00:00Z", "2026-10-13T02:00Z", RAW_B_POINTS))
)
A01_GAPS_XML = A03_GAPS_XML.replace("<curveType>A03</curveType>", "<curveType>A01</curveType>")

# c_dst_2025-10-26.xml positions 97-104: 2025-10-26 00:00-02:00Z, across the
# local DST change (01:00Z = 03:00 CEST -> 02:00 CET).
RAW_C_POINTS = {
    1: "3793.96", 2: "3800.49", 3: "3742.61", 4: "3748.88",
    5: "3713.69", 6: "3696.2", 7: "3701.34", 8: "3715.95",
}  # fmt: skip
DST_XML = _raw_document(
    _raw_series("A03", _raw_period("2025-10-26T00:00Z", "2025-10-26T02:00Z", RAW_C_POINTS))
)

UNKNOWN_CURVE_XML = _raw_document(
    _raw_series(
        "A02",
        _raw_period(
            "2026-10-20T12:00Z", "2026-10-20T13:00Z", {p: RAW_A_POINTS[p] for p in range(1, 5)}
        ),
    )
)


def _mean(values: list[str]) -> float:
    return sum(float(v) for v in values) / len(values)


VALID_OPENMETEO_JSON = {
    "hourly": {
        "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
        "temperature_2m": [5.0, 4.5],
        "relative_humidity_2m": [60, 62],
        "cloud_cover": [20, 25],
    }
}


@pytest.fixture
def entsoe_client():
    os.environ[ENV_ENTSO_E_API_KEY] = "fake_key"
    return EntsoEClient()


@pytest.fixture
def openmeteo_client():
    return OpenMeteoClient()


@responses.activate
def test_fetch_actual_load_happy_path(entsoe_client):
    responses.add(
        responses.GET,
        ENTSO_E_BASE_URL,
        body=VALID_ENTSOE_XML,
        status=200,
        content_type="application/xml",
    )

    start = datetime(2024, 1, 1, 0, 0)
    end = datetime(2024, 1, 1, 4, 0)

    df = entsoe_client.fetch_actual_load(start, end)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4  # Positions 1, 2, 3 (gap), 4
    assert list(df.columns) == ["timestamp", "value_mwh", "country", "source", "fetched_at"]
    assert df.iloc[0]["value_mwh"] == 5000.0
    assert pd.isna(df.iloc[2]["value_mwh"])  # Position 3 is missing
    # PT60M positions map one per hour (unchanged by spec 013).
    assert list(df["timestamp"]) == list(
        pd.date_range("2024-01-01T00:00Z", periods=4, freq="h", tz="UTC")
    )


@responses.activate
def test_fetch_actual_load_pt15m_aggregates_to_hourly_means(entsoe_client):
    responses.add(responses.GET, ENTSO_E_BASE_URL, body=PT15M_ENTSOE_XML, status=200)

    df = entsoe_client.fetch_actual_load(datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 2))

    assert list(df["timestamp"]) == [
        pd.Timestamp("2024-01-01T00:00Z"),
        pd.Timestamp("2024-01-01T01:00Z"),
    ]
    assert df.iloc[0]["value_mwh"] == 4150.0


@responses.activate
def test_fetch_actual_load_pt15m_hour_with_three_quarters_is_nan(entsoe_client):
    responses.add(responses.GET, ENTSO_E_BASE_URL, body=PT15M_ENTSOE_XML, status=200)

    df = entsoe_client.fetch_actual_load(datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 2))

    # 01:00 has only 3 of 4 quarters: no partial mean.
    assert pd.isna(df.iloc[1]["value_mwh"])


@responses.activate
def test_fetch_actual_load_unsupported_resolution_raises(entsoe_client):
    responses.add(responses.GET, ENTSO_E_BASE_URL, body=PT30M_ENTSOE_XML, status=200)

    with pytest.raises(ValueError, match=r"PT30M.*2024-01-01T00:00:00\+00:00"):
        entsoe_client.fetch_actual_load(datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 1))


@responses.activate
def test_fetch_actual_load_reads_every_period(entsoe_client):
    responses.add(responses.GET, ENTSO_E_BASE_URL, body=MULTI_PERIOD_XML, status=200)

    df = entsoe_client.fetch_actual_load(datetime(2026, 10, 20, 12), datetime(2026, 10, 20, 14))

    assert list(df["timestamp"]) == [
        pd.Timestamp("2026-10-20T12:00Z"),
        pd.Timestamp("2026-10-20T13:00Z"),
    ]
    assert df.iloc[0]["value_mwh"] == pytest.approx(_mean([RAW_A_POINTS[p] for p in range(1, 5)]))
    assert df.iloc[1]["value_mwh"] == pytest.approx(_mean([RAW_A_POINTS[p] for p in range(5, 9)]))


def test_parse_xml_reads_every_timeseries(entsoe_client):
    second = _raw_series(
        "A03",
        _raw_period(
            "2026-10-20T13:00Z", "2026-10-20T14:00Z", {p - 4: RAW_A_POINTS[p] for p in range(5, 9)}
        ),
    )
    first = _raw_series(
        "A03",
        _raw_period(
            "2026-10-20T12:00Z", "2026-10-20T13:00Z", {p: RAW_A_POINTS[p] for p in range(1, 5)}
        ),
    )

    rows = entsoe_client._parse_xml(_raw_document(first, second))

    assert [r["timestamp"] for r in rows] == [
        pd.Timestamp("2026-10-20T12:00Z"),
        pd.Timestamp("2026-10-20T13:00Z"),
    ]


@responses.activate
def test_fetch_actual_load_a03_repeats_previous_value(entsoe_client):
    responses.add(responses.GET, ENTSO_E_BASE_URL, body=A03_GAPS_XML, status=200)

    df = entsoe_client.fetch_actual_load(datetime(2026, 10, 13, 0), datetime(2026, 10, 13, 2))

    # 00:00: position 2 repeats position 1. 01:00: positions 6-8 repeat position 5
    # up to the end of the period.
    assert df.iloc[0]["value_mwh"] == pytest.approx(
        _mean([RAW_B_POINTS[1], RAW_B_POINTS[1], RAW_B_POINTS[3], RAW_B_POINTS[4]])
    )
    assert df.iloc[1]["value_mwh"] == pytest.approx(float(RAW_B_POINTS[5]))


def test_parse_xml_a01_missing_quarter_is_null(entsoe_client):
    rows = entsoe_client._parse_xml(A01_GAPS_XML)

    assert [r["value_mwh"] for r in rows] == [None, None]


@responses.activate
def test_fetch_actual_load_dst_day_has_no_gap_or_duplicate(entsoe_client):
    responses.add(responses.GET, ENTSO_E_BASE_URL, body=DST_XML, status=200)

    df = entsoe_client.fetch_actual_load(datetime(2025, 10, 26, 0), datetime(2025, 10, 26, 2))

    # The API is UTC: the local DST change is invisible, 4 quarters per hour.
    assert list(df["timestamp"]) == [
        pd.Timestamp("2025-10-26T00:00Z"),
        pd.Timestamp("2025-10-26T01:00Z"),
    ]
    assert df.iloc[0]["value_mwh"] == pytest.approx(_mean([RAW_C_POINTS[p] for p in range(1, 5)]))
    assert df.iloc[1]["value_mwh"] == pytest.approx(_mean([RAW_C_POINTS[p] for p in range(5, 9)]))


@responses.activate
def test_fetch_actual_load_unknown_curve_type_raises(entsoe_client):
    responses.add(responses.GET, ENTSO_E_BASE_URL, body=UNKNOWN_CURVE_XML, status=200)

    with pytest.raises(ValueError, match="curveType A02"):
        entsoe_client.fetch_actual_load(datetime(2026, 10, 20, 12), datetime(2026, 10, 20, 13))


def test_missing_curve_type_raises(entsoe_client):
    """T1b: no curveType element is unexpected content, not an implicit A01."""
    xml = A03_GAPS_XML.replace("<curveType>A03</curveType>", "")

    with pytest.raises(ValueError, match="curveType None"):
        entsoe_client._parse_xml(xml)


@pytest.mark.parametrize("unit", ["MWH", "KWT"])
def test_unit_other_than_maw_raises(entsoe_client, unit):
    """T1b: only MAW (MW) is accepted; the hourly mean of MW equals MWh."""
    xml = A03_GAPS_XML.replace(
        "<quantity_Measure_Unit.name>MAW</quantity_Measure_Unit.name>",
        f"<quantity_Measure_Unit.name>{unit}</quantity_Measure_Unit.name>",
    )

    with pytest.raises(ValueError, match=f"quantity unit {unit}; expected MAW"):
        entsoe_client._parse_xml(xml)


def test_missing_unit_raises(entsoe_client):
    xml = A03_GAPS_XML.replace("<quantity_Measure_Unit.name>MAW</quantity_Measure_Unit.name>", "")

    with pytest.raises(ValueError, match="quantity unit None"):
        entsoe_client._parse_xml(xml)


def test_raw_fixtures_contain_no_secrets():
    for xml in (MULTI_PERIOD_XML, A03_GAPS_XML, DST_XML, UNKNOWN_CURVE_XML):
        assert "securityToken" not in xml


@responses.activate
def test_fetch_actual_load_empty_response(entsoe_client):
    responses.add(responses.GET, ENTSO_E_BASE_URL, body="", status=200)

    df = entsoe_client.fetch_actual_load(datetime(2024, 1, 1), datetime(2024, 1, 2))
    assert df.empty
    assert "timestamp" in df.columns


def test_fetch_actual_load_raises_on_range_exceeded(entsoe_client):
    start = datetime(2024, 1, 1)
    end = start + timedelta(days=8)
    with pytest.raises(ValueError, match="Range exceeds maximum"):
        entsoe_client.fetch_actual_load(start, end)


@responses.activate
def test_fetch_actual_load_hides_api_key_in_errors(entsoe_client):
    responses.add(responses.GET, ENTSO_E_BASE_URL, body="Invalid XML", status=200)

    with pytest.raises(EntsoEParseError) as exc:
        entsoe_client.fetch_actual_load(datetime(2024, 1, 1), datetime(2024, 1, 2))

    assert "fake_key" not in str(exc.value)


@responses.activate
def test_fetch_weather_happy_path(openmeteo_client):
    responses.add(responses.GET, OPENMETEO_BASE_URL, json=VALID_OPENMETEO_JSON, status=200)

    df = openmeteo_client.fetch_weather(date(2024, 1, 1), date(2024, 1, 1))

    assert len(df) == 2
    # Check if timezone is UTC
    assert str(df["timestamp"].dt.tz) in ["UTC", "datetime.timezone.utc", "UTC0"]
    assert df.iloc[0]["temperature_c"] == 5.0
    assert "humidity_pct" in df.columns
    assert "cloud_cover_pct" in df.columns


@responses.activate
def test_fetch_weather_uses_configured_base_url():
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    responses.add(responses.GET, url, json=VALID_OPENMETEO_JSON, status=200)

    df = OpenMeteoClient(base_url=url).fetch_weather(date(2024, 1, 1), date(2024, 1, 1))

    assert len(df) == 2
    assert responses.calls[0].request.url.startswith(url)


def test_fetch_weather_fallback_imputation(openmeteo_client):
    # Create DF with NaNs
    # Note: 'h' is the new standard for hourly frequency in pandas/Python 3.14
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC"),
            "temperature_c": [10.0, 11.0, None, 13.0, None, 15.0, 16.0, 17.0, 18.0, 19.0],
            "humidity_pct": [60.0] * 10,
            "cloud_cover_pct": [20.0] * 10,
        }
    )

    imputed_df = openmeteo_client._fallback_weather(df, ["temperature_c"])

    assert not imputed_df["temperature_c"].isna().any()
    assert imputed_df.loc[2, "is_weather_imputed"]
    assert not imputed_df.loc[0, "is_weather_imputed"]


def test_build_retry_session_mounts_https():
    from src.api_client import build_retry_session

    session = build_retry_session()
    assert "https://" in session.adapters
    assert "http://" in session.adapters


@responses.activate
@freeze_time("2024-01-01")
def test_fetch_all_wrapper():
    # Mock both
    responses.add(responses.GET, ENTSO_E_BASE_URL, body=VALID_ENTSOE_XML, status=200)
    responses.add(responses.GET, OPENMETEO_BASE_URL, json=VALID_OPENMETEO_JSON, status=200)

    os.environ[ENV_ENTSO_E_API_KEY] = "fake_key"
    load_df, temp_df = fetch_all(datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 4))

    assert not load_df.empty
    assert not temp_df.empty
