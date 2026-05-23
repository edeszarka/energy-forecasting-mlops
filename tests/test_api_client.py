"""
Unit tests for API clients.

Uses pytest, responses for HTTP mocking, and freezegun for time mocking.
"""

import os
from datetime import datetime, timedelta, date

import pandas as pd
import pytest
import responses
from freezegun import freeze_time

from src.api_client import EntsoEClient, OpenMeteoClient, EntsoEParseError, fetch_all
from src.config import ENV_ENTSO_E_API_KEY, ENTSO_E_BASE_URL, OPENMETEO_BASE_URL

# MOCK DATA
VALID_ENTSOE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Acknowledgement_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-1:acknowledgementdocument:7:0">
    <TimeSeries>
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

VALID_OPENMETEO_JSON = {
    "hourly": {
        "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
        "temperature_2m": [5.0, 4.5]
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
        content_type="application/xml"
    )
    
    start = datetime(2024, 1, 1, 0, 0)
    end = datetime(2024, 1, 1, 4, 0)
    
    df = entsoe_client.fetch_actual_load(start, end)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4  # Positions 1, 2, 3 (gap), 4
    assert list(df.columns) == ["timestamp", "value_mwh", "country", "source", "fetched_at"]
    assert df.iloc[0]["value_mwh"] == 5000.0
    assert pd.isna(df.iloc[2]["value_mwh"])  # Position 3 is missing

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
def test_fetch_temperature_happy_path(openmeteo_client):
    responses.add(
        responses.GET,
        OPENMETEO_BASE_URL,
        json=VALID_OPENMETEO_JSON,
        status=200
    )
    
    df = openmeteo_client.fetch_temperature(date(2024, 1, 1), date(2024, 1, 1))
    
    assert len(df) == 2
    assert df["timestamp"].dt.tz.zone == "UTC"
    assert df.iloc[0]["temperature_c"] == 5.0

def test_fetch_temperature_fallback_imputation(openmeteo_client):
    # Create DF with NaNs
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="H", tz="UTC"),
        "temperature_c": [10.0, 11.0, None, 13.0, None, 15.0, 16.0, 17.0, 18.0, 19.0]
    })
    
    imputed_df = openmeteo_client._fallback_temperature(df)
    
    assert not imputed_df["temperature_c"].isna().any()
    assert imputed_df.loc[2, "is_temp_imputed"] == True
    assert imputed_df.loc[0, "is_temp_imputed"] == False

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
