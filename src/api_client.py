"""
API clients for ENTSO-E and OpenMeteo.

This module provides production-grade HTTP clients for retrieving electricity
load data and weather forecasts. It includes retry logic, error handling,
and data parsing into pandas DataFrames.
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from src.config import (
    ENTSO_E_BASE_URL,
    ENTSO_E_DOC_TYPE,
    ENTSO_E_MAX_RANGE_DAYS,
    ENTSO_E_PROCESS_TYPE,
    ENTSO_E_ZONE,
    ENV_ENTSO_E_API_KEY,
    HTTP_BACKOFF_FACTOR,
    HTTP_MAX_RETRIES,
    HTTP_TIMEOUT_SECONDS,
    OPENMETEO_BASE_URL,
    OPENMETEO_LAT,
    OPENMETEO_LON,
    OPENMETEO_TIMEZONE,
)

logger = logging.getLogger(__name__)

class EntsoEParseError(Exception):
    """Raised when ENTSO-E XML response cannot be parsed."""
    def __init__(self, message: str, xml_text: str) -> None:
        # Avoid logging the securityToken if it happens to be in the XML (unlikely but safe)
        safe_xml = xml_text[:200].replace("securityToken=", "token=***")
        super().__init__(f"{message}. Preview: {safe_xml}")

class OpenMeteoParseError(Exception):
    """Raised when OpenMeteo JSON response cannot be parsed."""
    def __init__(self, message: str, json_text: str) -> None:
        super().__init__(f"{message}. Preview: {json_text[:200]}")

def build_retry_session() -> requests.Session:
    """Creates a requests session with exponential backoff retries."""
    session = requests.Session()
    retry_strategy = Retry(
        total=HTTP_MAX_RETRIES,
        backoff_factor=HTTP_BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

class EntsoEClient:
    """Client for the ENTSO-E Transparency Platform API."""

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initializes the client.
        
        Args:
            api_key: Optional API key. If not provided, reads from environment.
            
        Raises:
            ValueError: If no API key is found.
        """
        self.api_key = api_key or os.environ.get(ENV_ENTSO_E_API_KEY)
        if not self.api_key:
            raise ValueError(f"ENTSO-E API key must be provided via param or {ENV_ENTSO_E_API_KEY} env var.")
        
        self.session = build_retry_session()

    def fetch_actual_load(
        self,
        start: datetime,
        end: datetime,
        zone: str = ENTSO_E_ZONE,
    ) -> pd.DataFrame:
        """
        Fetch actual total load for the given UTC time range.
        
        Args:
            start: Start datetime (UTC).
            end: End datetime (UTC).
            zone: Bidding zone domain code.
            
        Returns:
            DataFrame with columns: timestamp, value_mwh, country, source, fetched_at.
            
        Raises:
            ValueError: On invalid range.
            requests.HTTPError: On API failure.
            EntsoEParseError: On parsing failure.
        """
        if end <= start:
            raise ValueError("End time must be after start time.")
        
        if (end - start).days > ENTSO_E_MAX_RANGE_DAYS:
            raise ValueError(f"Range exceeds maximum of {ENTSO_E_MAX_RANGE_DAYS} days.")

        # Format: YYYYMMDDHHmm
        fmt = "%Y%m%d%H%M"
        params = {
            "documentType": ENTSO_E_DOC_TYPE,
            "processType": ENTSO_E_PROCESS_TYPE,
            "outBiddingZone_Domain": zone,
            "periodStart": start.strftime(fmt),
            "periodEnd": end.strftime(fmt),
            "securityToken": self.api_key
        }

        logger.info("Fetching ENTSO-E load from %s to %s", start, end)
        response = self.session.get(ENTSO_E_BASE_URL, params=params, timeout=HTTP_TIMEOUT_SECONDS)
        
        # Guard against secrets in logs/errors
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            msg = str(e).replace(self.api_key, "***")
            raise requests.HTTPError(msg, response=response) from e

        if not response.text.strip():
            logger.warning("ENTSO-E returned empty response body.")
            return self._empty_load_df()

        data = self._parse_xml(response.text)
        if not data:
            return self._empty_load_df()

        df = pd.DataFrame(data)
        df["country"] = zone
        df["source"] = "entsoe_api"
        df["fetched_at"] = pd.Timestamp.now(tz="UTC")
        
        # Deduplicate and sort
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        
        logger.info("Successfully fetched %d load records", len(df))
        return df

    def _parse_xml(self, xml_text: str) -> list[dict[str, Any]]:
        """Parses ENTSO-E XML response into a list of dictionaries."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            raise EntsoEParseError("Invalid XML", xml_text) from e

        # Namespaces are common in ENTSO-E XML
        namespace = ""
        if "}" in root.tag:
            namespace = root.tag.split("}")[0] + "}"

        rows = []
        for ts in root.findall(f".//{namespace}TimeSeries"):
            period = ts.find(f"{namespace}Period")
            if period is None:
                continue
            
            start_str = period.find(f"{namespace}timeInterval/{namespace}start").text # type: ignore
            period_start = pd.to_datetime(start_str).tz_convert("UTC")
            
            # Resolution is typically PT60M for hourly
            resolution = period.find(f"{namespace}resolution").text # type: ignore
            if resolution != "PT60M":
                logger.warning("Unexpected resolution: %s. Expected PT60M.", resolution)

            points = {int(p.find(f"{namespace}position").text): float(p.find(f"{namespace}quantity").text) # type: ignore
                      for p in period.findall(f"{namespace}Point")}
            
            if not points:
                continue

            # Fill gaps by iterating through expected positions
            max_pos = max(points.keys())
            for pos in range(1, max_pos + 1):
                val = points.get(pos)
                rows.append({
                    "timestamp": period_start + timedelta(hours=pos - 1),
                    "value_mwh": val
                })

        return rows

    def _empty_load_df(self) -> pd.DataFrame:
        """Returns an empty DataFrame with the expected schema."""
        return pd.DataFrame(columns=["timestamp", "value_mwh", "country", "source", "fetched_at"])

class OpenMeteoClient:
    """Client for the OpenMeteo Forecast API."""

    def __init__(self) -> None:
        """Initializes the client with a retry session."""
        self.session = build_retry_session()

    def fetch_temperature(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Fetch hourly temperature for Budapest.
        
        Args:
            start_date: Start date.
            end_date: End date.
            
        Returns:
            DataFrame with columns: timestamp, temperature_c, source, fetched_at.
        """
        params = {
            "latitude": OPENMETEO_LAT,
            "longitude": OPENMETEO_LON,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": "temperature_2m",
            "timezone": OPENMETEO_TIMEZONE
        }

        logger.info("Fetching OpenMeteo temperature from %s to %s", start_date, end_date)
        response = self.session.get(OPENMETEO_BASE_URL, params=params, timeout=HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()

        data = response.json()
        if "hourly" not in data or "temperature_2m" not in data["hourly"]:
            raise OpenMeteoParseError("Unexpected JSON schema", response.text)

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(data["hourly"]["time"]),
            "temperature_c": data["hourly"]["temperature_2m"]
        })

        # OpenMeteo returns timestamps in the requested timezone. Convert to UTC.
        df["timestamp"] = df["timestamp"].dt.tz_localize(OPENMETEO_TIMEZONE).dt.tz_convert("UTC")
        
        df["source"] = "openmeteo"
        df["fetched_at"] = pd.Timestamp.now(tz="UTC")

        # Impute if needed
        if df["temperature_c"].isna().any():
            df = self._fallback_temperature(df)
        else:
            df["is_temp_imputed"] = False

        logger.info("Successfully fetched %d temperature records", len(df))
        return df

    def _fallback_temperature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills NaN temperatures with a 72-hour rolling mean."""
        logger.warning("Detected NaNs in temperature data. Applying rolling mean imputation.")
        df["is_temp_imputed"] = df["temperature_c"].isna()
        # 72-hour rolling window, minimum 1 period to fill as much as possible
        df["temperature_c"] = df["temperature_c"].fillna(
            df["temperature_c"].rolling(window=72, min_periods=1, center=True).mean()
        )
        return df

def fetch_all(
    start: datetime,
    end: datetime,
    api_key: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper to fetch both load and temperature.
    
    Args:
        start: Start datetime (UTC).
        end: End datetime (UTC).
        api_key: ENTSO-E API key.
        
    Returns:
        Tuple of (load_df, temperature_df).
    """
    entsoe = EntsoEClient(api_key=api_key)
    openmeteo = OpenMeteoClient()

    load_df = entsoe.fetch_actual_load(start, end)
    temp_df = openmeteo.fetch_temperature(start.date(), end.date())

    return load_df, temp_df
