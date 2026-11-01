"""Point-in-time features for the 24h-ahead load forecast (spec 019, ADR-0008).

A forecast is issued at hour `issued_at` (τ, UTC) for the target hour
T = τ + FORECAST_HORIZON_HOURS. Every feature uses only what was known at τ:

- **Load** is used only for hours ≤ τ − LOAD_AVAILABILITY_LAG_HOURS (24h). ENTSO-E
  data arrives ~1.5h after the hour at the earliest, but the GitHub-scheduled
  ingestion has gaps of up to 7.5h (measured 2026-11-01). Requiring data at
  least 24h old costs 0.16 pp of backtest MAPE (3.78% → 3.95%) and leaves 22h of
  slack, so a late ingestion never forces the naive fallback.
- **Weather** is the Open-Meteo forecast for the hour, taken from the latest run
  initialised at least 48h before it (`*_previous_day2`). Only hours ≤ T are
  read, so every value comes from a run initialised ≤ τ − 24h, published long
  before τ. (`*_previous_day1` would use runs initialised up to τ itself, which
  are not published at τ.)
- **Calendar** features describe T in Europe/Budapest; they are known in advance.

Nothing is imputed from later values; a missing input stays NaN. The same
function builds training rows (every τ) and the serving row (τ = now), so
training and serving cannot diverge.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import holidays
import numpy as np
import pandas as pd

from src.config import FIXED_HOLIDAYS
from src.promotion import to_naive_utc

FORECAST_HORIZON_HOURS = 24
LOAD_AVAILABILITY_LAG_HOURS = 24
LOCAL_TZ = "Europe/Budapest"

# Open-Meteo Previous Runs API: hourly variable -> feature column.
WEATHER_FORECAST_VARIABLES: dict[str, str] = {
    "temperature_2m_previous_day2": "temperature_fc",
    "relative_humidity_2m_previous_day2": "humidity_fc",
    "cloud_cover_previous_day2": "cloud_cover_fc",
    "shortwave_radiation_previous_day2": "shortwave_fc",
}
WEATHER_COLUMNS = list(WEATHER_FORECAST_VARIABLES.values())

LOAD_FEATURES = [
    "load_t_minus_48h",  # load(T − 48h) = load(τ − 24h): the latest usable hour
    "load_t_minus_168h",  # load(T − 168h): the seasonal-naive forecast itself
    "load_t_minus_336h",
    "load_mean_24h",  # mean over (τ − 48h, τ − 24h]
    "load_mean_168h",  # mean over (τ − 192h, τ − 24h]
    "load_std_168h",
]
CALENDAR_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "is_day_before_holiday",
    "is_day_after_holiday",
]
WEATHER_FEATURES = WEATHER_COLUMNS + ["temperature_fc_mean_24h"]  # mean over (T − 24h, T]
FEATURES = LOAD_FEATURES + CALENDAR_FEATURES + WEATHER_FEATURES

# Baselines evaluated on the same rows as the model (spec 019 T3).
WEEKLY_NAIVE = "load_t_minus_168h"  # served fallback (ADR-0002)
DAILY_NAIVE = "load_t_minus_48h"  # same hour two days earlier: T − 24h is not known at τ


def _hourly_series(frame: pd.DataFrame, value_col: str) -> pd.Series:
    """`timestamp, value_col` rows as a naive-UTC series on a complete hourly grid."""
    s = pd.Series(
        frame[value_col].to_numpy(dtype=float),
        index=pd.DatetimeIndex([to_naive_utc(t) for t in frame["timestamp"]]),
    )
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if s.empty:
        return s
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))


def _at(series: pd.Series, times: pd.DatetimeIndex) -> np.ndarray:
    return series.reindex(times).to_numpy(dtype=float)


def _holiday_dates(years: range) -> set:
    dates = set(holidays.country_holidays("HU", years=years))
    for year in years:
        for month, days in FIXED_HOLIDAYS.items():
            dates.update(pd.Timestamp(year, month, d).date() for d in days)
    return dates


def calendar_features(targets: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar of each target hour in local time (known at issuance)."""
    local = targets.tz_localize("UTC").tz_convert(LOCAL_TZ)
    dates = local.normalize()
    years = range(local.year.min() - 1, local.year.max() + 2) if len(local) else range(0)
    hol = _holiday_dates(years)

    def is_hol(days: pd.DatetimeIndex) -> np.ndarray:
        return np.array([d.date() in hol for d in days])

    day = pd.Timedelta(days=1)
    return pd.DataFrame(
        {
            "hour": local.hour,
            "day_of_week": local.dayofweek,
            "month": local.month,
            "is_weekend": (local.dayofweek >= 5).astype(int),
            "is_holiday": is_hol(dates).astype(int),
            "is_day_before_holiday": is_hol(dates + day).astype(int),
            "is_day_after_holiday": is_hol(dates - day).astype(int),
        },
        index=targets,
    )


def build_features(
    load: pd.DataFrame,
    weather_forecast: pd.DataFrame,
    issued_at: Sequence[Any] | pd.DatetimeIndex,
    include_target: bool = True,
) -> pd.DataFrame:
    """One row per issuance time: features, the target hour, and (optionally) its actual.

    Args:
        load: `timestamp, value_mwh` hourly actual load (any span, gaps allowed).
        weather_forecast: `timestamp` (valid hour) plus the columns in WEATHER_COLUMNS,
            as delivered by the Previous Runs API `*_previous_day2` variables.
        issued_at: issuance hours τ (UTC).
        include_target: add `target_mwh` = load(T). Serving passes False.

    Returns:
        DataFrame indexed by `issued_at` with `target_timestamp`, FEATURES and,
        if requested, `target_mwh`.
    """
    issued = pd.DatetimeIndex([to_naive_utc(t).floor("h") for t in issued_at], name="issued_at")
    horizon = pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
    lag = pd.Timedelta(hours=LOAD_AVAILABILITY_LAG_HOURS)
    targets = issued + horizon
    latest = issued - lag
    h = pd.Timedelta(hours=1)

    values = _hourly_series(load, "value_mwh")
    # Trailing windows end at their own row, so reading them at τ − 24h uses
    # nothing later than τ − 24h.
    mean_24 = values.rolling(24, min_periods=12).mean()
    mean_168 = values.rolling(168, min_periods=72).mean()
    std_168 = values.rolling(168, min_periods=72).std()

    out = pd.DataFrame(index=issued)
    out["target_timestamp"] = targets
    out["load_t_minus_48h"] = _at(values, targets - 48 * h)
    out["load_t_minus_168h"] = _at(values, targets - 168 * h)
    out["load_t_minus_336h"] = _at(values, targets - 336 * h)
    out["load_mean_24h"] = _at(mean_24, latest)
    out["load_mean_168h"] = _at(mean_168, latest)
    out["load_std_168h"] = _at(std_168, latest)

    out = out.join(calendar_features(targets).set_axis(issued))

    for col in WEATHER_COLUMNS:
        series = (
            _hourly_series(weather_forecast, col)
            if col in weather_forecast and not weather_forecast.empty
            else pd.Series(dtype=float)
        )
        out[col] = _at(series, targets)
        if col == "temperature_fc":
            out["temperature_fc_mean_24h"] = _at(series.rolling(24, min_periods=12).mean(), targets)
    if "temperature_fc_mean_24h" not in out:
        out["temperature_fc_mean_24h"] = np.nan

    if include_target:
        out["target_mwh"] = _at(values, targets)
    return out[["target_timestamp", *FEATURES] + (["target_mwh"] if include_target else [])]


def weather_frame_from_api(hourly: dict[str, list]) -> pd.DataFrame:
    """Previous Runs API `hourly` JSON block -> `timestamp` + WEATHER_COLUMNS."""
    frame = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"], utc=True)})
    for api_name, col in WEATHER_FORECAST_VARIABLES.items():
        frame[col] = pd.to_numeric(pd.Series(hourly.get(api_name, [None] * len(frame))))
    return frame
