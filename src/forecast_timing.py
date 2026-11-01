"""Which hour a serving run forecasts (spec 019, ADR-0008).

A run at issuance hour τ forecasts exactly one hour, T = τ + FORECAST_HORIZON_HOURS.
Every serving path — production LightGBM, shadow LightGBM, seasonal naive —
writes that hour, so all of them are compared on the same targets. (Before spec
019 a run wrote 24 hours, run+25h … run+48h, from features of future anchors
that were never observed at serving time.)
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import PROMOTABLE_MODELS
from src.pit_features import FORECAST_HORIZON_HOURS

# The one model family serving loads (ADR-0002).
SERVED_MODEL: str = PROMOTABLE_MODELS[0]


def issuance_hour(forecast_run_at: Any) -> pd.Timestamp:
    """The run's issuance hour τ (UTC, tz-aware)."""
    ts = pd.Timestamp(forecast_run_at)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.floor("h")


def serving_target_timestamps(
    forecast_run_at: Any, horizon_hours: int = FORECAST_HORIZON_HOURS
) -> list[pd.Timestamp]:
    """The single target hour of one serving run: [τ + horizon]."""
    return [issuance_hour(forecast_run_at) + pd.Timedelta(hours=horizon_hours)]
