"""Maps an inference feature-anchor timestamp to the timestamp a prediction targets."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import PROMOTABLE_MODELS

# The one model family serving loads (ADR-0002).
SERVED_MODEL: str = PROMOTABLE_MODELS[0]


def resolve_target_timestamp(
    anchor_ts: pd.Timestamp,
    model_name: str,
    horizon_hours: int,
) -> pd.Timestamp:
    """Returns the timestamp a raw model prediction actually represents.

    LightGBM (energy_lgbm_*) is trained as a direct multi-step forecaster on
    target = value_mwh.shift(-horizon_hours): a feature row anchored at t
    predicts value(t + horizon_hours). Prophet (energy_prophet_*) has no such
    shift — predict(ds=t) already estimates value(t) directly. The mapping is
    therefore model-family-specific, dispatched the same way generate_forecasts()
    already dispatches for the prediction call itself ("lgbm" in model_name).
    """
    if "lgbm" in model_name:
        return anchor_ts + pd.Timedelta(hours=horizon_hours)
    return anchor_ts  # Prophet: predict(ds=t) already targets t


def serving_anchor_timestamps(forecast_run_at: Any, horizon_hours: int) -> list[pd.Timestamp]:
    """Feature anchors for one serving run: run+1h … run+horizon_hours."""
    start = pd.Timestamp(forecast_run_at).floor("h")
    return [start + pd.Timedelta(hours=i) for i in range(1, horizon_hours + 1)]


def serving_target_timestamps(forecast_run_at: Any, horizon_hours: int) -> list[pd.Timestamp]:
    """Target hours for one serving run, by the served LightGBM convention
    (anchors run+1…run+h → targets run+h+1…run+2h). Every serving path —
    LightGBM or the seasonal-naive fallback — writes exactly these hours, so
    the fallback must never re-derive them from its own model name (which
    would not shift).
    """
    return [
        resolve_target_timestamp(anchor, SERVED_MODEL, horizon_hours)
        for anchor in serving_anchor_timestamps(forecast_run_at, horizon_hours)
    ]
