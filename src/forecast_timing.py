"""Maps an inference feature-anchor timestamp to the timestamp a prediction targets."""

from __future__ import annotations

import pandas as pd


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
