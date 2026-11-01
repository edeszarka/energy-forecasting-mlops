"""Gold forecast rows: which row was served, and the seasonal-naive history (spec 017)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from src.baseline import seasonal_naive_forecast
from src.config import NAIVE_MODEL_NAME
from src.promotion import to_naive_utc

NAIVE_MODEL_VERSION = "seasonal_naive_t-168h"
NAIVE_LAG_HOURS = 168


def gold_forecast_id(model_name: str, horizon_hours: int, target_ts: pd.Timestamp) -> str:
    """The gold idempotency key: MD5(model_name, horizon, target timestamp)."""
    return hashlib.md5(f"{model_name}_{horizon_hours}_{target_ts.isoformat()}".encode()).hexdigest()


def combine_gold_rows(
    naive_df: pd.DataFrame, model_df: pd.DataFrame, model_served: bool
) -> pd.DataFrame:
    """Rows for one serving run with `is_served` set (ADR-0006).

    `model_df` holds LightGBM rows: the production run's (`model_served=True`) or
    the shadow run's (`model_served=False`). Naive rows are always written and are
    the served ones exactly when the model is not.
    """
    if model_served and model_df.empty:
        raise ValueError("model_served=True requires model rows.")
    frames = [
        df.assign(is_served=served)
        for df, served in ((model_df, model_served), (naive_df, not model_served))
        if not df.empty
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def reconstruct_naive_gold(
    silver: pd.DataFrame,
    start: Any,
    end: Any,
    horizon_hours: int,
    pipeline_run_id: str,
    created_at: datetime | None = None,
) -> pd.DataFrame:
    """Seasonal-naive gold rows for target hours in `[start, end)` from silver history.

    Exact reconstruction: predicted(T) = actual(T - 168h), issued by the first
    serving run that covers T (T - 2 * horizon_hours, the served-target
    convention), long after the source hour was observed. Rows are marked
    `is_backfilled = true` and `is_served = NULL`, so the monitor never counts
    them as served.
    """
    start_ts, end_ts = to_naive_utc(start), to_naive_utc(end)
    if end_ts <= start_ts:
        return pd.DataFrame()
    targets = list(pd.date_range(start_ts, end_ts, freq="h", inclusive="left").tz_localize(UTC))
    naive = seasonal_naive_forecast(silver, targets, lag_hours=NAIVE_LAG_HOURS)
    created_at = created_at or datetime.now(UTC)
    lead = pd.Timedelta(hours=2 * horizon_hours)
    return pd.DataFrame(
        {
            "forecast_id": [
                gold_forecast_id(NAIVE_MODEL_NAME, horizon_hours, t) for t in naive["timestamp"]
            ],
            "timestamp": naive["timestamp"],
            "forecast_run_at": [t - lead for t in naive["timestamp"]],
            "model_name": NAIVE_MODEL_NAME,
            "model_version": NAIVE_MODEL_VERSION,
            "horizon_hours": horizon_hours,
            "predicted_mwh": naive["predicted_mwh"].astype(float),
            "actual_mwh": None,
            "is_backfilled": True,
            "pipeline_run_id": pipeline_run_id,
            "created_at": created_at,
            "is_served": None,
        }
    )
