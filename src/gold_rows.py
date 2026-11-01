"""Gold forecast rows (spec 017, 019b) and the seasonal-naive history."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from src.baseline import seasonal_naive_forecast
from src.config import NAIVE_MODEL_NAME
from src.promotion import to_naive_utc

NAIVE_MODEL_VERSION = "seasonal_naive_t-168h"
# Served only when T - 168h is missing (spec 019b D8); name in src/config.py.
DAILY_NAIVE_MODEL_VERSION = "daily_naive_t-48h"
NAIVE_LAG_HOURS = 168


def gold_forecast_id(model_name: str, horizon_hours: int, target_ts: pd.Timestamp) -> str:
    """The gold idempotency key: MD5(model_name, horizon, target timestamp)."""
    return hashlib.md5(f"{model_name}_{horizon_hours}_{target_ts.isoformat()}".encode()).hexdigest()


# Gold columns in table order (spec 006, 017, 019b).
GOLD_COLUMNS = [
    "forecast_id",
    "timestamp",
    "forecast_run_at",
    "model_name",
    "model_version",
    "horizon_hours",
    "predicted_mwh",
    "actual_mwh",
    "is_backfilled",
    "pipeline_run_id",
    "created_at",
    "is_served",
    "issued_at",
    "is_catch_up",
    "fallback_reason",
]


def gold_row(
    *,
    model_name: str,
    model_version: str,
    target: pd.Timestamp,
    issued_at: pd.Timestamp,
    is_catch_up: bool,
    predicted_mwh: float,
    is_served: bool,
    forecast_run_at: Any,
    pipeline_run_id: str,
    created_at: datetime,
    fallback_reason: str | None = None,
    horizon_hours: int = 24,
) -> dict[str, Any]:
    """One gold row (spec 019b B1/B2).

    `issued_at` is the nominal issuance hour τ' (target − 24h); `created_at` is
    the actual write time (D5); `forecast_run_at` is the hour of the run that
    wrote it. `fallback_reason` is set only on a served naive row.
    """
    return {
        "forecast_id": gold_forecast_id(model_name, horizon_hours, target),
        "timestamp": target,
        "forecast_run_at": forecast_run_at,
        "model_name": model_name,
        "model_version": str(model_version),
        "horizon_hours": horizon_hours,
        "predicted_mwh": float(predicted_mwh),
        "actual_mwh": None,
        "is_backfilled": False,
        "pipeline_run_id": pipeline_run_id,
        "created_at": created_at,
        "is_served": is_served,
        "issued_at": issued_at,
        "is_catch_up": is_catch_up,
        "fallback_reason": fallback_reason,
    }


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
            # Legacy history (spec 019b D7): no nominal issuance, never served.
            "issued_at": None,
            "is_catch_up": None,
            "fallback_reason": None,
        }
    )
