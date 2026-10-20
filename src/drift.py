"""
Drift detection and model quality monitoring logic.

This module provides functions to extract results from monitoring reports
and calculate performance degradation metrics.
"""

from datetime import datetime
from typing import TypeAlias

import numpy as np
import pandas as pd

from src.promotion import to_naive_utc

TimestampLike: TypeAlias = datetime | pd.Timestamp | str


def extract_drift_results(report_dict: dict) -> dict:
    """
    Extracts structured results from an Evidently report.as_dict() output.
    Handles both evidently 0.4.x and 0.5.x key path differences.

    Returns dict with: dataset_drift, n_drifted_features,
    drifted_features, drift_score_target, drift_score_temp.
    """
    try:
        # Standard path for Evidently 0.4.x / 0.5.x metrics
        metrics = report_dict["metrics"]

        # 1. Data Drift Preset (usually first metric)
        drift_preset = next(m for m in metrics if m["metric"] == "DatasetDriftMetric")
        res = drift_preset["result"]

        dataset_drift = res["dataset_drift"]
        n_drifted = res["number_of_drifted_columns"]
        drifted_cols = res["drifted_columns"]

        # 2. Individual Column Drift (Value MWh)
        target_drift = next(
            m
            for m in metrics
            if m["metric"] == "ColumnDriftMetric" and m["parameters"]["column_name"] == "value_mwh"
        )
        score_target = target_drift["result"]["drift_score"]

        # 3. Individual Column Drift (Temperature)
        temp_drift = next(
            m
            for m in metrics
            if m["metric"] == "ColumnDriftMetric"
            and m["parameters"]["column_name"] == "temperature_c"
        )
        score_temp = temp_drift["result"]["drift_score"]

        return {
            "dataset_drift": dataset_drift,
            "n_drifted_features": n_drifted,
            "drifted_features": drifted_cols,
            "drift_score_target": score_target,
            "drift_score_temp": score_temp,
        }
    except (KeyError, StopIteration) as e:
        raise ValueError(
            f"Failed to extract metrics from report dict. Structure may have changed: {e}"
        ) from e


def compute_prediction_mae(
    forecasts_df: pd.DataFrame, model_name: str, horizon_hours: int = 24
) -> float | None:
    """
    Computes MAE of predictions vs actuals for a given window.
    Returns None if fewer than 24 rows have non-null actuals.
    """
    # Filter for relevant model and horizon
    df = forecasts_df[
        (forecasts_df["model_name"] == model_name)
        & (forecasts_df["horizon_hours"] == horizon_hours)
    ].copy()

    # Only use rows where actuals exist
    df = df.dropna(subset=["actual_mwh"])

    if len(df) < 24:
        return None

    mae = np.mean(np.abs(df["predicted_mwh"] - df["actual_mwh"]))
    return float(mae)


# ── Retrain trigger logic (spec 011) ─────────────────────────────────────────

# Deterministic calendar columns differ between any two time windows by
# construction (a 30-day reference vs a 7-day current window almost never
# share the same `month` mix), so testing them for drift flags seasonality,
# not a real distribution shift (spec 011 §1.4).
DRIFT_EXCLUDED_FEATURES: frozenset[str] = frozenset(
    {"hour_of_day", "day_of_week", "month", "is_weekend", "is_holiday"}
)


def select_drift_columns(feature_columns: list[str]) -> list[str]:
    """Returns the feature columns that are meaningful to test for drift."""
    return [c for c in feature_columns if c not in DRIFT_EXCLUDED_FEATURES]


def count_consecutive_drift_hours(
    history: list[tuple[TimestampLike, bool]],
    check_time: TimestampLike,
    current_any_drift: bool,
) -> int:
    """Counts consecutive HOURLY drift checks ending at `check_time`.

    `history` is (check_timestamp, any_drift_detected) pairs from drift_control,
    in any order. The streak only continues while each earlier check sits exactly
    one hour before the next one — a missing hour (job outage) breaks it, so
    "3 consecutive hours" can never span 3 rows spread over days (spec 011 §1.5).
    """
    if not current_any_drift:
        return 0

    drift_by_hour = {to_naive_utc(ts).floor("h"): bool(flag) for ts, flag in history}
    count = 1
    expected = to_naive_utc(check_time).floor("h") - pd.Timedelta(hours=1)
    while drift_by_hour.get(expected):
        count += 1
        expected -= pd.Timedelta(hours=1)
    return count


def decide_retrain(
    any_drift: bool,
    consecutive_hours: int,
    consecutive_hours_threshold: int,
    now: TimestampLike,
    last_retrain_at: TimestampLike | None = None,
    cooldown_hours: float = 24,
    pending_since: TimestampLike | None = None,
    pending_timeout_hours: float = 6,
) -> tuple[bool, str]:
    """Pure retrain-trigger decision for 03_drift_check.

    A pending request older than `pending_timeout_hours` is treated as stale
    (the retrain it asked for failed or never ran) and no longer blocks a new
    trigger — otherwise one failed retrain would disable triggering forever
    (spec 011 §1.2).
    """
    if not any_drift:
        return False, "No drift detected."

    if consecutive_hours < consecutive_hours_threshold:
        return (
            False,
            f"Drift detected for {consecutive_hours}h (threshold: {consecutive_hours_threshold}h).",
        )

    now = to_naive_utc(now)
    if last_retrain_at is not None:
        hours_since = (now - to_naive_utc(last_retrain_at)).total_seconds() / 3600
        if hours_since < cooldown_hours:
            return False, f"Cooldown active: last retrain {hours_since:.1f}h ago."

    if pending_since is not None:
        pending_hours = (now - to_naive_utc(pending_since)).total_seconds() / 3600
        if pending_hours < pending_timeout_hours:
            return False, f"Retrain already pending ({pending_hours:.1f}h old)."

    return True, f"Drift detected for {consecutive_hours} consecutive hours."
