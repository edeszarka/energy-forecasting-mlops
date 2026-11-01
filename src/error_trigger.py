"""Error-triggered retraining decision (spec 018, ADR-0007).

The served LightGBM run is judged only on the target hours it served itself
(`model_version = <run_id>`), paired with seasonal naive on the same hours, so
a freshly promoted run is never blamed for its predecessor's errors. Data drift
plays no part. Pure functions; `09_error_monitor` reads and writes the tables.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config import (
    ERROR_ALERT_WINDOW_HOURS,
    ERROR_MAPE_CEILING,
    ERROR_RETRAIN_COOLDOWN_HOURS,
    ERROR_RETRAIN_MAX_SKILL,
    ERROR_RETRAIN_MIN_HOURS,
    ERROR_RETRAIN_STREAK_HOURS,
    NAIVE_MODEL_NAME,
    RETRAIN_PENDING_TIMEOUT_HOURS,
)
from src.drift import count_consecutive_hours, decide_retrain
from src.error_monitoring import paired_metrics
from src.forecast_timing import SERVED_MODEL
from src.promotion import to_naive_utc

# Served LightGBM rows written before spec 018 carry this instead of a run id.
LEGACY_MODEL_VERSION = "run_latest"


@dataclass(frozen=True)
class RetrainThresholds:
    min_hours: int = ERROR_RETRAIN_MIN_HOURS
    streak_hours: int = ERROR_RETRAIN_STREAK_HOURS
    cooldown_hours: float = ERROR_RETRAIN_COOLDOWN_HOURS
    max_skill: float = ERROR_RETRAIN_MAX_SKILL
    mape_ceiling: float = ERROR_MAPE_CEILING
    pending_timeout_hours: float = RETRAIN_PENDING_TIMEOUT_HOURS


DEFAULT_RETRAIN_THRESHOLDS = RetrainThresholds()


@dataclass(frozen=True)
class Degradation:
    degraded: bool
    reason: str
    run_id: str | None = None
    n_hours: int = 0
    mape: float | None = None
    relative_skill: float | None = None


def _live_served(gold: pd.DataFrame) -> pd.DataFrame:
    live = gold[~gold["is_backfilled"].fillna(False).astype(bool)]
    return live[live["is_served"].isna() | (live["is_served"] == True)]  # noqa: E712


def current_served_run(gold: pd.DataFrame) -> tuple[str, str] | None:
    """(model_name, model_version) of the most recent served target hour."""
    served = _live_served(gold)
    if served.empty:
        return None
    latest = served.assign(_ts=[to_naive_utc(t) for t in served["timestamp"]]).sort_values("_ts")
    return str(latest["model_name"].iloc[-1]), str(latest["model_version"].iloc[-1])


def assess_degradation(
    gold: pd.DataFrame,
    check_time: Any,
    window_hours: int = ERROR_ALERT_WINDOW_HOURS,
    thresholds: RetrainThresholds = DEFAULT_RETRAIN_THRESHOLDS,
) -> Degradation:
    """Is the currently served run degraded? (spec 018 T2)

    Degraded when, on at least `min_hours` hours it served itself within the
    window, it is worse than naive on the same hours (skill < `max_skill`) or
    its MAPE is above the ceiling. Never degraded while naive serves.
    """
    current = current_served_run(gold)
    if current is None:
        return Degradation(False, "No served forecasts in gold.")
    model_name, run_id = current
    if model_name == NAIVE_MODEL_NAME:
        return Degradation(False, "Seasonal naive serves: no model to judge.")
    if run_id == LEGACY_MODEL_VERSION:
        return Degradation(False, "Served rows carry no run id (written before spec 018).")

    live = gold[~gold["is_backfilled"].fillna(False).astype(bool)]
    served = _live_served(gold)
    own = served[(served["model_name"] == SERVED_MODEL) & (served["model_version"] == run_id)]
    naive = live[live["model_name"] == NAIVE_MODEL_NAME]
    paired = paired_metrics(own, naive, check_time, window_hours)
    n, mape, skill = paired.model.n, paired.model.mape, paired.relative_skill

    if n < thresholds.min_hours:
        return Degradation(
            False,
            f"Run {run_id}: {n}h of own served hours paired with naive (need {thresholds.min_hours}).",
            run_id,
            n,
            mape,
            skill,
        )
    reasons = []
    if mape is not None and mape > thresholds.mape_ceiling:
        reasons.append(f"MAPE {mape:.2f}% > {thresholds.mape_ceiling:.2f}%")
    if skill is not None and skill < thresholds.max_skill:
        reasons.append(f"skill vs naive {skill:+.3f} < {thresholds.max_skill:+.3f}")
    if reasons:
        return Degradation(
            True, f"Run {run_id} over {n}h: {'; '.join(reasons)}.", run_id, n, mape, skill
        )
    skill_txt = f"{skill:+.3f}" if skill is not None else "n/a"
    return Degradation(
        False,
        f"Run {run_id} over {n}h: skill vs naive {skill_txt}, not degraded.",
        run_id,
        n,
        mape,
        skill,
    )


def decide_error_retrain(
    degradation: Degradation,
    history: Sequence[tuple[Any, bool]],
    check_time: Any,
    last_retrain_at: Any | None = None,
    pending_since: Any | None = None,
    thresholds: RetrainThresholds = DEFAULT_RETRAIN_THRESHOLDS,
) -> tuple[int, bool, str]:
    """(consecutive degraded hours, start a retrain?, reason) for this check.

    `history` is (check_timestamp, degraded) from earlier monitor rows; the
    streak, cooldown and pending rules are spec 011's (`decide_retrain`).
    """
    consecutive = count_consecutive_hours(list(history), check_time, degradation.degraded)
    if not degradation.degraded:
        return consecutive, False, degradation.reason
    should, reason = decide_retrain(
        True,
        consecutive,
        thresholds.streak_hours,
        now=check_time,
        last_retrain_at=last_retrain_at,
        cooldown_hours=thresholds.cooldown_hours,
        pending_since=pending_since,
        pending_timeout_hours=thresholds.pending_timeout_hours,
        signal_name="Forecast degradation",
    )
    return consecutive, should, f"{reason} {degradation.reason}"


def start_retrain(
    reason: str,
    job_id: str,
    start_job: Callable[[int], tuple[int | None, str]],
    write_flag: Callable[[str, int | None], Any],
) -> tuple[bool, str, int | None]:
    """Starts the retraining job and writes the pending flag (spec 018 T3).

    Returns (retrain_triggered, reason, job_run_id). A failure to start the job
    is recorded and returned, never raised: the monitor task fails only on an
    alert (spec 017). Without a job id only the flag is written, and the weekly
    retrain picks the request up.
    """
    run_id = None
    if job_id:
        try:
            run_id, message = start_job(int(job_id))
        except Exception as e:  # recorded; the next hourly check retries
            return False, f"Trigger failed: {type(e).__name__}: {e}", None
        reason = f"{reason} {message}"
    write_flag(reason, run_id)
    return True, reason, run_id
