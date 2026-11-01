"""Which target hours a serving run writes, and how each is served (spec 019b, D6).

Pure functions; `04_predict` reads gold and MLflow, calls these, and writes what
they decide.

- `plan_targets`: every uncovered target T in (τ_now, τ_now + 24h], each forecast
  as if issued at τ' = T − 24h (catch-up for missed runs, B1).
- `decide_serving`: the production model, or a fallback with its reason —
  seasonal naive (T − 168h), daily naive (T − 48h) when T − 168h is missing (D8),
  or a skip when neither naive input exists (B2).
- `raise_on_model_errors`: fails the task after the gold write when a model
  error forced a fallback (D4); quiet fallbacks never fail it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.forecast_timing import issuance_hour
from src.pit_features import FORECAST_HORIZON_HOURS

MODEL = "model"
NAIVE = "naive"  # seasonal naive, T − 168h
DAILY_NAIVE = "daily_naive"  # T − 48h (spec 019b D8)
SKIP = "skip"

NO_PRODUCTION_RUN = "no_production_run"
WEATHER_TABLE_MISSING = "weather_table_missing"
MODEL_INPUTS_MISSING = "model_inputs_missing"
# model_error:* fails the task after the gold write; the other reasons never do (D4).
MODEL_ERROR = "model_error"


@dataclass(frozen=True)
class PlannedTarget:
    issued_at: pd.Timestamp  # nominal issuance hour τ' = target − 24h (UTC, tz-aware)
    target: pd.Timestamp  # UTC, tz-aware
    is_catch_up: bool  # τ' < τ_now: the run for τ' did not write it


@dataclass(frozen=True)
class ServingDecision:
    kind: str  # MODEL, NAIVE, DAILY_NAIVE or SKIP
    fallback_reason: str | None = None

    @property
    def is_model_error(self) -> bool:
        return bool(self.fallback_reason and self.fallback_reason.startswith(f"{MODEL_ERROR}:"))


def _utc(ts: Any) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def covered_targets(gold: pd.DataFrame) -> set[pd.Timestamp]:
    """Target hours that already have a served row.

    `is_served IS NULL` counts as served: before spec 017 only served rows were
    written. A covered target is never written again (§6, known limitation).
    """
    if gold.empty:
        return set()
    served = gold[gold["is_served"].isna() | (gold["is_served"] == True)]  # noqa: E712
    return {_utc(t) for t in served["timestamp"]}


def plan_targets(
    run_at: Any,
    covered: Iterable[Any],
    horizon_hours: int = FORECAST_HORIZON_HOURS,
) -> list[PlannedTarget]:
    """Uncovered targets in (τ_now, τ_now + horizon], oldest first.

    Targets ≤ τ_now are never planned: a forecast written after its target hour
    is not a forecast.
    """
    now = issuance_hour(run_at)
    done = {_utc(t) for t in covered}
    horizon = pd.Timedelta(hours=horizon_hours)
    plan = []
    for k in range(1, horizon_hours + 1):
        target = now + pd.Timedelta(hours=k)
        if target in done:
            continue
        issued = target - horizon
        plan.append(PlannedTarget(issued_at=issued, target=target, is_catch_up=issued < now))
    return plan


def decide_serving(
    *,
    model_state: str,
    weather_table_present: bool,
    missing_model_inputs: Sequence[str],
    prediction_error: str | None,
    naive_available: bool,
    daily_naive_available: bool,
) -> ServingDecision:
    """How one target is served. First match wins (spec 019b B2).

    Args:
        model_state: "ok", NO_PRODUCTION_RUN, or the exception type name of a
            failed MLflow lookup or model load.
        weather_table_present: `bronze_weather_forecast` exists and has rows.
        missing_model_inputs: required feature columns that are NaN for this target.
        prediction_error: exception type name if `predict` raised, else None.
        naive_available: load at T − 168h exists.
        daily_naive_available: load at T − 48h exists.

    D8: for any fallback reason, the seasonal naive (T − 168h) serves if it
    exists, else the daily naive (T − 48h), else the target is skipped. The
    reason is always the triggering one, so `model_error:*` still fails the task
    (raise_on_model_errors); `model_name` tells which naive served.
    """
    if model_state == NO_PRODUCTION_RUN:
        reason = NO_PRODUCTION_RUN
    elif model_state != "ok":
        reason = f"{MODEL_ERROR}:{model_state}"
    elif not weather_table_present:
        reason = WEATHER_TABLE_MISSING
    elif missing_model_inputs:
        reason = f"{MODEL_INPUTS_MISSING}:{','.join(missing_model_inputs)}"
    elif prediction_error is not None:
        reason = f"{MODEL_ERROR}:{prediction_error}"
    else:
        return ServingDecision(MODEL)
    if naive_available:
        return ServingDecision(NAIVE, reason)
    if daily_naive_available:
        return ServingDecision(DAILY_NAIVE, reason)
    return ServingDecision(SKIP, reason)


def raise_on_model_errors(decisions: Iterable[tuple[PlannedTarget, ServingDecision]]) -> None:
    """Fails the task when a model error forced any fallback (D4).

    Called after the gold write, so the hour still has its fallback forecast and
    the job's failure email still goes out.
    """
    errors = [(p.target, d.fallback_reason) for p, d in decisions if d.is_model_error]
    if errors:
        listed = ", ".join(f"{t.isoformat()} ({r})" for t, r in errors)
        raise RuntimeError(f"Model error forced a fallback for {len(errors)} target(s): {listed}")


def summarize_decisions(
    decisions: Iterable[tuple[PlannedTarget, ServingDecision]], n_covered: int
) -> dict[str, int]:
    """Counts for the task exit value: model, naive by reason, skipped, already covered."""
    counts: dict[str, int] = {"already_covered": n_covered}
    for planned, decision in decisions:
        key = (
            decision.kind
            if decision.kind == MODEL
            else f"{decision.kind}:{decision.fallback_reason}"
        )
        counts[key] = counts.get(key, 0) + 1
        if planned.is_catch_up:
            counts["catch_up"] = counts.get("catch_up", 0) + 1
    return counts
