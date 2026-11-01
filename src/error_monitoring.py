"""Live forecast error monitoring (spec 017, ADR-0006).

Pure functions over gold rows (`timestamp, model_name, predicted_mwh, actual_mwh,
is_served, is_backfilled`): rolling-window error metrics for the served forecast,
seasonal naive and the shadow LightGBM run, a paired comparison on the same
target hours, and the alert rules. No Spark and no I/O; `09_error_monitor` reads
gold and writes `forecast_error_monitor`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    ERROR_ALERT_WINDOW_HOURS,
    ERROR_MAPE_CEILING,
    ERROR_MIN_COVERAGE,
    ERROR_MIN_SKILL,
    ERROR_SHADOW_MIN_ROWS,
    ERROR_SHADOW_SKILL,
    ERROR_WINDOWS_HOURS,
    NAIVE_MODEL_NAME,
    NAIVE_MODEL_NAMES,
)
from src.forecast_timing import SERVED_MODEL
from src.promotion import to_naive_utc

SERVED_STREAM = "served"
MODEL_STREAM = "served_model"  # served LightGBM rows only (spec 019b)
NAIVE_STREAM = "naive"
SHADOW_STREAM = "shadow_lgbm"
STREAMS: tuple[str, ...] = (SERVED_STREAM, MODEL_STREAM, NAIVE_STREAM, SHADOW_STREAM)

SERVED_MAPE_HIGH = "SERVED_MAPE_HIGH"
SERVED_WORSE_THAN_NAIVE = "SERVED_WORSE_THAN_NAIVE"
SHADOW_BEATS_NAIVE = "SHADOW_BEATS_NAIVE"
LOW_COVERAGE = "LOW_COVERAGE"
# Informational alerts are recorded but never fail the monitor task.
INFORMATIONAL_ALERTS: frozenset[str] = frozenset({SHADOW_BEATS_NAIVE})

ROW_COLUMNS = ["timestamp", "predicted_mwh", "actual_mwh"]


@dataclass(frozen=True)
class Thresholds:
    mape_ceiling: float = ERROR_MAPE_CEILING
    min_skill: float = ERROR_MIN_SKILL
    shadow_skill: float = ERROR_SHADOW_SKILL
    shadow_min_rows: int = ERROR_SHADOW_MIN_ROWS
    min_coverage: float = ERROR_MIN_COVERAGE


DEFAULT_THRESHOLDS = Thresholds()


@dataclass(frozen=True)
class WindowMetrics:
    n: int
    coverage: float
    mape: float | None
    mae: float | None
    bias: float | None  # mean(predicted - actual): positive = over-forecast


def _normalise(rows: pd.DataFrame) -> pd.DataFrame:
    """Rows with naive-UTC timestamps, one per target hour, sorted."""
    out = rows[ROW_COLUMNS].copy()
    out["timestamp"] = pd.DatetimeIndex(
        [to_naive_utc(t) for t in out["timestamp"]], dtype="datetime64[ns]"
    )
    return out.drop_duplicates("timestamp", keep="last").sort_values("timestamp")


def split_streams(gold: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Splits gold rows into the monitored streams.

    - served: what was delivered — `is_served = true`, or NULL on rows written
      before spec 017 (only served rows existed then), whichever forecaster
      served (model, seasonal naive or daily naive). Backfilled rows never count.
    - served_model: the served rows of the LightGBM model only; the model's live
      metrics and its skill against naive come from these (spec 019b).
    - naive: every live `seasonal_naive` row, served or not.
    - shadow_lgbm: LightGBM rows written with `is_served = false`.
    """
    live = gold[~gold["is_backfilled"].fillna(False).astype(bool)]
    is_served = live["is_served"]
    served = live[is_served.isna() | (is_served == True)]  # noqa: E712 (nullable column)
    naive = live[live["model_name"] == NAIVE_MODEL_NAME]
    shadow = live[(live["model_name"] == SERVED_MODEL) & (is_served == False)]  # noqa: E712
    return {
        SERVED_STREAM: _normalise(served),
        MODEL_STREAM: _normalise(served[served["model_name"] == SERVED_MODEL]),
        NAIVE_STREAM: _normalise(naive),
        SHADOW_STREAM: _normalise(shadow),
    }


def _served_by_time(gold: pd.DataFrame) -> pd.DataFrame:
    live = gold[~gold["is_backfilled"].fillna(False).astype(bool)]
    served = live[live["is_served"].isna() | (live["is_served"] == True)]  # noqa: E712
    return served.assign(_ts=[to_naive_utc(t) for t in served["timestamp"]]).sort_values("_ts")


def latest_served_model(gold: pd.DataFrame) -> str | None:
    """The model behind the most recent target hour that a model served.

    Naive fallbacks (seasonal or daily) never identify a model: hours they served
    are skipped, and None means no model served in the given rows.
    """
    served = _served_by_time(gold)
    models = served[~served["model_name"].isin(NAIVE_MODEL_NAMES)]
    return None if models.empty else str(models["model_name"].iloc[-1])


def latest_served_forecaster(gold: pd.DataFrame) -> str | None:
    """`model_name` of the most recent served row, naive or not (labels the served stream)."""
    served = _served_by_time(gold)
    return None if served.empty else str(served["model_name"].iloc[-1])


def _in_window(rows: pd.DataFrame, window_end: Any, window_hours: int) -> pd.DataFrame:
    end = to_naive_utc(window_end)
    start = end - pd.Timedelta(hours=window_hours)
    scored = rows.dropna(subset=["actual_mwh", "predicted_mwh"])
    return scored[(scored["timestamp"] > start) & (scored["timestamp"] <= end)]


def _metrics(scored: pd.DataFrame, window_hours: int) -> WindowMetrics:
    n = len(scored)
    if n == 0:
        return WindowMetrics(n=0, coverage=0.0, mape=None, mae=None, bias=None)
    actual = scored["actual_mwh"].to_numpy(dtype=float)
    predicted = scored["predicted_mwh"].to_numpy(dtype=float)
    errors = predicted - actual
    # Same MAPE definition as src.backtest.mape: actual = 0 is excluded.
    nonzero = actual != 0
    mape = (
        float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100) if nonzero.any() else None
    )
    return WindowMetrics(
        n=n,
        coverage=n / window_hours,
        mape=mape,
        mae=float(np.mean(np.abs(errors))),
        bias=float(np.mean(errors)),
    )


def window_metrics(rows: pd.DataFrame, window_end: Any, window_hours: int) -> WindowMetrics:
    """Error metrics over target hours in `(window_end - window_hours, window_end]`.

    Rows without an actual or a prediction are ignored; coverage is the share of
    the window's hours that could be scored.
    """
    return _metrics(_in_window(_normalise(rows), window_end, window_hours), window_hours)


@dataclass(frozen=True)
class PairedMetrics:
    model: WindowMetrics
    naive: WindowMetrics
    relative_skill: float | None  # 1 - model MAPE / naive MAPE; > 0 = model beats naive


def paired_metrics(
    model_rows: pd.DataFrame, naive_rows: pd.DataFrame, window_end: Any, window_hours: int
) -> PairedMetrics:
    """Model and naive metrics on the target hours both of them scored, and only those."""
    model = _in_window(_normalise(model_rows), window_end, window_hours)
    naive = _in_window(_normalise(naive_rows), window_end, window_hours)
    common = set(model["timestamp"]) & set(naive["timestamp"])
    model_m = _metrics(model[model["timestamp"].isin(common)], window_hours)
    naive_m = _metrics(naive[naive["timestamp"].isin(common)], window_hours)
    skill = None
    if model_m.mape is not None and naive_m.mape:
        skill = 1.0 - model_m.mape / naive_m.mape
    return PairedMetrics(model=model_m, naive=naive_m, relative_skill=skill)


def evaluate_alerts(
    served: WindowMetrics,
    model_vs_naive: PairedMetrics | None,
    shadow_vs_naive: PairedMetrics | None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> list[str]:
    """Alert codes for one check (spec 017 T5), evaluated on the alert window.

    `model_vs_naive` (LightGBM-served hours only) is None when no model served. LOW_COVERAGE
    suppresses the served alerts, which cannot be trusted on too few rows.
    """
    alerts = []
    if served.coverage < thresholds.min_coverage:
        alerts.append(LOW_COVERAGE)
    else:
        if served.mape is not None and served.mape > thresholds.mape_ceiling:
            alerts.append(SERVED_MAPE_HIGH)
        if (
            model_vs_naive is not None
            and model_vs_naive.relative_skill is not None
            and model_vs_naive.relative_skill < thresholds.min_skill
        ):
            alerts.append(SERVED_WORSE_THAN_NAIVE)
    if (
        shadow_vs_naive is not None
        and shadow_vs_naive.relative_skill is not None
        and shadow_vs_naive.model.n >= thresholds.shadow_min_rows
        and shadow_vs_naive.relative_skill >= thresholds.shadow_skill
    ):
        alerts.append(SHADOW_BEATS_NAIVE)
    return alerts


def new_failing_alerts(current: Iterable[str], previous: Iterable[str] | None) -> list[str]:
    """Failing alerts that started at this check (absent from the previous check).

    The monitor fails once per alert episode, so the job's failure email fires
    once; informational alerts never fail it.
    """
    before = set(previous or [])
    return [a for a in current if a not in INFORMATIONAL_ALERTS and a not in before]


def _row(
    check_timestamp: Any,
    window_hours: int,
    stream: str,
    model_name: str | None,
    metrics: WindowMetrics,
    paired: PairedMetrics | None,
    alerts: Sequence[str],
) -> dict[str, Any]:
    return {
        "check_timestamp": check_timestamp,
        "window_hours": window_hours,
        "stream": stream,
        "model_name": model_name,
        "n": metrics.n,
        "coverage": metrics.coverage,
        "mape": metrics.mape,
        "mae": metrics.mae,
        "bias": metrics.bias,
        "naive_mape_paired": paired.naive.mape if paired else None,
        "relative_skill": paired.relative_skill if paired else None,
        "alerts": list(alerts),
    }


def build_monitor_rows(
    gold: pd.DataFrame,
    check_timestamp: Any,
    windows: Sequence[int] = ERROR_WINDOWS_HOURS,
    alert_window: int = ERROR_ALERT_WINDOW_HOURS,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> tuple[list[dict[str, Any]], list[str]]:
    """One `forecast_error_monitor` row per (window, stream), and this check's alerts.

    Alerts are stored on the served row of the alert window only. The served
    row measures what was delivered (any forecaster) and carries no skill; the
    model's metrics and its skill against naive are on the `served_model` row,
    computed on the LightGBM-served hours only, so hours served by a naive
    fallback (seasonal or daily) never enter the model's skill or its alert.
    """
    streams = split_streams(gold)
    model = latest_served_model(gold)
    shadow_model = SERVED_MODEL if not streams[SHADOW_STREAM].empty else None

    rows: list[dict[str, Any]] = []
    alerts: list[str] = []
    for hours in windows:
        served = window_metrics(streams[SERVED_STREAM], check_timestamp, hours)
        model_paired = (
            paired_metrics(streams[MODEL_STREAM], streams[NAIVE_STREAM], check_timestamp, hours)
            if model
            else None
        )
        shadow_paired = (
            paired_metrics(streams[SHADOW_STREAM], streams[NAIVE_STREAM], check_timestamp, hours)
            if shadow_model
            else None
        )
        if hours == alert_window:
            alerts = evaluate_alerts(served, model_paired, shadow_paired, thresholds)

        rows.append(
            _row(
                check_timestamp,
                hours,
                SERVED_STREAM,
                latest_served_forecaster(gold),
                served,
                None,
                alerts if hours == alert_window else [],
            )
        )
        if model_paired is not None:
            model_metrics = window_metrics(streams[MODEL_STREAM], check_timestamp, hours)
            rows.append(
                _row(check_timestamp, hours, MODEL_STREAM, model, model_metrics, model_paired, [])
            )
        naive = window_metrics(streams[NAIVE_STREAM], check_timestamp, hours)
        rows.append(_row(check_timestamp, hours, NAIVE_STREAM, NAIVE_MODEL_NAME, naive, None, []))
        if shadow_paired is not None:
            shadow = window_metrics(streams[SHADOW_STREAM], check_timestamp, hours)
            rows.append(
                _row(check_timestamp, hours, SHADOW_STREAM, shadow_model, shadow, shadow_paired, [])
            )
    return rows, alerts
