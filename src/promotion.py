"""Champion/challenger comparison and promotion decision logic (spec 021, ADR-0009).

The challenger's refit trained on every row after the champion's cutoff, so it
cannot be scored there. Instead an as-of model, built by the challenger's code on
the rows up to the cutoff, is compared with the champion on the rows after it:
same data age, same rows, both out-of-sample. Metrics from different windows are
never compared.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROMOTABLE_MODELS

# Per-row backtest logged by 06, for audit.
BACKTEST_ARTIFACT: str = "backtest_predictions.json"
# Parameters logged by 06, with their real types; 07 trains the as-of model with them.
TRAINING_PARAMS_ARTIFACT: str = "training_params.json"

BENCHMARK_ONLY_REASON: str = "Benchmark-only model (ADR-0002) — never promoted."

# Fewer rows after the champion's cutoff than this: no comparison, the champion stays.
MIN_COMMON_ROWS: dict[str, int] = {"lgbm": 24}


def calculate_mape(actual: pd.Series, predicted: pd.Series) -> float:
    """MAPE in percent, skipping zero actuals (matches 05/06)."""
    actual = pd.Series(actual).reset_index(drop=True)
    predicted = pd.Series(predicted).reset_index(drop=True)
    mask = actual != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def to_naive_utc(ts: Any) -> pd.Timestamp:
    """Normalises a timestamp (str, datetime, tz-aware or naive) to naive UTC."""
    ts = pd.Timestamp(ts)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo is not None else ts


def pick_production_run(runs: list[Any], exclude_run_id: str | None = None) -> Any | None:
    """Returns the current production run from runs tagged `production=true`.

    If more than one run carries the tag (a crash between the two tag writes in
    08), the most recently promoted one wins, so 04/07/08 all agree.
    """
    candidates = [r for r in runs if r.info.run_id != exclude_run_id]
    if not candidates:
        return None

    def promoted_at(run: Any) -> pd.Timestamp:
        tag = run.data.tags.get("promoted_at")
        if tag:
            return to_naive_utc(tag)
        return to_naive_utc(datetime.fromtimestamp(run.info.start_time / 1000.0, tz=UTC))

    return max(candidates, key=promoted_at)


def served_data_end(run: Any) -> pd.Timestamp:
    """Latest timestamp the run's SERVED model may have seen during fitting.

    Runs from before spec 011 have no `served_data_end` tag; the run's start
    time is a safe upper bound (a model cannot have seen data that arrived
    after it started training).
    """
    tag = run.data.tags.get("served_data_end")
    if tag:
        return to_naive_utc(tag)
    return to_naive_utc(datetime.fromtimestamp(run.info.start_time / 1000.0, tz=UTC))


def _target_hours(rows: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(rows["target_timestamp"]).map(to_naive_utc)


def as_of_train_rows(train: pd.DataFrame, cutoff: Any) -> pd.DataFrame:
    """Rows the as-of model may train on: a target at or before the champion's cutoff."""
    return train[_target_hours(train) <= to_naive_utc(cutoff)]


def select_as_of_window(rows: pd.DataFrame, cutoff: Any) -> pd.DataFrame:
    """The common window: a target after the cutoff, so neither model trained on it."""
    return rows[_target_hours(rows) > to_naive_utc(cutoff)]


def decide_as_of_promotion(
    champion_mape: float | None,
    as_of_mape: float | None,
    challenger_mape: float | None,
    naive_mape: float | None,
    n_common: int,
    delta: float,
    first_run: bool,
    force: bool = False,
) -> tuple[bool, str]:
    """Promotes when (a) as-of <= champion * (1 + delta) and (b) challenger < naive.

    `challenger_mape` and `naive_mape` are the challenger's logged backtest metrics;
    the other two are scored on the common window. Without a champion only (b)
    counts (spec 014 T7). Too few common rows or an invalid metric keep the champion.
    """
    if force:
        return True, "Force promotion requested via widget"
    if first_run:
        return _decide_first_run(challenger_mape, naive_mape)

    min_rows = MIN_COMMON_ROWS["lgbm"]
    if n_common < min_rows:
        return False, f"Only {n_common} common rows (need {min_rows}). Keeping champion."
    if as_of_mape is None or pd.isna(as_of_mape):
        return False, "Invalid as-of metric. Keeping champion."
    if champion_mape is None or pd.isna(champion_mape) or champion_mape <= 0:
        return False, "Invalid champion metric. Keeping champion."
    if challenger_mape is None or pd.isna(challenger_mape):
        return False, "Invalid challenger backtest metric. Keeping champion."
    if naive_mape is None or pd.isna(naive_mape):
        return False, "Invalid naive metric. Keeping champion."

    limit = champion_mape * (1 + delta)
    if as_of_mape > limit:
        return False, (
            f"As-of model is worse than the limit ({as_of_mape:.3f}% > {limit:.3f}%, "
            f"champion {champion_mape:.3f}%). Keeping champion."
        )
    if challenger_mape >= naive_mape:
        return False, (
            f"Challenger does not beat naive ({challenger_mape:.3f}% >= {naive_mape:.3f}%). "
            "Keeping champion."
        )
    return True, (
        f"As-of model within the limit ({as_of_mape:.3f}% <= {limit:.3f}%, "
        f"champion {champion_mape:.3f}%) and challenger beats naive "
        f"({challenger_mape:.3f}% < {naive_mape:.3f}%)"
    )


def _decide_first_run(challenger_mape: float | None, naive_mape: float | None) -> tuple[bool, str]:
    """No production run exists: promote only if the challenger beats naive."""
    if challenger_mape is None or pd.isna(challenger_mape):
        return (
            False,
            "No Production run exists; invalid challenger metrics — serving seasonal naive",
        )
    if naive_mape is None or pd.isna(naive_mape):
        return False, "No Production run exists; no naive_mape logged — serving seasonal naive"
    if challenger_mape < naive_mape:
        return True, (
            f"No Production run exists; challenger beats naive "
            f"({challenger_mape:.3f}% < {naive_mape:.3f}%)"
        )
    return False, (
        f"No Production run exists; challenger does not beat naive "
        f"({challenger_mape:.3f}% >= {naive_mape:.3f}%) — serving seasonal naive"
    )


def is_promotable(model_name: str) -> bool:
    """Only PROMOTABLE_MODELS may ever be production-tagged (ADR-0002)."""
    return model_name in PROMOTABLE_MODELS


def apply_benchmark_policy(model_name: str, challenger_wins: bool, reason: str) -> tuple[bool, str]:
    """Overrides a promotion decision for benchmark-only models.

    Applied after decide_as_of_promotion, so it also covers first_run and force: the
    comparison is still computed and reported, but a benchmark never wins.
    """
    if is_promotable(model_name):
        return challenger_wins, reason
    return False, f"{BENCHMARK_ONLY_REASON} {reason}".strip()
