"""Champion/challenger comparison and promotion decision logic (spec 011).

The champion and challenger must be scored on the SAME rows: each run's
logged `mape` comes from its own holdout window, and after drift a newer
window is usually harder, so comparing logged metrics would bias the decision
towards keeping a stale champion (or promoting a worse challenger).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROMOTABLE_MODELS

# Name of the per-row backtest artifact logged by 05/06 and read by 07.
BACKTEST_ARTIFACT: str = "backtest_predictions.json"

BENCHMARK_ONLY_REASON: str = "Benchmark-only model (ADR-0002) — never promoted."

# Minimum number of shared out-of-sample rows needed for a same-window
# comparison. Prophet's rolling-origin backtest only has n_folds targets.
MIN_COMMON_ROWS: dict[str, int] = {"lgbm": 24, "prophet": 3}


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


def select_common_window(backtest: pd.DataFrame, champion_end: Any) -> pd.DataFrame:
    """Keeps backtest rows whose TARGET is out-of-sample for the champion too."""
    targets = pd.to_datetime(backtest["target_timestamp"]).map(to_naive_utc)
    return backtest[targets > to_naive_utc(champion_end)].reset_index(drop=True)


def decide_promotion(
    champion_mape: float | None,
    challenger_mape: float | None,
    threshold: float,
    first_run: bool,
    force: bool = False,
    challenger_naive_mape: float | None = None,
) -> tuple[bool, str]:
    """Champion/challenger rule: promote only on > `threshold` relative MAPE gain.

    When the challenger is not clearly better, the champion stays and nothing
    changes in production. Without a champion (`first_run`) the challenger must
    beat the naive baseline logged on its own test rows (`challenger_naive_mape`),
    otherwise serving stays on seasonal naive (spec 014 T7).
    """
    if force:
        return True, "Force promotion requested via widget"
    if first_run:
        return _decide_first_run(challenger_mape, challenger_naive_mape)
    if challenger_mape is None or pd.isna(challenger_mape):
        return False, "Invalid challenger metrics — keeping champion"
    if champion_mape is None or pd.isna(champion_mape) or champion_mape <= 0:
        return False, "Invalid champion metrics — keeping champion"

    improvement = (champion_mape - challenger_mape) / champion_mape
    if improvement > threshold:
        return True, (
            f"MAPE improved by {improvement:.2%} ({champion_mape:.3f}% -> {challenger_mape:.3f}%)"
        )
    return False, (
        f"Insufficient improvement ({improvement:.2%} <= {threshold:.2%}). Keeping champion."
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

    Applied after decide_promotion, so it also covers first_run and force: the
    comparison is still computed and reported, but a benchmark never wins.
    """
    if is_promotable(model_name):
        return challenger_wins, reason
    return False, f"{BENCHMARK_ONLY_REASON} {reason}".strip()
