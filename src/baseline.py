"""Metrics for naive forecasting baselines, and the seasonal-naive serving fallback."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.promotion import to_naive_utc


def compute_naive_baseline_metrics(
    actual: pd.Series, baseline_prediction: pd.Series
) -> dict[str, float]:
    """Compute MAE, RMSE, and MAPE for paired baseline predictions."""
    paired = pd.DataFrame({"actual": actual, "baseline_prediction": baseline_prediction}).dropna()
    if paired.empty:
        raise ValueError(
            "Cannot compute baseline metrics without comparable actuals and predictions."
        )

    errors = paired["actual"] - paired["baseline_prediction"]
    nonzero_actuals = paired["actual"] != 0
    mape = 0.0
    if nonzero_actuals.any():
        mape = float(
            np.mean(np.abs(errors[nonzero_actuals] / paired.loc[nonzero_actuals, "actual"])) * 100
        )

    return {
        "naive_mae": float(np.mean(np.abs(errors))),
        "naive_rmse": float(np.sqrt(np.mean(errors**2))),
        "naive_mape": mape,
    }


def seasonal_naive_forecast(
    history: pd.DataFrame,
    targets: Sequence[pd.Timestamp],
    lag_hours: int = 168,
) -> pd.DataFrame:
    """Seasonal-naive serving fallback (ADR-0002): predicted(T) = actual(T - lag_hours).

    Returns `timestamp, predicted_mwh` for the targets whose source hour is present
    in `history` (columns `timestamp`, `value_mwh`) with a non-null value; other
    targets are skipped. Timestamps are matched as UTC instants, tz-aware or naive.

    Precondition: with targets at most run+48h and the default one-week lag, the
    source hours are run-143h..run-120h, so silver must be no more than 120h stale
    for full coverage (Part A for spec 013, Q6).

    Raises:
        ValueError: If no target has a source value.
    """
    observed = history.dropna(subset=["value_mwh"])
    value_by_ts = pd.Series(
        observed["value_mwh"].to_numpy(dtype=float),
        index=pd.DatetimeIndex([to_naive_utc(t) for t in observed["timestamp"]]),
    )
    value_by_ts = value_by_ts[~value_by_ts.index.duplicated(keep="last")]

    lag = pd.Timedelta(hours=lag_hours)
    rows = []
    for target in targets:
        source = to_naive_utc(target) - lag
        if source in value_by_ts.index:
            rows.append({"timestamp": target, "predicted_mwh": float(value_by_ts[source])})

    if not rows:
        raise ValueError(
            f"Seasonal naive has no T-{lag_hours}h value for any of {len(targets)} targets."
        )
    return pd.DataFrame(rows, columns=["timestamp", "predicted_mwh"])
