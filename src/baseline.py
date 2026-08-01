"""Metrics for naive forecasting baselines."""

import numpy as np
import pandas as pd


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
