"""Unit tests for naive baseline metrics."""

import pandas as pd
import pytest

from src.baseline import compute_naive_baseline_metrics


def test_compute_naive_baseline_metrics_happy_path():
    metrics = compute_naive_baseline_metrics(pd.Series([100.0, 200.0]), pd.Series([90.0, 220.0]))

    assert metrics == {
        "naive_mae": 15.0,
        "naive_rmse": pytest.approx(15.811388300841896),
        "naive_mape": 10.0,
    }


def test_compute_naive_baseline_metrics_drops_null_pairs():
    metrics = compute_naive_baseline_metrics(
        pd.Series([100.0, None, 300.0]), pd.Series([90.0, 500.0, None])
    )

    assert metrics == {"naive_mae": 10.0, "naive_rmse": 10.0, "naive_mape": 10.0}


def test_compute_naive_baseline_metrics_handles_zero_actuals():
    metrics = compute_naive_baseline_metrics(pd.Series([0.0, 0.0]), pd.Series([10.0, -10.0]))

    assert metrics["naive_mae"] == 10.0
    assert metrics["naive_rmse"] == 10.0
    assert metrics["naive_mape"] == 0.0
