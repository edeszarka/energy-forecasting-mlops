"""
Unit tests for the monitoring logic in src/drift.py.
"""

import pandas as pd
import pytest
from src.drift import extract_drift_results, compute_prediction_mae

def test_extract_drift_results_v4_format():
    """Tests extraction from a mock evidently report dict."""
    mock_report = {
        "metrics": [
            {
                "metric": "DatasetDriftMetric",
                "result": {
                    "dataset_drift": True,
                    "number_of_drifted_columns": 2,
                    "drifted_columns": ["temp_c", "lag_24h"]
                }
            },
            {
                "metric": "ColumnDriftMetric",
                "parameters": {"column_name": "value_mwh"},
                "result": {"drift_score": 0.12}
            },
            {
                "metric": "ColumnDriftMetric",
                "parameters": {"column_name": "temperature_c"},
                "result": {"drift_score": 0.25}
            }
        ]
    }
    
    res = extract_drift_results(mock_report)
    
    assert res["dataset_drift"] is True
    assert res["n_drifted_features"] == 2
    assert "temp_c" in res["drifted_features"]
    assert res["drift_score_target"] == 0.12
    assert res["drift_score_temp"] == 0.25

def test_compute_prediction_mae_happy_path():
    """Tests MAE calculation with sufficient data."""
    df = pd.DataFrame({
        "model_name": ["energy_lgbm_24h"] * 30,
        "horizon_hours": [24] * 30,
        "predicted_mwh": [100.0] * 30,
        "actual_mwh": [110.0] * 30
    })
    
    mae = compute_prediction_mae(df, "energy_lgbm_24h", 24)
    assert mae == 10.0

def test_compute_prediction_mae_insufficient_data():
    """Tests that None is returned when fewer than 24 actuals exist."""
    df = pd.DataFrame({
        "model_name": ["energy_lgbm_24h"] * 10,
        "horizon_hours": [24] * 10,
        "predicted_mwh": [100.0] * 10,
        "actual_mwh": [110.0] * 10
    })
    
    mae = compute_prediction_mae(df, "energy_lgbm_24h", 24)
    assert mae is None

# FIX APPLIED: Added unit tests for drift result extraction and prediction MAE calculation.
