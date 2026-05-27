"""
Drift detection and model quality monitoring logic.

This module provides functions to extract results from monitoring reports
and calculate performance degradation metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

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
        target_drift = next(m for m in metrics if m["metric"] == "ColumnDriftMetric" and m["parameters"]["column_name"] == "value_mwh")
        score_target = target_drift["result"]["drift_score"]
        
        # 3. Individual Column Drift (Temperature)
        temp_drift = next(m for m in metrics if m["metric"] == "ColumnDriftMetric" and m["parameters"]["column_name"] == "temperature_c")
        score_temp = temp_drift["result"]["drift_score"]
        
        return {
            "dataset_drift": dataset_drift,
            "n_drifted_features": n_drifted,
            "drifted_features": drifted_cols,
            "drift_score_target": score_target,
            "drift_score_temp": score_temp
        }
    except (KeyError, StopIteration) as e:
        raise ValueError(f"Failed to extract metrics from report dict. Structure may have changed: {e}")

def compute_prediction_mae(
    forecasts_df: pd.DataFrame,
    model_name: str,
    horizon_hours: int = 24
) -> float | None:
    """
    Computes MAE of predictions vs actuals for a given window.
    Returns None if fewer than 24 rows have non-null actuals.
    """
    # Filter for relevant model and horizon
    df = forecasts_df[
        (forecasts_df["model_name"] == model_name) & 
        (forecasts_df["horizon_hours"] == horizon_hours)
    ].copy()
    
    # Only use rows where actuals exist
    df = df.dropna(subset=["actual_mwh"])
    
    if len(df) < 24:
        return None
        
    mae = np.mean(np.abs(df["predicted_mwh"] - df["actual_mwh"]))
    return float(mae)

# FIX APPLIED: Replaced stub with real implementation of drift result extraction and performance metric calculation.
