# Project Context: energy-forecasting-mlops

## Environment Constraints
- **Platform**: Databricks Free Edition (Serverless).
- **Network**: Air-gapped (no outbound internet). API calls must happen in GitHub Actions.
- **Unity Catalog**: Use catalog `workspace` and schema `energy_forecasting`.
- **Storage**: Use Unity Catalog Volumes for raw data landing and reports.
  Base path: `/Volumes/workspace/energy_forecasting/data/`.
- **Compute**: Use serverless compute (`environment_key: "default"` in databricks.yml).
  Do NOT use job_cluster_key or classic cluster definitions.

## Split Ingestion Architecture
Databricks Free Edition has NO outbound internet access. This project uses a **Split Ingestion** pattern:

```text
[ GitHub Actions ] (Internet)  ──>  [ UC Volumes ]  ──>  [ Databricks ] (Air-gapped)
       (ingest.yml)                 (/raw_ingestion/)        (01_ingest.py)
```

1. **GitHub Actions (ingest.yml)**: Calls ENTSO-E and OpenMeteo APIs, segments data into hourly JSON files, and uploads them to UC Volumes via the Databricks CLI.
2. **Databricks (01_ingest.py)**: Reads the local Volume paths, validates schema, and performs idempotent `MERGE INTO` the Bronze layer.

Never attempt to call external APIs directly from a notebook.

## Engineering Standards
- **Idempotency**: All writes must use `MERGE INTO` or deterministic hashing
  (`MD5(model_name + "_" + horizon_hours + "_" + timestamp.isoformat())`) for IDs.
- **Paths**: Use `pathlib.Path` for local/GHA logic.
  Use `/Volumes/workspace/energy_forecasting/data/` for Volume paths in notebooks.
- **Feature Consistency**: All feature engineering MUST use `src/features.py`
  to prevent training-serving skew.
- **Logging**: Use Python `logging` (INFO level) for system events.
  Format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
  Use `dbutils.notebook.exit()` with JSON payloads for cross-task metadata.
- **MLflow tags**: Every training run MUST set these tags:
    mlflow.set_tag("training_data_end", <ISO timestamp of last training row>)
    mlflow.set_tag("training_data_start", <ISO timestamp of first training row>)
  These are read by 03_drift_check.py to define the drift reference window.

## Data Schema Contracts

### Bronze Load: `workspace.energy_forecasting.bronze_load`
- `timestamp` (TimestampType, UTC)
- `country` (StringType)
- `value_mwh` (DoubleType, nullable)
- `source` (StringType)
- `fetched_at` (TimestampType)
- `run_id` (StringType)
- `is_gap` (BooleanType)

### Bronze Temperature: `workspace.energy_forecasting.bronze_temperature`
- `timestamp` (TimestampType, UTC)
- `temperature_c` (DoubleType, nullable)
- `is_temp_imputed` (BooleanType)
- `source` (StringType)
- `fetched_at` (TimestampType)
- `run_id` (StringType)

### Silver Features: `workspace.energy_forecasting.silver_features`
EXACT column names — use these verbatim in all notebooks, no aliases:
- `timestamp` (TimestampType, UTC)
- `country` (StringType)
- `value_mwh` (DoubleType, nullable — NULL for future hours)
- `is_gap` (BooleanType)
- `hour_of_day` (IntegerType, 0–23, Budapest local time)
- `day_of_week` (IntegerType, 0=Monday–6=Sunday)
- `month` (IntegerType, 1–12)
- `quarter` (IntegerType, 1–4)
- `is_weekend` (BooleanType)
- `is_holiday` (BooleanType, Hungarian public holidays via `holidays` package)
- `is_holiday_eve` (BooleanType)
- `days_since_epoch` (IntegerType, days since 2015-01-01)
- `lag_24h` (DoubleType, nullable)
- `lag_48h` (DoubleType, nullable)
- `lag_168h` (DoubleType, nullable)
- `has_lag_gap` (BooleanType)
- `rolling_7d_mean` (DoubleType, nullable)
- `rolling_7d_std` (DoubleType, nullable)
- `rolling_24h_mean` (DoubleType, nullable)
- `temperature_c` (DoubleType, nullable)
- `temperature_lag_24h` (DoubleType, nullable)
- `is_temp_imputed` (BooleanType)
- `temp_missing` (BooleanType)
- `feature_built_at` (TimestampType)
- `run_id` (StringType)

### Canonical Feature Column List (use EXACTLY this for model input)
```python
FEATURE_COLS = [
    'temperature_c', 'lag_24h', 'lag_48h', 'lag_168h',
    'rolling_7d_mean', 'rolling_7d_std', 'rolling_24h_mean',
    'hour_of_day', 'day_of_week', 'month',
    'is_weekend', 'is_holiday'
]
```

### Gold Forecasts: `workspace.energy_forecasting.gold_forecasts`
- `forecast_id` (StringType, PK — MD5 hash, never uuid4)
- `timestamp` (TimestampType)
- `forecast_run_at` (TimestampType)
- `model_name` (StringType, e.g. "energy_lgbm_24h")
- `model_version` (StringType)
- `horizon_hours` (IntegerType, 24 or 168)
- `predicted_mwh` (DoubleType)
- `actual_mwh` (DoubleType, nullable — backfilled retroactively)
- `is_backfilled` (BooleanType)
- `pipeline_run_id` (StringType)
- `created_at` (TimestampType)

### Drift Control: `workspace.energy_forecasting.drift_control`
See 03_drift_check.py for full schema.
Key columns used by other notebooks:
- `check_timestamp`, `any_drift_detected`, `consecutive_drift_hours`,
  `retrain_triggered`, `drifted_features`

### Model Evaluation: `workspace.energy_forecasting.model_evaluation`
- `run_id`, `model_name`, `horizon_hours`, `mae`, `rmse`, `mape`,
  `training_rows`, `test_rows`, `trained_at`, `promoted` (BooleanType)

### Promotion Log: `workspace.energy_forecasting.promotion_log`
Full audit trail of all promotion decisions. See 08_promote_model.py.

## MLflow Conventions
- **Model naming**: `energy_{model_type}_{horizon}h`
  Examples: `energy_lgbm_24h`, `energy_lgbm_168h`,
            `energy_prophet_24h`, `energy_prophet_168h`
- **Lifecycle**: Champion/Challenger. New models enter "None"/"Staging" stage.
  Promotion to "Production" requires 1% relative MAPE improvement.
- **Required tags on every training run**:
    `training_data_end`: ISO timestamp of last row in training set
    `training_data_start`: ISO timestamp of first row in training set
- **Reference windows**: 03_drift_check.py reads `training_data_end` tag
  from the current Production model run to define the drift reference window.
- **MLflow API note**: Use stage-based API (`get_latest_versions`, 
  `transition_model_version_stage`) for compatibility with Databricks Free Edition.
  Do NOT use aliases unless confirmed available.

## Flag File Convention
Retraining trigger: `/Volumes/workspace/energy_forecasting/data/flags/retrain_requested.flag`
Content: JSON with keys: triggered_at, reason, consecutive_drift_hours, drifted_features
Written by: 03_drift_check.py
Read and deleted by: 08_promote_model.py
If file exists when 03 runs: skip writing (retrain already pending)