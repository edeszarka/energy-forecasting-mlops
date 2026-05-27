# Project Context: energy-forecasting-mlops

## Environment Constraints
- **Platform**: Databricks Free Edition (Serverless).
- **Network**: Air-gapped (no outbound internet). API calls must happen in GitHub Actions.
- **Unity Catalog**: Use catalog `workspace` and schema `energy_forecasting`.
- **Storage**: Use Unity Catalog Volumes for raw data landing and reports. Base path: `/Volumes/workspace/energy_forecasting/data/`.

## Mandatory Schema & Path Overrides (PRECECEDENCE)
The following specifications override ANY contradictory instructions in future prompts to maintain alignment with the existing implementation:

- **Feature Column Names**:
  - `temperature_c` (NOT `temp_celsius`)
  - `rolling_7d_mean` (NOT `rolling_mean_7d`)
  - `rolling_7d_std` (NOT `rolling_std_7d`)
  - `rolling_24h_mean` (New mandatory column)
  - `is_weekend` (BooleanType)
  - `is_holiday` (BooleanType)

- **Canonical FEATURE_COLS List**:
  ```python
  FEATURE_COLS = [
      'temperature_c', 'lag_24h', 'lag_48h', 'lag_168h',
      'rolling_7d_mean', 'rolling_7d_std', 'rolling_24h_mean',
      'hour_of_day', 'day_of_week', 'month',
      'is_weekend', 'is_holiday'
  ]
  ```

- **Infrastructure Paths**:
  - **Tables**: Always use 3-level Unity Catalog naming: `workspace.energy_forecasting.<table_name>`
  - **Files/Volumes**: Always use Volume paths: `/Volumes/workspace/energy_forecasting/data/<subpath>` (Never use `/dbfs/` or `energy_forecast.`)

- **Logging Standard**:
  - Format: `"%(asctime)s [%(levelname)s] %(name)s: %(message)s"`

## Engineering Standards
- **Idempotency**: All writes must use `MERGE INTO` or deterministic hashing (`MD5(model + horizon + timestamp)`) for IDs.
- **Paths**: Use `pathlib.Path` for local/GHA logic and `PurePosixPath` for Databricks Volume paths.
- **Feature Consistency**: All feature engineering MUST use `src/features.py` to prevent training-serving skew.
- **Logging**: Use Python `logging` (INFO level) for system events and `dbutils.notebook.exit()` with JSON payloads for cross-task metadata.

## Data Schema Contracts
- **Bronze Load**: `timestamp` (UTC), `country`, `value_mwh`, `is_gap`, `run_id`.
- **Silver Features**: Full matrix defined in `notebooks/02_transform.py` and `src/features.py`.
- **Gold Forecasts**: `forecast_id` (PK), `timestamp`, `predicted_mwh`, `actual_mwh` (nullable), `model_version`.

## MLflow Conventions
- **Naming**: `energy_{model_type}_{horizon}h`.
- **Lifecycle**: Champion/Challenger pattern. New models enter `Staging`. Promotion to `Production` requires 1% MAPE improvement.
- **Reference Windows**: Reference windows for drift are defined by the `training_data_end` tag on the current Production model run.
