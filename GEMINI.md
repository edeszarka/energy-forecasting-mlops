# Project Context: energy-forecasting-mlops

## Environment Constraints
- **Platform**: Databricks Free Edition (Serverless).
- **Network**: Air-gapped (no outbound internet). API calls must happen in GitHub Actions.
- **Unity Catalog**: Use catalog `workspace` and schema `energy_forecasting`.
- **Storage**: Use Unity Catalog Volumes for raw data landing and reports. Base path: `/Volumes/workspace/energy_forecasting/data/`.

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
