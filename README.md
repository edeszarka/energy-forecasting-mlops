# Energy Consumption Forecasting — MLOps Pipeline on Databricks

End-to-end forecasting pipeline for Hungarian electricity consumption using live ENTSO-E data, Databricks, Delta Lake, and MLflow. Hourly predictions, weekly retraining with drift signals, CI/CD via GitHub Actions.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Databricks](https://img.shields.io/badge/Platform-Databricks-orange.svg)](https://www.databricks.com/)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://img.shields.io/badge/coverage-89%25-brightgreen)]()

> This project is a portfolio demonstration. Forecasts are not intended for operational grid management or energy trading decisions.

## Architecture

This project implements a **Split Ingestion Architecture** to overcome the outbound internet restrictions of the Databricks Free Edition.

```
[ GitHub Actions ] (Internet Access)
       │
       ├───> [ ENTSO-E API ] --------> (Hourly Load Data)
       │
       ├───> [ OpenMeteo API ] ------> (Budapest Temperature, Humidity, Cloud Cover)
       │
       ├───> [ Local Python Script ] -> (Slice into hourly .json files)
       │
       └───> [ Databricks CLI ] ------> (Upload to UC Volume: /raw_ingestion/)
                                              │
                                              ▼
                                      [ Databricks Workspace ] (No Internet)
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    │                                                   │
       [ energy_hourly_pipeline ]                   [ energy_retraining_pipeline ]
       (Every hour at :05 UTC)                      (Sundays 02:00 UTC)
                    │                                                   │
        ┌───────────┼───────────┐                       ┌───────────────┼───────────────┐
        ▼           ▼           ▼                       ▼               ▼               ▼
  01_ingest → 02_transform → 03_drift_check     05_train_prophet   06_train_lgbm    (parallel)
        │           │           │                       │               │               │
        └───────────┴───────────┤                       └───────┬───────┘               │
                                ▼                               ▼                       ▼
                          04_predict                     07_evaluate ←─────────────────┘
                           (Gold)                              │
                                                               ▼
                                                         08_promote_model
```

The pipeline is split into two independent Databricks Workflows: the **Hourly Job** (ingest → transform → drift check → predict) and the **Retraining Job** (train → evaluate → promote).

### Reliability Design
- **Separation of Concerns**: GitHub Actions handles all external connectivity, while Databricks remains an air-gapped environment focused on scalable processing and modeling.
- **Persistence Layer**: Unity Catalog Volumes act as the landing zone for raw data, ensuring a clear audit trail and enabling easy backfills.
- **Idempotency**: Bronze, silver, and gold forecast writes use `MERGE INTO`; control and audit tables use append-only writes. Forecast rows use a deterministic `forecast_id` (MD5 hash) — job retries never create duplicates.

## Data

### Source
- **ENTSO-E Transparency Platform** ([transparency.entsoe.eu](https://transparency.entsoe.eu))
  - Bidding zone: Hungary (10YHU-MAVIR----U)
  - Metric: Actual Total Load, hourly resolution, MWh
  - Access: public REST API
- **OpenMeteo** ([open-meteo.com](https://open-meteo.com))
  - Budapest hourly 2m temperature, humidity, cloud cover
  - Used as primary external regressor for load forecasting

### Why temperature matters
In the Hungarian energy market, electricity consumption is highly sensitive to ambient temperature due to the significant penetration of electric heating (in winter) and air conditioning (in summer). Temperature is the single most important external regressor.

## Delta Tables — Medallion Architecture

| Layer | Table | Schema | Key Columns |
|---|---|---|---|
| **Bronze** | `bronze_load` | 8 cols | timestamp, value_mwh, country, source, fetched_at, run_id, is_gap |
| **Bronze** | `bronze_temperature` | 8 cols | timestamp, temperature_c, humidity_pct, cloud_cover_pct, is_temp_imputed, source, fetched_at, run_id |
| **Silver** | `silver_features` | 29 cols | 18 engineered features + metadata (timestamp, value_mwh, is_gap, feature_built_at, run_id) |
| **Gold** | `gold_forecasts` | 11 cols | forecast_id (MD5 PK), timestamp, model_name, predicted_mwh, actual_mwh, is_backfilled, horizon_hours |
| **Control** | `drift_control` | 16 cols | check_timestamp, data_drift_detected, prediction_drift_detected, consecutive_drift_hours, retrain_triggered |
| **Control** | `model_evaluation` | 14 cols | model_name, horizon_hours, challenger/champion metrics (MAE, RMSE, MAPE), challenger_wins, promoted |
| **Control** | `promotion_log` | 15 cols | promotion_id, model_name, challenger/champion run_ids, MAPE values, promotion_reason, drift_triggered |
| **Audit** | `ingestion_log` | 9 cols | run_id, run_date, files_found/missing, rows_ingested, null_count, dry_run |

## Features — 18 Engineered Columns

| Category | Features |
|---|---|
| **Calendar** | hour_of_day, day_of_week, month, quarter, is_weekend, is_holiday, is_holiday_eve, days_since_epoch |
| **Lags** | lag_24h, lag_48h, lag_168h, has_lag_gap |
| **Rolling** | rolling_7d_mean, rolling_7d_std, rolling_24h_mean |
| **Weather** | temperature_c, temperature_lag_24h, humidity_pct, cloud_cover_pct, is_temp_imputed, temp_missing |

The 12 features used for model input: temperature_c, lag_24h, lag_48h, lag_168h, rolling_7d_mean, rolling_7d_std, rolling_24h_mean, hour_of_day, day_of_week, month, is_weekend, is_holiday.

## Models

| Model | Horizon | Strategy | Key Features | MLflow Name |
|---|---|---|---|---|
| LightGBM | 24h | Direct (shift-24 target) | lag features, temp, calendar | energy_lgbm_24h |
| LightGBM | 168h | Direct (shift-168 target) | lag features, temp, calendar | energy_lgbm_168h |
| Prophet | 24h | Built-in decomposition | temp regressor, HU holidays | energy_prophet_24h |
| Prophet | 168h | Built-in decomposition | temp regressor, HU holidays | energy_prophet_168h |

LightGBM training uses Optuna hyperparameter optimization with 25 trials by default. Each trial is evaluated with custom time-ordered 3-fold cross-validation and a 168-hour gap between train and validation data to reduce leakage.

### Champion/Challenger Pattern
Every retraining run produces a "Challenger" model. The `07_evaluate` notebook compares the Challenger's MAPE against the current "Production" model ("Champion"). A Challenger is promoted only if it achieves >1% relative MAPE improvement. The `08_promote_model` notebook tags the winning run with `production=true` (run-based MLOps pattern, since Free Edition blocks `mlflow.register_model()`).

## MLOps Design

### Drift Detection
Drift monitoring uses Evidently AI's `DataDriftPreset` in `03_drift_check`. It detects statistical shifts in features and target distributions. If drift persists for 3 consecutive hours, `03_drift_check` writes a durable, deduplicated retrain flag to a Volume, subject to a 24-hour cooldown. The flag is consumed by `08_promote_model.py` for audit metadata; retraining itself runs on the fixed weekly schedule.

### CI/CD

| Workflow | Trigger | Steps |
|---|---|---|
| **ci.yml** | PR to main, push to feature/* | `ruff check` → `mypy src/` → `pytest --cov-fail-under=80` |
| **deploy.yml** | Push to main | Validate + deploy Databricks Asset Bundle to dev & prod |
| **ingestion_hourly.yml** | Cron `5 * * * *`, manual | Fetch ENTSO-E + OpenMeteo → segment → upload to UC Volumes → trigger Databricks job |

## Repository Structure

```
energy-forecasting-mlops/
├── .github/
│   └── workflows/
│       ├── ci.yml                  # lint + type-check + unit tests
│       ├── deploy.yml              # DAB deploy to dev + prod
│       └── ingestion_hourly.yml    # API acquisition (Internet Bridge)
├── databricks.yml                  # Asset Bundle: 2 jobs, 2 environments
├── pyproject.toml                  # Project metadata, dependencies, tool config
├── requirements.txt                # Pinned deps for Databricks %pip install
├── notebooks/
│   ├── 01_ingest.py                # Volume → Bronze (MERGE INTO)
│   ├── 02_transform.py             # Bronze → Silver (feature engineering)
│   ├── 03_drift_check.py           # Evidently AI drift monitoring
│   ├── 04_predict.py               # Silver → Gold (batch inference)
│   ├── 05_train_prophet.py         # Prophet 24h + 168h training
│   ├── 06_train_lgbm.py            # LightGBM 24h + 168h training
│   ├── 07_evaluate.py              # Champion/Challenger comparison
│   └── 08_promote_model.py         # Tag-based promotion + audit log
├── src/
│   ├── __init__.py
│   ├── api_client.py               # ENTSO-E + OpenMeteo HTTP clients
│   ├── config.py                   # Central constants, paths, thresholds
│   ├── drift.py                    # Drift result extraction + MAE logic
│   └── features.py                 # Feature engineering (18 columns)
├── tests/
│   ├── test_api_client.py          # 8 tests (HTTP mocking)
│   ├── test_features.py            # 11 tests (feature correctness)
│   ├── test_drift.py               # 3 tests (drift parsing)
│   └── test_ingest_logic.py        # 6 tests (mock Spark/Delta)
├── dashboard/
│   └── energy_forecast.sql         # 6 SQL dashboard queries
└── README.md
```

## Getting Started

### Prerequisites
- Databricks Free Edition account + Databricks CLI installed
- GitHub account with secrets: `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `ENTSO_E_API_KEY`
- Python 3.11+

### Local Development
```bash
git clone https://github.com/edeszarka/energy-forecasting-mlops.git
cd energy-forecasting-mlops
pip install -e ".[dev]"
pytest tests/ -v
ruff check .
mypy src/
```

### Deploy to Databricks
```bash
databricks bundle validate --target prod
databricks bundle deploy --target prod
```

### Pipeline Execution
- **Hourly**: GitHub Actions fetches data → uploads to UC Volumes → triggers Databricks `energy_hourly_pipeline` (ingest → transform → drift_check → predict)
- **Weekly**: `energy_retraining_pipeline` runs Sundays 02:00 UTC (train → evaluate → promote); drift signals are consumed for audit metadata during the scheduled cycle

## Dashboard

The `dashboard/energy_forecast.sql` file contains 6 queries for a Databricks SQL Dashboard:

| Panel | Purpose |
|---|---|
| Pipeline Health Counter | Ingestion freshness monitoring |
| Actual vs Forecast (24h) | Last 7 days comparison |
| 7-Day Forecast (168h) | Forward-looking strategic view |
| Rolling MAPE Table | Weekly accuracy by horizon |
| Drift Monitoring Heatmap | Drifted features over last 30 days |
| Model Registry Status | Current production models & age |

## Results

Results will be populated after the pipeline runs on live data for 7+ days. Metrics are stored in MLflow runs (`mape`, `mae`, `rmse` metrics) and the `model_evaluation` Delta table.

| Model | Horizon | Test MAPE | Test MAE (MWh) | Test RMSE (MWh) | Test Period |
|---|---|---|---|---|---|
| LightGBM | 24h | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| LightGBM | 168h | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Prophet | 24h | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Prophet | 168h | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

Published benchmarks for Hungary report 2–5% MAPE for 24h horizons.

## Test Coverage

```
src/api_client.py    89%
src/config.py       100%
src/drift.py         92%
src/features.py      82%
---------------------------------
TOTAL                89%  (threshold: 80%)
```

## Known Limitations
1. **Moveable Holidays**: Easter and other moveable holidays are approximated with fixed dates.
2. **Weather Proxies**: Future temperature uses naive persistence (same hour, 7 days ago).
3. **Lag Uncertainty**: Direct multi-step models compounding errors at horizon edges.
4. **Free Tier Quotas**: Databricks Free Edition concurrency limits may cause queuing.
5. **Drift Counter**: Reliant on successful hourly job execution without gaps.
6. **Model Registry IAM**: Free Edition blocks `mlflow.register_model()`, so run-based tag pattern used instead.

## Planned Enhancements
- Cyclical encoding (sine/cosine for hour_of_day, day_of_week)
- Residual analysis (MAPE heatmap by hour/DOW)
- Quantile regression (prediction intervals)
- Feature importance drift monitoring (SHAP over time)

## License
MIT License.
