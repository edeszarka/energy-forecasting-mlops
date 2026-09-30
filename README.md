# Energy Consumption Forecasting — MLOps Pipeline on Databricks

End-to-end forecasting pipeline for Hungarian electricity consumption using live ENTSO-E data, Databricks, Delta Lake, and MLflow. Hourly predictions, weekly retraining with drift signals, CI/CD via GitHub Actions.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Databricks](https://img.shields.io/badge/Platform-Databricks-orange.svg)](https://www.databricks.com/)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://img.shields.io/badge/coverage-86%25-brightgreen)]()

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
- **Backlog Discovery**: `01_ingest` lists the raw-ingestion Volume each run and drains up to `max_files_per_run` (job default `200`) pending files oldest-first, rather than probing a fixed filename count (spec 008). GitHub Actions fetches a `lookback_hours` window (default 24).
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
| **Bronze** | `bronze_load` | 7 cols | timestamp, country, value_mwh, source, fetched_at, run_id, is_gap |
| **Bronze** | `bronze_temperature` | 8 cols | timestamp, temperature_c, humidity_pct, cloud_cover_pct, is_weather_imputed, source, fetched_at, run_id |
| **Silver** | `silver_features` | 29 cols | 21 feature columns + 8 metadata (timestamp, country, value_mwh, is_gap, source, fetched_at, feature_built_at, run_id) |
| **Gold** | `gold_forecasts` | 11 cols | forecast_id (MD5 PK), timestamp (prediction target time, spec 006), forecast_run_at, model_name, model_version, horizon_hours, predicted_mwh, actual_mwh, is_backfilled, pipeline_run_id, created_at |
| **Control** | `drift_control` | 16 cols | check_timestamp, data_drift_detected, prediction_drift_detected, consecutive_drift_hours, retrain_triggered |
| **Control** | `model_evaluation` | 18 cols | model_name, horizon_hours, challenger/champion metrics (MAE, RMSE, MAPE), baseline metrics, beats_baseline, challenger_wins, promoted |
| **Control** | `promotion_log` | 15 cols | promotion_id, model_name, challenger/champion run_ids, MAPE values, promotion_reason, drift_triggered |
| **Audit** | `ingestion_log` | 10 cols | run_id, run_date, files_found, files_missing (backlog remaining after cap, spec 008), rows_ingested, null_count, schema_errors, dry_run, written_at, date |

## Features — 18 Engineered Columns

`get_feature_columns()` (`src/features.py`) defines the 18 feature columns:

| Category | Features |
|---|---|
| **Calendar** | hour_of_day, day_of_week, month, quarter, is_weekend, is_holiday, is_holiday_eve, days_since_epoch |
| **Lags** | lag_24h, lag_48h, lag_168h |
| **Rolling** | rolling_7d_mean, rolling_7d_std, rolling_24h_mean |
| **Weather** | temperature_c, temperature_lag_24h, humidity_pct, cloud_cover_pct |

`silver_features` additionally stores the QA flags `has_lag_gap`, `is_weather_imputed`, and `temp_missing`.

The 14 features used for model input (`src/config.py` `MODEL_INPUT_FEATURES`): temperature_c, lag_24h, lag_48h, lag_168h, rolling_7d_mean, rolling_7d_std, rolling_24h_mean, hour_of_day, day_of_week, month, is_weekend, is_holiday, humidity_pct, cloud_cover_pct.

## Models

| Model | Horizon | Strategy | Key Features | MLflow Name |
|---|---|---|---|---|
| LightGBM | 24h | Direct t+24h target (time-aligned, spec 007) | lag features, temp, calendar | energy_lgbm_24h |
| LightGBM | 168h | Direct t+168h target (time-aligned, spec 007) | lag features, temp, calendar | energy_lgbm_168h |
| Prophet | 24h | Built-in decomposition | temperature regressor (+ built-in seasonality) | energy_prophet_24h |
| Prophet | 168h | Built-in decomposition | temperature regressor (+ built-in seasonality) | energy_prophet_168h |

LightGBM training uses Optuna hyperparameter optimization with 25 trials by default. Each trial is evaluated with a custom time-ordered 3-fold cross-validation, a 168-hour gap between train and test folds, and a 120-row test window per fold; the search covers eight parameters including the objective (`regression`, `regression_l1`, `huber`). Tuning requires at least 984 training rows (`3 × (120 + 168) + 120`).

### Champion/Challenger Pattern
Every retraining run produces a "Challenger" model. The `07_evaluate` notebook compares the Challenger's MAPE against the current "Production" model ("Champion"). A Challenger is promoted only if it achieves >1% relative MAPE improvement. The `08_promote_model` notebook tags the winning run with `production=true` (run-based MLOps pattern, since Free Edition blocks `mlflow.register_model()`).

## MLOps Design

### Drift Detection
Drift monitoring uses Evidently AI's `DataDriftPreset` in `03_drift_check`. It detects statistical shifts in features and target distributions. If drift persists for 3 consecutive hours, `03_drift_check` writes a durable, deduplicated retrain flag to a Volume, subject to a 24-hour cooldown. The flag is consumed by `08_promote_model.py` for audit metadata; retraining itself runs on the fixed weekly schedule.

### CI/CD

| Workflow | Trigger | Steps |
|---|---|---|
| **ci.yml** | PR to main; push to `feature/*` or `prediction_dashboard` | `ruff check` → `mypy src/` → `pytest --cov-fail-under=80` |
| **deploy.yml** | Push to main | Validate Bundle (`--target prod`), then deploy prod, then deploy dev — no staging gate (both targets on the same push) |
| **ingestion_hourly.yml** | Cron `5 * * * *`, manual | Fetch ENTSO-E + OpenMeteo (`lookback_hours`, default 24) → segment → upload to UC Volumes → trigger Databricks job |

## Repository Structure

```
energy-forecasting-mlops/
├── .github/
│   └── workflows/
│       ├── ci.yml                  # lint + type-check + unit tests
│       ├── deploy.yml              # DAB deploy to prod + dev
│       └── ingestion_hourly.yml    # API acquisition (Internet Bridge)
├── .streamlit/
│   └── example_secrets.toml        # Streamlit secrets template
├── .pre-commit-config.yaml         # ruff + ruff-format hooks
├── databricks.yml                  # Asset Bundle: 2 jobs, 2 targets
├── pyproject.toml                  # Project metadata, dependencies, tool config
├── requirements.txt                # Pinned deps for Databricks %pip install
├── notebooks/
│   ├── 01_ingest.py                # Volume → Bronze (backlog discovery, MERGE INTO)
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
│   ├── baseline.py                 # Naive baseline metrics
│   ├── config.py                   # Central constants, paths, thresholds
│   ├── dashboard.py                # Streamlit dashboard
│   ├── drift.py                    # Drift result extraction + MAE logic
│   ├── features.py                 # Feature engineering (18 columns)
│   ├── forecast_timing.py          # Inference anchor → target timestamp (spec 006)
│   ├── ingest_discovery.py         # Volume backlog selection (spec 008)
│   ├── splits.py                   # Train/val/test + rolling-origin splits
│   ├── target_construction.py      # Time-aligned horizon target (spec 007)
│   └── tuning.py                   # Optuna search space + CV
├── tests/                          # 103 pytest tests across 12 files
├── specs/                          # Design specs 001–008 (+005b)
├── dashboard/
│   └── energy_forecast.sql         # 7 SQL dashboard queries
├── GEMINI.md                       # Developer/agent working notes
└── README.md
```

## Specs

Design specs live in `specs/`:

| # | Title |
|---|---|
| 001 | Ingestion Gap Resilience |
| 002 | Naive Baseline Evaluation |
| 003 | LightGBM Training-Objective Tuning |
| 004 | Train/Validation/Test Split Integrity |
| 005 | Prophet Horizon Differentiation |
| 005b | Prophet Rolling-Origin Horizon Evaluation |
| 006 | Gold Forecast Timestamp Offset Fix |
| 007 | LightGBM Target Construction: Positional Shift vs. Time-Based Shift |
| 008 | Ingestion Volume Backlog Discovery (Replace Fixed-Count File Probing) |

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

The `dashboard/energy_forecast.sql` file contains 7 queries for a Databricks SQL Dashboard:

| Panel | Purpose |
|---|---|
| Pipeline Health Counter | Ingestion freshness monitoring |
| Actual vs Forecast (24h) | Last 7 days comparison |
| 7-Day Forecast (168h) | Forward-looking strategic view |
| Rolling MAPE Table | Weekly accuracy by horizon |
| Drift Monitoring Heatmap | Drifted features over last 30 days |
| Model Registry Status | Current production models & age |
| Retraining History Timeline | Promotion audit trail (last 90 days) |

An interactive Streamlit dashboard is also available at `src/dashboard.py` (install the optional extra with `pip install -e ".[dashboard]"`; secrets via `.streamlit/secrets.toml`).

## Results

Metrics are stored in MLflow runs (`mape`, `mae`, `rmse`) and the `model_evaluation` Delta table.

| Model | Horizon | Test MAPE | Test MAE (MWh) | Test RMSE (MWh) | Test Period |
|---|---|---|---|---|---|
| LightGBM | 24h | withdrawn | — | — | — |
| LightGBM | 168h | withdrawn | — | — | — |
| Prophet | 24h | 15.70% | 821.23 | 885.96 | 2026-08-19 – 2026-08-23 |
| Prophet | 168h | 51.75% | 2658.98 | 2666.76 | 2026-08-19 – 2026-08-23 |

> **LightGBM metrics withdrawn.** Before spec 007 (merged 2026-09-27), the LightGBM target was built with an order-dependent positional `shift(-horizon_hours)`, which paired rows with the wrong hour's value (target leakage). LightGBM metrics and MLflow runs from before that date are therefore invalid and not comparable to any other result; re-evaluation is pending the next retraining. Prophet (spec 005b) does not use `shift()` and is unaffected.

Prophet metrics are from a 5-fold rolling-origin backtest (spec 005b) and are not directly comparable to other evaluation methodologies.

## Test Coverage

```
src/api_client.py           89%
src/baseline.py             92%
src/config.py              100%
src/dashboard.py            59%
src/drift.py                92%
src/features.py             92%
src/forecast_timing.py     100%
src/ingest_discovery.py    100%
src/splits.py              100%
src/target_construction.py 100%
src/tuning.py              100%
---------------------------------
TOTAL                       86%  (threshold: 80%)
```

## Known Limitations
1. **Weather Proxies**: Future temperature uses naive persistence (same hour, 7 days ago).
2. **Lag Uncertainty**: Direct multi-step models compounding errors at horizon edges.
3. **Free Tier Quotas**: Databricks Free Edition concurrency limits may cause queuing.
4. **Drift Counter**: Reliant on successful hourly job execution without gaps. The 2026-09-19→09-23 Databricks outage stalled the counter; spec 008's backlog discovery now drains the accumulated raw-ingestion Volume backlog oldest-first (`max_files_per_run=200`), which self-heals ingestion after a gap.
5. **Model Registry IAM**: Free Edition blocks `mlflow.register_model()`, so run-based tag pattern used instead.
6. **Optuna Row Floor**: Tuning requires ≥ 984 training rows (`3 × (120 + 168) + 120`); the 168h training set can currently dip below that and abort the `train_lgbm` task.
7. **Historical Feature Gaps**: Older `silver_features` rows carry NaN lag/rolling features (roughly 2026-05-01 → 2026-08-25), leaving only ~1k rows usable for training after `dropna`.
8. **No Staging Gate**: `deploy.yml` deploys prod then dev on the same push to main, with no staging/approval gate.
9. **Leak-Affected Production Models**: as of 2026-09-28, the currently production-tagged LightGBM models were also trained under the spec-007 leakage and the champion/challenger gate cannot currently promote an honest challenger over their artificially low MAPE. Remediation is tracked separately.

## Planned Enhancements
- Cyclical encoding (sine/cosine for hour_of_day, day_of_week)
- Residual analysis (MAPE heatmap by hour/DOW)
- Quantile regression (prediction intervals)
- Feature importance drift monitoring (SHAP over time)

## License
MIT License.
