# Energy Consumption Forecasting — MLOps Pipeline on Databricks

End-to-end forecasting pipeline for Hungarian electricity consumption using live ENTSO-E data, Databricks, Delta Lake, and MLflow. Hourly predictions, live error monitoring against seasonal naive, error-triggered retraining (plus a weekly safety-net run), CI/CD via GitHub Actions.

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
                              │                                ▼
                       09_error_monitor                  08_promote_model
```

The pipeline is split into two independent Databricks Workflows: the **Hourly Job** (ingest → transform → drift check → predict → error monitor) and the **Retraining Job** (train → evaluate → promote).

### Backfill Path
Historical data takes its own path ([ADR-0005](docs/adr/0005-separate-backfill-path.md)), so re-ingesting years of data neither floods the hourly Volume nor blocks hourly ingestion:

```
[ backfill.yml ] (manual; inputs start_date, end_date — end exclusive; own concurrency group)
       │  ENTSO-E in 7-day chunks, Open-Meteo Historical Forecast API in monthly chunks
       ▼
[ UC Volume: /raw_backfill/{load,temperature}/ ]  one JSON file per chunk
       │
       ▼
[ 00_backfill_bronze ] (manual; mode = dry_run | execute)
       │  monthly row counts before/after, data-quality checks always enforced,
       │  replaceWhere on [start, end) — NULLs replace stored values on this path
       ▼
[ bronze_load, bronze_temperature ]
       │
       ▼
[ 02_transform, rebuild mode ] (rebuild_start, rebuild_end) → silver in 30-day chunks
```

The hourly path keeps its incremental `MERGE`, where a NULL never overwrites a stored load value. Fetching logic lives in `src/backfill.py` (`python -m src.backfill --start … --end … --out …`); the workflow only runs it and uploads the files.

### Data Versioning
Every model is tied to the data it was trained on ([ADR-0003](docs/adr/0003-data-versioning-of-models.md)). The silver table carries the Delta table property `pipeline.data_version`; only a full rebuild from verified bronze bumps it (version `2` since spec 014; a table without the property is version `1`). `05`/`06` tag each run with the version they read (`data_version`), and serving (`04`), evaluation (`07`), promotion (`08`) and the drift reference window (`03`) only consider runs whose tag equals `CURRENT_DATA_VERSION` in `src/config.py`. Older runs are marked once by `scripts/remediate_014_data_version.py` (`data_version=1`, `invalidated_reason`, `production=false`; dry run by default).

### Data-Quality Checks
Load data must pass `src/data_quality.py` before it is written ([ADR-0004](docs/adr/0004-data-quality-checks-fail-the-pipeline.md)):
- 24 distinct hourly rows per complete UTC day, no duplicate or off-hour timestamps;
- non-null values within `[LOAD_MIN_MW, LOAD_MAX_MW]`;
- a daily cycle: the correlation with the value 24h earlier is at least `DQ_MIN_CORR_24H` and higher than the correlation with the value 12h earlier, on hour-aligned pairs only.

`00_backfill_bronze` always enforces the checks, per calendar month. `02_transform` runs them on the silver frame before writing, controlled by `dq_mode`: `warn` logs violations, `enforce` fails the task and writes nothing. The thresholds in `src/config.py` are provisional until they are calibrated on the rebuilt silver; `databricks.yml` sets `dq_mode: warn` until then.

### Reliability Design
- **Separation of Concerns**: GitHub Actions handles all external connectivity, while Databricks remains an air-gapped environment focused on scalable processing and modeling.
- **Persistence Layer**: Unity Catalog Volumes act as the landing zone for raw data, ensuring a clear audit trail and enabling easy backfills.
- **Backlog Discovery**: `01_ingest` lists the raw-ingestion Volume each run and drains up to `max_files_per_run` (job default `200`) pending files oldest-first, rather than probing a fixed filename count (spec 008). GitHub Actions fetches a `lookback_hours` window (default 24).
- **Idempotency**: Bronze, silver, and gold forecast writes use `MERGE INTO`; control and audit tables use append-only writes. Forecast rows use a deterministic `forecast_id` (MD5 hash) — job retries never create duplicates.

## Data

### Source
- **ENTSO-E Transparency Platform** ([transparency.entsoe.eu](https://transparency.entsoe.eu))
  - Bidding zone: Hungary (10YHU-MAVIR----U)
  - Metric: Actual Total Load, MWh. PT60M responses are used as is; PT15M responses are aggregated to hourly means (an hour with fewer than 4 quarters is left empty). Every `Period` of every `TimeSeries` is read; curve type `A03` (an omitted position repeats the previous value) is expanded before aggregation. Any other resolution, a curve type other than `A01`/`A03` (or none), or a unit other than `MAW` fails the ingestion run (specs 013, 014).
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
| **Gold** | `gold_forecasts` | 12 cols | forecast_id (MD5 PK), timestamp (prediction target time, spec 006), forecast_run_at, model_name, model_version, horizon_hours, predicted_mwh, actual_mwh, is_backfilled, pipeline_run_id, created_at, is_served (spec 017; NULL on older rows = served) |
| **Control** | `drift_control` | 16 cols | check_timestamp, data_drift_detected, prediction_drift_detected, consecutive_drift_hours, retrain_triggered |
| **Control** | `forecast_error_monitor` | 13 cols | check_timestamp, window_hours (24/168/720), stream (served/naive/shadow_lgbm), model_name, n, coverage, mape, mae, bias, naive_mape_paired, relative_skill, alerts, created_at |
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

| Model | Horizon | Role | Strategy | Key Features | MLflow Name |
|---|---|---|---|---|---|
| LightGBM | 24h | Served (promotable) | Direct t+24h target (time-aligned, spec 007) | lag features, temp, calendar | energy_lgbm_24h |
| Prophet | 24h | Benchmark only | Built-in decomposition | temperature regressor (+ built-in seasonality) | energy_prophet_24h |
| Seasonal naive | 24h | Serving fallback | `predicted(T) = actual(T − 168h)` | — | `seasonal_naive` (gold only) |

Only the 24h horizon is forecast; the 168h horizon was dropped ([ADR-0001](docs/adr/0001-drop-168h-horizon.md)). Prophet is trained and evaluated for reporting but never promoted or served ([ADR-0002](docs/adr/0002-prophet-benchmark-only-naive-fallback.md)). The model roles live in `src/config.py` (`PROMOTABLE_MODELS`, `BENCHMARK_MODELS`).

LightGBM training uses Optuna hyperparameter optimization with 25 trials by default. Each trial is evaluated with a custom time-ordered 3-fold cross-validation, a 168-hour gap between train and test folds, and a 120-row test window per fold; the search covers eight parameters including the objective (`regression`, `regression_l1`, `huber`). Tuning requires at least 984 training rows (`3 × (120 + 168) + 120`).

### Champion/Challenger Pattern
Every retraining run produces a "Challenger" model. The `07_evaluate` notebook loads the run currently tagged `production=true` ("Champion") and scores **both** models on the same rows — the Challenger's logged backtest rows whose targets the Champion never saw during fitting (`served_data_end`) — so a harder post-drift window cannot bias the comparison. A Challenger is promoted only if it achieves >1% relative MAPE improvement; otherwise the Champion stays and nothing changes in production. Runs from before spec 011 have no backtest artifact and fall back to logged metrics (`comparison_method = logged_metrics`). The `08_promote_model` notebook tags the winning run with `production=true` (run-based MLOps pattern, since Free Edition blocks `mlflow.register_model()`). Benchmark models still get a `model_evaluation` row, but `challenger_wins` is always false and 08 refuses to tag them, even with `force_promote`. When no production run of the current data version exists (`first_run`), a Challenger is promoted only if its MAPE is below the naive baseline's MAPE on the same test rows; otherwise serving stays on seasonal naive (spec 014).

### Serving and Fallback
`04_predict` loads only the `energy_lgbm_24h` run tagged `production=true` (and `leak_affected=false`, `data_version` = `CURRENT_DATA_VERSION`), chosen by the same rule as `07`/`08` (latest `promoted_at`). It never falls back to an unpromoted run. If there is no production run, or loading, feature building or prediction fails, it logs a warning and writes a seasonal-naive forecast instead: `model_name = 'seasonal_naive'`, `model_version = 'seasonal_naive_t-168h'`, for the same target hours as LightGBM (run+25h … run+48h). Targets whose T−168h value is missing in silver are skipped; if none can be produced while naive serves, the task fails.

Since spec 017 ([ADR-0006](docs/adr/0006-live-error-monitoring-and-shadow-scoring.md)) the seasonal-naive rows are written **every hour**, and `is_served` marks which rows were served. While naive serves, the latest `energy_lgbm_24h` run of the current data version is scored in **shadow**: its rows go to gold with `is_served = false` and `model_version = <run_id>`, through the same feature path serving would use, and are never served. A shadow failure only logs a warning. Each target hour keeps its first forecast (insert-only MERGE), which is issued 48h ahead in normal operation.

## MLOps Design

### Drift Diagnostics
`03_drift_check` runs Evidently AI's `DataDriftPreset` on the target and non-calendar features (calendar columns such as `month` differ between any two windows by construction and are excluded) and stores the result in `drift_control` with an HTML report. **It never starts a retrain** (spec 018, [ADR-0007](docs/adr/0007-error-triggered-retraining.md)): between a 7-day current window and a 30-day reference it flagged drift on every hourly check for 60 days, almost always on every feature at once. That is the season changing, not a model failing. Drift tells you *what* changed; the error monitor decides *whether* it matters.

### Live Error Monitoring
`09_error_monitor` runs after `04_predict` (which backfills actuals) and scores three streams over the last 24h, 7d and 30d of target hours: the **served** forecast, **seasonal naive** and the **shadow LightGBM** run. Each model is also compared with naive on exactly the same target hours: `relative_skill = 1 − model MAPE / naive MAPE` (> 0 means it beats naive). MAPE is the mean of `|actual − predicted| / actual`, the same definition as training's naive metric. One row per window and stream goes to `forecast_error_monitor`.

Why the comparison and not just a MAPE target: seasonal naive on the rebuilt silver (2024-01 → 2026-10) has a weekly MAPE of 7.28% (median), 13.21% (p90) and 22.53% at worst (Easter Monday week 2026). A fixed target would fire on every holiday week.

Alerts, on the 7-day window (thresholds in `src/config.py`, provisional for the first month):

| Alert | Condition | Fails the task |
|---|---|---|
| `SERVED_MAPE_HIGH` | served MAPE > 25% (above the worst naive week since 2024) | yes |
| `SERVED_WORSE_THAN_NAIVE` | a served model's `relative_skill` < 0 | yes |
| `LOW_COVERAGE` | fewer than 80% of the window's hours scored; suppresses the two above | yes |
| `SHADOW_BEATS_NAIVE` | shadow `relative_skill` ≥ 0.05 on ≥ 120 hours: evidence for a promotion review | no |

A failing alert fails the task **once, when it starts** (it was not active at the previous check), after the rows are written; the hourly job's failure email is the alert channel. It does not fail again while the alert persists. The old LightGBM-only prediction-MAE check in `03_drift_check` is retired; its `drift_control` columns are written empty, and the Streamlit MAE tab shows N/A.

### Error-Triggered Retraining
`09_error_monitor` also decides whether to retrain (spec 018). The served LightGBM run is judged **only on the target hours it served itself** (gold rows carry its run id as `model_version`), paired with seasonal naive on the same hours, so a freshly promoted run is never blamed for its predecessor's errors. The run is *degraded* when, on at least 72 of its own hours in the 7-day window, it is worse than naive (`relative_skill < 0`) or its MAPE is above 25%. After 24 consecutive degraded hourly checks (a missing hour breaks the streak) it starts `energy_retraining_pipeline` via the Jobs API (fire-and-forget, so hourly forecasts are never blocked), subject to a 72-hour cooldown, and writes a retrain flag that marks the request as pending. `08_promote_model.py` deletes the flag at the end of a successful retrain; a flag older than 6 hours is treated as stale so a failed retrain cannot block future triggers. A failure to start the job is recorded and retried the next hour; it never fails the task. The decision is stored on the served 7-day monitor row (`degraded`, `consecutive_degraded_hours`, `retrain_triggered`, `retrain_reason`).

Why error and not drift: retraining fixes **concept drift**, a change in the relationship between inputs and load (rooftop solar lowering midday net load, a large consumer appearing or leaving). **Data drift** alone, such as autumn cooling, is a shift the model has already seen in two years of history. Concept drift cannot be read from the inputs, only from a larger error once actuals arrive, so the error trigger is the practical concept-drift detector. It also catches inputs moving outside the training range, where a retrain helps as well. Error caused by bad data is stopped earlier by the enforced data-quality checks and the promotion gate, and the failure email asks for a human look. Slow concept drift that never makes the model lose to naive for a full day is absorbed by the weekly retrain ([ADR-0007](docs/adr/0007-error-triggered-retraining.md)).

While seasonal naive serves there is no model to judge, so the Sunday 02:00 UTC run is the only retrain. All thresholds are in `src/config.py` and provisional for the first month.

### CI/CD

| Workflow | Trigger | Steps |
|---|---|---|
| **ci.yml** | PR to main; push to `feature/*` or `prediction_dashboard` | `ruff check` → `mypy src/` → `pytest --cov-fail-under=80` |
| **deploy.yml** | Push to main | Validate Bundle (`--target prod`), then deploy prod, then deploy dev — no staging gate (both targets on the same push) |
| **ingestion_hourly.yml** | Cron `5 * * * *`, manual | Fetch ENTSO-E + OpenMeteo (`lookback_hours`, default 24) → segment → upload to UC Volumes → trigger Databricks job |
| **backfill.yml** | Manual (`start_date`, `end_date`) | Fetch history in chunks → upload to `/raw_backfill/` (no job trigger; load with `00_backfill_bronze`) |

## Repository Structure

```
energy-forecasting-mlops/
├── .github/
│   └── workflows/
│       ├── backfill.yml            # Manual historical fetch in chunks (ADR-0005)
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
│   ├── 00_backfill_bronze.py       # Manual: backfill files → Bronze (replaceWhere, DQ enforced)
│   ├── 01_ingest.py                # Volume → Bronze (backlog discovery, MERGE INTO)
│   ├── 02_transform.py             # Bronze → Silver (feature engineering, rebuild mode, dq_mode)
│   ├── 03_drift_check.py           # Evidently AI drift diagnostics (never triggers, spec 018)
│   ├── 04_predict.py               # Silver → Gold (production LightGBM or seasonal-naive fallback)
│   ├── 05_train_prophet.py         # Prophet 24h benchmark training
│   ├── 06_train_lgbm.py            # LightGBM 24h training
│   ├── 07_evaluate.py              # Champion/Challenger comparison
│   ├── 08_promote_model.py         # Tag-based promotion + audit log
│   └── 09_error_monitor.py         # Live MAPE, alerts (spec 017), retrain trigger (spec 018)
├── src/
│   ├── __init__.py
│   ├── api_client.py               # ENTSO-E + OpenMeteo HTTP clients
│   ├── backfill.py                 # Chunked historical fetch for backfill.yml (spec 014)
│   ├── baseline.py                 # Naive baseline metrics + seasonal-naive fallback
│   ├── bronze_write.py             # Bronze MERGE columns + backfill preparation (spec 014)
│   ├── config.py                   # Central constants, paths, thresholds
│   ├── dashboard.py                # Streamlit dashboard
│   ├── data_quality.py             # Load data-quality checks, dq_mode (spec 014)
│   ├── data_version.py             # Silver data version + run remediation plan (spec 014)
│   ├── drift.py                    # Drift extraction, streak/cooldown trigger rules (spec 011)
│   ├── error_monitoring.py         # Live error metrics, paired naive comparison, alerts (spec 017)
│   ├── error_trigger.py            # Served-run degradation and retrain decision (spec 018)
│   ├── features.py                 # Feature engineering (18 columns)
│   ├── forecast_timing.py          # Inference anchor → target timestamp (spec 006)
│   ├── gold_rows.py                # is_served rows + naive gold history (spec 017)
│   ├── ingest_discovery.py         # Volume backlog selection (spec 008)
│   ├── promotion.py                # Champion/challenger comparison (spec 011)
│   ├── retrain_trigger.py          # Starts the retraining job, pending flag (spec 011, 018)
│   ├── silver_build.py             # Silver build shared by hourly path and rebuild (spec 014)
│   ├── splits.py                   # Train/val/test + rolling-origin splits
│   ├── target_construction.py      # Time-aligned horizon target (spec 007)
│   └── tuning.py                   # Optuna search space + CV
├── scripts/
│   ├── backfill_017_naive_gold.py     # One-time naive history in gold (dry run default)
│   └── remediate_014_data_version.py  # One-time MLflow tag remediation (dry run default)
├── tests/                          # pytest suite
├── specs/                          # Design specs 001–008 (+005b)
├── dashboard/
│   └── energy_forecast.sql         # 8 SQL dashboard queries
├── docs/adr/                       # Architecture decision records
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

## Decisions

Architecture decision records live in [`docs/adr/`](docs/adr/README.md).

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
- **Hourly**: GitHub Actions fetches data → uploads to UC Volumes → triggers Databricks `energy_hourly_pipeline` (ingest → transform → drift_check → predict → error_monitor)
- **On forecast degradation**: `09_error_monitor` starts `energy_retraining_pipeline` (train → evaluate → promote) after 24 consecutive hours in which the served run is worse than naive or above the MAPE ceiling
- **Weekly**: the same job also runs Sundays 02:00 UTC as a safety net

## Dashboard

The `dashboard/energy_forecast.sql` file contains 8 queries for a Databricks SQL Dashboard:

| Panel | Purpose |
|---|---|
| Pipeline Health Counter | Ingestion freshness monitoring |
| Actual vs Forecast (24h) | Last 7 days; one series per model, marked served or shadow |
| Live Forecast Error (7d) | Served vs naive vs shadow MAPE, relative skill, coverage, alerts (spec 017) |
| Active Forecast Error Alerts | Alerts of the latest monitor check |
| Seasonal-Naive Accuracy History | Weekly naive MAPE since 2024 (includes the reconstructed history) |
| Drift Monitoring Heatmap | Drifted features over last 30 days (diagnostic; never triggers retraining) |
| Model Registry Status | Current production models & age |
| Retraining History Timeline | Promotion audit trail (last 90 days) |

An interactive Streamlit dashboard is also available at `src/dashboard.py` (install the optional extra with `pip install -e ".[dashboard]"`; secrets via `.streamlit/secrets.toml`).

## Results

Metrics are stored in MLflow runs (`mape`, `mae`, `rmse`) and the `model_evaluation` Delta table.

**No results are published at the moment.** Every model and metric produced before the spec 014 cut-over was trained and evaluated on corrupted load data (see [Data Incident, October 2026](#data-incident-october-2026)). Results will be reported once a model trained on data version 2 has been evaluated against the seasonal-naive baseline. Live accuracy is measured hourly by `09_error_monitor` (see [Live Error Monitoring](#live-error-monitoring)).

The 168h horizon was dropped in spec 013 ([ADR-0001](docs/adr/0001-drop-168h-horizon.md)); its historic MLflow runs and gold rows remain but are no longer read.

> **Earlier withdrawal.** Before spec 007 (merged 2026-09-27), the LightGBM target was built with an order-dependent positional `shift(-horizon_hours)`, which paired rows with the wrong hour's value (target leakage). Those runs were already invalid before the data incident.

## Data Incident, October 2026

- **What happened.** ENTSO-E returns MAVIR actual load at 15-minute resolution (PT15M) — at least since 2026-08-03, and the 2024 history is served that way too. The parser assumed hourly data and mapped each 15-minute position to an hour, which stretched the daily load curve by a factor of four. It only logged a warning (`Unexpected resolution: PT15M`), which nobody read.
- **Impact.** Bronze and silver load history, every model trained on it, every reported metric, the drift reference windows and the gold actuals before the cut-over are invalid.
- **Detection.** The warning in every retained GitHub Actions ingestion log, and a correlation check on silver: in May, June and September 2026 the lag-24h correlation collapsed (0.02–0.22) below the lag-12h correlation (0.32–0.68), which a real daily cycle never shows; the other months were degraded.
- **Fix.** Spec 013 parses PT15M into hourly means and fails on any other resolution. Spec 014 reads every period, handles the `A03` curve type, fails on unexpected document content, re-ingests history from 2024-01-01 through the backfill path, rebuilds silver with hour-aligned features, makes data-quality checks fail the pipeline, invalidates all older models through data versioning, and requires the first clean model to beat seasonal naive before it can be served.

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
6. **Optuna Row Floor**: Tuning requires ≥ 984 training rows (`3 × (120 + 168) + 120`); a smaller training set aborts the `train_lgbm` task.
7. **Historical Feature Gaps**: Older `silver_features` rows carry NaN lag/rolling features (roughly 2026-05-01 → 2026-08-25), leaving only ~1k rows usable for training after `dropna`.
8. **No Staging Gate**: `deploy.yml` deploys prod then dev on the same push to main, with no staging/approval gate.
9. **No Served Model Until a Clean Retrain**: after the spec 014 remediation no run of the current data version exists, so serving uses seasonal naive until a version-2 LightGBM run beats the naive baseline.

## Planned Enhancements
- Cyclical encoding (sine/cosine for hour_of_day, day_of_week)
- Residual analysis (MAPE heatmap by hour/DOW)
- Quantile regression (prediction intervals)
- Feature importance drift monitoring (SHAP over time)

## License
MIT License.
