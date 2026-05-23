# Energy Consumption Forecasting — MLOps Pipeline on Databricks

End-to-end forecasting pipeline for Hungarian electricity consumption using live ENTSO-E data, Databricks, Delta Lake, and MLflow. Hourly predictions, automated retraining on drift, CI/CD via GitHub Actions.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Databricks](https://img.shields.io/badge/Platform-Databricks-orange.svg)](https://www.databricks.com/)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> ⚠️ This project is a portfolio demonstration. Forecasts are not intended for operational grid management or energy trading decisions.

## Architecture

This project implements a **Split Ingestion Architecture** to overcome the outbound internet restrictions of the Databricks Free Edition.

```
[ GitHub Actions ] (Internet Access)
       │
       ├───> [ ENTSO-E API ] --------> (Hourly Load Data)
       │
       ├───> [ OpenMeteo API ] ------> (Budapest Temperature)
       │
       ├───> [ Local Python Script ] -> (Slice into hourly .json files)
       │
       └───> [ Databricks CLI ] ------> (Upload to UC Volume: /raw_ingestion/load/)
                                             │
                                             ▼
                                     [ Databricks Workspace ] (No Internet)
                                             │
                                             ├───> [ 01_ingest ] (Read Volume, MERGE Bronze)
                                             │
                                             ├───> [ 02_transform ] (Silver Features)
                                             │
                                             ├───> [ 03_drift_check ] (Monitoring)
                                             │
                                             └───> [ 04_predict ] (Gold Forecasts)
```

The pipeline is split into two independent Databricks Workflows: the **Hourly Job** and the **Retraining Job**. 

### Reliability Design
- **Separation of Concerns**: GitHub Actions handles all external connectivity, while Databricks remains an air-gapped environment focused on scalable processing and modeling.
- **Persistence Layer**: Unity Catalog Volumes act as the landing zone for raw data, ensuring a clear audit trail and enabling easy backfills.

## Data

### Source
- **ENTSO-E Transparency Platform** ([transparency.entsoe.eu](https://transparency.entsoe.eu))
  - Bidding zone: Hungary (10YHU-MAVIR----U)
  - Metric: Actual Total Load, hourly resolution, MWh
  - Access: public REST API
- **OpenMeteo** ([open-meteo.com](https://open-meteo.com))
  - Budapest hourly 2m temperature
  - Used as primary external regressor for load forecasting.

### Why temperature matters
In the Hungarian energy market, electricity consumption is highly sensitive to ambient temperature due to the significant penetration of electric heating (in winter) and air conditioning (in summer). Temperature is the single most important external regressor, as it drives the "thermal inertia" of the grid, making it a critical feature for capturing seasonal and daily demand spikes.

## Models

| Model | Horizon | Strategy | Key Features | MLflow Name |
|---|---|---|---|---|
| LightGBM | 24h | Direct (shift-24 target) | lag features, temp, calendar | energy_lgbm_24h |
| LightGBM | 168h | Direct (shift-168 target) | lag features, temp, calendar | energy_lgbm_168h |
| Prophet | 24h | Built-in decomposition | temp regressor, HU holidays | energy_prophet_24h |
| Prophet | 168h | Built-in decomposition | temp regressor, HU holidays | energy_prophet_168h |

### Champion/Challenger Pattern
This project implements a programmatic model lifecycle. Every retraining run produces a "Challenger" model. The `07_evaluate` notebook compares the Challenger's performance (Mean Absolute Percentage Error - MAPE) against the current "Production" model ("Champion"). A Challenger is only promoted if it achieves at least a 1% relative improvement in MAPE.

## MLOps Design

### Drift Detection
Drift monitoring is implemented as a "sensor" rather than a "circuit breaker." The `03_drift_check` notebook uses Evidently AI to detect statistical shifts in features and target distributions. If drift persists for 3 consecutive hours, a retrain flag is raised.

### Idempotency
All data writes use Delta Lake `MERGE INTO` logic. Every forecast row is assigned a deterministic `forecast_id` (MD5 hash of model, horizon, and timestamp). This ensures that if a Databricks job is retried, no duplicate records are created.

### CI/CD
- **Testing**: `ruff` and `pytest` run on every PR.
- **Ingestion**: `ingest.yml` runs hourly to bridge API data to Databricks.
- **Deployment**: Databricks Asset Bundles (DABs) sync code on merge to `main`.

## Repository Structure

```
energy-forecasting-mlops/
├── .github/
│   └── workflows/
│       ├── ci.yml              # lint + unit tests on PR
│       ├── deploy.yml          # deploy to Databricks on merge to main
│       └── ingest.yml          # API acquisition (Internet Bridge)
├── databricks.yml              # Asset Bundle: jobs, clusters, environments
├── notebooks/
│   ├── 01_ingest.py            # Read Volume -> Bronze
│   ├── 02_transform.py         # Feature engineering (Silver)
│   ├── 03_drift_check.py       # Evidently AI monitoring
│   ├── 04_predict.py           # Batch inference (Gold)
│   ├── 05_train_prophet.py     # Prophet training
│   ├── 06_train_lgbm.py        # LightGBM training
│   ├── 07_evaluate.py          # Champ/Challenger logic
│   └── 08_promote_model.py     # Registry management
├── src/
│   ├── api_client.py           # HTTP Clients (used in GHA)
│   ├── features.py             # Feature logic
│   └── config.py               # Constants & Thresholds
├── dashboard/
│   └── energy_forecast.sql     # SQL Dashboard queries
└── README.md
```

## Getting Started

1. **Prerequisites**: Databricks Free Edition account, GitHub account, Databricks CLI installed.
2. **Secrets**: Add `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, and `ENTSO_E_API_KEY` to GitHub Repo Secrets.
3. **Deploy**: Run `databricks bundle deploy --target prod`.
4. **Ingest**: Manually trigger `Hourly Data Ingestion` workflow in GitHub Actions to seed the Volumes.

## Results

| Model | Horizon | Test MAPE | Test MAE (MWh) | Test RMSE (MWh) | Test Period |
|---|---|---|---|---|---|
| LightGBM | 24h | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| LightGBM | 168h | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Prophet | 24h | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Prophet | 168h | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

_Results will be populated after the pipeline runs on live data. Published benchmarks for Hungary report 2–5% MAPE for 24h horizons._

## Known Limitations
1. **Moveable Holidays**: Easter and other moveable holidays are approximated with fixed dates.
2. **Weather Proxies**: Future temperature uses naive persistence (same hour, 7 days ago).
3. **Lag Uncertainty**: Direct multi-step models compounding errors at horizon edges.
4. **Free Tier Quotas**: Databricks Free Edition concurrency limits may cause queuing.
5. **Drift Counter**: Reliant on successful hourly job execution without gaps.

## License
MIT License.
