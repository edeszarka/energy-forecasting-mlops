# Spec 019 — Leakage-free 24h model on point-in-time features

- **Status:** Implemented on this branch (owner request, 2026-11-01: remove leakage, make LightGBM comparable with the baseline, tune only if it still loses)
- **Branch:** `feature/019-leakage-free-24h-model`
- **Decisions:** ADR-0008 (point-in-time features and a true 24h target)
- **Closes:** audit BUG-01 (future weather fill), BUG-02 (serving skew), BUG-03 (calendar/holiday skew), BUG-04 (baseline mismatch)

## 1. Leakage and skew found (code as of 2026-11-01)

| # | Where | What | Effect |
|---|---|---|---|
| L1 | `src/features.py:205,207`, `src/api_client.py:358` | weather gaps filled with `bfill(3)` and a **centred** 72h mean | training features use weather from later hours |
| L2 | `06_train_lgbm` + `src/features.py` | row t's features include load at t (`rolling_24h_mean` window ends at t) while load arrives ≥ 1.5h later | training assumes data not available at issuance |
| L3 | `06` + `src/splits.make_holdout_splits` | no purge between train, validation and test: a training row's target (t + 24h) lies in the validation window, a validation row's target in the test window | early stopping and the test score see targets of the next split |
| L4 | `04_predict.prepare_inference_features` | serving anchors run+1…run+24 are in the future: lags fall back to the last actual, rolling stats are constant, weather is the same hour a week earlier, calendar in UTC, a hard-coded holiday list (`(4, 21)` = Easter Monday 2025 only) | the model is served inputs it never saw in training; effective lead 25–48h |
| L5 | `06:148` | the logged "naive" is load at the anchor (24h persistence, assuming zero latency), the served fallback is T − 168h | promotion gate compared against a baseline that is neither served nor available |
| L6 | `05_train_prophet` | trained on silver features with L1 | benchmark numbers not trustworthy |

## 2. Formulation

- A forecast is **issued at hour τ** for **T = τ + 24h**; every hourly run writes exactly one target.
- **Load:** only hours ≤ τ − 24h. Measured on 2026-11-01: ENTSO-E data for an hour arrives ≥ 1.45h after the hour starts, but GitHub-scheduled ingestion had gaps up to 7.5h. Backtest MAPE by availability lag: 2h 3.78%, 6h 3.90%, 12h 3.91%, 24h 3.95%. 24h costs 0.16 pp and gives 22h of slack.
- **Weather:** Open-Meteo Previous Runs API, model `icon_seamless`, `*_previous_day2` = the forecast for each hour from the latest run initialised ≥ 48h before it. Verified against the Single Runs API (valid date 2026-10-28): `previous_day1` matches the run initialised exactly T − 24h, i.e. up to τ, not yet published at τ; `previous_day2` matches T − 48h. Only valid hours ≤ T are read. Historical single runs are not archived, so an exact "latest published at τ" reconstruction is not possible.
- **Calendar** of T in Europe/Budapest (`holidays` + fixed dates).
- No imputation; missing inputs stay NaN (LightGBM handles them). At serving, a missing load(T − 48h), load(T − 168h) or temperature forecast → seasonal naive for that hour.

## 3. Changes

- **T1** `src/pit_features.py`: `build_features(load, weather_forecast, issued_at)`, shared by training, evaluation and serving.
- **T2** As-issued weather data path: `src/weather_forecast.py` (fetch + CLI); `ingestion_hourly.yml` fetches 3 days back / 2 ahead every hour; `backfill.yml` fetches history in yearly chunks; `01b_ingest_weather_forecast` merges into `bronze_weather_forecast` (latest fetch wins per hour) and archives the files.
- **T3** `src/backtest.py` and `06_train_lgbm`: rolling-origin backtest over the last 12 months with a 24h purge and purged early stopping; model, weekly naive (T − 168h) and daily naive (T − 48h; T − 24h is not known at τ) on the same rows; final model on all rows; tags `feature_set`, `data_version`; per-row backtest and per-month table artifacts.
- **T4** `04_predict`: one target, T = τ + 24h, features from `build_features`; refuses a model trained on other features; naive and shadow on the same hour. `src/forecast_timing.py` reduced to that rule.
- **T5** `07_evaluate`: champion scored on the same point-in-time features; Prophet removed from the retraining job and from `EVALUATED_MODELS`; `05_train_prophet.py`, `src/splits.py`, `src/target_construction.py` and the training-time naive helper deleted (no remaining callers).
- **T6** `FEATURE_SET = "pit_24h_v1"` tag; every run lookup (03, 04, 07, 08) filters on `data_version` **and** `feature_set`, so no pre-019 run can be served, shadowed, compared or promoted.
- **T7** Silver weather imputation uses only earlier hours (L1).
- **T8** Optuna: available, off by default (`n_trials = 0`); when on, tunes only on rows before the backtest window.
- **T9** `requirements.txt` holds runtime dependencies only (dev tools come from the `[dev]` extra); `requests` 2.32.0 (yanked) → 2.32.3; Prophet, cmdstanpy, shap dropped.

## 4. Results (offline, real data)

Rolling-origin backtest, Nov 2025 – Oct 2026, 8,745 rows, default LightGBM parameters:

| | MAPE | MAE |
|---|---|---|
| LightGBM | 3.95% | 196 MW |
| Weekly naive (T − 168h) | 8.71% | 420 MW |
| Daily naive (T − 48h) | 10.49% | 502 MW |

Better than the weekly naive in all 12 months; skill 0.547. Ablations: no weather 5.28%; calendar + two lags 5.49%; a deliberately leaked target feature 0.98% (the harness exposes a leak). Hyperparameter optimisation was not needed (owner rule: tune only if the baseline is not beaten).

## 5. Runbook (owner, after merge and a successful `deploy.yml`)

1. Run `backfill.yml` with `start_date=2024-01-01`, `end_date=<tomorrow, UTC>`; it now also uploads `raw_backfill/weather_forecast/` (3 files).
2. Run the hourly job once (or wait): `ingest_weather_forecast` merges the backfill and the hourly fetch into `bronze_weather_forecast`.
3. Trigger `energy_retraining_pipeline`; check the run's `feature_set = pit_24h_v1` tag, the backtest metrics (`mape`, `naive_mape`) and the 07/08 decision (first run: promoted only if `mape < naive_mape`).
4. Next hourly runs: gold gets one `energy_lgbm_24h` row (served) and one `seasonal_naive` row per hour; `09_error_monitor` starts measuring the live skill.

## 6. Acceptance criteria

| # | Criterion | Verification |
|---|---|---|
| AC1 | No feature uses load after τ − 24h or weather after T | `tests/test_pit_features.py`: perturbing those values leaves every feature unchanged; the last usable hour does change it |
| AC2 | No training target reaches into a backtest fold | `tests/test_backtest.py`: every fold's training targets end before the fold starts; validation is purged |
| AC3 | Model and baselines on the same rows | `tests/test_backtest.py` |
| AC4 | Serving uses the training feature function and one target τ + 24h | `tests/test_predict_timing_integration.py`: model input equals `build_features` for τ; one target; other feature sets are refused; missing inputs → naive; ingestion up to 22h late still serves the model |
| AC5 | Pre-019 runs are never used | `tests/test_data_version.py`: a data-version-2 run without `feature_set` is ignored |
| AC6 | Silver weather fill uses only earlier hours | `tests/test_features.py` |
| AC7 | Quality gates | `ruff`, `ruff format --check`, `mypy src`, `pytest --cov=src` green; coverage ≥ 93% |
| AC8 | Live (owner) | runbook steps 1–4; after a week, `forecast_error_monitor` shows the served LightGBM's live skill against naive |
