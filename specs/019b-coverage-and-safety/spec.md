# Spec 019b — Coverage and safety for the 24h model

- **Status:** Approved with changes (owner, 2026-11-01: model errors alert, no `written_at`, pure functions in `src/`, legacy rows, known limitation, BL-47; D8 daily-naive fallback and O1 decided the same day)
- **Branch:** `fix/019b-coverage-and-safety` (from `feature/019-leakage-free-24h-model`, PR #29)
- **Decisions:** D1–D3 (owner, 2026-11-01), see §2
- **Evidence:** Part A review of PR #29 (2026-11-01), points 2, 3, 5 and 7

## 1. Problem

1. **Lost target hours.** Since spec 019 each hourly run writes exactly one target, T = τ + 24h (`src/forecast_timing.py:34`, `04_predict.py:467`). A run that does not happen or fails leaves its target without any gold row; nothing catches up. Main wrote 24 targets per run, so neighbours covered a missed run.
   - Last 30 days (2026-10-02 12:00 → 2026-11-01 12:00 UTC): 250 of 720 issuance hours had no successful `predict` (Databricks job runs). Of the 470 that did, an estimated ~130 would have had neither model nor naive inputs (GitHub ingestion: 156 runs instead of ~720, gaps up to 180h, 24h lookback). Estimated ~380 of 720 targets (~53%) without a forecast under the spec 019 logic.
2. **The task fails on missing data.** When the naive inputs are missing too, `04_predict.py:375-376` re-raises, so the job fails every hour of an outage and sends a failure email each time.
3. **Missing weather table is an unhandled exception.** `bronze_weather_forecast` does not exist yet. `04_predict.py:204` reads it outside any `try`, so `predict` fails without even a naive row; `01b` exits before creating the table when there are no files (`01b:98-99`, creation only at `:81`); `06:82` fails on read.
4. **Two documentation claims are stronger than the evidence.**
   - "Leakage is tested directly": only the valid-time axis is tested (`tests/test_pit_features.py:37-59`). The issue-time axis rests on the measured API semantics (`previous_dayN(v)` = latest 6-hourly run initialised ≤ v − N·24h: 168/168 hours, 3 variables, 2026-10-25…31; publication delay ≤ 4h00m in one sample, so ~20h of margin), and bronze stores no issue time.
   - 3.95% is not a clean holdout: four choices were compared on the same 12-month window (§2 of the Part A report, point 5), with differences ≤ 0.16 pp; the backtest retrains monthly while production retrains weekly.
5. **History of the Prophet fix is no longer in the code.** Spec 005/005b and their code (`src/splits.py`, `05_train_prophet.py`) were removed in PR #29.

## 2. Decisions (owner)

- **D1** The Prophet removal stays. The README gets a "History" section linking spec 005, spec 005b and the commits [ff0beb8](https://github.com/edeszarka/energy-forecasting-mlops/commit/ff0beb8) (PR #9, spec 005), [1054143](https://github.com/edeszarka/energy-forecasting-mlops/commit/1054143) (PR #10, spec 005b) and [fcc3fe1](https://github.com/edeszarka/energy-forecasting-mlops/commit/fcc3fe1) (Prophet served model on all data). All three verified on `main`.
- **D2** Catch-up within the latency budget, plus a marked naive fallback.
- **D3** ADR-0003…0007 stay "Accepted".
- **D4** Missing input data and `no_production_run` are quiet fallbacks. An MLflow, model-load or prediction error writes the fallback rows (`fallback_reason = model_error:<Type>`) and then fails the task at the end of the run, so the failure email goes out.
- **D5** No `written_at` column: `created_at` (already the actual write time, set by 04) is the write time.
- **D6** Target selection (coverage, τ', window) and the serving decision (model / naive + reason / skip) are pure functions in `src/serving_plan.py`; `04_predict` only calls them. The AC1–AC5 tests test these functions.
- **D7** Transition: future rows written by the pre-019 logic are naive rows with the same value as the new naive (T − 168h); `issued_at IS NULL` marks them. The live monitor treats them as legacy. No extra code.
- **D8** Daily-naive fallback, for **every** fallback reason (O1 decided): when the model cannot be used and T − 168h is missing but T − 48h exists, a served `daily_naive` row (T − 48h, `model_version = daily_naive_t-48h`) is written exactly like the seasonal-naive row (insert-only, `forecast_id` with its own `model_name`). `fallback_reason` is always the triggering reason (`no_production_run`, `weather_table_missing`, `model_error:<Type>`, `model_inputs_missing:<columns>`); `model_name` tells which naive served. `raise_on_model_errors` is unchanged, so a `model_error:*` with a daily-naive fallback still fails the task after the gold write. If T − 48h is missing as well, the target is skipped without an exception. Reason: a real-time data hole longer than 24h removes T − 168h for a whole week later (window (b): 123 of 307 targets).
- **D9** Monitoring with naive fallbacks: the model's live metrics and its skill against naive are computed on the LightGBM-served hours only (stream `served_model`); hours served by a naive (seasonal or daily) get no skill and raise no `SERVED_WORSE_THAN_NAIVE`; `latest_served_model` and `current_served_run` never return a naive `model_name`. The `served` stream keeps measuring what was delivered.

## 3. Changes

### B1 · Catch-up (`src/serving_plan.py`, called by `04_predict`)
- τ_now = the run's issuance hour (`issuance_hour(run time)`). The run considers every target T in **(τ_now, τ_now + 24h]**, i.e. nominal issuance hours τ' = T − 24h in (τ_now − 24h, τ_now]. The current target (τ' = τ_now) is the last of them; there is no separate in-time path.
- A target is **covered** when gold already has a served row for it (`is_served = true`, or `is_served IS NULL` for rows written before spec 017, which were all served). Covered targets are skipped entirely (`covered_targets()`, `plan_targets()`).
- For an uncovered T: the same `build_features(history, weather, [τ'])` call as training, then the same serving rules as today (production model → else naive; naive always; shadow while naive serves).
- Targets ≤ τ_now are never written: a forecast written after its target hour is not a forecast.
- New gold columns (added with `ALTER TABLE … ADD COLUMNS` when missing, NULL on older rows):
  - `issued_at TIMESTAMP`: nominal issuance hour τ' (= T − 24h). NULL on rows written before this spec (legacy, D7).
  - `is_catch_up BOOLEAN`: `τ' < τ_now`.
  - The actual write time is the existing `created_at` (D5).
- `forecast_id` and insert-only MERGE are unchanged (`MD5(model_name, horizon, T)`).

### B2 · Fallback (`decide_serving()` in `src/serving_plan.py`)
- New gold column `fallback_reason STRING` on served naive rows (NULL otherwise): `no_production_run`, `weather_table_missing`, `model_inputs_missing:<columns>`, `model_error:<ExceptionType>`.
- Decision per target, first match wins: no production run → `no_production_run`; MLflow lookup or model load raised → `model_error:<Type>`; weather table missing or empty → `weather_table_missing`; model inputs missing → `model_inputs_missing:<columns>`; prediction raised → `model_error:<Type>`; otherwise the model serves.
- A fallback writes a served seasonal-naive row (T − 168h) with the reason; if T − 168h is missing, a served daily-naive row (T − 48h) with the same reason (D8); if T − 48h is missing too, the target is skipped: a warning, no row.
- **Quiet** reasons (`no_production_run`, `weather_table_missing`, `model_inputs_missing`) never fail the task (D4).
- **`model_error:*`** writes the fallback rows and fails the task *after* the gold write (`raise_on_model_errors()`), so the alert goes out and the hour still has a forecast (D4).
- Spark/Delta failures (silver, gold read or write) still fail the task immediately. `04_predict.py:375-376` (re-raise on missing naive inputs) goes away.
- The task exit value reports counts per decision (model, naive by reason, skipped, already covered).

### B3 · Missing or empty `bronze_weather_forecast`
- `01b_ingest_weather_forecast` creates the table idempotently at the start of every run (`CREATE TABLE IF NOT EXISTS` with the file schema), also when there are no files.
- `04_predict`: a missing or empty table → `weather_table_missing` (B2), a quiet fallback. No uncaught exception.
- `06_train_lgbm`: a missing or empty table → stops with a clear error naming the table and the runbook step (backfill), before any MLflow run is created.

### B4 · Tests
- AC1–AC5 test the pure functions in `src/serving_plan.py` (D6); the notebook tests only check that 04 calls them and writes what they decide.
- Catch-up equals in-time: for the same τ' and the same data, the catch-up row's feature vector and prediction equal those of a run at τ'.
- Missing table and empty table → naive fallback with the right `fallback_reason`, no exception (04); clear error, no MLflow run (06).
- Naive inputs missing → no row, no exception, a logged reason.
- Model error (MLflow lookup, load or predict raises) → fallback rows written with `model_error:<Type>`, then the task raises; a quiet fallback never raises.
- Synthetic run schedule with outages (gaps of 1h, 5h, 23h and 30h between runs, data complete): every target T has exactly one served row when some run happened in [T − 24h, T); targets after a gap ≥ 24h have none (and only those). Covered targets are never written twice.
- Re-run the 30-day coverage estimate with the new rule (catch-up window 24h), same inputs and assumptions as Part A, raw output.

### B5 · Documentation (README and PR description)
- Replace "Leakage is tested directly" with: the valid-time axis is tested by perturbation; the issue-time axis rests on the measured API semantics (168/168 hours over 7 days; ~20h margin from one publication-delay sample); bronze does not store issue time (BL-46).
- Results: 3.95% is not a clean holdout. List the four choices made on the same window (load lag 24h; dropping `load_latest`; objective l2 vs l1; the ablations, informational only), each ≤ 0.16 pp. The backtest retrains monthly, production weekly. The live monitor is the clean holdout.
- "History" section per D1.
- Gold table row in README: the three new columns (`issued_at`, `is_catch_up`, `fallback_reason`). Catch-up and fallback in "Serving and Fallback"; the known limitation and legacy rows (§6).

### B6 · Investigation only
- The ~+1h offset between GitHub `createdAt` and Databricks run start times: display time zone or a real difference? Raw output (`gh api` timestamps vs `gh run list`, Databricks `start_time`, the run IDs the ingestion step logged). No change.

### B7 · Backlog (local `backlog.md`)
- BL-46: store forecast versions with `issued_at` / `available_at`; an issue-time-axis leakage test on top.
- BL-41 extended: scheduler reliability, lookback longer than the longest gap, automatic re-fetch after an outage.
- BL-47: alert when a production run exists but the model's inputs stay missing (e.g. N consecutive targets with `model_inputs_missing` or `weather_table_missing`).

## 4. Acceptance criteria

| # | Criterion | Verification |
|---|---|---|
| AC1 | `plan_targets()` returns every uncovered target in (τ_now, τ_now + 24h] with τ' = T − 24h and `is_catch_up`, and nothing else; `covered_targets()` counts `is_served` true or NULL | `pytest tests/test_serving_plan.py -k "window or covered"` |
| AC2 | Catch-up row = in-time row for the same τ' (features and prediction) | `pytest tests/test_predict_timing_integration.py -k equals_in_time` |
| AC3 | Synthetic outage schedule driven through `plan_targets()`: a target has exactly one served row iff a run happened in [T − 24h, T); no target written twice | `pytest tests/test_serving_plan.py -k schedule` |
| AC4 | `gold_row()` writes `issued_at`, `is_catch_up`, `fallback_reason`, `created_at` as specified (no `written_at`); 04 adds missing columns to an existing table | `pytest tests/test_gold_rows.py tests/test_predict_timing_integration.py -k "columns or gold_row"` |
| AC5 | `decide_serving()`: any fallback reason → seasonal naive if T − 168h exists, else daily naive if T − 48h exists, else skip, no exception; the reason is always the triggering one (no `weekly_naive_input_missing`); `model_error` + missing T − 168h → daily naive with `model_error:<Type>` and the task fails after the gold write; order of reasons as in B2 | `pytest tests/test_serving_plan.py -k decide` and `pytest tests/test_predict_timing_integration.py -k d8` |
| AC5c | Monitor with a mixed served stream (LightGBM, seasonal naive, daily naive): the model's metrics and skill come only from LightGBM rows; daily-naive hours get skill `None` and no alert; `latest_served_model` / `current_served_run` never return a naive name | `pytest tests/test_error_monitoring.py -k "mixed or daily_naive or latest_served"` and `pytest tests/test_error_trigger.py -k d8` |
| AC5b | Model error → fallback rows written, then the task raises (`raise_on_model_errors()` after `write_forecasts`); quiet reasons never raise | `pytest tests/test_serving_plan.py -k model_error` and `pytest tests/test_predict_timing_integration.py -k model_error` |
| AC6 | Missing or empty weather table: 04 serves naive (`weather_table_missing`); 06 stops with a clear error before MLflow; 01b creates the table with no files | `pytest -k "weather_table"` (04, 06, 01b tests) |
| AC7 | 30-day coverage estimate re-run with the new rule | script output in the PR description (raw) |
| AC8 | README: leakage wording, non-holdout note with the four choices, History links, gold columns | `grep -n "valid-time axis\|not a clean holdout\|ff0beb8\|1054143\|fcc3fe1\|is_catch_up" README.md` |
| AC9 | Quality gates | `ruff check .`, `ruff format --check .`, `mypy src`, `pytest --cov=src` green; coverage ≥ 94% |
| AC10 | B6 finding documented with raw output; BL-41/BL-46 in the local backlog | report + `grep -n "BL-46\|BL-41" backlog.md` |

## 5. Runbook after merge (B3 applied)

1. `gh run list --workflow=deploy.yml --limit 3`: the merge commit's run must be `completed success`. Otherwise stop.
2. The next hourly run: `ingest_weather_forecast` succeeds and `bronze_weather_forecast` exists even before any file arrives (B3). Until forecasts exist, `predict` serves naive rows with `fallback_reason = weather_table_missing` (or `no_production_run`, which comes first while no spec 019 run is promoted) and succeeds.
3. `gh workflow run backfill.yml -f start_date=2024-01-01 -f end_date=<tomorrow UTC>`; `gh run watch`; the log shows "Uploaded 3 weather_forecast files."
4. After the next hourly run: `bronze_weather_forecast` has `min(timestamp) = 2024-01-01` and 468 empty hours in 2024-01, none after.
5. Trigger `energy_retraining_pipeline`; check `feature_set = pit_24h_v1`, `mape`, `naive_mape` and the first-run decision (promoted only if `mape < naive_mape`).
6. Following hourly runs: one served row per target (model or naive with reason); catch-up rows only after a missed run.

## 6. Known limitations and notes

- **O1 (decided 2026-11-01):** the daily naive applies to every fallback reason; see D8.

- **A covered target is never overwritten**, even a naive fallback, even if the model's inputs arrive later. The first served row for a target is final (insert-only).
- **Legacy rows (D7):** future targets written by the pre-019 logic are served naive rows with `issued_at IS NULL`; the live monitor treats them as legacy. Their value equals the new naive (T − 168h).
- **What a catch-up row may know:** its features use only data timestamped ≤ τ' − 24h and forecasts for hours ≤ T, but that data may have *arrived* after τ'. A catch-up forecast can therefore use input the in-time run would not have had; it is still written before T and marked `is_catch_up`, so the monitor can separate it later.
- **Transition:** future targets already written by the pre-019 logic (served naive rows up to τ + 48h at merge time) count as covered and are not rewritten (D7).
