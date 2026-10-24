# Spec 013 — Scope reduction and serving safety

- **Status:** Approved for Part B (rev. 2, after Part A 2026-10-24)
- **Branch:** `feature/013-scope-reduction-serving-safety`
- **Backlog:** BL-19, BL-20, BL-21
- **Decisions:** ADR-0001 (drop 168h), ADR-0002 (Prophet benchmark-only, seasonal-naive fallback)
- **Evidence:** Part A investigation (`main` @ `9dda44d`); Part A for spec 013 (`reports/partA_spec013_2026-10-24.md`)

## 1. Context

The pipeline trains, evaluates, promotes and serves four models (LightGBM and Prophet × 24h and 168h). Part A found:

- The 168h horizon has no hourly TSO benchmark, its split has no purge gap, and its stored rows are 336h-ahead forecasts.
- Prophet is evaluated, promoted and production-tagged independently. At 168h it loses to the seasonal-naive baseline.
- `load_best_model_from_runs` (`04_predict.py:120-172`) falls back to the best historic run by MAPE, so a never-promoted run can be served.
- 04 picks among production runs by `metrics.mape ASC`, while 07/08/03 use `pick_production_run` (latest `promoted_at`).
- The ENTSO-E parser only warns on a non-`PT60M` resolution and still maps positions to hours.

This spec narrows the system to one served model family at one horizon and makes serving fail safe. It does not change what the 24h product is (spec 015).

**Live state (owner, 2026-10-24):** exactly one `energy_lgbm_24h` run is `production=true`, `leak_affected=false` (`3845300403fd450ba382b1aa13a1bd88`, promoted 2026-10-21 21:15 UTC, MAPE 8.39%). After deploy, serving is expected to load it.

## 2. Goals

1. Only the 24h horizon is trained, evaluated, promoted and served.
2. Prophet 24h is trained and evaluated as a reported benchmark only. It is never promoted or served.
3. Serving loads only the production LightGBM 24h run, chosen by the same rule as 07/08. If none can be used, it serves a seasonal-naive forecast for the same target hours.
4. ENTSO-E responses at PT60M or PT15M are parsed correctly; any other resolution fails ingestion.
5. ADRs are committed and tracked.

## 3. Non-goals

- Day-ahead issuance, new gold table, point-in-time features, purge gap, naive promotion gate (spec 015). Consequence accepted: if no production LightGBM run ever exists, the next retrain still promotes via `first_run` without a naive gate.
- MAVIR ingestion, weather forecasts, history backfill, gold actuals revision (spec 014).
- Error-based monitoring (spec 016). Consequence accepted: while naive is serving, 03's prediction-MAE drift sees no `energy_lgbm_24h` rows and cannot fire (Part A gap 8).
- Changing the `naive_*` metrics logged by 05/06 (T−24h until spec 015).
- Mutating existing MLflow tags. Historic 168h and Prophet runs keep their tags.
- Deleting historic 168h rows from `gold_forecasts`.
- The pre-existing broken "Retraining History" dashboard query (`should_promote`) and the dead `make_prophet_holdout_split` (backlog BL-31, BL-32).
- Untracking `GEMINI.md` (owner's call; ignoring a tracked file has no effect).

## 4. Changes

### T0 · Housekeeping (separate commits, first on the branch)
1. `.gitignore`: replace `docs/` with `docs/*` followed by `!docs/adr/` (a negation cannot re-include a path whose parent directory is ignored). Add `/reports`. Keep `/interview`. Normalise to a final newline; keep existing line endings.
2. Commit the spec and ADRs: `git add specs/013-scope-reduction-serving-safety docs/adr`; verify with `git check-ignore -v docs/adr/README.md` (no output) and `git check-ignore -v docs/audit_2026-10-20.md` (still ignored).
3. Separate `style:` commit: `ruff format tests/test_api_client.py tests/test_features.py` (pre-existing formatting failures on `main`), so AC7 can be checked repo-wide.

### T1 · Drop the 168h horizon (BL-19, ADR-0001)
- `06_train_lgbm.py`, `05_train_prophet.py`: train only 24h; update notebook headers.
- `04_predict.py`: horizons fixed to `[24]`; remove the `horizon_hours="both"` widget path; update header.
- `07_evaluate.py`: model list from config (T2); `horizon = 24`.
- `08_promote_model.py`: delete the unused `CONFIG["model_names"]`.
- `src/config.py`: remove unused `FORECAST_HORIZON_7D` (and `FORECAST_HORIZON_24H` if still unused). Keep `LAG_HOURS`, `lag_168h`, rolling windows.
- `src/tuning.py`: docstrings no longer say "(24 or 168)".
- `src/dashboard.py`: remove the 168h tab.
- `dashboard/energy_forecast.sql`: remove the 7-Day Forecast panel (`:63-80`); Rolling MAPE groups by `model_name` as well as week and filters `horizon_hours = 24`; Actual vs Forecast shows `seasonal_naive` as its own series instead of hiding it (`model_name IN ('energy_lgbm_24h', 'seasonal_naive')`, series by `model_name`).
- Tests, per Part A Q1:
  - keep `[24, 168]` (horizon-generic functions): `test_forecast_timing.py:13`, `test_target_construction.py:19,38,74`, `test_splits.py:249`, and the `make_prophet_holdout_split` tests (`:93-140`);
  - reduce to 24: `test_forecast_timing.py:21`, `test_predict_timing_integration.py:227`, `test_splits.py:55`.
- README: horizons, architecture, fallback, results.

### T2 · Prophet as benchmark only (BL-20, ADR-0002)
- `src/config.py`: `PROMOTABLE_MODELS = ("energy_lgbm_24h",)`, `BENCHMARK_MODELS = ("energy_prophet_24h",)`, `EVALUATED_MODELS = PROMOTABLE_MODELS + BENCHMARK_MODELS`.
- `src/promotion.py`: pure helper `apply_benchmark_policy(model_name, challenger_wins, reason) -> tuple[bool, str]` returning `(False, "Benchmark-only model (ADR-0002) — never promoted. <original reason>")` for non-promotable models, unchanged otherwise.
- `07_evaluate.py`:
  - iterate `EVALUATED_MODELS`;
  - for benchmark models, do not `continue` on "latest run already production" (`:215-217`), so a benchmark row is always written;
  - call `apply_benchmark_policy` immediately after `decide_promotion` (`:232-234`), so it also covers `first_run`.
- `08_promote_model.py`, two guards:
  - `decide_promotions`: first branch, before `force_promote`: non-promotable → `should_promote = False` with the ADR-0002 reason;
  - `execute_promotions`: immediately before the `set_tag` calls, skip and log an error for any non-promotable model (last line of defence).

### T3 · Serving: production run or seasonal naive (BL-20, ADR-0002)
- Loader (`04_predict.py`):
  - load only runs with `tags.model_name = 'energy_lgbm_24h'`, `tags.production = 'true'`, `tags.leak_affected = 'false'`, selected with `pick_production_run` from `src/promotion.py` (same rule as 07/08);
  - remove the best-historic-run path and the Prophet fallback;
  - fix the `mape:.4f` crash when the metric is missing.
- Target hours: compute the serving target timestamps once, from the LightGBM convention (anchors run+1…run+24 → `resolve_target_timestamp(anchor, "energy_lgbm_24h", 24)` → run+25…run+48). Both the LightGBM path and the naive path use this list. The naive path never calls `resolve_target_timestamp` with its own model name (it would not shift; Part A gap 2), and never goes through the type-based family detection (`:426-433`).
- Fallback triggers: no production run, any loader error, any feature-building error, or any `predict` error. Each logs a warning with the reason. Data-availability errors that also break the naive path (e.g. empty silver) still raise.
- Seasonal naive (pure function in `src/baseline.py`): `seasonal_naive_forecast(history: pd.DataFrame, targets: Sequence[pd.Timestamp], lag_hours: int = 168) -> pd.DataFrame` returning `timestamp, predicted_mwh` for targets whose `T − 168h` value is present and non-null. Precondition, documented in the docstring: silver must be no more than 120h stale for full coverage (Part A Q6).
- Gold rows from the fallback: `model_name = 'seasonal_naive'`, `model_version = 'seasonal_naive_t-168h'`, `horizon_hours = 24`; `forecast_id` built as today from `model_name`, `horizon_hours`, target timestamp.
- Targets without a naive value are skipped and counted in the log; zero rows → raise.
- No `dbutils.jobs.taskValues` (no consumer; gold `model_name` plus the warning log are the signal).

### T4 · ENTSO-E resolution handling (BL-21)
- Pre-check (read-only, Part B step 0): search recent `ingestion_hourly` GHA logs for `Unexpected resolution` and report the count.
- `src/api_client.py`:
  - `PT60M`: unchanged;
  - `PT15M`: map position p to `period_start + 15min·(p−1)`, then aggregate to hourly mean; an hour with fewer than 4 quarters present gets `value_mwh = NaN` (no partial means);
  - any other resolution: raise `ValueError` naming the resolution and period start (spec 012 path turns this into a visible GHA failure, Part A Q7).

### T5 · ADRs
- `docs/adr/` committed and tracked (T0). README links to `docs/adr/README.md`.

## 5. Acceptance criteria

| # | Criterion | Verification |
|---|---|---|
| AC1 | No 168h product code path remains | `rg -n "energy_(lgbm\|prophet)_168h\|for h in \[24, 168\]\|horizons = \[24, 168\]\|horizon_hours = 168" notebooks src dashboard databricks.yml .github` returns nothing; `rg -n "168" tests` only shows the generic tests listed in T1 and `lag_168h` / rolling features |
| AC2 | Non-promotable models are never tagged | unit tests: `apply_benchmark_policy` for LGBM unchanged, for Prophet `False` (incl. `first_run=True`); 08 decision with Prophet row and `force_promote=True` → `should_promote = False` |
| AC3 | Serving never loads an unpromoted run | unit test: best-MAPE run without `production=true`, no production run → naive rows written, LightGBM loader never called with a non-production filter |
| AC4 | Same production rule as 07/08 | unit test: two production runs → 04 selects the latest `promoted_at`, not the lowest MAPE |
| AC5 | Fallback on every model-side error | unit tests: loader raises → naive; feature build raises → naive; `predict` raises → naive |
| AC6 | Naive correctness | unit tests: `predicted(T) == actual(T − 168h)`; naive targets equal the LightGBM targets (run+25…run+48); missing source hour → row skipped; none available → raises |
| AC7 | ENTSO-E parsing | unit tests: PT60M fixture unchanged; PT15M fixture → correct hourly means; PT15M hour with 3 quarters → NaN; PT30M → `ValueError` |
| AC8 | Quality gates | `ruff check .`, `ruff format --check .`, `mypy src`, `pytest --cov=src` green; total coverage ≥ 87.8% (Part A baseline) |
| AC9 | ADRs tracked | `git ls-files docs/adr` lists README, 0001, 0002; `git check-ignore docs/audit_2026-10-20.md` still matches |
| AC10 | Live (owner, after merge) | `deploy.yml` succeeded; one `energy_hourly_pipeline` run writes only `horizon_hours = 24`, `model_name = 'energy_lgbm_24h'` rows (query below); dashboard panels render |

AC10 query:
```sql
SELECT model_name, model_version, horizon_hours, COUNT(*)
FROM workspace.energy_forecasting.gold_forecasts
WHERE pipeline_run_id = '<run id>'
GROUP BY 1, 2, 3;
```

## 6. Risks

- **T4 behaviour change:** if MAVIR actual load already arrives as PT15M, the parser fix changes historic-equivalent values (correctly). The step-0 log search tells which case applies; report it before merging.
- **Naive and LightGBM rows can coexist for the same target hour** (`forecast_id` includes `model_name`). Dashboards must separate by `model_name` (handled in T1).
- **Fallback duration is unbounded** until a production run exists; visible only through gold `model_name` and logs.
