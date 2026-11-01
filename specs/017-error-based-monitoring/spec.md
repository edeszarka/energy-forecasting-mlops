# Spec 017 — Error-based monitoring (live MAPE)

- **Status:** Approved (owner, 2026-10-31: Q1–Q4 as proposed); implemented on this branch
- **Branch:** `feature/017-error-based-monitoring`
- **Decisions:** ADR-0006 (live error monitoring and shadow scoring), proposed with this spec
- **Evidence:** spec 014 runbook results (2026-10-31), seasonal-naive history on the rebuilt silver (§1)

## 1. Context

After the spec 014 runbook (2026-10-31) silver is clean from 2024-01-01 (`data_version = 2`), gold was archived and restarted empty, and serving is on seasonal naive: the first version-2 LightGBM run (MAPE 4.767%) did not beat its logged naive MAPE (4.685%).

Nothing measures live forecast error today:

- `03_drift_check.run_prediction_drift` reads only `primary_model_name` (LightGBM) rows, so it returns `None` while naive serves (ADR-0002 already notes this). Its windows use `forecast_run_at`, and the 20% threshold is hard-coded.
- The dashboard's "Rolling MAPE" query is a weekly average per model, with no comparison against a baseline on the same hours and no alert.
- Gold has no column saying which row was actually served.

The seasonal-naive history on the rebuilt silver (`predicted(T) = actual(T−168h)`, 2024-01 → 2026-10) shows how large the error is normally:

| Window | Median MAPE | p90 | p95 / p99 | Max |
|---|---|---|---|---|
| Calendar week (146 weeks) | 7.28% | 13.21% | 15.69% (p95) | 22.53% (week of Easter Monday 2026) |
| Calendar day (1027 days) | 6.57% | 15.91% | 28.97% (p99) | 58.33% |

Two things follow:

1. **An absolute MAPE threshold by itself cannot tell a bad model from a hard week.** Holidays push naive above 15% without anything being broken. The useful signal is a comparison with naive on the same hours, plus an absolute ceiling high enough to catch only breakage.
2. **The logged training baseline is not the served baseline.** 4.685% is below every month of seasonal naive since 2024 (lowest 4.96% in 2026-09; October 2026 is 5.92%). `06_train_lgbm` scores the baseline at the anchor value (audit BUG-04), not T−168h. So whether LightGBM actually beats the served naive is unknown. Scoring it live in shadow answers that with real errors.

## 2. Goals

1. Every hour, compute the live error (MAPE, MAE, bias, number of rows) of the **served** forecast over rolling windows, and store it.
2. Compare the served forecast with seasonal naive **on exactly the same target hours**.
3. Score the latest version-2 LightGBM run in **shadow** (written to gold, never served), so its live MAPE against naive is known before any promotion.
4. Raise a visible alert when the served forecast is broken or worse than naive, and when the monitor has too little data to judge.
5. Show the result on the dashboard and in the README.

## 3. Non-goals

- Retraining or promotion triggered by error (the data-drift trigger of spec 011 is unchanged). Error-based triggering can follow once the monitor has run for a few weeks.
- Fixing the training baseline (BUG-04) or serving skew (BUG-02). The shadow LightGBM scores show how much they matter.
- As-issued weather forecasts (spec 015), the day-ahead product (spec 016).
- Re-scoring archived v1 gold rows.

## 4. Changes

### T1 · Gold records what was served (ADR-0006)
- Add `is_served BOOLEAN` to `gold_forecasts` (`ALTER TABLE ... ADD COLUMNS`, done by `04` when the column is missing; rows from before have NULL and are treated as served, since only served rows were written).
- `forecast_id` is unchanged (`md5(model_name, horizon, target_ts)`), so served, shadow and naive rows for the same hour coexist.
- `write_forecasts` keeps insert-only semantics: each target keeps its first forecast, which is issued 48h ahead in normal operation (targets are run+25h … run+48h). The lead time `timestamp − forecast_run_at` is computed by the monitor, not stored.

### T2 · Naive is always written; LightGBM is written in shadow
- `04_predict` writes, every hour, for the same target hours:
  - `seasonal_naive` rows, always (`is_served = true` when naive serves, otherwise `false`);
  - `energy_lgbm_24h` rows from the production run when one exists (`is_served = true`);
  - otherwise, `energy_lgbm_24h` rows from the **latest** run with `data_version = '2'` (`is_served = false`, `model_version = <run_id>`), when one exists.
- Shadow scoring uses the same feature path as serving (`prepare_inference_features`), so it measures what serving would deliver, skew included.
- A failure in the shadow path logs a warning and never blocks the served rows.
- Pure function `select_gold_rows(served_mode, naive_df, lgbm_df)` in `src/` decides `is_served`; tested.

### T3 · Error metrics, pure module
New `src/error_monitoring.py`:
- `window_metrics(rows, window_end, window_hours)`: rows with an actual and `timestamp` in `(window_end − window_hours, window_end]` → `n`, `mape`, `mae`, `bias` (mean of predicted − actual), `coverage = n / window_hours`. MAPE is the mean of `|actual − predicted| / actual × 100` over rows with `actual > 0`, the same definition as `compute_naive_baseline_metrics`.
- `paired_metrics(model_rows, naive_rows, ...)`: the same metrics on the intersection of target hours only, returning model and naive MAPE side by side and `relative_skill = 1 − model_mape / naive_mape`.
- `evaluate_alerts(metrics, thresholds)` → list of alert codes (T5). No Spark and no I/O in this module.

### T4 · `09_error_monitor` notebook in the hourly job
- New task after `predict` (which backfills actuals). Reads the last `max(window) + 24h` of gold.
- Windows: 24h, 7d (168h), 30d (720h), keyed on target `timestamp`.
- For each window, writes one row per scored stream to a new table `forecast_error_monitor`:
  `check_timestamp, window_hours, stream (served | naive | shadow_lgbm), model_name, n, coverage, mape, mae, bias, naive_mape_paired, relative_skill, alerts (array<string>), created_at`. Alerts are stored on the served row of the 7d window.
- `served` is built per target hour from `is_served = true` rows (NULL counts as served).
- Idempotent per `(check_timestamp, window_hours, stream)` (MERGE).

### T5 · Alerts
Evaluated on the 7d window; thresholds in `src/config.py`:

| Code | Condition | Threshold (proposed) | Rationale |
|---|---|---|---|
| `SERVED_MAPE_HIGH` | served 7d MAPE above a ceiling | `ERROR_MAPE_CEILING = 25.0` | above the worst naive week since 2024 (22.53%); catches breakage, not holidays |
| `SERVED_WORSE_THAN_NAIVE` | served model ≠ naive and paired 7d `relative_skill < 0` | `ERROR_MIN_SKILL = 0.0` | a model that loses to the fallback should not be serving |
| `SHADOW_BEATS_NAIVE` | shadow paired 7d `relative_skill ≥ ERROR_SHADOW_SKILL` with `n ≥ 120` | `0.05` | informational: evidence for a promotion review |
| `LOW_COVERAGE` | served 7d coverage below a floor | `ERROR_MIN_COVERAGE = 0.8` | missing actuals or forecasts; the other alerts are not trusted |

Alerting (owner decision Q1, §7): the task **fails when an alert starts** (it was absent in the previous hour's row), so the job's failure email fires once per episode; it does not fail again while the alert persists. `SHADOW_BEATS_NAIVE` never fails the task. The monitor row is written before the task fails.

### T6 · Retire the old prediction-drift check
- `03_drift_check.run_prediction_drift` is removed (it only ever looked at LightGBM and is replaced by T4/T5); the retrain trigger uses data drift only. `drift_control` keeps its columns: `prediction_drift_detected = false` (the column is non-nullable), the MAE columns NULL.

### T7 · Naive history in gold (owner decision Q2, §7)
- One-time notebook `scripts/backfill_017_naive_gold.py` (`dry_run` default): writes `seasonal_naive` gold rows from silver for 2024-01-08 → the first live naive row (and never past the last silver hour), `is_backfilled = true`, `is_served = NULL`, `forecast_run_at = timestamp − 48h`. Seasonal naive uses only `actual(T − 168h)`, observed 120h before the forecast would have been issued, so the reconstruction is exact. Insert-only on `forecast_id`.
- The monitor excludes `is_backfilled` rows from `served` and from alerts; they are used only by the dashboard's history panel.

### T8 · Dashboard and README
- Replace the "Rolling MAPE" query with one reading `forecast_error_monitor`: served vs naive vs shadow MAPE (7d), relative skill, active alerts.
- Add a "Live accuracy" section to the README: definitions (MAPE, paired comparison, relative skill), the alert table, and the naive history numbers from §1. Still no model accuracy claim until the monitor has data.

### T9 · ADR-0006
Live error monitoring with shadow scoring: what is served is recorded per row; naive is always computed; a challenger is scored live before promotion; alerts compare against naive on the same hours.

## 5. Runbook (owner, after merge and a successful `deploy.yml`)

1. Run `scripts/backfill_017_naive_gold.py` in `dry_run`, review counts, then `execute` (only if Q2 = yes).
2. Wait for the next hourly run; check that gold has `seasonal_naive` (served) and `energy_lgbm_24h` (shadow) rows for the same target hours.
3. Expect the first `error_monitor` run to fail once with `LOW_COVERAGE` (gold restarted on 2026-10-31, so the 7d window is not yet covered); it does not fail again while the alert persists. After ~2 days, check that `forecast_error_monitor` has rows for all three streams.
4. After ~8 days `LOW_COVERAGE` clears. Review the shadow LightGBM `relative_skill`. That is the input for the promotion review (and for prioritising BUG-02 and BUG-04).

## 6. Acceptance criteria

| # | Criterion | Verification |
|---|---|---|
| AC1 | Metrics are correct | unit tests: known rows → exact MAPE, MAE, bias, coverage; rows without actual or outside the window ignored; `actual = 0` excluded from MAPE |
| AC2 | Paired comparison uses the same hours only | test: model and naive with partially overlapping hours → metrics only on the intersection |
| AC3 | `is_served` is right | tests for `select_gold_rows`: naive serving, production LightGBM serving, shadow LightGBM, shadow failure leaves served rows intact |
| AC4 | Alerts | tests per code at and around each threshold; `LOW_COVERAGE` suppresses the other failing alerts |
| AC5 | Fail once per episode | test: alert absent → present fails; present → present does not; `SHADOW_BEATS_NAIVE` never fails |
| AC6 | No leakage in the naive gold backfill | test: every backfilled prediction equals silver at `T − 168h`, and `T − 168h < forecast_run_at` |
| AC7 | Quality gates | `ruff check .`, `ruff format --check .`, `mypy src`, `pytest --cov=src` green; coverage not below the current level |
| AC8 | Live (owner, runbook) | after 7 days: `forecast_error_monitor` has hourly rows for all streams, coverage ≥ 0.8, and a shadow LightGBM skill value |

## 7. Owner decisions

- **Q1 — Alert channel.** Proposed: fail the `error_monitor` task when an alert starts, so the existing job failure email is the channel (no new integration on Free Edition). Alternative: dashboard and table only, no failure.
- **Q2 — Naive history in gold (T7).** Proposed: yes. The dashboard gets 2+ years of naive accuracy at once, and the reconstruction is exact. Alternative: skip and wait for live history.
- **Q3 — Shadow LightGBM (T2).** Proposed: yes. It is the only honest way to learn whether the v2 model beats the served naive before fixing BUG-02/BUG-04. Cost: one extra model load and prediction per hour.
- **Q4 — Thresholds.** The values in T5 are derived from §1 and are provisional for the first month, as with spec 014's DQ thresholds.

## 8. Risks

- **Holiday weeks** can produce `SERVED_WORSE_THAN_NAIVE` for a model that is better on average. Accepted for now; the alert fires once per episode, not every hour.
- **Shadow features have the serving skew (BUG-02).** Intended: the monitor measures what would be served. A good offline backtest with a poor shadow score points at BUG-02.
- **Schema change on gold.** Readers that select `*` (dashboard, `src/dashboard.py`) get one more column; checked in T8.
