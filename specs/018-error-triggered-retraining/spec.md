# Spec 018 — Error-triggered retraining

- **Status:** Approved (owner, 2026-11-01: Q1–Q3 as proposed); implemented on this branch
- **Branch:** `feature/018-error-triggered-retraining-impl`
- **Backlog:** BL-09 (data drift triggers on seasonality); supersedes the trigger part of spec 011
- **Decisions:** ADR-0007 (retraining is triggered by forecast error; data drift is diagnostic only), proposed with this spec
- **Depends on:** spec 017 (`forecast_error_monitor`, `is_served`, shadow scoring)

## 1. Context

Since spec 011, `03_drift_check` starts `energy_retraining_pipeline` when data drift (Evidently `DataDriftPreset`) or prediction-MAE drift persists for 3 consecutive hourly checks, with a 24h cooldown. Spec 017 retired the prediction-MAE part, so data drift is the only trigger left.

`drift_control`, last 60 days (read 2026-11-01):

| Week | Checks | Data drift | Retrains |
|---|---|---|---|
| 2026-09-07 | 168 | 168 | 1 |
| 2026-10-05 | 168 | 168 | 2 |
| 2026-10-19 | 152 | 152 | 6 |
| 2026-10-26 | 150 | 150 | 6 |

- **Data drift fired on every check for 60 days**, almost always on all non-calendar features at once (`value_mwh, temperature_c, lag_*, rolling_*, humidity_pct, cloud_cover_pct`). That is the season changing between a 7-day current window and a 30-day reference, not a broken model. Part of the period ran on corrupted pre-014 data, but the pattern (every hour, every feature) is structural.
- So the trigger carries no information. The retraining job runs about once a day, limited only by the cooldown. The promotion gate keeps bad models out, so this costs compute without causing harm. It also hides a real degradation, because "drift" is always on.
- Spec 017 now measures what matters directly: the served forecast's live error against seasonal naive on the same hours.

## 2. Goals

1. Retraining is started by **sustained live forecast degradation** of the served model, read from `forecast_error_monitor`.
2. Data drift stays computed and stored as a diagnostic (what changed), but never starts a retrain.
3. A retrained and promoted model is judged only on hours it served itself, so the old model's errors cannot re-trigger a retrain.
4. Keep spec 011's safety rules: hourly streak (a missing hour breaks it), cooldown, pending-request timeout, fire-and-forget job start, flag cleared by `08`.

## 3. Non-goals

- Changing the promotion rule or its baseline (BUG-04, spec 016). A retrain started by this trigger is promoted only if 07/08 say so.
- Promotion from shadow evidence (a later spec; `SHADOW_BEATS_NAIVE` stays informational).
- Automatic rollback to naive (possible follow-up; see §7 Q3).
- Fixing the data-drift reference window (it no longer triggers anything).

## 4. Changes

### T1 · Served rows carry the run id
- `04_predict`: production LightGBM rows get `model_version = <run_id>` (today the constant `"run_latest"`), like shadow rows. Naive keeps `seasonal_naive_t-168h`.
- Older rows keep `"run_latest"`; the monitor treats them as "unknown run", which never counts as evidence for the current run (§T2).

### T2 · Trigger condition, pure function
New functions in `src/error_monitoring.py` (or `src/error_trigger.py` if it grows):
- `current_run_metrics(gold, run_id, check_time, window_hours, min_hours)`: served metrics and paired naive skill on the hours **served by `run_id` only**, within the alert window. `None` if fewer than `min_hours` such hours.
- `degraded(metrics, thresholds) -> (bool, reason)`: true when the current run's paired `relative_skill < ERROR_RETRAIN_MAX_SKILL` (default `0.0`, i.e. worse than naive) **or** its MAPE > `ERROR_MAPE_CEILING` (25%, as in spec 017). False while naive serves, with low coverage, or without enough evidence.
- The streak uses the existing `count_consecutive_drift_hours` over the monitor's hourly `degraded` history (renamed `count_consecutive_hours`; it is generic already).
- The decision uses the existing `decide_retrain` rules (renamed arguments, same logic): streak threshold, cooldown, pending timeout.

| Parameter | Proposed | Why |
|---|---|---|
| `ERROR_RETRAIN_MIN_HOURS` | 72 | three days of the run's own served hours before it can be judged |
| `ERROR_RETRAIN_STREAK_HOURS` | 24 | a full day of consecutive degraded checks: one bad day of weather does not trigger |
| `ERROR_RETRAIN_COOLDOWN_HOURS` | 72 | a newly promoted run needs `MIN_HOURS` of evidence anyway |
| `ERROR_RETRAIN_MAX_SKILL` | 0.0 | retrain when the served model is worse than the fallback |
| pending timeout | 6h | unchanged (spec 011) |

All provisional, as in spec 017.

### T3 · `09_error_monitor` owns the trigger
- After writing the monitor rows, `09` evaluates T2 and, when it decides to retrain, starts `energy_retraining_pipeline` with `trigger_retraining_job` (fire-and-forget) and writes the retrain flag with the reason (as `03` does today, so `08` and `promotion_log.drift_triggered` keep working).
- The served 7d monitor row gets four new columns: `degraded BOOLEAN`, `consecutive_degraded_hours INT`, `retrain_triggered BOOLEAN`, `retrain_reason STRING`. Added with `ALTER TABLE ... ADD COLUMNS` when missing.
- `databricks.yml`: `retraining_job_id` (and cooldown/pending widgets) move from the `drift_check` task to `error_monitor`.
- A trigger failure logs and records `retrain_triggered = false` with the error as reason. It never fails the task (the alert rule of spec 017 stays the only reason `09` fails).

### T4 · `03_drift_check` becomes diagnostic only
- Keeps running Evidently, writing `drift_control` and the HTML report.
- Removes the trigger: no `should_trigger_retrain`, no job start, no flag write. `drift_control.retrain_triggered` is written `false` and `consecutive_drift_hours` keeps counting (useful history).
- README and dashboard label drift as "diagnostic".

### T5 · Weekly safety net unchanged
- The Sunday 02:00 UTC retrain stays. While naive serves (no production run), it is the only retrain: an error trigger needs a served model to judge, and a daily retrain on the same algorithm and almost the same data would not change the outcome.

### T6 · README, ADR-0007, backlog
- README "Drift Detection" → "Drift Diagnostics"; "Live Error Monitoring" gains the trigger rules.
- ADR-0007: error-triggered retraining; data drift is diagnostic only.
- BL-09 closed by this spec.

## 5. Runbook (owner, after merge and a successful `deploy.yml`)

1. Check the next hourly run: `drift_check` writes `retrain_triggered = false`; `error_monitor` rows have the new columns with `degraded = false` while naive serves.
2. Over the next week, `energy_retraining_pipeline` runs only on Sunday (no daily runs).

## 6. Acceptance criteria

| # | Criterion | Verification |
|---|---|---|
| AC1 | Only the current run's own served hours count | test: a promoted run with 10 good hours inside a 7d window of a bad previous run → not enough evidence, not degraded |
| AC2 | Degradation rule | tests at and around `MAX_SKILL` and the MAPE ceiling; naive serving → never degraded; low coverage → not degraded |
| AC3 | Streak and safety rules | tests: 23 vs 24 consecutive degraded checks; a missing hour breaks the streak; cooldown; pending timeout (reusing spec 011 cases) |
| AC4 | Data drift never triggers | test: `03` path with persistent data drift → no job start, no flag; `drift_control.retrain_triggered = false` |
| AC5 | Trigger failure does not fail the monitor | test: job start raises → `retrain_triggered = false`, reason recorded, no exception |
| AC6 | Served rows carry the run id | predict tests: production rows `model_version = run_id` |
| AC7 | Quality gates | `ruff check .`, `ruff format --check .`, `mypy src`, `pytest --cov=src` green; coverage not below 93% |
| AC8 | Live (owner) | one week after deploy: no retrain except the Sunday run while naive serves |

## 7. Owner decisions

- **Q1 — Thresholds.** The values in T2 (72h evidence, 24h streak, 72h cooldown, skill < 0 or MAPE > 25%). Proposed as listed, provisional for the first month.
- **Q2 — Keep the weekly retrain?** Proposed: yes, as the only retrain while naive serves and as a safety net afterwards. Alternative: drop it and rely on the error trigger only, which would mean no retrain at all while naive serves.
- **Q3 — Automatic fallback.** When the trigger fires, should serving also switch to naive until a new model is promoted? Proposed: **not in this spec**. It changes serving, needs its own tests and runbook, and the promotion baseline (BUG-04) should be fixed first. It can follow as spec 019.

## 8. Risks

- **No retrain trigger while naive serves.** Accepted: the weekly run covers it, and the shadow LightGBM score shows whether a model is close to beating naive.
- **Holiday weeks** can make a good model lose to naive for a day. The 24h streak and the 72h evidence floor reduce this; Easter or Christmas week may still trigger one retrain. That is cheap and the promotion gate decides.
- **Promotion still uses the BUG-04 baseline**, so an error-triggered retrain may produce a model that is better than the served one but not promoted. Out of scope (spec 016).
