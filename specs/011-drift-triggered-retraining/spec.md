# 011 — Drift-Triggered Retraining and Fair Champion/Challenger Comparison

**Status:** Implemented (pending first live run)
**Date:** 2026-10-20
**Priority:** Critical
**Dependencies:** Builds on spec 009's `leak_affected` tag filtering and spec 010's dtype/tz fixes.

---

## 1. Problem Statement

Required behaviour: *drift detected → retrain automatically → promote the new model only if it is better; otherwise leave production unchanged.*

### 1.1 Drift never triggered retraining
`03_drift_check` wrote `retrain_requested.flag`, but nothing read it to start training. `energy_retraining_pipeline` ran only on its Sunday cron, and `08_promote_model` read the flag only for audit metadata.

### 1.2 A stale flag blocked triggering forever
While the flag existed, `03` treated a retrain as pending. If the retrain failed before `08` ran, the flag was never deleted.

### 1.3 Champion and challenger were scored on different windows
Each run's logged `mape` comes from its own holdout. After drift, the challenger is scored on the newer, harder window and the champion on an older one, so the comparison is biased.

### 1.4 Calendar features made drift fire on seasonality
`month`, `day_of_week`, `hour_of_day`, `is_weekend` and `is_holiday` differ between any two time windows by construction.

### 1.5 The consecutive-drift counter counted rows, not hours
Rows spread across an outage counted as "consecutive hours".

### 1.6 Training-window tags didn't describe the served model
Prophet's served model is fitted on all rows, but `training_data_end` is the latest fold cutoff.

### 1.7 The drift reference window always fell back
It read `get_latest_versions(stages=["Production"])`, but models are never registered (promotion is tag-based).

### 1.8 Champion/challenger bookkeeping bugs
- `07` dropped the best-MAPE run on the assumption that it was the newest run, so it compared against the second-best historical run, not the production run.
- `08` ignored `07`'s decision and recomputed its own.
- `08`'s `UPDATE ... WHERE run_id` referenced a column that doesn't exist, so rows were never marked processed.
- `08` demoted the old champion before tagging the new one, so a crash in between left zero production runs.

## 2. Design

| Area | Change |
|---|---|
| Trigger | `03` calls `src/retrain_trigger.trigger_retraining_job` (Jobs API `run_now`, fire-and-forget, skipped if a run is already active). `databricks.yml` injects the job id as the `retraining_job_id` widget. The weekly cron stays as a safety net. If the trigger fails, nothing is recorded and the next hourly check retries; `predict` is never blocked. |
| Pending state | The flag stores `triggered_at` and `job_run_id`. `src/drift.decide_retrain` ignores a flag older than `pending_timeout_hours` (default 6). `08` deletes the flag only at the end of a successful run. |
| Drift columns | `src/drift.select_drift_columns` excludes calendar features; `DataDriftPreset(columns=...)`. |
| Counter | `src/drift.count_consecutive_drift_hours` requires contiguous hourly `check_timestamp` values. |
| Reference window | Production-tagged run (`src/promotion.pick_production_run`), `training_data_start`/`training_data_end` tags. |
| Backtest artifact | `05`/`06` log `backtest_predictions.json` (`timestamp`, `target_timestamp`, `actual`, `predicted`) and a `served_data_end` tag. |
| Evaluation | `07` takes the champion from `production=true`. It scores the champion artifact on the challenger's backtest rows whose `target_timestamp > champion.served_data_end`, and computes both MAPEs on those rows (`comparison_method = same_window`). If there are fewer than `MIN_COMMON_ROWS` (LGBM 24, Prophet 3), it falls back to logged metrics (`logged_metrics`). The decision comes from `src/promotion.decide_promotion` (>1% relative gain, otherwise keep the champion). |
| Promotion | `08` acts on `07`'s `challenger_wins`. It skips the row if production changed since evaluation or the challenger is already production. It tags the new run first, then demotes every other production run. It marks rows via `challenger_run_id`, ignores pre-011 rows (`comparison_method IS NULL`), and fails the task if tagging fails. |

## 3. Tests
- `tests/test_promotion.py`: decision rule, champion selection with duplicate tags, `served_data_end` fallback, common-window filtering.
- `tests/test_drift.py`: calendar exclusion, contiguous counter (including the outage gap), cooldown, and pending/stale flag.
- `tests/test_retrain_trigger.py`: starts when idle; doesn't start a second run.

## 4. Out of Scope / Follow-ups
- Backtest how often the trigger would have fired using `drift_control` history. Lag and rolling features still follow seasonal level changes.
- Prophet's backtest has only 5 targets, so same-window comparisons right after a recent retrain often fall back to logged metrics.
- P2 backlog items (Evidently install probing, unused config constants, the CI gate in `deploy.yml`).
