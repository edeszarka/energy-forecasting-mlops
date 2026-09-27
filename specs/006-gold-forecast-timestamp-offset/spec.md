# 006 — Gold Forecast Timestamp Offset Fix

**Status:** Draft
**Author:** Technical Architect
**Date:** 2026-09-21
**Priority:** High
**Dependencies:** Spec 002 §7 flagged this exact investigation as a future candidate ("whether this creates a +horizon labeling offset in gold_forecasts for LGBM deserves a separate investigation spec") — this spec is that investigation, now confirmed. Independent of specs 003/004/005/005b: LightGBM's *training* metrics (MAE/RMSE/MAPE, the numbers in `model_evaluation` and README) are unaffected — this bug lives entirely in the *inference* path (`04_predict.py`) and Prophet's forecast is provably unaffected (§3.2).

---

## 1. Problem Statement

### 1.1 The Bug

`notebooks/06_train_lgbm.py` trains LightGBM as a direct multi-step forecaster:

```python
df_model["target"] = df_model["value_mwh"].shift(-horizon_hours)
```

A training pair means `model(features(t)) → value_mwh(t + horizon_hours)`. The row's own timestamp `t` is the *anchor* (what's known), never the *target* (what's predicted).

`notebooks/04_predict.py::prepare_inference_features()` builds one feature row per future timestamp `t` in `[forecast_run_at+1h, ..., forecast_run_at+horizon_hours]`, with `lag_24h = value(t-24h)`, `hour_of_day = t.hour`, etc. — structurally identical to a training anchor row at `t`, not at `t - horizon_hours`.

`generate_forecasts()` then stores the model's raw output under `timestamp = t` (the anchor), not `t + horizon_hours` (what the model's own trained semantics say that output represents):

```python
for _i, (ts, pred) in enumerate(zip(features_df.index, preds, strict=False)):
    f_id = hashlib.md5(f"{model_name}_{horizon_hours}_{ts.isoformat()}".encode()).hexdigest()
    output_rows.append({"forecast_id": f_id, "timestamp": ts, "predicted_mwh": float(pred), ...})
```

### 1.2 Confirmed via Reproduction

An independent OpenCode investigation (2026-09-21) built a synthetic-data repro (`value_mwh = hour_of_day + 100*day_of_week`, chosen to be non-periodic-24h so an offset is visible) mirroring the exact training/inference construction. Result, horizon=24, all 24 future rows:

```
mean(pred - value(t))      = +100.0
mean(pred - value(t+24h))  =   +0.0
count closer to value(t)     : 0
count closer to value(t+24h) : 24
VERDICT: CONFIRMED OFFSET
```

The stored prediction is, in every case, `value(t+24h)` — mean error exactly `0.0` — while `value(t)` is off by the full synthetic offset. **Not a suspicion; a confirmed, reproducible mislabel.**

### 1.3 Downstream Impact

Every read path that joins on `gold_forecasts.timestamp` inherits the shift:

| Consumer | Effect |
|---|---|
| `04_predict.py::backfill_actuals()` | `MERGE ... ON target.timestamp = source.timestamp` pairs a `t+horizon`-ahead prediction with the actual at `t` — wrong pairing |
| `03_drift_check.py::run_prediction_drift()` | `mae_current`/`mae_reference` computed directly from `predicted_mwh - actual_mwh` on the (mispaired) row — feeds `drift_control`, `consecutive_drift_hours`, and the retrain-flag rationale |
| `dashboard/energy_forecast.sql` — "Actual vs Forecast Line Chart", "7-Day Forecast Forward View", "Rolling MAPE Table" | All join or aggregate on `gold_forecasts.timestamp` — all display mispaired actual/predicted values |

**Not affected:** the MAPE/MAE/RMSE published in MLflow, `model_evaluation`, and the README (0.32% / 1.78% for LGBM). Those are computed in `06_train_lgbm.py` directly from `y_test`/`y_pred` in memory — they never round-trip through `gold_forecasts`. This is a data-quality bug in the *serving* path, not a training-evaluation bug.

### 1.4 Prophet Is Not Affected

Prophet has no `shift()` — it trains on `(ds, y=value(ds))` pairs directly, and `model.predict(ds=t)["yhat"]` already estimates `value(t)`. `generate_forecasts()` already branches on model family for the prediction call itself:

```python
if "lgbm" in model_name:
    preds = np.clip(model.predict(X), a_min=0, a_max=None)
else:  # Prophet
    forecast = model.predict(p_df)
    preds = forecast["yhat"].clip(lower=0).values
```

Any fix that shifts `timestamp` unconditionally for both families would *introduce* a new bug into the currently-correct Prophet path. The fix must be model-family-specific.

---

## 2. Scope

### 2.1 In Scope

1. A pure, unit-testable function resolving "anchor timestamp" → "timestamp the prediction actually represents," dispatching on model family exactly where `generate_forecasts()` already dispatches on it for the prediction call.
2. Update `generate_forecasts()` to store predictions under the resolved target timestamp, and to compute `forecast_id` from that same corrected timestamp (preserving the MD5-hash idempotency convention documented in `GEMINI.md`).
3. Unit tests proving: LGBM → `target = anchor + horizon_hours`; Prophet → `target = anchor` (regression guard — Prophet must not move).
4. A documented, one-time manual cleanup of historical `gold_forecasts` rows for `energy_lgbm_24h`/`energy_lgbm_168h` (see §3.6) — this is data proven wrong, not a measurement-basis change, so the project's established "never rewrite history" precedent (specs 004 §3.6, 005 §3.6) does not apply here.
5. Confirm, by code review (not code change), that `backfill_actuals()`, `03_drift_check.py`'s prediction-drift query, and all three affected dashboard SQL panels become correct automatically once `timestamp` carries the right value — no code change needed in any of them.

### 2.2 Out of Scope

- **Redesigning the inference loop's anchor/curve semantics.** Fixing the label reveals a second, deeper issue: with anchors spanning `forecast_run_at+1h ... +horizon_hours`, the corrected targets become `forecast_run_at+horizon_hours+1h ... +2×horizon_hours` — i.e., the "24h forecast" will represent hours 25–48 ahead, not the next 24 hours. This is a legitimate architectural question (does a fixed-offset direct model even support a genuine "next N hours" curve from one anchor?) — deliberately deferred to a future spec (§7). This spec fixes the label to match what the model actually predicts; it does not change what the model predicts.
- Any change to `06_train_lgbm.py`, `05_train_prophet.py`, or their reported metrics (unaffected, §1.3).
- Any change to `07_evaluate.py`, `08_promote_model.py`, `model_evaluation`, or `promotion_log` — these are sourced from MLflow, not `gold_forecasts`, and are untouched by this bug.
- Any change to `03_drift_check.py`'s code, `dashboard/energy_forecast.sql`, or `src/dashboard.py` — verified unnecessary (§2.1.5).
- Any change to `src/features.py`, `src/splits.py`, `src/baseline.py`, `src/tuning.py`, `databricks.yml`.
- Prophet's inference path — confirmed already correct (§1.4).

---

## 3. Design

### 3.1 Pure Function — `src/forecast_timing.py` (new)

Mirrors the project's established pattern of extracting testable logic into `src/` (`baseline.py`, `splits.py`):

```python
"""Maps an inference feature-anchor timestamp to the timestamp a prediction targets."""

from __future__ import annotations

import pandas as pd


def resolve_target_timestamp(
    anchor_ts: pd.Timestamp,
    model_name: str,
    horizon_hours: int,
) -> pd.Timestamp:
    """Returns the timestamp a raw model prediction actually represents.

    LightGBM (energy_lgbm_*) is trained as a direct multi-step forecaster on
    target = value_mwh.shift(-horizon_hours): a feature row anchored at t
    predicts value(t + horizon_hours). Prophet (energy_prophet_*) has no such
    shift — predict(ds=t) already estimates value(t) directly. The mapping is
    therefore model-family-specific, dispatched the same way generate_forecasts()
    already dispatches for the prediction call itself ("lgbm" in model_name).
    """
    if "lgbm" in model_name:
        return anchor_ts + pd.Timedelta(hours=horizon_hours)
    return anchor_ts  # Prophet: predict(ds=t) already targets t
```

### 3.2 `generate_forecasts()` — Before/After

**Before (`04_predict.py`, current):**

```python
    output_rows = []
    for _i, (ts, pred) in enumerate(zip(features_df.index, preds, strict=False)):
        f_id = hashlib.md5(f"{model_name}_{horizon_hours}_{ts.isoformat()}".encode()).hexdigest()
        output_rows.append(
            {
                "forecast_id": f_id,
                "timestamp": ts,
                "forecast_run_at": forecast_run_at,
                ...
            }
        )
```

**After:**

```python
    output_rows = []
    for _i, (anchor_ts, pred) in enumerate(zip(features_df.index, preds, strict=False)):
        target_ts = resolve_target_timestamp(anchor_ts, model_name, horizon_hours)
        # IDEMPOTENCY: Deterministic Hash — keyed on the row's own (target)
        # timestamp, per GEMINI.md's MD5(model_name + horizon + timestamp)
        # convention. Using anchor_ts here would silently reintroduce the bug.
        f_id = hashlib.md5(f"{model_name}_{horizon_hours}_{target_ts.isoformat()}".encode()).hexdigest()
        output_rows.append(
            {
                "forecast_id": f_id,
                "timestamp": target_ts,
                "forecast_run_at": forecast_run_at,
                ...
            }
        )
```

Everything else in `generate_forecasts()` — the LGBM/Prophet prediction branch, `predicted_mwh`, `actual_mwh`, `is_backfilled` — is structurally unchanged.

### 3.3 Idempotency Preserved, Not Broken

`write_forecasts()`'s `MERGE ... ON target.forecast_id = source.forecast_id` dedup logic is untouched. Because `forecast_id` is now a deterministic function of `(model_name, horizon_hours, target_ts)`, repeated or overlapping hourly runs that happen to predict the same real-world target hour still collapse to one row — the same idempotency guarantee as today, just correctly keyed.

### 3.4 Why the Three Downstream Consumers Need Zero Code Changes

| Consumer | Current code | Why it self-corrects |
|---|---|---|
| `backfill_actuals()` | `ON target.timestamp = source.timestamp` | Once `timestamp` = target time, this join pairs each prediction with the actual value at the time it was actually predicting |
| `03_drift_check.py::get_mae()` | `mean(abs(predicted_mwh - actual_mwh))` on rows filtered by `forecast_run_at` | `actual_mwh` is populated via the now-correct `backfill_actuals()` join; the arithmetic itself never referenced timestamps directly |
| Dashboard SQL (3 panels) | `LEFT JOIN silver_features s ON f.timestamp = s.timestamp` | Same join-key correction as `backfill_actuals()` |

This is the payoff of a narrowly-scoped fix: correcting one write path (`generate_forecasts()`) fixes every read path without touching them.

### 3.5 Historical Data: Why This Breaks the Project's "Never Rewrite History" Precedent — Deliberately

Specs 004 §3.6 and 005 §3.6 established: don't retroactively rewrite historical rows when a *measurement methodology* changes (old and new numbers are both "valid," just under different bases, and comparing across the transition is an accepted one-cycle quirk).

This is a different situation. Historical `gold_forecasts` rows for `energy_lgbm_24h`/`energy_lgbm_168h` are not "measured under an older but valid basis" — they are **mislabeled**: the `timestamp` column has never, for any LGBM row since the pipeline's inception, held the value it claims to hold. Leaving them in place means the dashboard actively misrepresents predictions as being for hours they were never for. Self-healing (letting old rows age out of the dashboard's lookback window) is not an acceptable substitute here, because:

- `gold_forecasts` is append/MERGE-only; nothing currently deletes old rows.
- The fix changes the `forecast_id` hash formula (§3.2), so old buggy rows will never be matched/overwritten by new runs — they persist **permanently**, not just for one transition cycle.
- `src/dashboard.py`'s lookback slider defaults to 720h (30 days) and goes up to 2160h (90 days) — old mislabeled rows would linger in the dashboard for a long, user-controlled window, not a bounded few days.

**Proposed cleanup:** a one-time, documented, manually-reviewed deletion of **all** historical `gold_forecasts` rows where `model_name IN ('energy_lgbm_24h', 'energy_lgbm_168h')`. Not a time-bounded subset — the bug is inception-to-date, so any surviving LGBM row is wrong by construction. `energy_prophet_*` rows are untouched (§1.4). This does **not** touch `model_evaluation`, `promotion_log`, or MLflow — those are independent tables/systems sourced from training-run metrics, not from `gold_forecasts` (§1.3).

This is a decision for Ede to confirm before execution (§5, item 1) — it is data deletion, and per project working norms nothing gets executed without explicit review.

### 3.6 The Curve-Coverage Side Effect (Explicitly Not Fixed Here)

Once the label is corrected, the 24 anchor points (`forecast_run_at+1h ... +24h`) map to target timestamps `forecast_run_at+25h ... +48h` — the "24h model" now visibly covers hours 25–48 ahead, not the immediate next day. This was always what the model was predicting; the bug only hid it by mislabeling the output as the next 24 hours. Whether the *intended* product behavior is a true "next 24 hours" curve (which would require re-anchoring feature construction to `target - horizon_hours`, a materially larger change) is deferred to §7.

---

## 4. Acceptance Criteria

- [ ] **AC1**: `resolve_target_timestamp()` exists in `src/forecast_timing.py`, pure and typed. Unit tests confirm `"lgbm" in model_name → anchor + horizon_hours` and `"prophet" in model_name → anchor` (unchanged), for both horizon_hours=24 and =168.
  - *Verification:* `tests/test_forecast_timing.py`.
- [ ] **AC2**: `generate_forecasts()` stores `timestamp` and computes `forecast_id` from `resolve_target_timestamp()`'s output, not the raw anchor. A regression test confirms Prophet's stored `timestamp`/`forecast_id` are byte-identical to pre-fix output for a fixed synthetic input (Prophet must not shift).
  - *Verification:* code review of the diff; `tests/test_predict_timing_integration.py` (or equivalent) using a mocked model.
- [ ] **AC3**: Idempotency is preserved — two calls to `write_forecasts()` producing rows for the same `(model_name, horizon_hours, target_timestamp)` triple result in exactly one row after MERGE (no duplicate insert, no silent overwrite of `actual_mwh` outside the `is_backfill` path).
  - *Verification:* unit/integration test exercising the MERGE logic (mocked Delta table or logic extracted for testing, following the `tests/test_ingest_logic.py` mocking pattern).
- [ ] **AC4**: `backfill_actuals()`, `03_drift_check.py::run_prediction_drift()`, and all three dashboard SQL panels referencing `gold_forecasts.timestamp` require zero code changes — verified correct by code review against the corrected join semantics (§3.4).
  - *Verification:* code review only; `git diff` shows no changes to `03_drift_check.py` or `dashboard/energy_forecast.sql`.
- [ ] **AC5**: A one-time cleanup removes all historical `energy_lgbm_24h`/`energy_lgbm_168h` rows from `gold_forecasts`; `energy_prophet_*` rows, `model_evaluation`, `promotion_log`, and MLflow are confirmed untouched.
  - *Verification:* row counts before/after by `model_name`; manual sign-off logged in the PR per §5 item 1's resolution.
- [ ] **AC6**: `ruff`, `mypy src/`, and `pytest --cov-fail-under=80` all pass.
- [ ] **AC7**: Out-of-scope files untouched — `05_train_prophet.py`, `06_train_lgbm.py`, `07_evaluate.py`, `08_promote_model.py`, `src/features.py`, `src/splits.py`, `src/baseline.py`, `src/tuning.py`, `dashboard/energy_forecast.sql`, `databricks.yml` all byte-identical to `main`.
  - *Verification:* `git diff main..feature/006-gold-forecast-timestamp-offset` lists only `notebooks/04_predict.py`, `src/forecast_timing.py` (new), test files, and this spec.

---

## 5. Open Questions / Decisions Needed

| # | Question | Options | Recommendation |
|---|---|---|---|
| 1 | **Delete all historical LGBM `gold_forecasts` rows, or only unbackfilled ones?** | (a) All LGBM rows (bug is inception-to-date); (b) only rows with `actual_mwh IS NULL` (unscored predictions) | **(a) All** — a backfilled row is just as mislabeled as an unscored one; keeping "scored but wrong" rows is worse than deleting them, not better. Requires Ede's explicit sign-off before execution — this is data deletion. |
| 2 | **Cleanup before or after the code deploy?** | (a) Deploy fix, then clean up; (b) clean up, then deploy | **(a) After** — deploying first means any straggler old-format row simply sits inert (new runs use the new hash, never touch it); cleaning up first then deploying leaves a window where a not-yet-fixed hourly run could repopulate the table with fresh buggy rows. |
| 3 | **Document the `timestamp` = target-time convention in `GEMINI.md`?** | Yes / No | **Yes, as a one-line clarification** to the existing forecast_id convention note — cheap, prevents recurrence, but not a hard blocker on this spec; can land in the same PR or a fast-follow doc-only commit. |
| 4 | **Does the curve-coverage side effect (§3.6) block this spec?** | (a) Ship the label fix now, track redesign separately; (b) block until redesigned | **(a) Ship now** — correctness of what's already stored outweighs the UX question of what window it covers; matches the project's demonstrated pattern (003 shipped ahead of 004; 005 shipped, then was itself superseded by 005b) of narrow, provable fixes before larger redesigns. |

---

## 6. Implementation Plan

### 6.1 Files to Change

| File | Change |
|---|---|
| `src/forecast_timing.py` (new) | `resolve_target_timestamp(anchor_ts, model_name, horizon_hours) -> pd.Timestamp` per §3.1. |
| `notebooks/04_predict.py` | `generate_forecasts()`: replace `ts` with `resolve_target_timestamp(anchor_ts, model_name, horizon_hours)` for both `timestamp` and the `forecast_id` hash input, per §3.2. No other function in this file changes — `prepare_inference_features()` already correctly builds anchor-relative features and needs no edit. |
| `tests/test_forecast_timing.py` (new) | Unit tests per AC1. |
| `tests/test_predict_timing_integration.py` (new, or extend existing predict tests if present) | Regression tests per AC2/AC3. |
| `specs/006-gold-forecast-timestamp-offset/spec.md` | This spec. |
| `GEMINI.md` (optional, per §5 item 3) | One-line clarification to the forecast_id convention note. |

### 6.2 Files NOT to Change

- `notebooks/05_train_prophet.py`, `notebooks/06_train_lgbm.py` — training metrics unaffected (§1.3).
- `notebooks/07_evaluate.py`, `notebooks/08_promote_model.py` — sourced from MLflow, not `gold_forecasts`.
- `notebooks/03_drift_check.py` — correct automatically once `timestamp` is correct (§3.4, AC4).
- `dashboard/energy_forecast.sql`, `src/dashboard.py` — same reasoning as above.
- `src/features.py`, `src/splits.py`, `src/baseline.py`, `src/tuning.py`, `databricks.yml`.
- `notebooks/04_predict.py::prepare_inference_features()` — already correctly anchor-relative; only `generate_forecasts()` changes.

### 6.3 Rollout Sequence

1. Implement `src/forecast_timing.py` and the `04_predict.py` diff; run `ruff`, `mypy src/`, `pytest --cov-fail-under=80`.
2. Manually re-run the OpenCode reproduction script against the fixed code path (mocked model), confirm the stored timestamp now matches `value(t+24h)`'s real-world hour, not `value(t)`'s.
3. Get explicit sign-off on §5 item 1 (cleanup scope) before touching the live table.
4. Merge; confirm `deploy.yml` succeeded (per project's standing "code merged ≠ deployed" discipline) before doing anything else.
5. Execute the one-time cleanup (§3.5) as a reviewed, logged manual step — not baked into a notebook cell, matching how spec 001's manual repair run was handled.
6. Let the next few hourly `energy_hourly_pipeline` runs repopulate `gold_forecasts` for LGBM under the corrected labeling; spot-check the dashboard's "Actual vs Forecast" panel once `actual_mwh` starts backfilling on the new rows.

---

## 7. Future Spec Candidates (Not Implemented Here)

- **Anchor/curve semantics redesign**: decide whether the pipeline should produce a genuine "next N hours from now" curve (requiring feature construction anchored at `target − horizon_hours` instead of `target`) versus explicitly documenting and relabeling the dashboard panels to reflect what a fixed-offset direct model actually supports (an "N-to-2N hours ahead" band, not "next N hours"). This spec fixes the label to match the model; it does not decide what the model should be asked to do.
- **Generalized gold-layer reconciliation check**: a lightweight, repo-native diagnostic (in the spirit of Evidently's drift monitoring, but for label/schema correctness rather than statistical drift) that could have caught this class of bug earlier — e.g., a periodic sanity check that a model's stored prediction for `timestamp` is closer to `value(timestamp)` than to `value(timestamp ± horizon_hours)` once actuals backfill.
