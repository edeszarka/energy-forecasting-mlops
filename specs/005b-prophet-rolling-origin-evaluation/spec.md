# 005b — Prophet Rolling-Origin Horizon Evaluation

**Status:** Draft
**Author:** Technical Architect
**Date:** 2026-08-23
**Priority:** High
**Dependencies:** Spec 005 (this replaces its single-cutoff test-window design with a rolling-origin one; `make_prophet_holdout_split` from 005 is superseded, not extended)

---

## 1. Problem Statement

Spec 005 fixed `energy_prophet_24h` and `energy_prophet_168h` producing byte-identical forecasts, by introducing a `horizon_hours`-sized gap between one fixed `train_cutoff` and a `test_days`-wide test window. This works for AC1–AC4 as written, but exposes a second, distinct problem:

`test_mask` still selects the **entire** `test_days` (5-day / 120-hour) window from that single origin. A model fit once at `train_cutoff` is then scored against every row in that window — meaning the "24h model" is actually evaluated at lead times ranging from ~25h to ~144h ahead of its training cutoff (168h model: ~169h–288h ahead), not consistently at 24h. Live data confirms the effect: post-005-fix, `energy_prophet_24h`'s MAPE moved from ~10.9% (under the old, horizon-blind bug) to 22.5% — nearly double its own naive baseline — consistent with a meaningful share of "24h" test rows actually being scored at much longer, harder lead times.

A single fixed origin cannot answer "how accurate is this model N hours out?" with a usable sample size — evaluating only the one row that's exactly `horizon_hours` from the origin gives n=1, useless for MAPE. The standard fix is **rolling-origin (walk-forward) backtesting**: multiple training cutoffs (folds), each fit independently, each scored only at its own `horizon_hours`-ahead target, with per-fold errors pooled into the final metric.

---

## 2. Scope

### 2.1 In Scope

1. Replace the single-origin evaluation in `train_prophet_model()` with a K-fold rolling-origin backtest: for each fold, fit Prophet on data up to that fold's cutoff, and score only the row(s) exactly `horizon_hours` ahead of that cutoff.
2. Add `make_prophet_rolling_origins()` to `src/splits.py` (pure, unit-testable) — computes the K cutoff timestamps and, for each, the corresponding scoring window.
3. Preserve the naive-baseline pairing: pool all folds' `(actual, baseline_prediction)` pairs into one flat series pair, then call `compute_naive_baseline_metrics()` exactly once at the end, unchanged signature — the per-fold structure is invisible to `src/baseline.py`.
4. Default `n_folds = 5`, spaced evenly across the existing `test_days`-derived window, so total evaluated rows stay in a similar ballpark to before (prevents an accidental sample-size regression).
5. Log `n_folds` and per-fold cutoffs as an MLflow param/tag for auditability.
6. Deprecate `make_prophet_holdout_split()` from spec 005 — remove its call site in `05_train_prophet.py`, but leave the function itself in `src/splits.py` (do not delete) since it's still exercised by spec 005's existing tests; a follow-up cleanup spec can remove it once nothing references it.
7. Update `tests/test_splits.py` with tests for the new function; existing spec-005 tests for `make_prophet_holdout_split` stay green (function untouched, just unused in the notebook).

### 2.2 Out of Scope

- Any change to `06_train_lgbm.py`, `make_holdout_splits`, or LightGBM's evaluation (LightGBM's `shift(-horizon_hours)` target already IS a correct direct-horizon setup — this problem is Prophet-specific).
- Any change to `src/baseline.py` or `compute_naive_baseline_metrics()`'s signature.
- Re-tuning Prophet's model hyperparameters.
- Changing `n_folds` dynamically based on data volume — fixed default, revisit only if row-count problems appear in practice.
- Rewriting historical `model_evaluation` rows or the two runs already produced under spec 005's single-origin design (`284aa575...`, and its still-unretrieved 168h sibling) — those are superseded, not corrected in place.

---

## 3. Design

### 3.1 Fold Construction

```python
# src/splits.py
def make_prophet_rolling_origins(
    timestamps: pd.Series,
    horizon_hours: int,
    test_days: int,
    n_folds: int = 5,
) -> list[tuple[pd.Series, pd.Series]]:
    """Returns a list of (train_mask, target_mask) pairs for rolling-origin
    backtesting. Each fold's target_mask selects exactly the row(s) at
    horizon_hours ahead of that fold's train cutoff — NOT a wide window.

    Fold cutoffs are spaced evenly across the last `test_days` days, so
    origin_i = t_max - test_days_in_hours + i * step, for i in [0, n_folds).
    Each fold trains on [t_min, origin_i] and scores at
    (origin_i + horizon_hours - 0.5h, origin_i + horizon_hours + 0.5h]
    (a half-hour tolerance window to catch the single matching hourly row).
    """
```

### 3.2 `train_prophet_model()` Changes

```python
folds = make_prophet_rolling_origins(df["ds"], horizon_hours, CONFIG["test_days"], n_folds=5)

all_actual, all_pred, all_baseline = [], [], []
for train_mask, target_mask in folds:
    fold_train = df[train_mask]
    fold_target = df[target_mask]
    if fold_target.empty:
        continue  # target row didn't land exactly on an hourly timestamp; skip fold
    model = Prophet(...)  # same config as today
    model.add_regressor("temperature_c")
    model.fit(fold_train)
    forecast = model.predict(fold_target[["ds", "temperature_c"]])
    all_actual.extend(fold_target["y"].values)
    all_pred.extend(forecast["yhat"].values)
    all_baseline.extend(fold_target[baseline_column].values)

y_true = np.array(all_actual)
y_pred = np.array(all_pred)
# mae/rmse/mape computed on pooled y_true/y_pred exactly as today
naive_metrics = compute_naive_baseline_metrics(pd.Series(all_actual), pd.Series(all_baseline))
```

Each fold refits Prophet independently — 5 fits per horizon, 10 per model family per run. Given the current `train_prophet` task completes in ~5 minutes for 2 fits (one per horizon) today, 5x the fits is still comfortably inside the 3600s task timeout, but should be watched on the first live run (§3.4).

### 3.3 Naive Baseline Compatibility

`baseline_column = f"lag_{horizon_hours}h"` stays exactly as spec 002 defined it — pulled per-fold from `fold_target`, pooled at the end. `compute_naive_baseline_metrics()` is called once, on the pooled arrays, with zero changes to its own code — satisfying spec 002/004's "no change to `src/baseline.py`" constraint the same way spec 005 did.

### 3.4 Runtime Budget — Verify, Don't Assume

Unlike spec 005 (a pure arithmetic change with no runtime cost), this spec 5x's Prophet's fit count. The `train_prophet` task's `timeout_seconds: 3600` in `databricks.yml` is unchanged in-scope, but the first live run after merge must be watched for actual duration — if 10 total fits (2 horizons × 5 folds) approach the timeout, `n_folds` may need reducing before this is considered stable, not after a job times out unattended.

---

## 4. Acceptance Criteria

- [ ] **AC1**: `make_prophet_rolling_origins()` returns exactly `n_folds` (`train_mask`, `target_mask`) pairs; each `target_mask` selects rows at `horizon_hours` (± the tolerance window) ahead of that fold's train cutoff.
- [ ] **AC2**: Each fold's `train_mask` uses strictly earlier data than its `target_mask` (no overlap, no future leakage into any individual fold).
- [ ] **AC3**: Pooled evaluation set size across all folds is reported and sane (not accidentally n=5) — each fold should contribute exactly 1 row given hourly data and a tight tolerance window; assert `len(all_actual) == n_folds` in the happy path (Prophet's regressor requires no NaN temperature at the target row — if this fails often, revisit the tolerance window).
- [ ] **AC4**: `compute_naive_baseline_metrics()` is called exactly once, on pooled data, no signature change.
- [ ] **AC5**: The two live MLflow runs post-fix show `mae`/`rmse`/`mape` distinct from **both** the original spec-005-bug numbers (identical pair) **and** the single-origin spec-005-fixed numbers (`284aa575...` / sibling) — proving this is a materially different evaluation, not another no-op.
- [ ] **AC6**: `n_folds`, and per-fold cutoff timestamps, are logged as MLflow params/tags.
- [ ] **AC7**: `ruff`, `mypy src/`, `pytest --cov-fail-under=80` pass.
- [ ] **AC8**: First live post-merge `train_prophet` task duration is reported and confirmed well under the 3600s timeout (not just "it finished" — the actual elapsed seconds).

---

## 5. Open Questions

| # | Question | Recommendation |
|---|---|---|
| 1 | `n_folds` default? | **5** — enough for a stable pooled MAPE without excessive fit count; revisit if AC8 shows timeout pressure. |
| 2 | Tolerance window width for matching the target row? | **±30 min** — hourly data means the exact target timestamp should land within this easily; widen only if AC3 shows frequent fold-skipping. |
| 3 | Keep `make_prophet_holdout_split` (spec 005) in the codebase unused? | **Yes for now** — avoid churn on a function whose own tests still pass; remove in a later cleanup spec once confirmed unused elsewhere. |

---

## 6. Implementation Plan

### 6.1 Files to Change
| File | Change |
|---|---|
| `src/splits.py` | Add `make_prophet_rolling_origins()`. Existing functions untouched. |
| `notebooks/05_train_prophet.py` | Replace `make_prophet_holdout_split` call with the fold loop per §3.2. |
| `tests/test_splits.py` | New tests for `make_prophet_rolling_origins` (AC1–AC3 shape). |
| `specs/005b-prophet-rolling-origin-evaluation/spec.md` | This spec. |

### 6.2 Files NOT to Change
- `06_train_lgbm.py`, `src/baseline.py`, `src/tuning.py`, `07_evaluate.py`, `08_promote_model.py`, `src/config.py`, `databricks.yml` (timeout stays as-is pending AC8 evidence).

### 6.3 Rollout
1. Implement, test locally (`ruff`, `mypy`, `pytest`).
2. Merge via PR (confirm `deploy.yml` succeeds this time before triggering retraining — same discipline as spec 005).
3. Trigger `energy_retraining_pipeline` manually once; **watch task duration live**, don't just check `result_state == SUCCESS`.
4. Verify AC5 (numbers differ from both prior versions) before any README update.
5. Only then republish Prophet's README rows, with a footnote clarifying the evaluation methodology (rolling-origin, N=5 folds pooled) so a reader knows exactly what the number represents.