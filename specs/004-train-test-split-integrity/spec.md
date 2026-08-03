# 004 — Train/Validation/Test Split Integrity

**Status:** Draft  
**Author:** Technical Architect  
**Date:** 2026-08-03  
**Priority:** High  
**Dependencies:** Spec 002 (naive-baseline pairing must keep working unchanged); spec 003 (the 8-dimension Optuna search space is untouched; only how the final model is fit/evaluated after tuning changes)

---

## 1. Problem Statement

### 1.1 The Exact Issue (Early-Stopping-Selection Bias, NOT Full Leakage)

In `notebooks/06_train_lgbm.py`, `train_lgbm_model()` builds a single `test_df` (the last `CONFIG["test_days"]` days) and then uses that **exact same DataFrame twice**:

1. **As the early-stopping evaluation set** — `model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(50)])` (`06_train_lgbm.py:115`). LightGBM's early stopping watches MAE on `X_test`/`y_test` and selects the exact boosting round (i.e. the exact trained model) at which that MAE stops improving.
2. **As the source of the final reported metrics** — `mae`, `rmse`, `mape` are computed on `y_test` / `y_pred` (`06_train_lgbm.py:117-122`) and logged to MLflow, feeding `07_evaluate.py`'s champion/challenger MAPE comparison and, eventually, the README's Results table.

Because the stopping round was *chosen by watching performance on this exact data*, the final reported MAPE/MAE/RMSE is not a clean estimate of generalization to unseen data. The numbers carry a **mild but real optimistic (early-stopping-selection) bias**: every model is, in effect, evaluated on the same data its complexity was tuned against.

**This is NOT full data leakage.** The framing must be precise:

- Features and targets are still **correctly time-ordered** — no future information flows backward into the training rows.
- No feature value at prediction time leaks information unavailable at forecast time.
- The problem is **specifically that the model's stopping-point selection signal (early stopping) and the reported test metric come from the same rows**.

The severity is honest and moderate: the bias inflates the reported metrics, flows into every `model_evaluation` row and promotion decision, and would make any README benchmark numbers optimistic relative to true out-of-sample performance. It is a correctness defect in *measurement*, not in *fit legality*.

### 1.2 Distinguish From the Optuna Tuning Phase — Already Clean

`src/tuning.py`'s Optuna loop (`objective()`, `src/tuning.py:104-169`) is a **separate, already-clean code path** and is NOT part of this problem:

- The tuning phase runs on the training window **only** (`06_train_lgbm.py:103-105` passes `X_train, y_train` — the data *before* `split_date`).
- Inside `objective()`, `make_timeseries_splits()` (`src/tuning.py:59-100`) creates 3 time-ordered folds with a 168-hour gap; each fold's model is fit on `X_train_fold` and scored on the disjoint held-out `X_test_fold`, so every CV score is computed on data that fold's model never trained on. This is proper time-series CV with no overlap between the train and test windows of any fold.
- The tuning phase's output is only the *hyperparameter set* (`study.best_params`); it produces no published metric. Any mild per-fold optimism inherent to CV-based early stopping affects only *which hyperparameters win*, not the reported test MAPE of the deployed model.

The leaky double-use under investigation lives **only** in the final refit step in `train_lgbm_model()` (`06_train_lgbm.py:114-128`) — the model trained on `train_df` after tuning completes, whose stopping round is selected on `test_df`, and whose `mape` is then computed on that same `test_df`. This distinction is made explicit here precisely because it is easy to conflate.

---

## 2. Scope

### 2.1 In Scope

1. Introduce a **proper three-way time-based split** in `train_lgbm_model()` (`06_train_lgbm.py`): a contiguous `train_df` → `val_df` → `test_df` timeline, where LightGBM's early stopping uses `val_df` and the final reported `mae`/`rmse`/`mape` (and the naive-baseline pairing) use `test_df`. The stopping-round signal and the reported metric must come from **disjoint** windows.
2. Add a **`val_days` widget** alongside the existing `test_days` widget (see §3.3 for exact defaults and the additive-widget justification).
3. **Preserve the naive-baseline pairing exactly**: `compute_naive_baseline_metrics(test_df["target"], test_df["value_mwh"])` (`06_train_lgbm.py:123`) is computed on the final test window only — same call, same signature, same pairing semantics as spec 002. No change to `src/baseline.py`.
4. Extract the three-window split arithmetic into a small **pure, unit-testable function** in `src/` (pattern: `src/baseline.py` from spec 002), so the split boundaries and the `MIN_TRAINING_ROWS` arithmetic are covered by unit tests.
5. Add unit tests for the new pure function (split contiguity, disjointness, and the row-count guarantee at current data volume).
6. Add an explicit **rollout step**: after merge, the next retraining run recomputes all four models' metrics under the corrected split before any number is published (README stays `_TBD_` until then).

### 2.2 Out of Scope

- **Any change to the champion/challenger promotion logic** in `07_evaluate.py` or the 1% MAPE improvement threshold (`mape_improvement_threshold` widget).
- **Any change to `08_promote_model.py`.**
- **Any change to `src/baseline.py` or the naive-baseline comparison logic** from spec 002 — the corrected split must still produce `test_df`/`baseline_prediction` pairs compatible with `compute_naive_baseline_metrics()` with no signature change.
- **Any change to the objective-tuning work from spec 003** — `src/tuning.py`'s `get_lgbm_search_space()` and the 8-dimension search space stay exactly as-is. Only *how the final model is fit/evaluated after tuning* changes, never the tuning search itself.
- **Any change to `src/features.py`, `src/config.py`'s `MODEL_INPUT_FEATURES`, or the lag/rolling feature computation.**
- **Any change to `OPTUNA_N_TRIALS`, `OPTUNA_N_SPLITS`, `OPTUNA_GAP_HOURS`, or `make_timeseries_splits()` itself.**
- **Changing Prophet** unless a leakage finding justifies it (see §3.2 — it does NOT: Prophet is scoped out).
- **Retroactively rewriting** historical `model_evaluation` rows, MLflow runs, or promotion history computed under the old split (recompute-forward only, §3.6).
- Re-balancing `MIN_TRAINING_ROWS` (720) or `CONFIG["min_train_rows"]` (200) defaults — they are respected, not changed.

---

## 3. Design

### 3.1 Three-Way Split Scheme

#### 3.1.1 Window Definition

`train_lgbm_model()` operates on `df_model` (after the `target = value_mwh.shift(-horizon_hours)` shift and `dropna`, `06_train_lgbm.py:82-84`). Define three contiguous, non-overlapping windows from the maximum timestamp `t_max = df_model["timestamp"].max()`:

| Window | Timestamp range | Inclusive/exclusive | Role |
|---|---|---|---|
| `train_df` | `[t_min, t_max − (val_days + test_days)]` | `<= val_split_date` | Fit LightGBM |
| `val_df` | `(t_max − (val_days + test_days), t_max − test_days]` | `> val_split_date and <= test_split_date` | LightGBM `eval_set` for early stopping |
| `test_df` | `(t_max − test_days, t_max]` | `> test_split_date` | Final reported metrics + naive baseline |

Where:

- `val_split_date = t_max − timedelta(days=val_days + test_days)`
- `test_split_date = t_max − timedelta(days=test_days)`

This keeps the **test window exactly where it is today** (the last `test_days` days) and inserts the validation window immediately before it. The existing `test_df`-based metric, baseline pairing, and SHAP sample all keep their current meaning — the only behavioral change is that early stopping now watches `val_df` instead of `test_df`, and the training set shrinks by `val_days` (see §3.1.2 for why that is safe).

#### 3.1.2 Proposed Defaults and Arithmetic

Proposed defaults: **`val_days=5`, `test_days=5`** (kept symmetric with the current `test_days=5`).

Row-count arithmetic at current data volume. Silver features holds ≈ **1,900–1,920 rows** (1,920 as of 2026-07-26, per spec 001's AC8/AC9). After the `dropna` on the shifted target, the usable row count is `N_rows − horizon_hours` (the last `horizon` rows have NaN targets):

| Model | Usable rows (`N − horizon`) | Train rows @ 5+5 days (240 h holdout) | ≥ `MIN_TRAINING_ROWS` (720)? | ≥ `CONFIG["min_train_rows"]` (200)? |
|---|---|---|---|---|
| `energy_lgbm_24h` | 1,900 − 24 ≈ **1,876** | 1,876 − 240 ≈ **1,636** | ✅ (2.3×) | ✅ (8.2×) |
| `energy_lgbm_168h` | 1,900 − 168 ≈ **1,732** | 1,732 − 240 ≈ **1,492** | ✅ (2.1×) | ✅ (7.5×) |

Both models retain >2× the `MIN_TRAINING_ROWS` floor after carving out the validation window. The margin is comfortable: the `energy_lgbm_168h` case is the binding one (train ≈ 1,492 vs floor 720), leaving ~772 spare hours — more than 32 days of headroom before the 5+5-day split would approach the 720-row floor. Even in a hypothetical worst case of silver_features exactly at 1,000 rows, the 168h model would still have `1,000 − 168 − 240 = 592` train rows (a 3.6× margin over the `min_train_rows=200` widget default, though it would breach `MIN_TRAINING_ROWS`; see §3.1.3 for the low-volume policy).

#### 3.1.3 Low-Data-Volume Policy: Raise, Do Not Silently Fall Back

**Recommendation: raise** — keep the existing `ValueError` pattern in `train_lgbm_model()` (`06_train_lgbm.py:90-93`) rather than falling back to the old two-way split with a logged warning.

Justification:

1. **Silent fallback reintroduces the exact bias this spec removes.** If the notebook fell back to the leaky two-way split at low volume, the resulting metrics would be non-comparable to both the corrected numbers and each other — a run could report leaky numbers for one model and clean numbers for another, silently polluting the promotion gate and README. The whole point of spec 004 is a *consistent, corrected* measurement basis; a fallback path defeats it.
2. **The existing code already raises.** `train_lgbm_model()` raises `ValueError` when `len(train_df) < CONFIG["min_train_rows"]`. Extending that same strictness to the three-way split (check the post-carve `train_df`) is the consistent, least-surprise change.
3. **The pipeline already treats this class of failure as expected.** `02_transform.py` exits with `insufficient_data: true` and skips feature engineering when its window can't fill `MIN_TRAINING_ROWS`; drift/retraining is trigger-gated. A training-time `ValueError` on a genuinely-too-short history is a legitimate, observable signal — it matches how the system already surfaces under-provisioning.

Concretely: keep the guard, change its operand from the two-way `train_df` to the three-way `train_df`, and compare against `max(MIN_TRAINING_ROWS, CONFIG["min_train_rows"])` so the stricter of the two floors always wins. If data is too low, the notebook raises exactly as today (`Insufficient data for {model_name}. Need ..., got ...`), nothing is trained, nothing is published.

#### 3.1.4 Pure Function in `src/splits.py` (new)

Extract the split arithmetic into a unit-testable pure function (mirrors spec 002's `src/baseline.py` extraction pattern):

```python
# src/splits.py
def make_holdout_splits(
    timestamps: pd.Series,
    val_days: int,
    test_days: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (train_mask, val_mask, test_mask) for a contiguous three-way
    time split of `timestamps` (sorted, ascending).

    train  : [t_min, t_max - (val_days + test_days)]
    val    : (t_max - (val_days + test_days), t_max - test_days]
    test   : (t_max - test_days, t_max]

    The three masks are mutually exclusive and jointly exhaustive (no gaps,
    no overlap), and test is always the most recent window.
    """
```

`train_lgbm_model()` calls this with `df_model["timestamp"]` and the configured `val_days`/`test_days`, then applies the three masks to build `train_df`/`val_df`/`test_df`. This is the required test seam for §3.1.2's row-count and §3.1.4's disjointness guarantees.

### 3.2 Prophet Applicability — INVESTIGATED, NOT AFFECTED, SCOPED OUT

I read `train_prophet_model()` (`05_train_prophet.py:86-165`) in full and confirmed **Prophet has no analogous leakage**, so it is **explicitly out of scope** for the code change.

The exact evidence (with the lines read):

- `split_date = df["ds"].max() - pd.Timedelta(days=CONFIG["test_days"])` then `train_df` / `test_df` split (`05_train_prophet.py:90-92`) — same two-way split as LGBM.
- `model.fit(train_df)` (`05_train_prophet.py:112`) — Prophet is fit **only** on the training data. Prophet's `fit()` has **no `eval_set` parameter, no early-stopping callbacks, and no boosting-round selection mechanism**. The model's posterior/changepoint structure is fully determined by `train_df` + the fixed hyperparameters (`changepoint_prior_scale=0.05`, `seasonality_mode="multiplicative"`). There is no data-dependent "stopping round" to be biased.
- `forecast = model.predict(test_df[["ds", "temperature_c"]])` (`05_train_prophet.py:115`) — `test_df` is used **only** to evaluate the already-fit model.
- `mae`/`rmse`/`mape` on `y_true = test_df["y"]` vs `y_pred = forecast["yhat"]` (`05_train_prophet.py:118-123`), and `compute_naive_baseline_metrics(test_df["y"], test_df[baseline_column])` (`05_train_prophet.py:125`) — all on the same final test window, which is exactly the correct use.

**Conclusion:** for Prophet, the reported test MAPE is already a clean estimate of generalization (fit on train, scored on unseen test, with no round-selection signal derived from the test data). **Prophet is scoped OUT** — we do **not** apply the three-way split to it for consistency's sake alone, per the requirement to avoid changing code without a leakage finding that justifies it. `05_train_prophet.py` is not touched.

One note for the promotion gate: because LGBM's metric basis changes (cleaner) while Prophet's stays valid, the two model families remain comparable in the sense that each is compared against *its own* previous champion on *its own* basis. The rollout (§3.6) recomputes all four models on the same corrected training schedule, so every published number is internally consistent.

### 3.3 Widget/Config Changes (LightGBM Only)

Add a second widget, keep `test_days` exactly as-is (additive, backward-compatible).

**Before (`06_train_lgbm.py:54-63`):**

```python
dbutils.widgets.text("test_days", "5")
dbutils.widgets.text("min_train_rows", "200")
dbutils.widgets.text("n_trials", str(OPTUNA_N_TRIALS))

CONFIG = {
    "silver_table": PATHS.table_silver,
    "test_days": int(dbutils.widgets.get("test_days")),
    "min_train_rows": int(dbutils.widgets.get("min_train_rows")),
    "n_trials": int(dbutils.widgets.get("n_trials")),
}
```

**After:**

```python
dbutils.widgets.text("val_days", "5")   # NEW — early-stopping validation window
dbutils.widgets.text("test_days", "5")  # unchanged — final test window
dbutils.widgets.text("min_train_rows", "200")
dbutils.widgets.text("n_trials", str(OPTUNA_N_TRIALS))

CONFIG = {
    "silver_table": PATHS.table_silver,
    "val_days": int(dbutils.widgets.get("val_days")),   # NEW
    "test_days": int(dbutils.widgets.get("test_days")),
    "min_train_rows": int(dbutils.widgets.get("min_train_rows")),
    "n_trials": int(dbutils.widgets.get("n_trials")),
}
```

**Justification for an additive `val_days` widget (not splitting `test_days` into two new widgets):**

1. **Backward compatibility.** `test_days` already exists with default 5 and is referenced in ad-hoc runs and any external invocation. Keeping its name and default means existing invocations that pass only `test_days` keep their exact test-window semantics (the last 5 days) — they merely gain a validation window. Reusing/renaming `test_days` would change the meaning of an established parameter.
2. **Single lever per concern.** `test_days` controls "how much holdout is the reported metric" and `val_days` controls "how much holdout feeds early stopping". Two independent knobs map one-to-one to the two distinct roles in the three-way split; no hidden coupling.
3. **No job-config change required.** `databricks.yml`'s `energy_retraining_pipeline` passes no training parameters (verified in `databricks.yml:62-73`), so the scheduled retraining run picks up both widget defaults automatically. Zero YAML churn.
4. **Symmetric defaults.** `val_days=5` matches `test_days=5`, preserving today's total 5-day holdout for scoring while adding an equal 5-day window for stopping — the arithmetic in §3.1.2 shows both models keep >2× the 720 floor.

`05_train_prophet.py` keeps only `test_days` (unchanged) since Prophet is out of scope.

### 3.4 Confirmation: Optuna Tuning-Phase CV Needs No Change

Confirmed by reading `src/tuning.py` in full — **no change to `src/tuning.py` is required**:

- `make_timeseries_splits()` (`src/tuning.py:59-100`) produces time-ordered, disjoint train/test index tuples with a 168-hour gap; each fold's model trains on `X_train_fold` and is scored on the disjoint `X_test_fold` (`src/tuning.py:150-162`). The CV loop already has correct train/test separation within each fold.
- The tuning phase's only output is `study.best_params` (`src/tuning.py:212`), used as hyperparameters for the final refit. It does **not** produce any published metric, so the leakage under investigation (which is specifically about the *reported* test metric) is not present in the tuning path.
- The tuning loop receives only the training window (`06_train_lgbm.py:103-105`), i.e. the data strictly before the validation+test holdout. With the three-way split, the tuning phase continues to receive `train_df` — strictly more separated from `test_df` than today.

The sole change to how tuning interacts with the final fit is **indirect**: `train_df` is smaller by `val_days`, and the final refit's `eval_set` points at `val_df` instead of `test_df`. Neither touches the search space, the sampler, the folds, or `make_timeseries_splits()`.

### 3.5 Exact Refit Change (Before/After)

**Before (`06_train_lgbm.py:114-115`):**

```python
model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(50)])
```

**After:**

```python
model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)])
```

Everything downstream is unchanged in structure: `y_pred = model.predict(X_test)` (`:117`), the `mae`/`rmse`/`mape` computation on `y_test` (`:117-122`), the naive-baseline call on `test_df` (`:123`), and the MLflow logging (`:125-128`). Only the `eval_set` argument changes, and the `X_val, y_val` arrays come from the new `val_df` window. The final reported metrics are therefore computed on data **never** used for early-stopping round selection.

### 3.6 Backward Compatibility & Rollout

Historical `model_evaluation` rows, MLflow runs, and promotion-log entries were computed under the old (leaky) split. They are **not** retroactively rewritten. The comparison machinery (`07_evaluate.py`) compares each challenger against its own champion on the same model family, so after the corrected split lands:

- The **first post-fix retraining run** produces a challenger measured on the corrected basis. Its champion (the best historic run) was measured on the old basis — a one-cycle, apples-to-oranges comparison that is a known, accepted consequence of a measurement correction. On the *next* cycle the corrected challenger is the champion, and all subsequent comparisons are clean.
- The **test window date range is unchanged** (the last `test_days` days in both old and new code), so promotion-gate comparisons are still over the same holdout period — only the stopping-round signal source and the training-window size differ.

**Explicit rollout step (this is a stated requirement):** after merge, the next scheduled retraining run **must recompute all four models' metrics under the corrected split** (LightGBM on the three-way split; Prophet unchanged but re-run on the same schedule) before *any* number is published. The README Results table (`README.md:201-206`) stays `_TBD_` until all four models have been evaluated on the corrected basis.

---

## 4. Acceptance Criteria

- [ ] **AC1**: The final reported test metric is computed on data never used for early-stopping round selection — `eval_set` in the final refit is `(X_val, y_val)`, and `mae`/`rmse`/`mape` are computed on `(y_test, y_pred)`.
  - *Verification:* code review of `train_lgbm_model()` diff — `06_train_lgbm.py:115`'s `eval_set` references the validation window; `06_train_lgbm.py:117-122`'s scoring references the test window. Unit test in `tests/test_splits.py` asserts the `val_mask` and `test_mask` are disjoint.
- [ ] **AC2**: `MIN_TRAINING_ROWS` (720) is still respected with the three-way split at current data volume (~1,900–1,920 silver rows).
  - *Verification:* `tests/test_splits.py` includes a row-count test: build a synthetic 1,900-row (post-dropna-usable) timestamp series, apply `make_holdout_splits(ts, val_days=5, test_days=5)`, assert `train_mask.sum()` ≥ 720 for both the 24h (`1,900 − 24` usable) and 168h (`1,900 − 168` usable) cases; plus the explicit arithmetic table in §3.1.2.
- [ ] **AC3**: `compute_naive_baseline_metrics()` from spec 002 still receives correctly-paired series with no signature change.
  - *Verification:* `06_train_lgbm.py:123` still calls `compute_naive_baseline_metrics(test_df["target"], test_df["value_mwh"])`; `src/baseline.py` has zero diff; existing `tests/test_baseline.py` stays green and unchanged.
- [ ] **AC4**: `ruff`, `mypy src/`, and `pytest --cov-fail-under=80` all pass.
  - *Verification:* run `ruff check .`, `mypy src/ --ignore-missing-imports`, `pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=80` — all green on the feature branch.
- [ ] **AC5**: The Optuna tuning phase is untouched — `src/tuning.py` has zero diff (search space, sampler, folds, `make_timeseries_splits()` all byte-identical).
  - *Verification:* `git diff` on the branch shows no changes to `src/tuning.py`; existing `tests/test_tuning.py` passes unchanged.
- [ ] **AC6**: Prophet is not changed — `05_train_prophet.py` has zero diff, and its metrics remain valid (fit on train, scored on test, no stopping-round selection on test).
  - *Verification:* `git diff` on the branch shows no changes to `05_train_prophet.py`; STEP 2 finding (§3.2) documents the absence of an `eval_set`/early-stopping mechanism in `model.fit()`.
- [ ] **AC7**: Out-of-scope files are untouched — `07_evaluate.py`, `08_promote_model.py`, `src/features.py`, `src/config.py` (except importing `MIN_TRAINING_ROWS` for the low-volume guard), `src/baseline.py`, `src/tuning.py`.
  - *Verification:* `git diff main..feature/004-train-test-split-integrity` lists only `notebooks/06_train_lgbm.py`, `src/splits.py` (new), `tests/test_splits.py` (new), and `specs/004-train-test-split-integrity/spec.md`.
- [ ] **AC8**: Rollout — after merge, the next scheduled retraining run recomputes all four models' metrics under the corrected split, and no README number is published until that run completes.
  - *Verification:* `databricks.yml` retraining job run completes for all four models; MLflow runs show the corrected `mae`/`rmse`/`mape`; README Results table remains `_TBD_` until this run succeeds.

---

## 5. Open Questions / Decisions Needed

| # | Question | Options | Recommendation |
|---|---|---|---|
| 1 | **Exact `val_days` default** | 3, 5, 7 days | **5 days** — symmetric with the existing `test_days=5`, keeps the 5+5 holdout comfortably above `MIN_TRAINING_ROWS` at current volume, and is large enough for stable early-stopping MAE. |
| 2 | **Keep both `val_days` and `test_days` as separate widgets?** | (a) Add `val_days` alongside existing `test_days`; (b) rename/re-purpose existing widgets | **(a)** — additive and backward-compatible; `test_days` keeps its exact meaning (final holdout). See §3.3. |
| 3 | **Low-data-volume behavior** | (a) Raise `ValueError` (existing pattern); (b) fall back to two-way split with logged warning | **(a) Raise** — a silent fallback would reintroduce the leaky basis and make runs non-comparable (see §3.1.3). |
| 4 | **Should the low-volume guard compare against `max(MIN_TRAINING_ROWS, CONFIG["min_train_rows"])`?** | Yes / No | **Yes** — the stricter floor (720) wins; this is a one-line operand change to the existing guard, not a re-balancing of either threshold. |
| 5 | **Does Prophet get the three-way split too?** | Yes / No | **No** — Prophet has no early-stopping/round-selection mechanism (verified §3.2), so applying it would be change without a leakage finding. |
| 6 | **Is the first post-fix promotion-cycle cross-basis comparison (clean challenger vs leaky champion) acceptable?** | Accept as documented / add a `split_version` tag | **Accept as documented** — the README/promotion gate only sees new numbers after the recompute; a `split_version` tag is noted as a future-spec candidate (§7) if we want to enforce strict same-basis comparisons in `07`. |

---

## 6. Implementation Plan (for reference during coding)

### 6.1 Files to Change

| File | Change |
|---|---|
| `src/splits.py` (new) | `make_holdout_splits(timestamps, val_days, test_days) -> (train_mask, val_mask, test_mask)` — pure, typed, unit-testable (see §3.1.4). |
| `notebooks/06_train_lgbm.py` | Add `val_days` widget (default `"5"`) and `CONFIG["val_days"]`; replace the two-way split (lines 86-88) with three masks from `make_holdout_splits`; change `eval_set` from `(X_test, y_test)` to `(X_val, y_val)` (line 115); update the low-volume guard (lines 90-93) to compare the post-carve `train_df` against `max(MIN_TRAINING_ROWS, CONFIG["min_train_rows"])` (import `MIN_TRAINING_ROWS` from `src.config`). Leave `y_pred = model.predict(X_test)`, the metric computation, the naive-baseline call, MLflow logging, and the return dict unchanged. |
| `tests/test_splits.py` (new) | Unit tests: three masks disjoint & exhaustive & contiguous; `test_mask` is the most recent window; row-count guarantee (train ≥ 720) for both 24h/168h usable row counts at 1,900 rows with 5+5 defaults. |
| `specs/004-train-test-split-integrity/spec.md` | This spec. |

### 6.2 Files NOT to Change

- `notebooks/05_train_prophet.py` — Prophet is unaffected (no early stopping / no round selection on test data; §3.2).
- `notebooks/07_evaluate.py`, `notebooks/08_promote_model.py` — promotion gate and thresholds untouched.
- `src/baseline.py` — `compute_naive_baseline_metrics()` signature and logic untouched (AC3).
- `src/tuning.py` — search space, sampler, folds, and `make_timeseries_splits()` untouched (AC5).
- `src/config.py` — `MODEL_INPUT_FEATURES`, `LGBM_PARAMS`, `OPTUNA_*`, `MIN_TRAINING_ROWS` value all unchanged (only an import in `06`).
- `src/features.py`, `databricks.yml`, `.github/workflows/`, and all other notebooks.

### 6.3 Rollout Sequence

1. Implement per §3 (new `src/splits.py`, `06_train_lgbm.py` changes, new `tests/test_splits.py`).
2. Run `ruff check .`, `mypy src/ --ignore-missing-imports`, `pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=80` — all green.
3. Merge; the next scheduled `energy_retraining_pipeline` run (Sundays 02:00 UTC, per `databricks.yml:54-56`) retrains all four models under the corrected split. No `databricks.yml` change is required — the new `val_days` widget default applies automatically.
4. Verify MLflow shows the corrected `mae`/`rmse`/`mape` and that `07_evaluate` writes clean `model_evaluation` rows (LightGBM on the three-way basis; Prophet unchanged).
5. **Publish README Results only after** step 4's run succeeds and all four models have been evaluated on the corrected basis (stays `_TBD_` until then).

---

## 7. Future Spec Candidates (Not Implemented Here)

- **`split_version` tag for strict same-basis comparisons in `07_evaluate`**: tag each MLflow run with the split scheme used (e.g. `split_version="2way"` vs `"3way_5_5"`) and have `get_run_metrics()` prefer champions with the same `split_version`, so the promotion gate never compares across measurement bases. Out of scope here because the corrected numbers only appear after the recompute rollout; this would harden the gate for the long run.
- **Within-fold early-stopping hygiene in the Optuna CV loop**: each CV fold currently uses its fold-test both as `eval_set` and as the scoring window (`src/tuning.py:158-165`). This is standard, accepted CV practice and affects only hyperparameter *selection*, not published metrics — but a stricter scheme (fold-internal train/val/test) could be a low-priority follow-up if we ever want the tuned hyperparameters themselves to be as clean as the final refit.
- **Re-examine the LGBM target-offset / `gold_forecasts` labeling** already flagged in spec 002 §7: `06_train_lgbm.py` trains `features(t) → value(t+horizon)` while `04_predict.py` may store predictions under a differently-aligned timestamp. Independent of this spec; worth its own investigation.
