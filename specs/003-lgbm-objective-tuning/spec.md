# 003 — LightGBM Training-Objective Tuning

**Status:** Draft  
**Author:** Technical Architect  
**Date:** 2026-08-03  
**Priority:** Medium  
**Dependencies:** None (standalone change to `src/tuning.py` + `tests/test_tuning.py`)

---

## 1. Problem Statement

Every LightGBM model in this repo is trained by minimizing **L2 (squared-error) loss**, because `LGBM_PARAMS["objective"] = "regression"` (`src/config.py:118`) is a hard-coded default that is **never included in Optuna's search space**. `get_lgbm_search_space()` (`src/tuning.py:39`) tunes 7 hyperparameters, but the loss function the model actually minimizes is not one of them.

Meanwhile, every *decision* made about the model is made against a **different** loss/metric:

| Decision | Loss/metric actually used | Where |
|---|---|---|
| Training loss (what the tree-growing objective minimizes) | **L2 / squared error** (fixed, never tuned) | `LGBM_PARAMS["objective"]` |
| Early stopping (when to stop boosting) | **MAE** (`metric: "mae"`) | `LGBM_PARAMS["metric"]`, passed to `lgb.early_stopping` |
| Optuna hyperparameter selection ("which params win") | **Mean CV MAE**, computed independently via `sklearn.mean_absolute_error` | `src/tuning.py:objective()` |
| Champion/challenger promotion gate | **MAPE** (1% improvement threshold) | `07_evaluate.py` |

So the model is trained against one loss function while the tuning search, the early-stopping criterion, and the promotion gate are all evaluated against different ones. This is almost certainly an **inherited LightGBM default rather than a deliberate choice**: `src/config.py:114-116` only comments that `LGBM_PARAMS` is "Default LightGBM hyperparameters," and neither the existing specs (`001`, `002`) nor any code comment discusses the objective. There is no documented rationale for why L2 is the training loss, and — because it is fixed outside the search space — no experiment has ever checked whether a different objective produces a better model *as measured by the CV MAE the tuner already optimizes for*.

The data makes the question non-trivial: Hungarian hourly national load (≈1,400 rows, features including `is_holiday`, `is_weekend`, weather) contains outlier hours around holidays and extreme weather. L2 squares the residual on those hours, so a single spike hour can dominate the training signal; L1 and Huber are structurally more robust to such hours.

**The fix:** make `objective` a tunable categorical hyperparameter with candidates `regression` (L2), `regression_l1` (true L1/MAE loss), and `huber` (robust to outlier hours), and let Optuna's existing CV-MAE loop decide empirically which one wins. Because `objective()` already scores every trial with `sklearn.mean_absolute_error` regardless of the model's internal loss, this is a close-to-drop-in addition to the search space.

---

## 2. Scope

### 2.1 In Scope

1. Add `"objective"` as a new categorical dimension in `get_lgbm_search_space()` with exactly three candidates: `"regression"` (L2), `"regression_l1"` (L1/MAE), `"huber"` (robust).
2. Wire the sampled `objective` (and the fixed Huber `alpha`, see §3.3) through the `params` dict construction in `src/tuning.py:objective()` so it reaches `lgb.LGBMRegressor(**params)` correctly. **No change** to the CV-MAE scoring logic itself (`mean_absolute_error(y_test_fold, y_pred)` stays as-is).
3. Confirm and document that `LGBM_PARAMS["metric"] = "mae"` remains the early-stopping metric across all three objectives (a **no-code-change invariant**, §3.2).
4. Confirm and document that `LGBM_PARAMS["objective"] = "regression"` remains the default when tuning is skipped (`n_trials=0`) — a **no-code-change decision** (§3.4).
5. Update `tests/test_tuning.py`: fix the two exact-key-set assertions to include `"objective"`, and add coverage for the new dimension and the params wiring.

### 2.2 Out of Scope

- **Fixing the train/validation/test split leakage in `train_lgbm_model()`** (`06_train_lgbm.py`: `test_df` is used both as the early-stopping `eval_set` AND as the basis for the final reported MAE/RMSE/MAPE). Real and related, but deliberately deferred to a **separate spec (004)** to keep this change independently scoped and verifiable. See §7.
- Any change to `notebooks/05_train_prophet.py`, `src/features.py`, `MODEL_INPUT_FEATURES` in `src/config.py`, `07_evaluate.py`, or `08_promote_model.py`.
- Any change to the champion/challenger promotion decision logic (the 1% MAPE gate in `07_evaluate.py` is untouched).
- Any change to the number of Optuna trials (`OPTUNA_N_TRIALS`) or the existing 3-fold time-series CV structure in `make_timeseries_splits()`.
- Tuning Huber's `alpha` (held fixed; see §3.3 — a future-spec candidate if needed).
- Changing the `n_trials=0` default objective (kept as `"regression"`; see §3.4).
- Re-balancing the existing 7 hyperparameters or their ranges.

---

## 3. Design

### 3.1 Candidate Objectives

#### 3.1.1 Why the Objective Belongs in the Search Space

The existing CV loop already scores every trial with `sklearn.mean_absolute_error()` independent of the model's internal training loss (`src/tuning.py:161`). That means the search space and the scoring rule are already aligned with MAE — only the training loss is fixed. Adding `objective` to the search space simply makes the loss *consistent with what the tuner measures*, and lets the data (not a silent default) choose between L2, L1, and Huber.

#### 3.1.2 Comparison

| Objective | LightGBM name | What it optimizes | Fits | Preferable when | Risk |
|---|---|---|---|---|---|
| `regression` | L2 / squared error | Quadratic penalty on residuals | Mean-conditional; large-load hours dominate the gradient | Normal, homoskedastic load with no extreme hours | Outlier hours (holidays, extreme weather spikes) get quadratically overweighted and distort the fit on typical hours |
| `regression_l1` | L1 / absolute error | Linear penalty on residuals | Median-conditional; every hour weighted equally regardless of magnitude | MAE is the decision metric (it is — both Optuna and early stopping use it); data has spike hours you want to de-emphasize | Ignores the magnitude of extreme errors; noisier gradients, occasionally slightly worse on typical hours it previously matched well |
| `huber` | Smooth Huber loss | Quadratic within `[−alpha, alpha]`, linear beyond | Robust blend: treats typical residuals as L2, spikes as L1 | Load with occasional outlier hours where you want L2's precision on normal hours *and* L1's bounded influence on spikes | Extra `alpha` knob to manage (§3.3); needs a sensible delta to separate "normal" from "spike" |

#### 3.1.3 Trade-off of Adding a Third Dimension

The current space is deliberately narrow — 7 hyperparameters calibrated for low-to-moderate data (~1,400 rows) to avoid overfitting the validation splits (`src/tuning.py:44-45`). Adding `objective` grows it to **8 dimensions**.

- **Cost:** with `OPTUNA_N_TRIALS = 25` and a `TPESampler(seed=42)`, each additional dimension dilutes per-parameter exploration. However, a categorical with 3 choices is the cheapest kind of dimension to add: TPE models it as a small discrete distribution, and 25 trials samples each candidate ~8 times on average — enough for a coarse, robust comparison.
- **Benefit:** the added dimension is exactly the one the pipeline's decision metrics (MAE for tuning/stopping, MAPE for promotion) are most sensitive to, because the loss determines the shape of every tree fit. None of the existing 7 hyperparameters changes the loss function family.
- **Mitigations:** (a) Huber's `alpha` is **not** tuned (fixed at its LightGBM default — §3.3), so no conditional 9th dimension; (b) the trial budget and CV structure are untouched; (c) if "regression" wins the search, behavior is byte-identical to today, so the added dimension can only make the outcome as good as or better than the status quo on CV MAE.

**Net:** a favorable, low-risk trade — one cheap categorical dimension against the real risk of the current fixed L2 default.

### 3.2 Search Space and Params Wiring

#### 3.2.1 `get_lgbm_search_space()` — Before/After

```diff
     return {
         "num_leaves": IntDistribution(16, 64),
         "learning_rate": FloatDistribution(0.005, 0.1, log=True),
         "min_child_samples": IntDistribution(5, 30),
         "subsample": FloatDistribution(0.6, 1.0),
         "colsample_bytree": FloatDistribution(0.6, 1.0),
         "reg_alpha": FloatDistribution(0.0, 1.0),
         "reg_lambda": FloatDistribution(0.0, 1.0),
+        "objective": CategoricalDistribution(["regression", "regression_l1", "huber"]),
     }
```

The existing 7 entries and their ranges are **unchanged**; only the `"objective"` line is added.

#### 3.2.2 `objective()` params construction — Before/After

The existing dict comprehension already dispatches on distribution type and calls `trial.suggest_categorical(key, dist.choices)` for any `CategoricalDistribution` in the space (`src/tuning.py:125-134`). Because `"objective"` is a `CategoricalDistribution`, it is sampled automatically with **no change to the comprehension itself**. The only code change is the conditional Huber `alpha` wiring after the comprehension:

```python
    space = get_lgbm_search_space()
    params: dict[str, Any] = {
        # ... existing comprehension unchanged — "objective" is sampled via the
        # CategoricalDistribution branch, overriding LGBM_PARAMS["objective"] ...
        for key, dist in space.items()
    }

    # Huber's delta is held fixed at LightGBM's documented default (alpha=1.0),
    # see §3.3. Passed only for the "huber" objective so we never ship an
    # irrelevant param (LightGBM warns on unused "alpha" for L1/L2).
    if params["objective"] == "huber":
        params["alpha"] = 1.0

    params = {
        **LGBM_PARAMS,
        **params,
        "n_estimators": 100,
        "verbose": -1,
        "random_state": 42 + horizon_hours,
    }
```

Key points:

- **`objective` override is guaranteed:** `params` (sampled) is merged **over** `LGBM_PARAMS`, so the sampled objective wins even though `LGBM_PARAMS["objective"] = "regression"`. The `**params` line already placed after `**LGBM_PARAMS` handles this with no reordering.
- **CV MAE scoring is untouched:** `mean_absolute_error(y_test_fold, y_pred)` (`src/tuning.py:161`) remains the only score; the model's internal objective never feeds the tuner's score. This is exactly what makes the change drop-in.
- **`metric` stays fixed at `"mae"` (invariant):** LightGBM decouples the training objective from the evaluation metric — `metric` is used only for early stopping. Keeping `"mae"` (which passes through unchanged from `LGBM_PARAMS` for every trial) gives a **consistent stopping criterion regardless of which objective a given trial is evaluating**, so trials are comparable and early stopping never confounds the CV-MAE comparison. This is the recommended choice: a trial stopping "at its own optimum" under a metric that varied per objective would make the per-fold MAE numbers non-comparable. **No code change is required** — `"metric"` is never overridden per-objective.
- **`n_estimators` and other trial-time overrides are unchanged** (`100` for tuning trials, `verbose: -1`, seeded `random_state`).

### 3.3 Huber `alpha` Handling — Fixed at Default (Recommended)

**Recommendation:** do **not** add `alpha` to the search space. Hold it fixed at `alpha = 1.0`, LightGBM's own documented default for the `huber` objective, set explicitly in `objective()` only when `params["objective"] == "huber"`.

**Justification:**

1. **Search-space economy.** With 25 trials, ~1,400 rows, and an already-narrow space, a conditional 9th dimension would be sampled on only the ~⅓ of trials where `objective == "huber"`, diluting TPE's exploration of the primary lever — the objective choice itself. The objective family is the coarse, high-impact decision; `alpha` is a fine-grained knob within one family.
2. **Default equals default → zero plumbing.** `alpha=1.0` is LightGBM's default, and `06_train_lgbm.py` merges `{**LGBM_PARAMS, **tuned_params}` with no `alpha` key, so the final trained model behaves identically whether or not we set it. Fixing it at 1.0 means **no change to `06_train_lgbm.py` at all** — the final model, the `tuned_objective` MLflow log (the notebook already logs `tuned_{k}` for every tuned key), and the champion-challenger flow all work unchanged.
3. **Sane out-of-the-box behavior.** At `alpha=1.0` on load data scaled in thousands of MWh, the Huber transition region sits within typical residual magnitudes while still bounding the influence of holiday/weather spike hours — the standard, well-tested LightGBM configuration.
4. **Cheap to revisit.** If `huber` wins on real data and residual diagnostics show sensitivity to the delta, `alpha` can be promoted to a conditional tuned dimension in a later spec (§7) without changing this spec's wiring shape.

### 3.4 Default Behaviour When `n_trials=0` — Keep `"regression"` (Recommended)

**Recommendation:** leave `LGBM_PARAMS["objective"] = "regression"` in `src/config.py` unchanged. `src/config.py` is **not** in this spec's Files-to-Change list.

**Justification:**

1. **Backwards-compatible conservatism.** `LGBM_PARAMS` is used verbatim when tuning is skipped (`run_lgbm_tuning` returns `{}` at `n_trials=0`, and `06_train_lgbm.py` merges `{**LGBM_PARAMS, **tuned_params}`). Keeping `"regression"` means an untuned run is byte-identical to today — the status quo, which has produced the currently deployed champion.
2. **Promotion-gate continuity.** The MAPE-based champion/challenger gate in `07_evaluate.py` and every historical `model_evaluation` row were calibrated against L2-trained challengers. Silently changing the untuned default would alter served-model behavior *without* the CV evidence this spec exists to generate.
3. **The default rarely matters.** The whole point of this change is to let CV MAE decide; when tuning is on, the sampled objective overrides the default anyway. The default only surfaces when tuning is off — the correct answer there is "do what we've always done."

---

## 4. Acceptance Criteria

- [ ] **AC1**: `get_lgbm_search_space()` includes an `"objective"` key that is a `CategoricalDistribution` with exactly the choices `["regression", "regression_l1", "huber"]`.
  - *Verification:* `tests/test_tuning.py` — `TestGetLgbmSearchSpace.test_search_space_includes_objective` asserts `space["objective"].choices == ["regression", "regression_l1", "huber"]` and `isinstance(space["objective"], CategoricalDistribution)`.
- [ ] **AC2**: The existing 7 hyperparameters (`num_leaves`, `learning_rate`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`) and their exact ranges are **unchanged**.
  - *Verification:* code review of the `get_lgbm_search_space()` diff — only the `"objective"` line is added; existing tests that assert ranges (`test_returns_dict_with_expected_keys` updated to include `"objective"`, `test_contains_only_distribution_objects`) still pass.
- [ ] **AC3**: `tests/test_tuning.py` asserts `"objective"` is present in `get_lgbm_search_space()` output, and demonstrates that a non-`"regression"` objective can be selected/returned.
  - *Verification:* (a) `TestGetLgbmSearchSpace.test_search_space_includes_objective` (present + 3 candidates); (b) `TestObjective` extended so that `objective()` is called on trials enqueued for each of the three candidates — asserting a finite float score is returned for `"regression_l1"` and `"huber"` proves the params wiring passes the sampled objective (and fixed `alpha`) through to `lgb.LGBMRegressor(**params)` for non-regression objectives; (c) `TestRunLgbmTuning.test_run_lgbm_tuning_can_return_non_regression_objective` runs `run_lgbm_tuning(X, y, horizon_hours=24, n_trials=25)` (seeded) and asserts `result["objective"] in {"regression", "regression_l1", "huber"}`. As additional, real-data evidence during implementation: a manual Optuna run shows an objective other than `"regression"` was selected at least once across the 24h/168h multi-trial runs (checked in the logged `tuned_objective` MLflow params).
- [ ] **AC4**: `ruff`, `mypy src/`, and `pytest --cov-fail-under=80` all pass.
  - *Verification:* run `pre-commit` (ruff), `mypy src/`, `pytest --cov-fail-under=80` — all green on the feature branch.
- [ ] **AC5**: No changes to any file outside `src/tuning.py`, `src/config.py` (only if required for the `n_trials=0` decision — **not required here**, so it stays untouched), and `tests/test_tuning.py`.
  - *Verification:* `git diff` on the branch touches only `src/tuning.py` and `tests/test_tuning.py`. `notebooks/06_train_lgbm.py`, `src/config.py`, `05_train_prophet.py`, `07_evaluate.py`, `08_promote_model.py`, `src/features.py`, and `make_timeseries_splits()` are byte-identical to `main`.
- [ ] **AC6**: The early-stopping metric stays fixed at `"mae"` for every objective, and `LGBM_PARAMS["objective"]` stays `"regression"` for the `n_trials=0` path.
  - *Verification:* code review confirms `"metric"` is never overridden per-objective in `objective()` and `src/config.py` is unchanged; a `n_trials=0` call to `run_lgbm_tuning` still returns `{}` (existing `test_zero_trials_returns_empty_dict` stays green).

---

## 5. Open Questions / Decisions Needed

| # | Question | Options | Recommendation |
|---|---|---|---|
| 1 | **Should Huber's `alpha` be Optuna-tuned (conditional on `objective == "huber"`) or fixed?** | (a) Fixed `alpha=1.0`; (b) conditional tuned dimension | **(a) Fixed `1.0`** — avoids a sparse 9th dimension at 25 trials/1,400 rows; equals LightGBM's own default so the final `06` model needs no plumbing; trivially promotable to a tuned dimension later if needed (§3.3). |
| 2 | **Should `LGBM_PARAMS["metric"]` stay fixed at `"mae"` across all three objectives?** | (a) Stay `"mae"`; (b) couple to objective (e.g. `"rmse"` for L2) | **(a) Stay `"mae"`** — consistent early-stopping criterion keeps trials comparable; metric and objective are independent knobs in LightGBM; aligns the stopping metric with the tuner's sklearn MAE score (§3.2.2). |
| 3 | **Should `LGBM_PARAMS["objective"]` remain `"regression"` for `n_trials=0`?** | (a) Keep `"regression"`; (b) change default | **(a) Keep `"regression"`** — byte-identical untuned behavior, continuity with the MAPE promotion gate's historical calibration (§3.4). |
| 4 | **Is the change still worth shipping if the seeded tune picks `"regression"` as best on the real data?** | Yes / No | **Yes** — the value is de-risking an arbitrary silent default: if `regression` still wins on CV MAE, behavior is unchanged and the choice becomes evidence-backed rather than inherited. If a non-L2 objective wins and improves MAPE, we get a documented model improvement. |

---

## 6. Implementation Plan (for reference during coding)

### 6.1 Files to Change

| File | Change |
|---|---|
| `src/tuning.py` | `get_lgbm_search_space()`: add `"objective": CategoricalDistribution(["regression", "regression_l1", "huber"])` (other 7 entries untouched). `objective()`: after the search-space comprehension, add `if params["objective"] == "huber": params["alpha"] = 1.0` before the `{**LGBM_PARAMS, **params, ...}` merge. No change to CV-MAE scoring, splits, or trial plumbing. |
| `tests/test_tuning.py` | Update `TestGetLgbmSearchSpace.test_returns_dict_with_expected_keys` and `TestRunLgbmTuning.test_returns_dict_with_expected_keys` to include `"objective"`. Add: `test_search_space_includes_objective`; `TestObjective` coverage for enqueued `"regression_l1"`/`"huber"` trials returning finite floats; `TestRunLgbmTuning.test_run_lgbm_tuning_can_return_non_regression_objective` (25-trial seeded run asserting a valid objective is returned). |

### 6.2 Files NOT to Change

- `src/config.py` — `LGBM_PARAMS` keeps `objective="regression"`, `metric="mae"`, and all default values; `OPTUNA_N_TRIALS`/`OPTUNA_N_SPLITS`/`OPTUNA_GAP_HOURS` untouched.
- `notebooks/06_train_lgbm.py` — the existing `{**LGBM_PARAMS, **tuned_params}` merge and `tuned_{k}` MLflow logging already propagate a sampled `objective` to the final model with no changes.
- `notebooks/05_train_prophet.py`, `07_evaluate.py`, `08_promote_model.py`, `01_ingest.py`, `02_transform.py`, `03_drift_check.py`, `04_predict.py`.
- `src/features.py`, `src/baseline.py`, `src/drift.py`, `src/config.py`, `databricks.yml`, `.github/workflows/`.
- `make_timeseries_splits()` and all other existing tests (only additive/updating edits to the two exact-key-set assertions).

### 6.3 Rollout Sequence

1. Implement per §3.2 in `src/tuning.py`; update/extend `tests/test_tuning.py` per §4.
2. Run `pre-commit` (ruff), `mypy src/`, `pytest --cov-fail-under=80` — all green.
3. Merge; next scheduled retraining run on Databricks: verify MLflow logs `tuned_objective` for both `energy_lgbm_24h` and `energy_lgbm_168h`, and confirm it is one of the three candidates. Record whether either model selected a non-`"regression"` objective (the AC3 manual-evidence check).
4. Observe 1–2 evaluation cycles: confirm `07_evaluate.py` promotion decisions are unchanged unless an objective change actually wins on MAPE; if a non-L2 objective wins and beats the incumbent, document the improvement in the README results section.

---

## 7. Future Spec Candidates (Not Implemented Here)

- **Spec 004 — Fix train/validation/test leakage in `train_lgbm_model()` (`06_train_lgbm.py`)**: `test_df` is currently used both as the early-stopping `eval_set` AND as the basis for the final reported MAE/RMSE/MAPE. This inflates the reported metrics and lets early stopping peek at the very split used for final scoring. It is deliberately **not** fixed here — it is a real, related problem but belongs in its own spec so that the objective-tuning change (003) is independently scoped and verifiable.
- **Conditional Huber-`alpha` tuning**: if `"huber"` wins the objective search on real data, revisit `alpha` as a conditional 9th search dimension (with a larger trial budget and/or residual diagnostics), building on §3.3's fixed-default wiring.
