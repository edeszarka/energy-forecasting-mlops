# 002 — Naive Baseline Evaluation

**Status:** Approved  
**Author:** Technical Architect  
**Date:** 2026-08-01  
**Priority:** High  
**Dependencies:** None (standalone change to `05_train_prophet.py`, `06_train_lgbm.py`, `07_evaluate.py`, `src/baseline.py`, `tests/test_baseline.py`)

---

## 1. Problem Statement

`07_evaluate.py` only compares each challenger model against the previous champion (or promotes unconditionally on `first_run`). There is no comparison against a trivial baseline, so nothing answers the question **"does this model beat doing nothing clever?"**

The README's Results section (`README.md:195-206`) is currently all `_TBD_`. Before any numbers are published, we must be able to demonstrate that every published model is better than a seasonal-naive baseline. Without this comparison, a mediocre model could be published while a trivial `lag_24h`/`lag_168h` predictor would match or beat it.

### 1.1 Current Behaviour

- `05_train_prophet.py` and `06_train_lgbm.py` log only model metrics (`mae`, `rmse`, `mape`) to MLflow — no baseline reference.
- `07_evaluate.py` (`get_run_metrics`, lines 64-107) pulls only `mae`/`rmse`/`mape`/`n_train`/`n_test` from each run.
- The `model_evaluation` Delta table has no baseline columns; the printed summary (`07_evaluate.py:183-196`) shows only challenger vs champion MAPE.

---

## 2. Scope

### 2.1 In Scope

1. Compute a **seasonal-naive baseline per horizon** inside `05_train_prophet.py` and `06_train_lgbm.py`, on the **identical test split** each model already uses (same `test_df`, same date range — not a separately sampled window).
2. Log `naive_mae`, `naive_rmse`, `naive_mape` as real MLflow metrics on every training run, using the same actuals the model's own metrics use and the same `calculate_mape` semantics already defined in each notebook.
3. Extract a small pure function in `src/` (pattern: `compute_prediction_mae` in `src/drift.py:50`) so the baseline metric logic is unit-testable without Spark/MLflow mocking.
4. Pull the `naive_*` metrics from the challenger run in `07_evaluate.py`'s `get_run_metrics()`.
5. Add columns to `model_evaluation`: `baseline_mape`, `baseline_mae`, `baseline_rmse`, `beats_baseline` (Boolean: `challenger_mape < baseline_mape`).
6. Update the printed evaluation summary to show `baseline_mape` alongside challenger/champion MAPE, and **clearly flag** any model where `beats_baseline` is False.
7. Unit tests for the baseline calculation logic; existing tests must keep passing.

### 2.2 Out of Scope

- **Changing champion/challenger promotion logic.** `beats_baseline` is a **reported metric only**; it must NOT influence `challenger_wins` / `should_promote` / the 1% MAPE improvement gate. This spec deliberately keeps the baseline comparison decoupled from the promotion decision.
- Adding baseline metrics to `gold_forecasts` or the dashboard (separate future spec if wanted).
- Any change to `08_promote_model.py`.
- Any change to `04_predict.py`, `02_transform.py`, `src/features.py`, `src/config.py`.
- Fixing the LGBM training/serving target-offset quirk noted in §7 (pre-existing, out of scope here).

---

## 3. Design

### 3.1 Baseline Definition — Seasonal-Naive, Matched Per Horizon

The baseline is defined by a single principle:

> **For a forecast target at time `T`, the baseline prediction is the observed value at `T − horizon`.**

This is the standard seasonal-naive (persistence) forecast: for a 24h horizon, "same hour yesterday"; for a 168h horizon, "same hour last week". No new feature engineering is needed — both lag columns already exist in `silver_features`.

Because the two model families label their test rows differently, the column mapping differs:

| Model | Test-row timestamp `t` | Target actual | Baseline prediction at row `t` | Column to use |
|---|---|---|---|---|
| `energy_prophet_24h` | `ds` | `y` = value(t) | value(t − 24h) | `lag_24h` |
| `energy_prophet_168h` | `ds` | `y` = value(t) | value(t − 168h) | `lag_168h` |
| `energy_lgbm_24h` | `timestamp` | value(t + 24h) (`target` via `shift(-24)`) | value(t) | `value_mwh` |
| `energy_lgbm_168h` | `timestamp` | value(t + 168h) (`target` via `shift(-168)`) | value(t) | `value_mwh` |

**Why the LGBM baseline is `value_mwh`, not `lag_24h`/`lag_168h`:** in `06_train_lgbm.py` the target is `value_mwh.shift(-horizon_hours)` (line 80), so a test row at `t` is scored against the actual at `t + horizon`. The forecast target time is therefore `T = t + horizon`, and the naive prediction is `value(T − horizon) = value(t)` — which is the row's own `value_mwh`. Using `lag_24h`/`lag_168h` at the test row would instead predict `value(t − horizon)` against a target of `value(t + horizon)`, i.e. a **2×horizon-ahead persistence** (48h / 336h), which is *not* the horizon-matched seasonal naive and would systematically flatter the models.

The "last known value regardless of lag" alternative (e.g. `value(t − 1h)`) is **not** recommended: hourly load has strong intra-day variation, so a 1h-old value predicts a 24h-ahead target far worse than the same hour yesterday. Seasonal-naive-per-horizon is the correct trivial baseline for this data.

### 3.2 Where the Baseline Is Computed (Training Notebooks, not `07_evaluate.py`)

The baseline MUST be computed on the exact test split each model already uses, so the comparison is apples-to-apples:

- `05_train_prophet.py`: `test_df` is built at line 91 (`df["ds"] > split_date`).
- `06_train_lgbm.py`: `test_df` is built at line 85 (`df_model["timestamp"] > split_date`), after the target shift and `dropna` (lines 80-81).

Only these notebooks have access to that exact split — `07_evaluate.py` reads MLflow only and cannot reproduce it. Hence the baseline metrics are computed and logged in `05`/`06`, and `07` merely reads them.

### 3.3 Shared Pure Function — `src/baseline.py` (new)

Extract the baseline metric math into a testable pure function (mirrors the pattern of `compute_prediction_mae` in `src/drift.py:50`):

```python
def compute_naive_baseline_metrics(
    actual: pd.Series, baseline_prediction: pd.Series
) -> dict[str, float]:
    """MAE/RMSE/MAPE of a naive baseline vs actuals.

    NaN-safe: pairs where either series is NaN are dropped before computing.
    MAPE follows the same formula as the notebooks' `calculate_mape`
    (mask actual != 0, guard against an all-zero mask).
    Returns {"naive_mae": float, "naive_rmse": float, "naive_mape": float}.
    """
```

- Both notebooks import and call this function, passing the series in §3.1's table (prophet: `test_df["y"]` vs the lag column; lgbm: `test_df["target"]` vs `test_df["value_mwh"]`).
- This gives a single source of truth for the three metrics and keeps the logic unit-testable without Spark/MLflow. The MAPE formula matches the notebooks' existing `calculate_mape` exactly (identical mask + `np.mean(...) * 100`), so the baseline MAPE is directly comparable to the logged model `mape`.
- The function is the required test seam; its MAPE formula must remain identical to each notebook's existing `calculate_mape` semantics.
- In the normal case the baseline row set equals the model's test row set (lags for the final 5 test days are always populated given the months of history; LGBM test rows are fully historical). Any residual NaN pair (e.g. an old gap hour) is dropped from the baseline metrics only — a deliberate, negligible approximation.

### 3.4 MLflow Logging Per Training Run

In both `05` and `06`, after the model's own metrics are logged (`05` lines 120-134, `06` lines 110-115):

- `mlflow.log_metrics(naive_metrics)` — logs `naive_mae`, `naive_rmse`, `naive_mape`.
- Add `naive_mae`/`naive_rmse`/`naive_mape` to the returned result dict (lines `05:152-160`, `06:143-148`) so the training summary print also shows them.

### 3.5 `07_evaluate.py` Changes

1. **`get_run_metrics()` (lines 64-107):** also pull `naive_mae`, `naive_rmse`, `naive_mape` from `run.data.metrics` and include them in the returned dict. (Challenger runs trained after this change will have them; older runs return `None`.)

2. **Eval row (lines 141-157):** add
   - `baseline_mae` ← `challenger["naive_mae"]`
   - `baseline_rmse` ← `challenger["naive_rmse"]`
   - `baseline_mape` ← `challenger["naive_mape"]`
   - `beats_baseline` ← `True/False/None` where
     `beats_baseline = (challenger["mape"] is not None and challenger["naive_mape"] is not None) and challenger["mape"] < challenger["naive_mape"]`

3. **`EVAL_SCHEMA` (lines 163-178):** add four nullable fields: `baseline_mae` (Double), `baseline_rmse` (Double), `baseline_mape` (Double), `beats_baseline` (Boolean). The existing `.option("mergeSchema", "true")` (line 181) already supports adding columns to a live table.

4. **Printed summary (lines 183-196):** show `baseline_mape` and `beats_baseline` alongside `challenger_mape`/`champion_mape`, and after the summary print a warning line for every model where `beats_baseline is False`, e.g.:
   ```
   WARNING: energy_lgbm_24h does NOT beat the naive baseline (model 6.2% vs baseline 5.8%).
   ```
   The "Final Recommendations" section remains driven **only** by `challenger_wins` — the baseline flag is an additional, visible warning, never a promotion input.

### 3.6 `beats_baseline` Is Report-Only

`beats_baseline` does not enter any promotion decision. `challenger_wins` (07) and `should_promote` (08, untouched) keep their exact current semantics. The only consumers of `beats_baseline` are the summary print and the `model_evaluation` table. This is a deliberate, narrowly-scoped coupling boundary.

### 3.7 Null Semantics

For challenger runs that predate this change (no `naive_*` metrics logged), `baseline_mae`/`baseline_rmse`/`baseline_mape` are `NULL` and `beats_baseline` is `NULL` (shown blank in the summary, no warning emitted). Every row written after this change (from runs trained by the updated `05`/`06`) has all four populated.

---

## 4. Acceptance Criteria

- [ ] **AC1**: Training runs log `naive_mae`, `naive_rmse`, `naive_mape` as real MLflow metrics, computed on the same test split as the model's own metrics (prophet: `lag_24h`/`lag_168h` on `test_df`; lgbm: `value_mwh` on `test_df`).
  - *Verification:* inspect `05_train_prophet.py` / `06_train_lgbm.py`; unit tests on `src/baseline.py`; (Databricks) `mlflow` run shows the three metrics on a fresh training run.
- [ ] **AC2**: `model_evaluation` table has `baseline_mape`, `baseline_mae`, `baseline_rmse`, `beats_baseline` columns populated for every new row.
  - *Verification:* `EVAL_SCHEMA` + eval-row mapping reviewed; `mergeSchema` keeps existing rows compatible.
- [ ] **AC3**: `07_evaluate.py`'s printed summary clearly shows whether each challenger beats the naive baseline, and prints an explicit warning for any `beats_baseline = False` model.
  - *Verification:* code review of summary/print section.
- [ ] **AC4**: Existing promotion behaviour is unchanged — `08_promote_model.py` untouched, `challenger_wins` logic unchanged, `tests/test_ingest_logic.py` and all existing tests pass.
  - *Verification:* `pytest` green; diff shows no changes to `08_promote_model.py` or promotion decision code.
- [ ] **AC5**: Unit tests added for the baseline calculation logic (`src/baseline.py`), covering happy path, NaN-pair dropping, and the zero-actual MAPE guard. `ruff`, `mypy src/`, and `pytest --cov-fail-under=80` all pass.
  - *Verification:* `tests/test_baseline.py`; run `pre-commit`, `mypy src/`, `pytest`.

---

## 5. Approved Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | LGBM uses **`value_mwh` at the test row**; Prophet uses `lag_24h` / `lag_168h`. | LGBM scores `value(t + horizon)`, so `value(t)` is its horizon-matched seasonal-naive prediction. |
| 2 | Legacy runs write `NULL` baseline fields and `beats_baseline`. | Newly trained challengers must have all baseline fields populated. |

---

## 6. Implementation Plan (for reference during coding)

### 6.1 Files to Change

| File | Change |
|---|---|
| `src/baseline.py` (new) | `compute_naive_baseline_metrics(actual, baseline_prediction)` → `{"naive_mae", "naive_rmse", "naive_mape"}`; NaN-safe; MAPE identical to notebooks' `calculate_mape`; typed for mypy. |
| `notebooks/05_train_prophet.py` | Import `compute_naive_baseline_metrics`; after logging model metrics compute `naive_metrics` on `test_df["y"]` vs `test_df["lag_24h"|"lag_168h"]`; `mlflow.log_metrics(...)`; extend return dict. |
| `notebooks/06_train_lgbm.py` | Import `compute_naive_baseline_metrics`; after logging model metrics compute `naive_metrics` on `test_df["target"]` vs `test_df["value_mwh"]`; `mlflow.log_metrics(...)`; extend return dict. |
| `notebooks/07_evaluate.py` | `get_run_metrics` pulls `naive_mae`/`naive_rmse`/`naive_mape`; eval rows add `baseline_mae`/`baseline_rmse`/`baseline_mape`/`beats_baseline`; extend `EVAL_SCHEMA` (nullable, `mergeSchema` already on); summary print shows baseline + flags `beats_baseline = False`. |
| `tests/test_baseline.py` (new) | Unit tests: happy path (known MAE/RMSE/MAPE), NaN-pair dropping, all-zero actual guard. |

### 6.2 Files NOT to Change

- `notebooks/08_promote_model.py` (promotion logic untouched).
- `notebooks/01_ingest.py`, `02_transform.py`, `03_drift_check.py`, `04_predict.py`.
- `src/config.py`, `src/features.py`, `src/drift.py`, `src/tuning.py`, `GEMINI.md`.
- `tests/test_ingest_logic.py` and all other existing tests (only additive new tests).

### 6.3 Rollout Sequence

1. Implement per spec; run `pre-commit` (ruff), `mypy src/`, `pytest --cov-fail-under=80`.
2. Merge; deploy the retraining job (`databricks.yml`) with the updated notebooks.
3. Next retraining run: verify the three `naive_*` metrics appear in MLflow for all four models; verify `07_evaluate` writes populated `baseline_*`/`beats_baseline` rows and prints the summary + warnings.
4. Results may be published to README only once `beats_baseline` is True for all four models (or the failure is explicitly documented).

---

## 7. Future Spec Candidates (Not Implemented Here)

- **Baseline in `gold_forecasts` / dashboard**: expose a naive-baseline series alongside model forecasts for visual comparison (currently out of scope).
- **LGBM target-offset note**: `06_train_lgbm.py` trains `features(t) → value(t + horizon)` while `04_predict.py` builds features at future timestamps `t` and stores the prediction under `t`; whether this creates a `+horizon` labeling offset in `gold_forecasts` for LGBM deserves a separate investigation spec before any gold-layer baseline work.
- **Last-known-value baseline** as a second, cheaper reference column (rejected for the 24h/168h MAPE gate per §3.1 rationale; could be logged as extra context if ever needed).
