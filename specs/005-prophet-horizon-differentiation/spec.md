# 005 — Prophet Horizon Differentiation

**Status:** Draft
**Author:** Technical Architect
**Date:** 2026-08-23
**Priority:** High
**Dependencies:** Spec 002 (naive-baseline pairing per horizon must keep working unchanged — `baseline_column = f"lag_{horizon_hours}h"` logic is untouched); independent of specs 003/004 (LightGBM-only changes, Prophet was explicitly out of scope for both)

---

## 1. Problem Statement

### 1.1 The Bug

`train_prophet_model()` in `notebooks/05_train_prophet.py` accepts `horizon_hours` as a parameter but never uses it to affect the train/test split, the fit, or the forecast target:

```python
def train_prophet_model(df: pd.DataFrame, horizon_hours: int, model_name: str):
    split_date = df["ds"].max() - pd.Timedelta(days=CONFIG["test_days"])
    train_df = df[df["ds"] <= split_date].copy()
    test_df = df[df["ds"] > split_date].copy()
    ...
    model.fit(train_df)
    forecast = model.predict(test_df[["ds", "temperature_c"]])
    y_true = test_df["y"].values
    y_pred = forecast["yhat"].values
```

`split_date`, `train_df`, and `test_df` depend only on `CONFIG["test_days"]`. `energy_prophet_24h` and `energy_prophet_168h` are trained on identical data and evaluated on identical timestamps — the only difference between the two calls is the string passed as `model_name`. As a result:

- `y_pred` (and therefore `mae`, `rmse`, `mape`) are **byte-identical** across both horizons on every training run.
- The only value that legitimately differs between the two "models" is `naive_mape` (via `baseline_column = f"lag_{horizon_hours}h"` in `compute_naive_baseline_metrics`), because that logic is separate from the model fit itself.

Confirmed on live data: run `1eb4e34776a4463cb6f91c04325cda19` (`energy_prophet_24h`) and `ce02f00b52c945d8b643beb843503dfd` (`energy_prophet_168h`), both from parent run `prophet_training_20260823_0201`, report `mae=471.25`, `rmse=589.01` — identical to at least 2 decimal places — for both horizons. This pattern was confirmed present across the two prior retraining cycles as well, meaning it is not new; it has been present since `05_train_prophet.py` was written.

### 1.2 Why This Matters

There is no 168h-ahead Prophet forecast anywhere in the current pipeline, despite the README, `GEMINI.md`, and the champion/challenger architecture describing four independently-evaluated models (2 model families × 2 horizons). This is a correctness defect that produces misleading published metrics, not a cosmetic issue — it was caught only because `naive_mape` (which *does* vary correctly by horizon) exposed the inconsistency during a README-publication review.

### 1.3 Why LightGBM Doesn't Have This Problem

`06_train_lgbm.py` builds a horizon-specific target via `df_model["target"] = df_model["value_mwh"].shift(-horizon_hours)` — the model is a **direct multi-step forecaster**: it literally learns to predict `value(t + horizon_hours)` from `features(t)`. Prophet has no equivalent target-shifting step; it's a continuous-time model that forecasts whatever `ds` timestamps it's asked to predict, so `horizon_hours` was never plumbed into anything that would make the two calls diverge.

---

## 2. Scope

### 2.1 In Scope

1. Make Prophet's train/test split **horizon-aware**, so that `energy_prophet_24h` and `energy_prophet_168h` are genuinely different fits evaluated in a way that reflects their stated forecast horizon.
2. Extract the new split logic into a small, pure, unit-testable function (pattern: `src/splits.py`'s `make_holdout_splits`, per spec 004's precedent).
3. Preserve the naive-baseline pairing exactly as-is (`baseline_column = f"lag_{horizon_hours}h"`, `compute_naive_baseline_metrics(test_df["y"], test_df[baseline_column])`) — no change to `src/baseline.py` or its call signature.
4. Add an explicit acceptance criterion / test asserting the two horizons produce **non-identical** `y_pred` arrays on real or synthetic data — this is the regression test that would have caught the original bug and must never silently pass again.
5. Rollout step: after merge, the next scheduled retraining run must produce two genuinely distinct Prophet metric sets before any Prophet number is republished to the README (mirrors spec 004 §3.6's rollout discipline).

### 2.2 Out of Scope

- Any change to `LGBM_PARAMS`, `06_train_lgbm.py`, `src/splits.py`'s existing `make_holdout_splits` (LightGBM's function is untouched; this spec adds a new, separate function for Prophet).
- Any change to `07_evaluate.py`, `08_promote_model.py`, or the champion/challenger promotion logic.
- Any change to `src/baseline.py` or `compute_naive_baseline_metrics()`.
- Re-tuning Prophet's hyperparameters (`changepoint_prior_scale`, `seasonality_mode`) — out of scope; only the split/evaluation logic changes.
- Retroactively correcting historical `model_evaluation` rows or MLflow runs computed under the old (bugged) split.
- Deciding whether Prophet 168h should beat the naive baseline post-fix — that's an empirical outcome to observe, not a target to engineer toward.

---

## 3. Design

### 3.1 Horizon-Aware Split via Training Gap

Introduce a `horizon_hours`-sized gap between the training cutoff and the test window, so training data available to the model always ends at least `horizon_hours` before the earliest test timestamp — mirroring the gap convention already established elsewhere in this codebase (`OPTUNA_GAP_HOURS=168` in `src/tuning.py`'s CV folds).
test window : (t_max - test_days, t_max] — same absolute dates for both horizons
train window : [t_min, test_start - horizon_hours] — cutoff shifts earlier as horizon grows

Where `test_start = t_max - pd.Timedelta(days=test_days)`.

This has two effects:
1. **The test window (the dates being scored) stays identical across horizons** — so 24h vs 168h accuracy is being compared on the same real-world period, which is the fair comparison a README reader would assume.
2. **The training cutoff differs by horizon** — the 168h model's `train_df` excludes the final 7 days of history that the 24h model gets to see, forcing Prophet to demonstrate it can generalize `horizon_hours` into the future without having trained on anything inside that gap. This is the same principle as `make_timeseries_splits`'s gap parameter, applied to Prophet instead of the CV loop.

Concretely:

```python
# src/splits.py — new function, alongside the existing make_holdout_splits
def make_prophet_holdout_split(
    timestamps: pd.Series,
    horizon_hours: int,
    test_days: int,
) -> tuple[pd.Series, pd.Series]:
    """Returns (train_mask, test_mask) for Prophet's horizon-aware split.

    test  : (t_max - test_days, t_max]                    — fixed test window
    train : [t_min, test_start - horizon_hours]            — cutoff recedes with horizon

    Unlike make_holdout_splits (LightGBM's 3-way split), this leaves a gap
    of exactly `horizon_hours` between train and test, rather than carving
    out a validation window — Prophet has no early-stopping mechanism to
    protect, so the gap exists solely to prevent the model from training on
    data inside its own forecast lead time.
    """
    t_max = timestamps.max()
    test_start = t_max - pd.Timedelta(days=test_days)
    train_cutoff = test_start - pd.Timedelta(hours=horizon_hours)

    train_mask = timestamps <= train_cutoff
    test_mask = timestamps > test_start

    return train_mask, test_mask
```

Note this deliberately does **not** guarantee `train_mask | test_mask` covers every row (there's an intentional gap of unused rows between them, unlike `make_holdout_splits`'s exhaustive 3-way partition) — that gap is the point.

### 3.2 `train_prophet_model()` Changes

```python
# Before
split_date = df["ds"].max() - pd.Timedelta(days=CONFIG["test_days"])
train_df = df[df["ds"] <= split_date].copy()
test_df = df[df["ds"] > split_date].copy()

# After
train_mask, test_mask = make_prophet_holdout_split(
    df["ds"], horizon_hours, CONFIG["test_days"]
)
train_df = df[train_mask].copy()
test_df = df[test_mask].copy()
```

Everything downstream (`model.fit(train_df)`, `model.predict(test_df[["ds","temperature_c"]])`, metric computation, naive-baseline call, MLflow logging) stays structurally identical — only where `train_df`/`test_df` come from changes.

### 3.3 Low-Data-Volume Behavior

The existing guard (`if len(train_df) < CONFIG["min_train_rows"]: raise ValueError(...)`) is preserved unchanged and now naturally accounts for the shrinking 168h train window — no new threshold logic needed. At current data volume (~1,900+ silver rows), a 7-day (168h) gap removes ~168 rows from the 168h model's training set relative to the 24h model's — trivial against a `min_train_rows=200` floor at this volume, but worth a row-count sanity check in the unit tests (§4, AC3) so this doesn't silently break as historical data window size changes.

### 3.4 Regression Test — The Bug Must Not Silently Return

The single most important test in this spec is the one that would have caught the original bug:

```python
def test_24h_and_168h_produce_different_forecasts():
    """The exact regression this spec exists to prevent: horizon_hours must
    actually affect the fitted model, not just the naive-baseline column."""
    # fit two Prophet models on the same df with horizon_hours=24 vs 168,
    # assert their train_df row counts differ by ~168 rows (or at least
    # differ deterministically), and assert forecast["yhat"] arrays for
    # the shared test window are NOT element-wise identical.
```

This should live in `tests/test_splits.py` (mask-level: train windows differ, gap is `horizon_hours`-sized) and, if practical without a full Prophet fit in CI, a lighter integration-style check in a new `tests/test_prophet_split_integration.py` or equivalent that at minimum asserts the two `train_df` row counts differ — the full non-identical-forecast assertion may need to run as a manual verification step on Databricks (Prophet fits are slow; not necessarily CI-friendly) but should still be documented as a required rollout check (§3.5).

### 3.5 Rollout & Republication

Mirrors spec 004 §3.6's discipline: after merge, the next scheduled `energy_retraining_pipeline` run must produce two Prophet MLflow runs whose `mae`/`rmse`/`mape` are **not identical** before any Prophet number is republished to the README. The README's Prophet rows stay `_TBD_` (already reverted per the prior task) until that verification passes.

---

## 4. Acceptance Criteria

- [ ] **AC1**: `make_prophet_holdout_split()` exists in `src/splits.py`, is pure and typed, and returns `(train_mask, test_mask)` where the gap between the latest `True` in `train_mask` and the earliest `True` in `test_mask` equals exactly `horizon_hours`.
  - *Verification:* `tests/test_splits.py` — new test asserting the gap arithmetic for both horizon_hours=24 and =168.
- [ ] **AC2**: `test_mask` covers the same absolute date range regardless of `horizon_hours` (only `train_mask` changes with horizon).
  - *Verification:* unit test comparing `test_mask`-selected timestamps across two calls with different `horizon_hours`, same `timestamps`/`test_days` — must be identical.
- [ ] **AC3**: At current data volume (~1,900 silver rows), both the 24h and 168h train windows stay comfortably above `CONFIG["min_train_rows"]` (200).
  - *Verification:* unit test with a synthetic ~1,900-row hourly timestamp series; assert `train_mask.sum() >= 200` for both horizons.
- [ ] **AC4**: `train_prophet_model()` uses `make_prophet_holdout_split()` instead of the flat date-based split; naive-baseline call signature and `compute_naive_baseline_metrics()` are untouched.
  - *Verification:* code review of the diff; `tests/test_baseline.py` stays green and unchanged.
- [ ] **AC5**: The two horizons produce genuinely different forecasts — this is the core regression test for the bug this spec fixes.
  - *Verification:* a test (integration-level, may require an actual Prophet fit on synthetic seasonal data) asserting `forecast_24h["yhat"].values` and `forecast_168h["yhat"].values` are NOT element-wise equal for the shared test window. If a full-fit test is impractical for CI runtime, this must instead be a documented manual verification step performed on the first post-merge Databricks retraining run (see §3.5), with the run's two `mae`/`rmse` values confirmed different as the pass condition.
- [ ] **AC6**: `ruff`, `mypy src/`, and `pytest --cov-fail-under=80` all pass.
- [ ] **AC7**: Out-of-scope files untouched — `06_train_lgbm.py`, `src/splits.py`'s existing `make_holdout_splits`, `src/baseline.py`, `src/tuning.py`, `07_evaluate.py`, `08_promote_model.py` all byte-identical to `main`.
  - *Verification:* `git diff main..feature/005-prophet-horizon-differentiation` touches only `notebooks/05_train_prophet.py`, `src/splits.py` (additive function only), test files, and this spec.
- [ ] **AC8**: Rollout — the next scheduled retraining run produces two non-identical Prophet metric sets; README Prophet rows stay `_TBD_` until this is confirmed.

---

## 5. Open Questions / Decisions Needed

| # | Question | Options | Recommendation |
|---|---|---|---|
| 1 | Gap-based split (this spec) vs. Prophet's native multi-step forecast (`make_future_dataframe(periods=horizon_hours)`)? | (a) Gap-based train/test split; (b) native future-dataframe forecasting | **(a)** — smaller, more contained change; keeps the existing `test_df`-based evaluation shape (compatible with spec 002's baseline pairing) rather than restructuring how Prophet is queried. (b) is worth a future spec if (a) proves insufficient. |
| 2 | Should the CI-unfriendly AC5 full-fit test block merge, or is manual rollout verification acceptable? | (a) Require the integration test in CI; (b) manual verification on first live run | **(b)**, given Prophet fit time — but the manual check is a hard AC8 gate on README republication, not optional. |
| 3 | Is a `_24h`/`_168h` naming split still meaningful for Prophet once fixed, or should it collapse to one model? | (a) Keep both, now genuinely differentiated; (b) collapse to one `energy_prophet` | **(a)** — this spec's entire point is making the existing two-horizon architecture true, not abandoning it. |

---

## 6. Implementation Plan

### 6.1 Files to Change

| File | Change |
|---|---|
| `src/splits.py` | Add `make_prophet_holdout_split(timestamps, horizon_hours, test_days) -> (train_mask, test_mask)` per §3.1. Existing `make_holdout_splits` untouched. |
| `notebooks/05_train_prophet.py` | Replace the flat date-based split (lines ~90-92) with a call to `make_prophet_holdout_split`. No other structural change. |
| `tests/test_splits.py` | Add tests per AC1-AC3 for the new function. |
| `specs/005-prophet-horizon-differentiation/spec.md` | This spec. |

### 6.2 Files NOT to Change

- `notebooks/06_train_lgbm.py`, `src/splits.py`'s existing `make_holdout_splits`, `src/baseline.py`, `src/tuning.py`, `src/config.py`, `07_evaluate.py`, `08_promote_model.py`, `src/features.py`, `databricks.yml`.

### 6.3 Rollout Sequence

1. Implement per §3; run `ruff`, `mypy src/`, `pytest --cov-fail-under=80`.
2. Merge to `main`; next scheduled `energy_retraining_pipeline` run (or manual `workflow_dispatch`) retrains both Prophet models.
3. Verify via MLflow that `energy_prophet_24h` and `energy_prophet_168h`'s `mae`/`rmse`/`mape` are no longer identical.
4. Only then, republish Prophet's README rows — reusing the same verification-prompt pattern from spec 004's rollout (naive_mape present, imputation check, now plus the non-identical-forecast check).

---

## 7. Future Spec Candidates (Not Implemented Here)

- **Native multi-step Prophet forecasting** (`make_future_dataframe`) as an alternative to the gap-based split, if the gap approach turns out not to differentiate the horizons meaningfully in practice.
- **Prophet regressor lag-awareness**: currently `model.add_regressor("temperature_c")` uses same-timestamp temperature at predict time, which for a genuine 168h-ahead forecast may not be realistically available — worth revisiting once §1 of this spec's fix is live, analogous to `04_predict.py`'s existing 7-day-lag weather proxy strategy for LightGBM.