# 010 — Predict-Path Timezone Regression and Target-Construction Dtype Upcast

**Status:** Draft
**Author:** Technical Architect
**Date:** 2026-10-05
**Priority:** Critical
**Dependencies:** Problem A was dormant until spec 009's remediation (2026-09-27/28) cleared the LGBM `production` tag, which is what first forced `04_predict.py`'s fallback onto the Prophet branch — spec 009 didn't cause the bug, but is what exposed it. Problem B is a genuine regression introduced by spec 007's `build_horizon_target()`, first exercised by the 2026-10-04 02:00 UTC retraining (the first run after spec 007 deployed with a sufficiently gapped `silver_features`). Both block spec 009's gate from ever being exercised by a real run — this spec's resolution is the direct precondition for finally observing spec 009's `first_run=True` promotion logic live.

---

## 1. Problem Statement

### 1.1 Problem A — Prophet Serving Path Never Strips Timezone

`notebooks/05_train_prophet.py` trains on tz-naive timestamps — `pdf["ds"] = pd.to_datetime(pdf["ds"]).dt.tz_localize(None)`. `notebooks/04_predict.py::generate_forecasts()`'s Prophet branch has no equivalent: `p_df = X.reset_index().rename(columns={"timestamp": "ds"})` inherits `X`'s tz-aware (UTC) `DatetimeIndex` straight from `prepare_inference_features()` (itself built from `datetime.now(UTC)`), and passes it to `model.predict(p_df)` unmodified. Prophet rejects any tz-aware `ds` outright.

This was never caught because it was never *exercised*: `gold_forecasts` has zero Prophet rows in its entire history. Confirmed via MLflow/job history: the last fully successful `predict` task (2026-09-30 20:05 UTC) served LGBM; the very next run (2026-09-30 20:18 UTC) — thirteen minutes after spec 009's deploy (20:12 UTC) cleared the LGBM `production` tag — fell to the Prophet fallback for the first time and failed immediately with `Column ds has timezone specified, which is not supported. Remove timezone.` Confirmed by local reproduction mirroring the exact repo code.

### 1.2 Problem B — `build_horizon_target()` Upcasts Unrelated Columns on Any Gap

Spec 007's `build_horizon_target()` (`src/target_construction.py`) reindexes the **entire** input frame to a complete hourly grid before computing the shifted target — correct and necessary for the target column itself (spec 007's whole point), but a side effect nobody specified: any gap inserts an all-`NaN` row across **every** column, which upcasts typed columns (e.g. a `bool` column, since a plain NumPy boolean array cannot hold `NaN`) to `object`. Filtering the gap rows back out afterward does not revert the dtype.

`silver_features` currently has 152 missing hours. The 2026-10-04 02:00 UTC retraining — the first to run against `build_horizon_target()` with enough accumulated gaps to trigger this — failed `train_lgbm`: `pandas dtypes must be int, float or bool. Fields with bad pandas dtypes: is_weekend: object, is_holiday: object.` Confirmed by local reproduction: the same input produces `bool` dtypes with no gaps and `object` dtypes with a gap present. Live data confirms zero actual `NULL`s in either column — this is a dtype artifact of the reindex mechanism, not a data-quality defect.

### 1.3 Consequence — Spec 009's Gate Has Never Actually Run

`databricks.yml`'s `evaluate` task depends on both `train_prophet` and `train_lgbm` under the default `ALL_SUCCESS` condition. The 2026-10-04 run: `train_prophet` SUCCESS, `train_lgbm` FAILED (×3 attempts), `evaluate` and `promote_model` both SKIPPED/UPSTREAM_FAILED. Spec 009's `leak_affected`-exclusion gate-unblock logic — the entire point of that spec — has not been exercised by a single real run since it was deployed.

---

## 2. Scope

### 2.1 In Scope

1. **Problem A fix**: strip timezone from `p_df["ds"]` in `04_predict.py::generate_forecasts()`'s Prophet branch before `model.predict()`, mirroring `05_train_prophet.py`'s training-time `tz_localize(None)` exactly.
2. **Problem B fix**: rewrite `build_horizon_target()` to reindex and shift only the `value_mwh` series needed to compute the target, then map the result back onto the original (untouched) rows — so no column other than the new `target` is ever affected by the gap-filling reindex.
3. Tests for both, extending the existing test harnesses from specs 006 (`tests/test_predict_timing_integration.py`) and 007 (`tests/test_target_construction.py`) rather than duplicating them.

### 2.2 Out of Scope

- **Problem C** (`databricks.yml`'s `ALL_SUCCESS` evaluate/promote gate skipping both models when only one training task fails) — a real resilience gap, explicitly deferred to §7 by agreement: fixing A and B restores `train_lgbm` to passing, which removes the immediate blocker without changing the gate's dependency semantics. Revisit only if a future cycle has one model family fail while the other succeeds and the gate skip matters again.
- Any change to `src/features.py`'s feature math, `src/config.py`, `src/splits.py`, `src/baseline.py`, `src/forecast_timing.py`, `src/ingest_discovery.py`, `src/silver_window.py`.
- Any change to `01_ingest.py`, `02_transform.py`, `03_drift_check.py`, `07_evaluate.py`, `08_promote_model.py`, `databricks.yml`.
- Re-deriving or re-verifying spec 007's core correctness contract (gapless-matches-naive-shift; shuffled-input-matches-sorted; NaN-only-at-exact-target-on-gap) — unchanged by this fix, only re-asserted as a regression guard (§4 AC4a).
- Manually triggering a retraining run — tracked as an explicit, separately-approved rollout step (§6.3), not part of implementation.

---

## 3. Design

### 3.1 Problem A — Before/After

**Before (`04_predict.py::generate_forecasts()`, Prophet branch):**

```python
else:  # Prophet
    p_df = X.reset_index().rename(columns={"timestamp": "ds"})
    forecast = model.predict(p_df)
    preds = forecast["yhat"].clip(lower=0).values
```

**After:**

```python
else:  # Prophet
    p_df = X.reset_index().rename(columns={"timestamp": "ds"})
    # Prophet rejects tz-aware ds outright; 05_train_prophet.py strips tz
    # at training time (tz_localize(None)) — this path never had the
    # equivalent, and was never exercised against it until spec 009's
    # remediation first forced LGBM-fallback traffic onto Prophet
    # (spec 010 §1.1).
    p_df["ds"] = pd.to_datetime(p_df["ds"]).dt.tz_localize(None)
    forecast = model.predict(p_df)
    preds = forecast["yhat"].clip(lower=0).values
```

The LGBM branch is untouched.

### 3.2 Problem B — Before/After

**Before (`src/target_construction.py`):**

```python
def build_horizon_target(df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    tz = df["timestamp"].dt.tz
    full_range = pd.date_range(start=df["timestamp"].min(), end=df["timestamp"].max(), freq="h", tz=tz)
    original_timestamps = set(df["timestamp"])
    df = df.set_index("timestamp").reindex(full_range)
    df["target"] = df["value_mwh"].shift(-horizon_hours)
    df = df.reset_index().rename(columns={"index": "timestamp"})
    df = df[df["timestamp"].isin(original_timestamps)]
    return df
```

**After:**

```python
def build_horizon_target(df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    tz = df["timestamp"].dt.tz
    full_range = pd.date_range(start=df["timestamp"].min(), end=df["timestamp"].max(), freq="h", tz=tz)

    # Reindex and shift ONLY the value series needed for the target.
    # Reindexing the WHOLE frame (the prior implementation) inserts an
    # all-NaN row for every gap hour across every column — including typed
    # columns like bool, which a plain NumPy array can't hold NaN in, so
    # they upcast to object. Filtering the gap rows back out afterward
    # does not revert that upcast (spec 010 §1.2). Mapping the target back
    # onto the original, never-reindexed rows touches no column but the
    # new `target` itself.
    value_by_ts = df.set_index("timestamp")["value_mwh"]
    target_by_ts = value_by_ts.reindex(full_range).shift(-horizon_hours)
    df["target"] = df["timestamp"].map(target_by_ts)

    return df
```

This produces an identical `target` column to the prior implementation for every row (same reindex-then-shift arithmetic, same NaN-only-at-missing-target-hour contract from spec 007) — the only change is that no column besides `target` is ever derived from the reindexed grid.

---

## 4. Acceptance Criteria

**Problem A**
- [ ] **AC1**: `generate_forecasts()`'s Prophet branch strips `ds`'s timezone before `model.predict()`, per §3.1.
- [ ] **AC2**: Extend `tests/test_predict_timing_integration.py`: (a) a test proving the Prophet branch calls `predict()` with a tz-naive `ds` — using a fake Prophet model whose `predict()` raises on a tz-aware input (mirroring real Prophet's actual failure mode), so the test would have failed against the pre-fix code, not just asserted the fix's own output; (b) confirm the LGBM branch's behavior is byte-identical to before (regression guard).

**Problem B**
- [ ] **AC3**: `build_horizon_target()` reindexes/shifts only `value_mwh`, mapping the result onto the original rows, per §3.2.
- [ ] **AC4**: Extend `tests/test_target_construction.py`: (a) spec 007's existing three guarantees (gapless-matches-naive-shift; shuffled-matches-sorted; gap produces NaN only at the exact target-on-gap anchor) still pass unchanged — this fix must not regress spec 007's correctness contract; (b) a new test with a gapped input containing a `bool` column proves the output's `bool` column dtype stays `bool`, and explicitly demonstrates — on the same input — that reindexing the whole frame (the pre-fix approach) does upcast it to `object`, proving this fix addresses a real, reproducible defect and not a hypothetical one.

**Shared**
- [ ] **AC5**: `ruff`, `mypy src/`, `pytest --cov-fail-under=80` all pass.
- [ ] **AC6**: Out-of-scope files untouched — `src/features.py`, `src/config.py`, `src/splits.py`, `src/baseline.py`, `src/forecast_timing.py`, `src/ingest_discovery.py`, `src/silver_window.py`, `notebooks/01_ingest.py`, `notebooks/02_transform.py`, `notebooks/03_drift_check.py`, `notebooks/07_evaluate.py`, `notebooks/08_promote_model.py`, `databricks.yml` all byte-identical to `main`.

---

## 5. Open Questions / Decisions Needed

| # | Question | Options | Recommendation |
|---|---|---|---|
| 1 | **Address Problem C (`ALL_SUCCESS` gate) now?** | (a) Now; (b) defer | **(b) Defer (§7)** — already agreed; fixing A and B removes the immediate blocker, and the gate's dependency semantics are a separate resilience question worth its own spec once there's a concrete case (one family failing, the other not) to design against. |
| 2 | **Add a defensive dtype cast immediately before `model.fit()` in `06_train_lgbm.py`, as a second line of defense?** | Yes / No | **Yes, lightweight** — a one-line `X_train = X_train.astype({c: bool for c in BOOLEAN_FEATURE_COLS})`-style cast (or equivalent) costs little and would have caught this class of defect regardless of where an upstream dtype corruption originates, not just this specific `reindex` cause. Worth adding in this same PR since the fix is already in context; not a hard requirement if it meaningfully widens the diff. |
| 3 | **Manually trigger a retraining run after this deploys, or wait for the 2026-10-11 02:00 UTC native schedule?** | (a) Manual trigger, with sign-off; (b) wait | **(a)** — spec 009's entire gate-unblock logic has not been exercised by a real run since it deployed two weeks ago; waiting another week to find out whether it actually works end-to-end is a real cost for a low-risk, reversible, explicitly-approved manual trigger. |

---

## 6. Implementation Plan

### 6.1 Files to Change

| File | Change |
|---|---|
| `notebooks/04_predict.py` | `generate_forecasts()`'s Prophet branch: strip tz from `p_df["ds"]`, per §3.1. |
| `src/target_construction.py` | `build_horizon_target()`: reindex/shift only `value_mwh`, map target back, per §3.2. |
| `tests/test_predict_timing_integration.py` | Extend with AC2's tests (reuse the existing AST-extraction harness from spec 006). |
| `tests/test_target_construction.py` | Extend with AC4's tests (reuse spec 007's existing harness). |
| `specs/010-predict-tz-and-target-dtype-regressions/spec.md` | This spec. |

### 6.2 Files NOT to Change

- `notebooks/01_ingest.py`, `notebooks/02_transform.py`, `notebooks/03_drift_check.py`, `notebooks/05_train_prophet.py`, `notebooks/07_evaluate.py`, `notebooks/08_promote_model.py`.
- `src/features.py`, `src/config.py`, `src/splits.py`, `src/baseline.py`, `src/forecast_timing.py`, `src/ingest_discovery.py`, `src/silver_window.py`, `src/tuning.py`.
- `databricks.yml`, `.github/workflows/*`.

### 6.3 Rollout Sequence

1. Implement per §3; run `ruff`, `mypy src/`, `pytest --cov-fail-under=80`.
2. Merge; confirm `deploy.yml` succeeded on both `prod` and `dev` targets.
3. Per §5 item 3, with explicit sign-off: manually trigger `energy_retraining_pipeline`. Confirm `train_prophet` and `train_lgbm` both SUCCEED, `evaluate` and `promote_model` both actually RUN (not SKIPPED), and — the real payoff — that `energy_lgbm_24h`/`168h` show `first_run=True` and are promoted, carrying `leak_affected="false"`, finally exercising spec 009's gate end-to-end.
4. Confirm `04_predict.py` subsequently serves the newly-promoted LGBM model (not Prophet) without error.
5. Only then, republish the README's LightGBM Results rows (currently "withdrawn").

---

## 7. Future Spec Candidates (Not Implemented Here)

- **`ALL_SUCCESS` gate resilience (Problem C)**: decide whether `evaluate`/`promote_model` should proceed per-model-family when only one of `train_prophet`/`train_lgbm` fails, rather than skipping both — revisit with a concrete case once one occurs.
- **General dtype-contract enforcement before `model.fit()`**: a shared assertion/cast helper validating `X_train`'s dtypes against `MODEL_INPUT_FEATURES`' expected types, reusable beyond this specific `reindex`-induced defect.
