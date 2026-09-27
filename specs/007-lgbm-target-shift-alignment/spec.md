# 007 — LightGBM Target Construction: Positional Shift vs. Time-Based Shift

**Status:** Draft
**Author:** Technical Architect
**Date:** 2026-09-21
**Priority:** High
**Dependencies:** Spec 004 (the `max(MIN_TRAINING_ROWS, CONFIG["min_train_rows"])` guard this fix relies on as a safety net; the 3-way split whose row-count arithmetic is affected by this fix's row-count reduction). Independent of Spec 006 — different files (006: `04_predict.py`, inference-path timestamp labeling; 007: `06_train_lgbm.py`, training-path target construction) — but the same root-cause class: pandas `.shift()` semantics assumed to be time-based when they are actually positional.

---

## 1. Problem Statement

### 1.1 The Bug

`notebooks/06_train_lgbm.py::train_lgbm_model()` (line 93) builds the direct-multi-step training target with:

```python
df_model["target"] = df_model["value_mwh"].shift(-horizon_hours)
```

`pandas.Series.shift(n)` is **positional** — it moves values `n` rows, not `n` time units. It is only equivalent to an `n`-hour lookahead if the frame is (a) sorted ascending by timestamp and (b) has no missing hourly rows in the shift window. Neither is enforced here:

- `pdf = spark.read.table(CONFIG["silver_table"]).toPandas()` (line 186) has no `.orderBy("timestamp")`; Spark does not guarantee row order without an explicit sort, and `silver_features` is a heavily-compacted, deletion-vector-enabled Delta table where file/scan order has no relationship to chronological order.
- No `.sort_values("timestamp")` or reindex-to-complete-hourly-range precedes the `shift()` call.

This is the **only unguarded `shift()` in the repository**. Every other one is preceded by an explicit sort and reindex — `src/features.py::add_lag_features()` sorts, reindexes to a complete hourly `pd.date_range`, then shifts, with the comment *"Ensure complete hourly index to prevent shift() misalignment"* — and only afterward restores the original rows. `06_train_lgbm.py`'s target construction does not follow its own codebase's established pattern.

### 1.2 Confirmed Impact (Investigation, 2026-09-21)

Live compute access was unavailable (SQL warehouse stopped, cluster creation blocked). The investigation used UC table statistics, the raw-ingestion Volume archive (one file per processed hour — a faithful proxy for the true hourly sequence), and past job outputs instead. Two separate findings, at different confidence levels:

**Measured (best case — assumes row order happens to be correct):**

| Horizon | Rows misaligned by real gaps | % of 3,234-row history | vs. `MIN_TRAINING_ROWS` (720) | Realized lead time (should be exact) |
|---|---|---|---|---|
| 24 h | 132 | 4.1% | 18.3% of the floor | 25 h – 72 h |
| 168 h | 276 | 8.5% | 38.3% of the floor | 169 h – 312 h |

All 144 underlying missing hours fall in the documented 2026-07-03…07-13 GitHub Actions outage window (spec 001). **This directly contradicts spec 001 §9's claim that these gaps were "aging out naturally"**: `06_train_lgbm.py` reads `silver_features` with **no lookback window** (unlike `02_transform.py`/`03_drift_check.py`, which use bounded windows) — the table accumulates all history permanently, and old gaps never leave the training model's view. This is a documentation correction this spec surfaces, not something spec 001 itself needs to be edited to fix (specs are dated, historical records in this project's convention).

**Demonstrated but not measured live (order effect — Spark's actual scan order is unknown; compute was unavailable to check):** a synthetic reproduction shuffling row order showed positional `shift(-24)` computing the wrong value for 78/120 rows (65%), vs. 0 wrong once sorted + reindexed first. This is architecturally plausible given Delta's file layout, but **it is a demonstrated risk, not a measured fact about the live table** — stated plainly rather than folded into the "confirmed" gap numbers above.

### 1.3 What This Does *Not* Affect

- `notebooks/05_train_prophet.py` — Prophet trains directly on `y = value(ds)`, no `shift()` anywhere; unaffected.
- `src/splits.py::make_holdout_splits()` — splits by boolean timestamp comparison (`timestamps <= val_split_date`), not position; order-independent, produces correct masks regardless of input row order. (Its docstring's "sorted, ascending" precondition is a documentation nicety, not an actual dependency — noted here so it isn't conflated with the real bug.)
- `src/tuning.py` — consumes whatever `(X_train, y_train)` it's handed; agnostic to how the target was constructed.
- The already-published README MAPE (0.32%/1.78%): the test window itself (2026-08-11–08-16, per spec 004's rollout) sits well after the July gap window, so those specific test rows are not directly mispaired. What this bug *could* have degraded is upstream of the test metric — the **model's own fit**, trained partly on mislabeled examples from the gap window. See §5, item 3.

---

## 2. Scope

### 2.1 In Scope

1. A pure, unit-testable function that builds the horizon target correctly — sort by timestamp, reindex to a complete hourly grid, shift, restore original rows — mirroring `src/features.py::add_lag_features()`'s established pattern. **Corrected during implementation review (see §3.1's note):** this produces `target = NaN` only for the anchor whose target hour (`t + horizon_hours`) itself lands on a missing hour; a gap merely *somewhere* inside the window does not produce NaN — it is correctly re-paired to the true value at `t + horizon_hours`, which is a stronger outcome than "NaN anywhere a gap intrudes."
2. Replace the raw `.shift(-horizon_hours)` call in `06_train_lgbm.py::train_lgbm_model()` with this function. No other logic in that function changes.
3. Unit tests proving: (a) a complete, sorted, gapless series produces identical output to the naive `shift()` — no regression on the happy path; (b) a mid-window gap produces `NaN` at the affected rows, not a wrong value; (c) a shuffled/unsorted input produces the same correct output as the sorted input — the actual defect this spec fixes.
4. Confirm, by review, that `make_holdout_splits()`, `src/tuning.py`, and the existing `MIN_TRAINING_ROWS` guard (spec 004) all require zero code changes and continue to behave correctly once fed a correctly-constructed target.

### 2.2 Out of Scope

- Any change to `notebooks/05_train_prophet.py` (unaffected, §1.3).
- Any change to `04_predict.py` (spec 006's concern — inference-path labeling, not training-path target construction; disjoint bug, disjoint files).
- Any change to `07_evaluate.py`, `08_promote_model.py`, `03_drift_check.py`, `databricks.yml`.
- Any change to `src/features.py`, `src/splits.py`, `src/tuning.py`, `src/config.py` (`MODEL_INPUT_FEATURES`, `MIN_TRAINING_ROWS`, `LGBM_PARAMS` all unchanged).
- Retroactively correcting spec 001's §9 claim, or rewriting historical MLflow runs / `model_evaluation` rows computed under the old target construction — recompute-forward only, per the project's established precedent (specs 004 §3.6, 005 §3.6).
- Bounding `06_train_lgbm.py`'s read of `silver_features` to a lookback window. More training history is not the problem; incorrectly *shifting across* gaps in that history is. Changing the window size is a separate design question, not addressed here.
- A repo-wide audit for other unguarded Spark→pandas boundary crossings beyond this one confirmed instance (§7).

---

## 3. Design

### 3.1 Pure Function — `src/target_construction.py` (new)

A new, narrowly-scoped module — not added to `src/features.py`, because that module's documented purpose is features shared identically between training and serving (`02_transform.py` and `05`/`06`) to prevent training-serving skew. The horizon target is training-only; it belongs in its own module, following the project's established pattern of one small pure module per concern (`baseline.py`, `splits.py`, `forecast_timing.py`).

```python
"""Time-aligned target construction for LightGBM's direct multi-step forecasting."""

from __future__ import annotations

import pandas as pd


def build_horizon_target(df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    """Adds a `target` column equal to value_mwh exactly `horizon_hours` HOURS
    ahead of each row — not `horizon_hours` ROWS ahead.

    pandas .shift(-n) is positional: it is only equivalent to an n-hour
    lookahead if the frame is ascending-sorted by timestamp and has no
    missing hourly rows in the shift window. Neither is guaranteed by a raw
    Spark table read. This mirrors src/features.py::add_lag_features()'s
    reindex-before-shift pattern: sort, reindex to a complete hourly grid,
    shift, then restore only the original rows.

    Because shift() now runs on a grid where every hour has a real position
    (gap hours included, as NaN placeholders), a gap merely somewhere inside
    the (t, t+horizon_hours] window does NOT produce NaN — it is correctly
    re-paired to the true value at t+horizon_hours, exactly as it should be.
    target = NaN occurs only for the one anchor per missing hour whose own
    target lands exactly on that missing hour (caught by the existing
    dropna(subset=["target"] + FEATURE_COLS)). This is a materially smaller
    set of NaN rows than "every row a gap intrudes on" — see spec 007 §3.4
    for the corrected row-count accounting.
    """
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

Note: `tz` is read from the input rather than hardcoded to `"UTC"` (unlike `add_lag_features`, which assumes UTC because it only ever runs downstream of `add_calendar_features`'s explicit tz check). `06_train_lgbm.py` reads `silver_features` directly with a bare `pd.to_datetime(pdf["timestamp"])` and no tz validation step — inferring `tz` from the input is a small, defensive generalization rather than repeating an unvalidated assumption.

### 3.2 `train_lgbm_model()` — Before/After

**Before (`06_train_lgbm.py:92-94`):**

```python
    df_model = df.copy()
    df_model["target"] = df_model["value_mwh"].shift(-horizon_hours)
    df_model = df_model.dropna(subset=["target"] + FEATURE_COLS)
```

**After:**

```python
    df_model = build_horizon_target(df, horizon_hours)
    df_model = df_model.dropna(subset=["target"] + FEATURE_COLS)
```

Everything downstream — `make_holdout_splits()`, the Optuna tuning call, the LGBM fit, metric computation, the naive-baseline call, MLflow logging — is structurally unchanged. `build_horizon_target()` returns the same shape/columns as before, just correctly paired and, as a side effect, sorted ascending (the previous code preserved whatever order the Spark read happened to return).

### 3.3 Why the Existing Row-Count Guard Is Sufficient — No New Safety Net Needed

Spec 004 already added: `if len(train_df) < max(MIN_TRAINING_ROWS, CONFIG["min_train_rows"]): raise ValueError(...)`. This guard is untouched and does its job for free here — if this fix removes enough previously-mispaired-but-non-null rows to push `train_df` below the floor, the notebook raises exactly as it's designed to, rather than silently training on a shrunken set. No new guard is added in this spec.

### 3.4 Expected Row-Count Impact — Refreshed Once, Still Not a Merge-Ready Number

**Superseded, not final.** The 2026-09-21 estimate below (132/276 misaligned rows) was based on a 3,234-row snapshot that itself predated a second GitHub Actions/Databricks outage (2026-09-19 17:05 – 2026-09-23 14:05, ~93h). A 2026-09-26 live-SQL refresh found the true picture materially worse:

| Horizon | 2026-09-21 estimate | 2026-09-26 refresh (live SQL) | Change |
|---|---|---|---|
| 24h | 132 rows (4.1% of 3,234) | 203 rows (6.12% of 3,317) | +71 rows |
| 168h | 276 rows (8.5% of 3,234) | 635 rows (19.14% of 3,317) | +359 rows |

The refresh found 227 total missing hours (vs. 144 previously known): the July outage (144h), a previously-undetected 10h gap on 2026-09-07/08, and a new 73h gap from the September outage. As of the refresh, `energy_hourly_pipeline`'s `drift_check`/`predict` tasks were still failing — the 73h block sits inside `03_drift_check.py`'s 7-day current window and has not yet been backfilled or aged out. **The 227-hour / 635-row figures are themselves a snapshot of an unresolved, in-progress incident, not a stable baseline** — they will change again once that gap is closed (by backfill or natural aging-out), and should not be treated as the number to merge against either.

19.14% of the 168h horizon's rows being misaligned under the current buggy code is a substantially larger fraction than this spec originally assumed — still very likely to clear the 720-row floor given the ~3,300-row pool, but the margin is visibly tighter than the original "very unlikely to threaten the floor" framing suggested. This raises, rather than lowers, the importance of AC3 as a hard gate: the number moved by +359 rows (168h) in nine days between two live measurements, which is reason enough to re-measure immediately before merge rather than trust either figure in this table.

**2026-09-27 refresh — post spec-008 deploy, third and most stable measurement:**

| Horizon | 2026-09-21 | 2026-09-26 (mid-incident) | 2026-09-27 (post spec-008) |
|---|---|---|---|
| 24h | 132 rows | 203 rows | **179 rows** (July 132 + Sep 7-8 47) |
| 168h | 276 rows | 635 rows | **467 rows** (July 276 + Sep 7-8 191) |

Spec 008 (Volume backlog discovery) deployed and closed the 2026-09-19→09-22 73h gap that drove the 09-26 spike — confirmed absent from the current gap enumeration in both `bronze_load` and `silver_features`. Only two gaps remain: the original July outage and a previously-undetected 10h gap on 2026-09-07/08, neither of which the still-draining Volume backlog (post spec-008, ~798 files remaining at last check) can reach — the source backfill covered roughly the trailing 12.5 days from the drain's start, which predates both remaining gaps. **These figures are therefore expected to hold**, not because the backlog has fully drained, but because what remains in it cannot touch either surviving gap, and `MERGE INTO`'s idempotency means the already-closed September gap cannot reopen.

Against the 3,411-row pool, 467 misaligned 168h rows leaves an estimated train pool of ≈2,776 after this fix's `dropna` — comfortably above the ~960-row combined floor (720 + 240 val/test holdout), a wider margin than either prior snapshot showed. AC3 remains a hard gate in principle, but confidence in these numbers going into that check is now materially higher than at either previous measurement.

**Correction, post-implementation (discovered during Part B, not a new live measurement):** the 179/276/467/etc. figures throughout this section measure something real and unchanged — how many rows were *wrongly paired* under the old positional-shift bug (any missing hour anywhere in the row's `(t, t+horizon_hours]` window). They are **not**, however, the number of rows that become `NaN` after this fix, which was this section's original (incorrect) assumption. Under the fix, a row whose window merely *contains* a gap gets correctly re-paired to its true value — not NaN'd. `target = NaN` occurs only for the single anchor per missing hour whose own target lands exactly on that hour, roughly one NaN per missing hour, **independent of horizon length**. At the current 154 known missing hours (144 July + 10 Sep 7-8), that means an expected ≈154 additional NaN target rows for *either* horizon — not the horizon-scaled 179 (24h) / 467 (168h) this section previously implied would be dropped.

This makes the row-count margin **more comfortable than every estimate above**, not less: losing ≈154 rows against a ~3,400-row pool is a small fraction regardless of horizon. The 179/467 table remains useful as a measure of *how wrong the old code was* (a legitimate, unchanged finding), but should not be read as "rows lost from training after the fix." AC3's live check should confirm both numbers — the actual post-fix NaN-target count (expected ≈154) and the resulting `train_df` size against the floor — so this correction is verified against real data, not just re-reasoned on paper.

---

## 4. Acceptance Criteria

- [ ] **AC1**: `build_horizon_target()` exists in `src/target_construction.py`, pure and typed. Unit tests confirm: (a) a complete, sorted, gapless hourly series produces output identical to naive `df["value_mwh"].shift(-horizon_hours)`; (b) a single mid-window gap never produces a wrong (mispaired) value — rows whose window merely crosses the gap are correctly re-paired to their true `t+horizon_hours` value, and `NaN` appears only at the one anchor whose target hour lands exactly on the missing hour; (c) row-shuffled input produces output identical to the sorted-input case, for both horizon_hours=24 and =168.
  - *Verification:* `tests/test_target_construction.py::test_mid_window_gap_never_mispairs_across_the_gap`, asserting the corrected contract against the naive `shift()`'s actual wrong output, not just against the fixed function's own expectation.
- [ ] **AC2**: `train_lgbm_model()` uses `build_horizon_target()` in place of the raw `.shift()` call; no other line in the function changes.
  - *Verification:* code review of the diff.
- [ ] **AC3**: Post-fix `train_df` row counts, measured against live `silver_features` data once compute access is available, still clear `max(MIN_TRAINING_ROWS, CONFIG["min_train_rows"])` for both horizons. This is a live-data check, not a re-derivation of §3.4's estimate.
  - *Verification:* a manual (or Databricks-run) check logged in the PR before merge; if it fails, that is a valid, actionable finding — not grounds to weaken the guard.
- [ ] **AC4**: `make_holdout_splits()`, `src/tuning.py`, and the spec-004 row-count guard require zero code changes.
  - *Verification:* `git diff` shows no changes to `src/splits.py` or `src/tuning.py`.
- [ ] **AC5**: `ruff`, `mypy src/`, and `pytest --cov-fail-under=80` all pass.
- [ ] **AC6**: Out-of-scope files untouched — `05_train_prophet.py`, `04_predict.py`, `07_evaluate.py`, `08_promote_model.py`, `03_drift_check.py`, `src/features.py`, `src/splits.py`, `src/tuning.py`, `src/baseline.py`, `src/config.py`, `databricks.yml` all byte-identical to `main`.
  - *Verification:* `git diff main..feature/007-lgbm-target-shift-alignment` lists only `notebooks/06_train_lgbm.py`, `src/target_construction.py` (new), the new test file, and this spec.

---

## 5. Open Questions / Decisions Needed

| # | Question | Options | Recommendation |
|---|---|---|---|
| 1 | **Verify post-fix row counts on live data before or after merge?** | (a) Before — gate the merge on it; (b) after — merge, then check on the next scheduled run | **(a) Before** — this is exactly the kind of assumption (§3.4) that should be confirmed, not discovered in production; requires restoring compute access (the SQL warehouse was stopped during investigation) first. |
| 2 | **Correct spec 001 §9's "aging out naturally" claim?** | (a) Leave spec 001 as a historical record, note the correction only here; (b) also patch a line in GEMINI.md or README reflecting that `silver_features` accumulates permanently and old gaps don't age out of `06_train_lgbm.py`'s unfiltered read | **(b), lightly** — a one-line GEMINI.md clarification is cheap and prevents the next spec from repeating the same "it'll self-heal" assumption; spec 001 itself stays untouched (dated document). |
| 3 | **Does this bug mean the published README MAPE (0.32%/1.78%) needs re-verification, not just the training row count?** | (a) Yes — retrain and compare; (b) No — the test window is clean, leave as-is | **(a)** — the test *window* wasn't mispaired (§1.3), but roughly 4–9% of the *training* examples were mislabeled going into that fit. The model itself may be marginally different post-fix even though the evaluation methodology wasn't at fault. This spec's rollout (§6.3) should compare new-vs-old MAPE before quietly assuming "no change." |
| 4 | **Ship independently of spec 006, or bundle?** | (a) Independent, parallel; (b) bundle into one PR | **(a) Independent** — disjoint files (`06_train_lgbm.py` vs `04_predict.py`), no ordering dependency; bundling only makes the diff harder to review against two separate spec's ACs. |

---

## 6. Implementation Plan

### 6.1 Files to Change

| File | Change |
|---|---|
| `src/target_construction.py` (new) | `build_horizon_target(df, horizon_hours) -> pd.DataFrame` per §3.1. |
| `notebooks/06_train_lgbm.py` | `train_lgbm_model()`: replace the raw `.shift(-horizon_hours)` line with a call to `build_horizon_target()`, per §3.2. No other line changes. |
| `tests/test_target_construction.py` (new) | Unit tests per AC1, mirroring the investigation's `repro_shift.py` methodology (gap case, shuffle case, gapless-regression case). |
| `specs/007-lgbm-target-shift-alignment/spec.md` | This spec. |
| `GEMINI.md` (optional, per §5 item 2) | One-line clarification that `silver_features` accumulates permanently and gaps do not self-resolve out of `06_train_lgbm.py`'s unfiltered read. |

### 6.2 Files NOT to Change

- `notebooks/05_train_prophet.py` — no `shift()`, unaffected (§1.3).
- `notebooks/04_predict.py` — disjoint bug, spec 006's concern.
- `notebooks/07_evaluate.py`, `notebooks/08_promote_model.py`, `notebooks/03_drift_check.py`.
- `src/features.py` — `add_lag_features()` already correct; not the file with the bug.
- `src/splits.py` — `make_holdout_splits()` confirmed order-independent (§1.3); no change needed.
- `src/tuning.py`, `src/baseline.py`, `src/config.py`, `databricks.yml`.

### 6.3 Rollout Sequence

1. Implement per §3; run `ruff`, `mypy src/`, `pytest --cov-fail-under=80`.
2. Restore compute access (SQL warehouse or a notebook run) and execute §4 AC3's live row-count check for both horizons — do not merge on the §3.4 estimate alone.
3. Merge; confirm `deploy.yml` succeeded before triggering retraining (standing project discipline).
4. Next scheduled `energy_retraining_pipeline` run (or manual trigger): compare the new run's `mae`/`rmse`/`mape` and `n_train` against the current README numbers and spec 004's rollout figures.
5. If the metrics move materially, update the README's LightGBM rows with a note explaining why (training-target correction, not a promotion-gate or architecture change); if they don't move materially, that itself is worth a one-line note — it would mean the ~4-9% of mislabeled training examples had negligible influence on the fit, which is a real, useful finding either way.

---

## 7. Future Spec Candidates (Not Implemented Here)

- **Repo-wide unguarded-shift/order-assumption audit**: this investigation confirmed `06_train_lgbm.py`'s target construction as "the only unguarded `shift()` in the repository," but that check was scoped to `shift()` calls specifically. A broader audit for any other place data crosses the Spark→pandas boundary and is then processed under an implicit ordering assumption (not necessarily via `shift()`) would close out this class of bug more completely.
- **`silver_features` retention/archival policy**: `06_train_lgbm.py` reads the full, unfiltered, permanently-accumulating history. This spec does not change that (§2.2), but the fact that gaps never age out of it (§1.2) is worth a deliberate policy decision — keep-forever-and-correctly-handle-gaps (this spec's approach) vs. some form of bounded/pruned window — rather than the implicit assumption spec 001 §9 made and this investigation found to be false.
