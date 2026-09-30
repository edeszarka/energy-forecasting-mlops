# 009 — Silver Rolling-Window Corruption and Leak-Affected Champion Exclusion

**Status:** Draft
**Author:** Technical Architect
**Date:** 2026-09-28
**Priority:** Critical
**Dependencies:** Spec 007 established the exact leak boundary (merge commit `28608cc`, 2026-09-27T20:08:23Z) and the honest-MAPE baseline (13.33%/12.48%) this spec's gate-unblock relies on. Independent of specs 006/008 (different files, different mechanisms), but part of the same incident thread. This spec bundles two distinct fixes under one number at the user's explicit request, because the second is only useful in combination with the first: fixing the champion-selection gate without also stopping the ongoing training-data corruption would unblock promotion into a data-starved retrain; fixing the data corruption without unblocking the gate would leave an honest model with nowhere to go. Each is scoped, reviewed, and tested independently within this spec (§3 Design A / Design B), touching disjoint files.

---

## 1. Problem Statement

### 1.1 Problem A — `02_transform.py` Permanently Corrupts Aging Rows

`02_transform.py` fetches a window `[run_date − (lookback_hours + max_lag), run_date]` (Cell 4), where `max_lag = max(LAG_HOURS) = 168` is explicitly meant as margin — Cell 4's own comment says *"We extend the window by max(LAG_HOURS) to ensure we have history for the lags of the first row."* The notebook computes features over the **entire** fetched window, including that margin, and writes **every** computed row to `silver_features` via an unconditional `whenMatchedUpdateAll()` MERGE (Cell 11).

The margin exists so that rows *elsewhere* in the window have enough backward context — it was never meant to produce final output for rows sitting *inside* the margin itself. But nothing filters those margin rows out before the write. For a fixed calendar hour `T`, every hourly run whose window happens to place `T` in the last 168 hours of margin has insufficient backward context to compute `T`'s `lag_168h` (needs `value(T−168h)`, which falls before this run's `window_start`) — and writes `NaN` for it. Because `whenMatchedUpdateAll()` accepts this unconditionally, and because `T` eventually ages entirely out of every future run's window (at which point it is never touched again), **every row is guaranteed to be permanently frozen with `NaN` long-lag/rolling features once it exits the active window** — not as an occasional edge case, but as a structural certainty of the current design. This was independently confirmed on live data during spec 007's investigation: 979 `lag_168h`-null rows in a 6-week span, clustering at age ≈888h (720+168) — exactly the boundary this mechanism predicts.

Consequence: `06_train_lgbm.py` reads the full, unfiltered `silver_features` history (spec 007 §2.2 explicitly left this out of scope), so this corruption directly caps the usable training pool — the 2026-09-28 honest re-evaluation found `dropna` removing 3,434 → 1,285/1,141 rows, the large majority to this cause, not to real data gaps.

### 1.2 Problem B — The Promotion Gate Cannot Accept an Honest Model

A 2026-09-28 investigation confirmed the currently production-tagged LightGBM runs (`energy_lgbm_24h` = `3fca487c…`, mape 0.09596%; `energy_lgbm_168h` = `3258148a…`, mape 0.10309%) are themselves leak-affected (≈99.97% of model gain on the single `lag_{h}h` feature; part of a run history that oscillates between ~0.1% and 4–15% MAPE with no stable underlying skill). `04_predict.py` would load these exact runs to serve predictions today.

Both promotion gates compare purely on MAPE improvement:
- `08_promote_model.py::decide_promotions()`: promote iff `(champion_mape − challenger_mape) / champion_mape > 0.01`.
- `07_evaluate.py`: same arithmetic, `challenger_wins` gate.

Against a champion MAPE of ~0.10%, **no honest challenger can ever win** — the honest re-evaluation (13.33%/12.48%) computes as a −13,100%/−12,010% "improvement." Only `first_run` (no champion exists) or the `force_promote` widget escape this. Without intervention, the champion/challenger system is permanently stuck serving a leak-affected model and will reject every honest one indefinitely.

---

## 2. Scope

### 2.1 In Scope — Problem A

1. Filter `02_transform.py`'s computed feature rows to the "core" sub-window — `timestamp >= window_start + timedelta(hours=max_lag)` — before the schema cast and MERGE, so every row ever written to `silver_features` has full backward context for its lag/rolling features by construction. No row is ever written with margin-induced `NaN`, so `whenMatchedUpdateAll()` can no longer silently degrade a previously-good value.
2. Move the Cell 8 NaN-rate validation to run on the filtered frame, so its warning reflects genuine data gaps, not margin-row artifacts that are no longer written.
3. A test proving the filter boundary is correct and that re-processing overlapping windows never reduces a previously-written row's feature completeness.

### 2.2 Out of Scope — Problem A

- Implementing a real `force_full_rebuild` (currently a no-op stub, Cell 12) to repair the ~2,000+ already-corrupted historical rows. This fix stops **further** corruption; it does not retroactively heal what's already frozen. A real rebuild is a larger, separate refactor (extracting the read→feature→write logic into a chunk-callable function) and is tracked as the top item in §7 — deliberately deferred so this spec stays reviewable, not because it's unimportant.
- Any change to `01_ingest.py`, `03_drift_check.py`, `src/features.py`'s feature math itself, or `MIN_TRAINING_ROWS`/`lookback_hours` defaults.

### 2.3 In Scope — Problem B

1. A `leak_affected` MLflow tag convention, set **positively** (`"false"` on every clean run, not merely absent) to avoid relying on ambiguous NULL/missing-tag filter semantics (§3.4).
2. `05_train_prophet.py` and `06_train_lgbm.py` each set `leak_affected = "false"` on every future run (Prophet was never affected, but the shared evaluation query in §2.3.3 needs both families tagged consistently).
3. `07_evaluate.py::get_run_metrics()`, `08_promote_model.py`'s champion search, and `04_predict.py::load_best_model_from_runs()` (both its production-tag query and its historic-best fallback query) add `AND tags.leak_affected = 'false'` to their MLflow search filters.
4. A one-time, explicitly-reviewed remediation (not repo code — a documented, logged script run once): tag every `energy_lgbm_24h`/`energy_lgbm_168h` run with `start_time` before `2026-09-27T20:08:23Z` (spec 007's merge) as `leak_affected = "true"` with an `invalidated_reason` tag for auditability; tag every historical Prophet run (any date — confirmed never affected) and any LGBM run at/after the cutoff as `leak_affected = "false"`; clear `production = "false"` on the two currently-tagged LGBM champions.

### 2.4 Out of Scope — Problem B

- Any change to the >1% improvement threshold, the `challenger_wins`/`should_promote` arithmetic itself, or `force_promote`.
- Rewriting `promotion_log`'s historical rows. Unlike `gold_forecasts`' proven-wrong timestamps (spec 006 §3.5), `promotion_log` is an accurate record of decisions actually made, under the code and information available at the time — it stays untouched as legitimate audit history.
- `03_drift_check.py::get_reference_window()`'s stage-based `get_latest_versions(stages=["Production"])` call — already always falls back to its 30-day window in practice (Model Registry writes are IAM-blocked on Free Edition, so no registered versions ever exist to find), unrelated to this remediation.
- `04_predict.py`'s `model_version="run_latest"` traceability gap (noted during the service audit — `gold_forecasts` cannot recover which exact run produced a prediction). Real, useful, and unrelated to the leak-exclusion mechanism — tracked in §7.

---

## 3. Design

### 3.1 Problem A — Before/After

**Before (`02_transform.py`, after Cell 7's `build_feature_matrix` call):**

```python
feature_pd = build_feature_matrix(load_pd, temp_pd)
logger.info(f"Feature matrix built: {feature_pd.shape}")

# Cell 8: Post-feature validation
expected_cols = set(get_feature_columns())
...
nan_report = feature_pd[get_feature_columns()].isna().mean()
```

**After:**

```python
feature_pd = build_feature_matrix(load_pd, temp_pd)
logger.info(f"Feature matrix built (pre-filter): {feature_pd.shape}")

# Discard the leading max_lag-hour margin: those rows lack full backward
# context for lag/rolling features within THIS run's fetch window. Writing
# them would silently degrade any previously-good value for the same
# timestamp once a future run's MERGE (Cell 11, whenMatchedUpdateAll)
# overwrites it — every row is guaranteed to age past this margin exactly
# once, so an unfiltered write here permanently corrupts it (see spec 009
# §1.1). window_start and max_lag are already computed in Cell 4.
core_window_start = window_start + timedelta(hours=max_lag)
feature_pd = feature_pd[feature_pd["timestamp"] >= core_window_start].reset_index(drop=True)
logger.info(f"Feature matrix after margin filter: {feature_pd.shape}")

# Cell 8: Post-feature validation (now runs on the filtered, writable frame)
expected_cols = set(get_feature_columns())
...
nan_report = feature_pd[get_feature_columns()].isna().mean()
```

Cell 9's schema cast, Cell 10's silver-table creation, and Cell 11's MERGE are unchanged — they now simply operate on a smaller, always-fully-contextualized frame. No conditional-MERGE logic is needed; the fix prevents bad rows from being *produced* in the first place, which is simpler and more robust than trying to detect and selectively preserve good values at merge time.

### 3.2 Problem A — Why This Is Sufficient (Not Just Plausible)

For any row with `timestamp = T` surviving the filter, `T >= window_start + max_lag`, so `T − 168h >= window_start`, so the reindexed grid (`add_lag_features`, already correct per spec 007) includes `T − 168h` as an actual position — `lag_168h(T)` is `NaN` only if that specific hour is a genuine bronze gap, never because of insufficient window margin. This holds for every lag (24/48/168h ≤ max_lag) and comfortably covers the rolling windows' `min_periods` (72h for the 7-day stat, 12h for the 24-hour stat — both well under the 168h margin).

### 3.3 Problem B — Query Changes

`07_evaluate.py::get_run_metrics()`:

```python
runs = client.search_runs(
    experiment_ids=[r.experiment_id for r in client.search_experiments()],
    filter_string=f"tags.model_name = '{model_name}' AND tags.leak_affected = 'false'",
    order_by=[...],
    max_results=5,
)
```

`08_promote_model.py::decide_promotions()` (champion search — defense in depth; functionally redundant once §3.5's remediation clears the production tag, but protects against a future accidental re-tag):

```python
prod_runs = mlflow_client.search_runs(
    ...,
    filter_string=f"tags.model_name = '{model_name}' AND tags.production = 'true' AND tags.leak_affected = 'false'",
    max_results=1,
)
```

`04_predict.py::load_best_model_from_runs()` — both paths (this one is functionally required, not defensive: its fallback path is exactly what would re-surface a leak run today if the production tag were cleared without this filter):

```python
runs = mlflow_client.search_runs(
    ...,
    filter_string=f"tags.model_name = '{model_name}' AND tags.production = 'true' AND tags.leak_affected = 'false'",
    order_by=["metrics.mape ASC"], max_results=1,
)
...
runs = mlflow_client.search_runs(
    ...,
    filter_string=f"tags.model_name = '{model_name}' AND tags.leak_affected = 'false'",
    order_by=["metrics.mape ASC"], max_results=1,
)
```

`05_train_prophet.py` and `06_train_lgbm.py`, inside their respective `with mlflow.start_run(...)` blocks, add one line each:

```python
mlflow.set_tag("leak_affected", "false")
```

### 3.4 Why a Positive Tag, Not an Exclusion of `'true'`

The natural-looking design — `AND tags.leak_affected != 'true'`, tagging only the ~20 known-bad runs — was considered and rejected. MLflow's search filter follows SQL-like tag comparison semantics; whether a *missing* tag satisfies `!= 'true'` is not something this investigation could verify without a live query, and if it follows strict NULL semantics (`NULL != 'true'` evaluates to NULL, i.e. excluded, not included), every future clean run — which would have no tag at all under that design — would silently vanish from every champion/challenger/serving query. A positive `= 'false'` filter has no such ambiguity in either interpretation: an absent tag cannot equal the literal string `'false'` under any reasonable semantics, so old untagged-or-mistagged runs are excluded either way, and only runs that explicitly declare themselves clean are ever matched. This is still flagged as something to verify against a live query before treating it as airtight (§5 item 2).

### 3.5 One-Time Remediation — Explicit, Reviewed, Not Repo Code

Per this project's established pattern (spec 006 §3.5's historical `gold_forecasts` cleanup): this is data remediation, not a notebook change, executed once, manually, after explicit sign-off, with the exact run_id list logged for audit. Steps:
1. Enumerate every `energy_lgbm_24h`/`energy_lgbm_168h` run via `search_runs`, freshly queried (not assumed from this or prior investigations' partial listings).
2. For each with `start_time < 2026-09-27T20:08:23Z`: `set_tag(run_id, "leak_affected", "true")`, `set_tag(run_id, "invalidated_reason", "spec007_positional_shift_target_leak")`.
3. For each at/after that cutoff (if any exist by execution time) and every historical Prophet run (any date): `set_tag(run_id, "leak_affected", "false")`.
4. On `3fca487c…` and `3258148a…` specifically: `set_tag(run_id, "production", "false")`.
5. Log the full before/after tag state for every touched run_id.

---

## 4. Acceptance Criteria

**Problem A**
- [ ] **AC1**: `02_transform.py` filters `feature_pd` to `timestamp >= window_start + timedelta(hours=max_lag)` before the schema cast (Cell 9) and MERGE (Cell 11); Cell 8's NaN-rate validation runs after this filter.
  - *Verification:* code review of the diff.
- [ ] **AC2**: A test (mirroring spec 006's AST-extraction pattern for notebook logic) proves the filter boundary is exactly `window_start + max_lag`, and that simulating two overlapping runs (an earlier wide window, a later narrower one covering an overlapping range) never reduces a previously-written row's `lag_168h`/rolling non-null rate — the actual property this fix guarantees.
  - *Verification:* new test file, per §6.1.
- [ ] **AC3**: `06_train_lgbm.py`, `05_train_prophet.py`, `07_evaluate.py`, `08_promote_model.py`, `03_drift_check.py` require no changes for Problem A.

**Problem B**
- [ ] **AC4**: `05_train_prophet.py` and `06_train_lgbm.py` each set `leak_affected = "false"` on every run; verified by code review only (cannot be verified without a live training run).
- [ ] **AC5**: `07_evaluate.py::get_run_metrics()`, `08_promote_model.py`'s champion search, and both of `04_predict.py::load_best_model_from_runs()`'s search paths add `AND tags.leak_affected = 'false'`.
- [ ] **AC6**: Before relying on AC5 in production, a live query against the real MLflow tracking server confirms the positive-filter behaves as expected for a genuinely untagged run (§3.4) — this is a required live check, not assumed from reasoning about SQL semantics alone.
- [ ] **AC7**: The one-time remediation (§3.5) is executed only after explicit sign-off; its exact run_id list and resulting tag state is logged. Not part of this spec's code diff.
- [ ] **AC8**: Post-remediation, a dry-run/simulation (not a real training or predict run) confirms `get_run_metrics(..., "champion")` returns `None` for `energy_lgbm_24h`/`168h` (triggering `first_run = True` in both `07_evaluate.py` and `08_promote_model.py`), and `load_best_model_from_runs` finds no valid LGBM run for either horizon (falls through to the Prophet fallback without raising).
- [ ] **AC9**: `promotion_log`, `model_evaluation`'s historical rows, and MLflow experiment history are confirmed untouched — this is a tagging operation, not a data rewrite.

**Shared**
- [ ] **AC10**: `ruff`, `mypy src/`, `pytest --cov-fail-under=80` all pass.
- [ ] **AC11**: Out-of-scope files untouched — `src/features.py`, `src/config.py`, `src/splits.py`, `src/tuning.py`, `src/baseline.py`, `01_ingest.py`, `databricks.yml`, and everything under `specs/006-*`/`specs/007-*`/`specs/008-*` byte-identical to `main`.

---

## 5. Open Questions / Decisions Needed

| # | Question | Options | Recommendation |
|---|---|---|---|
| 1 | **Order: code deploy vs. one-time tag remediation?** | (a) Deploy first, then tag; (b) reverse | **(a)** — mirrors spec 008's precedent. Before tagging, the new filters simply match nothing extra (no runs are tagged `'false'` yet, so behavior is unchanged from today) — zero risk window, no race. |
| 2 | **Is the `tags.leak_affected = 'false'` filter's live behavior on an absent tag verified?** | Verify before relying on it / trust the reasoning in §3.4 | **Verify (AC6)** — this is exactly the class of assumption this project's specs have repeatedly found wrong when finally checked live (007's own row-count estimate, twice). A positive filter is the safer *design*, but "safer" still needs a live confirmation, not just better reasoning. |
| 3 | **Accept the temporary LGBM→Prophet serving fallback (§3.5's consequence) as-is, or mitigate further?** | (a) Accept — Prophet is honest, if worse; (b) something else | **(a) Accept and document** — this is the correct, honest behavior: serving a known-bad LGBM model to avoid a worse-looking gap is exactly the failure mode this spec exists to remove. Watch the next few `predict` task outputs post-remediation to confirm the fallback engages cleanly (no unhandled exception). |
| 4 | **Implement the real `force_full_rebuild` now or later?** | (a) Now, in this spec; (b) later, separate spec | **(b) Later (§7)** — Problem B's core goal ("an honest model *can* reach production") is satisfied by the gate unblock alone (`first_run=True` auto-promotes regardless of training-pool size); the rebuild improves eventual *quality*, not whether promotion is possible. Bundling it here would meaningfully widen an already wide diff. |

---

## 6. Implementation Plan

### 6.1 Files to Change

| File | Change |
|---|---|
| `notebooks/02_transform.py` | Add the core-window filter per §3.1 (Problem A). |
| `notebooks/05_train_prophet.py` | Add `mlflow.set_tag("leak_affected", "false")` (Problem B, one line). |
| `notebooks/06_train_lgbm.py` | Add `mlflow.set_tag("leak_affected", "false")` (Problem B, one line). |
| `notebooks/07_evaluate.py` | `get_run_metrics()`'s `filter_string` gains `AND tags.leak_affected = 'false'` (Problem B). |
| `notebooks/08_promote_model.py` | `decide_promotions()`'s champion search `filter_string` gains `AND tags.leak_affected = 'false'` (Problem B). |
| `notebooks/04_predict.py` | `load_best_model_from_runs()`'s two `filter_string`s each gain `AND tags.leak_affected = 'false'` (Problem B). |
| `tests/test_transform_window_filter.py` (new) | AC2. |
| `specs/009-silver-window-integrity-and-leak-exclusion/spec.md` | This spec. |

### 6.2 Files NOT to Change

- `notebooks/01_ingest.py`, `notebooks/03_drift_check.py`.
- `src/features.py`, `src/config.py`, `src/splits.py`, `src/tuning.py`, `src/baseline.py`, `src/forecast_timing.py`, `src/target_construction.py`, `src/ingest_discovery.py`.
- `databricks.yml`, `.github/workflows/*`.
- `promotion_log`, `model_evaluation` table schemas — this spec adds MLflow run tags, not Delta table columns.

### 6.3 Rollout Sequence

1. Implement per §3.1–§3.3; run `ruff`, `mypy src/`, `pytest --cov-fail-under=80`.
2. Merge; confirm `deploy.yml` succeeded on both `prod` and `dev` targets.
3. Run AC6's live filter-behavior check against the real MLflow tracking server.
4. Execute the one-time remediation (§3.5) only after explicit sign-off; log the touched run_id list and resulting tags.
5. Run AC8's dry-run/simulation to confirm the gate is genuinely unblocked (`first_run=True`) without triggering a real training or predict run.
6. Watch the next few `energy_hourly_pipeline` `predict` task outputs: confirm the LGBM→Prophet fallback (§5 item 3) engages cleanly if no valid LGBM run exists yet.
7. At the next scheduled `energy_retraining_pipeline` run (2026-10-04 02:00 UTC or later, once Problem A has had time to grow the clean training pool), confirm the new LightGBM challenger is auto-promoted via `first_run=True`, and that its MLflow run carries `leak_affected = "false"`.
8. Only then, republish the README's LightGBM Results rows (currently "withdrawn" per the 2026-09-28 docs sync) with the new, genuinely honest numbers.

---

## 7. Future Spec Candidates (Not Implemented Here)

- **Real `force_full_rebuild`**: extract `02_transform.py`'s read→feature→core-window-filter→MERGE logic into a chunk-callable function, and implement Cell 12's currently-stubbed loop to actually call it per 30-day chunk across the full bronze history — this is what would repair the ~2,000+ already-corrupted historical rows and meaningfully grow the training pool beyond what organic hourly accumulation would achieve in any reasonable time. Highest-priority follow-up; deliberately deferred here for scope (§5 item 4).
- **`gold_forecasts` run-id traceability**: `04_predict.py` stores `model_version = "run_latest"` rather than the actual `run_id`, so a served prediction's producing run can never be recovered after the fact (noted during the 2026-09-28 service audit). Independent of the leak-exclusion mechanism.
- **A minimum-quality floor for `first_run` auto-promotion**: currently a first-run challenger is promoted regardless of its own MAPE, even against the naive baseline (`beats_baseline` is report-only per spec 002). Worth revisiting once this spec's gate-unblock has run its course — the very next LGBM promotion will be exactly this scenario.
