# 001 — Ingestion Gap Resilience

**Status:** Draft  
**Author:** Technical Architect  
**Date:** 2026-07-26  
**Priority:** High  
**Dependencies:** None (standalone fix to ingestion_hourly.yml + 02_transform.py)

---

## 1. Problem Statement

### 1.1 Observed Behaviour

GitHub Actions' scheduled trigger (`cron: "5 * * * *"` in `ingestion_hourly.yml`) does not fire reliably. Analysis of the last 30 consecutive runs shows:

| Metric | Value |
|---|---|
| Scheduled cadence | 1 hour (cron `5 * * * *`) |
| Actual min gap | 60 minutes |
| Actual median gap | 118 minutes |
| Actual max gap | 265 minutes |
| Consecutive-day blackout window | ~00:10 – ~04:15 UTC (3-4 triggers skipped nightly) |

Each run completes in **1.3–3.1 minutes**, ruling out the `concurrency: { group: ingestion-pipeline, cancel-in-progress: false }` setting as the cause — no run is long enough to queue subsequent triggers for hours. GitHub Actions explicitly documents scheduled workflows as "best effort"; the root cause is a GitHub-side scheduling limitation outside this repo's control.

### 1.2 Downstream Impact

Despite every run reporting `conclusion: success`, the effective ~3-hourly ingestion cadence silently broke the pipeline:

1. **`bronze_load`** received ~8–12 rows/day (vs. expected 24) starting ~July 3, with the heaviest loss during the overnight blackout window.
2. **02_transform.py** — its 37-day sliding window (`lookback_hours=720` + `max_lag=168`) holds ~720 bronze rows at hourly cadence but only ~559 at the actual 3-hourly cadence. On July 15 the window fell below `MIN_TRAINING_ROWS=720`, and every run since has exited at Cell 7 with `{"status": "skipped", "reason": "insufficient_data"}` — while still returning `dbutils.notebook.exit(SUCCESS)`.
3. **`silver_features`** received zero new rows after July 14.
4. **03_drift_check.py** queried an empty current window → no drift assessment, no retrain trigger.
5. **04_predict.py** never ran → `gold_forecasts` stopped updating (last 24h forecast: July 8, last 168h forecast: July 14).
6. **06_train_lgbm.py** — training set shrank below CV minimum (needs 984 rows at 3-fold gap=168, got 966) → task failure.
7. **08_promote_model.py** never executed → stale champion, no evaluation updates since July 5.

---

## 2. Scope

### 2.1 In Scope

1. **Increase `lookback_hours` default** in `ingestion_hourly.yml`'s fetch step so that whenever a run fires — even after a multi-hour gap — it backfills the missed hours in that single run.
2. **Make the transform "insufficient_data" exit distinguishable** at the job level so the hourly pipeline's monitoring (or a future alert) can differentiate between "gap being backfilled" and "genuinely insufficient data (e.g. cold start)."
3. **Corresponding test updates** in `tests/test_ingest_logic.py`.

### 2.2 Out of Scope

- Changing `MIN_TRAINING_ROWS`, `lookback_hours` in `src/config.py`, or the 720-row threshold in `02_transform.py` (these were correctly calibrated for hourly-cadence data — the fix is restoring coverage, not loosening thresholds).
- Adding a naive/seasonal baseline model to evaluation (separate spec).
- Changing drift thresholds, model training logic, or the champion/challenger flow.
- Implementing an external cron service or any workaround for GitHub's scheduler (flag as possible future spec, do not implement here).
- Backfilling historical data already missing from July 3–15 (the window will self-heal as new hourly data accumulates at the restored cadence once this fix is deployed).

---

## 3. Design

### 3.1 Proposal A: Widen `lookback_hours` Default (Recommended)

#### 3.1.1 Rationale

The existing fetch segment in `ingestion_hourly.yml` already has `ENTSOSE_MAX_RANGE_DAYS` chunking logic and idempotent `MERGE INTO` in `01_ingest.py`. Widening the fetch window is a parameter change only — no new infrastructure or API calls.

#### 3.1.2 Determining the Default

| Factor | Value |
|---|---|
| Max observed gap | 265 min (~4.4 h) |
| Desired margin | 5.5× — cover the entire overnight blackout window in a single run |
| Lower bound | 4.4 h × 5.5 ≈ 24 h |
| Round to clean hour | **24 hours** (1440 minutes) |
| Does this exceed `ENTSOSE_MAX_RANGE_DAYS`? | No (24 h << 30 days). The chunking logic handles ranges larger than the API limit — but since we stay well under it, we do not exercise the code path at all. |

**Proposed default: `lookback_hours: 24`** (previously unset / defaults to 1 in the Python fetch script).

At 24h, a single run covers the full overnight blackout window (~00:10–04:15 UTC) plus margin in both directions. Even if only one run fires in a 24-hour period, it fetches a complete day of data. The `MERGE INTO` in `01_ingest.py` is idempotent; re-fetching already-ingested hours produces no duplicates.

#### 3.1.3 Trade-offs

- **ENTSO-E API rate limit**: ENTSO-E allows ~100 requests/min. A single 24-hour fetch at hourly resolution costs ~24 documents (≈24 requests). At the worst observed cadence (~8 runs/day), this totals ~192 documents/day — well within limits. At the expected recovered cadence (24 runs/day), this totals ~576 documents/day, still well under the per-minute limit.
- **Data volume**: 24 hours × ~24 runs/day = up to 576 rows written to bronze_load daily via MERGE (idempotent, so actual new rows ≈ 24). Negligible storage.
- **Latency**: The fetch script processes hourly documents sequentially; 24 documents at ~1–2 seconds each adds ~25–50 seconds to a run that currently completes in ~90 seconds. Total run time remains under 3 minutes.

**Conclusion**: The risk is minimal and well-contained.

### 3.2 Proposal B: Distinguishable Transform Exit (Recommended)

#### 3.2.1 Current Behaviour

```
cell_7_exit = {
    "status": "skipped",
    "reason": "insufficient_data",
    "message": f"Insufficient data for feature engineering. Required: {MIN_TRAINING_ROWS}, Actual: {load_count}"
}
dbutils.notebook.exit(json.dumps({"status": "SUCCESS", "results": cell_7_exit}))
```

This exits with `"SUCCESS"` — the hourly job's task graph sees SUCCESS and does not alert. The insufficient-data condition becomes visible only by inspecting notebook output JSON.

#### 3.2.2 Proposed Change

Replace the blanket `"SUCCESS"` exit with an explicit exit payload that makes the shortfall observable:

```python
exit_payload = {
    "status": "SUCCESS",
    "message": "Transform complete",
    "rows_written": row_count,
    "insufficient_data": False,
}
```

When the early-exit path triggers:

```python
exit_payload = {
    "status": "SUCCESS",
    "message": f"Insufficient data for feature engineering. Required: {MIN_TRAINING_ROWS}, Actual: {load_count}",
    "rows_written": 0,
    "insufficient_data": True,
}
dbutils.notebook.exit(json.dumps(exit_payload))
```

The downstream Watson/Monitoring task (or a future `databricks jobs run-now --wait` caller) can key on `insufficient_data: true` + `rows_written: 0` without parsing free-text.

---

## 4. Acceptance Criteria

- [ ] **AC1**: `ingestion_hourly.yml`'s fetch step uses `lookback_hours=24`; the Python fetch script receives it from the workflow input or environment and fetches an ENTSO-E document for each hour in `[now - 24h, now)`.
- [ ] **AC2**: After a multi-hour gap (e.g., 4h with no runs), the next run backfills `bronze_load` with rows for all missed hours. Verify via `SELECT COUNT(*) FROM bronze_load WHERE fetched_at > <gap_start>`.
- [ ] **AC3**: `01_ingest.py` remains idempotent — re-running with overlapping backfill windows produces zero duplicate rows.
- [ ] **AC4**: `02_transform.py` exits with `insufficient_data: true` when `load_count < MIN_TRAINING_ROWS`, and with `insufficient_data: false` when it succeeds. No silent SUCCESS-with-zero-rows.
- [ ] **AC5**: `MIN_TRAINING_ROWS` in `src/config.py` is untouched. `02_transform.py`'s window sizing logic is untouched.
- [ ] **AC6**: Existing tests in `tests/test_ingest_logic.py` pass with the new default. Coverage is added for the backfill window sizing logic (test that `lookback_hours=24` produces the correct set of target timestamps).

---

## 5. Open Questions / Decisions Needed

| # | Question | Options | Recommendation |
|---|---|---|---|
| 1 | **Exact `lookback_hours` value** | 6h, 12h, 24h | **24h** — covers the full overnight blackout window in a single run. 24 API calls/run is well within rate limits. Run-time budget stays under 3 minutes. Chosen by architect. |
| 2 | **How to pass `lookback_hours` to the fetch script** | CLI arg, env var, workflow `inputs` | **Workflow `inputs` with a default** — visible in the GHA run UI, overrideable on `workflow_dispatch`, and requires no env-var plumbing changes. See `ingestion_hourly.yml` section below. |
| 3 | **Should the same `lookback_hours` apply to OpenMeteo temperature fetches?** | Yes / No | **Yes** — temperature is also needed for feature engineering; gaps in temperature cause missing `temperature_c` in silver_features, which degrades model accuracy. OpenMeteo has generous rate limits. |
| 4 | **How to deploy the `lookback_hours` change** | Direct PR to default branch / staged rollout | **Direct PR** — this is a config-only, additive change with zero risk of regression (idempotent MERGE, chunking already handles large ranges, rate-limit risk is nil). |
| 5 | **Should we also increase OpenMeteo fetch lookback?** | See Q3 | Already covered — answer Yes keeps the fix symmetric and prevents a parallel temperature gap. |

---

## 6. Implementation Plan (for reference during coding)

### 6.1 Files to Change

| File | Change |
|---|---|
| `.github/workflows/ingestion_hourly.yml` | Add `inputs.lookback_hours` with default `24`. Add `--lookback-hours ${{ inputs.lookback_hours }}` (or equivalent env-var) to the fetch-pipeline step. |
| `scripts/fetch_entsoe.py` or equivalent | Accept `lookback_hours` parameter; generate target timestamps as `[now - lookback_hours, now)` at hourly steps. |
| `scripts/fetch_openmeteo.py` or equivalent | Same change as above (accept `lookback_hours` param). |
| `notebooks/02_transform.py` | Change early-exit payload to include `"insufficient_data": true` (see §3.2). |
| `tests/test_ingest_logic.py` | Add test for `lookback_hours=24` → correct target timestamp set. No change to existing tests. |

### 6.2 Files NOT to Change

- `src/config.py` (keep `MIN_TRAINING_ROWS=720`, `lookback_hours` widget default unchanged).
- `notebooks/01_ingest.py` (no change needed — idempotent MERGE handles overlaps already).
- `notebooks/03_drift_check.py`, `04_predict.py`, `05_train_prophet.py`, `06_train_lgbm.py`, `08_promote_model.py`.
- `src/features.py`.

### 6.3 Rollout Sequence

1. **Pre-merge verification**: In a branch or fork, manually run the fetch script locally (or via `workflow_dispatch` on the branch) with `lookback_hours=24` and confirm the output JSON array contains 24 hourly documents. Verify `bronze_load` row count increases by the expected number after `01_ingest.py` processes the uploads.
2. **Merge** PR with the changes above.
3. **Post-merge monitoring (24h)**: Check GitHub Actions run logs — confirm `lookback_hours=24` appears in the fetch step output. Verify `bronze_load` row counts show a return to ~24 new rows/day within 48 hours. Run the `energy_hourly_pipeline` manually via `databricks jobs run-now` to confirm the full cascade (ingest → transform → drift → predict) resumes.
4. **Recovery tracking**: After bronze_load accumulates ~720 rows (within ~30 days at hourly cadence starting from the current ~559 base), confirm `silver_features` resumes writing. Check `drift_control` for new `check_timestamp` entries. Verify `gold_forecasts` resumes producing 24h and 168h forecasts. This is a passive recovery timeline — accelerate by running the pipeline once per hour via `workflow_dispatch` if desired, or simply let the fixed cadence self-heal.

---

## 7. Future Spec Candidates (Not Implemented Here)

- **External cron/liveness monitor**: A lightweight external service (e.g., AWS EventBridge, uptime monitor) that calls `workflow_dispatch` if no run has completed in >2 hours. This bypasses GitHub's scheduled-trigger reliability entirely.
- **Bronze self-backfill in `01_ingest.py`**: Detect missing hours at the Databricks side during ingestion and fetch missed windows directly via the UC Volume uploads (requires network access from Databricks — currently blocked in Free Edition).

---

## 8. Finding 2 (Post-Implementation): `lookback_files` Mismatch

### 8.1 Discovery

After deploying the GHA `lookback_hours=24` fix (§3.1), the expected bronze_load row-count jump did not materialise. Bronze_load remained at **9–12 rows/day** — unchanged from the pre-fix pattern — despite successful GHA runs uploading 24 files per execution.

Investigation revealed a second bottleneck in the ingestion pipeline.

### 8.2 Root Cause

The GHA workflow uploads 24 JSON files per run (one per hour in the lookback window). However, `notebooks/01_ingest.py` only reads **2 files per run**, controlled by a separate widget:

```python
# notebooks/01_ingest.py, line 63
dbutils.widgets.text("lookback_files", "2")
```

The `databricks.yml` job definition for `energy_hourly_pipeline`'s ingest task has **no `base_parameters`**:

```yaml
- task_key: "ingest"
  notebook_task:
    notebook_path: ./notebooks/01_ingest.py
  environment_key: "default"
  timeout_seconds: 1800
```

Because no parameters are passed, the notebook's widget default `"2"` is used. Each ingest run scans only the 2 most recent hourly files (`run_date` and `run_date - 1h`). The other 22 uploaded files accumulate in the Volume unprocessed.

This was confirmed via `databricks runs get-output` across multiple consecutive runs:

| Run time | Ingest result | Files found |
|---|---|---|
| Jul 26 14:43 | `rows_ingested=1`, `files_found=1` | 1 file (14:00) |
| Jul 26 15:05 | `no_files_found` | 0 — expected 15:00 file not yet uploaded |
| Jul 26 15:08 | `no_files_found` | 0 |
| Jul 26 15:24 | `no_files_found` | 0 |

And via `bronze_load` remaining flat at 1728 total rows (only +3 from 1725) over the same period.

### 8.3 Fix

Add `base_parameters` to the `databricks.yml` ingest task to pass `lookback_files="24"`, matching the GHA upload window:

```yaml
- task_key: "ingest"
  notebook_task:
    notebook_path: ./notebooks/01_ingest.py
    base_parameters:
      - "--lookback_files"
      - "24"
  environment_key: "default"
  timeout_seconds: 1800
```

Do **not** change `01_ingest.py`'s own widget default (`"2"`). The job-level override is the only change. This ensures:
- The job-level override is explicit and version-controlled in `databricks.yml`.
- The notebook's default remains conservative for manual/adhoc runs.
- `energy_retraining_pipeline` does not reference `01_ingest.py` — no side effects.

### 8.4 Acceptance Criteria

- [ ] **AC7**: After deploy, the next `energy_hourly_pipeline` run ingests all 24 files uploaded by GHA. Bronze_load shows ~24 new rows/day (net) instead of ~2.
- [ ] **AC8**: `bronze_load` reaches ~720 rows within the 37-day sliding window within ~12 days at the restored cadence, allowing `02_transform` to resume writing to `silver_features`.
- [ ] **AC9**: The `lookback_files` widget default in `01_ingest.py` remains `"2"` — only the job-level parameter overrides it.
