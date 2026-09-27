# 008 — Ingestion Volume Backlog Discovery (Replace Fixed-Count File Probing)

**Status:** Draft
**Author:** Technical Architect
**Date:** 2026-09-27
**Priority:** High
**Dependencies:** Spec 001 §8 ("Finding 2") — this is the same bug class recurring a second time. That fix hard-coded `databricks.yml`'s `lookback_files: "24"` to match GitHub Actions' `lookback_hours=24` default. This spec exists because the coupling itself, not the specific number, was never fixed: a 2026-09-27 manual backfill (`lookback_hours=300` on the GHA side) is suspected to have silently orphaned most of its ~300 uploaded files in the Volume, because `01_ingest.py` still only probes for exactly 24 hourly filenames per run, independent of how many files were actually fetched or how large the backlog is. Independent of specs 006/007 (different notebook, different bug — those are model-output/training-target correctness bugs; this is an ingestion-plumbing bug).

---

## 1. Problem Statement

### 1.1 The Recurring Bug

`notebooks/01_ingest.py`'s file-discovery logic (Cell 5) does not look at what is actually sitting in the raw-ingestion Volume. It guesses filenames from `run_date` and a count:

```python
for i in range(lookback_files):
    target_hour = run_date - timedelta(hours=i)
    filename = target_hour.strftime("%Y-%m-%dT%H-00-00Z") + ".json"
    load_path = f"{VOLUME_LOAD_PATH}/{filename}"
    ...
    try:
        dbutils.fs.ls(load_path)
        found_load_files.append(load_path)
    except Exception:
        missing_load_files.append(load_path)
```

`lookback_files` comes from `databricks.yml`'s hard-coded `base_parameters: { lookback_files: "24" }` — a number with no relationship to `ingestion_hourly.yml`'s `lookback_hours` GitHub Actions input, which defaults to `24` but can be (and on 2026-09-27, was) set far higher for a manual backfill. Whatever GHA uploads beyond the ingest side's fixed count sits in the Volume, never archived, never merged into `bronze_load`/`bronze_temperature`, invisible to every downstream notebook.

This is not hypothetical: spec 001 §8 documented this exact bug once already (GHA's `lookback_hours` was widened to 24 to cover the July outage; `01_ingest.py`'s `lookback_files` stayed at its old default of `2` until an explicit `base_parameters` override was added). The fix at the time closed that specific gap by matching one fixed number to another fixed number — it did not remove the coupling. It has now recurred with a larger backfill (300 hours) against the same fixed ceiling (24).

### 1.2 Why the Native Hourly Schedule Doesn't Help Either

`databricks.yml`'s `energy_hourly_pipeline` has its own native cron (`0 5 * * * ?`), independent of GitHub Actions' "Trigger Databricks Workflow" step. Every triggered run — whether from GHA's `databricks jobs run-now "$JOB_ID" --no-wait` (which passes no parameter overrides) or the native schedule — always uses `databricks.yml`'s static `lookback_files`. A backlog beyond that ceiling cannot self-heal through normal hourly operation; it requires a human to notice and manually override the parameter, exactly as happened in spec 001 and exactly as is suspected to have happened again here.

### 1.3 Scope of the Fix

The fix is architectural, not numerical: stop guessing what filenames *should* exist from `run_date`, and instead discover what files *actually* exist in the Volume. Cell 11's existing archival step already moves successfully-processed files out of the landing directory — so whatever remains there, by construction, **is** the backlog. No synthetic filename generation is needed.

---

## 2. Scope

### 2.1 In Scope

1. A pure, unit-testable function that selects which files to process this run from a list of what's actually present, sorted oldest-first, capped at a configurable per-run maximum — so a run never exceeds its processing budget regardless of backlog size, and any excess is left for the next run to drain (safe, because Cell 8/9's `MERGE INTO` is already idempotent).
2. Replace `01_ingest.py` Cell 5's fixed-count filename-probing loop with an actual Volume directory listing (`dbutils.fs.ls`), filtered and capped via the new pure function.
3. Rename `lookback_files` → `max_files_per_run` throughout (widget, `databricks.yml`), because the semantics genuinely changed — "how many hours to look back and guess filenames for" and "how many actually-present files to process this run, at most" are different concepts, and keeping the old name would misrepresent the new behavior to anyone reading `databricks.yml`.
4. Repurpose (not schema-change) the existing `ingestion_log.files_missing` column: previously "an expected hourly filename that wasn't found," now "backlog remaining in the Volume after this run's cap" — documented explicitly so historical and post-fix rows aren't silently conflated under one column name meaning two different things.
5. Confirm, by review, that Cell 12 (Automated Backfilling Detection, which queries `bronze_load` directly for real data gaps) needs no changes — it is already a more authoritative completeness signal than the removed per-filename probe ever was (§3.4).

### 2.2 Out of Scope

- Changing `ingestion_hourly.yml`'s `lookback_hours` default or its fetch/chunking logic.
- Any change to Cell 6 (validation), Cell 7 (table creation), Cell 8/9 (MERGE INTO bronze), Cell 10 (audit log write, beyond the one repurposed column), Cell 12 (gap detection), or Cell 13 (notebook exit) beyond consuming the new discovery-based `found_load_files`/`found_temp_files` variables — their internal logic is unchanged.
- Manually remediating the specific backlog suspected to be currently stuck in the Volume from the 2026-09-27 backfill. Once this fix is deployed, the next several scheduled hourly runs will drain it automatically, `max_files_per_run` at a time — that is the fix, not a separate manual step (§6.3).
- Any change to `02_transform.py` through `08_promote_model.py`, `src/features.py`, `src/config.py`, or specs 006/007's files.
- Changing `01_ingest.py`'s Delta table schemas (`BRONZE_TABLE_SCHEMA`, `TEMPERATURE_TABLE_SCHEMA`) or the `MERGE INTO` conditions.

---

## 3. Design

### 3.1 Pure Function — `src/ingest_discovery.py` (new)

Following the project's established pattern (`baseline.py`, `splits.py`, `forecast_timing.py`, `target_construction.py`): the Databricks I/O (`dbutils.fs.ls`) stays in the notebook and can't be unit-tested without mocking, but the actual selection logic — sort, cap, report backlog — is pure and belongs in `src/`.

```python
"""Pure selection logic for the ingestion Volume backlog-discovery pattern."""

from __future__ import annotations


def select_pending_files(
    available_paths: list[str],
    max_files_per_run: int,
) -> tuple[list[str], int]:
    """Given every filename currently sitting in a raw-ingestion Volume directory
    (already-archived files were moved out by Cell 11's archival step in earlier
    runs, so whatever remains here IS the backlog), returns
    (files_to_process_this_run, backlog_remaining_after_this_run).

    Files are sorted ascending — the "%Y-%m-%dT%H-00-00Z.json" naming convention
    makes lexicographic sort equivalent to chronological order — and capped to
    max_files_per_run, oldest first, so a run never exceeds its processing
    budget regardless of how large the backlog is. Anything beyond the cap is
    left for the next run; because MERGE INTO in this pipeline is idempotent,
    draining a backlog across several runs is safe and self-healing.
    """
    ordered = sorted(available_paths)
    selected = ordered[:max_files_per_run]
    backlog_remaining = max(0, len(ordered) - max_files_per_run)
    return selected, backlog_remaining
```

### 3.2 `01_ingest.py` Cell 5 — Before/After

**Before:**

```python
for i in range(lookback_files):
    target_hour = run_date - timedelta(hours=i)
    filename = target_hour.strftime("%Y-%m-%dT%H-00-00Z") + ".json"
    load_path = f"{VOLUME_LOAD_PATH}/{filename}"
    temp_path = f"{VOLUME_TEMP_PATH}/{filename}"
    try:
        dbutils.fs.ls(load_path)
        found_load_files.append(load_path)
    except Exception:
        missing_load_files.append(load_path)
        logger.warning(f"Load file missing: {load_path}")
    # ... identical try/except for temp_path ...
```

**After:**

```python
def list_volume_json_files(path: str) -> list[str]:
    """Lists filenames currently present in a Volume directory. Returns an
    empty list (not an error) if the directory doesn't exist yet."""
    try:
        return [f.path for f in dbutils.fs.ls(path) if f.path.endswith(".json")]
    except Exception:
        return []

pending_load_files = list_volume_json_files(VOLUME_LOAD_PATH)
pending_temp_files = list_volume_json_files(VOLUME_TEMP_PATH)

found_load_files, load_backlog_remaining = select_pending_files(pending_load_files, max_files_per_run)
found_temp_files, temp_backlog_remaining = select_pending_files(pending_temp_files, max_files_per_run)

if load_backlog_remaining > 0:
    logger.warning(
        f"{load_backlog_remaining} load files still pending after this run's "
        f"cap of {max_files_per_run}; will be picked up by a subsequent run."
    )
```

The "no files found" exit path (currently referencing a synthetic `run_date`-derived expected filename) is adjusted to report that the load directory itself was empty, rather than naming a specific hour that no longer has meaning under discovery-based selection.

### 3.3 Widget Rename — Deliberate Break, Following Spec 001 §8.4's Established Precedent

`lookback_files` → `max_files_per_run` everywhere. This is a genuine semantic change (not a relabeling of the same concept), so a silent alias would be misleading. A repo check confirms `databricks.yml` is the only caller passing this parameter — no other job, dashboard, or ad-hoc script references the old name.

Following spec 001 §8.4's already-established and explicitly-justified pattern ("Do not change 01_ingest.py's own widget default... The job-level override is the only change... ensures the notebook's default remains conservative for manual/adhoc runs"):

```python
dbutils.widgets.text("max_files_per_run", "24")  # notebook default: conservative for manual/adhoc runs
```

```yaml
# databricks.yml
- task_key: "ingest"
  notebook_task:
    notebook_path: ./notebooks/01_ingest.py
    base_parameters:
      max_files_per_run: "200"   # production ceiling; see §5 item 1 for the exact value
  environment_key: "default"
  timeout_seconds: 1800
```

### 3.4 Why Cell 12 (Backfill Detection) Doesn't Need to Change

Cell 12 already queries `bronze_load` directly for `value_mwh IS NULL` over the trailing 30 days — a data-completeness signal based on what actually made it into bronze, not on which landing filenames existed. This is strictly more authoritative than the removed per-file "missing" probe ever was: a landing file can exist but carry an API-failure wrapper (`{"status": "fetch_failed", "error": ..., "data": []}`, per `ingestion_hourly.yml`'s fetch step) or a genuinely empty array for a missing hour — either way, Cell 12's bronze-level null check catches it, while the old Cell 5 probe only ever checked "did a file exist by this name," not "did it carry usable data." No monitoring signal is lost by removing Cell 5's per-filename check; it is consolidated into a check that was already better.

### 3.5 The `ingestion_log.files_missing` Column — Repurposed, Not Renamed

The Delta table `ingestion_log` has historical rows where `files_missing` meant "an expected hourly filename, by name, that was absent." Post-fix, the same column will store `load_backlog_remaining` — "how many already-uploaded files were still waiting after this run's cap." These are different measurements. Per the project's established precedent of not rewriting history while being explicit about a definitional shift (specs 004 §3.6, 005 §3.6's "first cross-basis comparison" acknowledgment), this spec keeps the column name (avoiding a schema migration on a live audit table) and documents the shift inline in the notebook and in this spec, rather than silently changing what a stored number means.

---

## 4. Acceptance Criteria

- [ ] **AC1**: `select_pending_files()` exists in `src/ingest_discovery.py`, pure and typed. Unit tests cover: backlog smaller than cap (all selected, 0 remaining); backlog larger than cap (oldest N selected, correct remaining count); unsorted input (output still chronologically ordered); empty input; and the exact-cap boundary (backlog size == cap, 0 remaining).
  - *Verification:* `tests/test_ingest_discovery.py`.
- [ ] **AC2**: `01_ingest.py` Cell 5 uses `dbutils.fs.ls`-based directory listing + `select_pending_files()` in place of the `range(lookback_files)` filename-guessing loop; no code path references `lookback_files` anymore.
  - *Verification:* code review of the diff; `grep -r lookback_files` returns no hits outside this spec's own historical-context prose.
- [ ] **AC3**: `databricks.yml`'s `base_parameters` and `01_ingest.py`'s widget are both renamed to `max_files_per_run`, with the notebook's own default kept conservative (`24`) and the job-level override set to a generous production value, independently — mirroring spec 001 §8.4's pattern exactly.
  - *Verification:* code review of both files' diffs.
- [ ] **AC4**: The `ingestion_log.files_missing` column's semantic shift (expected-filename-absent → backlog-remaining-after-cap) is documented in an inline comment in `01_ingest.py` and in this spec; no schema change to the Delta table.
  - *Verification:* code review.
- [ ] **AC5**: Cell 12 requires zero changes — confirmed by `git diff` showing no edits to that cell, and by the reasoning in §3.4.
- [ ] **AC6**: `MERGE INTO` idempotency (Cell 8/9, unchanged) is confirmed still safe when a backlog is drained across multiple runs — i.e., reprocessing is a no-op for already-merged timestamps, never a duplicate.
  - *Verification:* existing MERGE logic untouched; a review/test confirming the `whenMatchedUpdate`/`whenNotMatchedInsertAll` conditions are unaffected by how many files feed a given run.
- [ ] **AC7**: `ruff`, `mypy src/`, and `pytest --cov-fail-under=80` all pass.
- [ ] **AC8**: Out-of-scope files untouched — `02_transform.py` through `08_promote_model.py`, `src/features.py`, `src/config.py`, `src/forecast_timing.py`, `src/target_construction.py`, `ingestion_hourly.yml`, and specs 006/007's files all byte-identical to `main`.
  - *Verification:* `git diff main..feature/008-ingest-backlog-discovery` lists only `notebooks/01_ingest.py`, `src/ingest_discovery.py` (new), `databricks.yml`, the new test file, and this spec.

---

## 5. Open Questions / Decisions Needed

| # | Question | Options | Recommendation |
|---|---|---|---|
| 1 | **Exact production `max_files_per_run` value?** | (a) 200; (b) some other number | **200, pending a live timing check** — Cell 11's per-file `dbutils.fs.mv()` archival loop is the main cost that scales with file count (each call is a Volume metadata operation); 200 leaves wide margin under `timeout_seconds: 1800` at a normal ~24-files/hour steady state, but this is a reasoned estimate, not a measured one — worth a one-off timing check on the first live run before treating it as final. |
| 2 | **Silent break for any caller still passing the old `lookback_files` name?** | (a) Accept the break, since only `databricks.yml` references it (confirmed); (b) keep a deprecated alias | **(a)** — a repo-wide check found no other reference; keeping a dead alias for a parameter name nothing calls anymore just adds confusion. |
| 3 | **Document the `files_missing` semantic shift in `GEMINI.md` too?** | Yes / No | **Yes, lightly** — same one-line-clarification pattern already used for specs 006/007's `GEMINI.md` notes; cheap, prevents future confusion reading old vs. new `ingestion_log` rows. |
| 4 | **Manually remediate the suspected current Volume backlog from the 2026-09-27 backfill, or let this fix drain it automatically?** | (a) Manual one-off cleanup now; (b) let the fix self-heal it over the next several scheduled runs | **(b)** — this is precisely the scenario the fix is designed for; once deployed, each hourly run will drain up to `max_files_per_run` of the backlog until it's exhausted, with no separate manual step needed. |

---

## 6. Implementation Plan

### 6.1 Files to Change

| File | Change |
|---|---|
| `src/ingest_discovery.py` (new) | `select_pending_files(available_paths, max_files_per_run) -> (list[str], int)` per §3.1. |
| `notebooks/01_ingest.py` | Cell 5: replace the fixed-count filename-probing loop with Volume directory listing + `select_pending_files()`, per §3.2. Widget renamed `lookback_files` → `max_files_per_run` (default `"24"`, per §3.3). One inline comment documenting the `files_missing` semantic shift (§3.5). No other cell changes. |
| `databricks.yml` | `ingest` task's `base_parameters` key renamed `lookback_files` → `max_files_per_run`, value updated per §5 item 1. |
| `tests/test_ingest_discovery.py` (new) | Unit tests per AC1. |
| `specs/008-ingest-backlog-discovery/spec.md` | This spec. |
| `GEMINI.md` (optional, per §5 item 3) | One-line clarification of the `ingestion_log.files_missing` semantic shift. |

### 6.2 Files NOT to Change

- `.github/workflows/ingestion_hourly.yml` — GHA's fetch/chunking/upload logic is unaffected; only the Databricks-side consumption of what it uploads changes.
- `notebooks/02_transform.py` through `notebooks/08_promote_model.py`.
- `src/features.py`, `src/config.py`, `src/forecast_timing.py`, `src/target_construction.py`, `src/baseline.py`, `src/splits.py`, `src/tuning.py`.
- `specs/006-gold-forecast-timestamp-offset/spec.md`, `specs/007-lgbm-target-shift-alignment/spec.md`, and their associated code.
- `01_ingest.py`'s Cell 6 (validation), Cell 7 (table creation), Cell 8/9 (MERGE), Cell 12 (gap detection), Cell 13 (exit) — beyond consuming the new discovery-based variables, their internal logic is untouched.

### 6.3 Rollout Sequence

1. Implement per §3; run `ruff`, `mypy src/`, `pytest --cov-fail-under=80`.
2. Merge; confirm `deploy.yml` succeeded on **both** `dev` and `prod` targets (standing project discipline, doubly relevant here since both targets are updated by the same push).
3. Watch the first several scheduled `energy_hourly_pipeline` runs post-deploy: confirm `ingest`'s exit payload shows `files_found` consistently at or near `max_files_per_run` (200) while the suspected 2026-09-27 backlog is being drained, tapering down to normal steady-state (~24/hour) once it's exhausted — this is also the live confirmation of whether that backlog existed and how large it actually was, resolving the open question from the prior investigation without a separate diagnostic step.
4. Confirm `bronze_load`/`bronze_temperature` row counts climb accordingly, and that `03_drift_check.py`'s 7-day current window fills in enough to stop hitting the `min_rows` guard that has been failing since 2026-09-23 (unblocking `predict` and un-freezing `gold_forecasts`) — this is the direct resolution of the live incident tracked in the 006/007 conversation thread.
5. Record the actual observed per-run duration at `max_files_per_run=200` against the `timeout_seconds: 1800` budget (§5 item 1) — confirm the estimate, don't just assume it held.

---

## 7. Future Spec Candidates (Not Implemented Here)

- **Apply the same discovery-first pattern to `ingestion_hourly.yml`'s GHA-side file generation**, if a future incident reveals a matching fixed-assumption fragility on the fetch/upload side (none currently known — GHA already generates exactly `lookback_hours` files by construction, so there is no analogous guess-vs-reality gap there today, but worth keeping in mind).
- **A repo-wide "fixed count assumes matching reality" audit**, generalizing this and spec 001's finding into a deliberate check for other places a hard-coded count silently assumes it matches an independently-varying upstream quantity — this is now a two-time recurrence of the same bug shape in this codebase, which is itself worth treating as a pattern to design out elsewhere, not just patch twice.
