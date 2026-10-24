# Spec 014 — Data integrity: full re-ingest and data versioning

- **Status:** Approved for Part B (rev. 2, 2026-10-24)
- **Branch:** `feature/014-data-integrity-reingest`
- **Backlog:** BL-22, BL-23 (reworded below), new BL-33…BL-37
- **Decisions:** ADR-0003 (data versioning), ADR-0004 (data-quality checks fail the pipeline), ADR-0005 (separate backfill path)
- **Evidence:** `reports/step0_entsoe_resolution_2026-10-24.md`, `reports/partA_spec014_2026-10-24.md`, owner's silver correlation check and table inventory (2026-10-24)

## 1. Context

Until spec 013, the ENTSO-E parser mapped 15-minute positions to hours. ENTSO-E has returned PT15M for MAVIR actual load at least since 2026-08-03 (GHA logs), and the owner's correlation check shows the daily cycle stretched in May, June and September 2026 (c12 0.62 / 0.68 / 0.32; c24 0.02 / 0.22 / 0.17) and degraded in the other months. Consequences:

- Bronze/silver load from 2026-04-10 to the spec 013 deploy is unreliable.
- Every MLflow run was trained and evaluated on it; every reported metric is invalid.
- Gold `actual_mwh` and predictions are unreliable; gold rows cannot be tied to a model run.

Part A also found defects that would make a naive re-ingest incomplete:

1. The parser reads only the first `Period` per `TimeSeries` (`src/api_client.py:199`); quarter-hour completeness assumes every position is present (curve type unverified).
2. The bronze load MERGE never overwrites with NULL (`01_ingest.py:303-304`).
3. `bronze_temperature` never updates humidity and cloud cover (`01_ingest.py:331-337`); weather-only files are skipped when no load file is pending (`:203`).
4. No full silver rebuild exists (`02_transform.py:324-350` stub).
5. `rolling_*` and `temperature_lag_24h` are row-based, not hour-based (`src/features.py:163-168, 245`).
6. Existing ingestion cannot carry a multi-year backfill (~49k uploads, 200-file drain cap, shared concurrency group).
7. Data-quality checks only warn.
8. No data version exists; filtering out corrupted runs would turn the next retrain into an unconditional `first_run` promotion.

## 2. Goals

1. Load and weather history from **2024-01-01** to the cut-over date are re-ingested correctly and bronze is replaced over that range.
2. Silver is rebuilt from bronze with hour-aligned features.
3. No model trained on pre-014 data can be served, used as champion, or used as drift reference.
4. Implausible load data fails the pipeline instead of warning.
5. The first model trained on clean data is promoted only if it beats seasonal naive.

## 3. Non-goals

- MAVIR forecast ingestion, as-issued weather forecasts (spec 015). Decision (owner, 2026-10-24): weather features for target hours will come **only from forecasts issued before the forecast issuance time**, in training and serving alike (Open-Meteo Previous Runs API, available from 2024-01). The weather columns rebuilt here (stitched short-range model output) remain only for features about past hours. Spec 015 lands the forecasts in their own bronze table through the backfill path built here, so silver is not rebuilt again.
- Day-ahead product, new gold table, purge gap, point-in-time naive gate (spec 016).
- Error-based monitoring (spec 017).
- Re-evaluating or rewriting historic gold rows (they are archived, §4 T8).
- ENTSO-E rate limiting beyond the existing retries (147 requests for the full range).

## 4. Changes

### T0 · Verify the raw ENTSO-E format (Part B step 0, before any code)
- The agent writes a fetch script in `reports/` that prompts for the API key with `getpass` (never from files, never echoed); the owner runs it. The agent never handles the key. The script saves raw XML responses to `reports/entsoe_raw/` for: (a) a 24h window crossing UTC midnight, (b) a 7-day window, (c) a window crossing a DST change (e.g. 2025-10-25 → 2025-10-27).
- Report per response: number of `TimeSeries`, `Period`s per series, `resolution`, `curveType`, positions per period, and the header fields (`createdDateTime`, `revisionNumber` if present).
- The agent then analyses the saved files.

### T1 · Parser correctness (BL-33)
- `src/api_client.py`: iterate **all** `Period`s of all `TimeSeries`.
- Curve types: `A01` (every position present) as today; `A03` (a missing position repeats the previous value, per ENTSO-E) by forward-filling within the period; any other curve type raises `ValueError`.
- Completeness for PT15M stays "4 quarters present after curve expansion", otherwise NULL.
- Tests: fixtures derived from the T0 raw responses (they contain no secrets; strip the request URL if it is echoed): multi-period response, A03 response, DST day, unknown curve type.

### T2 · Bronze write semantics (BL-34)
- Hourly MERGE (`01_ingest.py`): unchanged rule for load (a NULL never overwrites a value; protects against partial re-fetches).
- Temperature MERGE: update `humidity_pct` and `cloud_cover_pct` together with `temperature_c`.
- Process temperature files even when no load file is pending.

### T3 · Separate backfill path (ADR-0005)
- New workflow `.github/workflows/backfill.yml` (`workflow_dispatch`, inputs `start_date`, `end_date`), own concurrency group, no Databricks job trigger.
  - Load: ENTSO-E in 7-day chunks via `EntsoEClient`; one JSON file per chunk.
  - Weather: Open-Meteo **Historical Forecast API** (`historical-forecast-api.open-meteo.com`; same models and parameters as `/v1/forecast`, first hours of each run stitched; coverage from about 2022) in monthly chunks; one JSON file per chunk.
  - Upload to `raw_backfill/load/` and `raw_backfill/temperature/` in the existing Volume (~150 + ~35 files).
- New notebook `notebooks/00_backfill_bronze.py` (manual run; widgets `start`, `end`, `mode = dry_run | execute`):
  - reads all `raw_backfill` files, de-duplicates by timestamp, validates them with the T5 checks;
  - in `execute`, writes `bronze_load` and `bronze_temperature` with `replaceWhere` on `[start, end)` (NULLs replace values in this path);
  - archives the processed backfill files; prints row counts per month before and after.

### T4 · Hour-aligned features and a real silver rebuild (BL-35)
- `src/features.py`: rolling means/stds and `temperature_lag_24h` computed on the full hourly grid (time-based windows), with the same `min_periods` semantics expressed in hours.
- Extract the silver build in `02_transform.py` into a pure function in `src/` (input: bronze frames and window; output: core-window silver frame), used by both the hourly path and the rebuild.
- `02_transform.py` rebuild mode (widgets `rebuild_start`, `rebuild_end`): processes 30-day chunks, each read with a 168h look-back margin and a 72h look-ahead margin (centred weather fill, `is_holiday_eve`), writes each chunk's core range with `replaceWhere`. Remove the `force_full_rebuild` stub.
- The hourly path is otherwise unchanged (720h window).

### T5 · Data-quality checks that fail (ADR-0004, BL-36)
- New pure module `src/data_quality.py`, returning a list of violations:
  - every complete UTC day in the window has 24 distinct hourly rows; no duplicate timestamps;
  - non-null load values within `[LOAD_MIN_MW, LOAD_MAX_MW]`;
  - daily cycle: `corr(value, value 24h earlier) ≥ DQ_MIN_CORR_24H` and greater than `corr(value, value 12h earlier)`, over the checked window, using only hour-aligned pairs.
- Thresholds in `src/config.py`: provisional values in this PR (marked PROVISIONAL). They are calibrated on the rebuilt 2024–2025 silver in Databricks (runbook step 6: monthly c12/c24, min/max) and set in the same follow-up PR that switches `dq_mode` to `enforce` (step 9).
- Used in `00_backfill_bronze` (always enforced) and in `02_transform` after the silver frame is built and before the write.
- `02_transform` widget `dq_mode = warn | enforce`. This PR sets `warn` in `databricks.yml`; a one-line follow-up PR sets `enforce` after the rebuild (runbook step 9). In `enforce`, violations raise and nothing is written.

### T6 · Data versioning of models (ADR-0003, BL-37)
- The silver table carries a Delta table property `pipeline.data_version`. The rebuild (T4) sets it to `2` after the last chunk is written; absent means `1`.
- `05`, `06`: read the property and tag the run `data_version=<value>`.
- `src/config.py`: `CURRENT_DATA_VERSION = "2"`. Every MLflow reader adds `AND tags.data_version = '2'`: 04 loader, 07 challenger and champion, 08 production lookup, 03 reference window (Part A Q6 table).
- One-time remediation, **in the repo this time**: `scripts/remediate_014_data_version.py` (run in a Databricks notebook; `dry_run` default). For every `energy_lgbm_24h` / `energy_prophet_24h` / legacy 168h run without `data_version`: set `data_version=1`, `invalidated_reason=spec014_pt15m_parsing`, and `production=false`. Prints the before/after tags of every touched run.

### T7 · First clean model must beat naive (minimal guard until spec 016)
- `src/promotion.py` `decide_promotion`: when `first_run`, promote only if `challenger_mape < challenger_naive_mape` (the logged `naive_mape`, same test rows). Otherwise keep serving seasonal naive, with the reason recorded.
- 07 passes the challenger's `naive_mape`.

### T1b · Fail on unexpected document content (ADR-0004; added after B2 review)
- A `TimeSeries` without `curveType` raises (B2 had treated it as `A01`).
- A `quantity_Measure_Unit.name` other than `MAW` raises.

### T9 · README (added after B2 review)
- Architecture: the backfill path (ADR-0005), data versioning (ADR-0003), DQ checks and `dq_mode` (ADR-0004).
- A short "Data incident, October 2026" section: what happened (PT15M parsed as hourly, warning-only), impact (all history, models and metrics before the cut-over invalid), detection (GHA logs + correlation check), fix (specs 013–014). No accuracy numbers until a version-2 model has been evaluated.

### T8 · Gold archive (runbook only, no code)
- `CREATE TABLE gold_forecasts_v1_archive AS SELECT * FROM gold_forecasts;` then `DELETE FROM gold_forecasts;` (keeps the table and schema for 04).
- Consequence: 03's prediction-MAE drift returns `None` until enough new rows exist; the dashboard history restarts.

## 5. Runbook (owner, after merge and a successful `deploy.yml`)

Preconditions: spec 013 has been deployed for at least 48h, so the hourly 24h look-back has already rewritten the most recent hours with the fixed parser.

1. Confirm `deploy.yml` succeeded for the merge commit.
2. Run `scripts/remediate_014_data_version.py` in `dry_run`, review, then `execute`. Serving now falls back to seasonal naive (expected).
3. Run `backfill.yml` with `start_date=2024-01-01`, `end_date=<today, UTC>` (end is exclusive, so this loads through yesterday 23:00).
4. Run `00_backfill_bronze` in `dry_run` for the same range; review counts and DQ output; then `execute`.
5. Run `02_transform` in rebuild mode from `2024-01-01` to now.
6. Re-run the correlation check (below); every month must show the healthy pattern.
7. Run T8 (gold archive).
8. Trigger `energy_retraining_pipeline`; check the run's `data_version=2` tag and the 07/08 decision (T7).
9. Merge the follow-up PR that sets `dq_mode=enforce`; confirm its deploy.

## 6. Acceptance criteria

| # | Criterion | Verification |
|---|---|---|
| AC1 | Parser handles every format seen in T0 | unit tests on fixtures from T0 raw responses: multi-period, A03, DST, unknown curve type raises |
| AC2 | Temperature MERGE updates all weather columns; weather-only batches are ingested | unit/integration tests on the extracted merge logic or notebook functions |
| AC3 | Backfill replaces bronze over the range, NULLs included | test of the `00_backfill_bronze` write function with a fake Delta table; dry run prints before/after counts |
| AC4 | Features are hour-aligned | tests: a gap of N hours does not shift `rolling_*` or `temperature_lag_24h` |
| AC5 | Rebuild equals hourly build on overlapping hours | test: silver frame from the rebuild function and from the hourly path are identical on the same core hours |
| AC6 | DQ checks | tests: stretched (4×) series fails the daily-cycle check; missing hour fails; out-of-range fails; clean series passes; `enforce` raises and writes nothing |
| AC7 | No pre-014 run is served or used as champion | tests: loader, 07 champion/challenger and 08 lookup ignore runs without `data_version='2'` |
| AC8 | First clean run must beat naive | tests: `first_run` with `mape >= naive_mape` → not promoted; `mape < naive_mape` → promoted |
| AC9 | Quality gates | `ruff check .`, `ruff format --check .`, `mypy src`, `pytest --cov=src` green; coverage ≥ 88.69% |
| AC10 | Live (owner, runbook) | correlation check healthy for every month 2024-01 → now; bronze/silver 24 rows per complete day; one retrain run tagged `data_version=2` with a recorded promotion decision |

Correlation check (runbook step 6):
```sql
WITH s AS (SELECT timestamp, value_mwh FROM workspace.energy_forecasting.silver_features WHERE value_mwh IS NOT NULL)
SELECT date_trunc('MONTH', a.timestamp) AS month,
       ROUND(corr(a.value_mwh, b12.value_mwh), 2) AS c12,
       ROUND(corr(a.value_mwh, b24.value_mwh), 2) AS c24,
       COUNT(*) AS n
FROM s a
JOIN s b12 ON b12.timestamp = a.timestamp - INTERVAL 12 HOURS
JOIN s b24 ON b24.timestamp = a.timestamp - INTERVAL 24 HOURS
GROUP BY 1 ORDER BY 1;
```

## 7. Risks and open points

- **T0 may change T1.** If responses use a format not covered here, Part B stops after T0 and reports.
- **Historical Forecast API vs the hourly `/v1/forecast` path:** the same models, but stitching may differ slightly from what the hourly path stored; spec 015 replaces both with as-issued forecasts.
- **Retrains between merge and runbook step 5** read silver without the property and are tagged `data_version=1`: never served, harmless.
- **Seasonal-naive serving during the runbook** uses partly corrupted silver for T−168h until the rebuild; accepted (hours, not days).
- **Silver rows outside the rebuilt range** (none expected: silver starts 2026-05-01, the rebuild starts 2024-01-01).
