# 012 — OpenMeteo DST Crash and Silent Ingestion Failures

**Status:** Implemented (pending first live run)
**Date:** 2026-10-20
**Priority:** Critical — the DST failure fires on 2026-10-25
**Source:** Repository audit of 2026-10-20 (findings C2, H4 and H7)

---

## 1. Problem Statement

### 1.1 OpenMeteo fetch crashes on DST transitions
`OpenMeteoClient.fetch_weather()` requested times in `Europe/Budapest` and converted them with `tz_localize("Europe/Budapest")`.
- In autumn, 02:00 local time occurs twice, so pandas raises `AmbiguousTimeError`.
- In spring, 02:00 doesn't exist, so it raises `NonExistentTimeError`.

**Reproduced live** with the repo's client for 2026-10-24 → 26: `AmbiguousTimeError: Cannot infer dst time from 2026-10-25 02:00:00`.

The workflow catches the exception, so every hourly run whose fetch window touches that day uploads no weather. Those hours age out of the 24h lookback before a healthy run can re-fetch them.

**Secondary bug:** `start_date`/`end_date` are interpreted in the requested timezone. A Budapest calendar day ends at 22:00 UTC, so the last 1–2 UTC hours of `fetch_end.date()` were never requested.

### 1.2 Ingestion failures were silent and could overwrite good data
- When ENTSO-E or OpenMeteo failed, the fetch step still exited 0.
- It wrote a `{"status": "fetch_failed", ...}` wrapper (or `[]` for missing hours) under the real hourly filename.
- The upload uses `--overwrite`, so a good file for the same hour that hadn't been ingested yet was replaced.
- The run showed ✅, and nothing alerted.

### 1.3 API key prefix printed in logs
The "Diagnose ENTSO-E key" step echoed `${ENTSO_E_API_KEY:0:8}`. GitHub masks only the exact secret value, not substrings.

## 2. Changes

| File | Change |
|---|---|
| `src/config.py` | `OPENMETEO_TIMEZONE = "UTC"`. |
| `src/api_client.py` | Timestamps are localised as UTC (no conversion; no ambiguous or non-existent hours). |
| `.github/workflows/ingestion_hourly.yml` | Removed the key-diagnosis step. A file is written only for hours that have data (no wrappers or placeholders). Each source's status is recorded in `GITHUB_ENV`, and the job summary shows it. A final **Fail on fetch errors** step fails the run after upload and trigger, so data from the healthy source still flows. An OpenMeteo response with no rows counts as a failure. |
| `tests/test_openmeteo_dst.py` | Fall-back and spring-forward responses parse to unique, correct UTC hours; the request asks for `timezone=UTC`. All three tests fail on the pre-fix code. |

## 3. Verification
- `pytest`: 150 passed. `ruff` and `mypy src/` (clean CI-equivalent environment) pass.
- Live OpenMeteo call with the fixed client for 2026-10-24 → 26: 72 unique UTC hours.
- The workflow's embedded fetch script was extracted from the YAML and run against stub clients. Scenarios: both sources OK; ENTSO-E raising a multi-line error; OpenMeteo returning no rows.
  - Correct files are written, and there are no wrappers or placeholders.
  - `FETCH_FAILURES` is set correctly.
  - `GITHUB_ENV` stays well-formed.

## 4. Behaviour Notes
- When load fails but weather succeeds, `01_ingest` exits `no_files_found` (no load files). The weather files stay in the Volume and are ingested by the next run that has load files. This is the same backlog mechanism as spec 008.
- Weather hours lost to the old bug before deploy are not backfilled; silver imputes them.

## 5. Out of Scope
- The other audit findings (the `04_predict` feature skew, the deploy gate, ENTSO-E PT15M parsing).
