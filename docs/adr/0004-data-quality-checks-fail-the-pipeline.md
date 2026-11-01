# ADR-0004: Data-quality checks fail the pipeline instead of warning

- **Status:** Accepted
- **Date:** 2026-10-24
- **Deciders:** Ede Szarka
- **Implemented by:** spec 013 (T4, resolution), spec 014 (T5)

## Context

ENTSO-E switched MAVIR actual load to 15-minute resolution at least by August 2026. The parser logged "Unexpected resolution: PT15M" on every run, and nobody read it. The data was silently corrupted for months, and all models, metrics and drift signals built on it were wrong. All other existing checks (gap rate, NaN share, duplicates) also only warned.

## Decision

1. Unexpected input formats raise (spec 013: any resolution other than PT60M/PT15M, any unknown curve type).
2. Load data must pass content checks before it is written to bronze (backfill) or silver (hourly and rebuild): 24 hourly rows per complete day, no duplicates, values within a plausible range, and a daily cycle (lag-24 correlation above a threshold and above lag-12).
3. A violation fails the task and nothing is written. Downstream tasks in the same job (drift check, predict) do not run for that hour.
4. Thresholds are calibrated on verified data and kept in config.

## Alternatives considered

- **Warn and alert.** Needs someone to read alerts; that is exactly what failed.
- **Quarantine bad rows and continue.** Hides partial failures and needs a quarantine process; overkill for a single-source pipeline.

## Consequences

- An upstream change stops forecasts until it is handled; for a portfolio pipeline that is the right trade.
- Rollout needs a warn phase until history is clean (spec 014 runbook), otherwise the check fails on the corrupted window.
