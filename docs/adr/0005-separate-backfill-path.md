# ADR-0005: A separate backfill path for historical data

- **Status:** Proposed
- **Date:** 2026-10-24
- **Deciders:** Ede Szarka
- **Implemented by:** spec 014 (T3)

## Context

Databricks Free Edition compute has no outbound internet, so GitHub Actions fetches external data and uploads it to a Unity Catalog Volume. The hourly workflow writes one file per hour per source; `01_ingest` drains at most 200 files per run and merges without letting NULLs overwrite values. Re-ingesting 2024-01-01 → today that way means about 49,000 uploads, more than a hundred job runs, and blocking hourly ingestion in the shared concurrency group. It also could not replace a corrupted value with NULL.

## Decision

Historical loads use their own path:
- a manually triggered GitHub workflow that fetches in large chunks (7-day ENTSO-E, monthly Open-Meteo Historical Forecast API) and uploads one file per chunk to a separate Volume folder, in its own concurrency group, without triggering the hourly job;
- a manually run notebook that validates the files and replaces bronze over the requested range (`replaceWhere`), with a dry-run mode.

The hourly path keeps its incremental, NULL-safe MERGE.

## Alternatives considered

- **Reuse the hourly workflow with a large look-back.** The throughput and NULL-overwrite problems above.
- **Fetch inside Databricks.** Not possible: no outbound internet.

## Consequences

- Two ingestion paths to maintain, sharing the same client and parser code.
- Backfills are explicit, auditable operations with before/after counts.
