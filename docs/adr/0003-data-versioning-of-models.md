# ADR-0003: Version the training data and serve only models trained on the current version

- **Status:** Proposed
- **Date:** 2026-10-24
- **Deciders:** Ede Szarka
- **Implemented by:** spec 014 (T6, T8)

## Context

Until spec 013 the ENTSO-E parser mapped 15-minute positions to hours, so load history from 2026-04 to 2026-10 was corrupted, and every model was trained and evaluated on it. Nothing in MLflow records which data a run saw, so there was no way to exclude those runs except a manual, out-of-repo tag remediation, as spec 009 did for a target leak. Excluding runs by filter alone also has side effects: 07 then sees no champion and promotes unconditionally, and 08 cannot demote a production run it no longer sees.

## Decision

1. The silver table carries a Delta table property `pipeline.data_version`. Only a full rebuild from verified bronze bumps it (to `2` in spec 014).
2. Training runs tag themselves with the version they read (`data_version`).
3. Serving, evaluation, promotion and the drift reference only consider runs whose `data_version` equals `CURRENT_DATA_VERSION` in config.
4. Older runs are explicitly marked (`data_version=1`, `invalidated_reason`, `production=false`) by a remediation script kept in the repo.
5. Gold rows produced before the cut-over are archived to `gold_forecasts_v1_archive` and removed from the live table.

## Alternatives considered

- **Another boolean tag like `leak_affected`.** Works once; a second data problem would need a third tag. A version generalises.
- **Tag by training date cut-off.** A retrain between deploy and rebuild would read old data after the cut-off and be wrongly trusted. The table property ties the tag to what was actually read.
- **Delta table version as data version.** Changes on every hourly write; too fine-grained to filter on.

## Consequences

- Until the first version-2 model passes promotion, serving uses seasonal naive (ADR-0002).
- Every new data correction that invalidates models is one property bump plus one config change.
- Gold history and dashboard trends restart at the cut-over.
