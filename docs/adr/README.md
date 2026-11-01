# Architecture Decision Records

Each significant decision gets one ADR: context, decision, alternatives, consequences. ADRs are never edited after acceptance except for status; a changed decision gets a new ADR that supersedes the old one.

Format: `NNNN-short-title.md`. Statuses: Proposed, Accepted, Superseded by ADR-NNNN.

| ADR | Title | Status | Spec |
|---|---|---|---|
| [0001](0001-drop-168h-horizon.md) | Drop the 168h forecast horizon | Accepted | 013 |
| [0002](0002-prophet-benchmark-only-naive-fallback.md) | Prophet benchmark-only; seasonal-naive serving fallback | Accepted | 013 |
| [0003](0003-data-versioning-of-models.md) | Version the training data; serve only current-version models | Proposed | 014 |
| [0004](0004-data-quality-checks-fail-the-pipeline.md) | Data-quality checks fail the pipeline instead of warning | Proposed | 013, 014 |
| [0005](0005-separate-backfill-path.md) | A separate backfill path for historical data | Proposed | 014 |
| [0006](0006-live-error-monitoring-and-shadow-scoring.md) | Live error monitoring against naive, with shadow scoring | Proposed | 017 |
| [0007](0007-error-triggered-retraining.md) | Retraining triggered by forecast error; data drift diagnostic only | Proposed | 018 |
