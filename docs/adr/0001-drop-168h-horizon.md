# ADR-0001: Drop the 168h forecast horizon

- **Status:** Accepted
- **Date:** 2026-10-24
- **Deciders:** Ede Szarka
- **Implemented by:** spec 013 (BL-19)

## Context

The pipeline forecast two horizons, 24h and 168h, with LightGBM and Prophet. An investigation of `main` @ `9dda44d` found:

- There is no hourly external benchmark for 168h. ENTSO-E publishes MAVIR's day-ahead total load forecast hourly, but its week-ahead forecast only as daily min/max.
- The 168h training split has no purge gap: validation and test windows (5 days) are shorter than the horizon (7 days), so training labels extend past the test forecast origins.
- Because the gold table keeps the first forecast per target hour, stored "168h" rows are 336h-ahead forecasts.
- Prophet 168h loses to the seasonal-naive baseline; LightGBM 168h only "wins" on the leaky split.

## Decision

Forecast only the 24h horizon. The next step (spec 015) turns it into a day-ahead product (issued once on D-1, covering hours 00–23 of D), which is the horizon of the day-ahead market and the one MAVIR's own forecast can be benchmarked against.

The `lag_168h` feature and the 168-hour rolling window stay: they are inputs, not horizons.

## Alternatives considered

- **Keep 168h and add a purge gap.** Fixes the leak but still leaves a horizon with no credible external benchmark, and doubles the work of specs 014–016.
- **Keep 168h as a daily min/max product** to match the week-ahead TSO forecast. A different product with a different target; out of scope for a portfolio pipeline.

## Consequences

- Half the models to train, evaluate, promote and monitor.
- The CV claim "24h and 168h horizons" must be updated.
- Historic 168h MLflow runs and gold rows remain but are no longer read.
