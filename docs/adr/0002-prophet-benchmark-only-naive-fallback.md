# ADR-0002: Prophet becomes a benchmark only; seasonal naive is the serving fallback

- **Status:** Accepted
- **Date:** 2026-10-24
- **Deciders:** Ede Szarka
- **Implemented by:** spec 013 (BL-20)

## Context

- Serving (`04_predict`) used LightGBM and fell back to Prophet when no LightGBM run was found.
- Prophet was evaluated, promoted and production-tagged independently of LightGBM.
- In the latest evaluation (2026-10-20), Prophet 168h lost to the seasonal-naive baseline (7.98% vs 3.26% MAPE). Prophet 24h won, but on only 5 backtest points and using the target hour's actual temperature.
- The loader also fell back to the best historic run by MAPE, so a never-promoted run could be served.

## Decision

1. Prophet is still trained and evaluated, and its metrics are reported, but it is never promoted or served.
2. Serving uses only the production-tagged LightGBM run, selected by the same rule as evaluation and promotion (latest `promoted_at`). If none exists, or loading, feature building or prediction fails, it serves a **seasonal-naive** forecast for the same target hours: the actual value one week earlier, `predicted(T) = actual(T − 168h)`.
3. The fallback is recorded in the output (`model_name = 'seasonal_naive'`) and logged as a warning, so it is visible rather than silent.

## Why T−168h for the fallback

At serving time targets reach run+48h, so T−24h is not yet observed for most of them, while T−168h is, provided silver is no more than 120h stale. It also keeps the weekday pattern.

The naive metric logged during training still uses T−24h. Spec 015 makes both use the same point-in-time definition.

## Alternatives considered

- **Keep Prophet as fallback.** A fallback that loses to naive is worse than naive.
- **No fallback; fail the run.** Safer data, but no forecast at all for the hour; the naive forecast is a well-understood floor.
- **Best historic run as fallback.** Serves models that never passed promotion.

## Consequences

- One promotable model family; the promotion and serving paths get simpler.
- A fallback period shows up in gold as `seasonal_naive` rows and can be measured.
- While naive is serving, prediction-error drift on the LightGBM model cannot fire; spec 016 replaces that monitor.
- Prophet still costs training time in the retraining job.
