# ADR-0006: Live error monitoring against naive, with shadow scoring

- **Status:** Accepted
- **Date:** 2026-11-01
- **Deciders:** Ede Szarka
- **Implemented by:** spec 017

## Context

- Nothing measured live forecast error. `03_drift_check`'s prediction-MAE check read only LightGBM rows, so it went silent while seasonal naive served (ADR-0002), and its 20% threshold was hard-coded.
- Gold had no record of which row was served.
- After the spec 014 rebuild, the first version-2 LightGBM run (4.767% MAPE) was not promoted because it did not beat its logged naive MAPE (4.685%). That logged baseline is not the served T−168h naive: seasonal naive on the rebuilt silver never had a month below 4.96% since 2024. Whether LightGBM beats the served fallback is unknown.
- Seasonal naive's weekly MAPE ranges from 7.28% (median) to 22.53% (Easter Monday week 2026), so a fixed MAPE target cannot tell a broken model from a hard week.

## Decision

1. Gold records `is_served` per row. Seasonal naive is written every hour, served or not.
2. While naive serves, the latest current-data-version LightGBM run is scored in **shadow**: written to gold with `is_served = false`, through the serving feature path, never served.
3. `09_error_monitor` runs hourly after `04_predict` and stores MAPE, MAE, bias and coverage for the served, naive and shadow streams over 24h, 7d and 30d windows, plus each model's skill against naive **on the same target hours** (`1 − model MAPE / naive MAPE`).
4. Alerts on the 7-day window: served MAPE above a breakage ceiling (25%), a served model worse than naive, low coverage (which suppresses the other two), and an informational "shadow beats naive". A failing alert fails the task once, when it starts, so the job failure email is the channel.
5. The old prediction-MAE drift check is retired.

## Alternatives considered

- **A fixed MAPE target (e.g. 5%) as the alert.** Fires on every holiday week; says nothing about whether a model is better than the fallback.
- **Fix the training baseline first (BUG-04), monitor later.** Leaves serving unmeasured, and an offline fix still would not show the effect of serving skew (BUG-02). Shadow scores show both.
- **Separate table for shadow forecasts.** Duplicates gold's schema, MERGE and actual backfill; one table with `is_served` keeps the paired comparison a simple join on the target hour.
- **Fail the task on every hour an alert is active.** One email per hour during an episode; failure fatigue.
- **A webhook or Slack integration.** Not available on Free Edition without extra secrets and services; the job failure email already works.

## Consequences

- One extra model load and prediction per hour while naive serves.
- Gold grows by up to 24 naive and 24 shadow rows per hour.
- Promotion decisions can cite live, paired evidence (`SHADOW_BEATS_NAIVE`), not only the offline backtest.
- Alerts are as good as the provisional thresholds; they are reviewed after the first month.
- During the first week after deploy the 7-day window cannot reach 80% coverage, so `LOW_COVERAGE` fails the task once, on the first run.
