# ADR-0007: Retraining is triggered by forecast error; data drift is diagnostic only

- **Status:** Proposed
- **Date:** 2026-11-01
- **Deciders:** Ede Szarka
- **Implemented by:** spec 018
- **Supersedes:** the retrain trigger of spec 011 (data or prediction drift for 3 consecutive hours)

## Context

- Since spec 011, `03_drift_check` started `energy_retraining_pipeline` when drift persisted for 3 hourly checks (24h cooldown). Spec 017 retired the prediction-MAE part, leaving Evidently data drift as the only trigger.
- In the 60 days before 2026-11-01, data drift fired on **every** hourly check, almost always on all non-calendar features at once. A 7-day window against a 30-day reference sees the season change; the trigger carried no information and retrained about daily, limited only by the cooldown.
- Spec 017 measures the served forecast's live error against seasonal naive on the same hours. That is the quantity a retrain is supposed to improve.

## Decision

1. `09_error_monitor` triggers retraining when the **served run** is degraded on **its own served hours** (run id in `model_version`): worse than naive on the same hours, or MAPE above the 25% ceiling, on at least 72 own hours in the 7-day window, for 24 consecutive hourly checks.
2. Spec 011's safety rules stay: a missing hour breaks the streak, a cooldown (now 72h), a pending-request flag with a 6h timeout, fire-and-forget job start, flag cleared by `08`.
3. `03_drift_check` keeps computing and storing data drift as a diagnostic and never starts a retrain.
4. While seasonal naive serves, there is no error trigger; the weekly Sunday retrain remains.

## Alternatives considered

- **Tune data drift (thresholds, reference window, per-feature tests).** Treats the symptom; even a well-tuned input-drift test says what changed, not whether forecasts got worse.
- **Trigger on any error alert (spec 017's 7-day window).** The window mixes the previous run's hours after a promotion, so a new model would be retrained for its predecessor's errors.
- **Trigger on absolute MAPE only.** Seasonal naive's weekly MAPE reaches 22.5% on holiday weeks; an absolute threshold fires on hard weeks, not on bad models.
- **Retrain on a fixed schedule only.** Simple, but slow to react to real degradation of a served model.

## Consequences

- No daily retrains while naive serves; compute drops to the weekly run.
- A served model must earn its verdict: three days of own hours before it can be judged, one full day of degradation before a retrain.
- Retraining reacts to what users of the forecast see, and the reason is stored next to the metrics in `forecast_error_monitor`.
- The trigger depends on the promotion rule for its effect: a retrain whose model does not pass 07/08 changes nothing (BUG-04 is still open).
