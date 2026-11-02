# ADR-0009: Promotion gate with an as-of proxy and non-inferiority

- **Status:** Accepted
- **Date:** 2026-11-02
- **Deciders:** Ede Szarka
- **Implemented by:** spec 021

## Context

Since spec 020 the served model refits on all rows. It therefore trained on every row after the champion's cutoff, and cannot be scored out-of-sample there.

The previous gate scored fold models instead, requiring a gain of more than 1%. A fold model's data is 29–60 days old, while a refitted champion's is at most a retrain old. In a synthetic simulation this froze the champion: 1 replacement in 36 weekly decisions on stable data, and clearly worse served error after a level shift than always replacing.

## Decision

1. `07_evaluate` trains an as-of model with the challenger's code and parameters on the rows with a target at or before `served_data_end(champion)`, using the served model's procedure.
2. The as-of model and the champion are scored on the rows with a target after that cutoff.
3. Promote if (a) as-of MAPE ≤ champion MAPE × (1 + δ), with δ = 2% from config, and (b) the challenger's backtest `mape` < `naive_mape`.
4. Fewer than 24 common rows, or a failed comparison: the champion stays. Metrics from different windows are never compared.
5. Without a champion, only (b) counts.

## Alternatives considered

- **Non-inferiority with the fold model.** One line of code, but δ would have to compensate for the fold model's age, and a stale fold model can still block a retrain after a shift.
- **A shadow period for the challenger.** The only option that measures the promoted model itself, but it adds a week of delay and state, and 168 autocorrelated rows are noisy.
- **Always replace, with absolute guards only.** The simplest, but a code or parameter regression would go straight to production.

## Consequences

- With unchanged code and data the as-of model reproduces the champion, so weekly replacement is the normal case.
- The gate tests the procedure, not the data or the promoted model, which is never scored before it serves.
- Fresh bad data is left to data-quality checks, condition (b) and the live monitor. Each has a limit: plausible wrong values pass DQ, (b) is a 12-month average, and the monitor has no automatic rollback (BL-53).
- 07 takes two more LightGBM fits.
