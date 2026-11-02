# Spec 021 — Promotion gate with an as-of proxy

- **Status:** Approved (owner, 2026-11-02). Implemented on this branch, pending owner review.
- **Branch:** `fix/final-model-full-refit`, on top of the uncommitted spec 020
- **Decision:** owner, 2026-11-02 (as-of proxy with non-inferiority, δ = 2%)
- **Evidence:** in this folder, synthetic, 6 seeds × 2 scenarios:
  - `gate_sim.py`, `gate_sim.json`: paths A–F. **Ran on the code before spec 021** (the working tree with spec 020). It imports the old gate from `src/`, so it no longer runs on later code.
  - `gate_sim_021.py`, `gate_sim_021.json`: paths A–F plus G, the new gate. Holds its own copy of the old gate, so A–F stay reproducible. **G here is an accelerated run:** condition (b) is computed on a 2-month backtest.
  - `gate_sim_021_12m.py`, `gate_sim_021_12m.json`: path G only, with the production 12-month backtest. It is `gate_sim_021.py` with `BACKTEST_MONTHS = 12` and the other paths removed, kept as it ran, so its docstring still says 2 months.

## 1. Problem

`07_evaluate` compares the champion's served model with the challenger's **fold-model** predictions, on rows with a target after the champion's `served_data_end` (`notebooks/07_evaluate.py:171,190-193`). Both are out-of-sample, but the two have trained on data of different ages:

- A fold model's fit rows end about 29 days before its month starts (`src/backtest.py:40-41,105`), so it forecasts with data 29–60 days old.
- After spec 020 a champion has trained up to its own retrain.

So a fresher champion is held to a comparison against an older stand-in, needing a gain of more than 1% to be replaced. In the simulation, starting from a refitted champion, the current gate replaced it in 1 of 36 weekly decisions on stable data. After a level shift it served 4.85% MAPE, against 4.34% when always replacing. This also undercuts spec 018: a degradation triggers a retrain, and the gate can then keep the old model.

The refitted challenger cannot be measured directly on that window, because it trained on it.

## 2. Decision

### The as-of model
- `07_evaluate` trains an as-of model with the challenger's code and parameters, on rows with a target at or before the champion's cutoff, using the same procedure as the served model (`fit_final_lgbm`).
- Its only purpose is the comparison. It is not logged as a served model and is never promoted. The challenger's own refit is what gets promoted.

### The common window
- Rows with a target after the champion's cutoff and an actual value. Both models score exactly these rows, on `build_features` output.
- Neither model trained on them.

### The rule

With a champion, promote only if both hold:

- **(a)** `as_of_mape ≤ champion_mape × (1 + δ)`, with `δ = PROMOTION_NON_INFERIORITY_DELTA = 0.02` in `src/config.py`.
- **(b)** The challenger's logged backtest `mape < naive_mape`.

| Case | Result |
|---|---|
| Fewer than 24 common rows (`MIN_COMMON_ROWS`) | The champion stays. Metrics from different windows are not compared. |
| No champion (first run) | Only (b) counts. If it fails, serving stays on seasonal naive, as today. |
| Any metric missing or invalid, or the champion or as-of model cannot be scored | The champion stays. |
| `force_promote` | Unchanged: it overrides the gate. |
| Benchmark policy (ADR-0002) | Unchanged. |

## 3. Owner decisions

**Q1. The cutoff is `served_data_end(champion)`, not `training_data_end`.**
- For spec 020 runs the two are equal.
- The current production run (`11cca81f…`) is pre-020: its `training_data_end` is an issuance hour (2026-10-31 19:00), 24h before its last target (`served_data_end`, 2026-11-01 19:00). Using it would put targets the champion early-stopped on into the window.

**Q2. The first comparison is not like-for-like, and that is accepted, with no special case.** The current champion's fit rows end around 2026-10-03, while the as-of model refits up to 2026-11-01. It has about four more weeks of data, which favours replacing the champion. This happens once.

## 4. Known limitations

- **The gate tests the procedure, not the data or the promoted model.** With unchanged code and unchanged data, the as-of model reproduces the champion, so (a) passes. Weekly replacement is therefore the expected normal behaviour. The model actually promoted is never scored before it serves.
- **Against fresh bad data, the defences are DQ, the naive condition and the monitor**, and each has a limit:
  - DQ checks format and plausibility, so a wrong but plausible value passes.
  - (b) is a 12-month average of fold models, which barely moves with a few bad days, and those days reach the fold models late.
  - `09_error_monitor` alerts and can trigger a retrain, but there is no automatic rollback.
- **Noise:** the window may have only 24–168 autocorrelated rows. δ is a fixed margin, not a significance test.
- **δ = 2% is not calibrated on real data.** It was only tried on synthetic data (path G).
- **The window of condition (b) must not be shortened, because after a level shift it would block the retrain.** On a 2-month window all 15 rejections after a shift came from (b), and one seed served 6.75%, worse than never replacing (6.15%). On the production 12 months (b) rejected nothing, which also shows it does not react to a recent shift. See §8.
- **A comparison that keeps failing freezes the champion without an alert.** `not_scored` and `insufficient_common_rows` appear only in `model_evaluation` and in the logs.
- **Live metrics per run get shorter:** with weekly replacement a run has at most about 168 served hours, above the 72h evidence floor of spec 018 (`ERROR_RETRAIN_MIN_HOURS`).
- **Small differences are expected** even with unchanged code: later runs can read revised load or re-fetched weather for the same hours. δ absorbs these. A larger change shows up as a failed gate.
- **Evaluation takes longer:** `07_evaluate` gains two LightGBM fits (early stopping, refit), within its 1800s timeout on current evidence; to be confirmed on the first run.

## 5. Changes (after approval)

- `src/promotion.py`, pure functions:
  - `as_of_train_rows(train, cutoff)`: rows with a target ≤ cutoff.
  - `select_as_of_window(rows, cutoff)`: rows with a target > cutoff.
  - `decide_as_of_promotion(...)`: the rule in §2, including the first-run, too-few-rows and invalid-metric cases.
  - The old `decide_promotion`, `select_common_window` and the logged-metrics fallback are removed.
- `src/pit_features.py`: `issuance_hours` moves here from notebook 06, as 07 now needs it too.
- `src/config.py`: `PROMOTION_NON_INFERIORITY_DELTA = 0.02`.
- `notebooks/06_train_lgbm.py`: logs its parameters as a `training_params.json` artifact, because logged MLflow params are strings and 07 must reuse them exactly.
- `notebooks/07_evaluate.py`:
  - Reads the challenger's parameters, trains the as-of model and scores both models on the common window.
  - Removes the `mape_improvement_threshold` widget.
  - `comparison_method` becomes `as_of`, or `insufficient_common_rows` / `not_scored` when the champion stays.
  - `challenger_mape` in `model_evaluation` holds the as-of MAPE for `as_of` rows, with no schema change.
- `notebooks/08_promote_model.py`: remove the `mape_improvement_threshold` widget if nothing else reads it.
- `README.md`: the Champion/Challenger section.
- `docs/adr/0009-*`: a new ADR for the gate, listed in `docs/adr/README.md`.
- `pyproject.toml`: exclude `specs/**/*.py` from ruff, so evidence scripts stay as they ran.
- `gate_sim_021.py`: a new simulation script next to the evidence, adding the new gate as path G. The old files are not changed.

## 6. Acceptance criteria

| # | Criterion | Verification |
|---|---|---|
| AC1 | Unchanged code → the gate passes | LightGBM on synthetic data: as-of model trained with the champion's rows and parameters; `decide_as_of_promotion` promotes |
| AC2 | A deliberately worsened parameter → the gate stops it | Same data, as-of model with a damaged parameter: as-of MAPE exceeds champion × 1.02, not promoted |
| AC3 | Fewer than 24 common rows → the champion stays | Unit test at 23 and 24 rows; no fallback to logged metrics |
| AC4 | First run → only the naive condition counts | Unit test: promoted when `mape < naive_mape`, otherwise not; as-of inputs are ignored |
| AC5 | Both conditions are required | Unit tests: (a) true and (b) false, (a) false and (b) true, the exact boundary at 1 + δ, and invalid metrics |
| AC6 | No row is in-sample | Tests: as-of train targets ≤ cutoff < window targets |
| AC7 | 07 uses the pure functions and the challenger's parameters | Notebook-level test, in the style of the existing tests |
| AC8 | Simulation rerun | Path G (stable and shift) beside A–F, raw output |
| AC9 | Quality gates | Commands below |
| AC10 | Live (owner, after merge) | Runbook in §9 |

## 7. Verification commands

```
ruff check .
ruff format --check .
mypy src
pytest --cov=src
git diff --stat
```

## 8. Simulation results

Synthetic data, 6 seeds × 2 scenarios, 6 weekly retrains, so 36 decisions per cell. "Shift" is a load drop starting 10 days after the first training (−300 MW, −600 MW at midday). These numbers show direction and do not carry over to real data.

| Path | Replacements, stable | Served MAPE, stable | Replacements, shift | Served MAPE, shift |
|---|---|---|---|---|
| A: old code, old gate | 9 | 2.848 | 6 | 5.125 |
| B: 020, old gate, from a pre-020 champion | 5 | 2.848 | 5 | 5.126 |
| C: 020, old gate, from a refitted champion | 1 | 2.830 | 6 | 4.845 |
| D: 020, 5% non-inferiority with the fold model | 19 | 2.803 | 13 | 4.674 |
| E: 020, always replace | 36 | 2.792 | 36 | 4.338 |
| F: never replace | 0 | 2.816 | 0 | 5.167 |
| G: 021, **accelerated run, (b) on 2 months** | 36 | 2.792 | 21 | 4.842 |
| G: 021, (b) on the production 12 months | 36 | 2.792 | 36 | 4.338 |

- A–F in `gate_sim_021.json` are identical to `gate_sim.json` (72 of 72 records).
- The as-of model reproduced the champion in every decision (largest prediction difference 0.0, fitted without the cache), so condition (a) never stopped an unchanged procedure.
- On the production setting the gate equals "always replace" in these scenarios. A damaged parameter is stopped in `tests/test_promotion.py` (AC2), which the simulation does not measure.

## 9. Runbook (owner, after merge and a green `deploy.yml`)

1. Start `energy_retraining_pipeline` manually.
2. Check that `07_evaluate` stays within its 1800s timeout.
3. In `model_evaluation`: `comparison_method = as_of`, and the as-of MAPE (`challenger_mape`) equals the champion's MAPE (unchanged code).
   - **Exception on the first retrain:** the current champion is pre-020, with fit rows ending around 2026-10-03, while the as-of model refits up to its `served_data_end` (Q2). The two MAPEs will differ, and equality can only be expected from the second retrain on.
   - Revised load or re-fetched weather can also cause small differences (§4); a larger one needs a look.
4. The promotion happened, and the next hourly run serves with the new `run_id`.
   - Only targets not yet covered get the new run (`covered_targets`), so check the newest target hour in gold.
