# Spec 020 — The served model trains on all rows

- **Status:** Implemented on this branch, pending owner review
- **Branch:** `fix/final-model-full-refit`
- **Decision:** owner, 2026-11-02

## 1. Problem

`06_train_lgbm` trains the served model with `fit_lgbm`, which holds out the last 28 days for early stopping and purges 24h before them (`src/backtest.py:39-41`). So the served model's trees never see about the last 29 days of issuances, and a weekly retrain does not learn from the latest month.

The docs and tags say otherwise:
- The README and the notebook say the final model is trained on all rows.
- The production run of 2026-11-01 (`11cca81f62da4b6f85a497d57e32c3ed`) has `training_data_end = 2026-10-31T19:00`, but its fit rows end at about 2026-10-02 19:00.

This is not a leak, and the backtest folds use the same split, so the logged metrics are not biased by it.

## 2. Decision

The served model is built in two steps:
1. Early stopping on the validation tail, exactly as now, gives `best_iteration`.
2. Refit on all rows with an actual, with `n_estimators = best_iteration` and no early stopping.

Backtest folds do not change: they still use `fit_lgbm` and score its predictions.

Tags on the run:
- `training_data_start`: the first target hour the model trained on. It was the first issuance hour, 24h earlier.
- `training_data_end`: the last target hour the model trained on. It was the last issuance hour, 24h earlier.
- `served_data_end`: unchanged in meaning; now equal to `training_data_end`.
- `n_estimators_from_validation`: new, the `best_iteration` used for the refit.

## 3. Changes

- `src/backtest.py`: `fit_final_lgbm(train, params, features)` returns the refitted model and `best_iteration`; `final_model_tags(train, n_estimators)` builds the tags above. `rolling_origin_backtest` and `fit_lgbm` are untouched.
- `notebooks/06_train_lgbm.py`: uses both; the description matches the code.
- `README.md`: the training paragraph matches the code.

## 4. Consequences and known limitations

- **The refit has no validation score of its own.** Every row is in training, and a tree count found on fewer rows is assumed to carry over. The logged `mape` still describes the fold models, not the served model. The live monitor (`09_error_monitor`) is what measures the served model.
- **The promotion gate does not see the refit.** `07_evaluate` scores the challenger from its logged backtest rows, which come from fold models. A refitted challenger therefore gets no credit for its fresher data, and needs more than 1% relative improvement from a fold model to replace the champion. Out of scope here.
- **The current production run stays as it is** until a new run is promoted. Its tags are not rewritten.
- `03_drift_check` reads both tags as its reference window and filters silver `timestamp` (the load hour) on them. For new runs both ends move 24h later compared with the old tag meaning, so the window is exactly the load hours the model learned as targets. Its columns are still silver's, not the model's (BL-44). Drift is diagnostic only (spec 018).
- `07_evaluate` selects rows with a target after the champion's `served_data_end`. This remains correct, and is now exact rather than about 29 days stricter than necessary.

## 5. Acceptance criteria

| # | Criterion | Verification |
|---|---|---|
| AC1 | The served model's fit rows reach the last available target hour | `tests/test_backtest.py`: the rows passed to the refit equal all train rows, up to the last `target_timestamp` |
| AC2 | The served model's tree count equals the value found on validation | `tests/test_backtest.py`: tree count = `fit_lgbm(...).best_iteration_` = the returned value |
| AC3 | Backtest folds are unchanged | `tests/test_backtest.py`: fold bounds are pinned; fold predictions equal a direct `fit_lgbm` on the same cutoff; the refit is never called |
| AC4 | Tags state what was trained | `tests/test_backtest.py`: `training_data_start` and `training_data_end` = first and last target hour; `n_estimators_from_validation` = `best_iteration`; the notebook logs these tags |
| AC5 | Quality gates pass | commands below |
| AC6 | Live (owner, after merge) | the next retrain's run shows the new tags, and `training_data_end` equals `served_data_end` |

## 6. Verification commands

```
ruff check .
ruff format --check .
mypy src
pytest --cov=src
git diff --stat
git diff notebooks/06_train_lgbm.py src/backtest.py
```
