"""
Hyperparameter tuning for LightGBM using Optuna with time-series cross-validation.

This module provides pure functions for defining search spaces, creating
time-series splits, Optuna objective evaluation, and orchestrating tuning runs.
It is designed to be imported by notebooks/06_train_lgbm.py and fully
unit-testable without Spark or Databricks dependencies.
"""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.distributions import BaseDistribution, FloatDistribution, IntDistribution
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error

from src.config import (
    LGBM_PARAMS,
    OPTUNA_GAP_HOURS,
    OPTUNA_N_SPLITS,
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


def get_lgbm_search_space() -> dict[str, BaseDistribution]:
    """
    Returns the Optuna search space for LightGBM hyperparameters as a dict
    of distribution objects compatible with `study.enqueue_trial` and `trial.suggest_*`.

    Designed for low-to-moderate data volume (~1000-5000 rows). Deliberately
    narrow to avoid overfitting the validation splits.
    """
    return {
        "num_leaves": IntDistribution(16, 64),
        "learning_rate": FloatDistribution(0.005, 0.1, log=True),
        "min_child_samples": IntDistribution(5, 30),
        "subsample": FloatDistribution(0.6, 1.0),
        "colsample_bytree": FloatDistribution(0.6, 1.0),
        "reg_alpha": FloatDistribution(0.0, 1.0),
        "reg_lambda": FloatDistribution(0.0, 1.0),
    }


def make_timeseries_splits(
    n_samples: int,
    n_splits: int = OPTUNA_N_SPLITS,
    gap: int = OPTUNA_GAP_HOURS,
    test_size: int = 120,
) -> list[tuple[range, range]]:
    """
    Creates time-based train/test index tuples for time-series cross-validation.

    Each fold moves forward in time. The test set is the last `test_size` rows
    of the available window at each fold. A `gap` of rows separates train from
    test to prevent shifted-target leakage in direct forecasting.

    Args:
        n_samples: Total number of rows in the dataset.
        n_splits: Number of CV folds.
        gap: Number of rows to skip between train and test (cooldown).
        test_size: Number of rows in each test fold.

    Returns:
        List of (train_indices, test_indices) tuples.

    Raises:
        ValueError: If there are not enough rows for the requested splits.
    """
    min_required = n_splits * (test_size + gap) + test_size
    if n_samples < min_required:
        raise ValueError(
            f"Insufficient data for {n_splits}-fold CV with gap={gap}, "
            f"test_size={test_size}. Need at least {min_required} rows, got {n_samples}."
        )

    splits: list[tuple[range, range]] = []
    for fold in range(n_splits):
        # The test fold moves forward each iteration
        test_end = n_samples - (n_splits - 1 - fold) * (test_size + gap)
        test_start = test_end - test_size
        train_end = test_start - gap
        train_start = 0

        splits.append((range(train_start, train_end), range(test_start, test_end)))

    return splits


def objective(
    trial: optuna.trial.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    horizon_hours: int,
) -> float:
    """
    Optuna objective function: mean CV MAE for a given LGBM parameter set.

    Uses TimeSeriesSplit to evaluate across multiple folds. Returns the
    average MAE across folds, which Optuna minimizes.

    Args:
        trial: Optuna trial object (used to suggest params and report scores).
        X: Feature matrix.
        y: Target vector.
        horizon_hours: Forecast horizon (24 or 168), used for gap/seed only.

    Returns:
        Mean MAE across CV folds.
    """
    space = get_lgbm_search_space()
    params = {
        key: trial.suggest_float(key, dist.low, dist.high, log=dist.log)
        if isinstance(dist, FloatDistribution)
        else trial.suggest_int(key, dist.low, dist.high, log=dist.log)
        if isinstance(dist, IntDistribution)
        else trial.suggest_categorical(key, dist.choices)
        for key, dist in space.items()
    }

    params = {
        **LGBM_PARAMS,
        **params,
        "n_estimators": 100,
        "verbose": -1,
        "random_state": 42 + horizon_hours,
    }

    splits = make_timeseries_splits(len(X), n_splits=OPTUNA_N_SPLITS, gap=OPTUNA_GAP_HOURS)

    mae_scores: list[float] = []
    for train_idx, test_idx in splits:
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_test_fold = y.iloc[test_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_test_fold, y_test_fold)],
            callbacks=[lgb.early_stopping(20)],
        )
        y_pred = model.predict(X_test_fold)
        mae = mean_absolute_error(y_test_fold, y_pred)
        mae_scores.append(mae)

    mean_mae = float(np.mean(mae_scores))
    return mean_mae


def run_lgbm_tuning(
    X: pd.DataFrame,
    y: pd.Series,
    horizon_hours: int,
    n_trials: int = OPTUNA_N_TRIALS,
    timeout_seconds: int | None = OPTUNA_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """
    Runs Optuna hyperparameter tuning for LightGBM.

    Args:
        X: Feature matrix.
        y: Target vector.
        horizon_hours: Forecast horizon (24 or 168).
        n_trials: Number of Optuna trials. If 0, tuning is skipped.
        timeout_seconds: Optional timeout for the tuning process.

    Returns:
        Best hyperparameter dict (can be merged over LGBM_PARAMS).
        Returns empty dict if n_trials == 0.
    """
    if n_trials <= 0:
        logger.info("Optuna tuning skipped (n_trials=0). Using default LGBM_PARAMS.")
        return {}

    logger.info(f"Starting Optuna tuning: {n_trials} trials, horizon={horizon_hours}h")

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        study_name=f"lgbm_tuning_{horizon_hours}h",
    )

    study.optimize(
        lambda trial: objective(trial, X, y, horizon_hours),
        n_trials=n_trials,
        timeout=timeout_seconds,
        show_progress_bar=False,
    )

    best_params = study.best_params
    logger.info(f"Best trial (value={study.best_value:.2f} MAE): {best_params}")
    return best_params
