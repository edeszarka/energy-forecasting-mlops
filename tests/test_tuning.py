"""
Unit tests for the Optuna hyperparameter tuning logic in src/tuning.py.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import LGBM_PARAMS
from src.tuning import (
    get_lgbm_search_space,
    make_timeseries_splits,
    objective,
    run_lgbm_tuning,
)


class TestGetLgbmSearchSpace:
    def test_returns_dict_with_expected_keys(self):
        space = get_lgbm_search_space()
        expected_keys = {
            "num_leaves",
            "learning_rate",
            "min_child_samples",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
        }
        assert set(space.keys()) == expected_keys

    def test_contains_only_distribution_objects(self):
        from optuna.distributions import BaseDistribution

        space = get_lgbm_search_space()
        for v in space.values():
            assert isinstance(v, BaseDistribution)


class TestMakeTimeseriesSplits:
    def test_returns_correct_number_of_splits(self):
        n = 1000
        splits = make_timeseries_splits(n, n_splits=3, gap=168, test_size=120)
        assert len(splits) == 3

    def test_each_split_has_train_and_test_indices(self):
        n = 1000
        splits = make_timeseries_splits(n, n_splits=3, gap=168, test_size=120)
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # No overlap
            assert set(train_idx).isdisjoint(set(test_idx))

    def test_gap_between_train_and_test(self):
        n = 1000
        splits = make_timeseries_splits(n, n_splits=2, gap=24, test_size=48)
        for train_idx, test_idx in splits:
            max_train = max(train_idx)
            min_test = min(test_idx)
            assert min_test - max_train > 24

    def test_first_fold_starts_at_beginning(self):
        n = 1000
        splits = make_timeseries_splits(n, n_splits=3, gap=168, test_size=120)
        first_train, _ = splits[0]
        assert min(first_train) == 0

    def test_raises_on_insufficient_data(self):
        n = 50
        with pytest.raises(ValueError, match="Insufficient data"):
            make_timeseries_splits(n, n_splits=3, gap=24, test_size=48)


class TestObjective:
    @pytest.fixture
    def mock_data(self):
        n = 1000
        rng = np.random.default_rng(42)
        X = pd.DataFrame(
            {
                "temperature_c": rng.normal(10, 5, n),
                "lag_24h": rng.normal(4000, 500, n),
                "hour_of_day": rng.integers(0, 24, n),
            }
        )
        y = pd.Series(rng.normal(4000, 500, n))
        return X, y

    def test_returns_float(self, mock_data):
        X, y = mock_data
        from optuna import create_study

        study = create_study(direction="minimize")
        trial = study.ask()
        score = objective(trial, X, y, horizon_hours=24)
        assert isinstance(score, float)
        assert score > 0


class TestRunLgbmTuning:
    @pytest.fixture
    def mock_data(self):
        n = 1000
        rng = np.random.default_rng(42)
        X = pd.DataFrame(
            {
                "temperature_c": rng.normal(10, 5, n),
                "lag_24h": rng.normal(4000, 500, n),
                "lag_48h": rng.normal(4000, 500, n),
                "lag_168h": rng.normal(4000, 500, n),
                "rolling_7d_mean": rng.normal(4000, 200, n),
                "rolling_7d_std": abs(rng.normal(200, 50, n)),
                "rolling_24h_mean": rng.normal(4000, 300, n),
                "hour_of_day": rng.integers(0, 24, n),
                "day_of_week": rng.integers(0, 7, n),
                "month": rng.integers(1, 13, n),
                "is_weekend": rng.integers(0, 2, n).astype(bool),
                "is_holiday": rng.integers(0, 2, n).astype(bool),
                "humidity_pct": rng.uniform(30, 90, n),
                "cloud_cover_pct": rng.uniform(0, 100, n),
            }
        )
        y = pd.Series(rng.normal(4000, 500, n))
        return X, y

    def test_returns_dict_with_expected_keys(self, mock_data):
        X, y = mock_data
        result = run_lgbm_tuning(X, y, horizon_hours=24, n_trials=2)
        expected_keys = {
            "num_leaves",
            "learning_rate",
            "min_child_samples",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
        }
        assert set(result.keys()) == expected_keys

    def test_zero_trials_returns_empty_dict(self, mock_data):
        X, y = mock_data
        result = run_lgbm_tuning(X, y, horizon_hours=24, n_trials=0)
        assert result == {}

    def test_can_override_defaults_via_merge(self, mock_data):
        """Verifies that tuned params can override LGBM_PARAMS correctly."""
        X, y = mock_data
        tuned = run_lgbm_tuning(X, y, horizon_hours=24, n_trials=2)
        merged = {**LGBM_PARAMS, **tuned}
        for key in tuned:
            assert merged[key] == tuned[key]
