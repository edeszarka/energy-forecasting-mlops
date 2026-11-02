"""Rolling-origin backtest: purge, folds, same-row scoring (spec 019 AC2, AC3)."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.backtest as bt
from src.pit_features import WEATHER_COLUMNS, build_features

PARAMS = {"objective": "regression", "num_leaves": 8, "learning_rate": 0.1, "random_state": 0}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    ts = pd.date_range("2025-01-01", "2025-07-15", freq="h", tz="UTC")
    hours = np.arange(len(ts))
    temp = 10 + 8 * np.sin(2 * np.pi * hours / (24 * 365)) + rng.normal(0, 1, len(ts))
    daily = 600 * np.sin(2 * np.pi * (ts.hour - 6) / 24)
    weekly = np.where(ts.dayofweek >= 5, -500, 0)
    load = pd.DataFrame({"timestamp": ts, "value_mwh": 5000 + daily + weekly - 30 * temp})
    weather = pd.DataFrame({"timestamp": ts, **{c: temp for c in WEATHER_COLUMNS}})
    issued = pd.date_range("2025-01-20", "2025-07-14", freq="h")
    return build_features(load, weather, issued)


def test_monthly_folds_cover_the_last_months():
    issued = pd.date_range("2025-01-01", "2025-03-10 05:00", freq="h")

    folds = bt.monthly_folds(issued, 2)

    assert folds == [
        (pd.Timestamp("2025-02-01"), pd.Timestamp("2025-03-01")),
        (pd.Timestamp("2025-03-01"), pd.Timestamp("2025-04-01")),
    ]
    assert bt.monthly_folds(issued[:0], 2) == []


def test_validation_tail_is_purged_from_the_fit_rows(data):
    fit, val = bt.split_fit_validation(data, validation_days=28)

    assert val.index.min() - fit.index.max() > bt.PURGE
    assert len(fit) + len(val) < len(data)  # the purge rows are in neither


def test_ac2_no_training_target_reaches_into_the_fold(data, monkeypatch):
    seen = []
    real_fit = bt.fit_lgbm

    def spy(train, params, features=bt.FEATURES):
        seen.append(train["target_timestamp"].max())
        return real_fit(train, params, features)

    monkeypatch.setattr(bt, "fit_lgbm", spy)
    rows = bt.rolling_origin_backtest(data, PARAMS, n_months=2)

    folds = bt.monthly_folds(data.dropna(subset=["target_mwh"]).index, 2)
    assert len(seen) == 2
    for last_target, (start, _) in zip(seen, folds, strict=True):
        assert last_target < start  # issued < start - 24h  =>  target < start
    assert set(rows["fold"]) == {"2025-06", "2025-07"}


def test_ac3_model_and_baselines_are_scored_on_the_same_rows(data):
    rows = bt.rolling_origin_backtest(data, PARAMS, n_months=1)

    assert rows[["actual", "predicted", "weekly_naive", "daily_naive"]].notna().all().all()
    source = data.loc[pd.DatetimeIndex(rows["issued_at"])]
    np.testing.assert_array_equal(rows["weekly_naive"], source["load_t_minus_168h"])
    np.testing.assert_array_equal(rows["actual"], source["target_mwh"])


def test_summary_and_monthly_table(data):
    rows = bt.rolling_origin_backtest(data, PARAMS, n_months=2)

    summary = bt.summarize(rows)
    table = bt.monthly_table(rows)

    # The synthetic load is a function of the features, so the model must beat naive.
    assert summary["lgbm_mape"] < summary["weekly_naive_mape"]
    assert summary["skill_vs_weekly_naive"] == pytest.approx(
        1 - summary["lgbm_mape"] / summary["weekly_naive_mape"]
    )
    assert summary["n_rows"] == len(rows)
    assert list(table["fold"]) == ["2025-06", "2025-07"]
    assert table["n"].sum() == len(rows)


# --- Spec 020: the served model trains on all rows --------------------------------


@pytest.fixture()
def train(data):
    return data.dropna(subset=["target_mwh"])


def _record_fits(monkeypatch) -> list[pd.DataFrame]:
    fits = []
    real_fit = bt.lgb.LGBMRegressor.fit

    def spy(self, X, y, **kwargs):
        fits.append(X)
        return real_fit(self, X, y, **kwargs)

    monkeypatch.setattr(bt.lgb.LGBMRegressor, "fit", spy)
    return fits


def test_ac1_served_model_fit_rows_reach_the_last_target(train, monkeypatch):
    fits = _record_fits(monkeypatch)

    bt.fit_final_lgbm(train, PARAMS)

    early_stopping, refit = fits
    pd.testing.assert_index_equal(refit.index, train.index)
    assert train.loc[refit.index.max(), "target_timestamp"] == train["target_timestamp"].max()
    assert early_stopping.index.max() < refit.index.max() - pd.Timedelta(days=28)


def test_ac2_tree_count_equals_the_validation_result(train):
    expected = bt.fit_lgbm(train, PARAMS).best_iteration_

    model, best_iteration = bt.fit_final_lgbm(train, PARAMS)

    assert best_iteration == expected
    assert model.booster_.num_trees() == expected


def test_ac3_backtest_folds_are_unchanged(data, monkeypatch):
    fold_rows = data.dropna(subset=["target_mwh", bt.WEEKLY_NAIVE, bt.DAILY_NAIVE])
    folds = bt.monthly_folds(fold_rows.index, 2)
    assert folds == [
        (pd.Timestamp("2025-06-01"), pd.Timestamp("2025-07-01")),
        (pd.Timestamp("2025-07-01"), pd.Timestamp("2025-08-01")),
    ]

    def no_refit(*args, **kwargs):
        raise AssertionError("A backtest fold must not use the refit.")

    monkeypatch.setattr(bt, "fit_final_lgbm", no_refit)
    rows = bt.rolling_origin_backtest(data, PARAMS, n_months=2)

    train = data.dropna(subset=["target_mwh"])
    for start, end in folds:
        model = bt.fit_lgbm(train[train.index < start - bt.PURGE], PARAMS)
        test = fold_rows[(fold_rows.index >= start) & (fold_rows.index < end)]
        np.testing.assert_array_equal(
            rows.loc[rows["fold"] == start.strftime("%Y-%m"), "predicted"],
            model.predict(test[bt.FEATURES]),
        )


def test_ac4_tags_state_what_was_trained(train):
    tags = bt.final_model_tags(train, 123)

    last_target = train["target_timestamp"].max().isoformat()
    assert tags == {
        "training_data_start": train["target_timestamp"].min().isoformat(),
        "training_data_end": last_target,
        "served_data_end": last_target,
        "n_estimators_from_validation": "123",
    }


def test_ac4_tags_are_target_hours_the_drift_reference_window_can_read(train):
    tags = bt.final_model_tags(train, 123)

    # 03_drift_check reads both tags with datetime.fromisoformat, then filters silver on them.
    start = datetime.fromisoformat(tags["training_data_start"])
    end = datetime.fromisoformat(tags["training_data_end"])
    assert start.tzinfo is None and end.tzinfo is None
    assert start == train.index.min() + bt.PURGE
    assert end == train.index.max() + bt.PURGE


def test_ac4_training_notebook_logs_the_refit_and_its_tags():
    source = (Path(__file__).parents[1] / "notebooks/06_train_lgbm.py").read_text(encoding="utf-8")

    assert "model, best_iteration = fit_final_lgbm(train, params)" in source
    assert "**final_model_tags(train, best_iteration)," in source
    assert "fit_lgbm(" not in source


def test_mape_ignores_zero_actuals_and_empty_summary_raises():
    assert bt.mape(np.array([100.0, 0.0]), np.array([110.0, 5.0])) == pytest.approx(10.0)
    with pytest.raises(ValueError, match="no rows"):
        bt.summarize(pd.DataFrame(columns=["actual"]))
