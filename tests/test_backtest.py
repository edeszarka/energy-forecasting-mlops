"""Rolling-origin backtest: purge, folds, same-row scoring (spec 019 AC2, AC3)."""

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


def test_mape_ignores_zero_actuals_and_empty_summary_raises():
    assert bt.mape(np.array([100.0, 0.0]), np.array([110.0, 5.0])) == pytest.approx(10.0)
    with pytest.raises(ValueError, match="no rows"):
        bt.summarize(pd.DataFrame(columns=["actual"]))
