"""Rolling-origin backtest of the 24h LightGBM against naive baselines (spec 019 T3).

Each fold is one calendar month of issuance hours. The model is trained only on
issuances whose target hour ends before the fold starts (a purge of
FORECAST_HORIZON_HOURS), with early stopping on its own last VALIDATION_DAYS
(purged the same way). Model and baselines are scored on exactly the same rows.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.pit_features import DAILY_NAIVE, FEATURES, FORECAST_HORIZON_HOURS, WEEKLY_NAIVE

PURGE = pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
VALIDATION_DAYS = 28
MAX_ESTIMATORS = 3000
EARLY_STOPPING_ROUNDS = 100


def monthly_folds(
    issued: pd.DatetimeIndex, n_months: int
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """[start, end) of the last `n_months` calendar months that contain issuances."""
    if len(issued) == 0 or n_months <= 0:
        return []
    last = issued.max().to_period("M")
    months = pd.period_range(last - (n_months - 1), last, freq="M")
    return [(m.start_time, (m + 1).start_time) for m in months]


def split_fit_validation(
    train: pd.DataFrame, validation_days: int = VALIDATION_DAYS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last `validation_days` for early stopping; fit rows end a purge earlier."""
    val_start = train.index.max() - pd.Timedelta(days=validation_days)
    return train[train.index < val_start - PURGE], train[train.index >= val_start]


def fit_lgbm(
    train: pd.DataFrame, params: dict[str, Any], features: list[str] = FEATURES
) -> lgb.LGBMRegressor:
    """LightGBM on `target_mwh`, early-stopped on the purged validation tail."""
    fit, val = split_fit_validation(train)
    model = lgb.LGBMRegressor(**{**params, "n_estimators": MAX_ESTIMATORS, "verbose": -1})
    model.fit(
        fit[features],
        fit["target_mwh"],
        eval_set=[(val[features], val["target_mwh"])],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / actual[mask]) * 100)


def rolling_origin_backtest(
    data: pd.DataFrame,
    params: dict[str, Any],
    n_months: int,
    features: list[str] = FEATURES,
) -> pd.DataFrame:
    """Per-row out-of-sample predictions of model and baselines over the last `n_months`.

    `data` is build_features() output indexed by issuance time. Rows without an
    actual or without either baseline are left out, so all three are scored on
    the same rows.
    """
    scored = data.dropna(subset=["target_mwh", WEEKLY_NAIVE, DAILY_NAIVE])
    frames = []
    for start, end in monthly_folds(scored.index, n_months):
        train = data.dropna(subset=["target_mwh"])
        train = train[train.index < start - PURGE]
        test = scored[(scored.index >= start) & (scored.index < end)]
        if test.empty or train.empty:
            continue
        model = fit_lgbm(train, params, features)
        frames.append(
            pd.DataFrame(
                {
                    "issued_at": test.index,
                    "target_timestamp": test["target_timestamp"].to_numpy(),
                    "fold": start.strftime("%Y-%m"),
                    "actual": test["target_mwh"].to_numpy(),
                    "predicted": model.predict(test[features]),
                    "weekly_naive": test[WEEKLY_NAIVE].to_numpy(),
                    "daily_naive": test[DAILY_NAIVE].to_numpy(),
                    "best_iteration": model.best_iteration_,
                }
            )
        )
    columns = ["issued_at", "target_timestamp", "fold", "actual", "predicted"]
    columns += ["weekly_naive", "daily_naive", "best_iteration"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def summarize(rows: pd.DataFrame) -> dict[str, float]:
    """MAPE/MAE/RMSE of the model and both baselines, and skill vs the weekly naive."""
    if rows.empty:
        raise ValueError("Backtest produced no rows.")
    y = rows["actual"].to_numpy(dtype=float)
    out: dict[str, float] = {"n_rows": float(len(rows))}
    for name, col in (
        ("lgbm", "predicted"),
        ("weekly_naive", "weekly_naive"),
        ("daily_naive", "daily_naive"),
    ):
        p = rows[col].to_numpy(dtype=float)
        out[f"{name}_mape"] = mape(y, p)
        out[f"{name}_mae"] = float(np.mean(np.abs(y - p)))
        out[f"{name}_rmse"] = float(np.sqrt(np.mean((y - p) ** 2)))
    out["skill_vs_weekly_naive"] = 1 - out["lgbm_mape"] / out["weekly_naive_mape"]
    return out


def monthly_table(rows: pd.DataFrame) -> pd.DataFrame:
    """MAPE per fold for model and baselines (README/MLflow artifact)."""
    return (
        rows.groupby("fold")
        .apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "lgbm_mape": mape(g["actual"].to_numpy(), g["predicted"].to_numpy()),
                    "weekly_naive_mape": mape(g["actual"].to_numpy(), g["weekly_naive"].to_numpy()),
                    "daily_naive_mape": mape(g["actual"].to_numpy(), g["daily_naive"].to_numpy()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
