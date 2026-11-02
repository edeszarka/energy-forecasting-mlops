"""Synthetic weekly-retrain simulation of the promotion gate. Uses the repo's real functions.

Investigation only. Run from the repo root: python <this file> [n_seeds]
Fold models cached by cutoff, since identical rows give an identical deterministic model.
"""

import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import src.backtest as bt
from src.config import LGBM_PARAMS
from src.pit_features import FEATURES, WEATHER_COLUMNS, build_features
from src.promotion import calculate_mape, decide_promotion, select_common_window

PARAMS = {k: v for k, v in LGBM_PARAMS.items() if k not in ("n_estimators", "verbose")}
WEEKS = 6
WEEK = pd.Timedelta(days=7)
BACKTEST_MONTHS = 2  # only the newest folds can reach the common window
DELTA_NONINF = 0.05

_real_fit = bt.fit_lgbm
_cache: dict = {}


def cached_fit(train, params, features=FEATURES):
    key = (train.index.max(), len(train))
    if key not in _cache:
        _cache[key] = _real_fit(train, params, features)
    return _cache[key]


bt.fit_lgbm = cached_fit


def make_series(seed: int, scenario: str, shift_at: pd.Timestamp):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", "2026-03-15", freq="h", tz="UTC")
    n = len(ts)
    h = np.arange(n)
    temp = 11 + 12 * np.sin(2 * np.pi * (h / (24 * 365.25) - 0.28)) + 5 * np.sin(2 * np.pi * h / 24 - 2)
    weather_err = np.zeros(n)
    noise = np.zeros(n)
    for i in range(1, n):
        weather_err[i] = 0.98 * weather_err[i - 1] + rng.normal(0, 0.45)
        noise[i] = 0.96 * noise[i - 1] + rng.normal(0, 40)
    temp_true = temp + weather_err
    level = 5000 + np.cumsum(rng.normal(0, 1.2, n))  # slow wandering level
    daily = 700 * np.sin(2 * np.pi * (ts.hour - 6) / 24)
    weekly = np.where(ts.dayofweek >= 5, -550, 0)
    climate = 18 * np.abs(temp_true - 16)
    load = level + daily + weekly + climate + noise
    if scenario == "shift":
        after = ts >= shift_at
        midday = np.isin(ts.hour, [9, 10, 11, 12, 13, 14])
        load = load + np.where(after, -300, 0) + np.where(after & midday, -300, 0)
    load_df = pd.DataFrame({"timestamp": ts, "value_mwh": load})
    weather = pd.DataFrame({"timestamp": ts, **{c: temp for c in WEATHER_COLUMNS}})
    weather["cloud_cover_fc"] = 50 + 20 * np.sin(2 * np.pi * h / (24 * 9))
    weather["humidity_fc"] = 60 + 10 * np.sin(2 * np.pi * h / (24 * 5))
    weather["shortwave_fc"] = np.clip(300 * np.sin(2 * np.pi * (ts.hour - 6) / 24), 0, None)
    issued = pd.date_range("2024-01-01", "2026-03-14", freq="h")
    return build_features(load_df, weather, issued).dropna(subset=["target_mwh"])


def available(data: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    return data[data["target_timestamp"] <= now]


def served_rows(data, start, end):
    return data[(data["target_timestamp"] > start) & (data["target_timestamp"] <= end)]


def score(model, rows) -> np.ndarray:
    return np.abs(rows["target_mwh"].to_numpy() - model.predict(rows[FEATURES])) / rows[
        "target_mwh"
    ].to_numpy()


def old_model(train):
    return bt.fit_lgbm(train, PARAMS)


def new_model(train):
    key = ("refit", train.index.max(), len(train))
    if key not in _cache:
        _cache[key] = bt.fit_final_lgbm(train, PARAMS)[0]
    return _cache[key]


def run(data, d0, policy, build, initial):
    """One path: each week builds a challenger, the gate decides, then a week is served."""
    train0 = available(data, d0)
    champ, champ_end = initial(train0), train0["target_timestamp"].max()
    errors, log = [], []
    for w in range(1, WEEKS + 1):
        now = d0 + w * WEEK
        avail = available(data, now)
        rows = bt.rolling_origin_backtest(avail, PARAMS, BACKTEST_MONTHS)
        common = select_common_window(rows, champ_end)
        feats = data.loc[pd.DatetimeIndex(common["issued_at"]), FEATURES]
        champ_mape = calculate_mape(common["actual"], pd.Series(champ.predict(feats)))
        proxy_mape = calculate_mape(common["actual"], common["predicted"])
        if policy == "current":
            win = decide_promotion(champ_mape, proxy_mape, 0.01, False)[0]
        elif policy == "noninferior":
            win = proxy_mape <= champ_mape * (1 + DELTA_NONINF)
        elif policy == "always":
            win = True
        else:
            win = False
        log.append(
            {
                "week": w,
                "n_common": len(common),
                "champ": round(champ_mape, 3),
                "proxy": round(proxy_mape, 3),
                "promoted": bool(win),
            }
        )
        if win:
            champ, champ_end = build(avail), avail["target_timestamp"].max()
        errors.append(score(champ, served_rows(data, now, now + WEEK)))
    return float(np.concatenate(errors).mean() * 100), log


PATHS = {
    "A pre-020 kod, jelenlegi kapu": ("current", old_model, old_model),
    "B 020 + jelenlegi kapu, pre-020 championrol": ("current", new_model, old_model),
    "C 020 + jelenlegi kapu, refit championrol": ("current", new_model, new_model),
    "D 020 + non-inferiority 5% fold-proxyval": ("noninferior", new_model, new_model),
    "E 020 + mindig csere (=as-of proxy valtozatlan kodnal)": ("always", new_model, new_model),
    "F soha nincs csere (refit champion)": ("never", new_model, new_model),
}


def main(n_seeds: int) -> None:
    out = []
    starts = ["2025-11-02", "2025-11-09", "2025-11-16", "2025-11-23", "2025-11-30", "2025-12-07"]
    for seed in range(n_seeds):
        d0 = pd.Timestamp(starts[seed % len(starts)])
        for scenario in ("stable", "shift"):
            shift_at = (d0 + pd.Timedelta(days=10)).tz_localize("UTC")
            data = make_series(seed, scenario, shift_at)
            _cache.clear()
            for name, (policy, build, initial) in PATHS.items():
                t = time.time()
                mape, log = run(data, d0, policy, build, initial)
                rec = {
                    "seed": seed,
                    "d0": str(d0.date()),
                    "scenario": scenario,
                    "path": name,
                    "served_mape": round(mape, 3),
                    "promotions": sum(x["promoted"] for x in log),
                    "first_promoted_week": next((x["week"] for x in log if x["promoted"]), None),
                    "log": log,
                }
                out.append(rec)
                print(seed, scenario, name, rec["served_mape"], rec["promotions"],
                      [(x["champ"], x["proxy"], int(x["promoted"])) for x in log],
                      f"{time.time() - t:.0f}s", flush=True)
            json.dump(out, open(sys.argv[2], "w"))

    df = pd.DataFrame(out)
    summary = df.groupby(["scenario", "path"]).agg(
        runs=("seed", "size"),
        mean_promotions=("promotions", "mean"),
        never_promoted=("promotions", lambda s: int((s == 0).sum())),
        served_mape=("served_mape", "mean"),
    )
    print(summary.round(3).to_string())


main(int(sys.argv[1]))
