"""Point-in-time features: nothing known after issuance may reach a feature (spec 019 AC1)."""

import numpy as np
import pandas as pd
import pytest

from src.pit_features import (
    DAILY_NAIVE,
    FEATURES,
    WEATHER_COLUMNS,
    WEEKLY_NAIVE,
    build_features,
    calendar_features,
    weather_frame_from_api,
)

START = pd.Timestamp("2025-03-01 00:00", tz="UTC")
HOURS = 24 * 60


def _load(hours=HOURS, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(START, periods=hours, freq="h")
    return pd.DataFrame({"timestamp": ts, "value_mwh": 4000 + rng.normal(0, 300, hours)})


def _weather(hours=HOURS + 48, seed=1):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(START, periods=hours, freq="h")
    return pd.DataFrame({"timestamp": ts, **{c: rng.normal(10, 5, hours) for c in WEATHER_COLUMNS}})


ISSUED = [pd.Timestamp("2025-04-20 10:00", tz="UTC"), pd.Timestamp("2025-04-25 03:00", tz="UTC")]


@pytest.mark.parametrize("issued", ISSUED)
def test_ac1_load_after_the_availability_cut_never_reaches_a_feature(issued):
    load = _load()
    base = build_features(load, _weather(), [issued], include_target=False)

    changed = load.copy()
    later = changed["timestamp"] > issued - pd.Timedelta(hours=24)
    changed.loc[later, "value_mwh"] = 1e9  # including the target hour itself

    assert later.any()
    pd.testing.assert_frame_equal(build_features(changed, _weather(), [issued], False), base)


@pytest.mark.parametrize("issued", ISSUED)
def test_ac1_weather_after_the_target_hour_never_reaches_a_feature(issued):
    weather = _weather()
    base = build_features(_load(), weather, [issued], include_target=False)

    changed = weather.copy()
    later = changed["timestamp"] > issued + pd.Timedelta(hours=24)
    changed.loc[later, WEATHER_COLUMNS] = 1e9

    assert later.any()
    pd.testing.assert_frame_equal(build_features(_load(), changed, [issued], False), base)


def test_ac1_the_last_usable_load_hour_does_reach_the_features():
    issued = ISSUED[0]
    load = _load()
    base = build_features(load, _weather(), [issued], False)
    changed = load.copy()
    changed.loc[changed["timestamp"] == issued - pd.Timedelta(hours=24), "value_mwh"] += 500

    assert build_features(changed, _weather(), [issued], False)["load_t_minus_48h"].iloc[0] == (
        base["load_t_minus_48h"].iloc[0] + 500
    )


def test_feature_values_follow_their_definitions():
    load, weather = _load(), _weather()
    issued = ISSUED[0]
    row = build_features(load, weather, [issued]).iloc[0]
    v = load.set_index("timestamp")["value_mwh"]
    w = weather.set_index("timestamp")
    target = issued + pd.Timedelta(hours=24)
    h = pd.Timedelta(hours=1)

    assert row["target_timestamp"] == target.tz_localize(None)
    assert row["target_mwh"] == v[target]
    assert row[WEEKLY_NAIVE] == v[target - 168 * h]
    assert row[DAILY_NAIVE] == v[target - 48 * h] == v[issued - 24 * h]
    assert row["load_mean_24h"] == pytest.approx(v[issued - 47 * h : issued - 24 * h].mean())
    assert row["load_mean_168h"] == pytest.approx(v[issued - 191 * h : issued - 24 * h].mean())
    assert row["temperature_fc"] == w.loc[target, "temperature_fc"]
    assert row["temperature_fc_mean_24h"] == pytest.approx(
        w.loc[target - 23 * h : target, "temperature_fc"].mean()
    )


def test_serving_row_equals_the_training_row_for_the_same_issuance():
    train = build_features(_load(), _weather(), ISSUED)
    serve = build_features(_load(), _weather(), ISSUED, include_target=False)

    pd.testing.assert_frame_equal(train.drop(columns="target_mwh"), serve)


def test_missing_inputs_stay_missing():
    issued = ISSUED[0]
    load = _load()
    load = load[load["timestamp"] != issued - pd.Timedelta(hours=24)]

    row = build_features(load, pd.DataFrame(columns=["timestamp"]), [issued]).iloc[0]

    assert np.isnan(row["load_t_minus_48h"])  # not filled from a neighbour
    assert all(np.isnan(row[c]) for c in WEATHER_COLUMNS + ["temperature_fc_mean_24h"])


def test_target_without_actual_is_nan():
    late = START + pd.Timedelta(hours=HOURS - 10)

    assert np.isnan(build_features(_load(), _weather(), [late])["target_mwh"].iloc[0])


def test_columns_and_aware_or_naive_issuance():
    aware = build_features(_load(), _weather(), ISSUED)
    naive = build_features(_load(), _weather(), [t.tz_localize(None) for t in ISSUED])

    assert list(aware.columns) == ["target_timestamp", *FEATURES, "target_mwh"]
    pd.testing.assert_frame_equal(aware, naive)


def test_calendar_is_local_time_of_the_target():
    # 2025-04-21 is Easter Monday; 2025-03-30 01:00 UTC is 03:00 CEST (after the DST switch).
    targets = pd.DatetimeIndex(
        ["2025-04-20 22:30", "2025-04-21 10:00", "2025-04-22 10:00", "2025-03-30 01:00"]
    ).floor("h")
    cal = calendar_features(targets)

    assert list(cal["is_holiday"]) == [1, 1, 0, 0]  # 22:00 UTC on 04-20 is 00:00 local on 04-21
    assert list(cal["is_day_after_holiday"])[2] == 1
    assert cal["hour"].iloc[3] == 3
    assert cal["day_of_week"].iloc[1] == 0


def test_fixed_holiday_from_config_and_day_before():
    cal = calendar_features(pd.DatetimeIndex(["2025-08-19 10:00", "2025-08-20 10:00"]))

    assert list(cal["is_holiday"]) == [0, 1]
    assert list(cal["is_day_before_holiday"]) == [1, 0]


def test_weather_frame_from_api():
    hourly = {
        "time": ["2026-11-02T00:00", "2026-11-02T01:00"],
        "temperature_2m_previous_day2": [5.0, None],
        "cloud_cover_previous_day2": [80, 90],
    }

    frame = weather_frame_from_api(hourly)

    assert list(frame.columns) == ["timestamp", *WEATHER_COLUMNS]
    assert frame["timestamp"].dt.tz is not None
    assert np.isnan(frame["temperature_fc"].iloc[1])
    assert frame["humidity_fc"].isna().all()
