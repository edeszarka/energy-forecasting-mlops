"""Unit tests for src.target_construction.build_horizon_target (spec 007 AC1)."""

import numpy as np
import pandas as pd
import pytest

from src.target_construction import build_horizon_target


def _hourly(n: int, start: str = "2026-08-01T00:00:00+00:00") -> pd.DataFrame:
    index = pd.date_range(start=start, periods=n, freq="h", tz="UTC")
    values = np.arange(n, dtype=float) + 10.0
    return pd.DataFrame({"timestamp": index, "value_mwh": values})


# ── AC1(a): regression guard on the happy path ──────────────────────────────


@pytest.mark.parametrize("horizon_hours", [24, 168])
def test_gapless_sorted_series_is_identical_to_naive_shift(horizon_hours: int):
    df = _hourly(400)

    result = build_horizon_target(df, horizon_hours)

    # Same rows, same order, and a target identical to the naive positional
    # shift (which is correct precisely because the series is gapless + sorted).
    assert list(result["timestamp"]) == list(df["timestamp"])
    pd.testing.assert_series_equal(
        result.set_index("timestamp")["target"],
        df.set_index("timestamp")["value_mwh"].shift(-horizon_hours),
        check_names=False,
    )


# ── AC1(b): a mid-window gap never yields a cross-gap mispaired value ────────


@pytest.mark.parametrize("horizon_hours", [24, 168])
def test_mid_window_gap_never_mispairs_across_the_gap(horizon_hours: int):
    full = _hourly(800)
    gap_hour = full["timestamp"].iloc[300]
    gapped = full[full["timestamp"] != gap_hour].reset_index(drop=True)

    result = build_horizon_target(gapped, horizon_hours)
    result_by_ts = result.set_index("timestamp")["target"]
    value_by_ts = gapped.set_index("timestamp")["value_mwh"]

    # The anchor whose target hour IS the missing hour must be NaN — not a
    # value silently pulled from a later hour.
    anchor = gap_hour - pd.Timedelta(hours=horizon_hours)
    assert anchor in result_by_ts.index
    assert pd.isna(result_by_ts.loc[anchor])

    # Every row is time-aligned: target == value(t + horizon_hours), or NaN
    # exactly when that target hour is absent. No row is ever paired with a
    # value from any other hour.
    for t, target in result_by_ts.items():
        target_hour = t + pd.Timedelta(hours=horizon_hours)
        if target_hour in value_by_ts.index:
            assert target == value_by_ts.loc[target_hour]
        else:
            assert pd.isna(target)

    # Prove it against the old broken behavior: the naive positional shift on
    # the same gappy, sorted frame pairs the anchor with a later hour's value.
    naive = gapped.set_index("timestamp")["value_mwh"].shift(-horizon_hours)
    assert not pd.isna(naive.loc[anchor])
    assert naive.loc[anchor] != result_by_ts.loc[anchor]


# ── AC1(c): order independence — the actual defect ──────────────────────────


@pytest.mark.parametrize("horizon_hours", [24, 168])
def test_shuffled_input_matches_sorted_but_naive_shift_differs(horizon_hours: int):
    df = _hourly(400)
    sorted_df = df.sort_values("timestamp").reset_index(drop=True)
    shuffled_df = df.sample(frac=1.0, random_state=7).reset_index(drop=True)

    # Order-independent: shuffled and sorted inputs produce identical output.
    sorted_result = build_horizon_target(sorted_df, horizon_hours)
    shuffled_result = build_horizon_target(shuffled_df, horizon_hours)
    pd.testing.assert_frame_equal(shuffled_result, sorted_result)

    # The old positional shift() is order-dependent: applied to the same
    # shuffled rows it does not reproduce the correct time-aligned target.
    correct_target = sorted_result.set_index("timestamp")["target"]
    naive_shuffled = shuffled_df["value_mwh"].shift(-horizon_hours)
    naive_shuffled.index = shuffled_df["timestamp"]
    assert not naive_shuffled.sort_index().equals(correct_target)
