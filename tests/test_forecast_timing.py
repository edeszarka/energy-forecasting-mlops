"""A serving run forecasts exactly τ + 24h (spec 019, ADR-0008)."""

from datetime import UTC, datetime

import pandas as pd

from src.forecast_timing import SERVED_MODEL, issuance_hour, serving_target_timestamps


def test_one_target_exactly_24h_after_the_issuance_hour():
    run_at = datetime(2026, 11, 1, 10, 7, 33, tzinfo=UTC)

    assert serving_target_timestamps(run_at) == [pd.Timestamp("2026-11-02 10:00", tz="UTC")]


def test_issuance_hour_handles_naive_and_aware_times():
    assert issuance_hour(pd.Timestamp("2026-11-01 10:59")) == pd.Timestamp(
        "2026-11-01 10:00", tz="UTC"
    )
    assert issuance_hour(pd.Timestamp("2026-11-01 11:30", tz="Europe/Budapest")) == pd.Timestamp(
        "2026-11-01 10:00", tz="UTC"
    )


def test_served_model_is_the_promotable_lightgbm():
    assert SERVED_MODEL == "energy_lgbm_24h"
