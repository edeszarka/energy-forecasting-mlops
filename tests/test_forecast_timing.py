"""Unit tests for src.forecast_timing (spec 006 AC1, spec 013 T3)."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from src.forecast_timing import (
    resolve_target_timestamp,
    serving_anchor_timestamps,
    serving_target_timestamps,
)

ANCHOR = pd.Timestamp("2026-08-11T00:00:00+00:00")


@pytest.mark.parametrize("horizon_hours", [24, 168])
def test_lgbm_shifts_anchor_forward_by_horizon(horizon_hours: int) -> None:
    result = resolve_target_timestamp(ANCHOR, f"energy_lgbm_{horizon_hours}h", horizon_hours)

    assert result == ANCHOR + pd.Timedelta(hours=horizon_hours)
    assert result.tzinfo is not None


def test_prophet_leaves_anchor_unchanged() -> None:
    result = resolve_target_timestamp(ANCHOR, "energy_prophet_24h", 24)

    assert result == ANCHOR


def test_lgbm_dispatch_is_family_based_not_horizon_based() -> None:
    # "lgbm" anywhere in the name triggers the shift, exactly like the
    # prediction-call branch in generate_forecasts().
    assert resolve_target_timestamp(ANCHOR, "some_lgbm_variant", 24) == ANCHOR + pd.Timedelta(
        hours=24
    )


def test_unknown_family_is_treated_as_no_shift() -> None:
    assert resolve_target_timestamp(ANCHOR, "energy_pyfunc_24h", 24) == ANCHOR


# ── Serving target hours (spec 013 T3) ──────────────────────────────────────

RUN_AT = datetime(2026, 10, 24, 5, 0, tzinfo=UTC)


def test_serving_anchors_are_run_plus_1_to_plus_24() -> None:
    anchors = serving_anchor_timestamps(RUN_AT, 24)

    assert anchors[0] == pd.Timestamp(RUN_AT) + pd.Timedelta(hours=1)
    assert anchors[-1] == pd.Timestamp(RUN_AT) + pd.Timedelta(hours=24)
    assert len(anchors) == 24


def test_serving_targets_are_run_plus_25_to_plus_48() -> None:
    targets = serving_target_timestamps(RUN_AT, 24)

    expected = [pd.Timestamp(RUN_AT) + pd.Timedelta(hours=h) for h in range(25, 49)]
    assert targets == expected


def test_serving_targets_follow_lgbm_convention_not_naive_name() -> None:
    # AC6: the naive path must write the LightGBM target hours. Resolving with
    # the naive model's own name would NOT shift (Part A gap 2).
    anchors = serving_anchor_timestamps(RUN_AT, 24)
    lgbm_targets = [resolve_target_timestamp(a, "energy_lgbm_24h", 24) for a in anchors]

    assert serving_target_timestamps(RUN_AT, 24) == lgbm_targets
    assert [resolve_target_timestamp(a, "seasonal_naive", 24) for a in anchors] == anchors
