"""Unit tests for src.forecast_timing.resolve_target_timestamp (spec 006 AC1)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.forecast_timing import resolve_target_timestamp

ANCHOR = pd.Timestamp("2026-08-11T00:00:00+00:00")


@pytest.mark.parametrize("horizon_hours", [24, 168])
def test_lgbm_shifts_anchor_forward_by_horizon(horizon_hours: int) -> None:
    result = resolve_target_timestamp(ANCHOR, f"energy_lgbm_{horizon_hours}h", horizon_hours)

    assert result == ANCHOR + pd.Timedelta(hours=horizon_hours)
    assert result.tzinfo is not None


@pytest.mark.parametrize("horizon_hours", [24, 168])
def test_prophet_leaves_anchor_unchanged(horizon_hours: int) -> None:
    result = resolve_target_timestamp(ANCHOR, f"energy_prophet_{horizon_hours}h", horizon_hours)

    assert result == ANCHOR


def test_lgbm_dispatch_is_family_based_not_horizon_based() -> None:
    # "lgbm" anywhere in the name triggers the shift, exactly like the
    # prediction-call branch in generate_forecasts().
    assert resolve_target_timestamp(ANCHOR, "some_lgbm_variant", 24) == ANCHOR + pd.Timedelta(
        hours=24
    )


def test_unknown_family_is_treated_as_no_shift() -> None:
    assert resolve_target_timestamp(ANCHOR, "energy_pyfunc_24h", 24) == ANCHOR
