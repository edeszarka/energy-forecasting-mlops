"""
Unit tests for champion/challenger logic in src/promotion.py (spec 011).
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from src.promotion import (
    BENCHMARK_ONLY_REASON,
    apply_benchmark_policy,
    calculate_mape,
    decide_promotion,
    is_promotable,
    pick_production_run,
    select_common_window,
    served_data_end,
    to_naive_utc,
)


def make_run(run_id, start_ms=0, tags=None, metrics=None):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, start_time=start_ms),
        data=SimpleNamespace(tags=tags or {}, metrics=metrics or {}),
    )


class TestDecidePromotion:
    def test_better_challenger_is_promoted(self):
        promote, reason = decide_promotion(5.0, 4.0, 0.01, first_run=False)
        assert promote is True
        assert "improved" in reason

    def test_worse_challenger_keeps_champion(self):
        promote, reason = decide_promotion(4.0, 5.0, 0.01, first_run=False)
        assert promote is False
        assert "Keeping champion" in reason

    def test_marginal_gain_below_threshold_keeps_champion(self):
        promote, _ = decide_promotion(5.0, 4.99, 0.01, first_run=False)
        assert promote is False

    def test_first_run_promotes_unconditionally(self):
        assert decide_promotion(None, 9.0, 0.01, first_run=True)[0] is True

    def test_force_overrides_worse_challenger(self):
        assert decide_promotion(4.0, 5.0, 0.01, first_run=False, force=True)[0] is True

    @pytest.mark.parametrize("challenger", [None, float("nan")])
    def test_invalid_challenger_keeps_champion(self, challenger):
        assert decide_promotion(4.0, challenger, 0.01, first_run=False)[0] is False

    @pytest.mark.parametrize("champion", [None, float("nan"), 0.0])
    def test_invalid_champion_keeps_champion(self, champion):
        assert decide_promotion(champion, 1.0, 0.01, first_run=False)[0] is False


class TestPickProductionRun:
    def test_none_when_no_runs(self):
        assert pick_production_run([]) is None

    def test_excludes_challenger(self):
        only = make_run("challenger", tags={"promoted_at": "2026-10-01T00:00:00+00:00"})
        assert pick_production_run([only], exclude_run_id="challenger") is None

    def test_most_recently_promoted_wins_when_tag_duplicated(self):
        old = make_run("old", tags={"promoted_at": "2026-09-01T00:00:00+00:00"})
        new = make_run("new", tags={"promoted_at": "2026-10-01T00:00:00+00:00"})
        assert pick_production_run([old, new]).info.run_id == "new"
        assert pick_production_run([new, old]).info.run_id == "new"

    def test_falls_back_to_start_time_without_promoted_at(self):
        a = make_run("a", start_ms=1_000)
        b = make_run("b", start_ms=2_000)
        assert pick_production_run([a, b]).info.run_id == "b"


class TestServedDataEnd:
    def test_uses_tag_when_present(self):
        run = make_run("r", tags={"served_data_end": "2026-10-10T12:00:00"})
        assert served_data_end(run) == pd.Timestamp("2026-10-10 12:00:00")

    def test_falls_back_to_run_start_time(self):
        start = pd.Timestamp("2026-10-01 02:00:00", tz="UTC")
        run = make_run("r", start_ms=int(start.timestamp() * 1000))
        assert served_data_end(run) == pd.Timestamp("2026-10-01 02:00:00")


class TestSelectCommonWindow:
    def test_keeps_only_targets_after_champion_end(self):
        backtest = pd.DataFrame(
            {
                "target_timestamp": pd.date_range("2026-10-01", periods=5, freq="h"),
                "actual": [1.0] * 5,
                "predicted": [1.0] * 5,
            }
        )
        common = select_common_window(backtest, "2026-10-01T02:00:00")
        assert list(common["target_timestamp"]) == list(
            pd.date_range("2026-10-01 03:00", periods=2, freq="h")
        )

    def test_handles_tz_aware_champion_end(self):
        backtest = pd.DataFrame(
            {"target_timestamp": pd.date_range("2026-10-01", periods=3, freq="h")}
        )
        common = select_common_window(backtest, pd.Timestamp("2026-10-01 00:00", tz="UTC"))
        assert len(common) == 2


class TestHelpers:
    def test_mape_skips_zero_actuals(self):
        assert calculate_mape(pd.Series([0.0, 100.0]), pd.Series([5.0, 90.0])) == pytest.approx(
            10.0
        )

    def test_mape_all_zero_actuals(self):
        assert calculate_mape(pd.Series([0.0]), pd.Series([5.0])) == 0.0

    def test_mape_ignores_index_alignment(self):
        actual = pd.Series([100.0, 200.0], index=[5, 6])
        predicted = pd.Series([110.0, 180.0], index=[0, 1])
        assert calculate_mape(actual, predicted) == pytest.approx(10.0)

    def test_to_naive_utc_converts_offsets(self):
        assert to_naive_utc("2026-10-01T02:00:00+02:00") == pd.Timestamp("2026-10-01 00:00:00")


class TestApplyBenchmarkPolicy:
    """AC2: benchmark-only models never win (ADR-0002)."""

    def test_promotable_model_decision_unchanged(self):
        for wins in (True, False):
            assert apply_benchmark_policy("energy_lgbm_24h", wins, "reason") == (wins, "reason")

    def test_benchmark_model_never_wins(self):
        promote, reason = decide_promotion(5.0, 4.0, 0.01, first_run=False)
        assert promote is True

        promote, reason = apply_benchmark_policy("energy_prophet_24h", promote, reason)

        assert promote is False
        assert reason.startswith(BENCHMARK_ONLY_REASON)
        assert "improved" in reason  # original comparison is still reported

    def test_benchmark_first_run_never_wins(self):
        promote, reason = decide_promotion(None, 9.0, 0.01, first_run=True)
        assert promote is True

        assert apply_benchmark_policy("energy_prophet_24h", promote, reason)[0] is False

    def test_only_lgbm_24h_is_promotable(self):
        assert is_promotable("energy_lgbm_24h") is True
        for name in ("energy_prophet_24h", "energy_lgbm_168h", "seasonal_naive"):
            assert is_promotable(name) is False
