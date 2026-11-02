"""
Unit tests for champion/challenger logic in src/promotion.py (spec 011).
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.backtest import fit_final_lgbm, mape
from src.pit_features import FEATURES, WEATHER_COLUMNS, build_features, issuance_hours
from src.promotion import (
    BENCHMARK_ONLY_REASON,
    MIN_COMMON_ROWS,
    apply_benchmark_policy,
    as_of_train_rows,
    calculate_mape,
    decide_as_of_promotion,
    is_promotable,
    pick_production_run,
    select_as_of_window,
    served_data_end,
    to_naive_utc,
)


def make_run(run_id, start_ms=0, tags=None, metrics=None):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, start_time=start_ms),
        data=SimpleNamespace(tags=tags or {}, metrics=metrics or {}),
    )


DELTA = 0.02
ENOUGH = MIN_COMMON_ROWS["lgbm"]


def gate(
    champion=5.0, as_of=5.0, challenger=4.0, naive=8.0, n=ENOUGH, first_run=False, force=False
):
    return decide_as_of_promotion(champion, as_of, challenger, naive, n, DELTA, first_run, force)


class TestDecideAsOfPromotion:
    """AC3-AC5 (spec 021)."""

    def test_both_conditions_hold_promotes(self):
        promote, reason = gate()
        assert promote is True
        assert "within the limit" in reason

    def test_as_of_worse_than_limit_keeps_champion_even_if_naive_is_beaten(self):
        promote, reason = gate(as_of=5.2)
        assert promote is False
        assert "worse than the limit" in reason

    def test_not_beating_naive_keeps_champion_even_if_as_of_is_fine(self):
        promote, reason = gate(challenger=8.0, naive=8.0)
        assert promote is False
        assert "does not beat naive" in reason

    def test_boundary_at_one_plus_delta(self):
        assert gate(as_of=5.0 * (1 + DELTA))[0] is True
        assert gate(as_of=np.nextafter(5.0 * (1 + DELTA), 99.0))[0] is False

    def test_ac3_too_few_common_rows_keeps_champion(self):
        """Good metrics do not help: there is no fallback to another window."""
        promote, reason = gate(as_of=1.0, n=ENOUGH - 1)
        assert promote is False
        assert f"{ENOUGH - 1} common rows" in reason
        assert gate(as_of=1.0, n=ENOUGH)[0] is True

    def test_ac3_no_comparison_at_all_keeps_champion(self):
        assert gate(champion=None, as_of=None, n=0)[0] is False

    @pytest.mark.parametrize("bad", [None, float("nan")])
    @pytest.mark.parametrize("field", ["as_of", "champion", "challenger", "naive"])
    def test_invalid_metric_keeps_champion(self, field, bad):
        assert gate(**{field: bad})[0] is False

    def test_zero_champion_mape_keeps_champion(self):
        assert gate(champion=0.0, as_of=0.0)[0] is False

    def test_ac4_first_run_only_the_naive_condition_counts(self):
        promote, reason = gate(champion=None, as_of=None, n=0, first_run=True)
        assert promote is True
        assert "beats naive" in reason
        # As-of inputs are ignored, however bad.
        assert gate(champion=1.0, as_of=99.0, n=0, first_run=True)[0] is True

    @pytest.mark.parametrize("naive", [4.0, 3.5])
    def test_ac4_first_run_not_beating_naive_stays_on_naive(self, naive):
        promote, reason = gate(champion=None, as_of=None, naive=naive, n=0, first_run=True)
        assert promote is False
        assert "serving seasonal naive" in reason

    @pytest.mark.parametrize(
        ("challenger", "naive"), [(9.0, None), (9.0, float("nan")), (None, 9.0)]
    )
    def test_first_run_without_valid_metrics_is_not_promoted(self, challenger, naive):
        assert gate(challenger=challenger, naive=naive, n=0, first_run=True)[0] is False

    def test_force_overrides_the_gate(self):
        assert gate(as_of=99.0, challenger=99.0, n=0, force=True)[0] is True
        assert gate(challenger=99.0, first_run=True, force=True)[0] is True


class TestAsOfGateOnRealModels:
    """AC1, AC2, AC6: real LightGBM, real features, synthetic load."""

    PARAMS = {"objective": "regression", "num_leaves": 8, "learning_rate": 0.1, "random_state": 0}
    CUTOFF = pd.Timestamp("2025-07-01 00:00")

    @pytest.fixture(scope="class")
    def data(self):
        rng = np.random.default_rng(0)
        ts = pd.date_range("2025-01-01", "2025-07-15", freq="h", tz="UTC")
        hours = np.arange(len(ts))
        temp = 10 + 8 * np.sin(2 * np.pi * hours / (24 * 365)) + rng.normal(0, 1, len(ts))
        daily = 600 * np.sin(2 * np.pi * (ts.hour - 6) / 24)
        weekly = np.where(ts.dayofweek >= 5, -500, 0)
        load = pd.DataFrame({"timestamp": ts, "value_mwh": 5000 + daily + weekly - 30 * temp})
        weather = pd.DataFrame({"timestamp": ts, **{c: temp for c in WEATHER_COLUMNS}})
        features = build_features(load, weather, issuance_hours(load, weather))
        return features.dropna(subset=["target_mwh"])

    def _gate(self, data, as_of_params) -> tuple[bool, float, float]:
        window = select_as_of_window(data, self.CUTOFF)
        train = as_of_train_rows(data, self.CUTOFF)
        y = window["target_mwh"].to_numpy()
        champion, _ = fit_final_lgbm(train, self.PARAMS)
        as_of, _ = fit_final_lgbm(train, as_of_params)
        champion_mape = mape(y, champion.predict(window[FEATURES]))
        as_of_mape = mape(y, as_of.predict(window[FEATURES]))
        # Condition (b) holds here, so the outcome depends only on condition (a).
        promote, _ = decide_as_of_promotion(
            champion_mape, as_of_mape, 4.0, 8.0, len(window), DELTA, False
        )
        return promote, as_of_mape, champion_mape

    def test_ac1_unchanged_code_passes_the_gate(self, data):
        promote, as_of_mape, champion_mape = self._gate(data, self.PARAMS)
        assert as_of_mape == champion_mape  # same rows, same parameters: the same model
        assert promote is True

    def test_ac2_a_damaged_parameter_is_stopped(self, data):
        no_splits = {**self.PARAMS, "min_child_samples": len(data) + 1}
        promote, as_of_mape, champion_mape = self._gate(data, no_splits)
        assert as_of_mape > champion_mape * (1 + DELTA)
        assert promote is False

    def test_ac6_no_row_is_in_sample(self, data):
        train = as_of_train_rows(data, self.CUTOFF)
        window = select_as_of_window(data, self.CUTOFF)
        assert train["target_timestamp"].max() == self.CUTOFF
        assert window["target_timestamp"].min() == self.CUTOFF + pd.Timedelta(hours=1)
        assert len(train) + len(window) == len(data)

    def test_tz_aware_cutoff_means_the_same_hour(self, data):
        aware = self.CUTOFF.tz_localize("UTC")
        assert len(select_as_of_window(data, aware)) == len(select_as_of_window(data, self.CUTOFF))


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
        promote, reason = gate()
        assert promote is True

        promote, reason = apply_benchmark_policy("energy_prophet_24h", promote, reason)

        assert promote is False
        assert reason.startswith(BENCHMARK_ONLY_REASON)
        assert "within the limit" in reason  # original comparison is still reported

    def test_benchmark_first_run_never_wins(self):
        promote, reason = gate(champion=None, as_of=None, n=0, first_run=True)
        assert promote is True

        assert apply_benchmark_policy("energy_prophet_24h", promote, reason)[0] is False

    def test_only_lgbm_24h_is_promotable(self):
        assert is_promotable("energy_lgbm_24h") is True
        for name in ("energy_prophet_24h", "energy_lgbm_168h", "seasonal_naive"):
            assert is_promotable(name) is False


def test_ac7_07_runs_the_as_of_gate_with_the_challengers_parameters():
    source = (Path(__file__).resolve().parents[1] / "notebooks" / "07_evaluate.py").read_text(
        encoding="utf-8"
    )
    assert "cutoff = served_data_end(champion)" in source
    assert "TRAINING_PARAMS_ARTIFACT" in source
    assert "fit_final_lgbm(as_of_train_rows(data, cutoff), params)" in source
    assert "select_as_of_window(data, cutoff)" in source
    assert "decide_as_of_promotion(" in source
    assert 'challenger.data.metrics.get("naive_mape")' in source
    for old in ("decide_promotion(", "select_common_window", "logged_metrics", "THRESHOLD"):
        assert old not in source


def test_ac7_06_logs_the_parameters_07_reads():
    source = (Path(__file__).resolve().parents[1] / "notebooks" / "06_train_lgbm.py").read_text(
        encoding="utf-8"
    )
    assert "mlflow.log_dict(params, TRAINING_PARAMS_ARTIFACT)" in source
