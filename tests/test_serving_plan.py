"""Target selection and serving decisions (spec 019b AC1, AC3, AC5, AC5b)."""

import pandas as pd
import pytest

from src.serving_plan import (
    DAILY_NAIVE,
    MODEL,
    NAIVE,
    NO_PRODUCTION_RUN,
    SKIP,
    WEATHER_TABLE_MISSING,
    PlannedTarget,
    ServingDecision,
    covered_targets,
    decide_serving,
    plan_targets,
    raise_on_model_errors,
    summarize_decisions,
)

NOW = pd.Timestamp("2026-11-02 10:00", tz="UTC")
H = pd.Timedelta(hours=1)


# ───────────── AC1: window, τ', catch-up flag, coverage ─────────────


def test_window_plans_every_target_in_the_next_24h_when_nothing_is_covered():
    plan = plan_targets(NOW + pd.Timedelta(minutes=7), covered=[])

    assert [p.target for p in plan] == [NOW + k * H for k in range(1, 25)]
    assert all(p.issued_at == p.target - 24 * H for p in plan)
    assert [p.is_catch_up for p in plan] == [True] * 23 + [False]  # τ' = now only for the last


def test_window_skips_covered_targets_and_never_plans_the_past():
    covered = [NOW + k * H for k in range(1, 24)] + [NOW, NOW - H]  # all but now+24h

    plan = plan_targets(NOW, covered)

    assert plan == [PlannedTarget(issued_at=NOW, target=NOW + 24 * H, is_catch_up=False)]


def test_window_accepts_naive_utc_covered_timestamps():
    covered = [(NOW + 24 * H).tz_localize(None)]

    assert NOW + 24 * H not in {p.target for p in plan_targets(NOW, covered)}


def test_covered_counts_served_and_legacy_null_rows_only():
    gold = pd.DataFrame(
        {
            "timestamp": [NOW + H, NOW + 2 * H, NOW + 3 * H, NOW + 4 * H],
            "is_served": [True, None, False, True],
        }
    )

    assert covered_targets(gold) == {NOW + H, NOW + 2 * H, NOW + 4 * H}
    assert covered_targets(gold.iloc[0:0]) == set()


# ───────────── AC3: synthetic run schedule with outages ─────────────


@pytest.mark.parametrize("gap_hours", [1, 5, 23, 30])
def test_schedule_every_target_with_a_run_in_the_previous_24h_gets_exactly_one_row(gap_hours):
    start = NOW
    runs = [start + k * H for k in range(48)]
    outage = set(runs[10 : 10 + gap_hours])  # runs that never happened
    written: dict[pd.Timestamp, int] = {}

    for run in (r for r in runs if r not in outage):
        for planned in plan_targets(run, covered=set(written)):
            written[planned.target] = written.get(planned.target, 0) + 1

    happened = [r for r in runs if r not in outage]
    for target in pd.date_range(start + 25 * H, runs[-1] + H, freq="h"):
        expected = any(target - 24 * H <= r < target for r in happened)
        assert (target in written) is expected, target
    assert set(written.values()) == {1}  # no target written twice


def test_schedule_gap_of_24h_or_more_loses_exactly_the_unreachable_targets():
    runs = [NOW, NOW + 30 * H]  # one run, then a 30h outage
    written = set()
    for run in runs:
        written |= {p.target for p in plan_targets(run, covered=written)}

    lost = set(pd.date_range(NOW + 25 * H, NOW + 30 * H, freq="h")) - written
    assert lost == set(pd.date_range(NOW + 25 * H, NOW + 30 * H, freq="h"))


# ───────────── AC5: decide_serving ─────────────

PLANNED = PlannedTarget(issued_at=NOW, target=NOW + 24 * H, is_catch_up=False)


def _decide(**overrides):
    args = {
        "model_state": "ok",
        "weather_table_present": True,
        "missing_model_inputs": [],
        "prediction_error": None,
        "naive_available": True,
        "daily_naive_available": True,
    }
    return decide_serving(**{**args, **overrides})


def test_decide_model_when_everything_is_there():
    assert _decide() == ServingDecision(MODEL)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"model_state": NO_PRODUCTION_RUN}, "no_production_run"),
        ({"model_state": "MlflowException"}, "model_error:MlflowException"),
        ({"weather_table_present": False}, WEATHER_TABLE_MISSING),
        (
            {"missing_model_inputs": ["temperature_fc", "load_t_minus_48h"]},
            "model_inputs_missing:temperature_fc,load_t_minus_48h",
        ),
        ({"prediction_error": "ValueError"}, "model_error:ValueError"),
    ],
)
def test_decide_naive_with_reason(overrides, reason):
    assert _decide(**overrides) == ServingDecision(NAIVE, reason)


def test_decide_reason_order_first_match_wins():
    everything_wrong = {
        "model_state": NO_PRODUCTION_RUN,
        "weather_table_present": False,
        "missing_model_inputs": ["temperature_fc"],
        "prediction_error": "ValueError",
    }
    assert _decide(**everything_wrong).fallback_reason == "no_production_run"
    assert _decide(**{**everything_wrong, "model_state": "OSError"}).fallback_reason == (
        "model_error:OSError"
    )
    assert _decide(**{**everything_wrong, "model_state": "ok"}).fallback_reason == (
        WEATHER_TABLE_MISSING
    )


def test_decide_daily_naive_when_t_minus_168h_is_missing_keeps_the_triggering_reason():
    """D8: T - 168h missing (a model input too), T - 48h present."""
    decision = _decide(missing_model_inputs=["load_t_minus_168h"], naive_available=False)

    assert decision == ServingDecision(DAILY_NAIVE, "model_inputs_missing:load_t_minus_168h")
    raise_on_model_errors([(PLANNED, decision)])  # quiet reason: no exception


def test_decide_weekly_naive_still_wins_over_daily_when_both_exist():
    decision = _decide(missing_model_inputs=["temperature_fc"], naive_available=True)

    assert decision == ServingDecision(NAIVE, "model_inputs_missing:temperature_fc")


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"model_state": NO_PRODUCTION_RUN}, "no_production_run"),
        ({"weather_table_present": False}, WEATHER_TABLE_MISSING),
        ({"model_state": "MlflowException"}, "model_error:MlflowException"),
        ({"prediction_error": "ValueError"}, "model_error:ValueError"),
    ],
)
def test_decide_daily_naive_for_every_fallback_reason(overrides, reason):
    """O1: the daily naive serves for any reason when T - 168h is missing; the
    reason stays the triggering one (model_name tells which naive served)."""
    decision = _decide(**overrides, naive_available=False)

    assert decision == ServingDecision(DAILY_NAIVE, reason)


def test_decide_model_error_with_daily_naive_still_fails_the_task():
    decision = _decide(model_state="MlflowException", naive_available=False)

    assert decision.kind == DAILY_NAIVE and decision.is_model_error
    with pytest.raises(RuntimeError, match="model_error:MlflowException"):
        raise_on_model_errors([(PLANNED, decision)])


def test_decide_skip_when_weekly_and_daily_naive_inputs_are_both_missing():
    decision = _decide(
        missing_model_inputs=["load_t_minus_48h", "load_t_minus_168h"],
        naive_available=False,
        daily_naive_available=False,
    )

    assert decision.kind == SKIP
    assert decision.fallback_reason == "model_inputs_missing:load_t_minus_48h,load_t_minus_168h"
    raise_on_model_errors([(PLANNED, decision)])


# ───────────── AC5b: model errors fail the task, quiet reasons never ─────────────


def test_model_error_raises_after_listing_the_targets():
    decisions = [(PLANNED, ServingDecision(NAIVE, "model_error:MlflowException"))]

    with pytest.raises(RuntimeError, match="model_error:MlflowException"):
        raise_on_model_errors(decisions)


@pytest.mark.parametrize(
    "decision",
    [
        ServingDecision(MODEL),
        ServingDecision(NAIVE, "no_production_run"),
        ServingDecision(NAIVE, WEATHER_TABLE_MISSING),
        ServingDecision(NAIVE, "model_inputs_missing:temperature_fc"),
        ServingDecision(SKIP, "model_inputs_missing:load_t_minus_168h"),
    ],
)
def test_quiet_reasons_never_raise(decision):
    raise_on_model_errors([(PLANNED, decision)])


def test_model_error_flag_also_on_a_skipped_target():
    assert ServingDecision(SKIP, "model_error:ValueError").is_model_error


def test_summary_counts_decisions_catch_up_and_covered():
    catch_up = PlannedTarget(issued_at=NOW - H, target=NOW + 23 * H, is_catch_up=True)
    decisions = [
        (catch_up, ServingDecision(MODEL)),
        (PLANNED, ServingDecision(NAIVE, "no_production_run")),
    ]

    assert summarize_decisions(decisions, n_covered=22) == {
        "already_covered": 22,
        "model": 1,
        "catch_up": 1,
        "naive:no_production_run": 1,
    }
