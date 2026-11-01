"""Tests for error-triggered retraining (spec 018 AC1–AC5)."""

import ast
from pathlib import Path

import pandas as pd
import pytest

from src.error_trigger import (
    Degradation,
    RetrainThresholds,
    assess_degradation,
    current_served_run,
    decide_error_retrain,
    start_retrain,
)

END = pd.Timestamp("2026-11-20 00:00", tz="UTC")
T = RetrainThresholds(
    min_hours=72, streak_hours=24, cooldown_hours=72, max_skill=0.0, mape_ceiling=25.0
)


def _gold(model_name, model_version, hours, predicted, actual, is_served, end=END):
    ts = pd.date_range(end=end, periods=hours, freq="h")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "model_name": model_name,
            "model_version": model_version,
            "predicted_mwh": predicted,
            "actual_mwh": float("nan") if actual is None else actual,
            "is_served": is_served,
            "is_backfilled": False,
        }
    )


def _naive(hours=168, predicted=110.0, end=END):
    return _gold("seasonal_naive", "seasonal_naive_t-168h", hours, predicted, 100.0, False, end)


def _concat(*frames):
    return pd.concat(frames, ignore_index=True)


# ───────────── AC1: only the current run's own served hours ─────────────


def test_ac1_new_run_is_not_blamed_for_the_previous_runs_hours():
    old_end = END - pd.Timedelta(hours=10)
    gold = _concat(
        _gold("energy_lgbm_24h", "old-run", 150, 200.0, 100.0, True, end=old_end),  # 100% off
        _gold("energy_lgbm_24h", "new-run", 10, 101.0, 100.0, True),
        _naive(),
    )

    d = assess_degradation(gold, END, 168, T)

    assert d.run_id == "new-run"
    assert d.n_hours == 10
    assert d.degraded is False
    assert "need 72" in d.reason


def test_ac1_current_run_is_the_latest_served_target_hour():
    gold = _concat(
        _gold("energy_lgbm_24h", "old-run", 5, 1.0, 1.0, True, end=END - pd.Timedelta(hours=5)),
        _gold("energy_lgbm_24h", "new-run", 5, 1.0, 1.0, True),
    )

    assert current_served_run(gold) == ("energy_lgbm_24h", "new-run")
    assert current_served_run(gold.iloc[0:0]) is None


def test_ac1_backfilled_and_shadow_rows_are_not_served():
    shadow = _gold("energy_lgbm_24h", "shadow-run", 100, 1.0, 1.0, False)
    backfill = _naive().assign(is_backfilled=True, is_served=None)

    assert current_served_run(_concat(shadow, backfill)) is None


# ───────────── AC2: degradation rule ─────────────


@pytest.mark.parametrize(
    ("model_pred", "degraded"),
    [(110.0, False), (110.01, True), (105.0, False)],  # naive is 10% off
)
def test_ac2_worse_than_naive_on_same_hours(model_pred, degraded):
    gold = _concat(_gold("energy_lgbm_24h", "run", 168, model_pred, 100.0, True), _naive())

    assert assess_degradation(gold, END, 168, T).degraded is degraded


@pytest.mark.parametrize(("model_pred", "degraded"), [(125.0, False), (125.5, True)])
def test_ac2_mape_ceiling(model_pred, degraded):
    # Naive is even worse here, so only the ceiling can fire.
    gold = _concat(
        _gold("energy_lgbm_24h", "run", 168, model_pred, 100.0, True), _naive(predicted=200.0)
    )

    d = assess_degradation(gold, END, 168, T)

    assert d.degraded is degraded
    if degraded:
        assert "MAPE" in d.reason


def test_ac2_evidence_floor_is_inclusive():
    gold = _concat(_gold("energy_lgbm_24h", "run", 72, 150.0, 100.0, True), _naive())

    assert assess_degradation(gold, END, 168, T).degraded is True
    assert assess_degradation(gold.iloc[1:], END, 168, T).degraded is False  # 71 hours


def test_ac2_naive_serving_is_never_degraded():
    gold = _naive().assign(is_served=True, predicted_mwh=500.0)

    d = assess_degradation(gold, END, 168, T)

    assert d == Degradation(False, "Seasonal naive serves: no model to judge.")


def test_ac2_legacy_rows_without_run_id_are_not_judged():
    gold = _concat(_gold("energy_lgbm_24h", "run_latest", 168, 500.0, 100.0, True), _naive())

    d = assess_degradation(gold, END, 168, T)

    assert d.degraded is False and "no run id" in d.reason


def test_ac2_hours_without_actuals_do_not_count():
    gold = _concat(_gold("energy_lgbm_24h", "run", 168, 150.0, None, True), _naive())

    d = assess_degradation(gold, END, 168, T)

    assert d.degraded is False and d.n_hours == 0


def test_ac2_empty_gold():
    assert assess_degradation(_naive().iloc[0:0], END, 168, T).degraded is False


# ───────────── AC3: streak and safety rules ─────────────

DEGRADED = Degradation(True, "Run r over 168h: skill vs naive -0.100 < +0.000.", "r", 168)


def _history(hours, end=END):
    return [(end - pd.Timedelta(hours=i), True) for i in range(1, hours + 1)]


def test_ac3_streak_threshold():
    _, should23, _ = decide_error_retrain(DEGRADED, _history(22), END, thresholds=T)
    n, should24, reason = decide_error_retrain(DEGRADED, _history(23), END, thresholds=T)

    assert should23 is False
    assert (n, should24) == (24, True)
    assert reason.startswith("Forecast degradation detected for 24 consecutive hours.")
    assert DEGRADED.reason in reason


def test_ac3_missing_hour_breaks_the_streak():
    history = [h for h in _history(30) if h[0] != END - pd.Timedelta(hours=5)]

    n, should, _ = decide_error_retrain(DEGRADED, history, END, thresholds=T)

    assert (n, should) == (5, False)


def test_ac3_cooldown():
    last = END - pd.Timedelta(hours=71)
    _, blocked, reason = decide_error_retrain(DEGRADED, _history(30), END, last, thresholds=T)
    _, allowed, _ = decide_error_retrain(
        DEGRADED, _history(30), END, END - pd.Timedelta(hours=73), thresholds=T
    )

    assert blocked is False and "Cooldown" in reason
    assert allowed is True


def test_ac3_pending_request_and_its_timeout():
    fresh = END - pd.Timedelta(hours=1)
    stale = END - pd.Timedelta(hours=7)

    assert (
        decide_error_retrain(DEGRADED, _history(30), END, pending_since=fresh, thresholds=T)[1]
        is False
    )
    assert (
        decide_error_retrain(DEGRADED, _history(30), END, pending_since=stale, thresholds=T)[1]
        is True
    )


def test_ac3_not_degraded_resets_the_streak():
    healthy = Degradation(False, "Run r over 168h: skill vs naive +0.200, not degraded.", "r", 168)

    n, should, reason = decide_error_retrain(healthy, _history(30), END, thresholds=T)

    assert (n, should, reason) == (0, False, healthy.reason)


# ───────────── AC4: data drift never triggers ─────────────

DRIFT_NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "03_drift_check.py"


def test_ac4_drift_check_cannot_start_a_retrain():
    tree = ast.parse(DRIFT_NOTEBOOK.read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}

    forbidden = {"trigger_retraining_job", "write_retrain_flag", "decide_retrain", "start_retrain"}
    assert not forbidden & (names | imported)


def test_ac4_drift_check_records_no_retrain():
    source = DRIFT_NOTEBOOK.read_text(encoding="utf-8")

    assert '"retrain_triggered": False,' in source


# ───────────── AC5: trigger failure never fails the monitor ─────────────


def test_ac5_job_start_failure_is_recorded_not_raised():
    flags = []

    def boom(job_id):
        raise RuntimeError("Jobs API down")

    triggered, reason, run_id = start_retrain("why", "123", boom, lambda r, i: flags.append(r))

    assert triggered is False and run_id is None
    assert reason == "Trigger failed: RuntimeError: Jobs API down"
    assert flags == []  # no flag: the next hourly check retries


def test_ac5_successful_start_writes_the_flag():
    flags = []

    triggered, reason, run_id = start_retrain(
        "why",
        "123",
        lambda job_id: (987, f"Started job {job_id}"),
        lambda r, i: flags.append((r, i)),
    )

    assert (triggered, run_id) == (True, 987)
    assert flags == [("why Started job 123", 987)]


def test_ac5_without_job_id_only_the_flag_is_written():
    flags = []

    triggered, _, run_id = start_retrain(
        "why", "", lambda job_id: pytest.fail("must not start"), lambda r, i: flags.append(r)
    )

    assert (triggered, run_id) == (True, None)
    assert flags == ["why"]


def test_d8_daily_naive_hours_do_not_hide_the_served_run():
    gold = _concat(
        _gold("energy_lgbm_24h", "run", 100, 150.0, 100.0, True, end=END - pd.Timedelta(hours=5)),
        _gold("daily_naive", "daily_naive_t-48h", 5, 100.0, 100.0, True),
        _naive(),
    )

    assert current_served_run(gold) == ("energy_lgbm_24h", "run")
    d = assess_degradation(gold, END, 168, T)
    assert d.run_id == "run" and d.n_hours == 100 and d.degraded is True


# ───────────── spec 019b: the trigger's skill is the served_model skill ─────────────


def _mixed(lgbm_pred, naive_served_pred):
    """100h served by LightGBM run "run" (seasonal naive 10% off on the same hours, unserved),
    then 40h served by seasonal naive and 28h by daily naive at `naive_served_pred`."""
    lgbm_end = END - pd.Timedelta(hours=68)
    return _concat(
        _gold("energy_lgbm_24h", "run", 100, lgbm_pred, 100.0, True, end=lgbm_end),
        _gold("seasonal_naive", "seasonal_naive_t-168h", 100, 110.0, 100.0, False, end=lgbm_end),
        _gold(
            "seasonal_naive",
            "seasonal_naive_t-168h",
            40,
            naive_served_pred,
            100.0,
            True,
            end=END - pd.Timedelta(hours=28),
        ),
        _gold("daily_naive", "daily_naive_t-48h", 28, naive_served_pred, 100.0, True),
    )


def _monitor_rows(gold):
    from src.error_monitoring import MODEL_STREAM, SERVED_STREAM, build_monitor_rows

    rows, _ = build_monitor_rows(gold, END, windows=(168,), alert_window=168)
    by_stream = {r["stream"]: r for r in rows}
    return by_stream[MODEL_STREAM], by_stream[SERVED_STREAM]


def test_trigger_fires_on_the_served_model_skill():
    gold = _mixed(
        lgbm_pred=115.0, naive_served_pred=100.0
    )  # model loses to naive; naive hours perfect

    model_row, served_row = _monitor_rows(gold)
    degradation = assess_degradation(gold, END, 168, T)

    assert (
        degradation.relative_skill
        == pytest.approx(model_row["relative_skill"])
        == pytest.approx(-0.5)
    )
    assert served_row["relative_skill"] is None  # the served stream carries no skill
    assert degradation.degraded is True
    _, should, _ = decide_error_retrain(degradation, _history(23), END, thresholds=T)
    assert should is True


def test_trigger_ignores_bad_naive_served_hours_when_the_model_beats_naive():
    gold = _mixed(lgbm_pred=102.0, naive_served_pred=160.0)  # model good; naive hours terrible

    model_row, _ = _monitor_rows(gold)
    degradation = assess_degradation(gold, END, 168, T)

    assert (
        degradation.relative_skill
        == pytest.approx(model_row["relative_skill"])
        == pytest.approx(0.8)
    )
    assert degradation.degraded is False
    _, should, _ = decide_error_retrain(degradation, _history(23), END, thresholds=T)
    assert should is False
