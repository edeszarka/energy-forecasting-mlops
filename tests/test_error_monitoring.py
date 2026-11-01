"""Tests for live error monitoring (spec 017 AC1, AC2, AC4, AC5)."""

import pandas as pd
import pytest

from src.error_monitoring import (
    LOW_COVERAGE,
    MODEL_STREAM,
    NAIVE_STREAM,
    SERVED_MAPE_HIGH,
    SERVED_STREAM,
    SERVED_WORSE_THAN_NAIVE,
    SHADOW_BEATS_NAIVE,
    SHADOW_STREAM,
    PairedMetrics,
    Thresholds,
    WindowMetrics,
    build_monitor_rows,
    evaluate_alerts,
    latest_served_forecaster,
    latest_served_model,
    new_failing_alerts,
    paired_metrics,
    split_streams,
    window_metrics,
)

END = pd.Timestamp("2026-11-08 00:00", tz="UTC")


def _rows(hours, predicted, actual, end=END):
    ts = pd.date_range(end=end, periods=hours, freq="h")
    return pd.DataFrame({"timestamp": ts, "predicted_mwh": predicted, "actual_mwh": actual})


def _gold(model_name, hours, predicted, actual, is_served, is_backfilled=False, end=END):
    df = _rows(hours, predicted, actual, end)
    return df.assign(model_name=model_name, is_served=is_served, is_backfilled=is_backfilled)


# ───────────── AC1: metrics ─────────────


def test_ac1_window_metrics_exact_values():
    rows = pd.DataFrame(
        {
            "timestamp": pd.date_range(end=END, periods=4, freq="h"),
            "predicted_mwh": [110.0, 90.0, 220.0, 180.0],
            "actual_mwh": [100.0, 100.0, 200.0, 200.0],
        }
    )

    m = window_metrics(rows, END, 4)

    assert m.n == 4
    assert m.coverage == 1.0
    assert m.mape == pytest.approx(10.0)  # every row is 10% off
    assert m.mae == pytest.approx(15.0)
    assert m.bias == pytest.approx(0.0)


def test_ac1_rows_without_actual_or_outside_window_are_ignored():
    rows = _rows(6, 110.0, 100.0)
    rows.loc[5, "actual_mwh"] = None  # latest hour not yet observed
    rows.loc[0, "predicted_mwh"] = 999.0  # outside a 5h window ending at END

    m = window_metrics(rows, END, 5)

    assert m.n == 4
    assert m.coverage == pytest.approx(0.8)
    assert m.mape == pytest.approx(10.0)


def test_ac1_window_is_left_open_right_closed():
    rows = _rows(3, 110.0, 100.0)  # END-2h, END-1h, END

    assert window_metrics(rows, END, 2).n == 2
    assert window_metrics(rows, END - pd.Timedelta(hours=1), 2).n == 2


def test_ac1_zero_actual_is_excluded_from_mape_only():
    rows = pd.DataFrame(
        {
            "timestamp": pd.date_range(end=END, periods=2, freq="h"),
            "predicted_mwh": [110.0, 5.0],
            "actual_mwh": [100.0, 0.0],
        }
    )

    m = window_metrics(rows, END, 2)

    assert m.mape == pytest.approx(10.0)
    assert m.mae == pytest.approx(7.5)


def test_ac1_empty_window():
    m = window_metrics(_rows(3, 1.0, None), END, 24)

    assert m == WindowMetrics(n=0, coverage=0.0, mape=None, mae=None, bias=None)


def test_ac1_bias_sign_is_predicted_minus_actual():
    assert window_metrics(_rows(4, 120.0, 100.0), END, 4).bias == pytest.approx(20.0)


def test_ac1_naive_and_aware_timestamps_match():
    rows = _rows(2, 110.0, 100.0)
    rows["timestamp"] = rows["timestamp"].dt.tz_localize(None)

    assert window_metrics(rows, END, 2).n == 2


# ───────────── AC2: paired comparison ─────────────


def test_ac2_paired_metrics_use_only_common_hours():
    model = _rows(10, 105.0, 100.0)  # 5% error, END-9h … END
    naive = _rows(10, 110.0, 100.0, end=END - pd.Timedelta(hours=4))  # 10%, END-13h … END-4h
    naive.loc[9, "predicted_mwh"] = 500.0  # END-4h is shared: 400% error

    p = paired_metrics(model, naive, END, 24)

    assert p.model.n == p.naive.n == 6  # END-9h … END-4h
    assert p.model.mape == pytest.approx(5.0)
    assert p.naive.mape == pytest.approx((5 * 10.0 + 400.0) / 6)
    assert p.relative_skill == pytest.approx(1 - p.model.mape / p.naive.mape)


def test_ac2_no_common_hours_gives_no_skill():
    model = _rows(3, 105.0, 100.0)
    naive = _rows(3, 110.0, 100.0, end=END - pd.Timedelta(hours=10))

    p = paired_metrics(model, naive, END, 24)

    assert p.model.n == 0 and p.relative_skill is None


def test_ac2_negative_skill_when_model_loses():
    p = paired_metrics(_rows(24, 120.0, 100.0), _rows(24, 110.0, 100.0), END, 24)

    assert p.relative_skill == pytest.approx(-1.0)


# ───────────── streams ─────────────


def test_streams_split_served_naive_shadow_and_skip_backfill():
    gold = pd.concat(
        [
            _gold("seasonal_naive", 4, 110.0, 100.0, is_served=True),
            _gold("energy_lgbm_24h", 4, 105.0, 100.0, is_served=False),
            _gold(
                "seasonal_naive",
                4,
                1.0,
                100.0,
                None,
                is_backfilled=True,
                end=END - pd.Timedelta(hours=10),
            ),
        ],
        ignore_index=True,
    )

    streams = split_streams(gold)

    assert len(streams[SERVED_STREAM]) == 4
    assert (streams[SERVED_STREAM]["predicted_mwh"] == 110.0).all()
    assert len(streams[NAIVE_STREAM]) == 4  # backfilled naive rows excluded
    assert (streams[SHADOW_STREAM]["predicted_mwh"] == 105.0).all()


def test_streams_null_is_served_counts_as_served():
    gold = _gold("seasonal_naive", 3, 110.0, 100.0, is_served=None)

    assert len(split_streams(gold)[SERVED_STREAM]) == 3


def test_latest_served_model_follows_the_newest_target_hour():
    gold = pd.concat(
        [
            _gold("seasonal_naive", 3, 1.0, 1.0, True, end=END - pd.Timedelta(hours=3)),
            _gold("energy_lgbm_24h", 3, 1.0, 1.0, True),
        ],
        ignore_index=True,
    )

    assert latest_served_model(gold) == "energy_lgbm_24h"
    assert latest_served_model(gold.iloc[0:0]) is None


# ───────────── AC4: alerts ─────────────

T = Thresholds(
    mape_ceiling=25.0, min_skill=0.0, shadow_skill=0.05, shadow_min_rows=120, min_coverage=0.8
)


def _wm(mape, coverage=1.0, n=168):
    return WindowMetrics(n=n, coverage=coverage, mape=mape, mae=1.0, bias=0.0)


def _paired(skill, n=168):
    return PairedMetrics(model=_wm(5.0, n=n), naive=_wm(5.0, n=n), relative_skill=skill)


@pytest.mark.parametrize(("mape", "alert"), [(25.0, False), (25.01, True)])
def test_ac4_served_mape_ceiling(mape, alert):
    assert (SERVED_MAPE_HIGH in evaluate_alerts(_wm(mape), None, None, T)) is alert


@pytest.mark.parametrize(("skill", "alert"), [(0.0, False), (-0.001, True)])
def test_ac4_served_worse_than_naive(skill, alert):
    assert (SERVED_WORSE_THAN_NAIVE in evaluate_alerts(_wm(5.0), _paired(skill), None, T)) is alert


def test_ac4_no_naive_comparison_while_naive_serves():
    assert evaluate_alerts(_wm(5.0), None, None, T) == []


@pytest.mark.parametrize(
    ("skill", "n", "alert"), [(0.05, 120, True), (0.0499, 120, False), (0.2, 119, False)]
)
def test_ac4_shadow_beats_naive(skill, n, alert):
    assert (SHADOW_BEATS_NAIVE in evaluate_alerts(_wm(5.0), None, _paired(skill, n), T)) is alert


@pytest.mark.parametrize(("coverage", "alert"), [(0.8, False), (0.79, True)])
def test_ac4_low_coverage(coverage, alert):
    assert (LOW_COVERAGE in evaluate_alerts(_wm(5.0, coverage), None, None, T)) is alert


def test_ac4_low_coverage_suppresses_served_alerts_but_not_shadow():
    alerts = evaluate_alerts(_wm(80.0, coverage=0.5), _paired(-0.5), _paired(0.3), T)

    assert alerts == [LOW_COVERAGE, SHADOW_BEATS_NAIVE]


# ───────────── AC5: fail once per episode ─────────────


def test_ac5_alert_start_fails():
    assert new_failing_alerts([SERVED_MAPE_HIGH], []) == [SERVED_MAPE_HIGH]
    assert new_failing_alerts([LOW_COVERAGE], None) == [LOW_COVERAGE]


def test_ac5_persisting_alert_does_not_fail_again():
    assert new_failing_alerts([SERVED_MAPE_HIGH], [SERVED_MAPE_HIGH]) == []


def test_ac5_shadow_alert_never_fails():
    assert new_failing_alerts([SHADOW_BEATS_NAIVE], []) == []


def test_ac5_only_the_new_alert_fails():
    current = [SERVED_MAPE_HIGH, SERVED_WORSE_THAN_NAIVE]

    assert new_failing_alerts(current, [SERVED_MAPE_HIGH]) == [SERVED_WORSE_THAN_NAIVE]


# ───────────── monitor rows ─────────────


def test_monitor_rows_while_naive_serves_with_shadow():
    gold = pd.concat(
        [
            _gold("seasonal_naive", 200, 110.0, 100.0, is_served=True),
            _gold("energy_lgbm_24h", 200, 104.0, 100.0, is_served=False),
        ],
        ignore_index=True,
    )

    rows, alerts = build_monitor_rows(gold, END, windows=(24, 168), alert_window=168, thresholds=T)

    by_key = {(r["window_hours"], r["stream"]): r for r in rows}
    assert set(by_key) == {
        (w, s) for w in (24, 168) for s in (SERVED_STREAM, NAIVE_STREAM, SHADOW_STREAM)
    }
    served = by_key[(168, SERVED_STREAM)]
    assert served["model_name"] == "seasonal_naive"
    assert served["mape"] == pytest.approx(10.0)
    assert served["relative_skill"] is None  # naive is not compared with itself
    shadow = by_key[(168, SHADOW_STREAM)]
    assert shadow["mape"] == pytest.approx(4.0)
    assert shadow["naive_mape_paired"] == pytest.approx(10.0)
    assert shadow["relative_skill"] == pytest.approx(0.6)
    assert alerts == [SHADOW_BEATS_NAIVE]
    assert served["alerts"] == alerts
    assert by_key[(24, SERVED_STREAM)]["alerts"] == []


def test_monitor_rows_when_lgbm_serves_and_loses_to_naive():
    gold = pd.concat(
        [
            _gold("energy_lgbm_24h", 200, 130.0, 100.0, is_served=True),
            _gold("seasonal_naive", 200, 110.0, 100.0, is_served=False),
        ],
        ignore_index=True,
    )

    rows, alerts = build_monitor_rows(gold, END, windows=(168,), alert_window=168, thresholds=T)

    assert {r["stream"] for r in rows} == {SERVED_STREAM, MODEL_STREAM, NAIVE_STREAM}
    assert SERVED_MAPE_HIGH in alerts and SERVED_WORSE_THAN_NAIVE in alerts


def test_monitor_rows_on_empty_gold_raise_low_coverage():
    gold = _gold("seasonal_naive", 0, 1.0, 1.0, True)

    rows, alerts = build_monitor_rows(gold, END, windows=(168,), alert_window=168, thresholds=T)

    assert alerts == [LOW_COVERAGE]
    assert rows[0]["n"] == 0 and rows[0]["model_name"] is None


# ───────────── spec 019b: mixed served stream (model, seasonal naive, daily naive) ─────────────


def _mixed_gold():
    """Last 168h: 100h served by LightGBM (2% off), 40h by seasonal naive, 28h by daily naive.

    Seasonal naive (10% off) exists for the LightGBM hours too, as the unserved
    comparison row; the daily-naive hours have no seasonal-naive row (T - 168h missing).
    """
    lgbm_end = END - pd.Timedelta(hours=68)
    naive_end = END - pd.Timedelta(hours=28)
    return pd.concat(
        [
            _gold("energy_lgbm_24h", 100, 102.0, 100.0, True, end=lgbm_end),
            _gold("seasonal_naive", 100, 110.0, 100.0, False, end=lgbm_end),
            _gold("seasonal_naive", 40, 130.0, 100.0, True, end=naive_end),
            _gold("daily_naive", 28, 150.0, 100.0, True),
        ],
        ignore_index=True,
    )


def test_mixed_stream_model_metrics_come_only_from_lgbm_rows():
    rows, alerts = build_monitor_rows(
        _mixed_gold(), END, windows=(168,), alert_window=168, thresholds=T
    )

    by_stream = {r["stream"]: r for r in rows}
    model = by_stream[MODEL_STREAM]
    assert model["model_name"] == "energy_lgbm_24h"
    assert model["n"] == 100
    assert model["mape"] == pytest.approx(2.0)  # the naive-served hours are not in it
    assert model["naive_mape_paired"] == pytest.approx(10.0)
    assert model["relative_skill"] == pytest.approx(0.8)
    served = by_stream[SERVED_STREAM]
    assert served["n"] == 168  # what was delivered: all three forecasters
    assert served["relative_skill"] is None and served["naive_mape_paired"] is None
    assert served["model_name"] == "daily_naive"  # label: latest served forecaster
    assert SERVED_WORSE_THAN_NAIVE not in alerts


def test_daily_naive_hours_get_no_skill_and_raise_no_alert():
    gold = pd.concat(
        [
            _gold("daily_naive", 168, 300.0, 100.0, True),  # 200% off, no seasonal-naive pair
        ],
        ignore_index=True,
    )

    rows, alerts = build_monitor_rows(gold, END, windows=(168,), alert_window=168, thresholds=T)

    assert MODEL_STREAM not in {r["stream"] for r in rows}  # no model served, no model row
    assert all(r["relative_skill"] is None for r in rows)
    assert SERVED_WORSE_THAN_NAIVE not in alerts


def test_latest_served_model_skips_naive_fallbacks():
    gold = _mixed_gold()

    assert latest_served_model(gold) == "energy_lgbm_24h"  # newest hours were daily naive
    assert latest_served_forecaster(gold) == "daily_naive"
    naive_only = gold[gold["model_name"] != "energy_lgbm_24h"]
    assert latest_served_model(naive_only) is None
