"""Tests for gold row assembly and the naive gold history (spec 017 AC3, AC6)."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from src.gold_rows import combine_gold_rows, gold_forecast_id, reconstruct_naive_gold


def _frame(model_name, n=3):
    ts = pd.date_range("2026-11-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "model_name": model_name, "predicted_mwh": 1.0})


def test_ac3_model_served_marks_only_model_rows():
    out = combine_gold_rows(_frame("seasonal_naive"), _frame("energy_lgbm_24h"), model_served=True)

    assert out.groupby("model_name")["is_served"].unique().to_dict() == {
        "energy_lgbm_24h": [True],
        "seasonal_naive": [False],
    }


def test_ac3_naive_served_with_shadow():
    out = combine_gold_rows(_frame("seasonal_naive"), _frame("energy_lgbm_24h"), model_served=False)

    assert out[out["model_name"] == "seasonal_naive"]["is_served"].all()
    assert not out[out["model_name"] == "energy_lgbm_24h"]["is_served"].any()


def test_ac3_missing_frames_are_skipped():
    naive_only = combine_gold_rows(_frame("seasonal_naive"), pd.DataFrame(), model_served=False)
    model_only = combine_gold_rows(pd.DataFrame(), _frame("energy_lgbm_24h"), model_served=True)

    assert len(naive_only) == 3 and naive_only["is_served"].all()
    assert len(model_only) == 3 and model_only["is_served"].all()
    assert combine_gold_rows(pd.DataFrame(), pd.DataFrame(), model_served=False).empty


def test_ac3_served_model_without_rows_is_an_error():
    with pytest.raises(ValueError, match="requires model rows"):
        combine_gold_rows(_frame("seasonal_naive"), pd.DataFrame(), model_served=True)


def test_forecast_id_matches_the_notebook_convention():
    ts = pd.Timestamp("2026-11-01 22:00", tz="UTC")

    expected = hashlib.md5(b"seasonal_naive_24_2026-11-01T22:00:00+00:00").hexdigest()
    assert gold_forecast_id("seasonal_naive", 24, ts) == expected


def _silver(start="2026-10-01", hours=24 * 21):
    ts = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "value_mwh": 4000.0 + np.arange(hours)})


def test_ac6_naive_gold_equals_silver_one_week_earlier_without_leakage():
    silver = _silver()
    value_by_ts = silver.set_index("timestamp")["value_mwh"]

    out = reconstruct_naive_gold(silver, "2026-10-10", "2026-10-15", 24, "backfill-017")

    assert len(out) == 5 * 24
    for _, row in out.iterrows():
        source = row["timestamp"] - pd.Timedelta(hours=168)
        assert row["predicted_mwh"] == value_by_ts[source]
        assert source < row["forecast_run_at"]
        assert row["forecast_run_at"] == row["timestamp"] - pd.Timedelta(hours=48)
    assert out["is_backfilled"].all()
    assert out["is_served"].isna().all()
    assert out["actual_mwh"].isna().all()  # filled by 04's actual backfill
    assert set(out["model_name"]) == {"seasonal_naive"}
    assert list(out["forecast_id"]) == [
        gold_forecast_id("seasonal_naive", 24, t) for t in out["timestamp"]
    ]


def test_ac6_targets_without_a_source_hour_are_skipped():
    silver = _silver()
    out = reconstruct_naive_gold(silver, "2026-10-01", "2026-10-09", 24, "backfill-017")

    # Silver starts 2026-10-01 00:00, so the first target with a source is 2026-10-08 00:00.
    assert out["timestamp"].min() == pd.Timestamp("2026-10-08", tz="UTC")
    assert len(out) == 24


def test_ac6_empty_range():
    assert reconstruct_naive_gold(_silver(), "2026-10-10", "2026-10-10", 24, "x").empty
