"""Tests for gold rows (spec 019b AC4) and the naive gold history (spec 017 AC6)."""

import hashlib

import numpy as np
import pandas as pd

from src.gold_rows import GOLD_COLUMNS, gold_forecast_id, gold_row, reconstruct_naive_gold

TARGET = pd.Timestamp("2026-11-03 10:00", tz="UTC")
CREATED = pd.Timestamp("2026-11-02 12:07", tz="UTC")


def _row(**overrides):
    args = {
        "model_name": "seasonal_naive",
        "model_version": "seasonal_naive_t-168h",
        "target": TARGET,
        "issued_at": TARGET - pd.Timedelta(hours=24),
        "is_catch_up": True,
        "predicted_mwh": 4321,
        "is_served": True,
        "forecast_run_at": pd.Timestamp("2026-11-02 12:00", tz="UTC"),
        "pipeline_run_id": "run-1",
        "created_at": CREATED,
        "fallback_reason": "no_production_run",
    }
    return gold_row(**{**args, **overrides})


def test_ac4_gold_row_has_every_gold_column_and_no_written_at():
    row = _row()

    assert list(row) == GOLD_COLUMNS
    assert "written_at" not in row


def test_ac4_gold_row_values():
    row = _row()

    assert row["forecast_id"] == gold_forecast_id("seasonal_naive", 24, TARGET)
    assert row["timestamp"] == TARGET
    assert row["issued_at"] == TARGET - pd.Timedelta(hours=24)
    assert row["is_catch_up"] is True
    assert row["created_at"] == CREATED  # the actual write time (D5)
    assert row["predicted_mwh"] == 4321.0 and isinstance(row["predicted_mwh"], float)
    assert row["fallback_reason"] == "no_production_run"
    assert row["actual_mwh"] is None and row["is_backfilled"] is False


def test_ac4_model_row_has_no_fallback_reason_by_default():
    row = _row(model_name="energy_lgbm_24h", model_version="run-9", fallback_reason=None)

    assert row["fallback_reason"] is None
    assert row["model_version"] == "run-9"


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
    assert out[["issued_at", "is_catch_up", "fallback_reason"]].isna().all().all()  # legacy (D7)
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
