"""Data-quality checks that fail the pipeline (spec 014 T5, AC6)."""

import logging

import numpy as np
import pandas as pd
import pytest

from src.data_quality import (
    DataQualityError,
    apply_dq_mode,
    check_daily_cycle,
    check_hourly_rows,
    check_load_quality,
    check_load_quality_by_month,
    check_value_range,
    lag_correlation,
    month_windows,
)

START = pd.Timestamp("2026-03-02T00:00Z")


def daily_cycle(hours: np.ndarray) -> np.ndarray:
    """MW with a 24h cycle (roughly the observed 3.5-7 GW range) and some noise."""
    rng = np.random.default_rng(0)
    return 5000 + 1200 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 60, len(hours))


def clean_load(days: int = 14, start: pd.Timestamp = START) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=days * 24, freq="h")
    return pd.DataFrame(
        {"timestamp": timestamps, "value_mwh": daily_cycle(np.arange(len(timestamps)))}
    )


def stretched_load(days: int = 14) -> pd.DataFrame:
    """The pre-014 parser bug: consecutive quarter-hour values labelled as hours (4x)."""
    quarters = daily_cycle(np.arange(days * 24) / 4)  # quarter k is at hour k/4
    timestamps = pd.date_range(START, periods=days * 24, freq="h")
    return pd.DataFrame({"timestamp": timestamps, "value_mwh": quarters})


def window(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    return df["timestamp"].min(), df["timestamp"].max() + pd.Timedelta(hours=1)


def test_clean_series_passes():
    df = clean_load()
    assert check_load_quality(df, *window(df)) == []


def test_stretched_series_fails_daily_cycle():
    df = stretched_load()

    violations = check_load_quality(df, *window(df))

    assert len(violations) == 1
    assert violations[0].startswith("daily cycle: corr24")


def test_missing_hour_fails():
    df = clean_load().drop(index=30)

    violations = check_hourly_rows(df, *window(df))

    assert violations == ["1 complete days without 24 hourly rows (2026-03-03: 23)"]


def test_duplicate_and_off_hour_timestamps_fail():
    df = clean_load(days=2)
    df = pd.concat([df, df.iloc[[5]]])
    df.loc[len(df)] = [START + pd.Timedelta(minutes=15), 5000.0]

    violations = check_hourly_rows(df, START, START + pd.Timedelta(days=2))

    assert any("duplicate timestamps" in v for v in violations)
    assert any("not on the hour" in v for v in violations)


def test_partial_days_at_window_edges_are_not_counted():
    df = clean_load(days=3).iloc[6:-6]
    assert check_hourly_rows(df, *window(df)) == []


def test_out_of_range_fails_and_nulls_are_ignored():
    df = clean_load()
    df.loc[10, "value_mwh"] = 12000.0
    df.loc[11, "value_mwh"] = np.nan

    violations = check_value_range(df, *window(df), min_mw=2500, max_mw=9000)

    assert len(violations) == 1
    assert violations[0].startswith("1 values outside [2500, 9000] MW")


def test_daily_cycle_uses_hour_aligned_pairs_only():
    df = clean_load()
    # Drop a block of hours: positional shifting would pair values 24 rows apart
    # that are no longer 24 hours apart.
    gapped = df.drop(index=range(100, 107))

    c24, n24 = lag_correlation(gapped, *window(df), 24)

    assert c24 > 0.9
    assert n24 == len(df) - 24 - 2 * 7


def test_daily_cycle_needs_enough_pairs():
    df = clean_load(days=1)
    assert check_daily_cycle(df, *window(df)) == [
        "daily cycle: only 0 hour-aligned 24h pairs (need 24)"
    ]


def test_corr24_must_exceed_corr12():
    df = clean_load()
    # A 12h cycle repeats at both 12h and 24h: corr24 is high but not above corr12.
    df["value_mwh"] = 5000 + 1000 * np.sin(2 * np.pi * np.arange(len(df)) / 12)

    violations = check_daily_cycle(df, *window(df), min_corr_24h=0.5)

    assert violations[0].startswith("daily cycle: corr24 1.00 <= corr12 1.00")


def test_month_windows_split_at_calendar_months():
    windows = month_windows(pd.Timestamp("2024-01-15T06:00Z"), pd.Timestamp("2024-03-02T00:00Z"))

    assert windows == [
        (pd.Timestamp("2024-01-15T06:00Z"), pd.Timestamp("2024-02-01T00:00Z")),
        (pd.Timestamp("2024-02-01T00:00Z"), pd.Timestamp("2024-03-01T00:00Z")),
        (pd.Timestamp("2024-03-01T00:00Z"), pd.Timestamp("2024-03-02T00:00Z")),
    ]


def test_by_month_reports_only_the_bad_month():
    good = clean_load(days=31, start=pd.Timestamp("2024-01-01T00:00Z"))
    bad = stretched_load(days=29)
    bad["timestamp"] = pd.date_range("2024-02-01T00:00Z", periods=len(bad), freq="h")
    df = pd.concat([good, bad], ignore_index=True)

    violations = check_load_quality_by_month(
        df, pd.Timestamp("2024-01-01T00:00Z"), pd.Timestamp("2024-03-01T00:00Z")
    )

    assert len(violations) == 1
    assert violations[0].startswith("2024-02: daily cycle")


def test_enforce_raises_on_violations():
    with pytest.raises(DataQualityError, match="1 data-quality violations: boom"):
        apply_dq_mode(["boom"], "enforce")


def test_enforce_passes_without_violations():
    apply_dq_mode([], "enforce")


def test_warn_logs_and_continues(caplog):
    with caplog.at_level(logging.WARNING, logger="src.data_quality"):
        apply_dq_mode(["boom"], "warn")
    assert "boom" in caplog.text


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown dq_mode"):
        apply_dq_mode([], "off")
