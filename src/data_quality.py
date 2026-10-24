"""Load data-quality checks that fail the pipeline (ADR-0004, spec 014 T5).

Pure functions over a pandas frame with a tz-aware UTC `timestamp` column and a
`value_mwh` column. Each check returns human-readable violations; an empty list
means the window passed. `apply_dq_mode` turns violations into a failure.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import DQ_MIN_CORR_24H, LOAD_MAX_MW, LOAD_MIN_MW

logger = logging.getLogger(__name__)

DQ_MODES: tuple[str, ...] = ("warn", "enforce")

# Fewer hour-aligned pairs than one day cannot show a daily cycle.
MIN_CYCLE_PAIRS: int = 24

_MAX_EXAMPLES: int = 5


class DataQualityError(ValueError):
    """Raised in `enforce` mode when a window has violations."""


def _examples(timestamps: pd.Series | pd.Index) -> str:
    items = [pd.Timestamp(t).isoformat() for t in list(timestamps)[:_MAX_EXAMPLES]]
    return ", ".join(items)


def check_hourly_rows(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    """No duplicate or off-hour timestamps; 24 hourly rows per complete UTC day in [start, end)."""
    ts = df.loc[(df["timestamp"] >= start) & (df["timestamp"] < end), "timestamp"]
    violations = []

    duplicated = ts[ts.duplicated()]
    if not duplicated.empty:
        violations.append(
            f"{duplicated.nunique()} duplicate timestamps (e.g. {_examples(duplicated.unique())})"
        )

    off_hour = ts[ts != ts.dt.floor("h")]
    if not off_hour.empty:
        violations.append(
            f"{len(off_hour)} timestamps not on the hour (e.g. {_examples(off_hour)})"
        )

    per_day = ts.drop_duplicates().dt.floor("D").value_counts()
    first_day = start.ceil("D")
    last_day = end.floor("D") - pd.Timedelta(days=1)
    if first_day <= last_day:
        days = pd.date_range(first_day, last_day, freq="D")
        short = per_day.reindex(days, fill_value=0)
        short = short[short != 24]
        if not short.empty:
            detail = ", ".join(f"{d.date()}: {n}" for d, n in list(short.items())[:_MAX_EXAMPLES])
            violations.append(f"{len(short)} complete days without 24 hourly rows ({detail})")
    return violations


def check_value_range(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_mw: float = LOAD_MIN_MW,
    max_mw: float = LOAD_MAX_MW,
) -> list[str]:
    """Non-null load values within [min_mw, max_mw]."""
    window = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
    values = window["value_mwh"]
    outside = window[values.notna() & ((values < min_mw) | (values > max_mw))]
    if outside.empty:
        return []
    return [
        f"{len(outside)} values outside [{min_mw:g}, {max_mw:g}] MW "
        f"(min {outside['value_mwh'].min():.1f}, max {outside['value_mwh'].max():.1f}; "
        f"e.g. {_examples(outside['timestamp'])})"
    ]


def lag_correlation(
    df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, lag_hours: int
) -> tuple[float, int]:
    """corr(value(t), value(t - lag)) for t in [start, end), on hour-aligned pairs only.

    The lagged value is looked up by timestamp anywhere in `df`, never by row
    position, so a gap cannot pair two values that are not `lag_hours` apart.
    """
    s = df.dropna(subset=["value_mwh"]).drop_duplicates("timestamp").set_index("timestamp")
    s = s["value_mwh"]
    current = s[(s.index >= start) & (s.index < end)]
    lagged = s.reindex(current.index - pd.Timedelta(hours=lag_hours))
    lagged.index = current.index
    pairs = pd.concat([current, lagged], axis=1, keys=["now", "lag"]).dropna()
    if len(pairs) < 2:
        return float("nan"), len(pairs)
    return float(pairs["now"].corr(pairs["lag"])), len(pairs)


def check_daily_cycle(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_corr_24h: float = DQ_MIN_CORR_24H,
) -> list[str]:
    """corr at lag 24h >= min_corr_24h and > corr at lag 12h.

    A load series with real hourly labels repeats every 24h; the pre-014
    parser's 4x-stretched series repeats every 96h, which drives lag-24
    correlation to around zero or below.
    """
    c24, n24 = lag_correlation(df, start, end, 24)
    c12, _ = lag_correlation(df, start, end, 12)
    if n24 < MIN_CYCLE_PAIRS:
        return [f"daily cycle: only {n24} hour-aligned 24h pairs (need {MIN_CYCLE_PAIRS})"]
    if pd.isna(c24) or c24 < min_corr_24h:
        return [f"daily cycle: corr24 {c24:.2f} < {min_corr_24h:.2f} ({n24} pairs)"]
    if not pd.isna(c12) and c24 <= c12:
        return [f"daily cycle: corr24 {c24:.2f} <= corr12 {c12:.2f} ({n24} pairs)"]
    return []


def check_load_quality(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_mw: float = LOAD_MIN_MW,
    max_mw: float = LOAD_MAX_MW,
    min_corr_24h: float = DQ_MIN_CORR_24H,
) -> list[str]:
    """All load checks for the window [start, end); empty list means it passed."""
    return (
        check_hourly_rows(df, start, end)
        + check_value_range(df, start, end, min_mw, max_mw)
        + check_daily_cycle(df, start, end, min_corr_24h)
    )


def month_windows(
    start: pd.Timestamp, end: pd.Timestamp
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Splits [start, end) at calendar-month boundaries (UTC)."""
    bounds = [start]
    month = (start + pd.offsets.MonthBegin(1)).normalize()
    while month < end:
        bounds.append(month)
        month = month + pd.offsets.MonthBegin(1)
    bounds.append(end)
    return list(zip(bounds[:-1], bounds[1:], strict=True))


def check_load_quality_by_month(
    df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> list[str]:
    """Runs check_load_quality per calendar month, so one bad month cannot hide
    inside a multi-year correlation."""
    violations = []
    for w_start, w_end in month_windows(start, end):
        violations += [f"{w_start:%Y-%m}: {v}" for v in check_load_quality(df, w_start, w_end)]
    return violations


def apply_dq_mode(violations: list[str], mode: str) -> None:
    """`warn` logs the violations; `enforce` raises DataQualityError if there are any."""
    if mode not in DQ_MODES:
        raise ValueError(f"Unknown dq_mode {mode!r}; expected one of {DQ_MODES}.")
    if not violations:
        return
    message = f"{len(violations)} data-quality violations: " + "; ".join(violations)
    if mode == "enforce":
        raise DataQualityError(message)
    logger.warning(message)
