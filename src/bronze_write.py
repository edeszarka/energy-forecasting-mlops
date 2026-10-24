"""Bronze write rules: the hourly MERGE and the backfill replace (spec 014 T2, T3).

The hourly path MERGEs and never lets a NULL load value overwrite a stored one,
which protects against partial re-fetches. The backfill path replaces a whole
range with `replaceWhere`, NULLs included (ADR-0005).
"""

from __future__ import annotations

import pandas as pd

LOAD_MERGE_CONDITION: str = "source.value_mwh IS NOT NULL"
LOAD_MERGE_COLUMNS: tuple[str, ...] = ("value_mwh", "source", "fetched_at", "run_id", "is_gap")

# Imputed weather never overwrites a measured value; all weather columns move together.
TEMPERATURE_MERGE_CONDITION: str = "source.is_weather_imputed = false"
TEMPERATURE_MERGE_COLUMNS: tuple[str, ...] = (
    "temperature_c",
    "humidity_pct",
    "cloud_cover_pct",
    "is_weather_imputed",
    "source",
    "fetched_at",
    "run_id",
)

BRONZE_LOAD_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "country",
    "value_mwh",
    "source",
    "fetched_at",
    "run_id",
    "is_gap",
)
BRONZE_TEMPERATURE_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "temperature_c",
    "humidity_pct",
    "cloud_cover_pct",
    "is_weather_imputed",
    "source",
    "fetched_at",
    "run_id",
)


def merge_set(columns: tuple[str, ...]) -> dict[str, str]:
    """`whenMatchedUpdate(set=...)` mapping: every column taken from the source."""
    return {col: f"source.{col}" for col in columns}


def should_skip_ingest(load_files: list[str], temp_files: list[str]) -> bool:
    """01_ingest has nothing to do only when neither source has pending files.

    Weather-only batches are ingested too (spec 014 T2).
    """
    return not load_files and not temp_files


def replace_where_predicate(
    start: pd.Timestamp, end: pd.Timestamp, country: str | None = None
) -> str:
    """Delta `replaceWhere` predicate for timestamps in [start, end)."""
    predicate = (
        f"timestamp >= '{start.strftime('%Y-%m-%d %H:%M:%S')}' "
        f"AND timestamp < '{end.strftime('%Y-%m-%d %H:%M:%S')}'"
    )
    if country is not None:
        predicate += f" AND country = '{country}'"
    return predicate


def _in_range(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
    # Chunks may overlap: keep the most recently fetched row per hour.
    return df.sort_values(["timestamp", "fetched_at"]).drop_duplicates("timestamp", keep="last")


def prepare_backfill_load(
    df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, country: str, run_id: str
) -> pd.DataFrame:
    """bronze_load rows for [start, end) from backfill files; NULL values are kept."""
    df = _in_range(df, start, end)
    df["country"] = country
    df["run_id"] = run_id
    df["is_gap"] = df["value_mwh"].isna()
    return df[list(BRONZE_LOAD_COLUMNS)].reset_index(drop=True)


def prepare_backfill_temperature(
    df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, run_id: str
) -> pd.DataFrame:
    """bronze_temperature rows for [start, end) from backfill files."""
    df = _in_range(df, start, end)
    # A missing flag means not imputed.
    df["is_weather_imputed"] = df["is_weather_imputed"].eq(True)
    df["run_id"] = run_id
    return df[list(BRONZE_TEMPERATURE_COLUMNS)].reset_index(drop=True)


def monthly_row_counts(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Rows and non-null values per UTC month, for before/after comparisons."""
    if df.empty:
        return pd.DataFrame(columns=["month", "rows", "non_null"])
    ts = pd.to_datetime(df["timestamp"], utc=True)
    grouped = df.assign(month=ts.dt.strftime("%Y-%m")).groupby("month")[value_column]
    return grouped.agg(rows="size", non_null="count").reset_index()
