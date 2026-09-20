"""Parquet loaders for train / benchmark data.

Uses polars for speed on the ~500k-row train file (§8: "train.parquet читать
через polars/pyarrow, не pandas с object-колонками"). decimal128 price/lat/lon
columns are cast to float64 on load -- decimal dtypes are exact but every
downstream consumer (numpy, catboost, torch) wants float anyway, and carrying
decimal128 through the pipeline just adds silent-cast risk later.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

# Columns stored as decimal128 in the source parquet; cast once, here, so
# nothing downstream has to know about it.
_DECIMAL_COLUMNS = ["item_price", "item_latitude", "item_longitude"]

# item_id / query_id must stay strings everywhere (16 lowercase hex chars) --
# never let them round-trip through anything that could turn them into
# numbers (§8, §11 risk 7).
_ID_COLUMNS = ["item_id", "query_id"]


def _cast_decimals(df: pl.DataFrame) -> pl.DataFrame:
    casts = [pl.col(c).cast(pl.Float64) for c in _DECIMAL_COLUMNS if c in df.columns]
    if casts:
        df = df.with_columns(casts)
    return df


def _enforce_id_strings(df: pl.DataFrame) -> pl.DataFrame:
    casts = [pl.col(c).cast(pl.Utf8) for c in _ID_COLUMNS if c in df.columns]
    if casts:
        df = df.with_columns(casts)
    return df


def _postprocess(df: pl.DataFrame) -> pl.DataFrame:
    return _enforce_id_strings(_cast_decimals(df))


def load_train(path: str | Path, columns: list[str] | None = None) -> pl.DataFrame:
    df = pl.read_parquet(path, columns=columns)
    return _postprocess(df)


def load_benchmark_queries(path: str | Path, columns: list[str] | None = None) -> pl.DataFrame:
    df = pl.read_parquet(path, columns=columns)
    return _postprocess(df)


def load_benchmark_items(path: str | Path, columns: list[str] | None = None) -> pl.DataFrame:
    df = pl.read_parquet(path, columns=columns)
    return _postprocess(df)


def assert_valid_item_ids(df: pl.DataFrame, column: str = "item_id") -> None:
    """Sanity check: item_id is exactly 16 lowercase hex chars.

    Cheap to run after every load; catches an upstream data issue (or an
    accidental numeric cast) immediately instead of silently degrading the
    submission metric later (task_description.md "Частые ошибки").
    """
    bad = df.filter(~pl.col(column).str.contains(r"^[0-9a-f]{16}$")).height
    if bad:
        raise ValueError(f"{bad} rows in column {column!r} are not 16 lowercase hex chars")


def assert_valid_query_ids(df: pl.DataFrame, column: str = "query_id") -> None:
    """Sanity check: query_id is exactly 16 characters, case-sensitive, any charset.

    Unlike item_id, query_id is NOT restricted to hex (e.g. "00WuFMaXSFZBxSzT"
    in task_description.md's example) -- only length and dtype matter.
    """
    lengths = df.select(pl.col(column).str.len_chars().alias("n")).to_series()
    bad = int((lengths != 16).sum())
    if bad:
        raise ValueError(f"{bad} rows in column {column!r} are not exactly 16 characters")
