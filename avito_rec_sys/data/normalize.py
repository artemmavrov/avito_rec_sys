"""Text normalization primitives.

Two distinct normalized forms are used downstream, and they must NOT be
confused (they are different keys for different purposes):

  - `normalize_query` (word-sorted): the query-paraphrase key. Used for
    pair dedup, the validation split key and the hard-negative denoise key. Sorting is required because "установка сэндвич
    панели" and "сэндвич панели установка" must collapse to the same key.
  - `normalize_title` (order-preserving): the duplicate-CLUSTER feature key
    (features/duplicates.py). Titles aren't paraphrased the way queries are,
    so a title's dup-cluster identity shouldn't collapse two titles that
    share a word set but were never actually written in scrambled order.
"""

from __future__ import annotations

import re

import polars as pl

# Keep only digits, latin/cyrillic letters (incl. ё) and whitespace as word
# separators. Cyrillic range: а-я (U+0430-U+044F) + ё (U+0451).
_ALLOWED_CHARS_RE = re.compile(r"[^0-9a-zа-яё\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_title(text: str | None) -> str:
    """Lowercase -> drop everything except [0-9a-zа-яё] -> collapse whitespace.
    No word-sorting -- see module docstring."""
    if text is None:
        return ""
    lowered = text.lower()
    stripped = _ALLOWED_CHARS_RE.sub(" ", lowered)
    return _WHITESPACE_RE.sub(" ", stripped).strip()


def normalize_query(text: str | None) -> str:
    """Python-level normalization for single strings (tests, ad-hoc use)."""
    cleaned = normalize_title(text)
    if not cleaned:
        return ""
    return " ".join(sorted(cleaned.split(" ")))


def normalize_title_expr(column: str) -> pl.Expr:
    """Vectorized `normalize_title` (no word-sort needed, so purely polars-native)."""
    return (
        pl.col(column)
        .str.to_lowercase()
        .str.replace_all(r"[^0-9a-zа-яё\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .fill_null("")
    )


def normalize_query_expr(column: str = "search_query") -> pl.Expr:
    """Same normalization as `normalize_query`, vectorized as a polars expression.

    polars has no built-in "sort words within a string" primitive, so the
    heavy lexical cleanup (lowercase, char filter, whitespace collapse) is
    vectorized here, and only the final word-sort is done via `map_elements`
    -- a small cost paid once per unique query, not once per (query, item) row.
    """
    return normalize_title_expr(column).map_elements(_sort_words, return_dtype=pl.Utf8)


def _sort_words(cleaned: str) -> str:
    if not cleaned:
        return ""
    return " ".join(sorted(cleaned.split(" ")))


def add_normalized_query_column(
    df: pl.DataFrame, source_col: str = "search_query", target_col: str = "search_query_norm"
) -> pl.DataFrame:
    return df.with_columns(normalize_query_expr(source_col).alias(target_col))


def add_normalized_title_column(
    df: pl.DataFrame, source_col: str = "item_title_raw", target_col: str = "item_title_norm"
) -> pl.DataFrame:
    return df.with_columns(normalize_title_expr(source_col).alias(target_col))
