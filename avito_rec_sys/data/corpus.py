"""Precomputed, cached text views of a corpus of items or a set of queries.

Everything the lexical stages need is derived once per item / query and kept
as plain Python lists / numpy arrays aligned to the row order of the source
frame, so downstream code indexes by integer position instead of joining.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from avito_rec_sys.data.lemmatize import lemmatize_batch
from avito_rec_sys.data.normalize import normalize_title
from avito_rec_sys.data.params_parser import extract_whitelisted


def prepare_items(items: pl.DataFrame, n_workers: int = 8) -> pl.DataFrame:
    """Add: params_filtered, title_norm, title_lemmas / params_lemmas / desc_lemmas (space-joined)."""
    params_filtered = [extract_whitelisted(t) for t in items["item_infm_params_text"].to_list()]
    title_norm = [normalize_title(t) for t in items["item_title_raw"].to_list()]
    title_lem = lemmatize_batch(items["item_title_raw"].fill_null("").to_list(), n_workers)
    params_lem = lemmatize_batch(params_filtered, n_workers)
    desc_lem = lemmatize_batch(items["item_description_raw"].fill_null("").to_list(), n_workers)
    return items.with_columns(
        pl.Series("params_filtered", params_filtered),
        pl.Series("title_norm", title_norm),
        pl.Series("title_lemmas", title_lem),
        pl.Series("params_lemmas", params_lem),
        pl.Series("desc_lemmas", desc_lem),
    )


def prepare_queries(queries: pl.DataFrame, n_workers: int = 8) -> pl.DataFrame:
    """Add: query_lemmas (space-joined)."""
    lem = lemmatize_batch(queries["search_query"].fill_null("").to_list(), n_workers)
    return queries.with_columns(pl.Series("query_lemmas", lem))


def cached(path: Path, build):
    """Parquet cache: load if present, else build() and write."""
    path = Path(path)
    if path.exists():
        return pl.read_parquet(path)
    df = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df


def tokens(column: pl.Series) -> list[list[str]]:
    return [s.split() if s else [] for s in column.to_list()]


def id_index(ids: pl.Series) -> dict[str, int]:
    return {v: i for i, v in enumerate(ids.to_list())}


def group_positions(values: np.ndarray) -> dict:
    """value -> np.ndarray of row positions (used for per-location / per-microcat pools)."""
    order = np.argsort(values, kind="stable")
    sorted_vals = values[order]
    uniq, start = np.unique(sorted_vals, return_index=True)
    ends = list(start[1:]) + [len(values)]
    return {u.item(): order[s:e] for u, s, e in zip(uniq, start, ends)}
