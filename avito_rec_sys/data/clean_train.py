"""Three-step cleaning of the training pairs.

    0. normalize query
    1. dedup (normalized_query, item_id) -- exact repeated pairs
    2. dedup (normalized_query, item_tower_text) within each query group --
       different item_id but identical encoder input is not a distinct
       training example
    3. cap pairs per normalized query at `cap` (default 16) -- flattens the
       pathological tail (one query had 6,540 pairs) without dropping any
       query entirely

Each step returns row counts so the reduction can be reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from avito_rec_sys.data.normalize import add_normalized_query_column
from avito_rec_sys.data.params_parser import build_item_tower_text


@dataclass
class CleaningStats:
    n_start: int
    n_after_pair_dedup: int
    n_after_tower_dedup: int
    n_after_cap: int
    n_queries: int

    def as_dict(self) -> dict:
        return {
            "0_start": self.n_start,
            "1_pair_dedup": self.n_after_pair_dedup,
            "2_tower_dedup": self.n_after_tower_dedup,
            "3_cap": self.n_after_cap,
            "n_queries_preserved": self.n_queries,
        }


def add_item_tower_text_column(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.struct(["item_title_raw", "item_infm_params_text"])
        .map_elements(
            lambda s: build_item_tower_text(s["item_title_raw"], s["item_infm_params_text"]),
            return_dtype=pl.Utf8,
        )
        .alias("item_tower_text")
    )


def clean_train(df: pl.DataFrame, cap_per_query: int = 16, seed: int = 42) -> tuple[pl.DataFrame, CleaningStats]:
    n_start = df.height

    df = add_normalized_query_column(df)
    df = add_item_tower_text_column(df)

    # Step 1: dedup (query, item_id) -- keep first occurrence.
    df = df.unique(subset=["search_query_norm", "item_id"], keep="first", maintain_order=True)
    n_after_pair_dedup = df.height

    # Step 2: dedup (query, item_tower_text) -- different item_id, identical
    # encoder input. Key is the full tower text, not the title alone.
    df = df.unique(subset=["search_query_norm", "item_tower_text"], keep="first", maintain_order=True)
    n_after_tower_dedup = df.height

    # Step 3: cap pairs per query. Random sample (not "first N") so the cap
    # doesn't systematically favor whatever order rows happened to appear in
    # the source parquet.
    rng = np.random.RandomState(seed)
    df = df.with_columns(pl.Series("_rand", rng.random(df.height)))
    df = (
        df.with_columns(pl.col("_rand").rank(method="ordinal").over("search_query_norm").alias("_rank"))
        .filter(pl.col("_rank") <= cap_per_query)
        .drop(["_rand", "_rank"])
    )
    n_after_cap = df.height

    n_queries = df["search_query_norm"].n_unique()

    stats = CleaningStats(
        n_start=n_start,
        n_after_pair_dedup=n_after_pair_dedup,
        n_after_tower_dedup=n_after_tower_dedup,
        n_after_cap=n_after_cap,
        n_queries=n_queries,
    )
    return df, stats
