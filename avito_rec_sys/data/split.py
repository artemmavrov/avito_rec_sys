"""Leak-free query hold-outs, built to mirror how the benchmark looks.

Facts measured on the real data that drive the design:

- Split key is the NORMALIZED query text (word-sorted), never the raw string:
  3.3% of normalized queries have several raw spellings and cover 25.4% of
  pairs, so splitting on raw text leaks positives into train.
- A query is one full `search_*` tuple (text + location + delivery flag +
  filters + category). The benchmark has 2,452 tuples over 2,448 distinct
  normalized texts, i.e. ~1 tuple per text (train averages 4.9), so each
  held-out text contributes ONE random tuple -- otherwise a few very
  popular texts would dominate the metric.
- 38.5% of benchmark texts also occur in train, 61.5% are new. A hold-out
  made only of unseen texts would misjudge every signal that exploits the
  logs (q->item, title bridge, p(microcat|query)). So a `seen_fraction` of
  hold-out queries keep their text in the training part (only that tuple
  and its positives are removed) and the rest are removed with all their
  tuples. Metrics are reported per stratum.
- Positives of held-out queries are removed from the training part entirely:
  only ~9.6% of benchmark corpus items ever appear in train.
- Two disjoint hold-outs are made: `val` (measurement) and
  `ranker` (train CatBoost). The encoders are trained on neither, otherwise
  the ranker would learn to over-trust neural scores on pairs the encoders
  memorised.
- The distractor pool is the real benchmark_items corpus (no benchmark
  relevance labels exist or are used).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from avito_rec_sys.data.normalize import add_normalized_query_column

QUERY_COLUMNS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]

ITEM_COLUMNS = [
    "item_id",
    "item_title_raw",
    "item_description_raw",
    "item_infm_params_text",
    "item_category_id",
    "item_microcat_id",
    "item_price",
    "item_rating",
    "item_rating_reviews_count",
    "item_location_id",
    "item_latitude",
    "item_longitude",
    "item_is_phone_hidden",
    "item_is_message_forbidden",
]

SEEN, NEW = "seen", "new"


@dataclass
class HeldOut:
    queries: pl.DataFrame  # query_id, search_query_norm, QUERY_COLUMNS, stratum
    qrels: pl.DataFrame  # (query_id, item_id)
    positive_items: pl.DataFrame  # ITEM_COLUMNS of every positive, deduped by item_id


@dataclass
class SplitResult:
    train_part: pl.DataFrame  # raw train rows + search_query_norm; used to train encoders / logs
    val: HeldOut
    ranker: HeldOut
    n_rows_dropped_for_item_overlap: int


def _hold_out(
    df: pl.DataFrame, n_queries: int, seen_fraction: float, rng: np.random.RandomState, id_prefix: str
) -> tuple[pl.DataFrame, HeldOut, int]:
    tuples = (
        df.group_by(["search_query_norm", *QUERY_COLUMNS])
        .agg(pl.col("item_id").unique(maintain_order=True).alias("item_ids"))
        .sort(["search_query_norm", *QUERY_COLUMNS])  # deterministic before sampling
    )
    n_tuples_per_text = tuples.group_by("search_query_norm").len().rename({"len": "n_tuples"})
    texts = n_tuples_per_text.sort("search_query_norm")
    text_arr = np.array(texts["search_query_norm"].to_list())
    multi_arr = np.array(texts.filter(pl.col("n_tuples") >= 2)["search_query_norm"].to_list())

    n_seen = min(int(round(n_queries * seen_fraction)), len(multi_arr))
    seen_texts = set(rng.choice(multi_arr, size=n_seen, replace=False).tolist()) if n_seen else set()
    rest_texts = np.array([t for t in text_arr if t not in seen_texts])
    n_new = min(n_queries - n_seen, len(rest_texts))
    new_texts = set(rng.choice(rest_texts, size=n_new, replace=False).tolist())

    chosen = tuples.filter(pl.col("search_query_norm").is_in(seen_texts | new_texts))
    chosen = chosen.with_columns(pl.Series("_rand", rng.random(chosen.height)))
    chosen = (
        chosen.with_columns(pl.col("_rand").rank(method="ordinal").over("search_query_norm").alias("_rank"))
        .filter(pl.col("_rank") == 1)
        .drop(["_rand", "_rank"])
        .with_columns(
            pl.when(pl.col("search_query_norm").is_in(seen_texts)).then(pl.lit(SEEN)).otherwise(pl.lit(NEW)).alias("stratum")
        )
        .with_row_index("_idx")
        .with_columns(pl.format(id_prefix + "{}", pl.col("_idx").cast(pl.Utf8).str.zfill(15 - len(id_prefix) + 1)).alias("query_id"))
    )

    queries = chosen.select(["query_id", "search_query_norm", *QUERY_COLUMNS, "stratum"])
    qrels = chosen.select(["query_id", "item_ids"]).explode("item_ids").rename({"item_ids": "item_id"})
    positive_ids = set(qrels["item_id"].to_list())
    positive_items = (
        df.filter(pl.col("item_id").is_in(positive_ids)).select(ITEM_COLUMNS).unique(subset=["item_id"], maintain_order=True)
    )

    # new texts leave the training part with every tuple; seen texts lose only
    # the chosen tuple (removed below through its positives) but keep the rest.
    remaining = df.filter(~pl.col("search_query_norm").is_in(new_texts))
    chosen_keys = chosen.select(["search_query_norm", *QUERY_COLUMNS]).with_columns(pl.lit(True).alias("_held"))
    remaining = remaining.join(chosen_keys, on=["search_query_norm", *QUERY_COLUMNS], how="left")
    remaining = remaining.filter(pl.col("_held").is_null()).drop("_held")
    before = remaining.height
    remaining = remaining.filter(~pl.col("item_id").is_in(positive_ids))
    return remaining, HeldOut(queries, qrels, positive_items), before - remaining.height


def make_splits(
    train_raw: pl.DataFrame,
    n_validation_queries: int = 2500,
    n_ranker_queries: int = 8000,
    seen_fraction: float = 0.385,
    seed: int = 42,
) -> SplitResult:
    df = add_normalized_query_column(train_raw)
    rng = np.random.RandomState(seed)
    df, val, dropped_v = _hold_out(df, n_validation_queries, seen_fraction, rng, "v")
    df, ranker, dropped_r = _hold_out(df, n_ranker_queries, seen_fraction, rng, "r")
    return SplitResult(df, val, ranker, dropped_v + dropped_r)


def build_corpus(positive_items: pl.DataFrame, benchmark_items: pl.DataFrame) -> pl.DataFrame:
    """Positives + the benchmark_items pool as distractors, deduped by item_id."""
    return pl.concat([positive_items.select(ITEM_COLUMNS), benchmark_items.select(ITEM_COLUMNS)]).unique(
        subset=["item_id"], keep="first", maintain_order=True
    )


def cold_start_ratio(corpus: pl.DataFrame, train_part: pl.DataFrame) -> float:
    """Share of corpus items that never appear in the training part."""
    train_items = set(train_part["item_id"].unique().to_list())
    return corpus.filter(~pl.col("item_id").is_in(train_items)).height / corpus.height
