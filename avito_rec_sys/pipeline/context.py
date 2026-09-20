"""Stages of the pipeline.

A stage is everything candidate generation needs for one set of queries:
    val        validation hold-out queries against the benchmark corpus + their own positives
    ranker     the second hold-out, used to train the CatBoost ranker
    benchmark  the real benchmark queries against `benchmark_items`, logs from ALL of train
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from avito_rec_sys.data.corpus import cached, prepare_items, prepare_queries
from avito_rec_sys.data.loading import (
    assert_valid_item_ids, assert_valid_query_ids, load_benchmark_items, load_benchmark_queries, load_train,
)
from avito_rec_sys.data.normalize import add_normalized_query_column
from avito_rec_sys.data.split import HeldOut, SplitResult, make_splits
from avito_rec_sys.features.bm25f import FieldBM25Index
from avito_rec_sys.features.logs import QueryLogs
from avito_rec_sys.features.tables import click_centroids, location_centroids
from avito_rec_sys.paths import Paths

GEO_COLUMNS = ["item_id", "item_location_id", "item_latitude", "item_longitude"]


@dataclass
class Stage:
    name: str
    corpus: pl.DataFrame  # items with the derived text columns of `prepare_items`
    queries: pl.DataFrame  # with `search_query_norm` and `query_lemmas`
    qrels: dict[str, set[str]] | None  # query_id -> relevant item_ids (None for the benchmark)
    logs: QueryLogs  # click history the stage may use
    centroids: dict[int, tuple[float, float]]  # search location -> (lat, lon)


def make_benchmark_stage(paths: Paths, n_workers: int = 6) -> Stage:
    """The real task: benchmark queries against benchmark items, history = the whole train set."""
    bench_items = load_benchmark_items(paths.benchmark_items_parquet)
    bench_queries = load_benchmark_queries(paths.benchmark_queries_parquet)
    assert_valid_item_ids(bench_items)
    assert_valid_query_ids(bench_queries)
    corpus = cached(paths.work_dir / "benchmark_items_prepared.parquet", lambda: prepare_items(bench_items, n_workers))
    queries = prepare_queries(add_normalized_query_column(bench_queries), n_workers)
    history = add_normalized_query_column(load_train(paths.train_parquet))
    return Stage("benchmark", corpus, queries, None, QueryLogs(history), _centroids(history, bench_items))


# ---------------------------------------------------------------- training-time stages (val / ranker)


@dataclass
class TrainContext:
    cfg: dict
    paths: Paths
    splits: SplitResult
    bench_items: pl.DataFrame
    master_items: pl.DataFrame  # every item that can appear in a stage corpus, with the derived text columns

    @property
    def work(self) -> Path:
        return self.paths.work_dir


def _save_heldout(h: HeldOut, d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    h.queries.write_parquet(d / "queries.parquet")
    h.qrels.write_parquet(d / "qrels.parquet")
    h.positive_items.write_parquet(d / "positive_items.parquet")


def _load_heldout(d: Path) -> HeldOut:
    return HeldOut(*(pl.read_parquet(d / f"{n}.parquet") for n in ("queries", "qrels", "positive_items")))


def build_train_context(cfg: dict, paths: Paths, n_workers: int = 6) -> TrainContext:
    """Load the data, create (or load the cached) hold-out splits and the lemmatized master corpus."""
    work = paths.work_dir
    bench_items = load_benchmark_items(paths.benchmark_items_parquet)

    sdir = work / "splits"
    if (sdir / "train_part.parquet").exists():
        splits = SplitResult(
            pl.read_parquet(sdir / "train_part.parquet"), _load_heldout(sdir / "val"), _load_heldout(sdir / "ranker"), 0
        )
    else:
        sp = cfg["split"]
        splits = make_splits(
            load_train(paths.train_parquet), sp["n_validation_queries"], sp["n_ranker_queries"], sp["seen_fraction"], cfg["seed"]
        )
        sdir.mkdir(parents=True, exist_ok=True)
        splits.train_part.write_parquet(sdir / "train_part.parquet")
        _save_heldout(splits.val, sdir / "val")
        _save_heldout(splits.ranker, sdir / "ranker")

    def build_master() -> pl.DataFrame:
        cols = bench_items.columns
        parts = [bench_items, splits.val.positive_items.select(cols), splits.ranker.positive_items.select(cols)]
        return prepare_items(pl.concat(parts).unique(subset=["item_id"], keep="first", maintain_order=True), n_workers)

    master = cached(work / "master_items.parquet", build_master)
    return TrainContext(cfg, paths, splits, bench_items, master)


def make_stage(ctx: TrainContext, name: str, n_workers: int = 6) -> Stage:
    """Hold-out stage. The corpus is the benchmark corpus plus the hold-out's own positives (which
    are not in `benchmark_items`), so retrieval is as hard as on the benchmark. History and location
    centroids never include held-out pairs."""
    if name not in ("val", "ranker"):
        raise ValueError(name)
    sp = ctx.splits
    held = sp.val if name == "val" else sp.ranker
    qrels: dict[str, set[str]] = {}
    for qid, item in zip(held.qrels["query_id"].to_list(), held.qrels["item_id"].to_list()):
        qrels.setdefault(qid, set()).add(item)

    ids = set(ctx.bench_items["item_id"].to_list()) | set(held.positive_items["item_id"].to_list())
    corpus = ctx.master_items.filter(pl.col("item_id").is_in(ids))
    held_positives = set(sp.val.positive_items["item_id"].to_list()) | set(sp.ranker.positive_items["item_id"].to_list())
    return Stage(
        name, corpus, prepare_queries(held.queries, n_workers), qrels, QueryLogs(sp.train_part),
        _centroids(sp.train_part, ctx.bench_items, exclude_items=held_positives),
    )


def _centroids(history: pl.DataFrame, bench_items: pl.DataFrame, exclude_items: set[str] | None = None) -> dict:
    """Search location -> (lat, lon). Item-based centroids where the location holds corpus items,
    click-based ones (mean of the clicked items' coordinates) for the locations without any."""
    geo = pl.concat([history.select(GEO_COLUMNS), bench_items.select(GEO_COLUMNS)]).unique(subset=["item_id"])
    if exclude_items:
        geo = geo.filter(~pl.col("item_id").is_in(exclude_items))
    return {**click_centroids(history), **location_centroids(geo)}


def build_index(corpus: pl.DataFrame, k1: float = 1.2, b: float = 0.75) -> FieldBM25Index:
    from avito_rec_sys.data.corpus import tokens

    return FieldBM25Index(k1, b).fit(
        {
            "title": tokens(corpus["title_lemmas"]),
            "params": tokens(corpus["params_lemmas"]),
            "desc": tokens(corpus["desc_lemmas"]),
        }
    )


def stage_qrels_by_position(stage: Stage) -> dict[int, set[str]]:
    """qrels keyed by query POSITION in the stage (the key used by the feature tables)."""
    ids = stage.queries["query_id"].to_list()
    return {q: stage.qrels[ids[q]] for q in range(len(ids))}


def label_candidates(feats, item_ids: list[str], qrels_by_position: dict[int, set[str]]) -> np.ndarray:
    """1 where the candidate is a relevant item of its query, else 0."""
    pos = {v: i for i, v in enumerate(item_ids)}
    n = len(item_ids)
    relevant = np.array(
        [q * n + pos[i] for q, ids in qrels_by_position.items() for i in ids if i in pos], dtype=np.int64
    )
    keys = feats["q"].to_numpy().astype(np.int64) * n + feats["i"].to_numpy().astype(np.int64)
    return np.isin(keys, relevant).astype(np.int8)
