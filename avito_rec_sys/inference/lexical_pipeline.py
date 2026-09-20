"""Glue for the CPU stages: cached splits / prepared items, and per-stage
(val | ranker | test) candidate generation + feature building.

A "stage" is a (corpus, queries, logs, centroids) bundle:
  val     validation queries against val positives + benchmark_items pool
  ranker  CatBoost training queries against ranker positives + pool
  test    the real benchmark queries against benchmark_items, logs from ALL of train
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from avito_rec_sys.data.corpus import cached, prepare_items, prepare_queries
from avito_rec_sys.data.loading import load_benchmark_items, load_benchmark_queries, load_train
from avito_rec_sys.data.normalize import add_normalized_query_column
from avito_rec_sys.data.split import HeldOut, SplitResult, make_splits
from avito_rec_sys.features.assemble import build_features
from avito_rec_sys.features.bm25f import FieldBM25Index
from avito_rec_sys.features.logs import QueryLogs, log_candidate_positions
from avito_rec_sys.features.tables import ItemTable, QueryTable, click_centroids, location_centroids
from avito_rec_sys.retrieval.lexical import LexicalResult, lexical_candidates
from avito_rec_sys.utils.io import ensure_dir, resolve_path

DEFAULT_FIELD_WEIGHTS = {"title": 1.0, "params": 0.5, "desc": 0.3}


@dataclass
class Context:
    cfg: dict
    work: Path
    train_raw: pl.DataFrame
    splits: SplitResult
    bench_items: pl.DataFrame
    bench_queries: pl.DataFrame
    master_items: pl.DataFrame  # every item that can appear in any corpus, with lemma columns


def _save_heldout(h: HeldOut, d: Path) -> None:
    ensure_dir(d)
    h.queries.write_parquet(d / "queries.parquet")
    h.qrels.write_parquet(d / "qrels.parquet")
    h.positive_items.write_parquet(d / "positive_items.parquet")


def _load_heldout(d: Path) -> HeldOut:
    return HeldOut(*(pl.read_parquet(d / f"{n}.parquet") for n in ("queries", "qrels", "positive_items")))


def build_context(cfg: dict, n_workers: int = 6) -> Context:
    work = ensure_dir(resolve_path(cfg, "work_dir"))
    train_raw = load_train(resolve_path(cfg, "train_parquet"))
    bench_items = load_benchmark_items(resolve_path(cfg, "benchmark_items_parquet"))
    bench_queries = load_benchmark_queries(resolve_path(cfg, "benchmark_queries_parquet"))

    sdir = work / "splits"
    if (sdir / "train_part.parquet").exists():
        splits = SplitResult(
            pl.read_parquet(sdir / "train_part.parquet"), _load_heldout(sdir / "val"), _load_heldout(sdir / "ranker"), 0
        )
    else:
        sp = cfg["split"]
        splits = make_splits(
            train_raw, sp["n_validation_queries"], sp["n_ranker_queries"], sp["seen_fraction"], cfg["seed"]
        )
        ensure_dir(sdir)
        splits.train_part.write_parquet(sdir / "train_part.parquet")
        _save_heldout(splits.val, sdir / "val")
        _save_heldout(splits.ranker, sdir / "ranker")

    def build_master() -> pl.DataFrame:
        cols = bench_items.columns
        parts = [bench_items, splits.val.positive_items.select(cols), splits.ranker.positive_items.select(cols)]
        items = pl.concat(parts).unique(subset=["item_id"], keep="first", maintain_order=True)
        return prepare_items(items, n_workers)

    master = cached(work / "master_items.parquet", build_master)
    return Context(cfg, work, train_raw, splits, bench_items, bench_queries, master)


@dataclass
class Stage:
    name: str
    corpus: pl.DataFrame
    queries: pl.DataFrame
    qrels: dict[str, set[str]] | None
    logs: QueryLogs
    centroids: dict


def make_stage(ctx: Context, name: str, n_workers: int = 6) -> Stage:
    bench_ids = set(ctx.bench_items["item_id"].to_list())
    sp = ctx.splits
    if name in ("val", "ranker"):
        held = sp.val if name == "val" else sp.ranker
        ids = bench_ids | set(held.positive_items["item_id"].to_list())
        queries = held.queries
        qrels: dict[str, set[str]] | None = {}
        for qid, item in zip(held.qrels["query_id"].to_list(), held.qrels["item_id"].to_list()):
            qrels.setdefault(qid, set()).add(item)
        history = sp.train_part
        legal_for_geo = pl.concat(
            [history.select("item_id", "item_location_id", "item_latitude", "item_longitude"),
             ctx.bench_items.select("item_id", "item_location_id", "item_latitude", "item_longitude")]
        ).unique(subset=["item_id"])
        held_pos = set(sp.val.positive_items["item_id"].to_list()) | set(sp.ranker.positive_items["item_id"].to_list())
        legal_for_geo = legal_for_geo.filter(~pl.col("item_id").is_in(held_pos))
    elif name == "test":
        ids, qrels = bench_ids, None
        queries = add_normalized_query_column(ctx.bench_queries)
        history = add_normalized_query_column(ctx.train_raw)
        legal_for_geo = pl.concat(
            [history.select("item_id", "item_location_id", "item_latitude", "item_longitude"),
             ctx.bench_items.select("item_id", "item_location_id", "item_latitude", "item_longitude")]
        ).unique(subset=["item_id"])
    else:
        raise ValueError(name)

    corpus = ctx.master_items.filter(pl.col("item_id").is_in(ids))
    if name == "test":
        corpus = ctx.master_items.filter(pl.col("item_id").is_in(bench_ids))
    queries = prepare_queries(queries, n_workers)
    # item-based centroids where the location holds items (unchanged), click-based ones for the rest
    centroids = {**click_centroids(history), **location_centroids(legal_for_geo)}
    return Stage(name, corpus, queries, qrels, QueryLogs(history), centroids)


def build_index(corpus: pl.DataFrame, k1: float = 1.2, b: float = 0.75) -> FieldBM25Index:
    from avito_rec_sys.data.corpus import tokens

    return FieldBM25Index(k1, b).fit(
        {
            "title": tokens(corpus["title_lemmas"]),
            "params": tokens(corpus["params_lemmas"]),
            "desc": tokens(corpus["desc_lemmas"]),
        }
    )


def candidates_and_features(
    stage: Stage,
    index: FieldBM25Index,
    field_weights: dict[str, float] = DEFAULT_FIELD_WEIGHTS,
    k_local: int = 150,
    k_global: int = 100,
    with_features: bool = True,
):
    items = ItemTable(stage.corpus, index.vocab)
    queries = QueryTable(stage.queries, stage.centroids)
    extra = log_candidate_positions(queries, items, stage.logs)
    cand: LexicalResult = lexical_candidates(
        index, queries.lemma_tokens, queries.loc, items.loc_positions, field_weights,
        k_local=k_local, k_global=k_global, extra_positions=extra, chunk=64,
    )
    feats = build_features(cand, items, queries, stage.logs, index) if with_features else None
    return items, queries, cand, feats


def labels_for(feats, items: ItemTable, queries_ids: list[str], qrels: dict[str, set[str]]) -> np.ndarray:
    """1 if the candidate is a relevant item for its query."""
    q, i = feats["q"].to_numpy(), feats["i"].to_numpy()
    y = np.zeros(len(q), dtype=np.int8)
    for k in range(len(q)):
        if items.ids[i[k]] in qrels.get(queries_ids[q[k]], ()):
            y[k] = 1
    return y
