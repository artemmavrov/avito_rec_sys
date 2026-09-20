"""Shared helpers for the GPU-stage scripts (s05-s11).

The master corpus (benchmark items + every hold-out positive) is encoded once
per encoder; each stage (val | ranker | test) is a row subset of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from avito_rec_sys.inference.lexical_pipeline import Context, Stage, build_index, make_stage
from avito_rec_sys.inference.neural_pipeline import StageOutput, item_tower_texts, run_stage
from avito_rec_sys.retrieval.encode_store import CorpusStore, build_store, load_store
from avito_rec_sys.retrieval.encoders import load_bi_encoder


def ensure_master_store(ctx: Context, models_cfg: dict, name: str, model_path: str | None, device: str = "cuda") -> CorpusStore:
    """Encode the master corpus with the given encoder (or load the cached store)."""
    out = ctx.work / f"store_{name}"
    if (out / "sparse.npz").exists():
        return load_store(out)
    model = load_bi_encoder(models_cfg, device, model_path=model_path)
    texts = item_tower_texts(ctx.master_items)
    store = build_store(model, texts, out, ctx.cfg["text"]["item_max_tokens"], batch_size=128)
    del model
    return store


def stage_inputs(ctx: Context, name: str, master: CorpusStore, max_queries: int | None = None):
    """(Stage, BM25 index over the stage corpus, stage store) for `name`."""
    stage = make_stage(ctx, name)
    if max_queries is not None and stage.queries.height > max_queries:
        keep = stage.queries.head(max_queries)  # hold-outs are already randomly ordered
        stage = Stage(stage.name, stage.corpus, keep, stage.qrels, stage.logs, stage.centroids)
    master_pos = {v: i for i, v in enumerate(ctx.master_items["item_id"].to_list())}
    positions = np.array([master_pos[i] for i in stage.corpus["item_id"].to_list()], dtype=np.int64)
    return stage, build_index(stage.corpus), master.subset(positions)


def stage_qrels(stage: Stage) -> dict[int, set[str]]:
    """qrels keyed by query POSITION in the stage (the key used everywhere in the pipeline)."""
    ids = stage.queries["query_id"].to_list()
    return {q: stage.qrels[ids[q]] for q in range(len(ids))}


def narrowing_mode(work: Path) -> str:
    """"colbert" or "rrf": the decision stage 5 stored (default colbert when it hasn't run)."""
    return load_json(work / "narrowing_mode.json", {"mode": "colbert"})["mode"]


def run_full_stage(
    ctx: Context, stage, index, store, bi_model, cross_scorer, field_weights, rrf_weights, device="cuda", thin=None
) -> StageOutput:
    return run_stage(
        stage.queries, stage.corpus, stage.logs, stage.centroids, index, store, bi_model, cross_scorer,
        ctx.cfg, field_weights, rrf_weights, device=device, narrowing=narrowing_mode(ctx.work), thin=thin,
    )


def load_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def ceiling_report(out: StageOutput, item_ids: list[str], qrels: dict[int, set[str]], ks=(50, 300, 1000)) -> dict:
    """Recall@K per tour, for the RRF pool and for the ColBERT-narrowed pool (§9.5)."""
    from avito_rec_sys.eval.metrics import ceiling_curve

    n = len(out.fused_pool)
    as_ids = lambda lists: {q: [item_ids[p] for p in lists[q]] for q in range(n)}  # noqa: E731
    tours = {name: as_ids(lists) for name, lists in out.tours.items()}
    tours["rrf_pool"] = as_ids(out.fused_pool)
    tours["narrowed_pool"] = as_ids(out.narrowed_pool)  # + log candidates
    curve = ceiling_curve(tours, qrels, ks)
    return {name: {str(k): v for k, v in d.items()} for name, d in curve.items()}


def tune_rrf(out: StageOutput, item_ids: list[str], qrels: dict[int, set[str]], cfg: dict) -> dict[str, float]:
    from avito_rec_sys.retrieval.rrf import tune_rrf_weights

    n = len(out.fused_pool)
    names = list(out.tours)
    per_query = {q: {t: [item_ids[p] for p in out.tours[t][q]] for t in names} for q in range(n)}
    return tune_rrf_weights(per_query, qrels, names, cfg["rrf"]["k"], cfg["pool_sizes"]["retrieval_k"])


def narrowing_decision(out: StageOutput, item_ids: list[str], qrels: dict[int, set[str]], cfg: dict) -> dict:
    """ColBERT narrowing vs plain truncation of the RRF pool, at the same size.

    Both pools are cut to the cross-encoder's budget and compared on Recall
    (log candidates excluded from both, they bypass the cut either way).
    ColBERT is kept only if it loses no more than narrowing.max_loss_pp
    against simply taking the first `colbert_k` of the fused ranking.
    """
    from avito_rec_sys.eval.metrics import recall_at_k

    n = len(out.fused_pool)
    as_ids = lambda lists: {q: [item_ids[p] for p in lists[q]] for q in range(n)}  # noqa: E731
    r_rrf = recall_at_k(as_ids(out.rrf_only), qrels, 10**9)
    r_colbert = recall_at_k(as_ids(out.colbert_only), qrels, 10**9)
    loss_pp = 100 * (r_rrf - r_colbert)
    keep = loss_pp <= cfg["narrowing"]["max_loss_pp"]
    return {
        "recall_rrf_truncation": r_rrf, "recall_colbert_narrowing": r_colbert, "loss_pp": loss_pp,
        "max_loss_pp": cfg["narrowing"]["max_loss_pp"], "mode": "colbert" if keep else "rrf",
    }
