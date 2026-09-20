"""Full candidate pipeline for one stage (val | ranker | test):

    queries -> bge-m3 (dense + sparse + ColBERT)
            -> tours: BM25F, dense, sparse                (local + global rankings)
            -> weighted RRF                               -> pool  (retrieval_k)
            -> ColBERT MaxSim narrowing                   -> pool  (colbert_k)
            -> + log-signal candidates (bypass the cut)
            -> feature bank incl. neural columns          -> CatBoost input

Everything heavy (encoders, cross-encoder) is injected, so the same code runs
at benchmark scale on the server and on a 40-query subset in the local
smoke test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp

from avito_rec_sys.features.assemble import FEATURE_COLUMNS, build_features
from avito_rec_sys.features.bm25f import FieldBM25Index
from avito_rec_sys.features.logs import QueryLogs, log_candidate_positions
from avito_rec_sys.features.tables import ItemTable, QueryTable
from avito_rec_sys.retrieval.colbert_narrow import maxsim_batch, narrow_pool, parallel_map
from avito_rec_sys.retrieval.encode_store import CorpusStore, sparse_to_csr
from avito_rec_sys.retrieval.encoders import encode_bi_encoder
from avito_rec_sys.retrieval.lexical import lexical_candidates
from avito_rec_sys.retrieval.pool import merge_pool, nearby_positions
from avito_rec_sys.retrieval.rrf import rrf_fuse
from avito_rec_sys.retrieval.tours import dense_tour, sparse_tour
from avito_rec_sys.training.biencoder_train import query_tower_text

NEURAL_COLUMNS = ["dense_cos", "sparse_score", "colbert_maxsim", "ce_logit", "cos_query_microcat_centroid"]
# Query-relative views of the retrieval signals: a raw cosine means different things for different
# queries, its rank / gap to the best candidate does not (the BM25 analogue is in FEATURE_COLUMNS).
# All are computed on the FULL pool before any row thinning, so training rows and inference agree.
_RELATIVE_SOURCES = ["dense_cos", "sparse_score", "colbert_maxsim"]
POOL_COLUMNS = ["pool_rank"] + [f"{c}_{kind}" for c in _RELATIVE_SOURCES for kind in ("rank_in_pool", "minus_max")]
FEATURE_COLUMNS_NEURAL = FEATURE_COLUMNS + NEURAL_COLUMNS + POOL_COLUMNS


def item_tower_texts(corpus: pl.DataFrame) -> list[str]:
    """Item-tower input (§3.1) -- identical to `build_item_tower_text` used
    for training data, computed from the already-filtered params column."""
    return [
        f"{t.strip()} {p}".strip()
        for t, p in zip(corpus["item_title_raw"].fill_null("").to_list(), corpus["params_filtered"].to_list())
    ]


def query_texts(queries: pl.DataFrame) -> list[str]:
    return [
        query_tower_text(q, p)
        for q, p in zip(queries["search_query"].to_list(), queries["search_infm_params_text"].to_list())
    ]


def microcat_centroids(dense: np.ndarray, items: ItemTable) -> dict[int, np.ndarray]:
    """Unit-length mean embedding per microcategory (§4). Covers every
    microcategory present in the corpus, needs no training (a classifier
    query->microcat could not cover the 540 microcats absent from train)."""
    out = {}
    for mc, pos in items.microcat_positions.items():
        v = dense[pos].astype(np.float32).mean(axis=0)
        n = np.linalg.norm(v)
        out[mc] = v / n if n > 0 else v
    return out


@dataclass
class StageOutput:
    feats: pd.DataFrame  # candidate pairs + all feature columns
    tours: dict[str, list[np.ndarray]] = field(default_factory=dict)  # global rankings per tour (for ceiling curves)
    fused_pool: list[np.ndarray] = field(default_factory=list)  # after RRF
    narrowed_pool: list[np.ndarray] = field(default_factory=list)  # narrowing result + log candidates
    colbert_only: list[np.ndarray] = field(default_factory=list)  # ColBERT-narrowed pool alone (gate diagnostic)
    rrf_only: list[np.ndarray] = field(default_factory=list)  # RRF-truncated pool alone (gate diagnostic)


def add_pool_features(feats: pd.DataFrame, fused: list[np.ndarray], missing_rank: int) -> pd.DataFrame:
    """`pool_rank` (position in the merged retrieval ranking; `missing_rank` for candidates that
    entered only through the log sources) and the query-relative dense / sparse / ColBERT features."""
    lens = [len(f) for f in fused]
    ranks = pd.DataFrame({
        "q": np.repeat(np.arange(len(fused)), lens).astype(feats["q"].dtype),
        "i": (np.concatenate(fused) if lens else np.empty(0)).astype(feats["i"].dtype),
        "pool_rank": np.concatenate([np.arange(n) for n in lens]).astype(np.float32) if lens else np.empty(0, np.float32),
    })
    feats = feats.merge(ranks, on=["q", "i"], how="left")  # left merge keeps the (query-sorted) row order
    feats["pool_rank"] = feats["pool_rank"].fillna(missing_rank).astype(np.float32)
    for c in _RELATIVE_SOURCES:
        g = feats.groupby("q")[c]
        feats[f"{c}_rank_in_pool"] = g.rank(ascending=False, method="first").astype(np.float32)
        feats[f"{c}_minus_max"] = (feats[c] - g.transform("max")).astype(np.float32)
    return feats


def _pairwise_dot_sparse(qs: sp.csr_matrix, ds: sp.csr_matrix, q: np.ndarray, i: np.ndarray, chunk=100_000):
    out = np.zeros(len(q), dtype=np.float32)
    for s in range(0, len(q), chunk):
        out[s : s + chunk] = np.asarray(qs[q[s : s + chunk]].multiply(ds[i[s : s + chunk]]).sum(axis=1)).ravel()
    return out


def run_stage(
    queries_df: pl.DataFrame,
    corpus: pl.DataFrame,
    logs: QueryLogs,
    centroids_geo: dict,
    index: FieldBM25Index,
    store: CorpusStore,
    bi_model,
    cross_scorer: Callable[[list[str], list[str]], np.ndarray] | None,
    cfg: dict,
    field_weights: dict,
    rrf_weights: dict,
    device: str = "cuda",
    enc_batch: int = 64,
    ce_query_chunk: int = 20,
    narrowing: str = "colbert",
    thin: Callable[[pd.DataFrame, ItemTable, QueryTable], np.ndarray] | None = None,
) -> StageOutput:
    """`narrowing`: "colbert" (MaxSim narrowing) or "rrf" (plain truncation of the fused
    pool); stage 5 decides which, see configs narrowing.max_loss_pp.
    `thin(feats, items, queries) -> bool mask` drops candidate rows BEFORE the
    cross-encoder runs (training-set construction only; pool-relative features have
    already been computed on the full pool)."""
    pool_cfg = cfg["pool_sizes"]
    text_cfg = cfg["text"]
    items = ItemTable(corpus, index.vocab)
    queries = QueryTable(queries_df, centroids_geo)
    n_q = queries.n

    # ---- encode queries once
    q_text = query_texts(queries_df)
    enc = encode_bi_encoder(bi_model, q_text, enc_batch, text_cfg["query_max_tokens"])
    q_sparse = sparse_to_csr(enc.sparse)

    # ---- tours (bm25f ranking comes from the lexical pass; its pair scores are re-gathered later)
    bm25 = lexical_candidates(
        index, queries.lemma_tokens, queries.loc, items.loc_positions, field_weights,
        k_local=pool_cfg["retrieval_k"] // 2, k_global=pool_cfg["retrieval_k"], k_ceiling=pool_cfg["retrieval_k"],
        chunk=64,
    )
    d_glob, d_loc = dense_tour(
        enc.dense, store.dense, queries.loc, items.loc_positions,
        k_global=pool_cfg["retrieval_k"], k_local=pool_cfg["retrieval_k"] // 2, device=device,
    )
    s_glob, s_loc = sparse_tour(
        q_sparse, store.sparse, queries.loc, items.loc_positions,
        k_global=pool_cfg["retrieval_k"], k_local=pool_cfg["retrieval_k"] // 2,
    )
    tours_g = {"bm25f": bm25.ranked_global, "dense": d_glob, "sparse": s_glob}
    tours_l = {"bm25f": bm25.ranked_local, "dense": d_loc, "sparse": s_loc}

    # ---- fusion: local and global rankings fused separately, then merged with a "nearby" list
    # (global candidates within pool_merge.near_radius_km of the query location), see retrieval/pool.py
    k_rrf, K = cfg["rrf"]["k"], pool_cfg["retrieval_k"]
    pm = cfg["pool_merge"]
    fused: list[np.ndarray] = []
    for q in range(n_q):
        f_loc = np.array(rrf_fuse({t: l[q] for t, l in tours_l.items()}, rrf_weights, k_rrf, K // 2), dtype=np.int64)
        f_glob = np.array(rrf_fuse({t: g[q] for t, g in tours_g.items()}, rrf_weights, k_rrf, K), dtype=np.int64)
        near = nearby_positions(f_glob, queries.lat[q], queries.lon[q], items.lat, items.lon, pm["near_radius_km"])
        fused.append(merge_pool(f_loc, f_glob, near, pm, K))

    # ---- narrowing to what the cross-encoder reads, then log candidates join outside the cut
    extra = log_candidate_positions(queries, items, logs)
    k_cb, k_ce = pool_cfg["colbert_k"], pool_cfg["reranker_k"]

    def narrow_one(q: int) -> np.ndarray:
        return narrow_pool(enc.colbert[q], fused[q], store.colbert, k_cb)[:k_ce] if len(fused[q]) else fused[q]

    narrowed: list[np.ndarray] = []
    colbert_only: list[np.ndarray] = parallel_map(narrow_one, n_q)  # the CPU-heavy part, one query per task
    rrf_only: list[np.ndarray] = []
    for q in range(n_q):
        # both variants are always built: the stage-5 gate compares them whichever one is active
        rr = fused[q][:k_cb][:k_ce]
        rrf_only.append(rr)
        keep = colbert_only[q] if narrowing == "colbert" else rr
        narrowed.append(np.unique(np.concatenate([keep, extra[q]]).astype(np.int64)))

    # ---- pair features: lexical part re-gathered for exactly these pairs
    cand = lexical_candidates(
        index, queries.lemma_tokens, queries.loc, items.loc_positions, field_weights,
        k_local=0, k_global=0, k_ceiling=0, extra_positions=narrowed, chunk=64,
    )
    feats = build_features(cand, items, queries, logs, index)
    q_idx, i_idx = feats["q"].to_numpy(), feats["i"].to_numpy()

    qd = enc.dense.astype(np.float32)
    dd = store.dense
    feats["dense_cos"] = np.concatenate(
        [np.einsum("ij,ij->i", qd[q_idx[s : s + 200_000]], dd[i_idx[s : s + 200_000]].astype(np.float32))
         for s in range(0, len(q_idx), 200_000)]
    ) if len(q_idx) else np.empty(0, dtype=np.float32)
    feats["sparse_score"] = _pairwise_dot_sparse(q_sparse, store.sparse, q_idx, i_idx)

    colbert = np.zeros(len(q_idx), dtype=np.float32)
    bounds = np.flatnonzero(np.diff(q_idx, prepend=-1))  # start row of each query's block
    ends = np.append(bounds[1:], len(q_idx))

    def maxsim_block(k: int) -> np.ndarray:
        b, e = bounds[k], ends[k]
        return maxsim_batch(enc.colbert[q_idx[b]], [store.colbert[int(p)] for p in i_idx[b:e]])

    for b, e, scores in zip(bounds, ends, parallel_map(maxsim_block, len(bounds))):
        colbert[b:e] = scores
    feats["colbert_maxsim"] = colbert

    centroids = microcat_centroids(store.dense, items)
    zero = np.zeros(qd.shape[1], dtype=np.float32)
    cen = np.stack([centroids.get(int(m), zero) for m in items.microcat[i_idx]]) if len(i_idx) else np.zeros((0, qd.shape[1]), np.float32)
    feats["cos_query_microcat_centroid"] = np.einsum("ij,ij->i", qd[q_idx], cen) if len(i_idx) else np.empty(0, np.float32)

    feats = add_pool_features(feats, fused, pool_cfg["retrieval_k"])

    if thin is not None:
        keep_rows = thin(feats, items, queries)
        feats = feats[keep_rows].reset_index(drop=True)
        q_idx, i_idx = feats["q"].to_numpy(), feats["i"].to_numpy()
        bounds = np.flatnonzero(np.diff(q_idx, prepend=-1))
        ends = np.append(bounds[1:], len(q_idx))

    if cross_scorer is not None:
        tower = item_tower_texts(corpus)
        ce = np.zeros(len(q_idx), dtype=np.float32)
        for s in range(0, len(bounds), ce_query_chunk):  # bounded RAM: a few queries' pairs at a time
            lo, hi = bounds[s], ends[min(s + ce_query_chunk, len(bounds)) - 1]
            ce[lo:hi] = cross_scorer([q_text[q] for q in q_idx[lo:hi]], [tower[i] for i in i_idx[lo:hi]])
        feats["ce_logit"] = ce
    else:
        feats["ce_logit"] = np.float32(0.0)  # constant column: keeps the feature layout stable when CE is skipped

    num = feats.select_dtypes("float64").columns
    feats[num] = feats[num].astype(np.float32)
    return StageOutput(feats, tours_g, fused, narrowed, colbert_only, rrf_only)
