"""Candidate generation and feature building for one stage.

    queries -> bge-m3 (dense + sparse + ColBERT vectors)
            -> three tours: BM25, dense, sparse, each with a local (same location) and a global ranking
            -> weighted RRF per ranking, merged with a "global but near the query location" list
            -> pool of `pool.size` candidates, plus the candidates suggested by the click logs
            -> feature table (lexical / geo / category / quality / log + neural scores + pool ranks)

Queries are processed in chunks: every step is per-query independent, so chunking changes nothing
in the result but bounds peak memory. The ColBERT scores, which need the 20 GB on-disk store, are
computed for all pairs in one pass at the end. Everything heavy (the encoder, the encoded corpus)
is injected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp

from avito_rec_sys.features.assemble import FEATURE_COLUMNS, build_features
from avito_rec_sys.features.logs import log_candidate_positions
from avito_rec_sys.features.tables import ItemTable, QueryTable
from avito_rec_sys.pipeline.context import Stage, build_index
from avito_rec_sys.retrieval.colbert_store import pair_maxsim
from avito_rec_sys.retrieval.encode_store import CorpusStore, sparse_to_csr
from avito_rec_sys.retrieval.encoders import EncodedBatch, encode_bi_encoder
from avito_rec_sys.retrieval.lexical import lexical_candidates
from avito_rec_sys.retrieval.pool import merge_pool, nearby_positions
from avito_rec_sys.retrieval.rrf import rrf_fuse
from avito_rec_sys.retrieval.tours import dense_tour, sparse_tour
from avito_rec_sys.training.biencoder_train import query_tower_text

NEURAL_COLUMNS = ["dense_cos", "sparse_score", "colbert_maxsim", "cos_query_microcat_centroid"]
# Query-relative views of the retrieval signals: a raw cosine means different things for different
# queries, its rank in the pool and its gap to the best candidate do not.
_RELATIVE_SOURCES = ["dense_cos", "sparse_score", "colbert_maxsim"]
POOL_COLUMNS = ["pool_rank"] + [f"{c}_{kind}" for c in _RELATIVE_SOURCES for kind in ("rank_in_pool", "minus_max")]
RANKER_FEATURES = FEATURE_COLUMNS + NEURAL_COLUMNS + POOL_COLUMNS  # the input layout of the CatBoost model


def item_tower_texts(corpus: pl.DataFrame) -> list[str]:
    """Item-tower input of the encoder: title + whitelist-filtered params."""
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
    """Unit-length mean embedding per microcategory. Covers every microcategory present in the
    corpus and needs no training."""
    out = {}
    for mc, pos in items.microcat_positions.items():
        v = dense[pos].astype(np.float32).mean(axis=0)
        n = np.linalg.norm(v)
        out[mc] = v / n if n > 0 else v
    return out


def add_relative_features(feats: pd.DataFrame, sources: list[str]) -> pd.DataFrame:
    """Rank inside the query's pool and gap to the pool's best value, for each score column."""
    for c in sources:
        g = feats.groupby("q")[c]
        feats[f"{c}_rank_in_pool"] = g.rank(ascending=False, method="first").astype(np.float32)
        feats[f"{c}_minus_max"] = (feats[c] - g.transform("max")).astype(np.float32)
    return feats


def add_pool_features(
    feats: pd.DataFrame, fused: list[np.ndarray], missing_rank: int, sources: list[str] = _RELATIVE_SOURCES
) -> pd.DataFrame:
    """`pool_rank` (position in the merged retrieval ranking; `missing_rank` for candidates that
    entered only through the log sources) and the query-relative views of the `sources` scores."""
    lens = [len(f) for f in fused]
    ranks = pd.DataFrame({
        "q": np.repeat(np.arange(len(fused)), lens).astype(feats["q"].dtype),
        "i": (np.concatenate(fused) if lens else np.empty(0)).astype(feats["i"].dtype),
        "pool_rank": np.concatenate([np.arange(n) for n in lens]).astype(np.float32) if lens else np.empty(0, np.float32),
    })
    feats = feats.merge(ranks, on=["q", "i"], how="left")  # a left merge keeps the (query-sorted) row order
    feats["pool_rank"] = feats["pool_rank"].fillna(missing_rank).astype(np.float32)
    return add_relative_features(feats, sources)


def _pairwise_dot_sparse(qs: sp.csr_matrix, ds: sp.csr_matrix, q: np.ndarray, i: np.ndarray, chunk=100_000):
    out = np.zeros(len(q), dtype=np.float32)
    for s in range(0, len(q), chunk):
        out[s : s + chunk] = np.asarray(qs[q[s : s + chunk]].multiply(ds[i[s : s + chunk]]).sum(axis=1)).ravel()
    return out


class CandidateGenerator:
    """Builds the candidate feature table of a stage.

    `store` must be aligned with `stage.corpus` (row i = item i). The result of `features()` has one
    row per (query position `q`, corpus position `i`) pair, sorted by `q`, with the columns of
    `RANKER_FEATURES` (in arbitrary order: select them by name).
    """

    def __init__(self, stage: Stage, store: CorpusStore, encoder, cfg: dict, device: str = "cuda", enc_batch: int = 64):
        self.stage, self.store, self.cfg, self.device = stage, store, cfg, device
        self.index = build_index(stage.corpus)
        self.items = ItemTable(stage.corpus, self.index.vocab)
        self.queries = QueryTable(stage.queries, stage.centroids)
        self.corpus_dense = store.dense
        self.microcat_centroids = microcat_centroids(store.dense, self.items)
        # queries are encoded in one go, exactly like the corpus, then sliced per chunk
        self.q_enc: EncodedBatch = encode_bi_encoder(encoder, query_texts(stage.queries), enc_batch, cfg["text"]["query_max_tokens"])
        self.q_sparse = sparse_to_csr(self.q_enc.sparse)

    def features(self, log=print) -> pd.DataFrame:
        n, chunk = self.queries.n, self.cfg["pool"]["query_chunk"]
        parts = []
        for start in range(0, n, chunk):
            sl = slice(start, min(start + chunk, n))
            part = self._chunk_features(sl)
            part["q"] += start
            parts.append(part)
            log(f"[candidates] {self.stage.name}: {sl.stop}/{n} queries")
        feats = pd.concat(parts, ignore_index=True)
        # ColBERT is scored for all pairs at once: one sequential pass over the on-disk store
        log(f"[candidates] {self.stage.name}: ColBERT MaxSim for {len(feats)} pairs")
        q_vecs = [v.astype(np.float32) for v in self.q_enc.colbert]
        feats["colbert_maxsim"] = pair_maxsim(self.store.colbert, q_vecs, feats["q"].to_numpy(), feats["i"].to_numpy())
        return add_relative_features(feats, ["colbert_maxsim"])

    # ------------------------------------------------------------------ one chunk of queries

    def _chunk_features(self, sl: slice) -> pd.DataFrame:
        cfg, items = self.cfg, self.items
        pool_cfg = cfg["pool"]
        field_weights, rrf_weights = cfg["bm25"]["field_weights"], cfg["rrf"]["weights"]
        queries = QueryTable(self.stage.queries.slice(sl.start, sl.stop - sl.start), self.stage.centroids)
        q_dense = self.q_enc.dense[sl]
        q_sparse = self.q_sparse[sl]

        # ---- tours: BM25, dense, sparse; each gives a global and a local (same location) ranking
        k_glob, k_loc = pool_cfg["retrieval_k"], pool_cfg["retrieval_k"] // 2
        bm25 = lexical_candidates(
            self.index, queries.lemma_tokens, queries.loc, items.loc_positions, field_weights,
            k_local=k_loc, k_global=k_glob, k_ceiling=k_glob, chunk=64,
        )
        d_glob, d_loc = dense_tour(
            q_dense, self.corpus_dense, queries.loc, items.loc_positions, k_global=k_glob, k_local=k_loc, device=self.device
        )
        s_glob, s_loc = sparse_tour(
            q_sparse, self.store.sparse, queries.loc, items.loc_positions, k_global=k_glob, k_local=k_loc
        )
        tours_g = {"bm25f": bm25.ranked_global, "dense": d_glob, "sparse": s_glob}
        tours_l = {"bm25f": bm25.ranked_local, "dense": d_loc, "sparse": s_loc}

        # ---- fusion: local and global rankings are fused separately, then merged with the "near" list
        pm, k_rrf = pool_cfg["merge"], cfg["rrf"]["k"]
        fused: list[np.ndarray] = []
        for q in range(queries.n):
            f_loc = np.array(rrf_fuse({t: r[q] for t, r in tours_l.items()}, rrf_weights, k_rrf, k_loc), dtype=np.int64)
            f_glob = np.array(rrf_fuse({t: r[q] for t, r in tours_g.items()}, rrf_weights, k_rrf, k_glob), dtype=np.int64)
            near = nearby_positions(f_glob, queries.lat[q], queries.lon[q], items.lat, items.lon, pm["near_radius_km"])
            fused.append(merge_pool(f_loc, f_glob, near, pm, k_glob))

        # ---- the pool is the head of the fused ranking plus the candidates suggested by the click logs
        extra = log_candidate_positions(queries, items, self.stage.logs)
        pool = [np.unique(np.concatenate([f[: pool_cfg["size"]], e]).astype(np.int64)) for f, e in zip(fused, extra)]

        # ---- pair features: the lexical part re-gathered for exactly these pairs, then the neural scores
        cand = lexical_candidates(
            self.index, queries.lemma_tokens, queries.loc, items.loc_positions, field_weights,
            k_local=0, k_global=0, k_ceiling=0, extra_positions=pool, chunk=64,
        )
        feats = build_features(cand, items, queries, self.stage.logs, self.index)
        q_idx, i_idx = feats["q"].to_numpy(), feats["i"].to_numpy()

        qd = q_dense.astype(np.float32)
        feats["dense_cos"] = np.concatenate(
            [np.einsum("ij,ij->i", qd[q_idx[s : s + 200_000]], self.corpus_dense[i_idx[s : s + 200_000]].astype(np.float32))
             for s in range(0, len(q_idx), 200_000)]
        ) if len(q_idx) else np.empty(0, dtype=np.float32)
        feats["sparse_score"] = _pairwise_dot_sparse(q_sparse, self.store.sparse, q_idx, i_idx)

        zero = np.zeros(qd.shape[1], dtype=np.float32)
        cen = (np.stack([self.microcat_centroids.get(int(m), zero) for m in items.microcat[i_idx]])
               if len(i_idx) else np.zeros((0, qd.shape[1]), np.float32))
        feats["cos_query_microcat_centroid"] = np.einsum("ij,ij->i", qd[q_idx], cen) if len(i_idx) else np.empty(0, np.float32)

        feats = add_pool_features(feats, fused, missing_rank=k_glob, sources=["dense_cos", "sparse_score"])
        num = feats.select_dtypes("float64").columns
        feats[num] = feats[num].astype(np.float32)
        return feats
