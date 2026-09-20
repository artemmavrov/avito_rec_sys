"""Dense and sparse retrieval tours over a pre-encoded corpus (§1).

Each tour returns, per query, a best-first GLOBAL ranking and a best-first
LOCAL ranking (restricted to the query's location) -- the same shape the
lexical tour produces, so all tours feed the same fusion and slot logic.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import scipy.sparse as sp
import torch

TourRanking = tuple[list[np.ndarray], list[np.ndarray]]  # (global lists, local lists)


def _rank_rows(
    scores: torch.Tensor,
    q_locs: np.ndarray,
    location_positions: Mapping[int, np.ndarray],
    k_global: int,
    k_local: int,
    q_offset: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    glob, loc = [], []
    for r in range(scores.shape[0]):
        row = scores[r]
        glob.append(torch.topk(row, min(k_global, row.shape[0])).indices.cpu().numpy())
        lp = location_positions.get(int(q_locs[q_offset + r]))
        if lp is not None and lp.size and k_local > 0:
            lp_t = torch.as_tensor(lp, device=row.device)
            top = torch.topk(row[lp_t], min(k_local, lp.size)).indices.cpu().numpy()
            loc.append(lp[top])
        else:
            loc.append(np.empty(0, dtype=np.int64))
    return glob, loc


def dense_tour(
    q_dense: np.ndarray,
    corpus_dense: np.ndarray,
    q_locs: np.ndarray,
    location_positions: Mapping[int, np.ndarray],
    k_global: int = 1000,
    k_local: int = 300,
    device: str = "cuda",
    chunk: int = 256,
) -> TourRanking:
    """Cosine retrieval (vectors are L2-normalized by the encoder, so dot = cosine)."""
    docs = torch.as_tensor(corpus_dense, device=device)
    if device == "cpu":
        docs = docs.float()  # fp16 matmul is very slow / poorly supported on CPU
    glob: list[np.ndarray] = []
    loc: list[np.ndarray] = []
    for s in range(0, len(q_dense), chunk):
        q = torch.as_tensor(q_dense[s : s + chunk], device=device).to(docs.dtype)
        scores = (q @ docs.T).float()
        g, l = _rank_rows(scores, q_locs, location_positions, k_global, k_local, s)
        glob += g
        loc += l
    return glob, loc


def sparse_tour(
    q_sparse: sp.csr_matrix,
    corpus_sparse: sp.csr_matrix,
    q_locs: np.ndarray,
    location_positions: Mapping[int, np.ndarray],
    k_global: int = 1000,
    k_local: int = 300,
    chunk: int = 128,
) -> TourRanking:
    """bge-m3 lexical-weight retrieval: score = sum over shared tokens of q_w * d_w.
    Zero-score items are dropped -- a query sharing no token with an item has
    no sparse evidence about it."""
    corpus_t = corpus_sparse.T.tocsr()
    glob: list[np.ndarray] = []
    loc: list[np.ndarray] = []
    for s in range(0, q_sparse.shape[0], chunk):
        scores = torch.as_tensor((q_sparse[s : s + chunk] @ corpus_t).toarray())
        g, l = _rank_rows(scores, q_locs, location_positions, k_global, k_local, s)
        for i in range(len(g)):
            row = scores[i].numpy()
            glob.append(g[i][row[g[i]] > 0])
            loc.append(l[i][row[l[i]] > 0] if l[i].size else l[i])
    return glob, loc
