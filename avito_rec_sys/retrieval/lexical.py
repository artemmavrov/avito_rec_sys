"""Lexical retrieval tour: multi-field BM25 with a local (same-location) and a
global ranking, plus per-pair field scores for the ranker (§1, §6.3).

For every query we keep
  - a global ranking over the whole corpus (used for the ceiling curve),
  - a local ranking restricted to items in the query's location,
  - a candidate pool = local top-K_local  U  global top-K_global  U  extra
    items supplied by log-based sources (q->item, title bridge).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from avito_rec_sys.features.bm25f import FieldBM25Index, combine_fields


@dataclass
class LexicalResult:
    cand_q: np.ndarray  # int32, query position per candidate pair
    cand_i: np.ndarray  # int32, corpus position per candidate pair
    s_title: np.ndarray
    s_params: np.ndarray
    s_desc: np.ndarray
    s_total: np.ndarray
    ranked_global: list[np.ndarray] = field(default_factory=list)  # best-first corpus positions per query
    ranked_local: list[np.ndarray] = field(default_factory=list)


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Positions of the k highest strictly-positive scores, best first."""
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return idx[scores[idx] > 0]


def lexical_candidates(
    index: FieldBM25Index,
    queries_tokens: Sequence[Sequence[str]],
    query_locations: np.ndarray,
    location_positions: Mapping[int, np.ndarray],
    field_weights: Mapping[str, float],
    k_local: int = 200,
    k_global: int = 100,
    k_ceiling: int = 1000,
    extra_positions: Sequence[np.ndarray] | None = None,
    chunk: int = 128,
) -> LexicalResult:
    n_q = len(queries_tokens)
    cq, ci, st, sp_, sd, stot = [], [], [], [], [], []
    ranked_global: list[np.ndarray] = []
    ranked_local: list[np.ndarray] = []

    for start in range(0, n_q, chunk):
        end = min(start + chunk, n_q)
        fields = index.score_fields(queries_tokens[start:end])
        total = combine_fields(fields, field_weights)
        for r in range(end - start):
            q = start + r
            row = total[r]
            g = _top_k(row, k_ceiling)
            local_pos = location_positions.get(int(query_locations[q]), np.empty(0, dtype=np.int64))
            if local_pos.size:
                loc_top = local_pos[_top_k(row[local_pos], k_local)]
            else:
                loc_top = np.empty(0, dtype=np.int64)
            ranked_global.append(g)
            ranked_local.append(loc_top)

            parts = [loc_top, g[:k_global]]
            if extra_positions is not None and len(extra_positions[q]):
                parts.append(np.asarray(extra_positions[q], dtype=np.int64))
            pos = np.unique(np.concatenate(parts))
            cq.append(np.full(pos.size, q, dtype=np.int32))
            ci.append(pos.astype(np.int32))
            st.append(fields["title"][r, pos])
            sp_.append(fields["params"][r, pos])
            sd.append(fields["desc"][r, pos])
            stot.append(row[pos])

    cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.empty(0, dtype=dt)  # noqa: E731
    return LexicalResult(
        cand_q=cat(cq, np.int32),
        cand_i=cat(ci, np.int32),
        s_title=cat(st, np.float32),
        s_params=cat(sp_, np.float32),
        s_desc=cat(sd, np.float32),
        s_total=cat(stot, np.float32),
        ranked_global=ranked_global,
        ranked_local=ranked_local,
    )
