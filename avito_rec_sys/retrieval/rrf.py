"""Weighted Reciprocal Rank Fusion across retrieval tours (§6.1).

    score(d) = sum_i  w_i / (k + rank_i(d))       (rank is 1-based)

k = 200 instead of the canonical 60: k=60 is tuned for precision at the top,
but the objective here is recall inside 50 slots, and a large k flattens the
gap between ranks so an item that is 4th in one tour and 300th in another
still beats one that is 1st in a single tour only.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from avito_rec_sys.eval.metrics import recall_at_k


def rrf_fuse(
    tours: Mapping[str, Sequence[str]],
    weights: Mapping[str, float],
    k: int = 200,
    top_k: int | None = None,
) -> list[str]:
    """Fuse best-first ranked lists for ONE query. Tours absent from
    `weights` get weight 1.0. Ties break by first appearance (stable)."""
    scores: dict[str, float] = {}
    for name, ranked in tours.items():
        w = weights.get(name, 1.0)
        if w == 0.0:
            continue
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
    fused = sorted(scores, key=scores.__getitem__, reverse=True)
    return fused[:top_k] if top_k else fused


def tune_rrf_weights(
    tours_per_query: Mapping[str, Mapping[str, Sequence[str]]],
    qrels: Mapping[str, set[str]],
    tour_names: Sequence[str],
    k_rrf: int = 200,
    pool_k: int = 1000,
    grid: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    n_rounds: int = 2,
) -> dict[str, float]:
    """Coordinate ascent on the fused pool's Recall@pool_k (CPU only, §6.1).

    Full grid search is |grid|^n_tours; coordinate ascent is n_rounds *
    n_tours * |grid| evaluations and lands on the same optimum for this
    smooth, low-dimensional objective in practice.
    `tours_per_query[qid][tour_name]` is that tour's ranked list.
    """
    weights = {name: 1.0 for name in tour_names}

    def score(w: Mapping[str, float]) -> float:
        preds = {qid: rrf_fuse(t, w, k_rrf, pool_k) for qid, t in tours_per_query.items()}
        return recall_at_k(preds, qrels, pool_k)

    best = score(weights)
    for _ in range(n_rounds):
        for name in tour_names:
            for value in grid:
                if value == weights[name]:
                    continue
                trial = {**weights, name: value}
                if not any(trial.values()):
                    continue
                s = score(trial)
                if s > best:
                    best, weights = s, trial
    return weights
