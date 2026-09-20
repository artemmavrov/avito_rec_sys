"""Weighted Reciprocal Rank Fusion across retrieval tours.

    score(d) = sum_i  w_i / (k + rank_i(d))       (rank is 1-based)

k = 200 instead of the canonical 60: k=60 is tuned for precision at the top,
but the objective here is recall inside 50 slots, and a large k flattens the
gap between ranks so an item that is 4th in one tour and 300th in another
still beats one that is 1st in a single tour only.
"""

from __future__ import annotations

from typing import Mapping, Sequence


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
