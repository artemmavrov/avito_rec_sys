"""Stratified hard-negative mining + denoising (§5.3).

Negatives are sampled by RANK QUANTILE within [rank_min, rank_max] of a
zero-shot (or current) retriever's ranking, not "top-N hardest" -- a
sharpest-only sample teaches the model to split near-duplicates and nothing
else about the overall preference structure.

Denoising: any candidate whose full item-tower text matches the positive's
tower text is dropped -- if the encoder input is identical, the pair is
unsolvable and training on it would teach the model to separate the
inseparable (§5.3, same key as §5.2 step 2 dedup).
"""

from __future__ import annotations

import numpy as np


def stratified_ranks(rank_min: int, rank_max: int, n: int, rng: np.random.RandomState) -> np.ndarray:
    """n rank positions spread across quantiles of [rank_min, rank_max)."""
    edges = np.linspace(rank_min, rank_max, n + 1)
    return np.array([rng.randint(int(edges[i]), max(int(edges[i + 1]), int(edges[i]) + 1)) for i in range(n)])


def mine_negatives_for_query(
    ranked_positions: np.ndarray,
    positive_positions: set[int],
    tower_text_by_position: list[str],
    positive_tower_text: str,
    n_negatives: int,
    rank_min: int,
    rank_max: int,
    rng: np.random.RandomState,
) -> list[int]:
    """Pick n_negatives corpus positions from `ranked_positions[rank_min:rank_max]`,
    excluding positives and anything denoised against the positive's tower text."""
    window = ranked_positions[rank_min : min(rank_max, len(ranked_positions))]
    pool = [
        p
        for p in window
        if p not in positive_positions and tower_text_by_position[p] != positive_tower_text
    ]
    if not pool:
        return []
    if len(pool) <= n_negatives:
        return list(pool)
    ranks = stratified_ranks(0, len(pool), n_negatives, rng)
    return [pool[r] for r in np.clip(ranks, 0, len(pool) - 1)]
