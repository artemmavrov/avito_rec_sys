"""Row thinning of the ranker's training pools.

The cross-encoder must score every row that CatBoost trains on, which is the
expensive part of stage 11. Thinning each pool from ~300 to `rows_per_query`
rows BEFORE scoring lets far more queries fit in the same budget (8000 x 130
instead of 3500 x 300).

Rules, chosen so the ranker still learns what separates the positive from the
confusable candidates it will meet at inference:
  * every positive is kept;
  * the remaining budget goes half to the HARDEST negatives (highest
    `hardness`, i.e. the ones the retrieval funnel ranks closest to a
    positive) and half to a uniform sample of the rest;
  * queries at or after `protect_from_q` are left untouched -- they form the
    early-stopping set, which must keep full pools (the distribution the
    ranker sees at inference).
YetiRank is a within-query pairwise objective, so dropping easy negatives
changes the class balance but not what a query's ranking has to get right.
Pool-relative features are computed on the full pool before this runs.
"""

from __future__ import annotations

import numpy as np


def thin_pools(
    q: np.ndarray,
    y: np.ndarray,
    hardness: np.ndarray,
    rows_per_query: int,
    protect_from_q: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Boolean keep-mask over rows. `q` must be sorted (rows of one query contiguous)."""
    keep = np.zeros(len(q), dtype=bool)
    keep[q >= protect_from_q] = True
    bounds = np.flatnonzero(np.diff(q, prepend=q[0] - 1))
    ends = np.append(bounds[1:], len(q))
    for b, e in zip(bounds, ends):
        if q[b] >= protect_from_q:
            continue
        idx = np.arange(b, e)
        pos = idx[y[b:e] == 1]
        neg = idx[y[b:e] == 0]
        keep[pos] = True
        budget = rows_per_query - len(pos)
        if budget <= 0:
            continue
        if len(neg) <= budget:
            keep[neg] = True
            continue
        n_hard = budget // 2
        order = neg[np.argsort(-hardness[neg], kind="stable")]
        hard, rest = order[:n_hard], order[n_hard:]
        keep[hard] = True
        keep[rng.choice(rest, size=budget - n_hard, replace=False)] = True
    return keep
