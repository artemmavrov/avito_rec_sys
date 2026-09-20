"""Recall@K and its stratified breakdown.

Predictions are `{query_id: ranked list of item_id}` and relevance is
`{query_id: set of item_id}`. Order inside a list matters only for
truncation to the top-K -- the target metric itself is order-insensitive
within the 50 returned items.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence


def per_query_recall(
    predictions: Mapping[str, Sequence[str]], qrels: Mapping[str, set[str]], k: int
) -> dict[str, float]:
    """|top-k ∩ relevant| / |relevant| per query. Queries missing from
    `predictions` score 0 (an empty answer), never silently skipped."""
    out: dict[str, float] = {}
    for qid, relevant in qrels.items():
        if not relevant:
            continue
        top = set(predictions.get(qid, ())[:k])
        out[qid] = len(top & relevant) / len(relevant)
    return out


def recall_at_k(predictions: Mapping[str, Sequence[str]], qrels: Mapping[str, set[str]], k: int) -> float:
    scores = per_query_recall(predictions, qrels, k)
    return sum(scores.values()) / len(scores) if scores else 0.0


def stratified_recall(
    per_query: Mapping[str, float], strata: Mapping[str, str]
) -> dict[str, tuple[float, int]]:
    """Mean recall and query count per stratum label. An aggregate
    hides exactly the slices that are most expensive to lose."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for qid, score in per_query.items():
        buckets[strata.get(qid, "unknown")].append(score)
    return {label: (sum(v) / len(v), len(v)) for label, v in buckets.items()}
