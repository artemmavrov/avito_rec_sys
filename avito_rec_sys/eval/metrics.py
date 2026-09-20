"""Recall@K, stratified breakdowns and the retrieval ceiling curve (§9).

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
    """Mean recall and query count per stratum label (§9.4). An aggregate
    hides exactly the slices that are most expensive to lose."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for qid, score in per_query.items():
        buckets[strata.get(qid, "unknown")].append(score)
    return {label: (sum(v) / len(v), len(v)) for label, v in buckets.items()}


def ceiling_curve(
    tours: Mapping[str, Mapping[str, Sequence[str]]],
    qrels: Mapping[str, set[str]],
    ks: Iterable[int] = (50, 300, 1000),
) -> dict[str, dict[int, float]]:
    """Recall@K for every tour and for the union of all tours' top-K (§9.5).

    The union row is the retrieval ceiling: `1 - union@1000` is what no tour
    finds at all, and `union@1000 - <final>@50` is the whole budget of the
    selection stages that come after retrieval.
    """
    ks = list(ks)
    curve: dict[str, dict[int, float]] = {}
    for name, preds in tours.items():
        curve[name] = {k: recall_at_k(preds, qrels, k) for k in ks}

    union_curve: dict[int, float] = {}
    for k in ks:
        union_preds: dict[str, list[str]] = {}
        for qid in qrels:
            seen: dict[str, None] = {}
            for preds in tours.values():
                for item in preds.get(qid, ())[:k]:
                    seen.setdefault(item, None)
            union_preds[qid] = list(seen)
        # union lists can exceed k; recall_at_k must not truncate them
        union_curve[k] = recall_at_k(union_preds, qrels, k=10**9)
    curve["union"] = union_curve
    return curve
