"""Signals mined from the training logs (§1 "лог-сигналы").

Built ONLY from the training part of a split, never from held-out queries.

  q -> item      items users chose for the same normalized query text
  title bridge   titles of those items; corpus items with the same normalized
                 title are candidates too. Benchmark positives are mostly cold
                 (unseen items), but their titles repeat those of warm items,
                 which is exactly what 48.5% duplicate-title clusters mean.
  p(microcat|q)  category prior: exact for a seen text, otherwise averaged
                 token-level distribution (the only prior available for the
                 ~61% of queries whose text was never seen).
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import polars as pl

from avito_rec_sys.data.normalize import normalize_title_expr


class QueryLogs:
    def __init__(self, train_part: pl.DataFrame):
        df = train_part.with_columns(normalize_title_expr("item_title_raw").alias("_title_norm"))

        self.items: dict[str, Counter] = defaultdict(Counter)
        self.titles: dict[str, Counter] = defaultdict(Counter)
        self.microcats: dict[str, Counter] = defaultdict(Counter)
        for text, item, title, mc in zip(
            df["search_query_norm"].to_list(),
            df["item_id"].to_list(),
            df["_title_norm"].to_list(),
            df["item_microcat_id"].to_list(),
        ):
            self.items[text][item] += 1
            self.titles[text][title] += 1
            self.microcats[text][mc] += 1

        # token -> microcat counts, for the unseen-text backoff
        self.token_microcats: dict[str, Counter] = defaultdict(Counter)
        for text, mcs in self.microcats.items():
            for tok in set(text.split()):
                self.token_microcats[tok].update(mcs)
        self._token_totals = {t: sum(c.values()) for t, c in self.token_microcats.items()}

    def seen(self, text: str) -> bool:
        return text in self.microcats

    def p_microcat(self, text: str, microcat: int) -> float:
        """Exact P(microcat | text) if the text was seen, else the mean of
        P(microcat | token) over its known tokens (0 if none are known)."""
        exact = self.microcats.get(text)
        if exact:
            return exact.get(microcat, 0) / sum(exact.values())
        probs = [
            self.token_microcats[t].get(microcat, 0) / self._token_totals[t]
            for t in text.split()
            if t in self._token_totals
        ]
        return float(np.mean(probs)) if probs else 0.0


def log_candidate_positions(queries, items, logs: QueryLogs, cap_per_title: int = 20, max_total: int = 150):
    """Corpus positions suggested by the logs for each query (q->item + title bridge).

    For a bridge title shared by many corpus items, same-location items are
    taken first: geo is what separates members of a duplicate cluster (§6.3).
    """
    out: list[np.ndarray] = []
    for q in range(queries.n):
        text = queries.norm[q]
        if not logs.seen(text):
            out.append(np.empty(0, dtype=np.int64))
            continue
        pos: dict[int, None] = {}
        for item_id in logs.items[text]:
            p = items.id2pos.get(item_id)
            if p is not None:
                pos[p] = None
        for title, _ in logs.titles[text].most_common():
            members = items.title_to_positions.get(title, ())
            local = [p for p in members if items.loc[p] == queries.loc[q]]
            rest = [p for p in members if items.loc[p] != queries.loc[q]]
            for p in (local + rest)[:cap_per_title]:
                pos[p] = None
            if len(pos) >= max_total:
                break
        out.append(np.fromiter(pos, dtype=np.int64)[:max_total] if pos else np.empty(0, dtype=np.int64))
    return out
