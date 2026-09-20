"""Array views of a corpus and of a query set, aligned by row position.

Feature code indexes these by integer position (item position / query
position) so a whole candidate table is built with vectorized gathers.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import polars as pl
import scipy.sparse as sp

from avito_rec_sys.data.corpus import group_positions, tokens
from avito_rec_sys.data.params_parser import parse_params


def location_centroids(items: pl.DataFrame) -> dict[int, tuple[float, float]]:
    """Mean (lat, lon) per item location. Callers pass only items that are
    legal for the split at hand (no held-out positives)."""
    g = (
        items.drop_nulls(["item_latitude", "item_longitude"])
        .group_by("item_location_id")
        .agg(pl.col("item_latitude").mean().alias("lat"), pl.col("item_longitude").mean().alias("lon"))
    )
    return {int(l): (float(a), float(o)) for l, a, o in zip(g["item_location_id"], g["lat"], g["lon"])}


def click_centroids(history: pl.DataFrame) -> dict[int, tuple[float, float]]:
    """Mean (lat, lon) of the items users CLICKED after searching in a location, per search location.

    `location_centroids` only knows locations that hold corpus items; 15.7% of validation queries
    search in a location with none (their positives sit in neighbouring localities), so the
    geo features and the "nearby" candidate list had no anchor for them. The clicks in
    `history` (train only, never held-out pairs) give one for every location seen in train.
    """
    g = (
        history.drop_nulls(["search_location_id", "item_latitude", "item_longitude"])
        .group_by("search_location_id")
        .agg(pl.col("item_latitude").mean().alias("lat"), pl.col("item_longitude").mean().alias("lon"))
    )
    return {int(l): (float(a), float(o)) for l, a, o in zip(g["search_location_id"], g["lat"], g["lon"])}


class ItemTable:
    def __init__(self, items: pl.DataFrame, vocab: dict[str, int]):
        self.ids = items["item_id"].to_list()
        self.id2pos = {v: i for i, v in enumerate(self.ids)}
        self.n = len(self.ids)

        f = lambda c, dt=np.float64: items[c].fill_null(np.nan).to_numpy().astype(dt)  # noqa: E731
        self.loc = items["item_location_id"].fill_null(-1).to_numpy().astype(np.int64)
        self.cat = items["item_category_id"].fill_null(-1).to_numpy().astype(np.int64)
        self.microcat = items["item_microcat_id"].fill_null(-1).to_numpy().astype(np.int64)
        self.lat, self.lon = f("item_latitude"), f("item_longitude")
        self.price, self.rating, self.reviews = f("item_price"), f("item_rating"), f("item_rating_reviews_count")
        self.phone_hidden = items["item_is_phone_hidden"].fill_null(False).to_numpy().astype(np.int8)

        self.loc_positions = group_positions(self.loc)
        self.loc_count = {k: len(v) for k, v in self.loc_positions.items()}
        self.microcat_positions = group_positions(self.microcat)
        self.microcat_size = {k: len(v) for k, v in self.microcat_positions.items()}

        # duplicate clusters: same normalized title (title key, NOT the encoder tower text)
        title_norm = items["title_norm"].to_list()
        codes: dict[str, int] = {}
        self.cluster = np.empty(self.n, dtype=np.int64)
        self.title_to_positions: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(title_norm):
            self.cluster[i] = codes.setdefault(t, len(codes))
            self.title_to_positions[t].append(i)
        self.title_norm = title_norm
        self.cluster_size = np.bincount(self.cluster)[self.cluster]

        # log1p(price) z-score inside the microcategory
        lp = np.log1p(np.where(self.price > 0, self.price, np.nan))
        self.price_z = np.full(self.n, np.nan)
        for pos in self.microcat_positions.values():
            vals = lp[pos]
            ok = ~np.isnan(vals)
            if ok.sum() >= 5:
                sd = vals[ok].std()
                if sd > 0:
                    self.price_z[pos] = (vals - vals[ok].mean()) / sd

        self.title_tokens = tokens(items["title_lemmas"])
        self.title_str = items["title_lemmas"].to_list()

        raw_params = items["item_infm_params_text"].fill_null("").to_list()
        self.raw_params = raw_params
        self.param_keys = [frozenset(k for k, _ in parse_params(t)) for t in raw_params]

        self.title_bin = self._binary(self.title_tokens, vocab)
        self.desc_bin = self._binary(tokens(items["desc_lemmas"]), vocab)

    @staticmethod
    def _binary(docs: list[list[str]], vocab: dict[str, int]) -> sp.csr_matrix:
        rows, cols = [], []
        for d, doc in enumerate(docs):
            for idx in {vocab[t] for t in doc if t in vocab}:
                rows.append(d)
                cols.append(idx)
        return sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(docs), len(vocab))
        )


class QueryTable:
    def __init__(self, queries: pl.DataFrame, centroids: dict[int, tuple[float, float]]):
        self.n = queries.height
        self.norm = queries["search_query_norm"].to_list()
        self.lemma_tokens = tokens(queries["query_lemmas"])
        self.lemma_str = queries["query_lemmas"].to_list()
        self.loc = queries["search_location_id"].fill_null(-1).to_numpy().astype(np.int64)
        self.cat = queries["search_category"].fill_null(-1).to_numpy().astype(np.int64)
        latlon = [centroids.get(int(l), (np.nan, np.nan)) for l in self.loc]
        self.lat = np.array([a for a, _ in latlon], dtype=np.float64)
        self.lon = np.array([o for _, o in latlon], dtype=np.float64)

        raw = queries["search_infm_params_text"].fill_null("").to_list()
        self.has_params = np.array([1 if t.strip() else 0 for t in raw], dtype=np.int8)
        self.param_pairs = [parse_params(t) for t in raw]
        self.param_keys = [frozenset(k for k, _ in p) for p in self.param_pairs]
