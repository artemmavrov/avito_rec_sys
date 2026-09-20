"""Feature bank for (query, candidate) pairs (§4).

`build_features` takes the candidate pairs from the lexical tour and returns
one pandas frame with the non-neural part of the bank. Neural columns
(dense_cos, sparse_score, colbert_maxsim, ce_logit, cos_query_microcat_
centroid) are added by the GPU stages through `extra_columns` and appended to
`FEATURE_COLUMNS` there.

Deviations from §4's list, all cheap, all in the non-neural part:
  + query_seen_in_train, log_qitem_count, log_bridge_count  (log signals, §1)
  + bm25f_rank_in_pool, bm25f_over_max                       (pool-relative)
  - the 5 neural columns listed above (added later)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from avito_rec_sys.features.logs import QueryLogs
from avito_rec_sys.features.tables import ItemTable, QueryTable
from avito_rec_sys.retrieval.lexical import LexicalResult

FEATURE_COLUMNS = [
    # text
    "bm25_title", "bm25_params", "bm25_desc", "bm25f_total", "ngram_sim",
    "qcov_title", "qcov_desc", "title_prefix_match",
    "bm25f_rank_in_pool", "bm25f_over_max",
    # geo
    "loc_match", "haversine_km", "log1p_dist", "item_loc_freq", "is_query_loc_empty",
    "dist_rank_in_pool", "dist_pct_in_dup_cluster",
    # categories
    "category_match", "microcat_size", "p_microcat_given_query", "microcat_rank", "query_seen_in_train",
    # params
    "n_matched_param_pairs", "jaccard_param_keys", "has_query_params",
    # quality
    "item_rating", "log1p_reviews", "price", "price_zscore_in_microcat", "is_phone_hidden",
    # duplicates
    "dup_cluster_size", "geo_rank_in_dup_cluster", "is_unique_title",
    # logs
    "log_qitem_count", "log_bridge_count",
]

_EARTH_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * _EARTH_KM * np.arcsin(np.sqrt(a))


def _trigrams(s: str) -> frozenset:
    s = f" {s} "
    return frozenset(s[i : i + 3] for i in range(len(s) - 2))


def _coverage(qm, item_bin, q, i, n_query_tokens, chunk=200_000) -> np.ndarray:
    """Share of the query's distinct tokens present in the item field."""
    out = np.zeros(len(q), dtype=np.float32)
    for s in range(0, len(q), chunk):
        e = s + chunk
        overlap = np.asarray(qm[q[s:e]].multiply(item_bin[i[s:e]]).sum(axis=1)).ravel()
        out[s:e] = overlap / np.maximum(n_query_tokens[q[s:e]], 1)
    return out


def build_features(
    cand: LexicalResult,
    items: ItemTable,
    queries: QueryTable,
    logs: QueryLogs,
    index,
) -> pd.DataFrame:
    q, i = cand.cand_q.astype(np.int64), cand.cand_i.astype(np.int64)
    n = len(q)
    df = pd.DataFrame({"q": q, "i": i})

    # ---- text
    df["bm25_title"], df["bm25_params"], df["bm25_desc"] = cand.s_title, cand.s_params, cand.s_desc
    df["bm25f_total"] = cand.s_total
    g = df.groupby("q")["bm25f_total"]
    df["bm25f_rank_in_pool"] = g.rank(ascending=False, method="min")
    df["bm25f_over_max"] = df["bm25f_total"] / g.transform("max").replace(0, np.nan)

    qm = index.query_matrix(queries.lemma_tokens)
    n_qtok = np.array([len(set(t)) for t in queries.lemma_tokens])
    df["qcov_title"] = _coverage(qm, items.title_bin, q, i, n_qtok)
    df["qcov_desc"] = _coverage(qm, items.desc_bin, q, i, n_qtok)

    q_tri = [_trigrams(s) for s in queries.lemma_str]
    item_tri: dict[int, frozenset] = {}
    ngram = np.zeros(n, dtype=np.float32)
    prefix = np.zeros(n, dtype=np.float32)
    qitem = np.zeros(n, dtype=np.float32)
    bridge = np.zeros(n, dtype=np.float32)
    for k in range(n):
        qi, ii = q[k], i[k]
        t = item_tri.get(ii)
        if t is None:
            t = item_tri[ii] = _trigrams(items.title_str[ii])
        qt = q_tri[qi]
        union = len(qt | t)
        ngram[k] = len(qt & t) / union if union else 0.0
        qtok, ttok = queries.lemma_tokens[qi], items.title_tokens[ii]
        m = 0
        while m < len(qtok) and m < len(ttok) and qtok[m] == ttok[m]:
            m += 1
        prefix[k] = m / len(qtok) if qtok else 0.0
        text = queries.norm[qi]
        if logs.seen(text):
            qitem[k] = logs.items[text].get(items.ids[ii], 0)
            bridge[k] = logs.titles[text].get(items.title_norm[ii], 0)
    df["ngram_sim"], df["title_prefix_match"] = ngram, prefix
    df["log_qitem_count"], df["log_bridge_count"] = qitem, bridge

    # ---- geo
    i_loc, q_loc = items.loc[i], queries.loc[q]
    df["loc_match"] = (i_loc == q_loc).astype(np.int8)
    dist = haversine_km(queries.lat[q], queries.lon[q], items.lat[i], items.lon[i])
    df["haversine_km"] = dist
    df["log1p_dist"] = np.log1p(dist)
    df["item_loc_freq"] = [items.loc_count.get(int(l), 0) for l in i_loc]
    df["is_query_loc_empty"] = np.array([items.loc_count.get(int(l), 0) == 0 for l in q_loc], dtype=np.int8)
    df["dist_rank_in_pool"] = df.groupby("q")["haversine_km"].rank(method="average")
    df["_cluster"] = items.cluster[i]
    by_cluster = df.groupby(["q", "_cluster"])["haversine_km"]
    df["dist_pct_in_dup_cluster"] = by_cluster.rank(pct=True)
    df["geo_rank_in_dup_cluster"] = by_cluster.rank(method="average")

    # ---- categories
    df["category_match"] = (items.cat[i] == queries.cat[q]).astype(np.int8)
    df["microcat_size"] = [items.microcat_size.get(int(m), 0) for m in items.microcat[i]]
    pm_cache: dict[tuple[int, int], float] = {}
    pm = np.zeros(n, dtype=np.float32)
    mcs = items.microcat[i]
    for k in range(n):
        key = (int(q[k]), int(mcs[k]))
        v = pm_cache.get(key)
        if v is None:
            v = pm_cache[key] = logs.p_microcat(queries.norm[q[k]], int(mcs[k]))
        pm[k] = v
    df["p_microcat_given_query"] = pm
    df["microcat_rank"] = df.groupby("q")["p_microcat_given_query"].rank(ascending=False, method="dense")
    df["query_seen_in_train"] = np.array([logs.seen(queries.norm[x]) for x in q], dtype=np.int8)

    # ---- params
    df["has_query_params"] = queries.has_params[q]
    n_matched = np.zeros(n, dtype=np.float32)
    jac = np.zeros(n, dtype=np.float32)
    with_params = np.nonzero(queries.has_params[q])[0]
    for k in with_params:
        qi, ii = q[k], i[k]
        raw = items.raw_params[ii]
        n_matched[k] = sum(1 for key, val in queries.param_pairs[qi] if f"{key} {val}" in raw)
        a, b = queries.param_keys[qi], items.param_keys[ii]
        jac[k] = len(a & b) / len(a | b) if (a or b) else 0.0
    df["n_matched_param_pairs"], df["jaccard_param_keys"] = n_matched, jac

    # ---- quality
    df["item_rating"] = items.rating[i]
    df["log1p_reviews"] = np.log1p(np.nan_to_num(items.reviews[i]))
    df["price"] = items.price[i]
    df["price_zscore_in_microcat"] = items.price_z[i]
    df["is_phone_hidden"] = items.phone_hidden[i]

    # ---- duplicates
    df["dup_cluster_size"] = items.cluster_size[i]
    df["is_unique_title"] = (items.cluster_size[i] == 1).astype(np.int8)

    return df.drop(columns=["_cluster"])
