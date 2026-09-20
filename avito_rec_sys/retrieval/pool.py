"""Candidate-pool construction: merge the local and global rankings with a geo-aware third list.

The naive pool is "the whole local ranking first, then the global one", cut to the pool size.
On the validation hold-out every positive the local part could reach was already inside 300
(1 miss in 2 683), while 22.6% of positives sit outside the search location and were pushed
out by the local list. Merging by reciprocal rank instead, with a third list of global
candidates lying within a radius of the query location, keeps the same budget and recovers
most of them (Recall of a 300-pool 0.924 -> 0.973 together with the click-based location
centroids of `features/tables.py`).
"""

from __future__ import annotations

import numpy as np

from avito_rec_sys.features.assemble import haversine_km
from avito_rec_sys.retrieval.rrf import rrf_fuse


def nearby_positions(
    ranked_global: np.ndarray,
    q_lat: float,
    q_lon: float,
    item_lat: np.ndarray,
    item_lon: np.ndarray,
    radius_km: float,
) -> np.ndarray:
    """The items of `ranked_global` within `radius_km` of the query location, order preserved.
    Empty when the query location has no coordinates or an item has none."""
    if len(ranked_global) == 0 or np.isnan(q_lat) or np.isnan(q_lon):
        return np.empty(0, dtype=np.int64)
    ranked_global = np.asarray(ranked_global, dtype=np.int64)
    dist = haversine_km(q_lat, q_lon, item_lat[ranked_global], item_lon[ranked_global])
    return ranked_global[dist < radius_km]  # NaN distances compare False and drop out


def merge_pool(local: np.ndarray, glob: np.ndarray, near: np.ndarray, pool_cfg: dict, top_k: int) -> np.ndarray:
    """Reciprocal-rank merge of the three best-first lists into one ranking of at most `top_k`."""
    fused = rrf_fuse(
        {"local": local, "global": glob, "near": near},
        {"local": 1.0, "global": 1.0, "near": pool_cfg["near_weight"]},
        pool_cfg["rrf_k"],
        top_k,
    )
    return np.array(fused, dtype=np.int64)
