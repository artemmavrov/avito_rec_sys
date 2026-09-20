import numpy as np
import polars as pl

from avito_rec_sys.features.tables import click_centroids
from avito_rec_sys.retrieval.pool import merge_pool, nearby_positions

CFG = {"rrf_k": 60, "near_radius_km": 100, "near_weight": 1.0}


def test_nearby_keeps_order_and_drops_far_or_unknown():
    lat = np.array([55.75, 59.93, 55.80, np.nan, 55.70])  # Moscow, St Petersburg, Moscow, unknown, Moscow
    lon = np.array([37.62, 30.33, 37.50, 37.6, 37.70])
    ranked = np.array([4, 1, 3, 0, 2])
    near = nearby_positions(ranked, 55.75, 37.62, lat, lon, 100)
    assert near.tolist() == [4, 0, 2]  # far (SPb) and coordinate-less items are gone, order preserved


def test_nearby_without_query_coordinates_is_empty():
    assert nearby_positions(np.array([0, 1]), np.nan, np.nan, np.zeros(2), np.zeros(2), 100).size == 0
    assert nearby_positions(np.empty(0, dtype=np.int64), 1.0, 1.0, np.zeros(2), np.zeros(2), 100).size == 0


def test_merge_pool_lets_nearby_global_items_in_next_to_local_ones():
    local = np.arange(0, 6)  # six local items
    glob = np.array([100, 101, 102, 0, 1, 2])
    near = np.array([102])  # only 102 is close by
    merged = merge_pool(local, glob, near, CFG, 1000).tolist()
    assert len(merged) == len(set(merged)) == 9
    # a plain "local first" cut to 6 would hold no global item at all; the merge does
    assert 102 in merged[:6] or 100 in merged[:6]
    assert merged.index(102) < merged.index(101)  # the nearby one outranks the far one


def test_click_centroids_cover_locations_without_items():
    hist = pl.DataFrame({
        "search_location_id": [7, 7, 8, 9],
        "item_latitude": [10.0, 20.0, 30.0, None],
        "item_longitude": [1.0, 3.0, 5.0, 6.0],
    })
    c = click_centroids(hist)
    assert c[7] == (15.0, 2.0) and c[8] == (30.0, 5.0)
    assert 9 not in c  # its only click has no coordinates


def test_add_pool_features_ranks_gaps_and_row_order():
    import pandas as pd

    from avito_rec_sys.inference.neural_pipeline import POOL_COLUMNS, add_pool_features

    feats = pd.DataFrame({
        "q": np.array([0, 0, 0, 1, 1], dtype=np.int32),
        "i": np.array([5, 7, 9, 5, 8], dtype=np.int32),
        "dense_cos": [0.2, 0.9, 0.5, 0.1, 0.3],
        "sparse_score": [0.0, 1.0, 2.0, 0.0, 0.0],
        "colbert_maxsim": [3.0, 1.0, 2.0, 5.0, 4.0],
    })
    fused = [np.array([9, 5, 7]), np.array([8, 5])]  # query 1 got item 5 late; item 9 of query 0 leads
    out = add_pool_features(feats, fused, missing_rank=1000)
    assert list(out["i"]) == [5, 7, 9, 5, 8] and list(out["q"]) == [0, 0, 0, 1, 1]  # order untouched
    assert out["pool_rank"].tolist() == [1, 2, 0, 1, 0]
    assert out["dense_cos_rank_in_pool"].tolist() == [3, 1, 2, 2, 1]
    assert np.allclose(out["colbert_maxsim_minus_max"], [0, -2, -1, 0, -1])
    assert set(POOL_COLUMNS) <= set(out.columns)
    # a candidate that only came from the log sources is not in `fused`
    extra = add_pool_features(feats.iloc[:1].assign(i=np.int32(99)), fused, missing_rank=1000)
    assert extra["pool_rank"].iloc[0] == 1000
