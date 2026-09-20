from avito_rec_sys.retrieval.slots import allocate_slots

CFG = {
    "output_k": 50,
    "small_local_pool_threshold": 1000,
    "small_pool_global_reserve_min": 5,
    "small_pool_global_reserve_max": 10,
    "large_pool_local_slots": 40,
}


def _ids(prefix, n):
    return [f"{prefix}{i}" for i in range(n)]


def test_no_local_items_all_global():
    out = allocate_slots([], _ids("g", 100), 0, CFG)
    assert out == _ids("g", 50)


def test_large_local_pool_is_40_plus_10():
    out = allocate_slots(_ids("l", 300), _ids("g", 100), 5000, CFG)
    assert len(out) == 50
    assert sum(i.startswith("l") for i in out) == 40
    assert sum(i.startswith("g") for i in out) == 10


def test_small_local_pool_keeps_all_local_and_reserve():
    out = allocate_slots(_ids("l", 20), _ids("g", 100), 20, CFG)
    assert len(out) == 50
    assert set(_ids("l", 20)) <= set(out)  # whole local pool kept


def test_small_local_pool_reserve_floor_of_5():
    out = allocate_slots(_ids("l", 200), _ids("g", 100), 200, CFG)
    assert sum(i.startswith("g") for i in out) == 5
    assert sum(i.startswith("l") for i in out) == 45


def test_global_list_exhausted_backfills_from_local():
    out = allocate_slots(_ids("l", 300), _ids("g", 3), 5000, CFG)
    assert len(out) == 50 and len(set(out)) == 50


def test_no_duplicates_when_lists_overlap():
    local = _ids("x", 60)
    out = allocate_slots(local, local + _ids("g", 60), 5000, CFG)
    assert len(out) == len(set(out)) == 50


def test_select_plain_is_top_k_by_score_only():
    import numpy as np
    import pandas as pd
    from types import SimpleNamespace

    from avito_rec_sys.inference.select import select_plain

    feats = pd.DataFrame({"q": [0, 0, 0, 1], "i": [0, 1, 2, 1]})
    items = SimpleNamespace(ids=["a", "b", "c"])
    out = select_plain(feats, np.array([0.1, 0.9, 0.5, 0.3]), items, SimpleNamespace(n=3), 2)
    assert out == [["b", "c"], ["b"], []]  # query 2 has no candidates -> empty list
