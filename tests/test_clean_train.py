import polars as pl

from avito_rec_sys.data.clean_train import clean_train


def _toy_df():
    # query "a b" and "b a" are the same normalized query and should merge.
    rows = [
        {"search_query": "a b", "item_id": "1" * 16, "item_title_raw": "t1", "item_infm_params_text": ""},
        {"search_query": "b a", "item_id": "1" * 16, "item_title_raw": "t1", "item_infm_params_text": ""},
        {"search_query": "a b", "item_id": "2" * 16, "item_title_raw": "t1", "item_infm_params_text": ""},
        {"search_query": "a b", "item_id": "3" * 16, "item_title_raw": "t3", "item_infm_params_text": ""},
    ]
    return pl.DataFrame(rows)


def test_step1_dedups_exact_normalized_pairs():
    df = _toy_df()
    cleaned, stats = clean_train(df, cap_per_query=16)
    # rows 0 and 1 are the same (query_norm, item_id) pair -> collapse to 1
    assert stats.n_start == 4
    assert stats.n_after_pair_dedup == 3


def test_step2_dedups_identical_tower_text_within_query():
    df = _toy_df()
    cleaned, stats = clean_train(df, cap_per_query=16)
    # item 1 and item 2 share the same title/params (identical tower text)
    # under the same normalized query -> one of them is dropped.
    assert stats.n_after_tower_dedup == 2
    assert cleaned["search_query_norm"].n_unique() == 1


def test_step3_caps_per_query():
    rows = [
        {
            "search_query": "q",
            "item_id": f"{i:016x}",
            "item_title_raw": f"title {i}",
            "item_infm_params_text": "",
        }
        for i in range(20)
    ]
    df = pl.DataFrame(rows)
    cleaned, stats = clean_train(df, cap_per_query=5, seed=42)
    assert stats.n_after_cap == 5
    assert cleaned.height == 5


def test_no_query_is_dropped_entirely_by_capping():
    rows = []
    for q in range(3):
        for i in range(30):
            rows.append(
                {
                    "search_query": f"query{q}",
                    "item_id": f"{q}{i:015x}",
                    "item_title_raw": f"title {q} {i}",
                    "item_infm_params_text": "",
                }
            )
    df = pl.DataFrame(rows)
    cleaned, stats = clean_train(df, cap_per_query=5, seed=42)
    assert stats.n_queries == 3
    assert cleaned.height == 15
