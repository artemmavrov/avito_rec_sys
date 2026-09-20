import polars as pl

from avito_rec_sys.data.split import ITEM_COLUMNS, SEEN, build_corpus, cold_start_ratio, make_splits


def _toy_train(n_texts: int = 60):
    rows = []
    for t in range(n_texts):
        for loc in range(3):
            for k in range(2):
                rows.append(
                    {
                        "search_query": f"query {t}" if loc else f"{t} query",  # two spellings, one normalized text
                        "search_location_id": loc,
                        "search_is_delivery_search": 0,
                        "search_infm_params_text": "",
                        "search_category": 1,
                        "item_id": f"{t:08x}{loc:04x}{k:04x}",
                        "item_title_raw": f"title {t}",
                        "item_description_raw": "d",
                        "item_infm_params_text": "",
                        "item_category_id": 1,
                        "item_microcat_id": 1,
                        "item_price": 1.0,
                        "item_rating": 5.0,
                        "item_rating_reviews_count": 1.0,
                        "item_location_id": loc,
                        "item_latitude": 55.0,
                        "item_longitude": 37.0,
                        "item_is_phone_hidden": False,
                        "item_is_message_forbidden": False,
                    }
                )
    return pl.DataFrame(rows)


def _split(**kw):
    return make_splits(_toy_train(), n_validation_queries=10, n_ranker_queries=8, seen_fraction=0.5, seed=0, **kw)


def test_positives_are_cold_to_train_part():
    res = _split()
    train_items = set(res.train_part["item_id"].to_list())
    for held in (res.val, res.ranker):
        assert not (set(held.qrels["item_id"].to_list()) & train_items)


def test_new_texts_never_in_train_seen_texts_always_are():
    res = _split()
    train_texts = set(res.train_part["search_query_norm"].to_list())
    q = res.val.queries
    for text in q.filter(pl.col("stratum") == "new")["search_query_norm"].to_list():
        assert text not in train_texts  # spelling variants included: same normalized key
    for text in q.filter(pl.col("stratum") == SEEN)["search_query_norm"].to_list():
        assert text in train_texts


def test_one_query_per_text_and_disjoint_holdouts():
    res = _split()
    v, r = res.val.queries, res.ranker.queries
    assert v["search_query_norm"].n_unique() == v.height == 10
    assert r.height == 8
    assert not (set(v["search_query_norm"].to_list()) & set(r["search_query_norm"].to_list()))
    assert not (set(res.val.qrels["item_id"].to_list()) & set(res.ranker.qrels["item_id"].to_list()))
    assert (v["stratum"] == SEEN).sum() == 5  # seen_fraction=0.5


def test_query_ids_are_16_chars_and_unique():
    res = _split()
    ids = res.val.queries["query_id"].to_list() + res.ranker.queries["query_id"].to_list()
    assert all(len(i) == 16 for i in ids)
    assert len(set(ids)) == len(ids)


def test_corpus_contains_positives_and_cold_ratio():
    res = _split()
    empty_bench = _toy_train().head(0).select(ITEM_COLUMNS)
    corpus = build_corpus(res.val.positive_items, empty_bench)
    assert set(res.val.qrels["item_id"].to_list()) <= set(corpus["item_id"].to_list())
    assert cold_start_ratio(corpus, res.train_part) == 1.0


def test_positive_items_order_is_deterministic():
    """The master corpus (and with it every encoded store) is aligned by row order, so the order of
    the positives must not depend on hash-table iteration order."""
    a, b = _split(), _split()
    assert a.val.positive_items["item_id"].to_list() == b.val.positive_items["item_id"].to_list()
    assert a.ranker.qrels["item_id"].to_list() == b.ranker.qrels["item_id"].to_list()
