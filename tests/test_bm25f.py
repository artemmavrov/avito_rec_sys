import numpy as np

from avito_rec_sys.features.bm25f import FieldBM25Index, combine_fields

TITLES = [["баня", "дрова"], ["автоподбор", "осмотр"], ["баня", "сауна", "парение"], []]
PARAMS = [["услуга"], ["услуга"], ["услуга"], []]
DESC = [["топим", "баня"], ["осмотр", "авто"], ["сауна"], []]


def _index():
    return FieldBM25Index().fit({"title": TITLES, "params": PARAMS, "desc": DESC})


def test_matching_docs_score_higher_and_shorter_title_wins():
    s = _index().score_fields([["баня"]])["title"][0]
    assert s[0] > s[2] > 0  # shorter title with same tf scores higher
    assert s[1] == 0 and s[3] == 0


def test_oov_and_empty_query_score_zero():
    idx = _index()
    assert idx.score_fields([["нет_такого"]])["title"].sum() == 0
    assert idx.score_fields([[]])["title"].sum() == 0


def test_common_term_has_lower_idf_than_rare_term():
    s = _index().score_fields([["услуга"], ["автоподбор"]])
    assert s["params"][0].max() < s["title"][1].max()


def test_combine_fields_is_weighted_sum():
    scores = _index().score_fields([["баня"]])
    total = combine_fields(scores, {"title": 2.0, "params": 0.0, "desc": 1.0})
    np.testing.assert_allclose(total, 2 * scores["title"] + scores["desc"])


def test_batched_equals_single():
    idx = _index()
    both = idx.score_fields([["баня"], ["осмотр"]])["title"]
    np.testing.assert_allclose(both[0], idx.score_fields([["баня"]])["title"][0])
    np.testing.assert_allclose(both[1], idx.score_fields([["осмотр"]])["title"][0])
