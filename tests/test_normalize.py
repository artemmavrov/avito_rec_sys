import polars as pl

from avito_rec_sys.data.normalize import add_normalized_query_column, normalize_query


def test_word_order_collapses():
    a = normalize_query("установка сэндвич панели")
    b = normalize_query("сэндвич панели установка")
    assert a == b


def test_case_and_punctuation_collapse():
    a = normalize_query("PlayStation 5 скупка")
    b = normalize_query("скупка playstation 5")
    assert a == b


def test_strips_punctuation_and_collapses_whitespace():
    assert normalize_query("баня,  на дровах!!") == "баня дровах на"


def test_none_and_empty():
    assert normalize_query(None) == ""
    assert normalize_query("") == ""
    assert normalize_query("   ") == ""


def test_vectorized_matches_scalar():
    queries = [
        "установка сэндвич панели",
        "сэндвич панели установка",
        "автоподбор",
        None,
    ]
    df = pl.DataFrame({"search_query": queries})
    out = add_normalized_query_column(df)["search_query_norm"].to_list()
    expected = [normalize_query(q) for q in queries]
    assert out == expected
