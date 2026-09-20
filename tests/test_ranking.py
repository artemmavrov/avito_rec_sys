import numpy as np
import pandas as pd

from avito_rec_sys.pipeline.context import label_candidates
from avito_rec_sys.ranking.ranker import select_top_k


def test_select_top_k_orders_by_score_and_keeps_empty_queries():
    feats = pd.DataFrame({"q": [0, 0, 0, 2], "i": [0, 1, 2, 1]})
    scores = np.array([0.1, 0.9, 0.5, 0.3])
    out = select_top_k(feats, scores, ["a", "b", "c"], n_queries=3, k=2)
    assert out == [["b", "c"], [], ["b"]]


def test_label_candidates_marks_only_relevant_pairs():
    feats = pd.DataFrame({"q": [0, 0, 1, 1], "i": [0, 1, 0, 2]})
    y = label_candidates(feats, ["a", "b", "c"], {0: {"b"}, 1: {"c", "zzz"}})
    assert y.tolist() == [0, 1, 0, 1]
