from avito_rec_sys.eval.metrics import ceiling_curve, per_query_recall, recall_at_k, stratified_recall


def test_task_description_example():
    # task_description.md: A 1/1, B 1/2, C 0/1 -> 0.5
    qrels = {"A": {"a1"}, "B": {"b1", "b2"}, "C": {"c1"}}
    preds = {"A": ["a1", "x"], "B": ["b1", "x"], "C": ["x", "y"]}
    assert recall_at_k(preds, qrels, 50) == 0.5


def test_truncation_to_k():
    qrels = {"A": {"a1"}}
    preds = {"A": ["x", "y", "a1"]}
    assert recall_at_k(preds, qrels, 2) == 0.0
    assert recall_at_k(preds, qrels, 3) == 1.0


def test_missing_prediction_scores_zero_not_skipped():
    qrels = {"A": {"a1"}, "B": {"b1"}}
    preds = {"A": ["a1"]}
    assert recall_at_k(preds, qrels, 50) == 0.5


def test_stratified():
    pq = {"A": 1.0, "B": 0.0, "C": 0.5}
    out = stratified_recall(pq, {"A": "seen", "B": "new", "C": "new"})
    assert out["seen"] == (1.0, 1)
    assert out["new"] == (0.25, 2)


def test_ceiling_curve_union_beats_each_tour():
    qrels = {"A": {"a1", "a2"}}
    tours = {
        "bm25": {"A": ["a1", "x"]},
        "dense": {"A": ["y", "a2"]},
    }
    curve = ceiling_curve(tours, qrels, ks=(2,))
    assert curve["bm25"][2] == 0.5
    assert curve["dense"][2] == 0.5
    assert curve["union"][2] == 1.0
    assert per_query_recall(tours["bm25"], qrels, 2) == {"A": 0.5}
