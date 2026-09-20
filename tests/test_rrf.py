from avito_rec_sys.retrieval.rrf import rrf_fuse, tune_rrf_weights


def test_agreement_beats_single_tour_top():
    tours = {"a": ["x", "y", "z"], "b": ["y", "x", "w"]}
    fused = rrf_fuse(tours, {}, k=200)
    assert set(fused[:2]) == {"x", "y"}  # both tours agree on x,y; z/w only one tour


def test_weight_zero_disables_tour():
    tours = {"a": ["x"], "b": ["y"]}
    assert rrf_fuse(tours, {"a": 0.0, "b": 1.0}) == ["y"]


def test_top_k_truncates():
    tours = {"a": [str(i) for i in range(10)]}
    assert len(rrf_fuse(tours, {}, top_k=3)) == 3


def test_large_k_flattens_rank_gap():
    # with small k an item ranked 1st in one tour beats one ranked 4th/5th in two
    tours = {"a": ["solo", "p", "q", "both"], "b": ["r", "s", "t", "u", "both"]}
    assert rrf_fuse(tours, {}, k=1)[0] == "solo"
    assert rrf_fuse(tours, {}, k=200)[0] == "both"


def test_tuning_downweights_useless_tour():
    qrels = {f"q{i}": {f"good{i}"} for i in range(20)}
    tours_per_query = {
        qid: {
            "good_tour": [f"good{qid[1:]}", "n1", "n2"],
            "junk_tour": ["j1", "j2", "j3"],
        }
        for qid in qrels
    }
    w = tune_rrf_weights(tours_per_query, qrels, ["good_tour", "junk_tour"], pool_k=1)
    assert w["good_tour"] >= w["junk_tour"]
