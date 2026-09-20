import numpy as np

from avito_rec_sys.training.hard_negatives import mine_negatives_for_query, stratified_ranks


def test_stratified_ranks_spread_across_window():
    rng = np.random.RandomState(0)
    ranks = stratified_ranks(10, 200, 5, rng)
    assert len(ranks) == 5
    assert (ranks >= 10).all() and (ranks < 200).all()
    # roughly increasing across quantiles (not all clustered at one end)
    assert ranks[0] < ranks[-1]


def test_denoising_drops_identical_tower_text():
    ranked = np.array([0, 1, 2, 3, 4])
    tower = ["same text", "same text", "different", "different2", "different3"]
    rng = np.random.RandomState(0)
    out = mine_negatives_for_query(ranked, {0}, tower, "same text", 3, 0, 5, rng)
    assert 1 not in out  # position 1 has identical tower text to the positive
    assert 0 not in out  # position 0 is the positive itself


def test_returns_fewer_when_pool_too_small():
    ranked = np.array([0, 1])
    tower = ["a", "b"]
    rng = np.random.RandomState(0)
    out = mine_negatives_for_query(ranked, set(), tower, "positive", 5, 0, 2, rng)
    assert set(out) == {0, 1}


def test_empty_pool_returns_empty():
    ranked = np.array([0])
    tower = ["positive"]
    rng = np.random.RandomState(0)
    out = mine_negatives_for_query(ranked, set(), tower, "positive", 3, 0, 1, rng)
    assert out == []
