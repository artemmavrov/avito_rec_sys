import numpy as np

from avito_rec_sys.training.thinning import thin_pools


def _pools(n_queries=6, per=300, seed=0):
    rng = np.random.RandomState(seed)
    q = np.repeat(np.arange(n_queries), per)
    y = np.zeros(len(q), dtype=np.int8)
    for k in range(n_queries):
        y[k * per + rng.randint(per)] = 1
    hard = rng.rand(len(q))
    return q, y, hard


def test_keeps_all_positives_and_hits_row_budget():
    q, y, hard = _pools()
    keep = thin_pools(q, y, hard, 130, protect_from_q=99, rng=np.random.RandomState(1))
    assert keep[y == 1].all()
    for k in range(6):
        assert keep[q == k].sum() == 130


def test_half_of_negatives_are_the_hardest():
    q, y, hard = _pools(n_queries=1)
    keep = thin_pools(q, y, hard, 130, protect_from_q=99, rng=np.random.RandomState(1))
    neg = np.flatnonzero(y == 0)
    hardest = neg[np.argsort(-hard[neg])[: (130 - 1) // 2]]
    assert keep[hardest].all()


def test_protected_queries_keep_full_pools():
    q, y, hard = _pools()
    keep = thin_pools(q, y, hard, 130, protect_from_q=4, rng=np.random.RandomState(1))
    assert keep[q >= 4].all()
    assert keep[q == 0].sum() == 130


def test_small_pool_untouched_and_deterministic():
    q = np.array([0, 0, 0])
    y = np.array([1, 0, 0], dtype=np.int8)
    h = np.array([0.1, 0.2, 0.3])
    assert thin_pools(q, y, h, 130, 99, np.random.RandomState(0)).all()
    q, y, hard = _pools()
    a = thin_pools(q, y, hard, 130, 99, np.random.RandomState(5))
    b = thin_pools(q, y, hard, 130, 99, np.random.RandomState(5))
    assert (a == b).all()
