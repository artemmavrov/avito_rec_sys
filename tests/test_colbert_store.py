import numpy as np

from avito_rec_sys.retrieval.colbert_store import ColbertStore, pair_maxsim


def _unit(rng, n, d=8):
    x = rng.randn(n, d)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def test_pair_maxsim_matches_brute_force(tmp_path):
    rng = np.random.RandomState(0)
    docs = [_unit(rng, n) for n in (3, 5, 1, 4)]
    store = ColbertStore.build(tmp_path / "c.bin", [d.astype(np.float16) for d in docs], 8)
    queries = [_unit(rng, 2), _unit(rng, 4), np.zeros((0, 8), np.float32)]
    q_idx = np.array([0, 1, 2, 1, 0, 2])
    i_idx = np.array([3, 0, 1, 3, 0, 2])  # unsorted on purpose: results must land on the right pair
    got = pair_maxsim(store, queries, q_idx, i_idx)
    for k, (q, i) in enumerate(zip(q_idx, i_idx)):
        d = store[int(i)].astype(np.float32)
        want = 0.0 if queries[q].size == 0 else (queries[q] @ d.T).max(axis=1).sum()
        assert np.isclose(got[k], want, atol=1e-6)
