import numpy as np
import scipy.sparse as sp

from avito_rec_sys.retrieval.tours import dense_tour, sparse_tour


def _unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def test_dense_tour_global_and_local():
    rng = np.random.RandomState(0)
    docs = _unit(rng.randn(50, 16))
    queries = docs[[3, 20]] + 0.01  # nearly identical to docs 3 and 20
    locs = np.array([1, 2])
    loc_pos = {1: np.array([3, 4, 5]), 2: np.array([30, 31])}  # doc 20 is NOT in query 1's location
    glob, local = dense_tour(queries, docs, locs, loc_pos, k_global=5, k_local=2, device="cpu")
    assert glob[0][0] == 3 and glob[1][0] == 20
    assert set(local[0]) <= {3, 4, 5} and local[0][0] == 3
    assert set(local[1]) <= {30, 31}


def test_sparse_tour_drops_zero_scores():
    docs = sp.csr_matrix(np.array([[1.0, 0, 0], [0, 2.0, 0], [1.0, 1.0, 0]], dtype=np.float32))
    q = sp.csr_matrix(np.array([[1.0, 0, 0]], dtype=np.float32))
    glob, local = sparse_tour(q, docs, np.array([1]), {1: np.array([1, 2])}, k_global=3, k_local=2)
    assert set(glob[0]) == {0, 2}  # doc 1 shares no token with the query -> excluded
    assert list(local[0]) == [2]  # only doc 2 is local AND has a positive score
