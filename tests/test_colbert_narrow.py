import numpy as np

from avito_rec_sys.retrieval.colbert_narrow import ColbertStore, maxsim_batch, narrow_pool


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_build_and_read_ragged_store(tmp_path):
    vecs = [_unit(np.random.randn(3, 8)).astype(np.float32), _unit(np.random.randn(0, 8)).astype(np.float32),
            _unit(np.random.randn(5, 8)).astype(np.float32)]
    store = ColbertStore.build(tmp_path / "colbert.mm", vecs, dim=8)
    assert store[0].shape == (3, 8)
    assert store[1].shape == (0, 8)
    assert store[2].shape == (5, 8)
    np.testing.assert_allclose(store[0], vecs[0], atol=1e-3)


def test_maxsim_identical_vector_scores_highest():
    q = _unit(np.random.randn(2, 8)).astype(np.float32)
    same = q.copy()
    other = _unit(np.random.randn(4, 8)).astype(np.float32)
    empty = np.empty((0, 8), dtype=np.float32)
    scores = maxsim_batch(q, [other, same, empty])
    assert scores[1] == max(scores)
    assert np.isclose(scores[1], 2.0, atol=1e-5)  # each query token maxes out at cosine 1 against itself
    assert scores[2] == 0.0


def test_narrow_pool_orders_and_truncates(tmp_path):
    rng = np.random.RandomState(0)
    vecs = [_unit(rng.randn(4, 8)).astype(np.float32) for _ in range(10)]
    store = ColbertStore.build(tmp_path / "c.mm", vecs, dim=8)
    query = vecs[7]  # should match item 7 best
    pos = np.arange(10)
    out = narrow_pool(query, pos, store, top_k=3)
    assert len(out) == 3
    assert out[0] == 7


def test_parallel_map_matches_serial():
    import numpy as np

    from avito_rec_sys.retrieval.colbert_narrow import parallel_map

    data = np.random.RandomState(0).rand(200, 5)
    fn = lambda i: (data[i] @ data[i].T, i)  # noqa: E731 - closure over shared data, like the real jobs
    assert parallel_map(fn, 200, workers=3, chunksize=5) == [fn(i) for i in range(200)]
    assert parallel_map(fn, 6, workers=3) == [fn(i) for i in range(6)]  # below the parallel threshold: serial
