"""ColBERT MaxSim narrowing of a retrieval pool (§6.2).

Corpus token vectors live on disk as a memmap (`189212 x 128 x 128 x fp16 =
6.2 GB`, §8) -- never loaded into RAM whole. A ragged token-count-per-item
layout is stored as (memmap, offsets) so short titles don't pay for the full
128-token slot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ColbertStore:
    """offsets[i]:offsets[i+1] in the flat (total_tokens, dim) memmap = item i's vectors."""

    def __init__(self, path: Path, offsets: np.ndarray, dim: int, dtype=np.float16):
        self.offsets = offsets
        self.dim = dim
        self.vecs = np.memmap(path, dtype=dtype, mode="r", shape=(int(offsets[-1]), dim))

    def __getitem__(self, i: int) -> np.ndarray:
        return np.asarray(self.vecs[self.offsets[i] : self.offsets[i + 1]])

    @staticmethod
    def build(path: Path, per_item_vecs: list[np.ndarray], dim: int, dtype=np.float16) -> "ColbertStore":
        lengths = np.array([0, *(v.shape[0] for v in per_item_vecs)])
        offsets = np.cumsum(lengths)
        mm = np.memmap(path, dtype=dtype, mode="w+", shape=(int(offsets[-1]), dim))
        for i, v in enumerate(per_item_vecs):
            mm[offsets[i] : offsets[i + 1]] = v.astype(dtype)
        mm.flush()
        return ColbertStore(path, offsets, dim, dtype)


def maxsim_batch(query_vecs: np.ndarray, doc_vecs_list: list[np.ndarray]) -> np.ndarray:
    """MaxSim(q, d) for one query against many docs: sum over query tokens of
    the max cosine similarity to any doc token. Vectors are pre-normalized."""
    q = query_vecs.astype(np.float32)
    out = np.zeros(len(doc_vecs_list), dtype=np.float32)
    for k, d in enumerate(doc_vecs_list):
        if q.size == 0 or d.size == 0:
            continue
        sims = q @ d.astype(np.float32).T
        out[k] = sims.max(axis=1).sum()
    return out


def narrow_pool(
    query_vecs: np.ndarray, candidate_positions: np.ndarray, store: ColbertStore, top_k: int
) -> np.ndarray:
    """Positions (subset of candidate_positions) ranked by MaxSim, best first, truncated to top_k."""
    docs = [store[int(p)] for p in candidate_positions]
    scores = maxsim_batch(query_vecs, docs)
    k = min(top_k, len(candidate_positions))
    order = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
    order = order[np.argsort(-scores[order], kind="stable")]
    return candidate_positions[order]


_JOB = None  # the callable forked workers run; set right before the pool is created


def _call_job(i: int):
    return _JOB(i)


def _one_thread_per_worker() -> None:
    """Workers already give one process per core: keep BLAS from oversubscribing them."""
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except ImportError:
        pass


def parallel_map(fn, n: int, workers: int | None = None, chunksize: int = 16) -> list:
    """[fn(0), ..., fn(n-1)], computed in forked worker processes.

    MaxSim over a 1000-candidate pool is ~0.4 s of single-core numpy per query, which
    left the GPU idle for over an hour on the 8k-query ranker set. Forked workers
    inherit `fn`'s closure copy-on-write and read the ColBERT store through one shared
    page cache. Results are identical to the serial loop. Falls back to it where `fork`
    is unavailable (Windows) or the job is small.
    """
    import multiprocessing as mp
    import os

    if workers is None:
        workers = int(os.environ.get("AVITO_WORKERS", max(1, min(14, (os.cpu_count() or 2) - 2))))
    if workers <= 1 or n < 4 * chunksize or "fork" not in mp.get_all_start_methods():
        return [fn(i) for i in range(n)]
    global _JOB
    _JOB = fn
    try:
        with mp.get_context("fork").Pool(workers, initializer=_one_thread_per_worker) as pool:
            return pool.map(_call_job, range(n), chunksize=chunksize)
    finally:
        _JOB = None


class ColbertWriter:
    """Append-only writer so a corpus can be encoded in chunks without ever
    holding all token vectors in RAM. Layout matches `ColbertStore`."""

    def __init__(self, path: Path, dim: int, dtype=np.float16):
        self.path, self.dim, self.dtype = Path(path), dim, dtype
        self._f = open(self.path, "wb")
        self._lengths: list[int] = []

    def append(self, per_item_vecs: list[np.ndarray]) -> None:
        for v in per_item_vecs:
            self._f.write(np.ascontiguousarray(v.astype(self.dtype)).tobytes())
            self._lengths.append(v.shape[0])

    def finalize(self) -> ColbertStore:
        self._f.close()
        offsets = np.concatenate([[0], np.cumsum(self._lengths)]).astype(np.int64)
        np.save(self.path.with_suffix(".offsets.npy"), offsets)
        return ColbertStore(self.path, offsets, self.dim, self.dtype)


def load_colbert_store(path: Path, dim: int, dtype=np.float16) -> ColbertStore:
    return ColbertStore(path, np.load(Path(path).with_suffix(".offsets.npy")), dim, dtype)


class SubsetColbertStore:
    """View of a `ColbertStore` restricted to `master_positions` (stage corpus
    = subset of the master corpus that was encoded once)."""

    def __init__(self, base: ColbertStore, master_positions: np.ndarray):
        self.base, self.master_positions = base, master_positions

    def __getitem__(self, i: int) -> np.ndarray:
        return self.base[int(self.master_positions[i])]
