"""ColBERT (late-interaction) token vectors of the corpus: an on-disk store and MaxSim scoring.

The vectors are 1024-d fp16 and take ~20 GB for the benchmark corpus, so they live in a memmap
that is never loaded into RAM whole. A ragged layout (offsets per item) means short titles do not
pay for a full-length slot.
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


def pair_maxsim(store, query_vecs: list[np.ndarray], q_idx: np.ndarray, i_idx: np.ndarray) -> np.ndarray:
    """ColBERT MaxSim of every (query, item) pair: for each query token the best cosine over the
    item's tokens, summed over the query tokens (vectors are L2-normalized by the encoder).

    The pairs are scored item by item in storage order, so the store is read exactly once and
    sequentially. Scoring query by query instead reads the candidates of every query at random,
    which makes a 20 GB store unusable on a machine whose RAM cannot cache it.
    `query_vecs[q]` are float32 token matrices; `q_idx[k], i_idx[k]` name the k-th pair.
    """
    out = np.zeros(len(q_idx), dtype=np.float32)
    order = np.argsort(i_idx, kind="stable")
    sorted_items = i_idx[order]
    starts = np.flatnonzero(np.diff(sorted_items, prepend=-1))
    ends = np.append(starts[1:], len(order))
    for s, e in zip(starts, ends):
        doc_t = store[int(sorted_items[s])].astype(np.float32).T
        if doc_t.size == 0:
            continue
        for k in order[s:e]:
            q = query_vecs[q_idx[k]]
            if q.size:
                out[k] = (q @ doc_t).max(axis=1).sum()
    return out


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
