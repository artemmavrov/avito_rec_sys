"""Multi-field BM25 over lemmas (§3.1 / §4).

Three independent BM25 indices (title, filtered params, full description)
share one vocabulary. `bm25f_total` is their weighted sum. This is the
"BM25F-lite" form the feature bank in §4 implies: it lists the per-field
scores as separate features next to the total, so a true term-level field
fusion would throw away information the ranker is given anyway.

Implementation is a precomputed sparse weight matrix per field:
    W[d, t] = idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len_d / avg_len))
so scoring a chunk of queries is one sparse matmul (Q x V) @ (V x D), with no
per-query Python loop.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp

FIELDS = ("title", "params", "desc")


class FieldBM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vocab: dict[str, int] = {}
        self.weights_t: dict[str, sp.csr_matrix] = {}  # per field: (V x D) BM25 weights
        self.n_docs = 0

    def fit(self, field_tokens: Mapping[str, Sequence[Sequence[str]]]) -> "FieldBM25Index":
        n_docs = len(next(iter(field_tokens.values())))
        self.n_docs = n_docs
        for tokens in field_tokens.values():
            for doc in tokens:
                for tok in doc:
                    if tok not in self.vocab:
                        self.vocab[tok] = len(self.vocab)
        v = len(self.vocab)
        for name, docs in field_tokens.items():
            self.weights_t[name] = self._build_weights(docs, v)
        return self

    def _build_weights(self, docs: Sequence[Sequence[str]], v: int) -> sp.csr_matrix:
        rows, cols, vals, lengths = [], [], [], np.zeros(len(docs), dtype=np.float32)
        for d, doc in enumerate(docs):
            lengths[d] = len(doc)
            counts: dict[int, int] = {}
            for tok in doc:
                idx = self.vocab[tok]
                counts[idx] = counts.get(idx, 0) + 1
            for idx, c in counts.items():
                rows.append(d)
                cols.append(idx)
                vals.append(c)
        tf = sp.csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)), shape=(len(docs), v))
        df = np.diff(tf.tocsc().indptr)  # docs containing each term
        idf = np.log(1.0 + (len(docs) - df + 0.5) / (df + 0.5)).astype(np.float32)
        avg_len = max(float(lengths.mean()), 1e-9)
        tf = tf.tocoo()
        norm = self.k1 * (1.0 - self.b + self.b * lengths[tf.row] / avg_len)
        data = idf[tf.col] * tf.data * (self.k1 + 1.0) / (tf.data + norm)
        w = sp.csr_matrix((data.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)
        return w.T.tocsr()

    def query_matrix(self, queries_tokens: Sequence[Sequence[str]]) -> sp.csr_matrix:
        """Binary (Q x V) matrix; out-of-vocabulary tokens are ignored."""
        rows, cols = [], []
        for q, toks in enumerate(queries_tokens):
            for idx in {self.vocab[t] for t in toks if t in self.vocab}:
                rows.append(q)
                cols.append(idx)
        return sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(queries_tokens), len(self.vocab))
        )

    def score_fields(self, queries_tokens: Sequence[Sequence[str]]) -> dict[str, np.ndarray]:
        """Dense (Q x D) float32 score matrix per field. Keep chunks modest
        (~128 queries): each matrix is Q x 190k x 4 bytes."""
        qm = self.query_matrix(queries_tokens)
        return {name: (qm @ wt).toarray() for name, wt in self.weights_t.items()}


def combine_fields(scores: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> np.ndarray:
    total = None
    for name, s in scores.items():
        contrib = weights.get(name, 0.0) * s
        total = contrib if total is None else total + contrib
    return total
