"""Encode a text corpus once with bge-m3 and keep all three representations on disk.

    dense    (N, 1024) fp16 .npy       -- loaded into RAM (0.4 GB for 190k items)
    sparse   scipy CSR (N, 250002)     -- lexical weights, loaded into RAM
    colbert  ragged fp16 memmap        -- NEVER loaded whole (§8)

Note on size: bge-m3's ColBERT vectors are 1024-d (colbert_dim=-1 keeps the
hidden size), so the store is ~total_tokens x 2 KB, not the 128-d figure
assumed in reports/02_architectures.md §6.2. Ragged storage keeps it to the
tokens that actually exist (~20 GB for the benchmark corpus); still a
memmap, still fits the 240 GB disk.

Encoding is chunked so RAM stays flat regardless of corpus size.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from avito_rec_sys.retrieval.colbert_narrow import (
    ColbertStore, ColbertWriter, SubsetColbertStore, load_colbert_store,
)
from avito_rec_sys.retrieval.encoders import EncodedBatch, encode_bi_encoder

SPARSE_VOCAB = 250_002  # XLM-R sentencepiece vocabulary size used by bge-m3
DENSE_DIM = 1024


@dataclass
class CorpusStore:
    dense: np.ndarray  # (N, 1024) float16
    sparse: sp.csr_matrix  # (N, SPARSE_VOCAB) float32
    colbert: ColbertStore | SubsetColbertStore

    def subset(self, master_positions: np.ndarray) -> "CorpusStore":
        """Stage corpus = rows `master_positions` of the master corpus."""
        return CorpusStore(
            self.dense[master_positions],
            self.sparse[master_positions],
            SubsetColbertStore(self.colbert, master_positions),
        )


def sparse_to_csr(weights: list[dict], vocab: int = SPARSE_VOCAB) -> sp.csr_matrix:
    rows, cols, vals = [], [], []
    for r, d in enumerate(weights):
        for tok, w in d.items():
            rows.append(r)
            cols.append(int(tok))
            vals.append(float(w))
    return sp.csr_matrix(
        (np.array(vals, dtype=np.float32), (rows, cols)), shape=(len(weights), vocab)
    )


def build_store(
    model, texts: list[str], out_dir: Path, max_length: int, batch_size: int, chunk: int = 4096
) -> CorpusStore:
    out_dir.mkdir(parents=True, exist_ok=True)
    dense = np.zeros((len(texts), DENSE_DIM), dtype=np.float16)
    sparse_parts: list[sp.csr_matrix] = []
    writer = ColbertWriter(out_dir / "colbert.bin", DENSE_DIM)
    for s in range(0, len(texts), chunk):
        enc: EncodedBatch = encode_bi_encoder(model, texts[s : s + chunk], batch_size, max_length)
        dense[s : s + chunk] = enc.dense.astype(np.float16)
        sparse_parts.append(sparse_to_csr(enc.sparse))
        writer.append(enc.colbert)
        print(f"[encode_store] {min(s + chunk, len(texts))}/{len(texts)}")
    sparse = sp.vstack(sparse_parts).tocsr() if sparse_parts else sp.csr_matrix((0, SPARSE_VOCAB), dtype=np.float32)
    colbert = writer.finalize()
    np.save(out_dir / "dense.npy", dense)
    sp.save_npz(out_dir / "sparse.npz", sparse)
    return CorpusStore(dense, sparse, colbert)


def load_store(out_dir: Path) -> CorpusStore:
    return CorpusStore(
        np.load(out_dir / "dense.npy"),
        sp.load_npz(out_dir / "sparse.npz").tocsr(),
        load_colbert_store(out_dir / "colbert.bin", DENSE_DIM),
    )
