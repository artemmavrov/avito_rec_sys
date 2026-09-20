"""bge-m3 wrapper: dense + sparse + ColBERT representations in one encoder pass.

A thin adapter over `FlagEmbedding.BGEM3FlagModel`, so the rest of the pipeline depends on our own
return shapes. bge-m3 needs no "query:" / "passage:" instruction prefix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from huggingface_hub import snapshot_download


@dataclass
class EncodedBatch:
    dense: np.ndarray  # (N, hidden) float32, L2-normalized
    sparse: list[dict[int, float]]  # token_id -> weight, per text
    colbert: list[np.ndarray]  # (n_tokens_i, hidden) float16 per text


def resolve_snapshot(name: str, revision: str) -> str:
    return snapshot_download(name, revision=revision)


def load_bi_encoder(models_cfg: dict, device: str = "cuda", use_fp16: bool = True, model_path: str | None = None):
    """Zero-shot model at the pinned revision, or a fine-tuned checkpoint dir via `model_path`."""
    from FlagEmbedding import BGEM3FlagModel

    cfg = models_cfg["bi_encoder"]
    path = model_path or resolve_snapshot(cfg["name"], cfg["revision"])
    return BGEM3FlagModel(path, use_fp16=use_fp16, devices=device)


def encode_bi_encoder(model, texts: list[str], batch_size: int, max_length: int) -> EncodedBatch:
    if not texts:
        return EncodedBatch(np.empty((0, model.model.model.config.hidden_size), dtype=np.float32), [], [])
    out = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    return EncodedBatch(
        dense=out["dense_vecs"].astype(np.float32),
        sparse=out["lexical_weights"],
        colbert=[v.astype(np.float16) for v in out["colbert_vecs"]],
    )


def encode_dense(model, texts: list[str], batch_size: int, max_length: int) -> np.ndarray:
    """Dense-only fast path (hard-negative mining needs neither sparse nor ColBERT)."""
    if not texts:
        return np.empty((0, 1024), dtype=np.float32)
    out = model.encode(
        texts, batch_size=batch_size, max_length=max_length,
        return_dense=True, return_sparse=False, return_colbert_vecs=False,
    )
    return out["dense_vecs"].astype(np.float32)
