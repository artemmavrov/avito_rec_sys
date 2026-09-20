"""bge-m3 wrapper: dense + sparse + ColBERT in one encoder pass (§1, §2.1).

Thin adapter over `FlagEmbedding.BGEM3FlagModel` so the rest of the pipeline
depends on our own return shapes, not FlagEmbedding's dict format. The model
revision is pinned in configs/models.yaml (§2.1) -- resolved to a local
snapshot path via huggingface_hub so `from_pretrained` can't silently pick up
a newer HF revision than the one recorded there.

§11 risk 3: bge-m3, unlike bge-v1.5, uses NO "query:"/"passage:" instruction
prefix (query_instruction is "" in models.yaml). This is asserted, not just
assumed -- `load_bi_encoder` fails loudly if models.yaml ever sets a
non-empty instruction without a matching code change here.
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
    if cfg["query_instruction"] or cfg["passage_instruction"]:
        raise NotImplementedError(
            "models.yaml sets a non-empty bge-m3 instruction prefix, but load_bi_encoder "
            "does not apply one yet -- update this function before changing that config value."
        )
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


def sparse_score(q: dict[int, float], d: dict[int, float]) -> float:
    """bge-m3's own sparse (lexical) score: dot product over shared token ids."""
    if len(q) > len(d):
        q, d = d, q
    return float(sum(w * d[t] for t, w in q.items() if t in d))


def colbert_maxsim(q: np.ndarray, d: np.ndarray) -> float:
    """Sum of max cosine similarity per query token (bge-m3's late-interaction score).
    Vectors are already L2-normalized by the encoder."""
    if q.size == 0 or d.size == 0:
        return 0.0
    sims = q.astype(np.float32) @ d.astype(np.float32).T
    return float(sims.max(axis=1).sum())
