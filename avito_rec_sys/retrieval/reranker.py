"""Cross-encoder scoring (§1): raw logit of (query, item) pairs.

Same interface for the zero-shot bge-reranker-v2-m3 and a fine-tuned
checkpoint -- pass its directory as `model_path`. The logit (not a sigmoid)
is the CatBoost feature `ce_logit`: sigmoid saturates and squeezes the
differences among top candidates that the ranker needs.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class CrossEncoderScorer:
    def __init__(self, model_path: str, device: str = "cuda", max_length: int = 192, batch_size: int = 128):
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)
        if device == "cuda":
            self.model.half()
        self.model.to(device).eval()

    @torch.inference_mode()
    def score(self, queries: list[str], docs: list[str]) -> np.ndarray:
        """Logits aligned with the input order. Pairs are processed sorted by
        length so padding stays small -- a large speed-up on short titles."""
        assert len(queries) == len(docs)
        n = len(queries)
        if n == 0:
            return np.empty(0, dtype=np.float32)
        order = np.argsort([len(q) + len(d) for q, d in zip(queries, docs)], kind="stable")
        out = np.empty(n, dtype=np.float32)
        for s in range(0, n, self.batch_size):
            idx = order[s : s + self.batch_size]
            enc = self.tokenizer(
                [queries[i] for i in idx],
                [docs[i] for i in idx],
                truncation="only_second",
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            out[idx] = self.model(**enc).logits.squeeze(-1).float().cpu().numpy()
        return out
