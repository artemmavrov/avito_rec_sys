"""Layer freezing shared by the bi-encoder and the cross-encoder (§5.1).

Frozen: the token-embedding table (250k x 1024 = 256M params, 45% of the
model -- a lookup, so freezing costs almost no quality but saves ~4 GB of
optimizer state and the sparse gradient scatter) and the lowest
`frozen_layers` transformer blocks (lexical / syntactic features that
transfer). Everything above, plus heads, trains.

Works on any HF XLM-R style encoder: parameters are matched by name
(`embeddings.` and `encoder.layer.<i>.`), so it applies equally to the bare
`XLMRobertaModel` inside the bge-m3 wrapper and to
`XLMRobertaForSequenceClassification` (whose encoder sits under `roberta.`).
"""

from __future__ import annotations

import re

import torch

_LAYER_RE = re.compile(r"(?:^|\.)encoder\.layer\.(\d+)\.")
_EMBED_RE = re.compile(r"(?:^|\.)embeddings\.")


def freeze_encoder_layers(model: torch.nn.Module, frozen_layers: int, freeze_embeddings: bool = True) -> dict:
    """Set requires_grad=False on the embedding block and the first
    `frozen_layers` blocks. Returns counts so callers can log/assert them."""
    n_frozen = n_trainable = 0
    frozen_layer_ids: set[int] = set()
    for name, p in model.named_parameters():
        m = _LAYER_RE.search(name)
        freeze = False
        if m and int(m.group(1)) < frozen_layers:
            freeze = True
            frozen_layer_ids.add(int(m.group(1)))
        elif freeze_embeddings and _EMBED_RE.search(name):
            freeze = True
        p.requires_grad = not freeze
        if freeze:
            n_frozen += p.numel()
        else:
            n_trainable += p.numel()
    return {
        "frozen_params": n_frozen,
        "trainable_params": n_trainable,
        "frozen_layers": sorted(frozen_layer_ids),
    }
