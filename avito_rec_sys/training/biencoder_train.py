"""bge-m3 fine-tuning (§5.4) on top of FlagEmbedding's own M3 trainer.

Reimplementing bge-m3's joint dense + sparse + ColBERT loss (with
self-distillation) from scratch is a high-risk way to lose points silently,
so we reuse `EncoderOnlyEmbedderM3Runner` and add only what the spec needs
on top of it:
  * our data pipeline (cleaned pairs + stratified, denoised hard negatives),
  * the layer-freezing scheme of §5.1 (FlagEmbedding only offers all-or-nothing
    `fix_encoder`),
  * hyper-parameters from configs/pipeline.yaml.

FlagEmbedding is MIT licensed; it is listed in README.md as required.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl

from avito_rec_sys.retrieval.encoders import resolve_snapshot
from avito_rec_sys.training.freezing import freeze_encoder_layers


def query_tower_text(query: str | None, params: str | None) -> str:
    """Query-tower input (§3.1): search text + raw filter text (already query-like)."""
    q = (query or "").strip()
    p = (params or "").strip()
    return f"{q} {p}".strip()


def write_training_jsonl(
    pairs: pl.DataFrame,
    negatives: dict[tuple[str, str], list[str]],
    out_path: Path,
    n_negatives: int,
) -> int:
    """One JSONL line per (query, positive) pair in FlagEmbedding's format:
    {"query": ..., "pos": [tower text], "neg": [tower text, ...]}.

    `negatives` maps (search_query_norm, item_id) -> list of negative tower
    texts (already denoised). Pairs with fewer than `n_negatives` mined
    negatives are skipped -- a short group would break the fixed
    `train_group_size` the collator expects.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in pairs.iter_rows(named=True):
            negs = negatives.get((row["search_query_norm"], row["item_id"]), [])
            if len(negs) < n_negatives:
                continue
            record = {
                "query": query_tower_text(row["search_query"], row["search_infm_params_text"]),
                "pos": [row["item_tower_text"]],
                "neg": negs[:n_negatives],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def _latest_checkpoint(output_dir: Path) -> str | None:
    ckpts = sorted((p for p in output_dir.glob("checkpoint-*") if p.is_dir()), key=lambda p: int(p.name.split("-")[1]))
    return str(ckpts[-1]) if ckpts else None


def build_arguments(
    cfg: dict, models_cfg: dict, train_jsonl: Path, output_dir: Path, n_groups: int, init_model_path: str | None = None
):
    """Translate configs/pipeline.yaml (+ smoke overrides) into FlagEmbedding's argument dataclasses."""
    from FlagEmbedding.abc.finetune.embedder import AbsEmbedderDataArguments
    from FlagEmbedding.finetune.embedder.encoder_only.m3 import (
        EncoderOnlyEmbedderM3ModelArguments,
        EncoderOnlyEmbedderM3TrainingArguments,
    )

    t = cfg["biencoder_train"]
    bi = models_cfg["bi_encoder"]
    model_args = EncoderOnlyEmbedderM3ModelArguments(
        model_name_or_path=init_model_path or resolve_snapshot(bi["name"], bi["revision"]), token=None, colbert_dim=-1
    )
    data_args = AbsEmbedderDataArguments(
        train_data=[str(train_jsonl)],
        train_group_size=1 + t["hard_negatives_per_positive"],  # 1 positive + N hard negatives
        query_max_len=cfg["text"]["query_max_tokens"],
        passage_max_len=cfg["text"]["item_max_tokens"],
    )
    max_steps = t.get("max_steps", -1)
    steps_per_epoch = max(1, math.ceil(n_groups / t["batch_size_groups"]))
    total_steps = max_steps if max_steps > 0 else steps_per_epoch * t["epochs"]
    training_args = EncoderOnlyEmbedderM3TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=t["batch_size_groups"],
        num_train_epochs=t["epochs"],
        max_steps=max_steps,
        learning_rate=t["lr"],
        weight_decay=t["weight_decay"],
        adam_beta1=t["betas"][0],
        adam_beta2=t["betas"][1],
        adam_epsilon=t["eps"],  # 1e-6, not 1e-8: below bf16 resolution (§5.4)
        max_grad_norm=t["max_grad_norm"],
        lr_scheduler_type="linear",
        warmup_steps=max(1, int(t["warmup_ratio"] * total_steps)),
        bf16=t["precision"] == "bf16",
        fp16=t["precision"] == "fp16",
        gradient_checkpointing=t["gradient_checkpointing"],
        temperature=t["temperature"],
        save_steps=t["checkpoint_every_steps"],  # no ECC on a 3090: checkpoint often (§11)
        save_total_limit=3,
        logging_steps=10,
        report_to="none",
        seed=cfg["seed"],
        sentence_pooling_method="cls",
        normalize_embeddings=True,
        unified_finetuning=True,  # train dense + sparse + ColBERT jointly
        use_self_distill=True,
        fix_encoder=False,  # freezing is layer-wise below
        dataloader_num_workers=2,
        resume_from_checkpoint=_latest_checkpoint(output_dir),  # crash-safe: a rerun continues (§11)
    )
    return model_args, data_args, training_args


def sparse_embedding_lowmem(self, hidden_state, input_ids, return_embedding: bool = True):
    """Drop-in for FlagEmbedding's M3 `_sparse_embedding` in training mode.

    Upstream materialises a dense (batch, seq_len, vocab=250k) tensor and takes
    a max over seq_len: with 256 groups x 3 passages that is ~46 GB and OOMs a 24 GB
    card at step 0. Here the per-token weights are scattered straight into a
    (batch, vocab) tensor with an "amax" reduction: same values (weights are
    ReLU >= 0, so the zero init is exact) and same gradient, ~100x less memory.
    """
    import torch

    token_weights = torch.relu(self.sparse_linear(hidden_state))
    if not return_embedding:
        return token_weights
    emb = torch.zeros(
        input_ids.size(0), self.vocab_size, dtype=token_weights.dtype, device=token_weights.device
    ).scatter_reduce(dim=-1, index=input_ids, src=token_weights.squeeze(-1), reduce="amax")
    # Mask special tokens out-of-place: scatter_reduce's backward reads its own output, so an in-place
    # `emb[:, unused] *= 0` (what upstream does) would break autograd here.
    keep = torch.ones(self.vocab_size, dtype=emb.dtype, device=emb.device)
    keep[[self.tokenizer.cls_token_id, self.tokenizer.eos_token_id,
          self.tokenizer.pad_token_id, self.tokenizer.unk_token_id]] = 0.0
    return emb * keep


def colbert_score_chunked(self, q_reps, p_reps, q_mask=None, chunk: int = 32):
    """Drop-in for M3's `compute_colbert_score`: same MaxSim, evaluated `chunk` queries at a time.

    Upstream builds the full (queries, q_len, passages, p_len) einsum output at once
    (~2.3 GB in bf16 for 256 x 768, doubled again in backward), which is the difference
    between fitting and not fitting on a 24 GB card next to the 250k-vocab sparse head.
    The max only needs the per-chunk output, so peak transient memory drops ~8x.
    """
    import torch

    per_chunk = []
    for s in range(0, q_reps.size(0), chunk):
        token_scores = torch.einsum("qin,pjn->qipj", q_reps[s : s + chunk], p_reps)
        per_chunk.append(token_scores.max(-1).values.sum(1))
    scores = torch.cat(per_chunk, 0) / q_mask[:, 1:].sum(-1, keepdim=True)
    return scores / self.temperature


def make_runner_class():
    """Built lazily so importing this module doesn't require FlagEmbedding."""
    import types

    from FlagEmbedding.finetune.embedder.encoder_only.m3 import EncoderOnlyEmbedderM3Runner

    class FrozenM3Runner(EncoderOnlyEmbedderM3Runner):
        frozen_layers: int = 0
        freeze_embeddings: bool = True
        freeze_report: dict = {}

        def load_tokenizer_and_model(self):
            tokenizer, model = super().load_tokenizer_and_model()
            type(self).freeze_report = freeze_encoder_layers(model, self.frozen_layers, self.freeze_embeddings)
            # this model instance is only used for training, where upstream's branch is the memory hog
            model._sparse_embedding = types.MethodType(sparse_embedding_lowmem, model)
            model.compute_colbert_score = types.MethodType(colbert_score_chunked, model)
            return tokenizer, model

    return FrozenM3Runner


def run_finetune(
    cfg: dict, models_cfg: dict, train_jsonl: Path, output_dir: Path, n_groups: int, init_model_path: str | None = None
) -> dict:
    model_args, data_args, training_args = build_arguments(
        cfg, models_cfg, train_jsonl, output_dir, n_groups, init_model_path
    )
    runner_cls = make_runner_class()
    runner_cls.frozen_layers = cfg["freezing"]["frozen_layers"]
    runner_cls.freeze_embeddings = cfg["freezing"]["freeze_embeddings"]
    runner = runner_cls(model_args, data_args, training_args)
    runner.run()
    return runner_cls.freeze_report


def final_model_dir(output_dir: Path) -> Path:
    """Directory of the trained bi-encoder: the run's root if the trainer saved
    the final model there, else its newest checkpoint."""
    if (output_dir / "colbert_linear.pt").exists():
        return output_dir
    ckpts = sorted((p for p in output_dir.glob("checkpoint-*") if p.is_dir()), key=lambda p: int(p.name.split("-")[1]))
    if not ckpts:
        raise FileNotFoundError(f"no trained model in {output_dir}")
    return ckpts[-1]
