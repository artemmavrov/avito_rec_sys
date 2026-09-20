"""bge-reranker-v2-m3 fine-tuning (§5.5): listwise softmax over
(1 positive + K negatives) per query, same freezing scheme as the bi-encoder.

Custom loop instead of FlagEmbedding's reranker CLI: the objective is a
plain cross-entropy over a group's logits, and owning the loop keeps the
data format, checkpoint/resume behaviour (no ECC on a 3090, §11) and
freezing under our control.

Training pairs must come from the pools of the FINE-TUNED retriever (§5.5):
the negatives have to look like what the cross-encoder will meet at
inference, not like zero-shot-retriever mistakes.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from avito_rec_sys.retrieval.encoders import resolve_snapshot
from avito_rec_sys.training.freezing import freeze_encoder_layers


@dataclass
class Group:
    query: str
    docs: list[str]  # docs[0] is the positive, the rest are negatives


class GroupDataset(Dataset):
    def __init__(self, groups: list[Group]):
        self.groups = groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i: int) -> Group:
        return self.groups[i]


def make_collate(tokenizer, max_length: int):
    def collate(batch: list[Group]):
        queries = [g.query for g in batch for _ in g.docs]
        docs = [d for g in batch for d in g.docs]
        enc = tokenizer(
            queries, docs, truncation="only_second", max_length=max_length, padding=True, return_tensors="pt"
        )
        return enc, [len(g.docs) for g in batch]

    return collate


def listwise_loss(logits: torch.Tensor, group_sizes: list[int]) -> torch.Tensor:
    """Cross-entropy with the positive (index 0) as the target inside each group."""
    losses, start = [], 0
    for size in group_sizes:
        g = logits[start : start + size].float()
        losses.append(torch.nn.functional.cross_entropy(g.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=g.device)))
        start += size
    return torch.stack(losses).mean()


def _lr_lambda(warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))

    return f


def _latest_checkpoint(out: Path) -> Path | None:
    ckpts = sorted((p for p in out.glob("checkpoint-*") if p.is_dir()), key=lambda p: int(p.name.split("-")[1]))
    return ckpts[-1] if ckpts else None


def train_reranker(
    cfg: dict, models_cfg: dict, groups: list[Group], output_dir: Path, device: str = "cuda", model_path: str | None = None
) -> dict:
    t = cfg["reranker_train"]
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    rr = models_cfg["reranker"]
    base_path = model_path or resolve_snapshot(rr["name"], rr["revision"])
    resume = _latest_checkpoint(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    model = AutoModelForSequenceClassification.from_pretrained(str(resume) if resume else base_path, num_labels=1)
    model.to(device)
    report = freeze_encoder_layers(model, cfg["freezing"]["frozen_layers"], cfg["freezing"]["freeze_embeddings"])
    if t["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # frozen embeddings would otherwise cut the checkpointed graph
    model.train()

    group_size = len(groups[0].docs)
    groups_per_batch = max(1, t["batch_size_pairs"] // group_size)
    steps_per_epoch = math.ceil(len(groups) / groups_per_batch)
    max_steps = t.get("max_steps", -1)
    total_steps = max_steps if max_steps > 0 else steps_per_epoch * t["epochs"]
    warmup = max(1, int(t["warmup_ratio"] * total_steps))

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params, lr=t["lr"], weight_decay=t["weight_decay"], betas=tuple(t["betas"]), eps=t["eps"]
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(warmup, total_steps))

    step = 0
    if resume and (resume / "training_state.pt").exists():
        state = torch.load(resume / "training_state.pt", map_location="cpu", weights_only=True)
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        step = state["step"]

    collate = make_collate(tokenizer, cfg["text"]["cross_encoder_max_tokens"])
    use_bf16 = t["precision"] == "bf16" and device == "cuda"
    history: list[dict] = []
    t0 = time.time()
    epoch = step // steps_per_epoch
    while step < total_steps:
        gen = torch.Generator().manual_seed(seed + epoch)  # deterministic order => resumable
        skip = step - epoch * steps_per_epoch  # batches of this epoch already trained before a resume
        loader = DataLoader(
            GroupDataset(groups), batch_size=groups_per_batch, shuffle=True, generator=gen, collate_fn=collate
        )
        for i, (enc, sizes) in enumerate(loader):
            if i < skip:
                continue
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(**enc).logits.squeeze(-1)
            loss = listwise_loss(logits, sizes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, t["max_grad_norm"])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            history.append({"step": step, "loss": float(loss.item()), "lr": sched.get_last_lr()[0]})
            if step % 10 == 0 or step == 1:
                print(f"[reranker] step {step}/{total_steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)")
            if step % t["checkpoint_every_steps"] == 0 or step == total_steps:
                ckpt = output_dir / f"checkpoint-{step}"
                model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "step": step}, ckpt / "training_state.pt")
            if step >= total_steps:
                break
        epoch += 1

    (output_dir / "train_history.json").write_text(json.dumps(history), encoding="utf-8")
    return {"freeze": report, "steps": step, "final_loss": history[-1]["loss"] if history else None}
