"""Stage 7 [GPU, ~4.3 h on a 3090]: bge-m3 fine-tuning, 2 epochs, with the
ANCE-style negative refresh between them (§5.3, §5.4).

Two one-epoch runs instead of one two-epoch run: the refresh re-encodes the
mining corpus with the epoch-1 model and re-mines negatives, which cannot
happen inside FlagEmbedding's trainer. Consequence (a small deviation from
§5.4's single schedule): warmup + linear decay restart in epoch 2, which
starts from the epoch-1 weights, so epoch 2 uses the gentler `biencoder_train.epoch2`
schedule (lr 5e-6, warmup 1%). Resumable: a finished epoch is skipped.

Usage: python scripts/s07_train_biencoder.py
"""

import copy
import shutil
import time

import polars as pl

from avito_rec_sys.inference.lexical_pipeline import build_context
from avito_rec_sys.retrieval.encoders import load_bi_encoder
from avito_rec_sys.training.biencoder_train import final_model_dir, run_finetune, write_training_jsonl
from avito_rec_sys.training.mining import build_mining_corpus, encode_mining_corpus, load_bm25_negatives, mine
from avito_rec_sys.utils.io import load_config, load_models_config


def count_lines(path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def main() -> None:
    cfg, models = load_config(), load_models_config()
    t = cfg["biencoder_train"]
    ctx = build_context(cfg)
    out_root = ctx.work / "biencoder_ft"
    jsonl = ctx.work / "biencoder_train.jsonl"
    one_epoch = copy.deepcopy(cfg)
    one_epoch["biencoder_train"]["epochs"] = 1
    epoch_cfg = {1: one_epoch, 2: copy.deepcopy(one_epoch)}
    epoch_cfg[2]["biencoder_train"].update(t["epoch2"])  # lr / warmup_ratio for the restart

    prev = None
    for epoch in (1, 2):
        out = out_root / f"epoch{epoch}"
        if (out / "colbert_linear.pt").exists():
            print(f"epoch {epoch} already trained, skipping")
        else:
            t0 = time.time()
            report = run_finetune(epoch_cfg[epoch], models, jsonl, out, count_lines(jsonl), init_model_path=prev)
            print(f"epoch {epoch} done in {(time.time() - t0) / 60:.0f} min; freeze report: {report}")
        prev = str(final_model_dir(out))
        if epoch == 1 and not (ctx.work / "biencoder_train_epoch1.jsonl").exists():  # refresh done once
            # refresh negatives with the model that just trained
            pairs = pl.read_parquet(ctx.work / "train_clean.parquet")
            mc = build_mining_corpus(ctx.splits.train_part)
            bm25 = load_bm25_negatives(ctx.work, mc)
            model = load_bi_encoder(models, model_path=prev)
            dense = encode_mining_corpus(model, mc, cfg)
            negatives = mine(
                pairs, model, mc, dense, cfg, t["hard_negatives_per_positive"],
                t["negative_rank_min"], t["negative_rank_max"], cfg["seed"] + 1, bm25_top=bm25,
            )
            del model
            shutil.copy(jsonl, ctx.work / "biencoder_train_epoch1.jsonl")
            n = write_training_jsonl(pairs, negatives, jsonl, t["hard_negatives_per_positive"])
            print(f"refreshed negatives: {n} groups")
    print("final bi-encoder:", prev)


if __name__ == "__main__":
    main()
