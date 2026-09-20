"""Stage 6 [GPU, ~15 min on a 3090]: zero-shot hard-negative mining (§5.3) and
the bi-encoder training file work/biencoder_train.jsonl.

Negatives: 2 per positive, stratified over ranks 10-200 of the hybrid
ranking (dense + the BM25 rankings precomputed by s03b, fused by RRF), denoised against the positive's tower text, never an item another
user chose for the same normalized query.
"""

import time

import polars as pl

from avito_rec_sys.inference.lexical_pipeline import build_context
from avito_rec_sys.retrieval.encoders import load_bi_encoder
from avito_rec_sys.training.biencoder_train import write_training_jsonl
from avito_rec_sys.training.mining import build_mining_corpus, encode_mining_corpus, load_bm25_negatives, mine
from avito_rec_sys.utils.io import load_config, load_models_config


def main() -> None:
    cfg, models = load_config(), load_models_config()
    t = cfg["biencoder_train"]
    t0 = time.time()
    ctx = build_context(cfg)
    pairs = pl.read_parquet(ctx.work / "train_clean.parquet")
    mc = build_mining_corpus(ctx.splits.train_part)
    print(f"mining corpus: {len(mc.ids)} items, {pairs.height} training pairs")
    bm25 = load_bm25_negatives(ctx.work, mc)
    print("BM25 negatives:", f"{len(bm25)} queries" if bm25 is not None else "not found -> dense only (run s03b)")

    model = load_bi_encoder(models)
    corpus_dense = encode_mining_corpus(model, mc, cfg)
    print(f"corpus encoded: {time.time() - t0:.0f}s")

    negatives = mine(
        pairs, model, mc, corpus_dense, cfg, t["hard_negatives_per_positive"],
        t["negative_rank_min"], t["negative_rank_max"], cfg["seed"], bm25_top=bm25,
    )
    n = write_training_jsonl(pairs, negatives, ctx.work / "biencoder_train.jsonl", t["hard_negatives_per_positive"])
    print(f"wrote {n} training groups ({pairs.height - n} pairs skipped: too few clean negatives); {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
