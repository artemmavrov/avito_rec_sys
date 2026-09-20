"""Step 3 [GPU, ~15 min]: zero-shot hard-negative mining and the bi-encoder training file.

Negatives: 2 per positive, stratified over ranks 10-200 of the hybrid ranking (zero-shot bge-m3
dense + the BM25 rankings of step 2, fused by RRF), denoised against the positive's tower text,
never an item another user chose for the same normalized query.

Writes biencoder_train.jsonl in FlagEmbedding's format.
"""

import argparse
import time

import polars as pl

from avito_rec_sys.config import load_config, load_models_config
from avito_rec_sys.paths import add_path_arguments, get_paths
from avito_rec_sys.pipeline.context import build_train_context
from avito_rec_sys.retrieval.encoders import load_bi_encoder
from avito_rec_sys.training.biencoder_train import write_training_jsonl
from avito_rec_sys.training.mining import build_mining_corpus, encode_mining_corpus, load_bm25_negatives, mine


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_arguments(ap)
    args = ap.parse_args()
    cfg, models = load_config(), load_models_config()
    paths = get_paths(args.data_dir, args.work_dir, args.weights_dir)
    t = cfg["biencoder_train"]
    t0 = time.time()
    ctx = build_train_context(cfg, paths)
    pairs = pl.read_parquet(ctx.work / "train_clean.parquet")
    mc = build_mining_corpus(ctx.splits.train_part)
    print(f"mining corpus: {len(mc.ids)} items, {pairs.height} training pairs")
    bm25 = load_bm25_negatives(ctx.work, mc)
    print("BM25 negatives:", f"{len(bm25)} queries" if bm25 is not None else "not found -> dense only (run step 2)")

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
