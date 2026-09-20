"""Stage 8 [GPU, ~12 min on a 3090]: with the FINE-TUNED bi-encoder,
  (a) re-encode the master corpus (dense + sparse + ColBERT) -> work/store_ft,
  (b) mine cross-encoder training groups from the fine-tuned retriever's pools (§5.5).

Order matters: the cross-encoder must train on negatives drawn from the
distribution it will see at inference (the fine-tuned retriever's pool), not
on zero-shot-retriever mistakes.

Groups: <= 2 positives per normalized query (query diversity beats positive
diversity for a reranker), each with 4 negatives from ranks 5-300,
stratified and denoised.
"""

import argparse
import pickle
import time

import numpy as np
import polars as pl

from avito_rec_sys.inference.lexical_pipeline import build_context
from avito_rec_sys.inference.stages import ensure_master_store
from avito_rec_sys.retrieval.encoders import load_bi_encoder
from avito_rec_sys.training.biencoder_train import final_model_dir, query_tower_text
from avito_rec_sys.training.mining import build_mining_corpus, encode_mining_corpus, mine
from avito_rec_sys.training.reranker_train import Group
from avito_rec_sys.utils.io import load_config, load_models_config


def cap_positives_per_query(pairs: pl.DataFrame, cap: int, seed: int) -> pl.DataFrame:
    rng = np.random.RandomState(seed)
    pairs = pairs.with_columns(pl.Series("_r", rng.random(pairs.height)))
    return (
        pairs.with_columns(pl.col("_r").rank(method="ordinal").over("search_query_norm").alias("_k"))
        .filter(pl.col("_k") <= cap)
        .drop(["_r", "_k"])
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-only", action="store_true",
                    help="only build work/store_ft (what s09 needs); mining can then run alongside s09")
    args = ap.parse_args()
    cfg, models = load_config(), load_models_config()
    t = cfg["reranker_train"]
    t0 = time.time()
    ctx = build_context(cfg)
    ft_dir = str(final_model_dir(ctx.work / "biencoder_ft" / "epoch2"))

    ensure_master_store(ctx, models, "ft", ft_dir)
    print(f"fine-tuned master store ready: {time.time() - t0:.0f}s")
    if args.store_only:
        return

    pairs = cap_positives_per_query(pl.read_parquet(ctx.work / "train_clean.parquet"), t["positives_per_query"], cfg["seed"])
    mc = build_mining_corpus(ctx.splits.train_part)
    model = load_bi_encoder(models, model_path=ft_dir)
    dense = encode_mining_corpus(model, mc, cfg)
    negatives = mine(
        pairs, model, mc, dense, cfg, t["negatives_per_positive"], t["negative_rank_min"], t["negative_rank_max"],
        cfg["seed"] + 2,
    )

    groups = []
    for row in pairs.iter_rows(named=True):
        negs = negatives.get((row["search_query_norm"], row["item_id"]), [])
        if len(negs) >= t["negatives_per_positive"]:
            groups.append(Group(query_tower_text(row["search_query"], row["search_infm_params_text"]),
                                [row["item_tower_text"], *negs[: t["negatives_per_positive"]]]))
    with open(ctx.work / "ce_groups.pkl", "wb") as f:
        pickle.dump(groups, f)
    print(f"{len(groups)} cross-encoder groups from {pairs.height} pairs, {pairs['search_query_norm'].n_unique()} queries; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
