"""Step 2 [CPU, ~25 min]: the BM25 half of the hard-negative pool.

For every distinct training query, stores the top-200 items of the training-item corpus by
multi-field BM25 (as positions in the mining corpus). Steps 3 and 4 fuse them with the dense
ranking by RRF, so the lexical negatives cost no GPU time.

Writes bm25_neg_part_*.parquet (search_query, top) and bm25_neg_meta.json (fingerprint of the mining
corpus that the positions refer to). Resumable: finished parts are skipped.
"""

import argparse
import json
import time

import numpy as np
import polars as pl

from avito_rec_sys.config import load_config
from avito_rec_sys.data.corpus import cached, prepare_items, tokens
from avito_rec_sys.data.lemmatize import lemmatize_batch
from avito_rec_sys.features.bm25f import FieldBM25Index, combine_fields
from avito_rec_sys.paths import add_path_arguments, get_paths
from avito_rec_sys.pipeline.context import build_train_context
from avito_rec_sys.training.mining import build_mining_corpus, corpus_fingerprint

TOP_K = 200
PART_SIZE = 20_000
CHUNK = 64


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_arguments(ap)
    args = ap.parse_args()
    cfg, paths = load_config(), get_paths(args.data_dir, args.work_dir, args.weights_dir)
    t0 = time.time()
    ctx = build_train_context(cfg, paths)
    weights = cfg["bm25"]["field_weights"]

    mc = build_mining_corpus(ctx.splits.train_part)
    items = ctx.splits.train_part.unique(subset=["item_id"], keep="first", maintain_order=True)
    assert items["item_id"].to_list() == mc.ids  # positions must match the mining corpus exactly

    lem = cached(ctx.work / "mining_items_lemmas.parquet", lambda: prepare_items(items).select(
        "item_id", "title_lemmas", "params_lemmas", "desc_lemmas"))
    index = FieldBM25Index().fit(
        {"title": tokens(lem["title_lemmas"]), "params": tokens(lem["params_lemmas"]), "desc": tokens(lem["desc_lemmas"])}
    )
    print(f"index over {len(mc.ids)} training items: {time.time() - t0:.0f}s")

    pairs = pl.read_parquet(ctx.work / "train_clean.parquet")
    queries = pairs["search_query"].unique(maintain_order=True).to_list()
    print(f"{len(queries)} distinct training queries")

    (ctx.work / "bm25_neg_meta.json").write_text(
        json.dumps({"fingerprint": corpus_fingerprint(mc), "n_items": len(mc.ids), "top_k": TOP_K, "weights": weights}),
        encoding="utf-8",
    )
    for part_no, start in enumerate(range(0, len(queries), PART_SIZE)):
        out = ctx.work / f"bm25_neg_part_{part_no:03d}.parquet"
        if out.exists():
            continue
        part = queries[start : start + PART_SIZE]
        q_tokens = [s.split() if s else [] for s in lemmatize_batch(part, n_workers=6)]
        tops: list[list[int]] = []
        for s in range(0, len(part), CHUNK):
            total = combine_fields(index.score_fields(q_tokens[s : s + CHUNK]), weights)
            for row in total:
                k = min(TOP_K, row.shape[0])
                idx = np.argpartition(-row, k - 1)[:k]
                idx = idx[np.argsort(-row[idx], kind="stable")]
                tops.append(idx[row[idx] > 0].astype(np.int32).tolist())
        pl.DataFrame({"search_query": part, "top": tops}).write_parquet(out)
        print(f"part {part_no}: {start + len(part)}/{len(queries)} queries, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
