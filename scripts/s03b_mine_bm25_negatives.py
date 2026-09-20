"""Stage 3b [CPU, ~25 min]: BM25 half of the hard-negative pool (§5.3).

Runs after s03 (it uses the tuned field weights). For every distinct training
query it stores the top-200 items of the training-item corpus by multi-field
BM25 (positions in the mining corpus). The GPU stage s06 (and the negative
refresh inside s07) fuses these with the dense ranking by RRF, so the lexical
negatives cost zero GPU minutes.

Output in work/: bm25_neg_part_*.parquet (search_query, top) and
bm25_neg_meta.json (fingerprint of the mining corpus the positions refer to).
Resumable: finished parts are skipped.
"""

import json
import time

import numpy as np
import polars as pl

from avito_rec_sys.data.corpus import cached, prepare_items, tokens
from avito_rec_sys.data.lemmatize import lemmatize_batch
from avito_rec_sys.features.bm25f import FieldBM25Index, combine_fields
from avito_rec_sys.inference.lexical_pipeline import build_context
from avito_rec_sys.training.mining import build_mining_corpus, corpus_fingerprint
from avito_rec_sys.utils.io import load_config

TOP_K = 200
PART_SIZE = 20_000
CHUNK = 64


def main() -> None:
    cfg = load_config()
    t0 = time.time()
    ctx = build_context(cfg)
    weights = json.loads((ctx.work / "s03_results.json").read_text())["weights"]

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
