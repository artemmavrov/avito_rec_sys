"""Stage 1: leak-free splits (val + ranker hold-outs) on normalized query text.

Writes work/splits/* and prints the statistics the report quotes.
"""

import json

import polars as pl

from avito_rec_sys.data.split import build_corpus, cold_start_ratio, make_splits
from avito_rec_sys.data.loading import load_benchmark_items, load_train
from avito_rec_sys.utils.io import ensure_dir, load_config, resolve_path


def main() -> None:
    cfg = load_config()
    work = ensure_dir(resolve_path(cfg, "work_dir"))
    train = load_train(resolve_path(cfg, "train_parquet"))
    bench = load_benchmark_items(resolve_path(cfg, "benchmark_items_parquet"))

    sp = cfg["split"]
    res = make_splits(train, sp["n_validation_queries"], sp["n_ranker_queries"], sp["seen_fraction"], cfg["seed"])

    sdir = ensure_dir(work / "splits")
    res.train_part.write_parquet(sdir / "train_part.parquet")
    stats = {"train_rows": train.height, "train_part_rows": res.train_part.height,
             "rows_dropped_for_item_overlap": res.n_rows_dropped_for_item_overlap}
    for name, held in (("val", res.val), ("ranker", res.ranker)):
        d = ensure_dir(sdir / name)
        held.queries.write_parquet(d / "queries.parquet")
        held.qrels.write_parquet(d / "qrels.parquet")
        held.positive_items.write_parquet(d / "positive_items.parquet")
        corpus = build_corpus(held.positive_items, bench)
        stats[name] = {
            "queries": held.queries.height,
            "strata": {r["stratum"]: r["count"] for r in held.queries["stratum"].value_counts().to_dicts()},
            "items_per_query": round(held.qrels.height / held.queries.height, 3),
            "corpus_size": corpus.height,
            "cold_start_ratio": round(cold_start_ratio(corpus, res.train_part), 4),
        }
    (work / "split_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
