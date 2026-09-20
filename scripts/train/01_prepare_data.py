"""Step 1 [CPU, ~15 min]: hold-out splits, lemmatized corpus, cleaned training pairs.

Writes to the work dir:
  splits/                leak-free `val` and `ranker` hold-outs and the remaining training part
  master_items.parquet   benchmark items + hold-out positives with lemma / filtered-params columns
  train_clean.parquet    deduplicated, capped training pairs for the bi-encoder (training part only)
"""

import argparse
import json

from avito_rec_sys.config import load_config
from avito_rec_sys.data.clean_train import clean_train
from avito_rec_sys.paths import add_path_arguments, get_paths
from avito_rec_sys.pipeline.context import build_train_context


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_arguments(ap)
    args = ap.parse_args()
    cfg, paths = load_config(), get_paths(args.data_dir, args.work_dir, args.weights_dir)

    ctx = build_train_context(cfg, paths)
    print(f"master corpus: {ctx.master_items.height} items")
    for name, held in (("val", ctx.splits.val), ("ranker", ctx.splits.ranker)):
        print(f"{name}: {held.queries.height} queries, {held.qrels.height} relevant pairs")

    cleaned, stats = clean_train(ctx.splits.train_part, cfg["cleaning"]["cap_pairs_per_query"], cfg["seed"])
    cleaned.write_parquet(ctx.work / "train_clean.parquet")
    print("train cleaning:", json.dumps(stats.as_dict()))


if __name__ == "__main__":
    main()
