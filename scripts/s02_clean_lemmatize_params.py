"""Stage 2: clean the encoder training pairs (§5.2), filter item params (§3.2)
and lemmatize every item that can appear in a corpus (§3.3).

Outputs (work/): train_clean.parquet (encoder/reranker training pairs, drawn
from the split's training part only) and master_items.parquet.
"""

import json
import time

import numpy as np

from avito_rec_sys.data.clean_train import clean_train
from avito_rec_sys.inference.lexical_pipeline import build_context
from avito_rec_sys.utils.io import load_config


def main() -> None:
    cfg = load_config()
    t = time.time()
    ctx = build_context(cfg)
    print(f"context + lemmatized master items: {time.time() - t:.0f}s, {ctx.master_items.height} items")

    cleaned, stats = clean_train(ctx.splits.train_part, cfg["cleaning"]["cap_pairs_per_query"], cfg["seed"])
    cleaned.write_parquet(ctx.work / "train_clean.parquet")

    before = ctx.master_items["item_infm_params_text"].str.len_chars().to_numpy()
    after = ctx.master_items["params_filtered"].str.len_chars().to_numpy()
    out = {
        "cleaning": stats.as_dict(),
        "params_chars_median_before": float(np.median(before)),
        "params_chars_median_after": float(np.median(after)),
        "params_volume_reduction_pct": round(100 * (1 - after.sum() / before.sum()), 1),
        "params_empty_after_pct": round(100 * float((after == 0).mean()), 2),
    }
    (ctx.work / "clean_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
