"""Step 5 [GPU + CPU, ~30 min per stage on a T4-class card]: candidate pools and features of a
hold-out, computed with the fine-tuned bi-encoder.

    python scripts/train/05_build_features.py --stage ranker    # rows the ranker is trained on (step 6)
    python scripts/train/05_build_features.py --stage val       # rows it is measured on

The whole master corpus (benchmark items + hold-out positives) is encoded once into work/store_ft
(~22 GB); each stage uses its own subset of it. Output: work/feats_<stage>.parquet.
"""

import argparse
import gc
import time

import numpy as np
import torch

from avito_rec_sys.config import load_config, load_models_config
from avito_rec_sys.paths import add_path_arguments, get_paths
from avito_rec_sys.pipeline.candidates import CandidateGenerator, item_tower_texts
from avito_rec_sys.pipeline.context import build_train_context, make_stage
from avito_rec_sys.retrieval.encode_store import load_or_build_store
from avito_rec_sys.retrieval.encoders import load_bi_encoder
from avito_rec_sys.training.biencoder_train import final_model_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_arguments(ap)
    ap.add_argument("--stage", choices=["ranker", "val"], required=True)
    ap.add_argument("--encoder-dir", help="default: work/biencoder_ft/epoch2")
    args = ap.parse_args()
    cfg, models = load_config(), load_models_config()
    paths = get_paths(args.data_dir, args.work_dir, args.weights_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    ctx = build_train_context(cfg, paths)
    encoder_dir = args.encoder_dir or str(final_model_dir(ctx.work / "biencoder_ft" / "epoch2"))
    encoder = load_bi_encoder(models, device, model_path=encoder_dir)
    master = load_or_build_store(
        encoder, item_tower_texts(ctx.master_items), ctx.work / "store_ft", cfg["text"]["item_max_tokens"]
    )
    print(f"master store ready: {time.time() - t0:.0f}s")

    stage = make_stage(ctx, args.stage)
    if args.stage == "ranker":
        stage.queries = stage.queries.head(cfg["catboost"]["train_queries"])  # hold-outs are randomly ordered
    master_pos = {v: i for i, v in enumerate(ctx.master_items["item_id"].to_list())}
    positions = np.array([master_pos[i] for i in stage.corpus["item_id"].to_list()], dtype=np.int64)
    work = ctx.work
    del ctx, master_pos  # the training history and the master corpus are not needed any more: free the RAM
    gc.collect()

    feats = CandidateGenerator(stage, master.subset(positions), encoder, cfg, device).features()
    feats.to_parquet(work / f"feats_{args.stage}.parquet")
    print(f"{args.stage}: {len(feats)} candidate pairs, {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
