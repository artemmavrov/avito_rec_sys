"""Stage 10 [GPU, ~2.3 h on a 3090]: fine-tune bge-reranker-v2-m3 (§5.5), 1 epoch.

Run only if the §12.1 gate (stage 9) says so. Resumable: rerunning continues
from the newest checkpoint in work/reranker_ft.
"""

import pickle
import time

from avito_rec_sys.inference.lexical_pipeline import build_context
from avito_rec_sys.training.reranker_train import train_reranker
from avito_rec_sys.utils.io import load_config, load_models_config


def main() -> None:
    cfg, models = load_config(), load_models_config()
    ctx = build_context(cfg)
    with open(ctx.work / "ce_groups.pkl", "rb") as f:
        groups = pickle.load(f)
    print(f"{len(groups)} training groups")
    t0 = time.time()
    report = train_reranker(cfg, models, groups, ctx.work / "reranker_ft")
    print(f"done in {(time.time() - t0) / 60:.0f} min: {report}")


if __name__ == "__main__":
    main()
