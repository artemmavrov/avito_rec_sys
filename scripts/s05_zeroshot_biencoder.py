"""Stage 5 [GPU, ~10 min on a 3090]: zero-shot bge-m3 over the whole pipeline.

Encodes the master corpus once (dense + sparse + ColBERT), runs the
validation stage without the cross-encoder, prints the ceiling curve per tour
(§9.5), tunes the RRF weights on it (§6.1, CPU) and stores them. Also decides whether ColBERT narrowing beats plain
truncation of the RRF pool (threshold: narrowing.max_loss_pp) -> work/narrowing_mode.json.
"""

import argparse
import json
import time

from avito_rec_sys.inference.lexical_pipeline import build_context
from avito_rec_sys.inference.stages import (
    ceiling_report, ensure_master_store, narrowing_decision, run_full_stage, stage_inputs, stage_qrels, tune_rrf,
)
from avito_rec_sys.retrieval.encoders import load_bi_encoder
from avito_rec_sys.utils.io import load_config, load_models_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="run the (cached-corpus) validation pass on CPU, e.g. while the GPU trains")
    device = "cpu" if ap.parse_args().cpu else "cuda"
    cfg, models = load_config(), load_models_config()
    t0 = time.time()
    ctx = build_context(cfg)
    master = ensure_master_store(ctx, models, "zeroshot", None, device)
    print(f"master store ready: {time.time() - t0:.0f}s")

    stage, index, store = stage_inputs(ctx, "val", master)
    field_weights = json.loads((ctx.work / "s03_results.json").read_text())["weights"]
    bi = load_bi_encoder(models, device)
    out = run_full_stage(ctx, stage, index, store, bi, None, field_weights, cfg["rrf"]["weights"], device=device)

    item_ids, qrels = stage.corpus["item_id"].to_list(), stage_qrels(stage)
    report = {"ceiling_default_weights": ceiling_report(out, item_ids, qrels)}
    decision = narrowing_decision(out, item_ids, qrels, cfg)
    report["narrowing"] = decision
    (ctx.work / "narrowing_mode.json").write_text(json.dumps(decision), encoding="utf-8")
    print("narrowing:", decision)
    weights = tune_rrf(out, item_ids, qrels, cfg)
    report["rrf_weights"] = weights
    (ctx.work / "rrf_weights_zeroshot.json").write_text(json.dumps(weights), encoding="utf-8")
    (ctx.work / "s05_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
