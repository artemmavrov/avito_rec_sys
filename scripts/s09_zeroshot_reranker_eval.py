"""Stage 9 [GPU, ~35 min on a 3090]: zero-shot cross-encoder on the validation
queries with the FINE-TUNED retriever, and the decision gate of §12.1.

Measures Recall@300 (the pool the cross-encoder reads) against Recall@50
(what survives when the pool is cut to 50 by the cross-encoder alone). That
gap is what a better reranker could still recover:

    gap <  3 pp   -> selection loses almost nothing: do NOT fine-tune the CE,
                     spend the ~136 min on a third bi-encoder epoch instead
    3-10 pp       -> fine-tune the CE (§5.5) -- the base case
    > 10 pp       -> selection is the bottleneck: fine-tune the CE and consider
                     raising the reranker K to 500

Also re-tunes the RRF weights on the fine-tuned tours (CPU) and reports the
share of positives no tour finds at all (1 - oracle-union Recall@1000).
"""

import argparse
import json
import time

import pandas as pd

from avito_rec_sys.eval.metrics import recall_at_k
from avito_rec_sys.inference.lexical_pipeline import build_context
from avito_rec_sys.inference.stages import (
    ceiling_report, ensure_master_store, narrowing_decision, run_full_stage, stage_inputs, stage_qrels, tune_rrf,
)
from avito_rec_sys.retrieval.encoders import load_bi_encoder, resolve_snapshot
from avito_rec_sys.retrieval.reranker import CrossEncoderScorer
from avito_rec_sys.training.biencoder_train import final_model_dir
from avito_rec_sys.utils.io import load_config, load_models_config


def top_by_score(feats: pd.DataFrame, col: str, item_ids: list[str], n_q: int, k: int) -> dict[int, list[str]]:
    df = feats[["q", "i", col]].sort_values(["q", col], ascending=[True, False], kind="stable")
    preds: dict[int, list[str]] = {q: [] for q in range(n_q)}
    for q, g in df.groupby("q", sort=False):
        preds[int(q)] = [item_ids[i] for i in g["i"].to_numpy()[:k]]
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-queries", type=int, default=None,
                    help="score only the first N validation queries (the gate needs a coarse gap, not all 2500)")
    max_queries = ap.parse_args().max_queries
    cfg, models = load_config(), load_models_config()
    t0 = time.time()
    ctx = build_context(cfg)
    ft_dir = str(final_model_dir(ctx.work / "biencoder_ft" / "epoch2"))
    master = ensure_master_store(ctx, models, "ft", ft_dir)
    stage, index, store = stage_inputs(ctx, "val", master, max_queries)
    item_ids, qrels = stage.corpus["item_id"].to_list(), stage_qrels(stage)
    field_weights = json.loads((ctx.work / "s03_results.json").read_text())["weights"]

    bi = load_bi_encoder(models, model_path=ft_dir)
    ce = CrossEncoderScorer(
        resolve_snapshot(models["reranker"]["name"], models["reranker"]["revision"]),
        max_length=cfg["text"]["cross_encoder_max_tokens"],
    )
    out = run_full_stage(ctx, stage, index, store, bi, ce.score, field_weights, cfg["rrf"]["weights"])
    n_q = len(qrels)

    pool: dict[int, list[str]] = {q: [] for q in range(n_q)}
    for q_, i_ in zip(out.feats["q"].to_numpy(), out.feats["i"].to_numpy()):
        pool[int(q_)].append(item_ids[i_])
    r_pool = recall_at_k(pool, qrels, 10**9)
    r50 = recall_at_k(top_by_score(out.feats, "ce_logit", item_ids, n_q, 50), qrels, 50)
    r50_colbert = recall_at_k(top_by_score(out.feats, "colbert_maxsim", item_ids, n_q, 50), qrels, 50)
    gap_pp = 100 * (r_pool - r50)

    g = cfg["reranker_gate"]
    decision = (
        "SKIP CE fine-tuning; give the time to a third bi-encoder epoch" if gap_pp < g["recall_gap_no_finetune_below_pp"]
        else "fine-tune CE, consider reranker K=500" if gap_pp > g["recall_gap_finetune_above_pp"]
        else "fine-tune CE (base case, §5.5)"
    )
    ceiling = ceiling_report(out, item_ids, qrels)
    union_1000 = ceiling["union"]["1000"]
    report = {
        "recall@pool(300+log)": r_pool, "recall@50_by_zero_shot_ce": r50, "recall@50_by_colbert": r50_colbert,
        "gap_pp": gap_pp, "decision": decision,
        "positives_no_tour_finds": 1 - union_1000, "ceiling": ceiling,
        "rrf_weights_ft": tune_rrf(out, item_ids, qrels, cfg),
        # ColBERT quality changes with fine-tuning, so the stage-5 decision is redone here
        "narrowing_ft": narrowing_decision(out, item_ids, qrels, cfg),
    }
    (ctx.work / "narrowing_mode.json").write_text(json.dumps(report["narrowing_ft"]), encoding="utf-8")
    (ctx.work / "rrf_weights_ft.json").write_text(json.dumps(report["rrf_weights_ft"]), encoding="utf-8")
    (ctx.work / "s09_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "ceiling"}, indent=2))
    print(f"total {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
