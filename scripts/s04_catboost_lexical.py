"""Stage 4: CatBoost YetiRank on the non-neural feature bank, evaluated on the
validation queries, plus a CPU-only baseline answer.csv for the benchmark.

Training data comes from the `ranker` hold-out (encoders never see it); the
last 10% of its queries are the early-stopping set so the `val` hold-out
stays untouched by model selection.
"""

import gc
import json
import time

import numpy as np
import pandas as pd

from avito_rec_sys.eval.metrics import ceiling_curve, per_query_recall, recall_at_k, stratified_recall
from avito_rec_sys.features.assemble import FEATURE_COLUMNS
from avito_rec_sys.inference.answer_writer import write_answer
from avito_rec_sys.inference.lexical_pipeline import (
    build_context, build_index, candidates_and_features, labels_for, make_stage,
)
from avito_rec_sys.inference.select import select_top
from avito_rec_sys.inference.validator import validate_answer
from avito_rec_sys.training.catboost_train import build_pool, predict_scores, train_ranker
from avito_rec_sys.utils.io import load_config


def run_stage(ctx, name, weights):
    stage = make_stage(ctx, name)
    index = build_index(stage.corpus)
    items, queries, cand, feats = candidates_and_features(stage, index, weights)
    num = feats.select_dtypes("float64").columns
    feats[num] = feats[num].astype(np.float32)
    return stage, items, queries, cand, feats


def main() -> None:
    cfg = load_config()
    t0 = time.time()
    ctx = build_context(cfg)
    weights = json.loads((ctx.work / "s03_results.json").read_text())["weights"]
    print("field weights from s03:", weights)

    # ---- train on the ranker hold-out
    stage, items, queries, cand, feats = run_stage(ctx, "ranker", weights)
    qids = stage.queries["query_id"].to_list()
    y = labels_for(feats, items, qids, stage.qrels)
    n_pos_in_pool = int(y.sum())
    n_pos_total = sum(len(v) for v in stage.qrels.values())
    print(f"ranker: {len(feats)} pairs, positives in pool {n_pos_in_pool}/{n_pos_total} ({n_pos_in_pool / n_pos_total:.3f}), {time.time() - t0:.0f}s")
    cut = int(0.9 * queries.n)
    tr, va = feats["q"] < cut, feats["q"] >= cut
    model = train_ranker(feats[tr], y[tr], feats[va], y[va], FEATURE_COLUMNS, cfg["catboost"], cfg["seed"])
    model.save_model(str(ctx.work / "catboost_lexical.cbm"))
    imp = sorted(
        zip(FEATURE_COLUMNS, model.get_feature_importance(build_pool(feats[tr], y[tr], FEATURE_COLUMNS))),
        key=lambda x: -x[1],
    )
    print("top features:", [(k, round(float(v), 2)) for k, v in imp[:12]])
    del stage, items, queries, cand, feats
    gc.collect()

    # ---- evaluate on val
    stage, items, queries, cand, feats = run_stage(ctx, "val", weights)
    vq = stage.queries["query_id"].to_list()
    qrels = {q: stage.qrels[vq[q]] for q in range(queries.n)}
    scores = predict_scores(model, feats, FEATURE_COLUMNS)
    preds = select_top(feats, scores, items, queries, cfg["slots"])
    preds_d = {q: preds[q] for q in range(queries.n)}
    r50 = recall_at_k(preds_d, qrels, 50)

    # pool ceiling: what a perfect ranker could get from this candidate pool
    pool = {q: [] for q in range(queries.n)}
    for q_, i_ in zip(feats["q"].to_numpy(), feats["i"].to_numpy()):
        pool[int(q_)].append(items.ids[i_])
    pool_ceiling = recall_at_k(pool, qrels, 10**9)
    pq = per_query_recall(preds_d, qrels, 50)
    stratum = dict(zip(range(queries.n), stage.queries["stratum"].to_list()))
    result = {
        "val_recall@50": r50,
        "val_pool_ceiling": pool_ceiling,
        "pool_size_mean": len(feats) / queries.n,
        "by_stratum": {k: {"recall@50": v[0], "n": v[1]} for k, v in stratified_recall(pq, stratum).items()},
        "top_features": [(k, float(v)) for k, v in imp[:15]],
    }
    print(json.dumps(result, indent=2))
    (ctx.work / "s04_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    del stage, items, queries, cand, feats
    gc.collect()

    # ---- benchmark answer (CPU-only baseline)
    stage, items, queries, cand, feats = run_stage(ctx, "test", weights)
    scores = predict_scores(model, feats, FEATURE_COLUMNS)
    preds = select_top(feats, scores, items, queries, cfg["slots"])
    qids = stage.queries["query_id"].to_list()
    out = write_answer(ctx.work / "answer_lexical.csv", qids, {qids[q]: preds[q] for q in range(queries.n)})
    problems = validate_answer(out, qids, set(items.ids))
    print("answer.csv problems:", problems or "none", "| empty answers:", sum(1 for p in preds if not p))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
