"""Stage 11 [GPU + CPU, ~1.5-2 h on a 3090]: final ranker and answer.csv.

  1. build neural features for the `ranker` hold-out, `val` and `test`
     (cached to work/feats_*.parquet: the cross-encoder is the expensive part).
     Ranker pools are thinned to `catboost.train_rows_per_query` rows before the
     cross-encoder runs (see training/thinning.py); the last `es_fraction` of
     the queries keeps full pools for early stopping,
  2. train CatBoost YetiRank on the ranker hold-out,
  3. report Recall@50 on val (per stratum, plus pool ceilings),
  4. predict the benchmark, allocate slots, write + validate answer.csv.

Flags: --zero-shot-ce  use the base reranker (when the §12.1 gate says skip).
       --no-ce  drop the ce_logit column from the ranker (it added nothing measurable, see reports/04)
       --iterations N  fixed CatBoost iterations on ALL ranker queries, no early stopping
       --features-only STAGE  (ranker | val | test) only build and cache that stage's feature table.
           Stages are independent, so running the three concurrently on one GPU hides each one's
           CPU-only phases (indexes, BM25, feature bank) behind another's cross-encoder scoring;
           a final plain run then finds all three caches and only trains + writes the answer.
"""

import argparse
import gc
import json
import shutil
import time

import numpy as np
import pandas as pd

from avito_rec_sys.eval.metrics import per_query_recall, recall_at_k, stratified_recall
from avito_rec_sys.features.tables import ItemTable, QueryTable
from avito_rec_sys.inference.answer_writer import write_answer
from avito_rec_sys.inference.lexical_pipeline import build_context, labels_for
from avito_rec_sys.inference.neural_pipeline import FEATURE_COLUMNS_NEURAL
from avito_rec_sys.inference.select import select_plain, select_top
from avito_rec_sys.inference.stages import ensure_master_store, load_json, run_full_stage, stage_inputs, stage_qrels
from avito_rec_sys.inference.validator import validate_answer
from avito_rec_sys.retrieval.encoders import load_bi_encoder, resolve_snapshot
from avito_rec_sys.retrieval.reranker import CrossEncoderScorer
from avito_rec_sys.training.biencoder_train import final_model_dir
from avito_rec_sys.training.catboost_train import build_pool, predict_scores, train_ranker
from avito_rec_sys.training.thinning import thin_pools
from avito_rec_sys.utils.io import load_config, load_models_config, resolve_path


def es_cut(n_queries: int, es_fraction: float) -> int:
    """Queries at positions >= cut form the early-stopping set."""
    return int((1 - es_fraction) * n_queries)


def make_thin(stage, rows_per_query: int, es_fraction: float, seed: int):
    qids = stage.queries["query_id"].to_list()
    cut = es_cut(len(qids), es_fraction)

    def thin(feats, items, queries):
        y = labels_for(feats, items, qids, stage.qrels)
        return thin_pools(
            feats["q"].to_numpy(), y, feats["colbert_maxsim"].to_numpy(), rows_per_query, cut, np.random.RandomState(seed)
        )

    return thin


def stage_features(ctx, name, master, bi, ce, field_weights, rrf_weights, max_queries=None, thin_cfg=None):
    """Features for a stage, cached on disk. Returns (stage, items, queries, feats)."""
    stage, index, store = stage_inputs(ctx, name, master, max_queries)
    cache = ctx.work / f"feats_{name}.parquet"
    if cache.exists():
        feats = pd.read_parquet(cache)
    else:
        thin = make_thin(stage, **thin_cfg) if thin_cfg else None
        feats = run_full_stage(ctx, stage, index, store, bi, ce.score, field_weights, rrf_weights, thin=thin).feats
        feats.to_parquet(cache)
    items = ItemTable(stage.corpus, index.vocab)
    queries = QueryTable(stage.queries, stage.centroids)
    return stage, items, queries, feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zero-shot-ce", action="store_true")
    ap.add_argument("--features-only", choices=["ranker", "val", "test"])
    ap.add_argument("--no-ce", action="store_true")
    ap.add_argument("--iterations", type=int, default=None)
    args = ap.parse_args()

    cfg, models = load_config(), load_models_config()
    t0 = time.time()
    ctx = build_context(cfg)
    ft_dir = str(final_model_dir(ctx.work / "biencoder_ft" / "epoch2"))
    master = ensure_master_store(ctx, models, "ft", ft_dir)
    field_weights = json.loads((ctx.work / "s03_results.json").read_text())["weights"]
    rrf_weights = load_json(ctx.work / "rrf_weights_ft.json", cfg["rrf"]["weights"])
    print("rrf weights:", rrf_weights)

    ce_path = resolve_snapshot(models["reranker"]["name"], models["reranker"]["revision"])
    if not args.zero_shot_ce:
        ckpts = sorted((ctx.work / "reranker_ft").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        ce_path = str(ckpts[-1])
    print("cross-encoder:", ce_path)
    bi = load_bi_encoder(models, model_path=ft_dir)
    ce = CrossEncoderScorer(ce_path, max_length=cfg["text"]["cross_encoder_max_tokens"])

    cb = cfg["catboost"]
    thin_cfg = {"rows_per_query": cb["train_rows_per_query"], "es_fraction": cb["es_fraction"], "seed": cfg["seed"]}
    if args.features_only:
        is_ranker = args.features_only == "ranker"
        stage_features(ctx, args.features_only, master, bi, ce, field_weights, rrf_weights,
                       cb["train_queries"] if is_ranker else None, thin_cfg if is_ranker else None)
        print(f"features cached: {args.features_only}; total {(time.time() - t0) / 60:.0f} min")
        return

    # ---- 1-2. train the ranker
    stage, items, queries, feats = stage_features(
        ctx, "ranker", master, bi, ce, field_weights, rrf_weights, cb["train_queries"], thin_cfg
    )
    y = labels_for(feats, items, stage.queries["query_id"].to_list(), stage.qrels)
    print(f"ranker: {len(feats)} pairs, positives in pool {int(y.sum())}/{sum(len(v) for v in stage.qrels.values())}")
    cut = es_cut(queries.n, cb["es_fraction"])  # rows of queries >= cut are unthinned full pools
    tr, va = feats["q"] < cut, feats["q"] >= cut
    columns = [c for c in FEATURE_COLUMNS_NEURAL if not (args.no_ce and c == "ce_logit")]
    cb_cfg = dict(cb)
    if args.iterations:  # fixed iterations, every ranker query is training data
        cb_cfg.update(iterations=args.iterations, early_stopping_rounds=0)
        tr = feats["q"] >= 0
    tag = "_v2" if (args.no_ce or args.iterations) else ""  # keeps the v1 artifacts intact
    model = train_ranker(feats[tr], y[tr], feats[va], y[va], columns, cb_cfg, cfg["seed"])
    model.save_model(str(ctx.work / f"catboost_final{tag}.cbm"))
    imp = sorted(
        zip(columns, model.get_feature_importance(build_pool(feats[tr], y[tr], columns))),
        key=lambda x: -x[1],
    )
    del feats, items, queries, stage
    gc.collect()

    # ---- 3. validation
    stage, items, queries, feats = stage_features(ctx, "val", master, bi, ce, field_weights, rrf_weights)
    qrels = stage_qrels(stage)
    val_scores = predict_scores(model, feats, columns)
    # two selection rules, the better one on validation is used for the benchmark
    candidates = {
        "slots": select_top(feats, val_scores, items, queries, cfg["slots"]),
        "plain": select_plain(feats, val_scores, items, queries, cfg["slots"]["output_k"]),
    }
    by_rule = {k: recall_at_k({q: v[q] for q in range(queries.n)}, qrels, 50) for k, v in candidates.items()}
    selection = max(by_rule, key=by_rule.get)
    print("selection rule recall@50 on val:", by_rule, "->", selection)
    preds = candidates[selection]
    preds_d = {q: preds[q] for q in range(queries.n)}
    pool: dict[int, list[str]] = {q: [] for q in range(queries.n)}
    for q_, i_ in zip(feats["q"].to_numpy(), feats["i"].to_numpy()):
        pool[int(q_)].append(items.ids[i_])
    pq = per_query_recall(preds_d, qrels, 50)
    stratum = dict(zip(range(queries.n), stage.queries["stratum"].to_list()))
    report = {
        "val_recall@50": recall_at_k(preds_d, qrels, 50),
        "selection": selection, "val_recall@50_by_rule": by_rule,
        "val_pool_ceiling": recall_at_k(pool, qrels, 10**9),
        "by_stratum": {k: {"recall@50": v[0], "n": v[1]} for k, v in stratified_recall(pq, stratum).items()},
        "top_features": [(k, float(v)) for k, v in imp[:15]],
    }
    (ctx.work / f"s11_results{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    del feats, items, queries, stage
    gc.collect()

    # ---- 4. benchmark answer
    stage, items, queries, feats = stage_features(ctx, "test", master, bi, ce, field_weights, rrf_weights)
    test_scores = predict_scores(model, feats, columns)
    preds = (select_plain(feats, test_scores, items, queries, cfg["slots"]["output_k"]) if selection == "plain"
             else select_top(feats, test_scores, items, queries, cfg["slots"]))
    qids = stage.queries["query_id"].to_list()
    out = write_answer(ctx.work / f"answer{tag}.csv", qids, {qids[q]: preds[q] for q in range(queries.n)})
    problems = validate_answer(out, qids, set(items.ids))
    print("answer.csv problems:", problems or "none", "| empty answers:", sum(1 for p in preds if not p))
    if not problems:
        shutil.copy(out, resolve_path(cfg, "answer_csv"))
    print(f"total {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
