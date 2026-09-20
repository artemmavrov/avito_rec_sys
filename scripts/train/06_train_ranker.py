"""Step 6 [CPU, ~25 min]: train the CatBoost ranker and measure it on the validation hold-out.

Needs work/feats_ranker.parquet and work/feats_val.parquet (step 5). Writes work/catboost_ranker.cbm
and work/ranker_report.json (Recall@50 on val, overall and per stratum, and the pool ceiling).

Use the result with:
  python predict.py --encoder-dir work/biencoder_ft/epoch2 --ranker-model work/catboost_ranker.cbm
"""

import argparse
import json

import pandas as pd

from avito_rec_sys.config import load_config
from avito_rec_sys.eval.metrics import per_query_recall, recall_at_k, stratified_recall
from avito_rec_sys.paths import add_path_arguments, get_paths
from avito_rec_sys.pipeline.context import build_train_context, label_candidates, make_stage, stage_qrels_by_position
from avito_rec_sys.ranking.ranker import predict_scores, select_top_k, train_ranker


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_arguments(ap)
    args = ap.parse_args()
    cfg, paths = load_config(), get_paths(args.data_dir, args.work_dir, args.weights_dir)
    ctx = build_train_context(cfg, paths)

    # ---- train on the ranker hold-out (the encoders never saw these queries)
    stage = make_stage(ctx, "ranker")
    feats = pd.read_parquet(ctx.work / "feats_ranker.parquet")
    y = label_candidates(feats, stage.corpus["item_id"].to_list(), stage_qrels_by_position(stage))
    print(f"ranker: {len(feats)} pairs over {feats['q'].nunique()} queries, positives in the pool: {int(y.sum())}")
    model = train_ranker(feats, y, cfg["catboost"], cfg["seed"])
    model.save_model(str(ctx.work / "catboost_ranker.cbm"))
    del feats, y

    # ---- measure on the validation hold-out
    stage = make_stage(ctx, "val")
    feats = pd.read_parquet(ctx.work / "feats_val.parquet")
    item_ids = stage.corpus["item_id"].to_list()
    qrels = stage_qrels_by_position(stage)
    n_val = stage.queries.height
    top = select_top_k(feats, predict_scores(model, feats), item_ids, n_val, cfg["pool"]["output_k"])
    pool: dict[int, list[str]] = {q: [] for q in range(n_val)}
    for q, i in zip(feats["q"].to_numpy(), feats["i"].to_numpy()):
        pool[int(q)].append(item_ids[i])
    preds = dict(enumerate(top))
    strata = dict(enumerate(stage.queries["stratum"].to_list()))
    by_stratum = stratified_recall(per_query_recall(preds, qrels, 50), strata)
    report = {
        "val_recall@50": recall_at_k(preds, qrels, 50),
        "val_pool_ceiling": recall_at_k(pool, qrels, 10**9),
        "by_stratum": {k: {"recall@50": v[0], "n": v[1]} for k, v in by_stratum.items()},
    }
    (ctx.work / "ranker_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
