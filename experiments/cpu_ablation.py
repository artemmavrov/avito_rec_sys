# CPU-only ablation while the GPU trains the cross-encoder: new pool + pool features + selection rule.
# Neural columns come from the fine-tuned bi-encoder; ce_logit is a constant here (no CE yet).
import json, time, numpy as np, pandas as pd
from avito_rec_sys.eval.metrics import recall_at_k
from avito_rec_sys.features.tables import ItemTable, QueryTable
from avito_rec_sys.inference.lexical_pipeline import build_context, labels_for
from avito_rec_sys.inference.neural_pipeline import FEATURE_COLUMNS_NEURAL, NEURAL_COLUMNS, POOL_COLUMNS
from avito_rec_sys.features.assemble import FEATURE_COLUMNS
from avito_rec_sys.inference.select import select_top
from avito_rec_sys.inference.stages import ensure_master_store, stage_inputs, run_full_stage, stage_qrels, load_json
from avito_rec_sys.retrieval.encoders import load_bi_encoder
from avito_rec_sys.training.biencoder_train import final_model_dir
from avito_rec_sys.training.catboost_train import train_ranker, predict_scores
from avito_rec_sys.utils.io import load_config, load_models_config
cfg, models = load_config(), load_models_config(); t0 = time.time()
ctx = build_context(cfg); ft = str(final_model_dir(ctx.work / "biencoder_ft" / "epoch2"))
master = ensure_master_store(ctx, models, "ft", ft, device="cpu")
bi = load_bi_encoder(models, "cpu", model_path=ft)
fw = json.loads((ctx.work / "s03_results.json").read_text())["weights"]
rw = load_json(ctx.work / "rrf_weights_ft.json", cfg["rrf"]["weights"])
def build(name, nq):
    stage, index, store = stage_inputs(ctx, name, master, nq)
    out = run_full_stage(ctx, stage, index, store, bi, None, fw, rw, device="cpu")
    items, queries = ItemTable(stage.corpus, index.vocab), QueryTable(stage.queries, stage.centroids)
    y = labels_for(out.feats, items, stage.queries["query_id"].to_list(), stage.qrels)
    print(name, "features ready", len(out.feats), "rows, %.0fs" % (time.time() - t0), flush=True)
    return stage, items, queries, out.feats, y
sr, ir, qr, fr, yr = build("ranker", 3000)
sv, iv, qv, fv, yv = build("val", None)
fr.to_parquet(ctx.work / "cpuabl_ranker.parquet"); fv.to_parquet(ctx.work / "cpuabl_val.parquet")
cut = int(0.9 * qr.n); tr, va = fr["q"] < cut, fr["q"] >= cut
cb = dict(cfg["catboost"]); cb.update(iterations=800, learning_rate=0.08, early_stopping_rounds=100)
qrels = stage_qrels(sv)
def evaluate(cols, label):
    m = train_ranker(fr[tr], yr[tr], fr[va], yr[va], cols, cb, cfg["seed"], thread_count=8)
    sc = predict_scores(m, fv, cols)
    with_slots = select_top(fv, sc, iv, qv, cfg["slots"])
    d = fv[["q", "i"]].assign(s=sc).sort_values(["q", "s"], ascending=[True, False], kind="stable")
    top = {q: [iv.ids[i] for i in g["i"].to_numpy()[:50]] for q, g in d.groupby("q", sort=False)}
    r_slots = recall_at_k({q: with_slots[q] for q in range(qv.n)}, qrels, 50)
    r_top = recall_at_k({q: top.get(q, []) for q in range(qv.n)}, qrels, 50)
    print(f"[{label}] val Recall@50: slots {r_slots:.4f} | plain top-50 {r_top:.4f} | iters {m.tree_count_} | %.0fs" % (time.time() - t0), flush=True)
    return m
pool_ceiling = recall_at_k({q: [iv.ids[i] for i in fv.loc[fv["q"] == q, "i"]] for q in range(0, qv.n)}, qrels, 10**9)
print("val pool ceiling: %.4f" % pool_ceiling, flush=True)
base_cols = FEATURE_COLUMNS + [c for c in NEURAL_COLUMNS if c != "ce_logit"]
evaluate(base_cols, "A: old features")
m = evaluate(base_cols + POOL_COLUMNS, "B: + pool features")
imp = sorted(zip(base_cols + POOL_COLUMNS, m.get_feature_importance()), key=lambda x: -x[1])[:12]
print("top features B:", [(k, round(float(v), 1)) for k, v in imp])
