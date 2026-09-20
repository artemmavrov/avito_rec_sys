# CPU tuning on the cached final feature tables: iterations, CE on/off. Val Recall@50 with plain top-50.
import sys, time, numpy as np, pandas as pd
from catboost import CatBoost, Pool
from avito_rec_sys.eval.metrics import recall_at_k
from avito_rec_sys.inference.lexical_pipeline import build_context, make_stage, labels_for
from avito_rec_sys.inference.neural_pipeline import FEATURE_COLUMNS_NEURAL
from avito_rec_sys.inference.stages import stage_qrels
from avito_rec_sys.utils.io import load_config
cfg = load_config(); ctx = build_context(cfg); t0 = time.time()
class It: pass
def load(name):
    st = make_stage(ctx, name); it = It(); it.ids = st.corpus["item_id"].to_list()
    f = pd.read_parquet(ctx.work / f"feats_{name}.parquet")
    return st, it, f, labels_for(f, it, st.queries["query_id"].to_list(), st.qrels)
sr, ir, fr, yr = load("ranker"); sv, iv, fv, yv = load("val"); qv = stage_qrels(sv); nq = len(qv)
print("loaded %.0fs" % (time.time() - t0), flush=True)
def run(cols, iters, lr=0.05, depth=8, seed=0, tag=""):
    m = CatBoost({"loss_function": "YetiRank", "iterations": iters, "learning_rate": lr, "depth": depth, "random_seed": seed,
                  "thread_count": 16, "verbose": 0, "allow_writing_files": False})
    m.fit(Pool(fr[cols].to_numpy(np.float32), label=yr, group_id=fr["q"].to_numpy()))
    d = fv[["q", "i"]].assign(s=m.predict(fv[cols].to_numpy(np.float32))).sort_values(["q", "s"], ascending=[True, False], kind="stable")
    top = {q: [iv.ids[i] for i in g["i"].to_numpy()[:50]] for q, g in d.groupby("q", sort=False)}
    r = recall_at_k({q: top.get(q, []) for q in range(nq)}, qv, 50)
    print(f"{tag:28s} it={iters} lr={lr} depth={depth}: val R@50 {r:.4f}  ({time.time()-t0:.0f}s)", flush=True); return r
noce = [c for c in FEATURE_COLUMNS_NEURAL if c != "ce_logit"]
run(FEATURE_COLUMNS_NEURAL, 600, tag="with CE")
run(noce, 600, tag="without CE")
run(FEATURE_COLUMNS_NEURAL, 1200, tag="with CE, longer")
run(FEATURE_COLUMNS_NEURAL, 600, depth=6, lr=0.08, tag="with CE, depth 6")
