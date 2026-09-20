# Robust re-run of the feature ablation on the cached CPU feature tables: fixed iterations (no noisy early
# stopping on 300 queries), 3 seeds, plain top-50 selection. CPU only.
import numpy as np, pandas as pd, time
from catboost import CatBoost, Pool
from avito_rec_sys.eval.metrics import recall_at_k
from avito_rec_sys.features.assemble import FEATURE_COLUMNS
from avito_rec_sys.features.tables import ItemTable
from avito_rec_sys.inference.lexical_pipeline import build_context, build_index, labels_for, make_stage
from avito_rec_sys.inference.neural_pipeline import NEURAL_COLUMNS, POOL_COLUMNS
from avito_rec_sys.inference.stages import stage_qrels
from avito_rec_sys.utils.io import load_config
cfg = load_config(); ctx = build_context(cfg); t0 = time.time()
def labels(name, feats, nq=None):
    st = make_stage(ctx, name)
    if nq: st = type(st)(st.name, st.corpus, st.queries.head(nq), st.qrels, st.logs, st.centroids)
    class It: pass
    it = It(); it.ids = st.corpus["item_id"].to_list()
    return st, labels_for(feats, it, st.queries["query_id"].to_list(), st.qrels)
fr = pd.read_parquet(ctx.work / "cpuabl_ranker.parquet"); fv = pd.read_parquet(ctx.work / "cpuabl_val.parquet")
sr, yr = labels("ranker", fr, 3000); sv, yv = labels("val", fv)
qrels = stage_qrels(sv); ids = sv.corpus["item_id"].to_list(); nq = len(qrels)
print("data ready %.0fs" % (time.time() - t0), flush=True)
def fit_eval(cols, iters=400, lr=0.08, depth=8, seed=0, loss="YetiRank"):
    m = CatBoost({"loss_function": loss, "iterations": iters, "learning_rate": lr, "depth": depth, "random_seed": seed,
                  "thread_count": 8, "verbose": 0, "allow_writing_files": False})
    m.fit(Pool(fr[cols].to_numpy(np.float32), label=yr, group_id=fr["q"].to_numpy()))
    sc = m.predict(fv[cols].to_numpy(np.float32))
    d = fv[["q", "i"]].assign(s=sc).sort_values(["q", "s"], ascending=[True, False], kind="stable")
    top = {q: [ids[i] for i in g["i"].to_numpy()[:50]] for q, g in d.groupby("q", sort=False)}
    return recall_at_k({q: top.get(q, []) for q in range(nq)}, qrels, 50)
base = FEATURE_COLUMNS + [c for c in NEURAL_COLUMNS if c != "ce_logit"]
sets = {"A old": base, "B +pool feats": base + POOL_COLUMNS}
for name, cols in sets.items():
    r = [fit_eval(cols, iters=300, seed=s) for s in range(2)]
    print(f"{name:20s} R@50 {np.mean(r):.4f}  (seeds {', '.join('%.4f' % x for x in r)})  %.0fs" % (time.time() - t0), flush=True)
