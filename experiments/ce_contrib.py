# Does the fine-tuned CE help INSIDE the ranker? Val-only split (2000 train / 500 test queries), fixed iterations, CPU.
import numpy as np, pandas as pd
from catboost import CatBoost, Pool
from avito_rec_sys.eval.metrics import recall_at_k
from avito_rec_sys.inference.lexical_pipeline import build_context, make_stage, labels_for
from avito_rec_sys.inference.neural_pipeline import FEATURE_COLUMNS_NEURAL
from avito_rec_sys.inference.stages import stage_qrels
from avito_rec_sys.utils.io import load_config
ctx = build_context(load_config()); st = make_stage(ctx, "val"); qrels = stage_qrels(st); ids = st.corpus["item_id"].to_list()
f = pd.read_parquet(ctx.work / "feats_val.parquet")
class It: pass
it = It(); it.ids = ids
y = labels_for(f, it, st.queries["query_id"].to_list(), st.qrels)
rng = np.random.RandomState(0); order = rng.permutation(2500)
def run(cols, fold, iters=250):
    te_q = set(order[fold * 500:(fold + 1) * 500].tolist()); mask = f["q"].isin(te_q).to_numpy()
    tr, te = f[~mask], f[mask]
    m = CatBoost({"loss_function": "YetiRank", "iterations": iters, "learning_rate": 0.1, "depth": 8, "random_seed": 0,
                  "thread_count": 8, "verbose": 0, "allow_writing_files": False})
    m.fit(Pool(tr[cols].to_numpy(np.float32), label=y[~mask], group_id=tr["q"].to_numpy()))
    d = te[["q", "i"]].assign(s=m.predict(te[cols].to_numpy(np.float32))).sort_values(["q", "s"], ascending=[True, False], kind="stable")
    top = {q: [ids[i] for i in g["i"].to_numpy()[:50]] for q, g in d.groupby("q", sort=False)}
    return recall_at_k({q: top.get(q, []) for q in te_q}, {q: qrels[q] for q in te_q}, 50), m
noce = [c for c in FEATURE_COLUMNS_NEURAL if c != "ce_logit"]
for fold in (0, 1):
    a, _ = run(noce, fold); b, m = run(FEATURE_COLUMNS_NEURAL, fold)
    print(f"fold {fold}: without CE {a:.4f} | with CE {b:.4f} | delta {100*(b-a):+.2f} pp", flush=True)
imp = sorted(zip(FEATURE_COLUMNS_NEURAL, m.get_feature_importance()), key=lambda x: -x[1])[:10]
print("top features:", [(k, round(float(v), 1)) for k, v in imp])
