# Quick read on the cached val table: how good is each single signal at picking the top 50 from the pool?
import numpy as np, pandas as pd
from avito_rec_sys.eval.metrics import recall_at_k
from avito_rec_sys.inference.lexical_pipeline import build_context, make_stage
from avito_rec_sys.inference.stages import stage_qrels
from avito_rec_sys.utils.io import load_config
ctx = build_context(load_config()); st = make_stage(ctx, "val"); qrels = stage_qrels(st); ids = st.corpus["item_id"].to_list(); n = len(qrels)
f = pd.read_parquet(ctx.work / "feats_val.parquet")
print("rows", len(f), "queries", f["q"].nunique(), "| pool recall", recall_at_k({q: [ids[i] for i in g["i"]] for q, g in f.groupby("q")}, qrels, 10**9))
for col in ["ce_logit", "colbert_maxsim", "dense_cos", "bm25f_total", "pool_rank"]:
    sign = -1 if col == "pool_rank" else 1
    d = f[["q", "i", col]].assign(s=sign * f[col]).sort_values(["q", "s"], ascending=[True, False], kind="stable")
    top = {q: [ids[i] for i in g["i"].to_numpy()[:50]] for q, g in d.groupby("q", sort=False)}
    print(f"{col:16s} alone -> Recall@50 {recall_at_k({q: top.get(q, []) for q in range(n)}, qrels, 50):.4f}")
