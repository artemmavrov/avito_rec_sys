# CPU-only: dump the fine-tuned tours (global + local rankings) for the val stage, so that pool
# composition experiments run in seconds instead of re-running the pipeline. GPU is busy training.
import pickle, sys, json, time
import numpy as np
from avito_rec_sys.features.logs import log_candidate_positions
from avito_rec_sys.features.tables import ItemTable, QueryTable
from avito_rec_sys.inference.lexical_pipeline import build_context
from avito_rec_sys.inference.neural_pipeline import query_texts
from avito_rec_sys.inference.stages import ensure_master_store, stage_inputs, stage_qrels
from avito_rec_sys.retrieval.encode_store import sparse_to_csr
from avito_rec_sys.retrieval.encoders import encode_bi_encoder, load_bi_encoder
from avito_rec_sys.retrieval.lexical import lexical_candidates
from avito_rec_sys.retrieval.tours import dense_tour, sparse_tour
from avito_rec_sys.training.biencoder_train import final_model_dir
from avito_rec_sys.utils.io import load_config, load_models_config

name = sys.argv[1]
cfg, models = load_config(), load_models_config()
t0 = time.time()
ctx = build_context(cfg)
ft_dir = str(final_model_dir(ctx.work / "biencoder_ft" / "epoch2"))
master = ensure_master_store(ctx, models, "ft", ft_dir, device="cpu")
stage, index, store = stage_inputs(ctx, name, master)
items, queries = ItemTable(stage.corpus, index.vocab), QueryTable(stage.queries, stage.centroids)
bi = load_bi_encoder(models, "cpu", model_path=ft_dir)
enc = encode_bi_encoder(bi, query_texts(stage.queries), 64, cfg["text"]["query_max_tokens"])
q_sparse = sparse_to_csr(enc.sparse)
fw = json.loads((ctx.work / "s03_results.json").read_text())["weights"]
K = cfg["pool_sizes"]["retrieval_k"]
bm25 = lexical_candidates(index, queries.lemma_tokens, queries.loc, items.loc_positions, fw,
                          k_local=K // 2, k_global=K, k_ceiling=K, chunk=64)
d_g, d_l = dense_tour(enc.dense, store.dense, queries.loc, items.loc_positions, k_global=K, k_local=K // 2, device="cpu")
s_g, s_l = sparse_tour(q_sparse, store.sparse, queries.loc, items.loc_positions, k_global=K, k_local=K // 2)
extra = log_candidate_positions(queries, items, stage.logs)
out = {
    "tours_g": {"bm25f": bm25.ranked_global, "dense": d_g, "sparse": s_g},
    "tours_l": {"bm25f": bm25.ranked_local, "dense": d_l, "sparse": s_l},
    "extra": extra, "qrels": stage_qrels(stage), "item_ids": items.ids,
    "item_loc": items.loc, "item_lat": items.lat, "item_lon": items.lon,
    "q_loc": queries.loc, "q_lat": queries.lat, "q_lon": queries.lon,
    "q_norm": queries.norm, "q_stratum": stage.queries["stratum"].to_list() if "stratum" in stage.queries.columns else None,
    "colbert_q": None,
}
pickle.dump(out, open(ctx.work / f"analysis_{name}_tours.pkl", "wb"))
print("dumped", name, len(queries.loc), "queries in", round(time.time() - t0), "s")
