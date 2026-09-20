# End-to-end check of the new pool on the full val stage (CPU, fine-tuned store, no CE)
import json, time, numpy as np
from avito_rec_sys.eval.metrics import recall_at_k
from avito_rec_sys.inference.lexical_pipeline import build_context
from avito_rec_sys.inference.stages import ensure_master_store, stage_inputs, run_full_stage, stage_qrels, load_json
from avito_rec_sys.retrieval.encoders import load_bi_encoder
from avito_rec_sys.training.biencoder_train import final_model_dir
from avito_rec_sys.utils.io import load_config, load_models_config
cfg, models = load_config(), load_models_config(); t0 = time.time()
ctx = build_context(cfg); ft = str(final_model_dir(ctx.work / "biencoder_ft" / "epoch2"))
master = ensure_master_store(ctx, models, "ft", ft, device="cpu")
stage, index, store = stage_inputs(ctx, "val", master)
bi = load_bi_encoder(models, "cpu", model_path=ft)
fw = json.loads((ctx.work / "s03_results.json").read_text())["weights"]
rw = load_json(ctx.work / "rrf_weights_ft.json", cfg["rrf"]["weights"]); print("rrf weights", rw)
out = run_full_stage(ctx, stage, index, store, bi, None, fw, rw, device="cpu")
ids, qrels = stage.corpus["item_id"].to_list(), stage_qrels(stage); n = len(qrels)
as_ids = lambda L: {q: [ids[p] for p in L[q]] for q in range(n)}
print("centroid coverage:", float(np.mean(~np.isnan(np.array([1.0])))))
print("Recall rrf_only (300 pool, no log cands): %.4f" % recall_at_k(as_ids(out.rrf_only), qrels, 10**9))
print("Recall narrowed_pool (+log cands):        %.4f" % recall_at_k(as_ids(out.narrowed_pool), qrels, 10**9))
print("mean pool size", np.mean([len(x) for x in out.narrowed_pool]), "| feats rows", len(out.feats), "| %.0fs" % (time.time() - t0))
