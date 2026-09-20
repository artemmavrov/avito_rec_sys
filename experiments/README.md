# One-off analysis scripts

Not part of the pipeline. They produced the numbers in `reports/04_gpu_run_log.md` and need the artifacts in `work/`
(cached feature tables, tour dumps) of a finished GPU run; paths are relative to the repo root.

| Script | What it measured |
|---|---|
| `dump_tours.py`, `pool_exp.py` | pool composition variants, geo "nearby" list, centroid fill from train clicks |
| `verify_pool.py` | end-to-end check of the new pool on the full validation stage |
| `cpu_ablation.py`, `ablation2.py` | CatBoost feature ablation (pool features, slots vs plain top-50) |
| `ce_quick.py`, `ce_contrib.py` | single-signal Recall@50 and the cross-encoder's contribution inside CatBoost |
| `tune_ranker.py` | CatBoost iterations / depth / with-without cross-encoder |
