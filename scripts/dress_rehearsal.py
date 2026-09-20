"""Dress rehearsal: run the REAL stage scripts s01..s11 end to end on a tiny
slice of the real data, with the real models and the smoke-test profile.

Purpose: find wiring bugs (paths, shapes, caches, resume, checkpoint reloads,
file formats between stages) on a laptop instead of on a rented GPU. It says
nothing about quality -- the slice is a few thousand items and every training
stage runs a handful of steps.

    python scripts/dress_rehearsal.py [--out ../work_rehearsal] [--start s05]

Everything lands in --out (separate data slice, work dir and answer file), so
the real work/ directory and answer.csv are never touched. ~30-60 min on a
4 GB GPU, mostly checkpoint I/O.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

REPO = Path(__file__).resolve().parents[1]
STAGES = [
    "s01_split_and_normalize", "s02_clean_lemmatize_params", "s03_bm25f_baseline", "s03b_mine_bm25_negatives", "s04_catboost_lexical",
    "s05_zeroshot_biencoder", "s06_mine_hard_negatives", "s07_train_biencoder", "s08_reencode_mine_pools",
    "s09_zeroshot_reranker_eval", "s10_train_reranker", "s11_final_inference",
]


def make_slice(real_cfg: dict, out: Path, n_texts: int, n_bench_items: int, n_bench_queries: int, seed: int = 0) -> Path:
    from avito_rec_sys.utils.io import resolve_path

    rng = np.random.RandomState(seed)
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)

    train_path = resolve_path(real_cfg, "train_parquet")
    queries = pl.read_parquet(train_path, columns=["search_query"])["search_query"].unique().sort().to_list()
    keep = set(rng.choice(queries, size=n_texts, replace=False).tolist())
    pl.scan_parquet(train_path).filter(pl.col("search_query").is_in(keep)).collect().write_parquet(data / "train.parquet")

    items_path = resolve_path(real_cfg, "benchmark_items_parquet")
    ids = pl.read_parquet(items_path, columns=["item_id"])["item_id"].sort().to_list()
    keep_ids = set(rng.choice(ids, size=n_bench_items, replace=False).tolist())
    pl.scan_parquet(items_path).filter(pl.col("item_id").is_in(keep_ids)).collect().write_parquet(data / "benchmark_items.parquet")

    bq = pl.read_parquet(resolve_path(real_cfg, "benchmark_queries_parquet"))
    bq.sample(n=n_bench_queries, seed=seed).write_parquet(data / "benchmark_queries.parquet")
    return data


def write_config(out: Path, data: Path) -> Path:
    from avito_rec_sys.utils.io import load_config

    os.environ.pop("AVITO_PIPELINE_CONFIG", None)
    cfg = load_config("smoke_test")
    cfg["paths"].update(
        {
            "data_dir": str(data),
            "train_parquet": str(data / "train.parquet"),
            "benchmark_queries_parquet": str(data / "benchmark_queries.parquet"),
            "benchmark_items_parquet": str(data / "benchmark_items.parquet"),
            "work_dir": str(out / "work"),
            "reports_dir": str(out / "reports"),
            "answer_csv": str(out / "answer_final.csv"),
        }
    )
    cfg["split"].update({"n_validation_queries": 40, "n_ranker_queries": 60})
    cfg["catboost"].update(
        {"iterations": 30, "early_stopping_rounds": 5, "depth": 4, "learning_rate": 0.1, "train_queries": 50,
         "train_rows_per_query": 12}  # < pool size, so row thinning is exercised
    )
    cfg["biencoder_train"]["checkpoint_every_steps"] = cfg["biencoder_train"]["max_steps"]
    cfg["reranker_train"]["checkpoint_every_steps"] = cfg["reranker_train"]["max_steps"]
    path = out / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO.parent / "work_rehearsal"))
    ap.add_argument("--start", default="s01", help="resume from this stage prefix, e.g. s05")
    ap.add_argument("--texts", type=int, default=2500, help="distinct train query texts in the slice")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    from avito_rec_sys.utils.io import load_config

    real_cfg = load_config()
    if not (out / "data" / "train.parquet").exists():
        print("building data slice ...")
        data = make_slice(real_cfg, out, args.texts, n_bench_items=2000, n_bench_queries=40)
    else:
        data = out / "data"
    cfg_path = write_config(out, data)

    env = {**os.environ, "AVITO_PIPELINE_CONFIG": str(cfg_path), "PYTHONIOENCODING": "utf-8"}
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    started = False
    for stage in STAGES:
        started = started or stage.startswith(args.start)
        if not started:
            continue
        t0 = time.time()
        log = logs / f"{stage}.log"
        with open(log, "w", encoding="utf-8") as f:
            rc = subprocess.run(
                [sys.executable, str(REPO / "scripts" / f"{stage}.py")], env=env, cwd=REPO, stdout=f, stderr=subprocess.STDOUT
            ).returncode
        print(f"{stage:32s} {'OK ' if rc == 0 else 'FAIL'} {time.time() - t0:6.0f}s")
        if rc != 0:
            print("---- last lines of", log)
            print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]))
            sys.exit(1)
    print("dress rehearsal finished; answer:", out / "answer_final.csv")


if __name__ == "__main__":
    main()
