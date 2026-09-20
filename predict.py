"""Inference: benchmark queries -> answer.csv (up to 50 item_ids per query).

    python predict.py --data-dir /path/to/data

What it does (all local, no external API; only the model weights are downloaded, once):
  1. downloads the released weights from Google Drive and verifies their sha256,
  2. encodes the 189k benchmark items with the fine-tuned bge-m3 (dense + sparse + ColBERT), cached in --work-dir,
  3. builds the candidate pool of every query and its features,
  4. scores the candidates with the CatBoost ranker and keeps the best 50,
  5. writes answer.csv, validates its format and compares it with the submitted reference answer.
"""

import argparse
import time
from pathlib import Path

from avito_rec_sys.config import load_config, load_models_config
from avito_rec_sys.inference.answer import compare_answers, write_answer
from avito_rec_sys.inference.validator import validate_answer
from avito_rec_sys.inference.weights import ensure_weights, sha256_of
from avito_rec_sys.paths import REPO_ROOT, add_path_arguments, get_paths
from avito_rec_sys.pipeline.candidates import CandidateGenerator, item_tower_texts
from avito_rec_sys.pipeline.context import make_benchmark_stage
from avito_rec_sys.ranking.ranker import load_ranker, predict_scores, select_top_k
from avito_rec_sys.retrieval.encode_store import load_or_build_store
from avito_rec_sys.retrieval.encoders import load_bi_encoder

REFERENCE_ANSWER = REPO_ROOT / "reference" / "answer.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_arguments(ap)
    ap.add_argument("--output", default="answer.csv", help="where to write the answer (default: ./answer.csv)")
    ap.add_argument("--device", default=None, help="cuda (default when available) or cpu (very slow)")
    ap.add_argument("--reference", default=str(REFERENCE_ANSWER), help="answer to compare with; '' to skip")
    ap.add_argument("--encoder-dir", help="use your own fine-tuned bge-m3 directory instead of the released weights")
    ap.add_argument("--ranker-model", help="use your own CatBoost model (.cbm) instead of the released one")
    args = ap.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg, models = load_config(), load_models_config()
    paths = get_paths(args.data_dir, args.work_dir, args.weights_dir)
    t0 = time.time()

    def step(msg: str) -> None:
        print(f"\n=== [{(time.time() - t0) / 60:5.1f} min] {msg}", flush=True)

    step("weights")
    released = None if (args.encoder_dir and args.ranker_model) else ensure_weights(paths.weights_dir, models)
    encoder_dir = Path(args.encoder_dir) if args.encoder_dir else released.encoder_dir
    ranker_path = Path(args.ranker_model) if args.ranker_model else released.ranker_path
    encoder_id = sha256_of(encoder_dir / "model.safetensors")[:12] if args.encoder_dir else released.encoder_id

    step("benchmark data (lemmatization is cached in --work-dir)")
    stage = make_benchmark_stage(paths)
    print(f"{stage.queries.height} queries, {stage.corpus.height} items")

    step("encoding the corpus with the fine-tuned bge-m3 (cached in --work-dir, ~20 GB)")
    encoder = load_bi_encoder(models, device, model_path=str(encoder_dir))
    store = load_or_build_store(
        encoder, item_tower_texts(stage.corpus), paths.work_dir / f"store_benchmark_{encoder_id}", cfg["text"]["item_max_tokens"]
    )

    step("candidate pools and features")
    generator = CandidateGenerator(stage, store, encoder, cfg, device)
    feats = generator.features()
    print(f"{len(feats)} candidate pairs")

    step("ranking")
    scores = predict_scores(load_ranker(ranker_path), feats)
    n_queries = stage.queries.height
    item_ids = generator.items.ids
    top = select_top_k(feats, scores, item_ids, n_queries, cfg["pool"]["output_k"])

    step(f"writing {args.output}")
    query_ids = stage.queries["query_id"].to_list()
    out = write_answer(args.output, query_ids, dict(zip(query_ids, top)), max_len=cfg["pool"]["output_k"])
    problems = validate_answer(out, query_ids, set(item_ids), max_len=cfg["pool"]["output_k"])
    print("format check:", "OK" if not problems else problems)
    if args.reference and Path(args.reference).exists():
        print("comparison with the submitted answer:", compare_answers(out, args.reference))
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {Path(out).resolve()}")


if __name__ == "__main__":
    main()
