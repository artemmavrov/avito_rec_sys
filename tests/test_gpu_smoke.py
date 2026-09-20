"""GPU code-path smoke tests on the local RTX 3050 (4 GB).

These exist so the rented 3090 is not spent debugging: they run the REAL
bge-m3 / bge-reranker-v2-m3 code paths (tokenizer, freezing, loss, optimizer
step, checkpoint save + reload) on a tiny synthetic dataset with the
`smoke_test` config profile. They check that things run and behave
(frozen params stay frozen, checkpoints reload), NOT that anything learns.

Run explicitly:  pytest tests/test_gpu_smoke.py -m gpu -s
"""

import json
from pathlib import Path

import pytest
import torch

from avito_rec_sys.utils.io import load_config, load_models_config

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]

TOPICS = [
    ("баня на дровах", "Русская баня на дровах под ключ"),
    ("автоподбор", "Автоподбор разовый осмотр автомобиля"),
    ("скупка телевизоров", "Скупка телевизоров дорого"),
    ("монтаж видеодомофонов", "Монтаж и установка видеодомофонов"),
    ("обзвон по базе", "Обзвон клиентской базы колл-центр"),
    ("ремонт стиральных машин", "Ремонт стиральных машин на дому"),
    ("маникюр", "Маникюр гель-лак"),
    ("грузчики", "Услуги грузчиков переезд"),
]


def _write_synthetic_jsonl(path: Path, n_negatives: int) -> int:
    with open(path, "w", encoding="utf-8") as f:
        for i, (q, pos) in enumerate(TOPICS * 2):
            negs = [TOPICS[(i + 1 + k) % len(TOPICS)][1] for k in range(n_negatives)]
            f.write(json.dumps({"query": q, "pos": [pos], "neg": negs}, ensure_ascii=False) + "\n")
    return len(TOPICS) * 2


@pytest.fixture(scope="module")
def smoke_cfg():
    return load_config("smoke_test"), load_models_config()


def test_biencoder_finetune_smoke(tmp_path, smoke_cfg):
    from avito_rec_sys.retrieval.encoders import encode_bi_encoder
    from avito_rec_sys.training.biencoder_train import run_finetune

    cfg, models_cfg = smoke_cfg
    n_neg = cfg["biencoder_train"]["hard_negatives_per_positive"]
    data = tmp_path / "train.jsonl"
    n_groups = _write_synthetic_jsonl(data, n_neg)

    out = tmp_path / "bi_ckpt"
    report = run_finetune(cfg, models_cfg, data, out, n_groups)

    # freezing applied as configured: embeddings + first 22 of 24 layers
    assert report["frozen_layers"] == list(range(cfg["freezing"]["frozen_layers"]))
    assert report["trainable_params"] > 0

    # trainer wrote a checkpoint (checkpoint_every_steps=2, max_steps=4)
    ckpts = sorted(p for p in out.iterdir() if p.name.startswith("checkpoint-"))
    assert ckpts, f"no checkpoint written in {list(out.iterdir())}"
    assert (ckpts[-1] / "colbert_linear.pt").exists() and (ckpts[-1] / "sparse_linear.pt").exists()

    # the fine-tuned checkpoint loads through the same wrapper used at inference
    from FlagEmbedding import BGEM3FlagModel

    torch.cuda.empty_cache()
    model = BGEM3FlagModel(str(ckpts[-1]), use_fp16=True, devices="cuda")
    enc = encode_bi_encoder(model, [t[0] for t in TOPICS[:2]], batch_size=2, max_length=32)
    assert enc.dense.shape == (2, 1024)
    assert len(enc.sparse) == 2 and enc.colbert[0].shape[1] == 1024


def _synthetic_groups(n_neg: int):
    from avito_rec_sys.training.reranker_train import Group

    groups = []
    for i, (q, pos) in enumerate(TOPICS * 2):
        negs = [TOPICS[(i + 1 + k) % len(TOPICS)][1] for k in range(n_neg)]
        groups.append(Group(query=q, docs=[pos, *negs]))
    return groups


def test_reranker_finetune_resume_and_inference_smoke(tmp_path, smoke_cfg):
    import copy

    from avito_rec_sys.retrieval.reranker import CrossEncoderScorer
    from avito_rec_sys.training.reranker_train import train_reranker

    cfg, models_cfg = smoke_cfg
    groups = _synthetic_groups(cfg["reranker_train"]["negatives_per_positive"])
    out = tmp_path / "ce_ckpt"

    r1 = train_reranker(cfg, models_cfg, groups, out)
    assert r1["steps"] == cfg["reranker_train"]["max_steps"]
    assert r1["freeze"]["frozen_layers"] == list(range(cfg["freezing"]["frozen_layers"]))
    assert (out / f"checkpoint-{r1['steps']}" / "training_state.pt").exists()

    # resume: extend the run by 2 steps -- must continue from checkpoint, not restart
    cfg2 = copy.deepcopy(cfg)
    cfg2["reranker_train"]["max_steps"] = r1["steps"] + 2
    torch.cuda.empty_cache()
    r2 = train_reranker(cfg2, models_cfg, groups, out)
    assert r2["steps"] == r1["steps"] + 2

    # fine-tuned checkpoint loads through the inference wrapper and orders sensibly-shaped output
    torch.cuda.empty_cache()
    scorer = CrossEncoderScorer(str(out / f"checkpoint-{r2['steps']}"), max_length=64, batch_size=4)
    logits = scorer.score([TOPICS[0][0]] * 3, [TOPICS[0][1], TOPICS[1][1], TOPICS[2][1]])
    assert logits.shape == (3,) and not (logits != logits).any()  # no NaNs


def test_neural_pipeline_end_to_end_smoke(tmp_path, smoke_cfg):
    """Real bge-m3 + real reranker over a slice of the real validation stage:
    encode -> tours -> RRF -> ColBERT narrowing -> CE -> feature bank ->
    CatBoost -> slots. Needs the artifacts of stages 1-2 in work/."""
    import numpy as np
    import polars as pl

    from avito_rec_sys.features.tables import ItemTable, QueryTable
    from avito_rec_sys.inference.lexical_pipeline import build_index, labels_for
    from avito_rec_sys.inference.neural_pipeline import FEATURE_COLUMNS_NEURAL, NEURAL_COLUMNS, item_tower_texts, run_stage
    from avito_rec_sys.inference.select import select_top
    from avito_rec_sys.retrieval.encode_store import build_store
    from avito_rec_sys.retrieval.encoders import load_bi_encoder, resolve_snapshot
    from avito_rec_sys.retrieval.reranker import CrossEncoderScorer
    from avito_rec_sys.training.catboost_train import predict_scores, train_ranker
    from avito_rec_sys.data.corpus import prepare_queries
    from avito_rec_sys.features.logs import QueryLogs
    from avito_rec_sys.features.tables import location_centroids
    from avito_rec_sys.utils.io import resolve_path

    cfg, models_cfg = smoke_cfg
    work = resolve_path(cfg, "work_dir")
    if not (work / "master_items.parquet").exists():
        pytest.skip("run scripts/s01 and s02 first")

    # Load only the slice this test needs (the machine has ~7 GB free RAM; the
    # full context plus two models does not fit -- the 64 GB server is fine).
    val = work / "splits" / "val"
    queries_df = pl.read_parquet(val / "queries.parquet").head(30)
    ids = queries_df["query_id"].to_list()
    qrels_df = pl.read_parquet(val / "qrels.parquet").filter(pl.col("query_id").is_in(ids))
    qrels = {}
    for qid, item in zip(qrels_df["query_id"].to_list(), qrels_df["item_id"].to_list()):
        qrels.setdefault(qid, set()).add(item)
    positives = set(qrels_df["item_id"].to_list())

    master = pl.scan_parquet(work / "master_items.parquet")
    rng = np.random.RandomState(0)
    others = master.select("item_id").collect()["item_id"].to_list()
    others = [i for i in others if i not in positives]
    keep = positives | set(rng.choice(others, size=1500, replace=False).tolist())
    corpus = master.filter(pl.col("item_id").is_in(keep)).collect()

    texts = set(queries_df["search_query_norm"].to_list())
    history = pl.scan_parquet(work / "splits" / "train_part.parquet").filter(
        pl.col("search_query_norm").is_in(texts)
    ).collect()
    logs = QueryLogs(history)  # log signals for the seen queries among these 30
    centroids = location_centroids(corpus)
    queries_df = prepare_queries(queries_df, n_workers=1)

    class stage:  # minimal stand-in for lexical_pipeline.Stage
        pass

    stage.qrels, stage.logs, stage.centroids = qrels, logs, centroids

    bi = load_bi_encoder(models_cfg, "cuda")
    store = build_store(bi, item_tower_texts(corpus), tmp_path / "store", cfg["text"]["item_max_tokens"], batch_size=32, chunk=512)
    ce = CrossEncoderScorer(
        resolve_snapshot(models_cfg["reranker"]["name"], models_cfg["reranker"]["revision"]),
        max_length=cfg["text"]["cross_encoder_max_tokens"], batch_size=32,
    )
    index = build_index(corpus)
    fw = {"title": 1.0, "params": 0.5, "desc": 1.0}
    out = run_stage(queries_df, corpus, stage.logs, stage.centroids, index, store, bi, ce.score, cfg, fw,
                    cfg["rrf"]["weights"], device="cuda", enc_batch=16, ce_query_chunk=5)

    feats = out.feats
    assert list(feats["q"]) == sorted(feats["q"])  # blocks contiguous: CatBoost groups and CE chunking rely on it
    assert set(FEATURE_COLUMNS_NEURAL) <= set(feats.columns)
    for col in NEURAL_COLUMNS:
        assert np.isfinite(feats[col]).all(), col
    assert feats["dense_cos"].between(-1.001, 1.001).all()
    assert (feats["colbert_maxsim"] > 0).all()
    assert feats["ce_logit"].std() > 0  # the cross-encoder actually discriminates

    # retrieval sanity on a 1.5k-item corpus: most positives must survive the funnel
    items = ItemTable(corpus, index.vocab)
    pool_hit = np.mean([
        len(stage.qrels[ids[q]] & {items.ids[i] for i in feats.loc[feats.q == q, "i"]}) / len(stage.qrels[ids[q]])
        for q in range(len(ids))
    ])
    assert pool_hit > 0.7, pool_hit

    # ranker + slots run end to end on the produced features
    y = labels_for(feats, items, ids, stage.qrels)
    tr = {"loss_function": "YetiRank", "iterations": 20, "learning_rate": 0.1, "depth": 3,
          "task_type": "CPU", "early_stopping_rounds": 5}
    model = train_ranker(feats, y, feats, y, FEATURE_COLUMNS_NEURAL, tr, thread_count=4)
    scores = predict_scores(model, feats, FEATURE_COLUMNS_NEURAL)
    preds = select_top(feats, scores, items, QueryTable(queries_df, stage.centroids), cfg["slots"])
    all_ids = set(items.ids)
    for p in preds:
        assert len(p) <= cfg["slots"]["output_k"] and len(set(p)) == len(p) and set(p) <= all_ids
