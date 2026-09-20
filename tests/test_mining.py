import hashlib
import json

import numpy as np
import polars as pl

from avito_rec_sys.data.params_parser import build_item_tower_text
from avito_rec_sys.training.biencoder_train import write_training_jsonl
from avito_rec_sys.training.mining import build_mining_corpus, mine


class FakeEncoder:
    """Deterministic text -> unit vector, standing in for bge-m3 (same `encode` contract)."""

    def encode(self, texts, batch_size=None, max_length=None, return_dense=True, return_sparse=False,
               return_colbert_vecs=False):
        vecs = []
        for t in texts:
            seed = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:8], 16)
            v = np.random.RandomState(seed).randn(16)
            vecs.append(v / np.linalg.norm(v))
        return {"dense_vecs": np.array(vecs, dtype=np.float32)}


def _train_part(n_queries=12, items_per_query=2):
    rows = []
    for q in range(n_queries):
        for k in range(items_per_query):
            rows.append({
                "search_query": f"запрос {q}", "search_query_norm": f"{q} запрос", "search_infm_params_text": "",
                "item_id": f"{q:08x}{k:08x}", "item_title_raw": f"Объявление {q} вариант {k}",
                "item_infm_params_text": "",
            })
    df = pl.DataFrame(rows)
    return df.with_columns(
        pl.struct(["item_title_raw", "item_infm_params_text"])
        .map_elements(lambda s: build_item_tower_text(s["item_title_raw"], s["item_infm_params_text"]), return_dtype=pl.Utf8)
        .alias("item_tower_text")
    )


def test_mined_negatives_are_clean_and_deterministic():
    tp = _train_part()
    mc = build_mining_corpus(tp)
    model = FakeEncoder()
    dense = model.encode(mc.tower)["dense_vecs"].astype(np.float16)
    cfg = {"text": {"query_max_tokens": 32, "item_max_tokens": 64}}

    args = (tp, model, mc, dense, cfg, 2, 0, 30, 7)
    a = mine(*args, device="cpu")
    b = mine(*args, device="cpu")
    assert a == b  # same seed, same negatives

    for row in tp.iter_rows(named=True):
        negs = a[(row["search_query_norm"], row["item_id"])]
        assert len(negs) == 2
        chosen_for_query = {mc.tower[p] for p in mc.positives_by_query[row["search_query_norm"]]}
        assert not (set(negs) & chosen_for_query)  # never another positive of the same query
        assert row["item_tower_text"] not in negs  # denoised


def test_write_training_jsonl_format_and_skips_short_groups(tmp_path):
    tp = _train_part(n_queries=3, items_per_query=1)
    rows = tp.to_dicts()
    negatives = {
        (rows[0]["search_query_norm"], rows[0]["item_id"]): ["n1", "n2"],
        (rows[1]["search_query_norm"], rows[1]["item_id"]): ["only one"],  # too short -> skipped
    }
    out = tmp_path / "train.jsonl"
    n = write_training_jsonl(tp, negatives, out, n_negatives=2)
    assert n == 1
    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert rec["query"] == "запрос 0" and rec["pos"] == [rows[0]["item_tower_text"]] and rec["neg"] == ["n1", "n2"]


def test_bm25_ranking_is_fused_into_the_window():
    tp = _train_part(n_queries=30)  # 60 items, so positions 40..43 exist
    mc = build_mining_corpus(tp)
    model = FakeEncoder()
    dense = model.encode(mc.tower)["dense_vecs"].astype(np.float16)
    cfg = {"text": {"query_max_tokens": 32, "item_max_tokens": 64}, "rrf": {"k": 200}}
    # BM25 (lexical) says: for every query, only item positions 40..43 are relevant
    lexical = np.arange(40, 44)
    bm25 = {q: lexical for q in tp["search_query"].unique().to_list()}

    args = (tp, model, mc, dense, cfg, 2, 0, 8, 3)
    with_bm25 = mine(*args, device="cpu", bm25_top=bm25)
    plain = mine(*args, device="cpu")
    fused_towers = {mc.tower[p] for p in lexical}
    # the lexical candidates enter the (small) rank window only when BM25 is merged
    assert any(set(v) & fused_towers for v in with_bm25.values())
    assert with_bm25 != plain
    for row in tp.iter_rows(named=True):
        negs = with_bm25[(row["search_query_norm"], row["item_id"])]
        assert row["item_tower_text"] not in negs  # denoising still applies
