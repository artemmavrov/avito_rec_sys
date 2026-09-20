"""End-to-end check of the candidate pipeline on a tiny synthetic stage with a fake encoder (CPU only)."""

import copy
import zlib

import numpy as np
import pandas as pd
import polars as pl
import pytest

from avito_rec_sys.config import load_config
from avito_rec_sys.data.corpus import prepare_items, prepare_queries
from avito_rec_sys.data.normalize import add_normalized_query_column
from avito_rec_sys.features.logs import QueryLogs
from avito_rec_sys.pipeline.candidates import RANKER_FEATURES, CandidateGenerator, item_tower_texts
from avito_rec_sys.pipeline.context import Stage
from avito_rec_sys.retrieval.colbert_store import ColbertStore
from avito_rec_sys.retrieval.encode_store import CorpusStore, sparse_to_csr
from avito_rec_sys.retrieval.encoders import encode_bi_encoder

SERVICES = ["ремонт стиральных машин", "маникюр гель лак", "услуги грузчиков", "баня на дровах", "монтаж видеодомофонов"]


class FakeEncoder:
    """Deterministic stand-in for BGEM3FlagModel.encode: vectors depend only on the words of a text."""

    def _vec(self, word, dim):
        return np.random.RandomState(zlib.crc32(word.encode())).randn(dim)

    def encode(self, texts, batch_size, max_length, return_dense, return_sparse, return_colbert_vecs):
        dense, sparse, colbert = [], [], []
        for t in texts:
            words = t.lower().split() or ["_"]
            toks = np.stack([self._vec(w, 8) for w in words])
            toks /= np.linalg.norm(toks, axis=1, keepdims=True)
            d = toks.mean(axis=0)
            dense.append(d / np.linalg.norm(d))
            sparse.append({zlib.crc32(w.encode()) % 1000: 1.0 for w in words})
            colbert.append(toks)
        return {"dense_vecs": np.stack(dense), "lexical_weights": sparse, "colbert_vecs": colbert}


def _stage_and_store(tmp_path):
    rng = np.random.RandomState(0)
    n = 60
    corpus = pl.DataFrame({
        "item_id": [f"{i:016x}" for i in range(n)],
        "item_title_raw": [f"{SERVICES[i % 5]} {i % 7}" for i in range(n)],
        "item_description_raw": [f"Профессионально выполним {SERVICES[i % 5]}, гарантия" for i in range(n)],
        "item_infm_params_text": ["Вид услуги Красота" if i % 2 else "" for i in range(n)],
        "item_category_id": [114] * n,
        "item_microcat_id": [i % 5 for i in range(n)],
        "item_price": rng.rand(n) * 1000,
        "item_rating": rng.rand(n) * 5,
        "item_rating_reviews_count": rng.randint(0, 50, n).astype(float),
        "item_location_id": [1 + i % 3 for i in range(n)],
        "item_latitude": 55.0 + rng.rand(n),
        "item_longitude": 37.0 + rng.rand(n),
        "item_is_phone_hidden": [bool(i % 2) for i in range(n)],
    })
    corpus = prepare_items(corpus, n_workers=1)
    queries = pl.DataFrame({
        "query_id": [f"q{i:015d}" for i in range(7)],
        "search_query": [SERVICES[i % 5] for i in range(7)],
        "search_location_id": [1 + i % 4 for i in range(7)],  # location 4 holds no items
        "search_infm_params_text": ["", "Вид услуги Красота"] * 3 + [""],
        "search_category": [114] * 7,
    })
    queries = prepare_queries(add_normalized_query_column(queries), n_workers=1)
    history = add_normalized_query_column(pl.DataFrame({
        "search_query": [SERVICES[0]] * 2,
        "item_id": [corpus["item_id"][0], corpus["item_id"][5]],
        "item_title_raw": [corpus["item_title_raw"][0], corpus["item_title_raw"][5]],
        "item_microcat_id": [0, 0],
    }))
    stage = Stage("test", corpus, queries, None, QueryLogs(history), {1: (55.5, 37.5), 2: (55.4, 37.4), 4: (55.6, 37.2)})

    encoder = FakeEncoder()
    enc = encode_bi_encoder(encoder, item_tower_texts(corpus), 16, 128)
    colbert = ColbertStore.build(tmp_path / "colbert.bin", enc.colbert, 8)
    store = CorpusStore(enc.dense.astype(np.float16), sparse_to_csr(enc.sparse), colbert)
    return stage, store, encoder


@pytest.fixture(scope="module")
def cfg():
    c = copy.deepcopy(load_config())
    c["pool"].update(retrieval_k=20, size=12)
    return c


def test_candidate_features_have_the_ranker_layout(tmp_path, cfg):
    stage, store, encoder = _stage_and_store(tmp_path)
    cfg = copy.deepcopy(cfg)
    cfg["pool"]["query_chunk"] = 256
    feats = CandidateGenerator(stage, store, encoder, cfg, device="cpu").features(log=lambda *_: None)

    assert set(RANKER_FEATURES) <= set(feats.columns) and len(RANKER_FEATURES) == 46
    assert (np.diff(feats["q"].to_numpy()) >= 0).all() and set(feats["q"]) == set(range(7))
    assert not feats.duplicated(["q", "i"]).any()
    assert feats.groupby("q").size().max() >= 12  # the pool is at least `size` (the logs may add more)
    assert feats["colbert_maxsim"].notna().all() and feats["dense_cos"].between(-1.01, 1.01).all()


def test_chunking_does_not_change_the_result(tmp_path, cfg):
    stage, store, encoder = _stage_and_store(tmp_path)
    whole = copy.deepcopy(cfg)
    whole["pool"]["query_chunk"] = 256
    small = copy.deepcopy(cfg)
    small["pool"]["query_chunk"] = 3
    a = CandidateGenerator(stage, store, encoder, whole, device="cpu").features(log=lambda *_: None)
    b = CandidateGenerator(stage, store, encoder, small, device="cpu").features(log=lambda *_: None)
    pd.testing.assert_frame_equal(a, b)
