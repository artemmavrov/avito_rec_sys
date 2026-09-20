"""Dense hard-negative mining over the training items (§5.3 / §5.5).

Shared by the bi-encoder (zero-shot mining, then the ANCE-style refresh with
the epoch-1 model; 2 negatives from ranks 10-200) and the cross-encoder
(fine-tuned retriever, 4 negatives from ranks 5-300).

Mining is a matmul over pre-encoded vectors, not an encoder pass, so it is
cheap; the encoding of the mining corpus is the only real cost and is done
by the caller-provided `model`.

§5.3 asks for a BM25 + dense hybrid pool. The BM25 half costs ~25 min of CPU
over ~150k training queries, so it is precomputed once, off the GPU budget, by
scripts/s03b_mine_bm25_negatives.py and merged here (`bm25_top`) by RRF with
the dense ranking. Without it, `mine` falls back to the dense ranking alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from avito_rec_sys.data.params_parser import build_item_tower_text
from avito_rec_sys.retrieval.encoders import encode_dense
from avito_rec_sys.retrieval.rrf import rrf_fuse
from avito_rec_sys.training.biencoder_train import query_tower_text
from avito_rec_sys.training.hard_negatives import mine_negatives_for_query


@dataclass
class MiningCorpus:
    ids: list[str]
    tower: list[str]
    id2pos: dict[str, int]
    positives_by_query: dict[str, set[int]]  # normalized query text -> positions of ALL items chosen for it


def build_mining_corpus(train_part: pl.DataFrame) -> MiningCorpus:
    """Unique training items + the full (uncapped) positive set per normalized query,
    so an item another user chose for the same text is never mined as a negative."""
    items = train_part.unique(subset=["item_id"], keep="first", maintain_order=True)
    ids = items["item_id"].to_list()
    tower = [
        build_item_tower_text(t, p)
        for t, p in zip(items["item_title_raw"].to_list(), items["item_infm_params_text"].to_list())
    ]
    id2pos = {v: i for i, v in enumerate(ids)}
    pos: dict[str, set[int]] = {}
    for text, item in zip(train_part["search_query_norm"].to_list(), train_part["item_id"].to_list()):
        pos.setdefault(text, set()).add(id2pos[item])
    return MiningCorpus(ids, tower, id2pos, pos)


def corpus_fingerprint(mc: MiningCorpus) -> str:
    """Order-sensitive hash of the mining corpus item ids: BM25 rankings store corpus
    POSITIONS, so they are only valid for exactly this corpus and order."""
    return hashlib.md5("\n".join(mc.ids).encode("utf-8")).hexdigest()


def load_bm25_negatives(work: Path, mc: MiningCorpus) -> dict[str, np.ndarray] | None:
    """search_query -> best-first mining-corpus positions from BM25, or None if not precomputed."""
    meta_path = Path(work) / "bm25_neg_meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["fingerprint"] != corpus_fingerprint(mc):
        raise ValueError("bm25_negatives were mined for a different mining corpus; rerun s03b")
    out: dict[str, np.ndarray] = {}
    for part in sorted(Path(work).glob("bm25_neg_part_*.parquet")):
        df = pl.read_parquet(part)
        for q, top in zip(df["search_query"].to_list(), df["top"].to_list()):
            out[q] = np.asarray(top, dtype=np.int64)
    return out


def encode_mining_corpus(model, mc: MiningCorpus, cfg: dict, batch_size: int = 256) -> np.ndarray:
    return encode_dense(model, mc.tower, batch_size, cfg["text"]["item_max_tokens"]).astype(np.float16)


def mine(
    pairs: pl.DataFrame,
    model,
    mc: MiningCorpus,
    corpus_dense: np.ndarray,
    cfg: dict,
    n_negatives: int,
    rank_min: int,
    rank_max: int,
    seed: int,
    device: str = "cuda",
    batch_size: int = 256,
    chunk: int = 256,
    bm25_top: dict[str, np.ndarray] | None = None,
) -> dict[tuple[str, str], list[str]]:
    """(search_query_norm, item_id) -> list of negative tower texts.

    With `bm25_top` the ranking is the RRF of the dense and BM25 rankings
    (both cut to `rank_max`), otherwise the dense ranking alone."""
    q_text = [
        query_tower_text(q, p)
        for q, p in zip(pairs["search_query"].to_list(), pairs["search_infm_params_text"].to_list())
    ]
    uniq = list(dict.fromkeys(q_text))
    q_row = {t: i for i, t in enumerate(uniq)}
    q_dense = encode_dense(model, uniq, batch_size, cfg["text"]["query_max_tokens"])

    docs = torch.as_tensor(corpus_dense, device=device)
    if device == "cpu":
        docs = docs.float()
    top: list[np.ndarray] = []
    k = min(rank_max, docs.shape[0])
    for s in range(0, len(uniq), chunk):
        q = torch.as_tensor(q_dense[s : s + chunk], device=device).to(docs.dtype)
        top.extend(torch.topk(q @ docs.T, k, dim=1).indices.cpu().numpy())

    if bm25_top is not None:
        raw_query = dict(zip(q_text, pairs["search_query"].to_list()))
        for i, qt in enumerate(uniq):
            lexical = bm25_top.get(raw_query[qt], np.empty(0, dtype=np.int64))[:rank_max]
            fused = rrf_fuse({"dense": top[i], "bm25": lexical}, {}, cfg["rrf"]["k"], rank_max)
            top[i] = np.asarray(fused, dtype=np.int64)

    rng = np.random.RandomState(seed)
    out: dict[tuple[str, str], list[str]] = {}
    for row, qt in zip(pairs.iter_rows(named=True), q_text):
        norm, item = row["search_query_norm"], row["item_id"]
        pos_positions = mc.positives_by_query.get(norm, set())
        negs = mine_negatives_for_query(
            top[q_row[qt]], pos_positions, mc.tower, row["item_tower_text"], n_negatives, rank_min, rank_max, rng
        )
        out[(norm, item)] = [mc.tower[p] for p in negs]
    return out
