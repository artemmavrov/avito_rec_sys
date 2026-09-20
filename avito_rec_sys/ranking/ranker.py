"""CatBoost YetiRank ranker over the candidate pool, and the final top-K selection.

Group = query. The training metric is CatBoost's built-in `RecallAt:top=50`: the fraction of a
group's relevant items inside its top 50, i.e. the target metric restricted to the pool. (AUC or
NDCG would reward ordering inside the 50 slots, which the task ignores.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoost, Pool

from avito_rec_sys.pipeline.candidates import RANKER_FEATURES


def build_pool(feats: pd.DataFrame, y: np.ndarray | None = None, columns: list[str] = RANKER_FEATURES) -> Pool:
    # CatBoost needs the rows of one group to be contiguous
    assert (np.diff(feats["q"].to_numpy()) >= 0).all(), "rows must be sorted by query"
    return Pool(feats[columns].to_numpy(np.float32), label=y, group_id=feats["q"].to_numpy())


def train_ranker(feats: pd.DataFrame, y: np.ndarray, cfg: dict, seed: int = 42) -> CatBoost:
    """Fit on all rows of `feats` for a fixed number of iterations: Recall@50 on a few hundred
    early-stopping queries is flat and noisy, and stopped the ranker too early in practice."""
    params = {
        "loss_function": cfg["loss_function"],
        "eval_metric": "RecallAt:top=50",
        "iterations": cfg["iterations"],
        "learning_rate": cfg["learning_rate"],
        "depth": cfg["depth"],
        "task_type": "CPU",
        "random_seed": seed,
        "thread_count": -1,
        "verbose": 100,
        "allow_writing_files": False,  # no catboost_info/ scratch dir
    }
    model = CatBoost(params)
    model.fit(build_pool(feats, y))
    return model


def load_ranker(path: str) -> CatBoost:
    model = CatBoost()
    model.load_model(str(path))
    return model


def predict_scores(model: CatBoost, feats: pd.DataFrame) -> np.ndarray:
    return model.predict(feats[RANKER_FEATURES].to_numpy(np.float32))


def select_top_k(feats: pd.DataFrame, scores: np.ndarray, item_ids: list[str], n_queries: int, k: int) -> list[list[str]]:
    """The `k` best-scoring candidates of every query (best first); empty for a query without
    candidates. No local / global quota: the ranker already sees `loc_match` and the distance."""
    df = pd.DataFrame({"q": feats["q"].to_numpy(), "i": feats["i"].to_numpy(), "s": scores})
    df = df.sort_values(["q", "s"], ascending=[True, False], kind="stable")
    out: list[list[str]] = [[] for _ in range(n_queries)]
    for q, g in df.groupby("q", sort=False):
        out[int(q)] = [item_ids[i] for i in g["i"].to_numpy()[:k]]
    return out
