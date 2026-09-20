"""CatBoost YetiRank ranker with early stopping on Recall@50 (§4).

Group = query. The eval metric is CatBoost's built-in `RecallAt:top=50`
(fraction of a group's relevant items inside its top 50), which is exactly
the target metric restricted to the candidate pool -- AUC / NDCG would
reward ordering inside the 50 slots, which the task ignores. Built-in
instead of a Python custom metric because a Python metric would be called
every iteration on millions of rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoost, Pool


def build_pool(df: pd.DataFrame, y: np.ndarray, cols: list[str]) -> Pool:
    # CatBoost needs rows of one group to be contiguous.
    assert (np.diff(df["q"].to_numpy()) >= 0).all(), "rows must be sorted by query"
    return Pool(df[cols].to_numpy(np.float32), label=y, group_id=df["q"].to_numpy())


def train_ranker(
    train: pd.DataFrame,
    y_train: np.ndarray,
    valid: pd.DataFrame,
    y_valid: np.ndarray,
    feature_cols: list[str],
    cfg: dict,
    seed: int = 42,
    thread_count: int = -1,
) -> CatBoost:
    params = {
        "loss_function": cfg["loss_function"],
        "eval_metric": "RecallAt:top=50",
        "iterations": cfg["iterations"],
        "learning_rate": cfg["learning_rate"],
        "depth": cfg["depth"],
        "task_type": cfg["task_type"],
        **({"od_type": "Iter", "od_wait": cfg["early_stopping_rounds"]} if cfg["early_stopping_rounds"] else {}),
        "random_seed": seed,
        "thread_count": thread_count,
        "verbose": 100,
        "allow_writing_files": False,  # no catboost_info/ scratch dir in the repo
    }
    model = CatBoost(params)
    if not cfg["early_stopping_rounds"]:
        # fixed number of iterations: Recall@50 on a few hundred early-stopping queries is flat and
        # noisy, and stopped the ranker too early in practice (see reports/04_gpu_run_log.md)
        model.fit(build_pool(train, y_train, feature_cols))
    else:
        model.fit(build_pool(train, y_train, feature_cols), eval_set=build_pool(valid, y_valid, feature_cols), use_best_model=True)
    return model


def predict_scores(model: CatBoost, feats: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return model.predict(feats[feature_cols].to_numpy(np.float32))
