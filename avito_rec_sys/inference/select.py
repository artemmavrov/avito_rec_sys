"""Turn scored candidates into the final top-50 per query via slot allocation (§6.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from avito_rec_sys.retrieval.slots import allocate_slots


def select_top(
    feats: pd.DataFrame, scores: np.ndarray, items, queries, slots_cfg: dict
) -> list[list[str]]:
    """One list of item_id per query position 0..queries.n-1 (empty if the
    query has no candidates). Local = item in the query's search location."""
    df = pd.DataFrame({"q": feats["q"].to_numpy(), "i": feats["i"].to_numpy(), "s": scores})
    df["local"] = items.loc[df["i"].to_numpy()] == queries.loc[df["q"].to_numpy()]
    df = df.sort_values(["q", "s"], ascending=[True, False], kind="stable")

    out: list[list[str]] = [[] for _ in range(queries.n)]
    for q, g in df.groupby("q", sort=False):
        local = g.loc[g["local"], "i"].to_numpy()
        other = g.loc[~g["local"], "i"].to_numpy()
        n_local_total = items.loc_count.get(int(queries.loc[q]), 0)
        chosen = allocate_slots(
            [items.ids[k] for k in local], [items.ids[k] for k in other], n_local_total, slots_cfg
        )
        out[int(q)] = chosen
    return out


def select_plain(feats: pd.DataFrame, scores: np.ndarray, items, queries, output_k: int) -> list[list[str]]:
    """Top `output_k` by ranker score, no local/global quota. The ranker already sees `loc_match`
    and the distance, and on validation this beat the fixed quota of `select_top` by ~0.5 pp
    (see reports/04_gpu_run_log.md)."""
    df = pd.DataFrame({"q": feats["q"].to_numpy(), "i": feats["i"].to_numpy(), "s": scores})
    df = df.sort_values(["q", "s"], ascending=[True, False], kind="stable")
    out: list[list[str]] = [[] for _ in range(queries.n)]
    for q, g in df.groupby("q", sort=False):
        out[int(q)] = [items.ids[k] for k in g["i"].to_numpy()[:output_k]]
    return out
