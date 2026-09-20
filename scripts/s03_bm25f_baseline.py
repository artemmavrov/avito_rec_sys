"""Stage 3: BM25F + priors + adaptive slots -> the first honest Recall@50.

  1. tune the three field weights on the validation queries (coordinate grid),
  2. report Recall@50 with slot allocation, the ceiling curve
     (@50/@300/@1000 for global, local and their union) and strata (§9.4).
"""

import itertools
import json
import time

import numpy as np

from avito_rec_sys.data.corpus import tokens
from avito_rec_sys.eval.metrics import ceiling_curve, per_query_recall, recall_at_k, stratified_recall
from avito_rec_sys.features.bm25f import combine_fields
from avito_rec_sys.features.tables import ItemTable, QueryTable
from avito_rec_sys.inference.lexical_pipeline import build_context, build_index, make_stage
from avito_rec_sys.retrieval.slots import allocate_slots
from avito_rec_sys.utils.io import load_config


def rank_lists(index, items, queries, weights, chunk=64, k=1000, k_local=300):
    """Per query: best-first global and local rankings by weighted BM25F."""
    glob, loc = [], []
    for s in range(0, queries.n, chunk):
        fields = index.score_fields(queries.lemma_tokens[s : s + chunk])
        total = combine_fields(fields, weights)
        for r in range(total.shape[0]):
            row = total[r]
            g = _top(row, k)
            lp = items.loc_positions.get(int(queries.loc[s + r]), np.empty(0, dtype=np.int64))
            glob.append(g)
            loc.append(lp[_top(row[lp], k_local)] if lp.size else np.empty(0, dtype=np.int64))
    return glob, loc


def _top(row, k):
    k = min(k, row.shape[0])
    idx = np.argpartition(-row, k - 1)[:k]
    idx = idx[np.argsort(-row[idx], kind="stable")]
    return idx[row[idx] > 0]


def final_lists(glob, loc, items, queries, slots_cfg):
    out = {}
    for q in range(queries.n):
        local_ids = [items.ids[p] for p in loc[q]]
        seen = set(local_ids)
        other = [items.ids[p] for p in glob[q] if items.ids[p] not in seen]
        out[q] = allocate_slots(local_ids, other, items.loc_count.get(int(queries.loc[q]), 0), slots_cfg)
    return out


def main() -> None:
    cfg = load_config()
    t0 = time.time()
    ctx = build_context(cfg)
    stage = make_stage(ctx, "val")
    index = build_index(stage.corpus)
    items = ItemTable(stage.corpus, index.vocab)
    queries = QueryTable(stage.queries, stage.centroids)
    qids = stage.queries["query_id"].to_list()
    qrels = {q: stage.qrels[qids[q]] for q in range(queries.n)}
    print(f"prepared val stage in {time.time() - t0:.0f}s: {items.n} items, {queries.n} queries")

    # 1. field-weight grid (params/desc relative to title=1). The first run peaked at
    # desc=1.0, the edge of [0, .3, .6, 1] with the gain still growing, so the grid
    # is extended along desc at the best params weight (0.5).
    grid = list(itertools.product([0.0, 0.5, 1.0], [0.0, 0.3, 0.6, 1.0])) + [(0.5, d) for d in (1.5, 2.0, 3.0)]
    best = (-1.0, None)
    for wp, wd in grid:
        w = {"title": 1.0, "params": wp, "desc": wd}
        glob, loc = rank_lists(index, items, queries, w, k=300, k_local=100)
        preds = final_lists(glob, loc, items, queries, cfg["slots"])
        r50 = recall_at_k(preds, qrels, 50)
        print(f"  weights params={wp} desc={wd}: recall@50={r50:.4f}")
        if r50 > best[0]:
            best = (r50, w)
    weights = best[1]
    print("best weights:", weights, f"recall@50={best[0]:.4f}")

    # 2. final evaluation at full depth
    glob, loc = rank_lists(index, items, queries, weights, k=1000, k_local=300)
    preds = final_lists(glob, loc, items, queries, cfg["slots"])
    ids = lambda lists: {q: [items.ids[p] for p in lists[q]] for q in range(queries.n)}  # noqa: E731
    curve = ceiling_curve({"bm25f_global": ids(glob), "bm25f_local": ids(loc)}, qrels, ks=(50, 300, 1000))
    r50 = recall_at_k(preds, qrels, 50)

    pq = per_query_recall(preds, qrels, 50)
    stratum = dict(zip(range(queries.n), stage.queries["stratum"].to_list()))
    loc_in_corpus = {q: ("loc_has_items" if items.loc_count.get(int(queries.loc[q]), 0) else "loc_empty") for q in range(queries.n)}
    result = {
        "weights": weights,
        "recall@50_with_slots": r50,
        "ceiling": {k: {str(kk): vv for kk, vv in v.items()} for k, v in curve.items()},
        "by_stratum": {k: {"recall@50": v[0], "n": v[1]} for k, v in stratified_recall(pq, stratum).items()},
        "by_location": {k: {"recall@50": v[0], "n": v[1]} for k, v in stratified_recall(pq, loc_in_corpus).items()},
    }
    (ctx.work / "s03_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
