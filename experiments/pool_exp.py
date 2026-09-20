import pickle, numpy as np, collections
from avito_rec_sys.retrieval.rrf import rrf_fuse
D = pickle.load(open("../work/analysis_val_tours.pkl", "rb"))
tg, tl, extra, qrels, ids = D["tours_g"], D["tours_l"], D["extra"], D["qrels"], D["item_ids"]
iloc, qloc = D["item_loc"], D["q_loc"]
pos_of = {v: i for i, v in enumerate(ids)}
n = len(qloc); W = {"bm25f": 2.0, "dense": 4.0, "sparse": 0.0}; K = 1000; KR = 200
F_loc = [rrf_fuse({t: l[q] for t, l in tl.items()}, W, KR, K // 2) for q in range(n)]
F_glob = [rrf_fuse({t: g[q] for t, g in tg.items()}, W, KR, K) for q in range(n)]
pos = [np.array([pos_of[i] for i in qrels[q]]) for q in range(n)]
in_loc = [iloc[p] == qloc[q] for q, p in enumerate(pos)]
tot = sum(len(p) for p in pos)
print("queries", n, "| positives in search loc: %.3f" % (sum(m.sum() for m in in_loc) / tot))
n_loc_items = np.array([int((iloc == l).sum()) for l in qloc])
print("queries with 0 local items: %.3f | <=300: %.3f | >1000: %.3f" % ((n_loc_items == 0).mean(), (n_loc_items <= 300).mean(), (n_loc_items > 1000).mean()))

def recall(pools, with_extra=True):
    r = []
    for q in range(n):
        s = set(pools[q].tolist()) | (set(extra[q].tolist()) if with_extra else set())
        r.append(np.mean([p in s for p in pos[q]]))
    return float(np.mean(r))

def cur(N):  # current: local-first merge, cut to N
    return [np.array(list(dict.fromkeys([*F_loc[q], *F_glob[q]]))[:N]) for q in range(n)]
def mix(N, L, adaptive=False):  # local[:L] then global-only fill
    out = []
    for q in range(n):
        l = list(F_loc[q][:L]); ls = set(l)
        g = [x for x in F_glob[q] if x not in ls]
        out.append(np.array((l + g)[:N]))
    return out
def rrf2(N, wl=1.0, wg=1.0, k=60):  # RRF merge of the local and global fused lists
    out = []
    for q in range(n):
        sc = collections.defaultdict(float)
        for r, x in enumerate(F_loc[q]): sc[x] += wl / (k + r)
        for r, x in enumerate(F_glob[q]): sc[x] += wg / (k + r)
        out.append(np.array(sorted(sc, key=sc.get, reverse=True)[:N]))
    return out

for N in (300, 400, 500):
    print(f"\n=== pool size {N}")
    print(f"current (local first)     : {recall(cur(N)):.4f}   no-extra {recall(cur(N), False):.4f}")
    for L in (100, 150, 200, 250):
        if L < N: print(f"mix local[:{L}] + global    : {recall(mix(N, L)):.4f}")
    print(f"rrf-merge(local, global)  : {recall(rrf2(N)):.4f}")
    print(f"pure global fused         : {recall([np.array(F_glob[q][:N]) for q in range(n)]):.4f}")
# where the misses of the current pool are
c = cur(300); miss_out = miss_in = 0
for q in range(n):
    s = set(c[q].tolist()) | set(extra[q].tolist())
    for p, m in zip(pos[q], in_loc[q]):
        if p not in s: (miss_in if m else miss_out).__class__  # noqa
        if p not in s:
            if m: miss_in += 1
            else: miss_out += 1
print(f"\nmissed positives @300: in-loc {miss_in} ({miss_in/tot:.3%}), out-of-loc {miss_out} ({miss_out/tot:.3%}) of {tot}")

# ---------- geo: how far are out-of-loc positives from the query location centroid, and does a "nearby" list help?
def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))
qlat, qlon, ilat, ilon = D["q_lat"], D["q_lon"], D["item_lat"], D["item_lon"]
print("\nqueries with centroid: %.3f" % np.mean(~np.isnan(qlat)))
d_out = []
for q in range(n):
    for p, m in zip(pos[q], in_loc[q]):
        if not m and not np.isnan(qlat[q]) and not np.isnan(ilat[p]): d_out.append(hav(qlat[q], qlon[q], ilat[p], ilon[p]))
d_out = np.array(d_out); print("out-of-loc positives with distance: n=%d, median %.0f km; share <50km %.2f <100km %.2f <300km %.2f" % (len(d_out), np.median(d_out), (d_out<50).mean(), (d_out<100).mean(), (d_out<300).mean()))

def near_lists(R):
    out = []
    for q in range(n):
        g = np.array(F_glob[q])
        if np.isnan(qlat[q]) or len(g) == 0: out.append(np.empty(0, dtype=np.int64)); continue
        d = hav(qlat[q], qlon[q], ilat[g], ilon[g])
        out.append(g[(d < R) & ~np.isnan(d)])
    return out
def rrf3(N, near, wn, k=60, wl=1.0, wg=1.0):
    out = []
    for q in range(n):
        sc = collections.defaultdict(float)
        for r, x in enumerate(F_loc[q]): sc[x] += wl / (k + r)
        for r, x in enumerate(F_glob[q]): sc[x] += wg / (k + r)
        for r, x in enumerate(near[q]): sc[x] += wn / (k + r)
        out.append(np.array(sorted(sc, key=sc.get, reverse=True)[:N]))
    return out
print("\nN=300 baseline rrf2: %.4f" % recall(rrf2(300)))
for R in (30, 60, 120):
    nl = near_lists(R)
    for wn in (0.5, 1.0, 2.0):
        print(f"R={R:4d} km wn={wn}: N=300 {recall(rrf3(300, nl, wn)):.4f}  N=400 {recall(rrf3(400, nl, wn)):.4f}")

# ---------- fill missing query-location centroids from where train users' clicked items are (leak-free: train_part only)
import polars as pl
from avito_rec_sys.inference.lexical_pipeline import build_context
from avito_rec_sys.utils.io import load_config
ctx = build_context(load_config())
tp = ctx.splits.train_part
cc = (tp.drop_nulls(["item_latitude", "item_longitude", "search_location_id"])
        .group_by("search_location_id").agg(pl.col("item_latitude").mean().alias("lat"), pl.col("item_longitude").mean().alias("lon")))
click_c = {int(l): (float(a), float(o)) for l, a, o in zip(cc["search_location_id"], cc["lat"], cc["lon"])}
qlat2, qlon2 = qlat.copy(), qlon.copy()
for q in range(n):
    if np.isnan(qlat2[q]) and int(qloc[q]) in click_c: qlat2[q], qlon2[q] = click_c[int(qloc[q])]
print("\ncentroid coverage: before %.3f  after click-fill %.3f" % (np.mean(~np.isnan(qlat)), np.mean(~np.isnan(qlat2))))
qlat, qlon = qlat2, qlon2
for R in (60, 100, 150):
    nl = near_lists(R)
    print(f"filled centroids R={R}: N=300 {recall(rrf3(300, nl, 1.0)):.4f}  N=400 {recall(rrf3(400, nl, 1.0)):.4f}  N=500 {recall(rrf3(500, nl, 1.0)):.4f}")
