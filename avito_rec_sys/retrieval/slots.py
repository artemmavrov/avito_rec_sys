"""Adaptive local/global slot allocation (§6.3).

"Local" = the item is in the query's search location. A hard geo filter
would zero out the 17.4% of queries with no local items and lose the 16.9%
of train positives that were chosen outside the search location, so a
global reserve is always kept:

    no local items          -> all slots global
    local pool <= 1000      -> as many local as fit, global reserve 5-10
    local pool  > 1000      -> 40 local + 10 global

There is deliberately NO title de-duplication here: we don't know which
member of a duplicate cluster the user picked, and geo is what tells them
apart (§6.3).
"""

from __future__ import annotations

from typing import Sequence


def allocate_slots(
    local_ranked: Sequence[str],
    global_ranked: Sequence[str],
    n_local_total: int,
    slots_cfg: dict,
) -> list[str]:
    """Pick the final `output_k` ids from two ranked lists.

    `local_ranked` / `global_ranked` are best-first. `n_local_total` is the
    number of corpus items in the query's location (not just how many made it
    into the ranked pool) and selects the branch. Shortfalls in one list are
    back-filled from the other, so the output is `output_k` long whenever
    enough distinct candidates exist.
    """
    k = slots_cfg["output_k"]
    local_set = set(local_ranked)
    # keep the global list disjoint from local so an item can't take two slots
    global_only = [i for i in global_ranked if i not in local_set]

    if n_local_total == 0 or not local_ranked:
        n_local = 0
    elif n_local_total <= slots_cfg["small_local_pool_threshold"]:
        reserve = min(max(k - len(local_ranked), slots_cfg["small_pool_global_reserve_min"]),
                      slots_cfg["small_pool_global_reserve_max"])
        n_local = k - reserve
    else:
        n_local = slots_cfg["large_pool_local_slots"]

    chosen: dict[str, None] = {}
    for item in local_ranked[:n_local]:
        chosen.setdefault(item, None)
    for item in global_only:
        if len(chosen) >= k:
            break
        chosen.setdefault(item, None)
    # global list exhausted before filling k: top up with remaining local
    for item in local_ranked[n_local:]:
        if len(chosen) >= k:
            break
        chosen.setdefault(item, None)
    return list(chosen)[:k]
