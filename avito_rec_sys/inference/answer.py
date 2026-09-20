"""Write answer.csv in the exact format required by task_description.md."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


def write_answer(
    path: str | Path,
    query_ids: Sequence[str],
    predictions: Mapping[str, Sequence[str]],
    max_len: int = 50,
) -> Path:
    """One row per query_id (in the given order); ids joined by single spaces.

    Duplicates inside a row are dropped preserving rank order and the list is
    truncated to `max_len`, so a caller bug can't produce an invalid file.
    Strings are written as-is (no dtype inference) with index=False.
    """
    answers = []
    for qid in query_ids:
        seen: dict[str, None] = {}
        for item in predictions.get(qid, ()):
            seen.setdefault(item, None)
            if len(seen) == max_len:
                break
        answers.append(" ".join(seen))

    df = pd.DataFrame({"query_id": list(query_ids), "answer": answers}, dtype=str)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def read_answer(path: str | Path) -> dict[str, list[str]]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {q: a.split() for q, a in zip(df["query_id"], df["answer"])}


def compare_answers(path_a: str | Path, path_b: str | Path) -> dict[str, float]:
    """How close two answer files are: mean share of common item_ids per query and the share of
    queries whose id sets are identical."""
    a, b = read_answer(path_a), read_answer(path_b)
    common = a.keys() & b.keys()
    overlap = [len(set(a[q]) & set(b[q])) / max(len(set(a[q]) | set(b[q])), 1) for q in common]
    identical = [set(a[q]) == set(b[q]) for q in common]
    return {
        "queries_compared": len(common),
        "mean_jaccard": sum(overlap) / max(len(overlap), 1),
        "identical_sets": sum(identical) / max(len(identical), 1),
    }
