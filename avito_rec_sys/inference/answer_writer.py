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
