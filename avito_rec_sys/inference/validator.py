"""Standalone validator for answer.csv (§11 risk 7).

Malformed item_ids never raise an error at submission time -- the metric just
silently drops. So every rule from task_description.md is checked here
explicitly, and the file is read back from disk as strings (no dtype
inference) exactly as the grader would see it.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

_ITEM_ID_RE = re.compile(r"^[0-9a-f]{16}$")
MAX_ANSWER_LEN = 50


def validate_answer(
    answer_path: str | Path,
    query_ids: list[str],
    corpus_item_ids: set[str],
    max_len: int = MAX_ANSWER_LEN,
) -> list[str]:
    """Return a list of human-readable problems; an empty list means valid."""
    problems: list[str] = []
    path = Path(answer_path)

    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows or rows[0] != ["query_id", "answer"]:
        problems.append(f"header must be exactly ['query_id', 'answer'], got {rows[0] if rows else None}")
        return problems
    body = rows[1:]

    bad_width = [i for i, r in enumerate(body) if len(r) != 2]
    if bad_width:
        problems.append(f"{len(bad_width)} rows do not have exactly 2 columns (first: line {bad_width[0] + 2})")
        return problems

    got_ids = [r[0] for r in body]
    if len(set(got_ids)) != len(got_ids):
        problems.append("duplicate query_id rows")
    expected = set(query_ids)
    missing = expected - set(got_ids)
    extra = set(got_ids) - expected
    if missing:
        problems.append(f"{len(missing)} query_ids missing from answer")
    if extra:
        problems.append(f"{len(extra)} unexpected query_ids in answer")
    if len(body) != len(query_ids):
        problems.append(f"expected {len(query_ids)} rows, got {len(body)}")

    n_bad_format = n_not_in_corpus = n_dups = n_too_long = 0
    for qid, answer in body:
        items = answer.split(" ") if answer else []
        if len(items) > max_len:
            n_too_long += 1
        if len(set(items)) != len(items):
            n_dups += 1
        for item in items:
            if not _ITEM_ID_RE.match(item):
                n_bad_format += 1
            elif item not in corpus_item_ids:
                n_not_in_corpus += 1

    if n_too_long:
        problems.append(f"{n_too_long} rows have more than {max_len} item_ids")
    if n_dups:
        problems.append(f"{n_dups} rows contain repeated item_ids")
    if n_bad_format:
        problems.append(f"{n_bad_format} item_ids are not 16 lowercase hex chars")
    if n_not_in_corpus:
        problems.append(f"{n_not_in_corpus} item_ids do not exist in the corpus")
    return problems
