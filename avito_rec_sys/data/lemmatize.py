"""Russian lemmatization for lexical features (BM25F, n-gram overlap).

§3.3: pymorphy3, dictionary form, multiprocessing across cores, cached to
disk once. Only feeds BM25 / lexical features -- the encoders get raw text
(they have their own subword tokenizer).
"""

from __future__ import annotations

import re
from functools import lru_cache
from multiprocessing import Pool

import pymorphy3

_TOKEN_RE = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)

# One analyzer per process (module-level global, initialized lazily in each
# worker) -- pymorphy3.MorphAnalyzer is not cheap to construct and is not
# picklable in a way that's worth passing through Pool arguments.
_analyzer: pymorphy3.MorphAnalyzer | None = None


def _get_analyzer() -> pymorphy3.MorphAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = pymorphy3.MorphAnalyzer()
    return _analyzer


@lru_cache(maxsize=200_000)
def _lemma_cached(word: str) -> str:
    return _get_analyzer().parse(word)[0].normal_form


def lemmatize_text(text: str | None) -> str:
    """Lowercase, tokenize, lemmatize each token, rejoin with single spaces."""
    if not text:
        return ""
    tokens = _TOKEN_RE.findall(text.lower())
    return " ".join(_lemma_cached(t) for t in tokens)


def _lemmatize_chunk(texts: list[str]) -> list[str]:
    return [lemmatize_text(t) for t in texts]


def lemmatize_batch(texts: list[str], n_workers: int = 16, chunk_size: int = 2000) -> list[str]:
    """Lemmatize a large list of texts in parallel.

    Each worker builds its own MorphAnalyzer + lru_cache the first time it's
    used, so the cache hit rate improves within a chunk but doesn't carry
    across workers -- fine at corpus scale (~37M tokens over 16 cores runs
    in a couple of minutes per §8).
    """
    if not texts:
        return []
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    if n_workers <= 1 or len(chunks) <= 1:
        results = [_lemmatize_chunk(c) for c in chunks]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(_lemmatize_chunk, chunks)
    out: list[str] = []
    for r in results:
        out.extend(r)
    return out
