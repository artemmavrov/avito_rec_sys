"""Filesystem layout: where the input data, the intermediate artifacts and the model weights live.

Every entry point accepts explicit directories; otherwise the environment variables
AVITO_DATA_DIR / AVITO_WORK_DIR / AVITO_WEIGHTS_DIR are used, and finally the defaults
`<repo>/data`, `<repo>/work`, `<repo>/weights`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    work_dir: Path
    weights_dir: Path

    @property
    def train_parquet(self) -> Path:
        return self.data_dir / "train.parquet"

    @property
    def benchmark_queries_parquet(self) -> Path:
        return self.data_dir / "benchmark_queries.parquet"

    @property
    def benchmark_items_parquet(self) -> Path:
        return self.data_dir / "benchmark_items.parquet"


def get_paths(data_dir: str | None = None, work_dir: str | None = None, weights_dir: str | None = None) -> Paths:
    def pick(explicit: str | None, env: str, default: str) -> Path:
        return Path(explicit or os.environ.get(env) or REPO_ROOT / default).expanduser().resolve()

    paths = Paths(
        pick(data_dir, "AVITO_DATA_DIR", "data"),
        pick(work_dir, "AVITO_WORK_DIR", "work"),
        pick(weights_dir, "AVITO_WEIGHTS_DIR", "weights"),
    )
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    return paths


def add_path_arguments(parser) -> None:
    """The three directory flags shared by all entry points."""
    parser.add_argument("--data-dir", help="directory with train.parquet, benchmark_queries.parquet, benchmark_items.parquet")
    parser.add_argument("--work-dir", help="cache for intermediate artifacts (encoded corpus, features, ...)")
    parser.add_argument("--weights-dir", help="model weights (downloaded here by predict.py)")
