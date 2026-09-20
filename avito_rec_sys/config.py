"""Loading of the YAML configs in `configs/`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def load_config() -> dict[str, Any]:
    """Pipeline hyper-parameters (configs/pipeline.yaml)."""
    with open(CONFIGS_DIR / "pipeline.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_models_config() -> dict[str, Any]:
    """Pinned open-source model revision and the released weight files (configs/models.yaml)."""
    with open(CONFIGS_DIR / "models.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)
