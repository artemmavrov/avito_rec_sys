"""Config loading and small filesystem helpers shared across the pipeline."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`, returning a new dict.

    Used to apply configs/smoke_test.yaml on top of configs/pipeline.yaml
    without duplicating the full config file.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(profile: str = "pipeline") -> dict[str, Any]:
    """Load configs/pipeline.yaml, optionally layering configs/smoke_test.yaml on top.

    profile="pipeline" -> production config as-is.
    profile="smoke_test" -> pipeline.yaml with smoke_test.yaml overrides merged in.

    The env var AVITO_PIPELINE_CONFIG, if set, names a complete, already merged
    YAML that replaces both (used by scripts/dress_rehearsal.py to run the real
    stage scripts against a tiny data slice).
    """
    override = os.environ.get("AVITO_PIPELINE_CONFIG")
    if override:
        with open(override, encoding="utf-8") as f:
            return yaml.safe_load(f)

    with open(CONFIGS_DIR / "pipeline.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if profile == "smoke_test":
        with open(CONFIGS_DIR / "smoke_test.yaml", encoding="utf-8") as f:
            overrides = yaml.safe_load(f)
        cfg = _deep_merge(cfg, overrides)
    elif profile != "pipeline":
        raise ValueError(f"Unknown config profile: {profile!r}")

    return cfg


def load_models_config() -> dict[str, Any]:
    with open(CONFIGS_DIR / "models.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a paths.* entry in the config relative to configs/ (where the yaml lives)."""
    raw = cfg["paths"][key]
    return (CONFIGS_DIR / raw).resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
