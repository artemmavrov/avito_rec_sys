"""Download and verify the released model weights.

The trained weights are not part of the repository. They live on Google Drive (see
configs/models.yaml) and are fetched with `gdown`; every file is checked against its sha256.
The tokenizer and config of the fine-tuned bge-m3 are the base model's, taken from the pinned
Hugging Face revision (only these few small files are downloaded, not the base weights).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download

# small files of the base model needed next to the fine-tuned weights
BASE_MODEL_FILES = [
    "config.json", "tokenizer.json", "tokenizer_config.json", "sentencepiece.bpe.model", "special_tokens_map.json",
]


@dataclass(frozen=True)
class Weights:
    encoder_dir: Path  # a bge-m3 directory: config + tokenizer + fine-tuned weights and heads
    ranker_path: Path  # CatBoost model
    encoder_id: str  # short hash of the encoder weights; keys the cache of the encoded corpus


def sha256_of(path: Path, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def _is_valid(path: Path, sha256: str) -> bool:
    return path.is_file() and sha256_of(path) == sha256


def _download(name: str, spec: dict, target: Path, folder_url: str) -> None:
    import gdown

    print(f"[weights] downloading {name} from Google Drive ...")
    try:
        gdown.download(id=spec["gdrive_id"], output=str(target), quiet=False)
    except Exception as e:  # gdown raises on quota / permission errors
        raise RuntimeError(_manual_hint(name, target, folder_url)) from e
    if not _is_valid(target, spec["sha256"]):
        target.unlink(missing_ok=True)
        raise RuntimeError(f"{name}: sha256 mismatch after download. " + _manual_hint(name, target, folder_url))


def _manual_hint(name: str, target: Path, folder_url: str) -> str:
    return f"Could not fetch {name}. Download it manually from {folder_url} and put it at {target}."


def ensure_weights(weights_dir: Path, models_cfg: dict) -> Weights:
    """Make sure every released file is present and intact in `weights_dir`, downloading what is not."""
    wcfg = models_cfg["weights"]
    encoder_dir = weights_dir / "bge-m3-finetuned"
    encoder_dir.mkdir(parents=True, exist_ok=True)

    for name, spec in wcfg["encoder"].items():
        if not _is_valid(encoder_dir / name, spec["sha256"]):
            _download(name, spec, encoder_dir / name, wcfg["folder_url"])
    if not all((encoder_dir / f).exists() for f in BASE_MODEL_FILES):
        base = models_cfg["bi_encoder"]
        snapshot_download(base["name"], revision=base["revision"], allow_patterns=BASE_MODEL_FILES, local_dir=encoder_dir)

    (ranker_name, ranker_spec), = wcfg["ranker"].items()
    ranker_path = weights_dir / ranker_name
    if not _is_valid(ranker_path, ranker_spec["sha256"]):
        _download(ranker_name, ranker_spec, ranker_path, wcfg["folder_url"])
    return Weights(encoder_dir, ranker_path, wcfg["encoder"]["model.safetensors"]["sha256"][:12])
