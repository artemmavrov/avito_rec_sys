import pytest

from avito_rec_sys.inference.weights import _is_valid, ensure_weights, sha256_of


def test_sha256_and_validity(tmp_path):
    f = tmp_path / "w.bin"
    f.write_bytes(b"abc")
    digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha256_of(f) == digest
    assert _is_valid(f, digest) and not _is_valid(f, "0" * 64)
    assert not _is_valid(tmp_path / "missing", digest)


def test_ensure_weights_uses_intact_local_files_without_network(tmp_path, monkeypatch):
    def fake(name, content):
        p = tmp_path / name
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(content)
        return {"gdrive_id": "unused", "sha256": sha256_of(p)}

    enc = tmp_path / "bge-m3-finetuned"
    enc.mkdir()
    models_cfg = {
        "bi_encoder": {"name": "x/y", "revision": "r"},
        "weights": {
            "folder_url": "http://example",
            "encoder": {n: fake(f"bge-m3-finetuned/{n}", n.encode()) for n in ("model.safetensors", "sparse_linear.pt")},
            "ranker": {"r.cbm": fake("r.cbm", b"rank")},
        },
    }
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "sentencepiece.bpe.model", "special_tokens_map.json"):
        (enc / f).write_text("{}")
    w = ensure_weights(tmp_path, models_cfg)
    assert w.encoder_dir == enc and w.ranker_path == tmp_path / "r.cbm"
    assert w.encoder_id == models_cfg["weights"]["encoder"]["model.safetensors"]["sha256"][:12]

    (tmp_path / "r.cbm").write_bytes(b"corrupted")  # a corrupted file must trigger a re-download, not be used
    def refuse(*args, **kwargs):
        raise RuntimeError("download attempted")

    monkeypatch.setattr("avito_rec_sys.inference.weights._download", refuse)
    with pytest.raises(RuntimeError, match="download attempted"):
        ensure_weights(tmp_path, models_cfg)
