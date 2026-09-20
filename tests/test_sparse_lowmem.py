"""The low-memory sparse embedding must equal FlagEmbedding's training branch, values and gradients."""

from types import SimpleNamespace

import torch

from avito_rec_sys.training.biencoder_train import sparse_embedding_lowmem


def _reference(self, hidden_state, input_ids):
    """FlagEmbedding's training-mode `_sparse_embedding` (dense (B, L, V) then max over L)."""
    token_weights = torch.relu(self.sparse_linear(hidden_state))
    emb = torch.zeros(input_ids.size(0), input_ids.size(1), self.vocab_size, dtype=token_weights.dtype)
    emb = torch.scatter(emb, dim=-1, index=input_ids.unsqueeze(-1), src=token_weights)
    emb = torch.max(emb, dim=1).values
    emb[:, [self.tokenizer.cls_token_id, self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id, self.tokenizer.unk_token_id]] *= 0.0
    return emb


def test_matches_upstream_values_and_grads():
    torch.manual_seed(0)
    vocab, hidden = 40, 8
    self = SimpleNamespace(
        sparse_linear=torch.nn.Linear(hidden, 1),
        vocab_size=vocab,
        tokenizer=SimpleNamespace(cls_token_id=0, pad_token_id=1, eos_token_id=2, unk_token_id=3),
    )
    ids = torch.randint(0, vocab, (5, 7))
    ids[:, 0] = 0                      # cls
    ids[0, 3] = ids[0, 4] = 10         # repeated token: the max of the two weights must win
    ids[:, -1] = 1                     # padding
    h1 = torch.randn(5, 7, hidden, requires_grad=True)
    h2 = h1.detach().clone().requires_grad_(True)

    ref = _reference(self, h1, ids)
    new = sparse_embedding_lowmem(self, h2, ids)
    assert torch.allclose(ref, new, atol=1e-6)
    assert (new[:, :4] == 0).all()

    w = torch.randn_like(ref)
    (ref * w).sum().backward()
    (new * w).sum().backward()
    assert torch.allclose(h1.grad, h2.grad, atol=1e-6)


def test_return_token_weights_path():
    self = SimpleNamespace(sparse_linear=torch.nn.Linear(4, 1), vocab_size=10, tokenizer=None)
    out = sparse_embedding_lowmem(self, torch.randn(2, 3, 4), torch.zeros(2, 3, dtype=torch.long), return_embedding=False)
    assert out.shape == (2, 3, 1)


def test_chunked_colbert_matches_upstream():
    from FlagEmbedding.finetune.embedder.encoder_only.m3.modeling import EncoderOnlyEmbedderM3Model as M3

    from avito_rec_sys.training.biencoder_train import colbert_score_chunked

    torch.manual_seed(1)
    self = SimpleNamespace(temperature=0.05)
    q = torch.nn.functional.normalize(torch.randn(7, 5, 6), dim=-1).requires_grad_(True)
    p = torch.nn.functional.normalize(torch.randn(11, 9, 6), dim=-1).requires_grad_(True)
    mask = torch.ones(7, 6)
    mask[:, 4:] = 0
    ref = M3.compute_colbert_score(self, q, p, q_mask=mask)
    new = colbert_score_chunked(self, q, p, q_mask=mask, chunk=3)
    assert torch.allclose(ref, new, atol=1e-5)
    g_ref = torch.autograd.grad(ref.sum(), [q, p])
    g_new = torch.autograd.grad(new.sum(), [q, p])
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(g_ref, g_new))
