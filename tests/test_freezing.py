import torch
from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaModel

from avito_rec_sys.training.freezing import freeze_encoder_layers


def _tiny_config(n_layers=4):
    return XLMRobertaConfig(
        vocab_size=100, hidden_size=16, num_hidden_layers=n_layers, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=64,
    )


def test_freezes_embeddings_and_lowest_layers_only():
    model = XLMRobertaModel(_tiny_config(4))
    info = freeze_encoder_layers(model, frozen_layers=2, freeze_embeddings=True)
    assert info["frozen_layers"] == [0, 1]
    for name, p in model.named_parameters():
        if name.startswith("embeddings.") or ".layer.0." in name or ".layer.1." in name:
            assert not p.requires_grad, name
        elif ".layer.2." in name or ".layer.3." in name:
            assert p.requires_grad, name
    assert info["frozen_params"] > 0 and info["trainable_params"] > 0


def test_works_on_sequence_classification_wrapper():
    model = XLMRobertaForSequenceClassification(_tiny_config(4))
    freeze_encoder_layers(model, frozen_layers=3)
    assert not model.roberta.embeddings.word_embeddings.weight.requires_grad
    assert not model.roberta.encoder.layer[2].attention.self.query.weight.requires_grad
    assert model.roberta.encoder.layer[3].attention.self.query.weight.requires_grad
    assert model.classifier.dense.weight.requires_grad  # the head always trains


def test_frozen_params_unchanged_after_optimizer_step():
    torch.manual_seed(0)
    model = XLMRobertaForSequenceClassification(_tiny_config(4))
    freeze_encoder_layers(model, frozen_layers=2)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    ids = torch.randint(3, 100, (2, 8))
    out = model(input_ids=ids, labels=torch.tensor([0, 1]))
    out.loss.backward()
    opt.step()
    changed_trainable = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            changed_trainable += int(not torch.equal(before[n], p.detach()))
        else:
            assert torch.equal(before[n], p.detach()), f"frozen param changed: {n}"
    assert changed_trainable > 0
