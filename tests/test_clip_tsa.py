"""CLIP-TSA faithful pieces: Perturbed Top-K HardAttention (the real "TSA")."""

import torch

from src.models.clip_tsa import CLIPTSAConfig, CLIPTSAForVideoAnomalyDetection
from src.models.clip_tsa.perturbed_topk import HardAttention, PerturbedTopKFunction


def test_perturbed_topk_indicator_and_grad():
    b, t = 2, 16
    x = torch.randn(b, t, requires_grad=True)
    ind = PerturbedTopKFunction.apply(x, 4, 200, 0.05)  # select 4 of 16
    assert ind.shape == (b, 4, t)
    # each selection row is a soft distribution over t (sums ~1)
    assert torch.allclose(ind.sum(-1), torch.ones(b, 4), atol=1e-4)
    ind.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_hard_attention_preserves_shape_and_grads():
    ha = HardAttention(k=0.5, num_samples=50, input_dim=32)
    x = torch.randn(3, 20, 32)
    out = ha(x)
    assert out.shape == x.shape
    out.sum().backward()
    g = ha.scorer.fc1.weight.grad
    assert g is not None and torch.isfinite(g).all()


def test_model_uses_hardattention_and_aggregate_not_transformer():
    model = CLIPTSAForVideoAnomalyDetection(CLIPTSAConfig())
    names = dict(model.named_modules())
    assert any("hard_attention" in n for n in names)  # the real TSA
    assert any("aggregate" in n for n in names)  # RTFM backbone
    assert not any("encoder" in n and "Transformer" in type(m).__name__
                   for n, m in names.items())  # no stand-in Transformer
