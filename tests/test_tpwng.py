"""Faithful TPWNG: CLIP-grounded CoOp text + Normality Visual Prompt (offline)."""

import pytest
import torch

pytest.importorskip("open_clip")

from src.models.tpwng import TPWNGConfig, TPWNGForVideoAnomalyDetection

B, T, D = 4, 32, 512


def _cfg(**kw):
    return TPWNGConfig(use_clip_text=True, clip_pretrained=None, **kw)


def test_text_frozen_except_prompt_and_projection():
    model = TPWNGForVideoAnomalyDetection(_cfg())
    trainable = {n for n, p in model.text_encoder.named_parameters() if p.requires_grad}
    # only the CoOp context and the final text projection fine-tune
    assert "text_prompt_embeddings.weight" in trainable
    assert any("text_projection" in n for n in trainable)
    # the frozen transformer tower gets no trainable params
    assert not any("transformer" in n for n in trainable)


def test_forward_backward_nvp_active():
    model = TPWNGForVideoAnomalyDetection(_cfg(use_nvp=True)).train()
    out = model(
        video=torch.randn(B, 1, T, D),
        abnormal_labels=torch.ones(B // 2),
        normal_labels=torch.zeros(B // 2),
    )
    assert out.scores.shape == (B, T, 1)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    # NVP FFN learns from normal frames
    g = model.nvp.ffn[0].weight.grad
    assert g is not None and torch.isfinite(g).all() and g.norm() > 0
    # CoOp context learns
    gp = model.text_encoder.text_prompt_embeddings.weight.grad
    assert gp is not None and gp.norm() > 0


def test_eval_scores_bounded():
    model = TPWNGForVideoAnomalyDetection(_cfg()).eval()
    with torch.no_grad():
        out = model(video=torch.randn(2, 1, T, D))
    assert out.loss is None
    assert out.scores.shape == (2, T, 1)
    assert float(out.scores.min()) >= 0.0 and float(out.scores.max()) <= 1.0
