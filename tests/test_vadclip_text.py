"""Faithful VadCLIP text branch (encode_textprompt) — offline, random CLIP weights.

Uses ``clip_pretrained=None`` so no download happens; we only check the wiring,
shapes, gradient flow (CoOp context learns, CLIP tower frozen), and that CLASM is
active when per-video class labels are supplied.
"""

import numpy as np
import pytest
import torch

torch = pytest.importorskip("torch")
pytest.importorskip("open_clip")

from src.models.vadclip import VadCLIPConfig, VadCLIPForVideoAnomalyDetection

B, T, D = 4, 32, 512


def _cfg():
    # small visual_length keeps the test light; random CLIP tower (no download)
    return VadCLIPConfig(use_clip_text=True, clip_pretrained=None, visual_length=T)


def test_text_encoder_shapes_and_frozen_clip():
    model = VadCLIPForVideoAnomalyDetection(_cfg())
    tf = model._text_features()
    assert tf.shape == (model.config.num_class, model.config.embed_dim)

    # the CLIP text tower is frozen; only the CoOp context learns inside the encoder
    enc = model.text_encoder
    trainable = [n for n, p in enc.named_parameters() if p.requires_grad]
    assert trainable == ["text_prompt_embeddings.weight"]


def test_forward_backward_clasm_active():
    model = VadCLIPForVideoAnomalyDetection(_cfg()).train()
    video = torch.randn(B, 1, T, D)
    normal = torch.zeros(B // 2)
    abnormal = torch.ones(B // 2)
    class_labels = torch.tensor([0, 0, 7, 9])  # normal-first; 7=Fighting 9=Robbery

    out = model(
        video=video,
        abnormal_labels=abnormal,
        normal_labels=normal,
        class_labels=class_labels,
    )
    assert out.scores.shape == (B, T, 1)
    assert out.alignment_logits.shape == (B, T, model.config.num_class)
    assert torch.isfinite(out.loss)

    out.loss.backward()
    # CoOp context receives gradient (text branch actually trains)
    g = model.text_encoder.text_prompt_embeddings.weight.grad
    assert g is not None and torch.isfinite(g.norm()) and g.norm() > 0
    # frozen CLIP transformer gets no gradient
    for n, p in model.text_encoder.transformer.named_parameters():
        assert p.grad is None


def test_eval_scores_only():
    model = VadCLIPForVideoAnomalyDetection(_cfg()).eval()
    with torch.no_grad():
        out = model(video=torch.randn(2, 1, T, D))
    assert out.loss is None
    assert out.scores.shape == (2, T, 1)
    assert float(out.scores.min()) >= 0.0 and float(out.scores.max()) <= 1.0
