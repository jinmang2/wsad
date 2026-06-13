"""Faithful VadCLIP text branch (encode_textprompt) — offline, random CLIP tower.

The faithful port builds a self-contained CLIP text tower (``clipmodel.*``,
random-init here so no download), a learnable CoOp context
(``text_prompt_embeddings``), and reproduces the official ``encode_textprompt``.
Numerical equivalence vs the official checkpoint lives in
``scripts/verify_vadclip.py`` (matches official logits to ~1e-6, UCF AUC 0.8802).
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("open_clip")

from src.models.vadclip import VadCLIPConfig, VadCLIPForVideoAnomalyDetection

B, T, D = 4, 32, 512


def _cfg():
    # input length must equal visual_length for the faithful temporal branch
    return VadCLIPConfig(visual_length=T)


def test_encode_textprompt_shape_and_frozen_clip():
    model = VadCLIPForVideoAnomalyDetection(_cfg())
    tf = model.encode_textprompt()
    assert tf.shape == (model.config.num_class, model.config.embed_dim)

    # the CLIP text tower (clipmodel.*) is frozen; only the CoOp context learns
    frozen = [n for n, p in model.clipmodel.named_parameters() if not p.requires_grad]
    assert len(frozen) == len(list(model.clipmodel.parameters()))
    assert model.text_prompt_embeddings.weight.requires_grad


def test_forward_backward_clasm_active():
    model = VadCLIPForVideoAnomalyDetection(_cfg()).train()
    video = torch.randn(B, 1, T, D)
    out = model(
        video=video,
        abnormal_labels=torch.ones(B // 2),
        normal_labels=torch.zeros(B // 2),
        class_labels=torch.tensor([0, 0, 7, 9]),  # normal-first; 7=Fighting 9=Robbery
    )
    assert out.scores.shape == (B, T, 1)
    assert out.alignment_logits.shape == (B, T, model.config.num_class)
    assert torch.isfinite(out.loss)

    out.loss.backward()
    # CoOp context receives gradient (text branch actually trains)
    g = model.text_prompt_embeddings.weight.grad
    assert g is not None and torch.isfinite(g.norm()) and g.norm() > 0
    # frozen CLIP transformer gets no gradient
    for _, p in model.clipmodel.transformer.named_parameters():
        assert p.grad is None


def test_eval_scores_only():
    model = VadCLIPForVideoAnomalyDetection(_cfg()).eval()
    with torch.no_grad():
        out = model(video=torch.randn(2, 1, T, D))
    assert out.loss is None
    assert out.scores.shape == (2, T, 1)
    assert float(out.scores.min()) >= 0.0 and float(out.scores.max()) <= 1.0
