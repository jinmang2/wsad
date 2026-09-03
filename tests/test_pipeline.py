"""HF-style AnomalyDetectionPipeline (issue #20).

Covers the extraction-free path (cached/dummy features), the compatibility guard
at construction, backbone-dim auto-matching, and the visualize hook. The raw-video
path (``__call__``) needs a backbone + decord and is exercised manually, not here.
"""

import numpy as np
import pytest

import src.features  # noqa: F401  (registration)
import src.models  # noqa: F401  (registration)
from src.pipeline import AnomalyDetectionPipeline

T = 32
NCROPS = 10


def test_guard_rejects_text_head_on_visual_backbone():
    with pytest.raises(ValueError, match="text-aligned"):
        AnomalyDetectionPipeline("i3d", "vadclip")


def test_from_features_mgfn_i3d():
    pipe = AnomalyDetectionPipeline("i3d", "mgfn")
    feats = np.random.randn(T, NCROPS, 2048).astype(np.float32)  # (T, ncrops, D)
    scores = pipe.from_features(feats)
    assert scores.shape == (T,)
    assert np.all(np.isfinite(scores))


def test_backbone_dim_auto_matched_for_clip():
    pipe = AnomalyDetectionPipeline("clip", "mgfn")
    assert pipe.model.config.channels == 512  # CLIP dim, not the 2048 default
    feats = np.random.randn(T, 512).astype(np.float32)  # single-crop CLIP
    scores = pipe.from_features(feats)
    assert scores.shape == (T,)


def test_from_features_sultani_clip_single_crop():
    pipe = AnomalyDetectionPipeline("clip", "sultani")
    assert pipe.model.config.feature_size == 512
    feats = np.random.randn(T, 512).astype(np.float32)
    scores = pipe.from_features(feats)
    assert scores.shape == (T,)


def test_visualize_from_scores(tmp_path):
    pipe = AnomalyDetectionPipeline("i3d", "mgfn")
    scores = np.random.rand(T)
    out = tmp_path / "p.png"
    fig, ax = pipe.visualize(scores, gt=[(5, 12)], save_path=str(out))
    assert out.exists()
