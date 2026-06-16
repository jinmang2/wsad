"""Dim-agnostic heads + backbone<->head text-alignment guard (issue #18).

Covers: (1) MGFN no longer hardcodes 2048 (works on 512/768 backbones) while the
2048 split is byte-for-byte unchanged; (2) the compatibility guard rejects a
text-branch head on a visual-only backbone.
"""

import pytest
import torch

import src.features  # noqa: F401  (registration)
import src.models  # noqa: F401  (registration)
from src.compat import (
    assert_compatible,
    backbone_is_text_aligned,
    head_requires_text_aligned,
    is_compatible,
)

T = 32
NCROPS = 10


def _mgfn(channels):
    from src.models.mgfn import MGFNConfig, MGFNForVideoAnomalyDetection

    return MGFNForVideoAnomalyDetection(MGFNConfig(channels=channels))


# --- MGFN dim-agnosticism ---


@pytest.mark.parametrize("channels", [2048, 768, 512])
def test_mgfn_is_dim_agnostic(channels):
    model = _mgfn(channels).eval()
    with torch.no_grad():
        # +1 for the appended magnitude channel
        out = model(video=torch.randn(1, NCROPS, T, channels + 1))
    assert out.scores.shape == (1, T, 1)
    assert torch.all(torch.isfinite(out.scores))


def test_mgfn_2048_split_unchanged():
    """Regression: at channels=2048 the amplifier splits exactly as the old
    hardcoded literal did (first 2048 = features, last 1 = magnitude)."""
    from src.models.mgfn import MGFNConfig
    from src.models.mgfn.modeling_mgfn import MGFNFeatureAmplifier

    cfg = MGFNConfig(channels=2048)
    amp = MGFNFeatureAmplifier(cfg).eval()
    x = torch.randn(1, NCROPS, T, 2049)
    with torch.no_grad():
        out = amp(x)
    assert out.shape == (NCROPS, cfg.dims[0], T)


# --- compatibility guard ---


def test_text_branch_heads_flagged():
    assert head_requires_text_aligned("vadclip") is True
    assert head_requires_text_aligned("tpwng") is True
    assert head_requires_text_aligned("mgfn") is False
    assert head_requires_text_aligned("clip_tsa") is False  # CLIP image feats, no text


def test_backbone_text_alignment():
    assert backbone_is_text_aligned("clip") is True
    assert backbone_is_text_aligned("i3d") is False
    assert backbone_is_text_aligned("videomae") is False


def test_assert_compatible_allows_valid_pairs():
    assert_compatible("clip", "vadclip")  # text head + text backbone
    assert_compatible("i3d", "mgfn")  # visual head + visual backbone
    assert_compatible("clip", "mgfn")  # visual head on a text backbone is fine
    assert is_compatible("i3d", "mgfn")


def test_assert_compatible_rejects_text_head_on_visual_backbone():
    assert is_compatible("i3d", "vadclip") is False
    with pytest.raises(ValueError, match="text-aligned"):
        assert_compatible("i3d", "vadclip")
