"""Guards and protocol checks for the Cosmos-Embed1 extractor.

These run without downloading the 2.4 GB checkpoint — they cover the parts that are
easy to get silently wrong: the leakage guard, the output-mode contract, and the
16-frame-snippet -> 8-model-frame subsampling that keeps the cache on the same
temporal grid as I3D/VideoMAE.
"""

import numpy as np
import pytest
from PIL import Image

from src.features.cosmos import CosmosEmbed1FeatureExtractor


def _bare(**attrs):
    """An instance without the (heavy) model, for pure-protocol methods."""
    ex = object.__new__(CosmosEmbed1FeatureExtractor)
    for k, v in attrs.items():
        setattr(ex, k, v)
    return ex


def test_anomaly_finetuned_checkpoint_is_refused():
    """Cosmos-Embed1-448p-anomaly-detection is trained on Vad-Reasoning, which contains
    UCF-Crime — scoring our test split with it would be label leakage."""
    with pytest.raises(ValueError, match="leakage"):
        CosmosEmbed1FeatureExtractor(model_name="nvidia/Cosmos-Embed1-448p-anomaly-detection")


def test_leakage_guard_can_be_overridden_explicitly():
    """The guard must be an opt-out, not a wall — but the opt-out has to be deliberate."""
    with pytest.raises(Exception) as e:
        CosmosEmbed1FeatureExtractor(
            model_name="nvidia/Cosmos-Embed1-448p-anomaly-detection",
            allow_leaky_checkpoint=True,
            device="cpu",
        )
    assert "leakage" not in str(e.value)


def test_unknown_output_mode_is_rejected():
    with pytest.raises(ValueError, match="output must be"):
        CosmosEmbed1FeatureExtractor(output="pooled")


def test_snippet_is_subsampled_to_the_model_frame_count():
    ex = _bare(model_frames=8)
    clip = [Image.fromarray(np.full((12, 10, 3), i, dtype=np.uint8)) for i in range(16)]
    got = ex._to_model_frames(clip)

    assert got.shape == (8, 3, 12, 10)  # (frames, C, H, W), channels-first uint8
    assert got.dtype == np.uint8
    # uniformly spaced over the snippet, endpoints included
    assert [int(f[0, 0, 0]) for f in got] == [0, 2, 4, 6, 9, 11, 13, 15]


def test_segment_pooling_matches_the_32_seg_convention():
    ex = _bare()
    feats = np.arange(40, dtype=np.float32).reshape(20, 2)
    got = ex._segment(feats, 5)
    assert got.shape == (5, 2)
    assert np.allclose(got[0], feats[0:4].mean(0))
    assert np.allclose(got[-1], feats[16:20].mean(0))
