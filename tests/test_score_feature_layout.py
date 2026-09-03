"""`score_feature` must agree with the reported-eval path on both cache layouts.

The two caches are shaped differently and cannot be distinguished from the shape alone —
I3D test features are `(T, ncrops, D)`, CLIP features are `(ncrops, T, D)`. Assuming the
I3D one silently transposed CLIP features, so a 20-snippet video produced 16 frame scores
instead of 320, with nothing raising.
"""

import numpy as np
import torch

from src.eval_matrix import score_video
from src.inference import score_feature
from src.models.ur_dmu.configuration_ur_dmu import URDMUConfig
from src.models.ur_dmu.modeling_ur_dmu import URDMUForVideoAnomalyDetection


def _model(dim=64):
    torch.manual_seed(0)
    cfg = URDMUConfig(feature_size=dim, hidden_size=64, num_layers=1, num_heads=2, mem_size=8)
    return URDMUForVideoAnomalyDetection(cfg).eval()


@torch.no_grad()
def test_i3d_layout_matches_the_reported_path():
    feat = np.random.RandomState(0).rand(20, 10, 64).astype(np.float32)  # (T, ncrops, D)
    m = _model()
    assert np.allclose(score_video(m, feat, "i3d", "ur_dmu"), score_feature(m, feat), atol=1e-6)


@torch.no_grad()
def test_clip_layout_needs_the_backbone_and_then_matches():
    feat = np.random.RandomState(1).rand(1, 20, 64).astype(np.float32)  # (ncrops, T, D)
    m = _model()
    want = score_video(m, feat, "clip", "sultani")

    assert len(score_feature(m, feat, backbone="clip")) == len(want) == 20 * 16
    assert np.allclose(score_feature(m, feat, backbone="clip"), want, atol=1e-6)

    # the old unconditional assumption: T read as the crop axis, 20 snippets -> 1
    assert len(score_feature(m, feat)) == 16


@torch.no_grad()
def test_per_crop_split_is_a_memory_optimization_not_a_semantic_change():
    """`eval_matrix` scores UR-DMU/BN-WVAD one crop at a time to fit 8 GB. That must be
    numerically identical to one forward over all crops, or the reported numbers depend on
    a memory workaround."""
    feat = np.random.RandomState(2).rand(16, 10, 64).astype(np.float32)
    m = _model()
    per_crop = score_video(m, feat, "i3d", "ur_dmu")     # loops crops, averages
    all_crops = score_feature(m, feat)                    # one forward
    assert np.allclose(per_crop, all_crops, atol=1e-6), np.abs(per_crop - all_crops).max()
