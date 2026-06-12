"""Forward/backward contract tests for registered WSVAD models.

These run offline (synthetic features, no HF download) and verify the shared
runner contract: ``forward(video, abnormal_labels, normal_labels) -> .loss``
and ``forward(video) -> .scores`` of shape ``(B, T, 1)``.
"""

import pytest
import torch

import src.models  # noqa: F401  (triggers registration)
from src.registry import MODELS

NCROPS = 10
T = 32
DIM = 2049  # 2048 features + appended magnitude channel


def _build(name):
    if name == "mgfn":
        from src.models.mgfn import MGFNConfig, MGFNForVideoAnomalyDetection

        return MGFNForVideoAnomalyDetection(MGFNConfig())
    if name == "rtfm":
        from src.models.rtfm import RTFMConfig, RTFMForVideoAnomalyDetection

        return RTFMForVideoAnomalyDetection(RTFMConfig())
    if name == "sultani":
        from src.models.sultani import (
            SultaniConfig,
            SultaniForVideoAnomalyDetection,
        )

        return SultaniForVideoAnomalyDetection(SultaniConfig())
    if name == "clip_tsa":
        from src.models.clip_tsa import (
            CLIPTSAConfig,
            CLIPTSAForVideoAnomalyDetection,
        )

        return CLIPTSAForVideoAnomalyDetection(CLIPTSAConfig())
    if name == "ur_dmu":
        from src.models.ur_dmu import (
            URDMUConfig,
            URDMUForVideoAnomalyDetection,
        )

        return URDMUForVideoAnomalyDetection(URDMUConfig())
    if name == "vadclip":
        from src.models.vadclip import (
            VadCLIPConfig,
            VadCLIPForVideoAnomalyDetection,
        )

        return VadCLIPForVideoAnomalyDetection(VadCLIPConfig())
    if name == "gs_moe":
        from src.models.gs_moe import (
            GSMoEConfig,
            GSMoEForVideoAnomalyDetection,
        )

        # smaller for a fast test (fewer experts than the 13-class default)
        return GSMoEForVideoAnomalyDetection(GSMoEConfig(num_experts=3))
    if name == "tpwng":
        from src.models.tpwng import (
            TPWNGConfig,
            TPWNGForVideoAnomalyDetection,
        )

        return TPWNGForVideoAnomalyDetection(TPWNGConfig())
    if name == "s3r":
        from src.models.s3r import (
            S3RConfig,
            S3RForVideoAnomalyDetection,
        )

        return S3RForVideoAnomalyDetection(S3RConfig())
    if name == "bn_wvad":
        from src.models.bn_wvad import (
            BNWVADConfig,
            BNWVADForVideoAnomalyDetection,
        )

        return BNWVADForVideoAnomalyDetection(BNWVADConfig())
    raise ValueError(name)


MODEL_NAMES = [
    "mgfn",
    "rtfm",
    "sultani",
    "clip_tsa",
    "ur_dmu",
    "vadclip",
    "gs_moe",
    "tpwng",
    "s3r",
    "bn_wvad",
]

# models whose anomaly score is an unbounded rank-score, not a [0,1] probability
UNBOUNDED_SCORE = {"bn_wvad"}


def test_registry_has_all_models():
    assert set(MODEL_NAMES).issubset(set(MODELS.keys()))


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_inference_scores_shape(name):
    model = _build(name).eval()
    with torch.no_grad():
        out = model(video=torch.randn(1, NCROPS, T, DIM))
    assert out.scores.shape == (1, T, 1)
    assert torch.all(torch.isfinite(out.scores)) and torch.all(out.scores >= 0)
    if name not in UNBOUNDED_SCORE:
        assert torch.all(out.scores <= 1)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_train_loss_and_backward(name):
    model = _build(name).train()
    # 2 normal + 2 abnormal samples (runner concatenates normal-first)
    video = torch.randn(4, NCROPS, T, DIM)
    out = model(
        video=video,
        abnormal_labels=torch.ones(2),
        normal_labels=torch.zeros(2),
    )
    assert out.loss is not None
    assert torch.isfinite(out.loss)
    out.loss.backward()
    grads = [
        p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
    ]
    assert len(grads) > 0


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_registry_build_matches_direct(name):
    cls = MODELS.get(name)
    assert cls is type(_build(name))
