"""Characterization tests for the unbounded-gradient hazards found by the audit.

These do **not** assert that the code is safe — two of these sites produce NaN gradients on
a reachable input, and `docs/GRADIENT_HAZARDS.md` explains why they are deliberately left
alone. The tests exist so that the hazard is documented executably and so that anyone who
changes the behaviour does it on purpose rather than by accident.
"""

import torch

from src.models.bn_wvad.modeling_bn_wvad import BNWVADForVideoAnomalyDetection
from src.models.mgfn.modeling_mgfn import MGFNLayerNorm
from src.models.pel4vad.modeling_pel4vad import _POWER_NORM_EPS


def test_mgfn_layernorm_gradient_is_nan_on_a_constant_slice():
    """`eps` guards the division, not the sqrt — so `var == 0` still yields NaN.

    Reachable: I3D features are ReLU outputs, so an all-zero snippet has zero variance.
    """
    ln = MGFNLayerNorm(4)
    x = torch.zeros(1, 4, 3, requires_grad=True)
    ln(x).sum().backward()
    assert torch.isnan(x.grad).any(), "hazard is gone — update docs/GRADIENT_HAZARDS.md"


def test_bn_wvad_mahalanobis_gradient_is_nan_when_feats_equal_the_anchor():
    """The sibling at line 224 already adds 1e-12; this one does not."""
    feats = torch.zeros(1, 4, 3, requires_grad=True)
    d = BNWVADForVideoAnomalyDetection.get_mahalanobis_distance(
        feats, torch.zeros(4), torch.ones(4)
    )
    d.sum().backward()
    assert torch.isnan(feats.grad).any(), "hazard is gone — update docs/GRADIENT_HAZARDS.md"


def test_the_guarded_sites_stay_finite():
    """The contrast that makes the audit actionable: guarding costs one term."""
    x = torch.zeros(1, 4, 3, requires_grad=True)
    guarded = torch.sqrt((x**2).sum(dim=1) + 1e-12)
    guarded.sum().backward()
    assert torch.isfinite(x.grad).all()

    y = torch.zeros(5, requires_grad=True)
    (torch.sign(y) * torch.sqrt(y.abs() + _POWER_NORM_EPS)).sum().backward()
    assert torch.isfinite(y.grad).all()
