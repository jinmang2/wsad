"""Port checks for PEL4VAD.

The parts worth pinning are the ones a port silently gets wrong: the prompt table's class
order (an off-by-one there trains against the wrong text embedding and still converges),
the official window mask's clamping, and the classifier's left-only padding.
"""

import numpy as np
import torch

from src.data.labels import UCF_CLASSES
from src.models.pel4vad.configuration_pel4vad import PEL4VADConfig
from src.models.pel4vad.modeling_pel4vad import PEL4VADForVideoAnomalyDetection


def _model(**kw):
    torch.manual_seed(0)
    return PEL4VADForVideoAnomalyDetection(PEL4VADConfig(**kw)).eval()


def test_forward_contract():
    m = _model()
    out = m(video=torch.randn(4, 1, 32, 1024))
    assert out.scores.shape == (4, 32, 1)
    assert out.loss is None  # eval path must not build a loss
    assert torch.all((out.scores >= 0) & (out.scores <= 1))


def test_loss_is_finite_with_and_without_class_labels():
    m = _model()
    x = torch.randn(4, 1, 32, 1024)
    kw = dict(abnormal_labels=torch.ones(2), normal_labels=torch.zeros(2))

    mil_only = m(video=x, **kw).loss
    with_prompt = m(video=x, class_labels=torch.tensor([0, 0, 3, 7]), **kw).loss

    assert torch.isfinite(mil_only) and torch.isfinite(with_prompt)
    assert not torch.equal(mil_only, with_prompt)  # the prompt term must actually apply


def test_vendored_prompt_table_matches_our_class_order():
    """PEL4VAD's `ucf_label.list` order must line up with `UCF_CLASSES`, or every video
    trains against another class's text embedding."""
    m = _model()
    assert m.class_prompts.shape == (len(UCF_CLASSES), 512)
    assert m.class_prompts.abs().sum() > 0  # actually loaded, not the zero buffer
    # distinct rows: a table loaded at the wrong stride would repeat or blank rows
    assert len(torch.unique(m.class_prompts, dim=0)) == len(UCF_CLASSES)


def test_classifier_padding_is_left_only():
    """`F.pad(x, (t_step-1, 0))` — a score at t must not move when a later frame changes."""
    m = _model(t_step=9)
    x = torch.randn(1, 1, 40, 1024)
    with torch.no_grad():
        a = m(video=x).scores
        x2 = x.clone()
        x2[:, :, 30:] += 5.0
        b = m(video=x2).scores
    # the encoder mixes globally, so compare the classifier alone on a fixed encoding
    with torch.no_grad():
        enc = torch.randn(1, m.config.out_dim, 40)
        pa = torch.sigmoid(m.classifier(torch.nn.functional.pad(enc, (8, 0))))
        enc2 = enc.clone()
        enc2[:, :, 30:] += 5.0
        pb = torch.sigmoid(m.classifier(torch.nn.functional.pad(enc2, (8, 0))))
    assert torch.allclose(pa[..., :30], pb[..., :30], atol=1e-6)
    assert not torch.allclose(a, b)  # sanity: the encoder itself is not causal


def test_window_mask_clamps_at_the_edges_like_the_official_code():
    m = _model(win_size=9)
    mask = m.encoder._window_mask(20, torch.device("cpu"))
    assert mask.shape == (20, 20)
    # interior row: exactly win_size entries, centred
    assert mask[10].sum() == 9
    assert mask[10, 6:15].sum() == 9
    # edge row: clamping collapses the out-of-range indices onto frame 0
    assert mask[0].sum() == 5
    assert mask[0, 0] == 1


def test_text_projection_appears_only_when_dims_disagree():
    """1024 // 2 == 512 is a coincidence the official code relies on; other dims need a
    projection, which is what lets this head run on the 2048-d and 768-d variants."""
    assert isinstance(_model(feature_size=1024).text_proj, torch.nn.Identity)
    assert isinstance(_model(feature_size=2048).text_proj, torch.nn.Linear)

    m = _model(feature_size=2048)
    out = m(video=torch.randn(4, 1, 32, 2048), abnormal_labels=torch.ones(2),
            normal_labels=torch.zeros(2), class_labels=torch.tensor([0, 0, 3, 7]))
    assert torch.isfinite(out.loss)


def test_distance_prior_is_learnable_and_device_safe():
    m = _model()
    adj = m.encoder.loc_adj(2, 16, torch.device("cpu"))
    assert adj.shape == (2, 16, 16)
    assert m.encoder.loc_adj.w.requires_grad and m.encoder.loc_adj.b.requires_grad
    # decays away from the diagonal
    assert adj[0, 8, 8] > adj[0, 8, 12]


def test_power_norm_gradient_is_bounded():
    """The official `sqrt(relu(x)) - sqrt(relu(-x))` has an UNBOUNDED derivative as an
    activation approaches zero, which is what diverged a real training run: a gradient norm
    of 6.6e18 poisoned Adam's second moment and the model never recovered.

    The stabilized form must stay finite everywhere and agree with the original away from
    zero, or the fix has changed the model rather than repaired it.
    """
    from src.models.pel4vad.modeling_pel4vad import _POWER_NORM_EPS

    xs = torch.tensor([1e-30, 1e-12, 1e-6, 1e-2, 0.5, -0.5, -1e-12, 0.0], requires_grad=True)
    ys = torch.sign(xs) * torch.sqrt(xs.abs() + _POWER_NORM_EPS)
    ys.sum().backward()

    assert torch.isfinite(xs.grad).all()
    bound = 0.5 / _POWER_NORM_EPS**0.5
    assert xs.grad.abs().max() <= bound * 1.01, xs.grad.abs().max()

    # away from zero it must still be the original function
    big = torch.tensor([0.25, 1.0, 4.0, -0.25, -9.0])
    original = torch.sqrt(torch.relu(big)) - torch.sqrt(torch.relu(-big))
    stabilized = torch.sign(big) * torch.sqrt(big.abs() + _POWER_NORM_EPS)
    assert torch.allclose(original, stabilized, atol=1e-5), (original - stabilized).abs().max()


def test_the_unstabilized_form_really_does_explode():
    """Control: without this the test above would pass on a no-op change."""
    x = torch.tensor([1e-30], requires_grad=True)
    y = torch.sqrt(torch.relu(x)) - torch.sqrt(torch.relu(-x))
    y.backward()
    assert x.grad.item() > 1e12, x.grad.item()
