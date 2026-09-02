"""Numerics for the ``WSAD_ATTN=window`` band attention (Spec 2, docs/SPEC2_EFFICIENCY.md).

The windowed path must be a *near-exact* drop-in, not a loose approximation:

* branch-2 (the fixed ``exp(-|i-j|/e)`` decay) is computed exactly on the terms it
  keeps, including the dense path's ``attn2.sum(-1)`` normalizer, so the only error is
  the dropped tail ``r^W`` (~6e-11 at W=64);
* branch-1 (learned ``qkᵀ``) is the genuine approximation — with a full-width window it
  must reproduce ``eager`` exactly, which is what pins the implementation down.
"""

import math

import pytest
import torch

from src.modules import translayer as T


@pytest.fixture(autouse=True)
def _restore_impl():
    saved = (T._ATTN_IMPL, T._ATTN_WINDOW, T._ATTN_CHUNK)
    yield
    T._ATTN_IMPL, T._ATTN_WINDOW, T._ATTN_CHUNK = saved


def _dense_decay(n: int) -> torch.Tensor:
    """The dense branch-2 matrix, transcribed from the official/eager path."""
    tmp_n = torch.linspace(1, n, n)
    tg = torch.abs(tmp_n * torch.ones(n) - tmp_n.view(-1, 1))
    attn2 = torch.exp(-tg / torch.exp(torch.tensor(1.0)))
    return attn2 / attn2.sum(-1)


def test_decay_column_sums_match_dense_normalizer():
    n = 137
    dense = torch.exp(-torch.cdist(torch.arange(n).float().view(-1, 1), torch.arange(n).float().view(-1, 1)) / math.e)
    got = T._decay_column_sums(n, torch.device("cpu"), torch.float32)
    assert torch.allclose(got, dense.sum(-1), atol=1e-5), (got - dense.sum(-1)).abs().max()


def test_banded_decay_matches_dense_when_window_covers_sequence():
    torch.manual_seed(0)
    n = 64
    t = torch.randn(2, 3, n, 8)
    want = torch.einsum("nm,bhmd->bhnd", _dense_decay(n), t)
    got = T._banded_decay(t, window=n)
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()


def test_banded_decay_tail_is_negligible_at_w64():
    """Truncating the decay at W=64 changes it by ~r^64, not by anything measurable."""
    torch.manual_seed(0)
    n = 300
    t = torch.randn(1, 2, n, 8)
    want = torch.einsum("nm,bhmd->bhnd", _dense_decay(n), t)
    got = T._banded_decay(t, window=64)
    assert (got - want).abs().max() < 1e-6


def _forward(impl, x, attn, window=64, chunk=256):
    T._ATTN_IMPL, T._ATTN_WINDOW, T._ATTN_CHUNK = impl, window, chunk
    with torch.no_grad():
        return attn(x)


def test_full_width_window_reproduces_eager():
    """With W >= n the band is the whole sequence, so both branches must match eager."""
    torch.manual_seed(0)
    n = 96
    x = torch.randn(2, n, 128)
    attn = T.Attention(128, heads=4, dim_head=32).eval()

    eager = _forward("eager", x, attn)
    windowed = _forward("window", x, attn, window=n)
    assert torch.allclose(eager, windowed, atol=1e-5), (eager - windowed).abs().max()


def test_chunking_does_not_change_the_result():
    torch.manual_seed(0)
    n = 200
    x = torch.randn(1, n, 128)
    attn = T.Attention(128, heads=4, dim_head=32).eval()

    a = _forward("window", x, attn, window=32, chunk=256)
    b = _forward("window", x, attn, window=32, chunk=17)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_window_stays_close_to_eager_at_seg200():
    """The real operating point: seg200 with W=64. Only branch-1 is approximated."""
    torch.manual_seed(0)
    n = 200
    x = torch.randn(2, n, 128)
    attn = T.Attention(128, heads=4, dim_head=32).eval()

    eager = _forward("eager", x, attn)
    windowed = _forward("window", x, attn, window=64)
    rel = (eager - windowed).abs().max() / eager.abs().max()
    assert rel < 0.25, rel


def test_long_sequence_runs_without_quadratic_tensors():
    """8k frames: the dense path would need a (b,h,8k,8k) score tensor; this must not."""
    torch.manual_seed(0)
    n = 8192
    x = torch.randn(1, n, 64)
    attn = T.Attention(64, heads=2, dim_head=16).eval()
    out = _forward("window", x, attn, window=64, chunk=512)
    assert out.shape == (1, n, 64)
    assert torch.isfinite(out).all()


def _causal_dense_decay(n: int) -> torch.Tensor:
    """Reference for `causal`: the decay, future masked, scaled by the *interior* limit.

    The causal backend deliberately drops the whole-clip ``attn2.sum(-1)`` normalizer for
    its position-independent limit — see `_causal_banded_decay` — because the clip-wide
    version cannot be computed on an unfinished stream.
    """
    tmp_n = torch.linspace(1, n, n)
    tg = torch.abs(tmp_n * torch.ones(n) - tmp_n.view(-1, 1))
    attn2 = torch.exp(-tg / torch.exp(torch.tensor(1.0))) / T._DECAY_SUM_LIMIT
    return attn2 * torch.tril(torch.ones(n, n))


def test_causal_normalizer_is_the_interior_limit_of_the_bidirectional_one():
    """Away from the ends the two normalizers agree, so the causal change is an edge effect."""
    n = 200
    exact = T._decay_column_sums(n, torch.device("cpu"), torch.float32)
    assert abs(float(exact[n // 2]) - T._DECAY_SUM_LIMIT) < 1e-5
    assert float(exact[0]) < T._DECAY_SUM_LIMIT  # the ends are where they differ


def test_causal_decay_matches_the_masked_dense_decay():
    torch.manual_seed(0)
    n = 64
    t = torch.randn(2, 3, n, 8)
    want = torch.einsum("nm,bhmd->bhnd", _causal_dense_decay(n), t)
    got = T._causal_banded_decay(t, window=n)
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()


def test_causal_output_never_depends_on_the_future():
    """The property that makes online scoring possible: changing frame k must leave every
    output before k bit-identical."""
    torch.manual_seed(0)
    n, k = 128, 80
    x = torch.randn(1, n, 128)
    attn = T.Attention(128, heads=4, dim_head=32).eval()

    a = _forward("causal", x, attn, window=32)
    x2 = x.clone()
    x2[:, k:] = torch.randn_like(x2[:, k:])
    b = _forward("causal", x2, attn, window=32)

    assert torch.equal(a[:, :k], b[:, :k]), (a[:, :k] - b[:, :k]).abs().max()


def test_bidirectional_window_does_depend_on_the_future():
    """Control for the test above — otherwise it could pass on a broken implementation."""
    torch.manual_seed(0)
    n, k = 128, 80
    x = torch.randn(1, n, 128)
    attn = T.Attention(128, heads=4, dim_head=32).eval()

    a = _forward("window", x, attn, window=32)
    x2 = x.clone()
    x2[:, k:] = torch.randn_like(x2[:, k:])
    b = _forward("window", x2, attn, window=32)

    assert not torch.equal(a[:, :k], b[:, :k])


def test_causal_full_width_matches_a_dense_causal_reference():
    """With W >= n the band is unlimited, so `causal` must equal plain causal attention."""
    torch.manual_seed(0)
    n = 96
    x = torch.randn(2, n, 128)
    attn = T.Attention(128, heads=4, dim_head=32).eval()

    got = _forward("causal", x, attn, window=n)

    with torch.no_grad():  # dense reference, built the same way as the eager path
        qkvt = attn.to_qkv(x).chunk(4, dim=-1)
        from einops import rearrange

        q, k, v, t = (rearrange(u, "b n (h d) -> b h n d", h=attn.heads) for u in qkvt)
        dots = (q @ k.transpose(-1, -2)) * attn.scale
        dots = dots.masked_fill(torch.tril(torch.ones(n, n)) == 0, float("-inf"))
        out1 = attn.attend(dots) @ v
        out2 = torch.einsum("nm,bhmd->bhnd", _causal_dense_decay(n), t)
        want = attn.to_out(rearrange(torch.cat([out1, out2], dim=-1), "b h n d -> b n (h d)"))

    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()
