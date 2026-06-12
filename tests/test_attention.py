"""Optimization-path tests for the shared attention module.

The ``eager`` and ``sdpa`` backends must agree numerically when fed identical
weights, so that switching ``attn_impl`` for speed never changes results enough
to break pretrained-weight equivalence (on the SDPA math backend they match to
floating-point tolerance; flash/mem-efficient GPU kernels may differ ~1e-3).
"""

import torch

from src.modules import MultiHeadSelfAttention


def test_eager_sdpa_equivalence():
    torch.manual_seed(0)
    x = torch.randn(2, 32, 512)

    eager = MultiHeadSelfAttention(512, heads=8, attn_impl="eager").eval()
    sdpa = MultiHeadSelfAttention(512, heads=8, attn_impl="sdpa").eval()
    sdpa.load_state_dict(eager.state_dict())

    with torch.no_grad():
        a, b = eager(x), sdpa(x)

    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max().item()


def test_additive_bias_supported_in_both_backends():
    torch.manual_seed(0)
    x = torch.randn(2, 16, 128)
    bias = torch.randn(1, 1, 16, 16)

    eager = MultiHeadSelfAttention(128, heads=4, attn_impl="eager").eval()
    sdpa = MultiHeadSelfAttention(128, heads=4, attn_impl="sdpa").eval()
    sdpa.load_state_dict(eager.state_dict())

    with torch.no_grad():
        a = eager(x, attn_bias=bias)
        b = sdpa(x, attn_bias=bias)

    assert torch.allclose(a, b, atol=1e-4)
