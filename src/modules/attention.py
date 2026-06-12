"""Shared multi-head self-attention with switchable kernels.

Two backends, selected per-module by ``attn_impl``:
  - ``"eager"`` (default): explicit ``softmax(QKᵀ/√d + bias)·V``. **Bit-exact** —
    use this when loading official pretrained weights and verifying numerical
    equivalence against a reference implementation.
  - ``"sdpa"``: ``torch.nn.functional.scaled_dot_product_attention``, which fuses
    the kernel and dispatches to **FlashAttention / memory-efficient** backends on
    supported GPUs. Faster + lower memory, but the flash/mem-efficient math may
    differ from eager at the ~1e-3 level, so it is opt-in (not for bit-exact
    weight verification).

An additive attention bias (e.g. UR-DMU's distance prior) is supported in both
backends — SDPA takes it as the additive ``attn_mask``, so even biased attention
keeps the fused-kernel speedup.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
        attn_impl: str = "eager",
    ):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        assert attn_impl in ("eager", "sdpa")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.attn_impl = attn_impl
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x: (B, T, D); attn_bias: broadcastable to (B, heads, T, T) or (T, T)
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.attn_impl == "sdpa":
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, T, T)
            if attn_bias is not None:
                attn = attn + attn_bias
            attn = attn.softmax(dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            out = attn @ v

        out = out.transpose(1, 2).reshape(b, t, d)
        return self.proj_drop(self.proj(out))


class TransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer block (MHSA + FFN) used by temporal encoders."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_impl: str = "eager",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim, heads=heads, dropout=dropout, attn_impl=attn_impl
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_bias=attn_bias)
        x = x + self.mlp(self.norm2(x))
        return x
