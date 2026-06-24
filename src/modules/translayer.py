"""Dual-branch temporal Transformer shared by UR-DMU and BN-WVAD.

Faithful, device-safe port of the official ``translayer.py`` (identical across
henrryzh1/UR-DMU and cool-xuan/BN-WVAD). The ``Attention`` is a **dual branch**:
``to_qkv`` projects to (q, k, v, t); branch 1 is standard ``softmax(qkᵀ/√d)·v``,
branch 2 is a **fixed** temporal distance-decay attention ``exp(-|i-j|/e)`` over a
4th projection ``t``; the two are concatenated and fused by ``to_out``. Module and
parameter names mirror the official state dict so the official checkpoints load
directly (``selfatt.layers.{i}.{0,1}.{norm,fn...}``).
"""

import os

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

# Opt-in memory-efficient attention for the dual-branch temporal Transformer. Default
# "eager" keeps the bit-exact path the official-checkpoint verification relies on
# (UR-DMU max|Δ|=5.96e-08). "mem" routes branch-1 through scaled_dot_product_attention
# (flash/mem-efficient kernel, ~1e-7 deviation) and computes branch-2 without
# materializing the (b, h, n, n) decay matrix — so seg200 fits batch 64 on 8 GB GPUs
# instead of OOM-ing in the O(T²) ``q·kᵀ``. Set WSAD_ATTN=mem for training the memory
# heads at the official batch size; leave unset for verification/eval.
_ATTN_IMPL = os.environ.get("WSAD_ATTN", "eager").lower()


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Dual-branch attention: learned ``qk·v`` + fixed distance-decay over ``t``."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 4, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(2 * inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.size()
        qkvt = self.to_qkv(x).chunk(4, dim=-1)
        q, k, v, t = map(lambda u: rearrange(u, "b n (h d) -> b h n d", h=self.heads), qkvt)

        # fixed temporal distance-decay attention (official, device-safe); (n, n)
        tmp_ones = torch.ones(n, device=x.device)
        tmp_n = torch.linspace(1, n, n, device=x.device)
        tg_tmp = torch.abs(tmp_n * tmp_ones - tmp_n.view(-1, 1))
        attn2 = torch.exp(-tg_tmp / torch.exp(torch.tensor(1.0, device=x.device)))
        attn2 = attn2 / attn2.sum(-1)  # row-normalized (n, n), identical across b/h

        if _ATTN_IMPL == "mem":
            # branch-1: flash/mem-efficient kernel (no materialized (b,h,n,n) dots);
            # branch-2: einsum reuses the shared (n,n) decay (no b*h repeat). Both avoid
            # the O(T²) tensors that OOM at batch 64 / seg200.
            out1 = F.scaled_dot_product_attention(q, k, v)
            out2 = torch.einsum("nm,bhmd->bhnd", attn2, t)
            out = torch.cat([out1, out2], dim=-1)
        else:  # eager: the bit-exact path the official-ckpt verification depends on
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn1 = self.attend(dots)
            attn2b = attn2.unsqueeze(0).unsqueeze(1).repeat(b, self.heads, 1, 1)
            out = torch.cat([torch.matmul(attn1, v), torch.matmul(attn2b, t)], dim=-1)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
                for _ in range(depth)
            ]
        )
        # activation checkpointing (off by default — eval/verify keep full activations).
        # The O(T²) attention at seg200 is the activation bulk; recomputing it in the
        # backward lets the full batch-64 forward fit 8 GB (BN sees the real batch).
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    @staticmethod
    def _block(attn, ff, x):
        x = attn(x) + x
        return ff(x) + x

    def forward(self, x):
        for attn, ff in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(self._block, attn, ff, x, use_reentrant=False)
            else:
                x = self._block(attn, ff, x)
        return x
