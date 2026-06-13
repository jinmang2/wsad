"""Dual-branch temporal Transformer shared by UR-DMU and BN-WVAD.

Faithful, device-safe port of the official ``translayer.py`` (identical across
henrryzh1/UR-DMU and cool-xuan/BN-WVAD). The ``Attention`` is a **dual branch**:
``to_qkv`` projects to (q, k, v, t); branch 1 is standard ``softmax(qkᵀ/√d)·v``,
branch 2 is a **fixed** temporal distance-decay attention ``exp(-|i-j|/e)`` over a
4th projection ``t``; the two are concatenated and fused by ``to_out``. Module and
parameter names mirror the official state dict so the official checkpoints load
directly (``selfatt.layers.{i}.{0,1}.{norm,fn...}``).
"""

import torch
from einops import rearrange
from torch import nn


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

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn1 = self.attend(dots)

        # fixed temporal distance-decay attention (official, device-safe)
        tmp_ones = torch.ones(n, device=x.device)
        tmp_n = torch.linspace(1, n, n, device=x.device)
        tg_tmp = torch.abs(tmp_n * tmp_ones - tmp_n.view(-1, 1))
        attn2 = torch.exp(-tg_tmp / torch.exp(torch.tensor(1.0, device=x.device)))
        attn2 = (attn2 / attn2.sum(-1)).unsqueeze(0).unsqueeze(1).repeat(b, self.heads, 1, 1)

        out = torch.cat([torch.matmul(attn1, v), torch.matmul(attn2, t)], dim=-1)
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

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
