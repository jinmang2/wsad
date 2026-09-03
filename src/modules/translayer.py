"""Dual-branch temporal Transformer shared by UR-DMU and BN-WVAD.

Faithful, device-safe port of the official ``translayer.py`` (identical across
henrryzh1/UR-DMU and cool-xuan/BN-WVAD). The ``Attention`` is a **dual branch**:
``to_qkv`` projects to (q, k, v, t); branch 1 is standard ``softmax(qkᵀ/√d)·v``,
branch 2 is a **fixed** temporal distance-decay attention ``exp(-|i-j|/e)`` over a
4th projection ``t``; the two are concatenated and fused by ``to_out``. Module and
parameter names mirror the official state dict so the official checkpoints load
directly (``selfatt.layers.{i}.{0,1}.{norm,fn...}``).
"""

import math
import os

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

# Opt-in attention backends for the dual-branch temporal Transformer. Default
# "eager" keeps the bit-exact path the official-checkpoint verification relies on
# (UR-DMU max|Δ|=5.96e-08).
#
#   mem     branch-1 through scaled_dot_product_attention (flash/mem-efficient kernel,
#           ~1e-7 deviation) and branch-2 without materializing the (b, h, n, n) decay
#           matrix — so seg200 fits batch 64 on 8 GB GPUs instead of OOM-ing in the
#           O(T²) ``q·kᵀ``. Use for training the memory heads at the official batch size.
#   window  sliding-window (band) attention, O(T·W) memory and compute — see
#           ``docs/SPEC2_EFFICIENCY.md``. Branch-2 is computed **exactly** (only terms
#           with |i-j| > W are dropped, and those are ≤ exp(-W/e) ≈ 6e-11 at W=64), so
#           only branch-1 is approximated. Peak memory is linear in T, which is what
#           lets full-length / streaming sequences run without per-crop splitting.
#
#   causal  the same band, past-only (``j <= i``) in both branches. This is the online /
#           streaming mode: a frame is scored without any lookahead, so peak memory is
#           O(W) and a live deployment can emit a score the moment a frame arrives. The
#           AUC delta against `window` is the price of causality specifically, since the
#           two share the same band width.
#
# Leave unset for verification/eval.
_ATTN_IMPL = os.environ.get("WSAD_ATTN", "eager").lower()
# Half-width of the band for WSAD_ATTN=window; query i attends to |i-j| <= W.
_ATTN_WINDOW = int(os.environ.get("WSAD_ATTN_WINDOW", "64"))
# Query-block size for the windowed path. Peak mask/score memory is
# O(T · (chunk + 2·window)); larger = fewer kernel launches, more memory.
_ATTN_CHUNK = int(os.environ.get("WSAD_ATTN_CHUNK", "256"))
# Kernel behind the banded softmax: "sdpa" (default, the chunked masked-SDPA loop) or
# "flex" (torch.compile'd FlexAttention with a block-sparse mask). Measured on the
# RTX 2070 SUPER, both numerically equal to ~2e-7, flex is 3.5-5.9x faster at T >= 8192
# and slightly lighter; below that they trade places and flex pays a compile cost. Default
# stays sdpa — flex is the choice for long/streaming sequences. See docs/SPEC2_EFFICIENCY.md.
_ATTN_KERNEL = os.environ.get("WSAD_ATTN_KERNEL", "sdpa").lower()

# The decay is exp(-|i-j|/e) with e = exp(1), i.e. a geometric sequence in |i-j|
# with ratio r = exp(-1/e). Naming it once keeps the windowed path provably equal
# to the dense one on the terms it keeps.
_DECAY_RATIO = math.exp(-1.0 / math.e)


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


def _decay_column_sums(n: int, device, dtype) -> torch.Tensor:
    """The ``attn2.sum(-1)`` normalizer of the dense path, in closed form (O(n) memory).

    The official code divides the (symmetric) decay matrix by ``attn2.sum(-1)``, which
    broadcasts over the last axis — element (i, j) is scaled by the sum of column j.
    That sum is a two-sided geometric series around j and has an exact closed form, so
    the windowed path can reproduce the dense normalizer without ever building (n, n)::

        s[j] = 1 + Σ_{d=1..j} r^d + Σ_{d=1..n-1-j} r^d,   r = exp(-1/e)
    """
    r = _DECAY_RATIO
    j = torch.arange(n, device=device, dtype=torch.float64)
    left = r * (1.0 - r ** j) / (1.0 - r)
    right = r * (1.0 - r ** (n - 1 - j)) / (1.0 - r)
    return (1.0 + left + right).to(dtype)


def _banded_decay(t: torch.Tensor, window: int) -> torch.Tensor:
    """Branch-2 restricted to ``|i-j| <= window``, as a depthwise 1-D convolution.

    ``out[i] = Σ_j (r^|i-j| / s[j]) · t[j]`` is a convolution of ``t / s`` with the
    fixed symmetric kernel ``r^|d|``, so it costs O(T·W) and never materializes the
    (n, n) decay. The dropped tail is bounded by ``r^window`` (6e-11 at W=64), which
    is why windowing this branch is near-exact rather than an approximation.
    """
    b, h, n, d = t.shape
    w = min(window, n - 1)
    s = _decay_column_sums(n, t.device, t.dtype)
    u = t / s.view(1, 1, n, 1)

    offsets = torch.arange(-w, w + 1, device=t.device, dtype=torch.float32)
    kernel = torch.exp(-offsets.abs() / math.e).to(t.dtype).view(1, 1, -1)

    u = u.permute(0, 1, 3, 2).reshape(b * h * d, 1, n)
    out = F.conv1d(u, kernel, padding=w)
    return out.reshape(b, h, d, n).permute(0, 1, 3, 2)


# Band masks depend only on the block geometry, not on the data, and the same few
# shapes recur for every layer of every video — so build each one once.
_MASK_CACHE: dict = {}


def _band_mask(n_q: int, n_k: int, offset: int, window: int, causal: bool, device) -> torch.Tensor:
    """``True`` where query ``i`` may attend to key ``j``, for a query block starting
    ``offset`` into its key span: ``|i - j| <= window``, and additionally ``j <= i``
    when ``causal``."""
    key = (n_q, n_k, offset, window, causal, str(device))
    mask = _MASK_CACHE.get(key)
    if mask is None:
        qi = torch.arange(n_q, device=device).unsqueeze(1) + offset
        kj = torch.arange(n_k, device=device).unsqueeze(0)
        delta = qi - kj
        mask = (delta.abs() <= window) & (delta >= 0) if causal else (delta.abs() <= window)
        _MASK_CACHE[key] = mask
    return mask


# The interior limit of the decay normalizer: s[j] -> 1 + 2·Σ_{d>=1} r^d = (1+r)/(1-r).
# `_decay_column_sums` converges to this exponentially away from the sequence ends.
_DECAY_SUM_LIMIT = (1.0 + _DECAY_RATIO) / (1.0 - _DECAY_RATIO)


def _causal_banded_decay(t: torch.Tensor, window: int) -> torch.Tensor:
    """Branch-2 with the future dropped: ``out[i] = Σ_{i-W <= j <= i} (r^(i-j)/s) · t[j]``.

    Note the normalizer: the bidirectional path divides by ``attn2.sum(-1)``, which is a
    **whole-clip** quantity — it depends on the sequence length and on how far ``j`` sits
    from either end. That makes the official branch-2 unstreamable *in principle*: an
    online scorer does not know the length of a stream that has not finished, and scoring
    the same frame inside two differently-sized windows would give two different answers.

    So the causal backend uses the interior limit ``(1+r)/(1-r)`` instead, which the true
    normalizer approaches exponentially (``r^17 ~ 2e-3``). The consequence is deliberate
    and worth stating: causal scores are position-independent and therefore *exactly*
    reproducible chunk-by-chunk, at the cost of a small deviation from the offline model
    confined to the first and last ~20 snippets of a clip.

    Implemented as a causal 1-D convolution (left padding only), so it stays O(T·W) and,
    unlike the bidirectional version, never reads a frame that has not happened yet.
    """
    b, h, n, d = t.shape
    w = min(window, n - 1)
    u = (t / _DECAY_SUM_LIMIT).permute(0, 1, 3, 2).reshape(b * h * d, 1, n)

    lags = torch.arange(w, -1, -1, device=t.device, dtype=torch.float32)  # r^W ... r^0
    kernel = torch.exp(-lags / math.e).to(t.dtype).view(1, 1, -1)

    out = F.conv1d(F.pad(u, (w, 0)), kernel)[..., :n]
    return out.reshape(b, h, d, n).permute(0, 1, 3, 2)


_FLEX_CACHE: dict = {}


def _flex_window_attend(q, k, v, window: int, causal: bool):
    """Banded attention through FlexAttention's block-sparse kernel.

    Returns ``None`` if FlexAttention is unavailable or refuses this input, so the caller
    can fall back rather than fail — this path is an optimization, never a requirement.

    Two details matter and are easy to get wrong:
      * ``flex_attention`` **must** be wrapped in ``torch.compile``. Called eagerly it warns
        and materializes the full (T, T) score matrix, which throws away the entire point.
      * ``create_block_mask`` needs ``_compile=True`` at long T. Without it, mask
        construction itself allocates a dense mask and OOMs at T=32768 on an 8 GB card.
    """
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    except Exception:
        return None

    n = q.shape[2]
    key = (n, window, causal, str(q.device))
    entry = _FLEX_CACHE.get(key)
    if entry is None:
        def mask_mod(b, h, q_idx, kv_idx):
            delta = q_idx - kv_idx
            return (delta >= 0) & (delta <= window) if causal else (delta.abs() <= window)

        try:
            block_mask = create_block_mask(
                mask_mod, B=None, H=None, Q_LEN=n, KV_LEN=n, device=q.device, _compile=True
            )
        except Exception:
            _FLEX_CACHE[key] = False
            return None
        compiled = _FLEX_CACHE.get("_fn")
        if compiled is None:
            compiled = torch.compile(flex_attention, dynamic=False)
            _FLEX_CACHE["_fn"] = compiled
        entry = _FLEX_CACHE[key] = (block_mask, compiled)
    if entry is False:
        return None

    block_mask, compiled = entry
    try:
        return compiled(q, k, v, block_mask=block_mask)
    except Exception:
        _FLEX_CACHE[key] = False
        return None


def _window_attend(q, k, v, window: int, chunk: int, causal: bool = False) -> torch.Tensor:
    """Softmax attention restricted to a band, in query blocks (O(T·(chunk+2W)) memory).

    Each query block attends only to the key span it can reach, so neither the scores
    nor the mask are ever (n, n). Every row keeps at least its own position (``j == i``
    survives both the band and the causal constraint), so no row is fully masked.
    """
    n = q.shape[2]
    if window >= n - 1:
        # the band already covers the reachable span — only causality still constrains
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    if _ATTN_KERNEL == "flex":
        out = _flex_window_attend(q, k, v, window, causal)
        if out is not None:
            return out

    outs = []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        ks = max(0, start - window)
        ke = min(n, end if causal else end + window)
        mask = _band_mask(end - start, ke - ks, start - ks, window, causal, q.device)
        outs.append(
            F.scaled_dot_product_attention(
                q[:, :, start:end], k[:, :, ks:ke], v[:, :, ks:ke], attn_mask=mask
            )
        )
    return torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]


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

        if _ATTN_IMPL in ("window", "causal"):
            # Neither branch builds an (n, n) tensor: branch-1 is banded softmax over
            # query blocks, branch-2 is the exact banded decay as a convolution.
            causal = _ATTN_IMPL == "causal"
            out1 = _window_attend(q, k, v, _ATTN_WINDOW, _ATTN_CHUNK, causal=causal)
            out2 = (_causal_banded_decay if causal else _banded_decay)(t, _ATTN_WINDOW)
            out = torch.cat([out1, out2], dim=-1)
            out = rearrange(out, "b h n d -> b n (h d)")
            return self.to_out(out)

        # fixed temporal distance-decay attention (official, device-safe); (n, n)
        tmp_ones = torch.ones(n, device=x.device)
        tmp_n = torch.linspace(1, n, n, device=x.device)
        tg_tmp = torch.abs(tmp_n * tmp_ones - tmp_n.view(-1, 1))
        attn2 = torch.exp(-tg_tmp / torch.exp(torch.tensor(1.0, device=x.device)))
        attn2 = attn2 / attn2.sum(-1)  # (n, n), identical across b/h

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
