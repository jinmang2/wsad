"""GS-MoE: Mixture of Experts Guided by Gaussian Splatters (ICCV'25).

Slot mapping (WSAD_INTEGRATION_PLAN.md section 4):
  - slot 2 (temporal encoder): per-expert Transformer block + projection.
  - slot 4 (scoring head): N class-specialized experts each emit a per-snippet
    score; a Gate model fuses the expert score-sequences into the final score.
  - slot 5 (loss): top-k-norm MIL + smoothness + sparsity + Temporal Gaussian
    Splatting (TGS) self-training on the abnormal curve.

Honors the shared runner contract:
``forward(video, abnormal_labels=None, normal_labels=None) -> ModelOutput`` with
``.loss`` (None at inference) and ``.scores`` of shape ``(B, T, 1)``.

Official code unreleased — this is a **paper-faithful** reimplementation
(arXiv 2508.06318, GS-MoE ICCV'25), built section-by-section from the paper:
  - Expert = Transformer block (2-head; LN→MHSA→res, LN→proj 1024→512→ReLU→1024→res)
    + MLP ``1024→512→256→128→64→1`` (GELU between layers, sigmoid out).
  - Gate (3 stages, §gate): (a) refine N expert scores → 1024; (b) **bi-directional
    cross-attention** — branch 1: Q,K = task-aware features, V = projected scores;
    branch 2: Q,K = projected scores, V = task-aware features; concat → 2048;
    (c) Transformer block (4-head, 2048) + MLP ``2048→1024→512→256→128→1``.
  - TGS loss (Eqs. 3–6): peak detection (prominence 0.2) → Gaussian splat →
    pseudo-label → ``L_topk-norm + BCE(y, ŷ)``; warm-up (epoch 1) = MIL only.
The "task-aware features" are UR-DMU encoder outputs in the paper; here the input
projection stands in for that pipeline (documented). Class-specialized expert
supervision activates when per-video ``class_labels`` are provided.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from src.loss import SparsityLoss, TemporalSmoothnessLoss
from src.loss.tgs import TemporalGaussianSplattingLoss
from src.modules import MultiHeadSelfAttention
from src.registry import MODELS

from .configuration_gs_moe import GSMoEConfig


@dataclass
class GSMoEVideoAnomalyDetectionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    scores: Optional[torch.FloatTensor] = None
    expert_scores: Optional[torch.FloatTensor] = None  # (B, num_experts, T)


class _AttnBlock(nn.Module):
    """LN -> MHSA -> residual -> LN -> projection-FFN -> residual.

    The FFN is the paper's ``h -> 512 -> h`` projection (not a 4x FFN), which
    keeps each expert light (~paper's ~500K-1M budget) rather than ballooning.
    """

    def __init__(self, h: int, heads: int, dropout: float, attn_impl: str):
        super().__init__()
        self.norm1 = nn.LayerNorm(h)
        self.attn = MultiHeadSelfAttention(
            h, heads=heads, dropout=dropout, attn_impl=attn_impl
        )
        self.norm2 = nn.LayerNorm(h)
        self.ffn = nn.Sequential(nn.Linear(h, 512), nn.ReLU(), nn.Linear(512, h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Expert(nn.Module):
    """One class-specialized scorer: attention block + 1024->512->256->128->64->1 MLP."""

    def __init__(self, config: GSMoEConfig):
        super().__init__()
        h = config.hidden_size
        self.block = _AttnBlock(
            h, config.expert_heads, config.dropout_rate, config.attn_impl
        )
        self.mlp = nn.Sequential(
            nn.Linear(h, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        # x: (B, T, h) -> score (B, T, 1), feature (B, T, h)
        h = self.block(x)
        return self.mlp(h), h


class _BiCrossAttention(nn.Module):
    """Paper's bi-directional cross-attention between task-aware features ``A`` and
    projected expert scores ``B`` (both ``(B,T,dim)``):
      branch 1: Q,K = A, V = B  ->  out1
      branch 2: Q,K = B, V = A  ->  out2
    concat([out1, out2]) -> ``(B, T, 2*dim)``.
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.a_q = nn.Linear(dim, dim, bias=False)
        self.a_k = nn.Linear(dim, dim, bias=False)
        self.a_v = nn.Linear(dim, dim, bias=False)
        self.b_q = nn.Linear(dim, dim, bias=False)
        self.b_k = nn.Linear(dim, dim, bias=False)
        self.b_v = nn.Linear(dim, dim, bias=False)

    def _attn(self, q, k, v):
        b, t, d = q.shape
        h = self.heads
        q = q.view(b, t, h, d // h).transpose(1, 2)
        k = k.view(b, t, h, d // h).transpose(1, 2)
        v = v.view(b, t, h, d // h).transpose(1, 2)
        a = (q @ k.transpose(-2, -1)) * self.scale
        a = a.softmax(dim=-1)
        o = a @ v
        return o.transpose(1, 2).reshape(b, t, d)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        out1 = self._attn(self.a_q(a), self.a_k(a), self.b_v(b))  # Q,K=A, V=B
        out2 = self._attn(self.b_q(b), self.b_k(b), self.a_v(a))  # Q,K=B, V=A
        return torch.cat([out1, out2], dim=-1)  # (B, T, 2*dim)


class GateModel(nn.Module):
    """Paper's 3-stage gate: (a) refine N expert scores -> ``dim``; (b) bi-directional
    cross-attention with the task-aware features -> ``2*dim``; (c) Transformer block
    (4-head) + MLP ``2*dim->dim->512->256->128->1``.
    """

    def __init__(self, config: GSMoEConfig):
        super().__init__()
        h = config.hidden_size
        g = 2 * h
        self.refine = nn.Linear(config.num_experts, h)
        self.cross = _BiCrossAttention(h, config.expert_heads)
        self.block = _AttnBlock(g, config.gate_heads, config.dropout_rate, config.attn_impl)
        self.mlp = nn.Sequential(
            nn.Linear(g, h),
            nn.GELU(),
            nn.Linear(h, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, expert_scores: torch.Tensor, task_aware: torch.Tensor) -> torch.Tensor:
        # expert_scores: (B, E, T); task_aware: (B, T, dim) -> final score (B, T, 1)
        scores = self.refine(expert_scores.permute(0, 2, 1))  # (B, T, dim)
        x = self.cross(task_aware, scores)  # (B, T, 2*dim)
        x = self.block(x)
        return self.mlp(x)


class GSMoEPreTrainedModel(PreTrainedModel):
    config_class = GSMoEConfig
    base_model_prefix = "gs_moe"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    @property
    def dummy_inputs(self):
        return {"video": torch.randn(8, 10, 32, 2049)}


@MODELS.register("gs_moe")
class GSMoEForVideoAnomalyDetection(GSMoEPreTrainedModel):
    def __init__(self, config: GSMoEConfig):
        super().__init__(config)
        self.input_proj = nn.Linear(config.feature_size, config.hidden_size)
        self.experts = nn.ModuleList(
            [Expert(config) for _ in range(config.num_experts)]
        )
        self.gate = GateModel(config)

        self.smooth = TemporalSmoothnessLoss(config.lambda_smooth)
        self.sparse = SparsityLoss(config.lambda_sparse)
        self.tgs = TemporalGaussianSplattingLoss(config.tgs_prominence)
        self.bce = nn.BCELoss()
        # toggled by a training callback: warm-up (epoch 1) trains MIL only.
        self.tgs_enabled = True
        self._force_split = False

    @property
    def force_split(self) -> bool:
        return self._force_split

    @force_split.setter
    def force_split(self, val: bool):
        self._force_split = val

    def _topk_mil(self, frame_scores: torch.Tensor) -> torch.Tensor:
        # frame_scores: (B, T) -> (B,) mean of top-(T/k_ratio) snippets
        t = frame_scores.size(1)
        k = max(t // self.config.k_ratio, 1)
        return torch.topk(frame_scores, k, dim=1)[0].mean(dim=1)

    def forward(
        self,
        video: torch.FloatTensor,
        abnormal_labels: Optional[torch.FloatTensor] = None,
        normal_labels: Optional[torch.FloatTensor] = None,
        class_labels: Optional[torch.FloatTensor] = None,
    ) -> GSMoEVideoAnomalyDetectionOutput:
        f = self.config.feature_size
        x = video[..., :f].mean(dim=1)  # (B, T, f) — mean over crops
        h = self.input_proj(x)  # (B, T, hidden)

        expert_out = [e(h)[0] for e in self.experts]  # list of (B, T, 1)
        expert_scores = torch.cat(expert_out, dim=2).permute(0, 2, 1)  # (B, E, T)

        # gate fuses expert scores with the task-aware features (h) via the
        # paper's bi-directional cross-attention.
        scores = self.gate(expert_scores, h)  # (B, T, 1)

        loss = None
        if abnormal_labels is not None and normal_labels is not None:
            loss = self._compute_loss(scores, expert_scores, class_labels)

        return GSMoEVideoAnomalyDetectionOutput(
            loss=loss, scores=scores, expert_scores=expert_scores
        )

    def _compute_loss(self, scores, expert_scores, class_labels):
        bs = scores.size(0)
        half = bs // 2
        device = scores.device
        frame = scores.squeeze(-1)  # (B, T)

        # top-k-norm MIL (binary; normal-first batch)
        y = torch.cat(
            [torch.zeros(half, device=device), torch.ones(half, device=device)]
        )
        vid = self._topk_mil(frame).clamp(1e-6, 1 - 1e-6)
        loss = self.bce(vid, y)

        # smoothness + sparsity (on abnormal scores)
        loss = loss + self.smooth(scores) + self.sparse(frame[half:].reshape(-1))

        # TGS self-training on the abnormal curve (epoch 2+; warm-up skips)
        if self.tgs_enabled:
            loss = loss + self.config.w_tgs * self.tgs(frame[half:])

        # optional class-specialized expert supervision (needs per-video class id)
        if class_labels is not None:
            # class_labels: (B,) int, 0=Normal, 1..13 anomaly classes. Route each
            # abnormal video to its class expert (anomaly class i -> expert i-1).
            n_exp = expert_scores.size(1)
            cls = (class_labels[half:].long().to(device) - 1).clamp(0, n_exp - 1)
            abn_expert = expert_scores[half:]  # (half, E, T)
            tgt = torch.zeros(half, abn_expert.size(1), device=device)
            tgt[torch.arange(half), cls] = 1.0
            expert_vid = torch.stack(
                [self._topk_mil(abn_expert[:, e]) for e in range(abn_expert.size(1))],
                dim=1,
            ).clamp(
                1e-6, 1 - 1e-6
            )  # (half, E)
            loss = loss + self.bce(expert_vid, tgt)

        return loss
