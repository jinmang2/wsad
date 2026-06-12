"""UR-DMU: Dual Memory Units with Uncertainty Regulation (AAAI'23).

Slot mapping (WSAD_INTEGRATION_PLAN.md section 4):
  - slot 2 (temporal encoder): distance-biased Transformer self-attention
    (shared ``src.modules`` MHSA + a learnable Gaussian distance prior).
  - slot 4 (scoring head): dual ``Memory_Unit`` banks (normal / abnormal) read by
    attention, a variational uncertainty latent, then an MLP classifier on the
    concatenation of the encoded + uncertainty features. Video score = top-k mean
    of frame scores.
  - slot 5 (loss): MIL BCE + memory-activation MIL (normal/abnormal banks) +
    triplet separation + KL uncertainty regulation.

Honors the shared runner contract:
``forward(video, abnormal_labels=None, normal_labels=None) -> ModelOutput`` with
``.loss`` (None at inference) and ``.scores`` of shape ``(B, T, 1)``.

Reference: https://arxiv.org/abs/2302.05160

Repro note: crops are mean-pooled up front (memory/attention operate per video);
the official repo handles 10-crop and averages at score level. Exact memory size
(60), top-k ratio (T/16) and loss weights follow the official config — verify the
precise UCF AUC (~87.0) against the official hyperparameters when training.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from src.modules import TransformerEncoderLayer
from src.registry import MODELS

from .configuration_ur_dmu import URDMUConfig


@dataclass
class URDMUVideoAnomalyDetectionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    scores: Optional[torch.FloatTensor] = None
    a_attention: Optional[torch.FloatTensor] = None
    n_attention: Optional[torch.FloatTensor] = None


class MemoryUnit(nn.Module):
    """Attention read over a learnable memory bank (UR-DMU Memory_Unit)."""

    def __init__(self, mem_size: int, dim: int):
        super().__init__()
        self.dim = dim
        self.mem_size = mem_size
        self.memory_block = nn.Parameter(torch.empty(mem_size, dim))
        self.sig = nn.Sigmoid()
        stdv = 1.0 / (dim**0.5)
        self.memory_block.data.uniform_(-stdv, stdv)

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # data: (B, T, D)
        attention = self.sig(
            torch.einsum("btd,kd->btk", data, self.memory_block) / (self.dim**0.5)
        )  # (B, T, mem_size)
        topk = max(self.mem_size // 16, 1)
        temporal_att = torch.topk(attention, topk, dim=-1)[0].mean(-1)  # (B, T)
        augment = torch.einsum("btk,kd->btd", attention, self.memory_block)  # (B, T, D)
        return temporal_att, augment


class DistanceAdj(nn.Module):
    """Learnable Gaussian distance prior as an additive attention bias.

    Returns ``-softplus(sigma) * |i-j|^2`` shaped ``(1, 1, T, T)``; after softmax
    this multiplies the attention by a Gaussian of temporal distance (the UR-DMU
    distance-adjacency, expressed in additive-bias form so it composes with both
    the eager and SDPA attention kernels).
    """

    def __init__(self):
        super().__init__()
        self.sigma = nn.Parameter(torch.full((1,), 0.1))

    def forward(self, t: int, device) -> torch.Tensor:
        arith = torch.arange(t, device=device).float()
        dist = (arith[:, None] - arith[None, :]).abs() ** 2  # (T, T)
        bias = -F.softplus(self.sigma) * dist
        return bias.view(1, 1, t, t)


class URDMUPreTrainedModel(PreTrainedModel):
    config_class = URDMUConfig
    base_model_prefix = "ur_dmu"

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
        return {"video": torch.randn(32, 10, 32, 2049)}


@MODELS.register("ur_dmu")
class URDMUForVideoAnomalyDetection(URDMUPreTrainedModel):
    def __init__(self, config: URDMUConfig):
        super().__init__(config)
        f, h = config.feature_size, config.hidden_size

        self.embedding = nn.Sequential(nn.Linear(f, h), nn.ReLU())
        self.encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    dim=h,
                    heads=config.num_heads,
                    dropout=config.dropout_rate,
                    attn_impl=config.attn_impl,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.distance_adj = DistanceAdj()

        self.a_memory = MemoryUnit(config.mem_size, h)
        self.n_memory = MemoryUnit(config.mem_size, h)

        self.encoder_mu = nn.Linear(h, h)
        self.encoder_var = nn.Linear(h, h)

        self.cls_head = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(h, 1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU()
        self.bce = nn.BCELoss()
        self.triplet = nn.TripletMarginLoss(margin=config.margin)
        self._force_split = False

    @property
    def force_split(self) -> bool:
        return self._force_split

    @force_split.setter
    def force_split(self, val: bool):
        self._force_split = val

    @staticmethod
    def _reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def _topk_mean(self, frame_vals: torch.Tensor) -> torch.Tensor:
        # frame_vals: (B, T) -> (B,) mean of top-(T/ratio) values
        t = frame_vals.size(1)
        k = max(t // self.config.topk_ratio, 1)
        return torch.topk(frame_vals, k, dim=1)[0].mean(dim=1)

    def forward(
        self,
        video: torch.FloatTensor,
        abnormal_labels: Optional[torch.FloatTensor] = None,
        normal_labels: Optional[torch.FloatTensor] = None,
    ) -> URDMUVideoAnomalyDetectionOutput:
        f = self.config.feature_size
        x = video[..., :f].mean(dim=1)  # (B, T, f) — mean over crops
        bs, t, _ = x.size()

        h = self.embedding(x)  # (B, T, hidden)
        bias = self.distance_adj(t, h.device)
        for layer in self.encoder:
            h = layer(h, attn_bias=bias)

        a_att, a_aug = self.a_memory(h)  # (B,T), (B,T,hidden)
        n_att, n_aug = self.n_memory(h)
        aug = self.relu(h + a_aug + n_aug)

        mu = self.encoder_mu(aug)
        logvar = self.encoder_var(aug)
        z = self._reparameterize(mu, logvar) if self.training else mu

        feat = torch.cat([h, z], dim=-1)  # (B, T, 2*hidden)
        scores = self.cls_head(feat)  # (B, T, 1)

        loss = None
        if abnormal_labels is not None and normal_labels is not None:
            loss = self._compute_loss(scores, a_att, n_att, aug, mu, logvar, bs)

        return URDMUVideoAnomalyDetectionOutput(
            loss=loss, scores=scores, a_attention=a_att, n_attention=n_att
        )

    def _compute_loss(self, scores, a_att, n_att, aug, mu, logvar, bs):
        half = bs // 2
        frame = scores.squeeze(-1)  # (B, T)
        y = torch.cat(
            [
                torch.zeros(half, device=scores.device),
                torch.ones(half, device=scores.device),
            ]
        )

        # MIL BCE on top-k mean video score
        vid_score = self._topk_mean(frame)
        loss_mil = self.bce(vid_score.clamp(1e-6, 1 - 1e-6), y)

        # memory-activation MIL: abnormal bank fires on abnormal, normal on normal
        a_score = self._topk_mean(a_att)
        n_score = self._topk_mean(n_att)
        loss_mem = self.bce(a_score.clamp(1e-6, 1 - 1e-6), y) + self.bce(
            n_score.clamp(1e-6, 1 - 1e-6), 1.0 - y
        )

        # triplet: pull normal-aug together, push from abnormal-aug
        pooled = aug.mean(dim=1)  # (B, hidden)
        nor, abn = pooled[:half], pooled[half:]
        loss_triplet = pooled.new_zeros(())
        if half >= 2:
            anchor = nor[:-1]
            positive = nor[1:]
            negative = abn[: anchor.size(0)]
            loss_triplet = self.triplet(anchor, positive, negative)

        # KL uncertainty regulation
        loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return (
            loss_mil
            + self.config.w_mem * loss_mem
            + self.config.w_triplet * loss_triplet
            + self.config.w_kl * loss_kl
        )
