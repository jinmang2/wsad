"""UR-DMU: Dual Memory Units with Uncertainty Regulation (AAAI'23).

Faithful port of the official ``WSAD`` (henrryzh1/UR-DMU, ``model.py`` +
``translayer.py`` + ``memory.py``) into this repo's HF style. Module/param names
mirror the official state dict so ``models/ucf_trans_2022.pkl`` loads directly.

Architecture (official):
  - ``embedding`` (``Temporal``): ``Conv1d(1024->512, k=3) + ReLU``.
  - ``selfatt`` (``Transformer``): pre-norm blocks whose ``Attention`` is a
    **dual branch** — ``to_qkv`` projects to (q,k,v,t); branch 1 is standard
    ``softmax(qkᵀ)·v``, branch 2 is a **fixed** temporal distance-decay
    ``exp(-|i-j|/e)`` attention over ``t``; the two are concatenated and fused.
  - ``Amemory`` / ``Nmemory`` (``Memory_Unit``): attention read over a learnable
    memory bank; top-(nums//16+1) attention = temporal activation.
  - ``encoder_mu`` / ``encoder_var``: variational uncertainty latent (reparam at
    train; ``mu`` at eval).
  - ``cls_head`` (``ADCLS_head``): ``Linear(1024,128)->ReLU->Linear(128,1)->Sigmoid``
    over ``cat([selfatt_feature, augment])``.

Input is official 1024-d I3D RGB (10-crop, no magnitude). ``.scores`` = per-frame
anomaly probability ``(B, T, 1)``.

Verified vs the official checkpoint in ``scripts/verify_ur_dmu.py``.
Ref: https://github.com/henrryzh1/UR-DMU — https://arxiv.org/abs/2302.05160
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from src.modules.translayer import Transformer
from src.registry import MODELS

from .configuration_ur_dmu import URDMUConfig


@dataclass
class URDMUVideoAnomalyDetectionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    scores: Optional[torch.FloatTensor] = None
    a_attention: Optional[torch.FloatTensor] = None
    n_attention: Optional[torch.FloatTensor] = None


def _norm(data: torch.Tensor) -> torch.Tensor:  # official utils.norm
    return data / (torch.norm(data, p=2, dim=-1, keepdim=True) + 1e-10)


class Temporal(nn.Module):
    """Official ``Temporal`` embedding: Conv1d(k=3) + ReLU."""

    def __init__(self, input_size: int, out_size: int):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(input_size, out_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        return x.permute(0, 2, 1)


class Memory_Unit(nn.Module):
    """Official ``Memory_Unit``: attention read over a learnable bank."""

    def __init__(self, nums: int, dim: int):
        super().__init__()
        self.dim = dim
        self.nums = nums
        self.memory_block = nn.Parameter(torch.empty(nums, dim))
        self.sig = nn.Sigmoid()
        stdv = 1.0 / math.sqrt(dim)
        self.memory_block.data.uniform_(-stdv, stdv)

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attention = self.sig(
            torch.einsum("btd,kd->btk", data, self.memory_block) / (self.dim**0.5)
        )
        temporal_att = torch.topk(attention, self.nums // 16 + 1, dim=-1)[0].mean(-1)
        augment = torch.einsum("btk,kd->btd", attention, self.memory_block)
        return temporal_att, augment


class ADCLS_head(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, out_dim), nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(x)


class URDMUPreTrainedModel(PreTrainedModel):
    config_class = URDMUConfig
    base_model_prefix = "ur_dmu"

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if getattr(module, "weight", None) is not None:
                nn.init.xavier_uniform_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias, 0)

    @property
    def dummy_inputs(self):
        return {"video": torch.randn(2, 10, 32, self.config.feature_size)}


@MODELS.register("ur_dmu")
class URDMUForVideoAnomalyDetection(URDMUPreTrainedModel):
    def __init__(self, config: URDMUConfig):
        super().__init__(config)
        h = config.hidden_size
        self.a_nums = config.mem_size
        self.n_nums = config.mem_size

        self.embedding = Temporal(config.feature_size, h)
        self.triplet = nn.TripletMarginLoss(margin=config.margin)
        self.cls_head = ADCLS_head(h * 2, 1)
        self.Amemory = Memory_Unit(nums=config.mem_size, dim=h)
        self.Nmemory = Memory_Unit(nums=config.mem_size, dim=h)
        self.selfatt = Transformer(h, config.num_layers, config.num_heads, 128, h, dropout=config.dropout_rate)
        self.encoder_mu = nn.Sequential(nn.Linear(h, h))
        self.encoder_var = nn.Sequential(nn.Linear(h, h))
        self.relu = nn.ReLU()
        self._force_split = False

    @property
    def force_split(self) -> bool:
        return self._force_split

    @force_split.setter
    def force_split(self, val: bool):
        self._force_split = val

    @staticmethod
    def _reparameterize(mu, logvar):
        std = torch.exp(logvar).sqrt()
        return mu + torch.randn_like(std) * std

    @staticmethod
    def _latent_loss(mu, var):
        return torch.mean(-0.5 * torch.sum(1 + var - mu**2 - var.exp(), dim=1))

    def forward(
        self,
        video: torch.FloatTensor,
        abnormal_labels: Optional[torch.FloatTensor] = None,
        normal_labels: Optional[torch.FloatTensor] = None,
    ) -> URDMUVideoAnomalyDetectionOutput:
        x = video[..., : self.config.feature_size]
        if x.dim() == 4:
            b, n, t, d = x.size()
            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1

        x = self.embedding(x)
        x = self.selfatt(x)

        train = abnormal_labels is not None and normal_labels is not None
        if train:
            out = self._forward_train(x, b, n, t)
            scores = out["frame"].unsqueeze(-1)
            loss = self._compute_loss(out, abnormal_labels, normal_labels)
            return URDMUVideoAnomalyDetectionOutput(
                loss=loss, scores=scores, a_attention=out["A_att"], n_attention=out["N_att"]
            )

        # eval (official else-branch): both memories read full x, mu only
        _, A_aug = self.Amemory(x)
        _, N_aug = self.Nmemory(x)
        A_aug = self.encoder_mu(A_aug)
        N_aug = self.encoder_mu(N_aug)
        feat = torch.cat([x, A_aug + N_aug], dim=-1)
        pre_att = self.cls_head(feat).reshape((b, n, -1)).mean(1)  # (b, T)
        return URDMUVideoAnomalyDetectionOutput(loss=None, scores=pre_att.unsqueeze(-1))

    def _forward_train(self, x, b, n, t):
        N_x = x[: b * n // 2]
        A_x = x[b * n // 2 :]
        A_att, A_aug = self.Amemory(A_x)
        N_Aatt, N_Aaug = self.Nmemory(A_x)
        A_Natt, A_Naug = self.Amemory(N_x)
        N_att, N_aug = self.Nmemory(N_x)
        d = x.size(-1)

        def sel(src, att):
            _, idx = torch.topk(att, t // 16 + 1, dim=-1)
            return torch.gather(src, 1, idx.unsqueeze(2).expand([-1, -1, d])).mean(1).reshape(b // 2, n, -1).mean(1), idx

        negative_ax, A_index = sel(A_x, A_att)
        anchor_nx, N_index = sel(N_x, N_att)
        positive_nx, _ = sel(A_x, N_Aatt)
        triplet_margin = self.triplet(_norm(anchor_nx), _norm(positive_nx), _norm(negative_ax))

        N_aug_mu = self.encoder_mu(N_aug)
        N_aug_var = self.encoder_var(N_aug)
        N_aug_new = self._reparameterize(N_aug_mu, N_aug_var)
        anchor_nx_new = torch.gather(N_aug_new, 1, N_index.unsqueeze(2).expand([-1, -1, d])).mean(1).reshape(b // 2, n, -1).mean(1)
        A_aug_new = self.encoder_mu(A_aug)
        negative_ax_new = torch.gather(A_aug_new, 1, A_index.unsqueeze(2).expand([-1, -1, d])).mean(1).reshape(b // 2, n, -1).mean(1)
        kl_loss = self._latent_loss(N_aug_mu, N_aug_var)

        A_Naug = self.encoder_mu(A_Naug)
        N_Aaug = self.encoder_mu(N_Aaug)
        distance = torch.relu(100 - torch.norm(negative_ax_new, p=2, dim=-1) + torch.norm(anchor_nx_new, p=2, dim=-1)).mean()
        feat = torch.cat((x, torch.cat([N_aug_new + A_Naug, A_aug_new + N_Aaug], dim=0)), dim=-1)
        pre_att = self.cls_head(feat).reshape((b, n, -1)).mean(1)
        return {
            "frame": pre_att,
            "triplet_margin": triplet_margin,
            "kl_loss": kl_loss,
            "distance": distance,
            "A_att": A_att.reshape((b // 2, n, -1)).mean(1),
            "N_att": N_att.reshape((b // 2, n, -1)).mean(1),
            "A_Natt": A_Natt.reshape((b // 2, n, -1)).mean(1),
            "N_Aatt": N_Aatt.reshape((b // 2, n, -1)).mean(1),
        }

    def _compute_loss(self, out, abnormal_labels, normal_labels):
        # official UCF loss: BCE(frame top-k) over normal/abnormal + memory MIL +
        # triplet + KL + distance (weights from official ucf_main/config).
        import torch.nn.functional as F

        from src.modules.amp import safe_bce  # autocast-safe BCE (fp16 training)

        frame = out["frame"]
        t = frame.size(1)
        k = t // 16 + 1
        device = frame.device
        half = frame.size(0) // 2
        y = torch.cat([torch.zeros(half, device=device), torch.ones(half, device=device)])
        vid = torch.topk(frame, k, dim=1)[0].mean(1)
        loss_mil = safe_bce(vid.clamp(1e-6, 1 - 1e-6), y)

        a_score = torch.topk(out["A_att"], k, dim=1)[0].mean(1)
        n_score = torch.topk(out["N_att"], k, dim=1)[0].mean(1)
        an_score = torch.topk(out["A_Natt"], k, dim=1)[0].mean(1)
        na_score = torch.topk(out["N_Aatt"], k, dim=1)[0].mean(1)
        loss_mem = (
            safe_bce(a_score.clamp(1e-6, 1 - 1e-6), torch.ones_like(a_score))
            + safe_bce(n_score.clamp(1e-6, 1 - 1e-6), torch.ones_like(n_score))
            + safe_bce(an_score.clamp(1e-6, 1 - 1e-6), torch.zeros_like(an_score))
            + safe_bce(na_score.clamp(1e-6, 1 - 1e-6), torch.zeros_like(na_score))
        )
        return (
            loss_mil
            + self.config.w_mem * loss_mem
            + self.config.w_triplet * out["triplet_margin"]
            + self.config.w_kl * out["kl_loss"]
            + self.config.w_distance * out["distance"]
        )
