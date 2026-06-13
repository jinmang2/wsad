"""S3R: Self-supervised Sparse Representation for VAD (ECCV'22).

Faithful port of the official ``S3R`` (louisYen/S3R, ``detector.py`` +
``modules/{memory_module,residual_attention}.py``). Module/param names mirror the
official state dict so a weight transfer maps 1:1.

Architecture (official):
  - ``en_normal`` (``enNormal.en_normal_module``): dictionary attention —
    ``softmax(q·kᵀ)·v`` reading a normal-event **macro dictionary** ``(B,S,C)``.
  - ``video_embedding`` / ``macro_embedding`` = ``Sequential(Aggregate, Dropout)``;
    ``Aggregate`` is the RTFM MTN with **GroupNorm(8)** and an ``attn/T`` non-local.
  - ``de_normal`` (``deNormal.channel_attention.channel_gate``): channel attention
    from the video−macro avg-pool residual; excites anomaly channels in video,
    normal channels in macro.
  - ``video_classifier`` (3-FC + sigmoid) → per-snippet score; ``macro_classifier``
    (``GlobalStatistics``: avg-pool over T → 3-FC → sigmoid) → video-level score;
    top-k feature-magnitude MIL (RTFM head).

The official ``macro`` dictionary is precomputed offline (dictionary learning on
normal features). Our ``forward`` accepts it as an optional input (faithful path);
when absent it falls back to a learnable ``dictionary`` parameter so the model also
fits the shared ``forward(video, ...)`` runner contract. ``video_projection`` /
``macro_projection`` exist in the official module (unused by its forward) and are
kept here for state-dict parity.

Verified vs official by weight transfer in ``scripts/verify_s3r.py``.
Ref: https://github.com/louisYen/S3R — https://arxiv.org/abs/2207.10448
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from src.loss import RTFMLoss, SparsityLoss, TemporalSmoothnessLoss
from src.modules.mil import topk_magnitude_select
from src.registry import MODELS

from .configuration_s3r import S3RConfig


@dataclass
class S3RVideoAnomalyDetectionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    scores: Optional[torch.FloatTensor] = None
    macro_scores: Optional[torch.FloatTensor] = None
    memory_attn: Optional[torch.FloatTensor] = None


class _NonLocalBlock1D(nn.Module):
    """S3R non-local block: ``attn = (q·kᵀ)/T`` (no softmax). Keys value/query/key/alter."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.inter_channels = max(in_channels // 2, 1)
        self.value = nn.Conv1d(in_channels, self.inter_channels, 1)
        self.alter = nn.Sequential(
            nn.Conv1d(self.inter_channels, in_channels, 1), nn.BatchNorm1d(in_channels)
        )
        nn.init.constant_(self.alter[1].weight, 0)
        nn.init.constant_(self.alter[1].bias, 0)
        self.query = nn.Conv1d(in_channels, self.inter_channels, 1)
        self.key = nn.Conv1d(in_channels, self.inter_channels, 1)

    def forward(self, x):  # (B, C, T)
        b, _, t = x.shape
        d = self.inter_channels
        value = self.value(x).view(b, d, -1).transpose(-2, -1)  # BTD
        query = self.query(x).view(b, d, -1).transpose(-2, -1)  # BTD
        key = self.key(x).view(b, d, -1)  # BDT
        attn = (query @ key) / t  # BTT
        out = (attn @ value).permute(0, 2, 1).contiguous().view(b, d, t)
        return self.alter(out) + x


class Aggregate(nn.Module):
    """RTFM MTN, GroupNorm(8) variant (official S3R ``Aggregate``)."""

    def __init__(self, dim: int = 2048, reduction: int = 4):
        super().__init__()
        di = dim // reduction

        def gn(c):
            return nn.GroupNorm(num_groups=8, num_channels=c, eps=1e-5)

        self.conv_1 = nn.Sequential(nn.Conv1d(dim, di, 3, dilation=1, padding=1), gn(di), nn.ReLU())
        self.conv_2 = nn.Sequential(nn.Conv1d(dim, di, 3, dilation=2, padding=2), gn(di), nn.ReLU())
        self.conv_3 = nn.Sequential(nn.Conv1d(dim, di, 3, dilation=4, padding=4), gn(di), nn.ReLU())
        self.conv_4 = nn.Sequential(nn.Conv1d(dim, di, 1, bias=False), nn.ReLU())
        self.conv_5 = nn.Sequential(nn.Conv1d(dim, dim, 3, padding=1, bias=False), gn(dim), nn.ReLU())
        self.non_local = _NonLocalBlock1D(di)

    def forward(self, x):  # (B, T, C)
        out = x.transpose(-2, -1)  # BCT
        residual = out
        out_d = torch.cat((self.conv_1(out), self.conv_2(out), self.conv_3(out)), dim=1)
        out = self.conv_4(out)
        out = self.non_local(out)
        out = torch.cat((out_d, out), dim=1)
        out = self.conv_5(out) + residual
        return out.transpose(-2, -1)  # BTC


class enNormalModule(nn.Module):
    def __init__(self, dim: int = 2048, shrink_thres: float = 0.0):
        super().__init__()
        self.shrink_thres = shrink_thres
        self.query_embedding = nn.Linear(dim, dim // 4)
        self.cache_embedding = nn.Linear(dim, dim // 4)
        self.value_embedding = nn.Linear(dim, dim)

    def forward(self, query, cache):
        # query: (BN, T, C); cache: (B, S, C)
        _, t, _ = query.shape
        b = cache.shape[0]
        query = rearrange(query, "(b n) t c -> b (n t) c", b=b)
        x_query = self.query_embedding(query)
        x_cache = self.cache_embedding(cache)
        x_value = self.value_embedding(cache)
        affinity = (x_query @ x_cache.transpose(1, 2)).softmax(dim=-1)  # BLS
        out = affinity @ x_value  # BLC
        out = rearrange(out, "b (n t) c -> (b n) t c", t=t)
        return out, affinity


class enNormal(nn.Module):
    def __init__(self, dim: int = 2048, num_univ: int = 1001, modality: str = "taskaware"):
        super().__init__()
        self.num_univ = num_univ
        self.modality = modality
        self.en_normal_module = enNormalModule(dim, shrink_thres=0.0)

    def forward(self, video, macro):
        out, attn = self.en_normal_module(query=video, cache=macro)
        return out, attn.transpose(1, 2)  # attn -> BST


class ChannelGate(nn.Module):
    def __init__(self, gate_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )

    def forward(self, video, macro):  # (BN, C, T)
        t = video.size(2)
        avg = F.avg_pool1d(video, t, stride=t) - F.avg_pool1d(macro, t, stride=t)
        return torch.sigmoid(self.mlp(avg)).unsqueeze(2)


class ChannelAttention(nn.Module):
    def __init__(self, dim_input: int, reduction: int = 16):
        super().__init__()
        self.channel_gate = ChannelGate(dim_input, reduction)

    def forward(self, video, macro):
        scale = self.channel_gate(video, macro)
        video = video * scale.expand_as(video)
        macro = macro * (1.0 - scale).expand_as(macro)
        return video, macro


class deNormal(nn.Module):
    def __init__(self, dim_input: int = 2048, dim_inner: int = 1024, reduction: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(dim_input, reduction)

    def forward(self, video, macro):  # (BN, T, C)
        video, macro = video.transpose(1, 2), macro.transpose(1, 2)  # BCT
        video, macro = self.channel_attention(video, macro)
        return video.transpose(1, 2), macro.transpose(1, 2)


class GlobalStatistics(nn.Module):
    """Avg-pool over T -> mlp -> sigmoid (official macro head)."""

    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.mlp = mlp

    def forward(self, x):  # (BN, C, T)
        t = x.size(2)
        pooled = F.avg_pool1d(x, t, stride=t).view(x.size(0), -1)
        return torch.sigmoid(self.mlp(pooled))


def _gn_proj(dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim, eps=1e-5), nn.ReLU()
    )


class S3RPreTrainedModel(PreTrainedModel):
    config_class = S3RConfig
    base_model_prefix = "s3r"

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    @property
    def dummy_inputs(self):
        return {"video": torch.randn(2, 10, 32, self.config.feature_size)}


@MODELS.register("s3r")
class S3RForVideoAnomalyDetection(S3RPreTrainedModel):
    def __init__(self, config: S3RConfig):
        super().__init__(config)
        self.k = config.k
        d = config.feature_size

        self.video_embedding = nn.Sequential(Aggregate(d, config.reduction), nn.Dropout(config.dropout_rate))
        self.macro_embedding = nn.Sequential(Aggregate(d, config.reduction), nn.Dropout(config.dropout_rate))
        self.en_normal = enNormal(d, modality=config.modality)
        self.de_normal = deNormal(d, d // 2, reduction=config.denormal_reduction)
        # present in the official module (unused by its forward) — kept for parity
        self.video_projection = _gn_proj(d)
        self.macro_projection = _gn_proj(d)

        self.video_classifier = nn.Sequential(
            nn.Linear(d, d // 4), nn.ReLU(), nn.Dropout(config.dropout_rate),
            nn.Linear(d // 4, d // 16), nn.ReLU(), nn.Dropout(config.dropout_rate),
            nn.Linear(d // 16, 1), nn.Sigmoid(),
        )
        macro_mlp = nn.Sequential(
            nn.Linear(d, d // 4), nn.ReLU(), nn.Dropout(config.dropout_rate),
            nn.Linear(d // 4, d // 16), nn.ReLU(), nn.Dropout(config.dropout_rate),
            nn.Linear(d // 16, 1),
        )
        self.macro_classifier = GlobalStatistics(mlp=macro_mlp)

        # runner-contract fallback dictionary (official passes a precomputed macro)
        self.dictionary = nn.Parameter(torch.randn(config.dict_size, d) * 0.02)
        self.drop_out = nn.Dropout(config.dropout_rate)
        self._force_split = False

    @property
    def force_split(self) -> bool:
        return self._force_split

    @force_split.setter
    def force_split(self, val: bool):
        self._force_split = val

    def forward(
        self,
        video: torch.FloatTensor,
        macro: Optional[torch.FloatTensor] = None,
        abnormal_labels: Optional[torch.FloatTensor] = None,
        normal_labels: Optional[torch.FloatTensor] = None,
    ) -> S3RVideoAnomalyDetectionOutput:
        f = self.config.feature_size
        video = video[..., :f]
        bs, ncrops, t, _ = video.size()
        x = rearrange(video, "b n t c -> (b n) t c")

        cache = macro if macro is not None else self.dictionary.unsqueeze(0).expand(bs, -1, -1)
        macro_stream, memory_attn = self.en_normal(x, cache)

        x_video = self.video_embedding(x)
        x_macro = self.macro_embedding(macro_stream)
        x_video, x_macro = self.de_normal(x_video, x_macro)

        video_scores = self.video_classifier(x_video)  # (BN, T, 1)
        scores = video_scores.view(bs, ncrops, -1).mean(1).unsqueeze(2)  # (B, T, 1)
        macro_scores = self.macro_classifier(x_macro.transpose(1, 2))  # (BN, 1)
        macro_scores = macro_scores.view(-1, ncrops, 1).mean(1)  # (B, 1)

        feat_mag = torch.norm(x_video, p=2, dim=2).view(bs, ncrops, -1).mean(1)  # (B, T)

        if self.force_split or self.training:
            half = bs // 2
            n_feat, a_feat = x_video[: half * ncrops], x_video[half * ncrops :]
            n_score, a_score = scores[:half], scores[half:]
            n_mag, a_mag = feat_mag[:half], feat_mag[half:]
        else:
            n_feat = a_feat = x_video
            n_score = a_score = scores
            n_mag = a_mag = feat_mag

        a_sel, score_abn = topk_magnitude_select(a_mag, a_feat, a_score, self.k, ncrops, t, f, self.drop_out)
        n_sel, score_nor = topk_magnitude_select(n_mag, n_feat, n_score, self.k, ncrops, t, f, self.drop_out)

        loss = None
        if abnormal_labels is not None and normal_labels is not None:
            half = bs // 2
            loss_mag = RTFMLoss(alpha=self.config.alpha, margin=self.config.margin)(
                normal_scores=score_nor, abnormal_scores=score_abn,
                normal_labels=normal_labels, abnormal_labels=abnormal_labels,
                nor_feamagnitude=n_sel, abn_feamagnitude=a_sel,
            )
            labels = torch.cat([normal_labels, abnormal_labels], dim=0)
            loss_macro = F.binary_cross_entropy(
                macro_scores.squeeze(-1).clamp(1e-6, 1 - 1e-6), labels
            )
            loss_smooth = TemporalSmoothnessLoss(self.config.lambda_smooth)(scores)
            loss_sparse = SparsityLoss(self.config.lambda_sparse)(scores[half:].reshape(-1))
            loss = loss_mag + self.config.w_macro * loss_macro + loss_smooth + loss_sparse

        return S3RVideoAnomalyDetectionOutput(
            loss=loss, scores=scores, macro_scores=macro_scores, memory_attn=memory_attn
        )
