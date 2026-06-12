"""S3R: Self-supervised Sparse Representation for VAD (ECCV'22).

Slot mapping (WSAD_INTEGRATION_PLAN.md section 4):
  - slot 2 (temporal encoder): ``Aggregate`` (pyramid dilated conv + non-local;
    the RTFM/MTN lineage, GroupNorm variant) on two streams.
  - slot 4 (scoring head): ``enNormal`` reads a normal-event **dictionary** to
    reconstruct a normal-pattern stream; ``deNormal`` channel-filters it out of
    the video stream; a video classifier + macro classifier + top-k magnitude MIL.
  - slot 5 (loss): RTFM-style magnitude separation + video/macro BCE + smooth + sparse.

Honors the runner contract: ``forward(video, abnormal_labels, normal_labels) ->
ModelOutput`` with ``.loss`` and ``.scores`` (B, T, 1).

Cross-checked vs official louisYen/S3R (detector.py, memory_module.py,
residual_attention.py). **Adaptation:** the official dictionary is precomputed
offline by dictionary learning on normal features and passed in as ``macro``;
here it is a **learnable parameter** trained end-to-end, so the model fits the
shared ``forward(video, ...)`` contract. Non-local uses S3R's ``attn/T`` scaling
(no softmax), faithfully.
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


class NonLocalBlock1D(nn.Module):
    """S3R non-local block: attention scaled by ``1/T`` (no softmax)."""

    def __init__(self, dim: int):
        super().__init__()
        self.inter = max(dim // 2, 1)
        self.value = nn.Conv1d(dim, self.inter, 1)
        self.query = nn.Conv1d(dim, self.inter, 1)
        self.key = nn.Conv1d(dim, self.inter, 1)
        self.alter = nn.Sequential(nn.Conv1d(self.inter, dim, 1), nn.BatchNorm1d(dim))
        nn.init.constant_(self.alter[1].weight, 0)
        nn.init.constant_(self.alter[1].bias, 0)

    def forward(self, x):  # x: (B, C, T)
        b, _, t = x.shape
        v = self.value(x).view(b, self.inter, -1).transpose(-2, -1)  # BTD
        q = self.query(x).view(b, self.inter, -1).transpose(-2, -1)  # BTD
        k = self.key(x).view(b, self.inter, -1)  # BDT
        attn = (q @ k) / t  # BTT
        out = (attn @ v).permute(0, 2, 1).contiguous().view(b, self.inter, t)
        return x + self.alter(out)


class Aggregate(nn.Module):
    """MTN-style multi-scale temporal aggregator (GroupNorm variant, S3R)."""

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        di = dim // reduction

        def gn(c):
            return nn.GroupNorm(8, c, eps=1e-5)

        self.conv_1 = nn.Sequential(
            nn.Conv1d(dim, di, 3, padding=1, dilation=1), gn(di), nn.ReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(dim, di, 3, padding=2, dilation=2), gn(di), nn.ReLU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(dim, di, 3, padding=4, dilation=4), gn(di), nn.ReLU()
        )
        self.conv_4 = nn.Sequential(nn.Conv1d(dim, di, 1, bias=False), nn.ReLU())
        self.conv_5 = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1, bias=False), gn(dim), nn.ReLU()
        )
        self.non_local = NonLocalBlock1D(di)

    def forward(self, x):  # x: (B, T, C)
        out = x.transpose(-2, -1)  # BCT
        residual = out
        out_d = torch.cat((self.conv_1(out), self.conv_2(out), self.conv_3(out)), dim=1)
        out_g = self.non_local(self.conv_4(out))
        out = self.conv_5(torch.cat((out_d, out_g), dim=1)) + residual
        return out.transpose(-2, -1)  # BTC


class EnNormal(nn.Module):
    """Read a normal-event dictionary by attention (sparse reconstruction)."""

    def __init__(self, dim: int):
        super().__init__()
        self.query_embedding = nn.Linear(dim, dim // 4)
        self.cache_embedding = nn.Linear(dim, dim // 4)
        self.value_embedding = nn.Linear(dim, dim)

    def forward(self, query, cache):
        # query: (BN, T, C); cache (dictionary): (B, S, C)
        _, t, _ = query.shape
        b = cache.shape[0]
        query = rearrange(query, "(b n) t c -> b (n t) c", b=b)
        aq = self.query_embedding(query)  # B(NT)D
        ak = self.cache_embedding(cache)  # BSD
        av = self.value_embedding(cache)  # BSC
        affinity = (aq @ ak.transpose(1, 2)).softmax(dim=-1)  # B(NT)S
        out = affinity @ av  # B(NT)C
        out = rearrange(out, "b (n t) c -> (b n) t c", t=t)
        return out, affinity.transpose(1, 2)


class ChannelGate(nn.Module):
    """deNormal channel gate from the video-vs-macro residual."""

    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim, dim // reduction),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim),
        )

    def forward(self, video, macro):  # (BN, C, T)
        t = video.size(2)
        residual = F.avg_pool1d(video, t) - F.avg_pool1d(macro, t)
        return torch.sigmoid(self.mlp(residual)).unsqueeze(2)


class DeNormal(nn.Module):
    """Excite anomaly-different channels in video; normal-same in macro."""

    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        self.gate = ChannelGate(dim, reduction)

    def forward(self, video, macro):  # (BN, T, C)
        v, m = video.transpose(1, 2), macro.transpose(1, 2)  # BCT
        scale = self.gate(v, m)
        v = v * scale.expand_as(v)
        m = m * (1.0 - scale).expand_as(m)
        return v.transpose(1, 2), m.transpose(1, 2)


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
        return {"video": torch.randn(32, 10, 32, 2049)}


@MODELS.register("s3r")
class S3RForVideoAnomalyDetection(S3RPreTrainedModel):
    def __init__(self, config: S3RConfig):
        super().__init__(config)
        self.k = config.k
        d = config.feature_size

        # learnable normal-event dictionary (official: precomputed offline)
        self.dictionary = nn.Parameter(torch.randn(config.dict_size, d) * 0.02)
        self.en_normal = EnNormal(d)
        self.de_normal = DeNormal(d, config.denormal_reduction)

        self.video_embedding = nn.Sequential(
            Aggregate(d, config.reduction), nn.Dropout(config.dropout_rate)
        )
        self.macro_embedding = nn.Sequential(
            Aggregate(d, config.reduction), nn.Dropout(config.dropout_rate)
        )

        def head(out_sigmoid: bool):
            layers = [
                nn.Linear(d, d // 4),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(d // 4, d // 16),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(d // 16, 1),
            ]
            if out_sigmoid:
                layers.append(nn.Sigmoid())
            return nn.Sequential(*layers)

        self.video_classifier = head(out_sigmoid=True)
        self.macro_mlp = head(out_sigmoid=False)
        self.dropout = nn.Dropout(config.dropout_rate)
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
        abnormal_labels: Optional[torch.FloatTensor] = None,
        normal_labels: Optional[torch.FloatTensor] = None,
    ) -> S3RVideoAnomalyDetectionOutput:
        f = self.config.feature_size
        video = video[..., :f]
        bs, ncrops, t, _ = video.size()
        x = rearrange(video, "b n t c -> (b n) t c")

        cache = self.dictionary.unsqueeze(0).expand(bs, -1, -1)  # (B, S, C)
        macro, memory_attn = self.en_normal(x, cache)

        xv = self.video_embedding(x)
        xm = self.macro_embedding(macro)
        xv, xm = self.de_normal(xv, xm)

        scores = self.video_classifier(xv)  # (BN, T, 1)
        scores = scores.view(bs, ncrops, -1).mean(dim=1).unsqueeze(2)  # (B, T, 1)
        macro_scores = torch.sigmoid(
            self.macro_mlp(xm.mean(dim=1)).view(bs, ncrops, 1).mean(dim=1)
        )  # (B, 1)

        feat_mag = torch.norm(xv, p=2, dim=2).view(bs, ncrops, -1).mean(dim=1)  # (B, T)

        if self.force_split or self.training:
            half = bs // 2
            n_feat, a_feat = xv[: half * ncrops], xv[half * ncrops :]
            n_score, a_score = scores[:half], scores[half:]
            n_mag, a_mag = feat_mag[:half], feat_mag[half:]
        else:
            n_feat = a_feat = xv
            n_score = a_score = scores
            n_mag = a_mag = feat_mag

        a_sel, score_abn = topk_magnitude_select(
            a_mag, a_feat, a_score, self.k, ncrops, t, f, self.dropout
        )
        n_sel, score_nor = topk_magnitude_select(
            n_mag, n_feat, n_score, self.k, ncrops, t, f, self.dropout
        )

        loss = None
        if abnormal_labels is not None and normal_labels is not None:
            half = bs // 2
            loss_mag = RTFMLoss(alpha=self.config.alpha, margin=self.config.margin)(
                normal_scores=score_nor,
                abnormal_scores=score_abn,
                normal_labels=normal_labels,
                abnormal_labels=abnormal_labels,
                nor_feamagnitude=n_sel,
                abn_feamagnitude=a_sel,
            )
            labels = torch.cat([normal_labels, abnormal_labels], dim=0)
            loss_macro = F.binary_cross_entropy(
                macro_scores.squeeze(-1).clamp(1e-6, 1 - 1e-6), labels
            )
            loss_smooth = TemporalSmoothnessLoss(self.config.lambda_smooth)(scores)
            loss_sparse = SparsityLoss(self.config.lambda_sparse)(
                scores[half:].reshape(-1)
            )
            loss = (
                loss_mag + self.config.w_macro * loss_macro + loss_smooth + loss_sparse
            )

        return S3RVideoAnomalyDetectionOutput(
            loss=loss, scores=scores, macro_scores=macro_scores, memory_attn=memory_attn
        )
