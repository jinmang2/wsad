"""CLIP-TSA: CLIP-Assisted Temporal Self-Attention for WSVAD (ICIP'23).

Faithful port of the official ``Model`` (joos2010kj/CLIP-TSA, ``model.py`` +
``utils/hard_attention.py``). CLIP-TSA = CLIP frame features dropped into the
**RTFM** backbone (``Aggregate``: multi-scale dilated Conv1d 1/2/4 + Non-Local +
fusion + residual; top-k feature-magnitude MIL) **plus** the defining **TSA =
Perturbed Top-K HardAttention** (differentiable snippet selection) applied before
the aggregate.

Slot mapping (WSAD_INTEGRATION_PLAN.md section 4):
  - slot 1 (features): CLIP ViT-B/16 frame features (512-d).
  - slot 2 (temporal): RTFM ``Aggregate`` (MTN) over snippets.
  - slot 3 (TSA): Perturbed Top-K ``HardAttention`` snippet selector.
  - slot 4 (head): RTFM-style top-k feature-magnitude MIL.
  - slot 5 (loss): RTFM loss + temporal smoothness + sparsity.

Earlier this file approximated "TSA" with a plain Transformer and dropped the
RTFM Aggregate — both the paper's defining pieces. This restores them. The
official used 10-crop CLIP features; the architecture is crop-agnostic so it also
runs on single-crop (ncrops=1) caches.

Reference: https://arxiv.org/abs/2212.05136 — https://github.com/joos2010kj/CLIP-TSA
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from src.loss import RTFMLoss, SparsityLoss, TemporalSmoothnessLoss
from src.modules.mil import topk_magnitude_select
from src.registry import MODELS

from .configuration_clip_tsa import CLIPTSAConfig
from .perturbed_topk import HardAttention


@dataclass
class CLIPTSAVideoAnomalyDetectionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    abnormal_scores: Optional[torch.FloatTensor] = None
    normal_scores: Optional[torch.FloatTensor] = None
    abn_feamagnitude: Optional[torch.FloatTensor] = None
    nor_feamagnitude: Optional[torch.FloatTensor] = None
    scores: Optional[torch.FloatTensor] = None


# ---- RTFM backbone (faithful port of official Aggregate + Non-Local) ----
class _NonLocalBlock1D(nn.Module):
    def __init__(self, in_channels, sub_sample=False, bn_layer=True):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = max(in_channels // 2, 1)
        self.g = nn.Conv1d(in_channels, self.inter_channels, 1)
        self.theta = nn.Conv1d(in_channels, self.inter_channels, 1)
        self.phi = nn.Conv1d(in_channels, self.inter_channels, 1)
        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv1d(self.inter_channels, in_channels, 1),
                nn.BatchNorm1d(in_channels),
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv1d(self.inter_channels, in_channels, 1)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)
        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool1d(2))
            self.phi = nn.Sequential(self.phi, nn.MaxPool1d(2))

    def forward(self, x):  # x: (b, c, t)
        b = x.size(0)
        g_x = self.g(x).view(b, self.inter_channels, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(b, self.inter_channels, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(b, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = f / f.size(-1)
        y = torch.matmul(f_div_C, g_x).permute(0, 2, 1).contiguous()
        y = y.view(b, self.inter_channels, *x.size()[2:])
        return self.W(y) + x


class Aggregate(nn.Module):
    """RTFM multi-scale temporal network with division-scaled channels (official)."""

    def __init__(self, len_feature: int):
        super().__init__()
        self.len_feature = len_feature
        self.division = max(2048 // len_feature, 1)
        c = 512 // self.division
        wide = 2048 // self.division
        self.conv_1 = nn.Sequential(
            nn.Conv1d(len_feature, c, 3, dilation=1, padding=1), nn.ReLU(), nn.BatchNorm1d(c)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(len_feature, c, 3, dilation=2, padding=2), nn.ReLU(), nn.BatchNorm1d(c)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(len_feature, c, 3, dilation=4, padding=4), nn.ReLU(), nn.BatchNorm1d(c)
        )
        self.conv_4 = nn.Sequential(nn.Conv1d(wide, c, 1, bias=False), nn.ReLU())
        self.conv_5 = nn.Sequential(
            nn.Conv1d(wide, wide, 3, padding=1, bias=False), nn.ReLU(), nn.BatchNorm1d(wide)
        )
        self.non_local = _NonLocalBlock1D(c, sub_sample=False, bn_layer=True)

    def forward(self, x):  # x: (b, t, len_feature)
        out = x.permute(0, 2, 1)
        residual = out
        out1, out2, out3 = self.conv_1(out), self.conv_2(out), self.conv_3(out)
        out_d = torch.cat((out1, out2, out3), dim=1)
        out = self.conv_4(out)
        out = self.non_local(out)
        out = torch.cat((out_d, out), dim=1)
        out = self.conv_5(out)
        out = out + residual
        return out.permute(0, 2, 1)  # (b, t, len_feature)


class CLIPTSAPreTrainedModel(PreTrainedModel):
    config_class = CLIPTSAConfig
    base_model_prefix = "clip_tsa"

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    @property
    def dummy_inputs(self):
        return {"video": torch.randn(2, 1, self.config.num_segments, 512)}


@MODELS.register("clip_tsa")
class CLIPTSAForVideoAnomalyDetection(CLIPTSAPreTrainedModel):
    def __init__(self, config: CLIPTSAConfig):
        super().__init__(config)
        self.k = config.k
        f = config.feature_size
        div = max(2048 // f, 1)

        self.apply_ha = getattr(config, "apply_ha", True)
        self.hard_attention = HardAttention(
            k=getattr(config, "topk_ratio", 0.7),
            num_samples=getattr(config, "num_samples", 100),
            input_dim=f,
        )
        self.aggregate = Aggregate(len_feature=f)
        self.fc1 = nn.Linear(f, 512 // div)
        self.fc2 = nn.Linear(512 // div, 128 // div)
        self.fc3 = nn.Linear(128 // div, 1)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self._force_split = False
        self.post_init()

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
    ) -> CLIPTSAVideoAnomalyDetectionOutput:
        f = self.config.feature_size
        video = video[..., :f]
        bs, ncrops, t, _ = video.size()

        x = video.reshape(bs * ncrops, t, f)
        if self.apply_ha:
            x = self.hard_attention(x)  # TSA: differentiable top-k snippet selection
        features = self.dropout(self.aggregate(x))  # RTFM Aggregate

        scores = self.relu(self.fc1(features))
        scores = self.dropout(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.dropout(scores)
        scores = self.sigmoid(self.fc3(scores))  # (bs*ncrops, T, 1)
        scores = scores.view(bs, ncrops, -1).mean(dim=1).unsqueeze(2)  # (bs, T, 1)

        feat_mag = torch.norm(features, p=2, dim=2).view(bs, ncrops, -1).mean(dim=1)

        if self.force_split or self.training:
            half = bs // 2
            normal_features = features[: half * ncrops]
            abnormal_features = features[half * ncrops :]
            normal_scores, abnormal_scores = scores[:half], scores[half:]
            nor_mag, abn_mag = feat_mag[:half], feat_mag[half:]
        else:
            normal_features = abnormal_features = features
            normal_scores = abnormal_scores = scores
            nor_mag = abn_mag = feat_mag

        abn_feamagnitude, score_abnormal = topk_magnitude_select(
            abn_mag, abnormal_features, abnormal_scores, self.k, ncrops, t, f, self.dropout
        )
        nor_feamagnitude, score_normal = topk_magnitude_select(
            nor_mag, normal_features, normal_scores, self.k, ncrops, t, f, self.dropout
        )

        loss = None
        if abnormal_labels is not None and normal_labels is not None:
            loss_rtfm = RTFMLoss(alpha=self.config.alpha, margin=self.config.margin)(
                normal_scores=score_normal,
                abnormal_scores=score_abnormal,
                normal_labels=normal_labels,
                abnormal_labels=abnormal_labels,
                nor_feamagnitude=nor_feamagnitude,
                abn_feamagnitude=abn_feamagnitude,
            )
            loss_smooth = TemporalSmoothnessLoss(self.config.lambda_smooth)(scores)
            loss_sparse = SparsityLoss(self.config.lambda_sparse)(
                abnormal_scores.view(-1)
            )
            loss = loss_rtfm + loss_smooth + loss_sparse

        return CLIPTSAVideoAnomalyDetectionOutput(
            loss=loss,
            abnormal_scores=score_abnormal,
            normal_scores=score_normal,
            abn_feamagnitude=abn_feamagnitude,
            nor_feamagnitude=nor_feamagnitude,
            scores=scores,
        )
