"""RTFM: Robust Temporal Feature Magnitude learning (Tian et al., ICCV'21).

Architecture (slot mapping to WSAD_INTEGRATION_PLAN.md section 4):
  - slot 2 (temporal encoder): ``MTN`` — a multi-scale temporal network of
    pyramid dilated convolutions (PDC, dilations 1/2/4) plus a non-local global
    temporal-context branch, fused with a residual connection.
  - slot 4 (scoring head): a 3-layer MLP producing per-snippet anomaly scores,
    plus top-k feature-magnitude selection.
  - slot 5 (loss): ``RTFMLoss`` + temporal smoothness + sparsity.

The forward signature matches the runner contract shared with MGFN:
``forward(video, abnormal_labels=None, normal_labels=None)`` returning a
``ModelOutput`` with ``.loss`` (None at inference) and ``.scores`` of shape
``(B, T, 1)``.

Reference: https://arxiv.org/abs/2101.10030
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from src.loss import RTFMLoss, SparsityLoss, TemporalSmoothnessLoss
from src.registry import MODELS

from .configuration_rtfm import RTFMConfig


@dataclass
class RTFMVideoAnomalyDetectionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    abnormal_scores: Optional[torch.FloatTensor] = None
    normal_scores: Optional[torch.FloatTensor] = None
    abn_feamagnitude: Optional[torch.FloatTensor] = None
    nor_feamagnitude: Optional[torch.FloatTensor] = None
    scores: Optional[torch.FloatTensor] = None


class NonLocalBlock1D(nn.Module):
    """Embedded-Gaussian non-local block over the temporal axis (1D)."""

    def __init__(self, dim: int, bn_layer: bool = True):
        super().__init__()
        self.dim = dim
        self.inter_dim = dim // 2 or 1

        self.g = nn.Conv1d(dim, self.inter_dim, kernel_size=1)
        self.theta = nn.Conv1d(dim, self.inter_dim, kernel_size=1)
        self.phi = nn.Conv1d(dim, self.inter_dim, kernel_size=1)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv1d(self.inter_dim, dim, kernel_size=1),
                nn.BatchNorm1d(dim),
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv1d(self.inter_dim, dim, kernel_size=1)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        b = x.size(0)
        g_x = self.g(x).view(b, self.inter_dim, -1).permute(0, 2, 1)  # (B, T, C')
        theta_x = self.theta(x).view(b, self.inter_dim, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(b, self.inter_dim, -1)  # (B, C', T)

        f = torch.matmul(theta_x, phi_x)  # (B, T, T)
        f_div_c = f / f.size(-1)  # official RTFM: dot-product / N (NOT softmax)

        y = torch.matmul(f_div_c, g_x)  # (B, T, C')
        y = y.permute(0, 2, 1).contiguous().view(b, self.inter_dim, -1)
        return x + self.W(y)


class MTN(nn.Module):
    """Multi-scale Temporal Network (RTFM ``Aggregate`` module)."""

    def __init__(self, len_feature: int):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(len_feature, 512, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(len_feature, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(512),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(len_feature, 512, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            nn.BatchNorm1d(512),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv1d(len_feature, 512, kernel_size=1, bias=False),
            nn.ReLU(),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(2048, len_feature, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(len_feature),
        )
        self.non_local = NonLocalBlock1D(512, bn_layer=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out = x.permute(0, 2, 1)  # (B, F, T)
        residual = out

        out1 = self.conv_1(out)
        out2 = self.conv_2(out)
        out3 = self.conv_3(out)
        out_d = torch.cat((out1, out2, out3), dim=1)  # (B, 1536, T) local pyramid

        out_g = self.conv_4(out)  # (B, 512, T)
        out_g = self.non_local(out_g)  # global temporal context

        out = torch.cat((out_d, out_g), dim=1)  # (B, 2048, T)
        out = self.conv_5(out)  # (B, F, T)
        out = out + residual
        out = out.permute(0, 2, 1)  # (B, T, F)
        return out


class RTFMPreTrainedModel(PreTrainedModel):
    config_class = RTFMConfig
    base_model_prefix = "rtfm"

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    @property
    def dummy_inputs(self):
        # (batch_size, n_crops, n_segments, feature_dim)
        return {"video": torch.randn(32, 10, self.config.num_segments, 2049)}


@MODELS.register("rtfm")
class RTFMForVideoAnomalyDetection(RTFMPreTrainedModel):
    def __init__(self, config: RTFMConfig):
        super().__init__(config)
        self.k = config.k
        f = config.feature_size

        self.mtn = MTN(len_feature=f)
        self.fc1 = nn.Linear(f, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self._force_split = False

    @property
    def force_split(self) -> bool:
        """Force the normal/abnormal split during evaluation (debugging)."""
        return self._force_split

    @force_split.setter
    def force_split(self, val: bool):
        self._force_split = val

    def _topk_select(self, magnitudes, features, scores, ncrops, t, f):
        """Select top-k snippets by feature magnitude; return selected feats + score."""
        device = features.device
        n = magnitudes.shape[0]

        select_idx = self.dropout(torch.ones_like(magnitudes, device=device))
        mag_drop = magnitudes * select_idx
        idx = torch.topk(mag_drop, self.k, dim=1)[1]  # (n, k)

        # gather features per crop
        idx_feat = idx.unsqueeze(2).expand([-1, -1, features.shape[2]])
        feats = features.view(n, ncrops, t, f).permute(1, 0, 2, 3)
        selected = torch.zeros(0, device=device)
        for crop_feat in feats:
            sel = torch.gather(crop_feat, 1, idx_feat)
            selected = torch.cat([selected, sel])

        # gather scores
        idx_score = idx.unsqueeze(2).expand([-1, -1, scores.shape[2]])
        score = torch.mean(torch.gather(scores, 1, idx_score), dim=1)
        return selected, score

    def forward(
        self,
        video: torch.FloatTensor,
        abnormal_labels: Optional[torch.FloatTensor] = None,
        normal_labels: Optional[torch.FloatTensor] = None,
    ) -> RTFMVideoAnomalyDetectionOutput:
        # video: (B, ncrops, T, C). Slice off any appended magnitude channel.
        f = self.config.feature_size
        video = video[..., :f]
        bs, ncrops, t, _ = video.size()

        x = video.reshape(bs * ncrops, t, f)
        x = self.mtn(x)
        features = self.dropout(x)

        scores = self.relu(self.fc1(features))
        scores = self.dropout(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.dropout(scores)
        scores = self.sigmoid(self.fc3(scores))  # (bs*ncrops, T, 1)
        scores = scores.view(bs, ncrops, -1).mean(dim=1).unsqueeze(2)  # (bs, T, 1)

        feat_mag = torch.norm(features, p=2, dim=2)  # (bs*ncrops, T)
        feat_mag = feat_mag.view(bs, ncrops, -1).mean(dim=1)  # (bs, T)

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

        abn_feamagnitude, score_abnormal = self._topk_select(
            abn_mag, abnormal_features, abnormal_scores, ncrops, t, f
        )
        nor_feamagnitude, score_normal = self._topk_select(
            nor_mag, normal_features, normal_scores, ncrops, t, f
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

        return RTFMVideoAnomalyDetectionOutput(
            loss=loss,
            abnormal_scores=score_abnormal,
            normal_scores=score_normal,
            abn_feamagnitude=abn_feamagnitude,
            nor_feamagnitude=nor_feamagnitude,
            scores=scores,
        )
