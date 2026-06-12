"""BN-WVAD: BatchNorm-based Weakly Supervised VAD (Zhou et al., 2023).

Slot mapping (WSAD_INTEGRATION_PLAN.md section 4):
  - slot 2 (temporal encoder): Conv1d embedding + self-attention.
  - slot 4 (scoring head): a ``NormalHead`` whose BatchNorm running stats define
    normality; anomaly = **DFM** (Divergence of Feature from Mean — a Mahalanobis
    distance to the BN running mean/var) times a learned normal score.
  - slot 5 (loss): NormalLoss (normal videos -> low scores) + MPP triplet (push
    selected abnormal feats far from the BN mean, normal feats close).

Honors the runner contract: ``forward(video, abnormal_labels, normal_labels) ->
ModelOutput`` with ``.loss`` and ``.scores`` (B, T, 1). Note BN-WVAD's score is an
**unbounded** rank-score (distance x score), not a [0,1] probability — fine for
ROC-AUC (rank-based).

Cross-checked vs official cool-xuan/BN-WVAD (models/model.py, normal_head.py,
losses/{mpp,normal}_loss.py). Faithful: NormalHead, DFM, MPP triplet (per-BN
weights [5,20], margin 1), NormalLoss (L2 of normal scores), selection by top-DFM.
**Simplified (documented):** features are crop-averaged before the pos/neg
selection (cleaner than the official per-crop reshape); self-attention is this
repo's shared block.
"""

from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from src.modules import TransformerEncoderLayer
from src.registry import MODELS

from .configuration_bn_wvad import BNWVADConfig


@dataclass
class BNWVADVideoAnomalyDetectionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    scores: Optional[torch.FloatTensor] = None


def _mahalanobis(x, mu, var):
    # x: (..., C); mu/var: (C,) -> (...) distance
    return torch.sqrt(torch.sum((x - mu) ** 2 / var, dim=-1) + 1e-12)


class NormalHead(nn.Module):
    """Conv stack with two BatchNorms whose running stats define normality."""

    def __init__(self, in_channel=512, ratios=(16, 32), kernel_sizes=(1, 1, 1)):
        super().__init__()
        r1, r2 = ratios
        k = kernel_sizes
        self.conv1 = nn.Conv1d(in_channel, in_channel // r1, k[0], 1, k[0] // 2)
        self.bn1 = nn.BatchNorm1d(in_channel // r1)
        self.conv2 = nn.Conv1d(in_channel // r1, in_channel // r2, k[1], 1, k[1] // 2)
        self.bn2 = nn.BatchNorm1d(in_channel // r2)
        self.conv3 = nn.Conv1d(in_channel // r2, 1, k[2], 1, k[2] // 2)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.bns = [self.bn1, self.bn2]

    def forward(self, x):  # x: (BN, C, T)
        feats = []
        x = self.conv1(x)
        feats.append(x)  # pre-bn1 feature
        x = self.conv2(self.act(self.bn1(x)))
        feats.append(x)  # pre-bn2 feature
        scores = self.sigmoid(self.conv3(self.act(self.bn2(x))))  # (BN, 1, T)
        return feats, scores


class BNWVADPreTrainedModel(PreTrainedModel):
    config_class = BNWVADConfig
    base_model_prefix = "bn_wvad"

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    @property
    def dummy_inputs(self):
        return {"video": torch.randn(32, 10, 32, 2049)}


@MODELS.register("bn_wvad")
class BNWVADForVideoAnomalyDetection(BNWVADPreTrainedModel):
    def __init__(self, config: BNWVADConfig):
        super().__init__(config)
        h = config.hidden_size
        self.embedding = nn.Sequential(
            nn.Conv1d(config.feature_size, h, 3, padding=1), nn.ReLU()
        )
        self.selfatt = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    h,
                    heads=config.num_heads,
                    mlp_ratio=1.0,
                    dropout=config.dropout_rate,
                    attn_impl=config.attn_impl,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.normal_head = NormalHead(h, config.ratios, config.kernel_sizes)
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
    ) -> BNWVADVideoAnomalyDetectionOutput:
        f = self.config.feature_size
        video = video[..., :f]
        bs, ncrops, t, _ = video.size()
        x = rearrange(video, "b n t c -> (b n) t c")

        x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)  # (BN, T, h)
        for layer in self.selfatt:
            x = layer(x)

        feats, scores = self.normal_head(
            x.permute(0, 2, 1)
        )  # feats:[(BN,C,T)], scores:(BN,1,T)
        normal_scores = scores.view(bs, ncrops, -1).mean(dim=1)  # (B, T)

        # DFM: Mahalanobis distance of each pre-BN feature to its BN running stats,
        # crop-averaged. Sum across BN layers -> anomaly weighting.
        distances = []
        for feat, bn in zip(feats, self.normal_head.bns):
            d = _mahalanobis(
                feat.permute(0, 2, 1), bn.running_mean, bn.running_var
            )  # (BN, T)
            distances.append(d.view(bs, ncrops, -1).mean(dim=1))  # (B, T)
        distance_sum = sum(distances)  # (B, T)

        scores_out = (distance_sum * normal_scores).unsqueeze(
            -1
        )  # (B, T, 1) rank-score

        loss = None
        if abnormal_labels is not None and normal_labels is not None:
            loss = self._compute_loss(feats, normal_scores, distances, bs, ncrops)

        return BNWVADVideoAnomalyDetectionOutput(loss=loss, scores=scores_out)

    # ---- losses (official NormalLoss + MPPLoss) ----
    def _select(self, feat_bctn, distance, half, t):
        """Top-DFM abnormal snippets + count-matched top-DFM normal snippets.

        feat_bctn: (B, C, T) crop-averaged; distance: (B, T).
        """
        c = feat_bctn.size(1)
        nor_d, abn_d = distance[:half], distance[half:]
        nor_f = feat_bctn[:half].permute(0, 2, 1)  # (B, T, C)
        abn_f = feat_bctn[half:].permute(0, 2, 1)

        k_samp = max(int(t * self.config.ratio_sample), 1)
        k_batch = max(int(half * t * self.config.ratio_batch), 1)

        m_samp = torch.zeros_like(abn_d, dtype=torch.bool)
        m_samp.scatter_(1, torch.topk(abn_d, k_samp, dim=1)[1], True)
        m_batch = torch.zeros_like(abn_d.reshape(-1), dtype=torch.bool)
        m_batch.scatter_(0, torch.topk(abn_d.reshape(-1), k_batch)[1], True)
        mask = m_batch | m_samp.reshape(-1)
        sel_abn = abn_f.reshape(-1, c)[mask]  # (M, C)

        m = sel_abn.size(0)
        k_nor = m // half + 1
        idx_nor = torch.topk(nor_d, min(k_nor, t), dim=1)[1]
        sel_nor = torch.gather(nor_f, 1, idx_nor[..., None].expand(-1, -1, c))
        sel_nor = sel_nor.reshape(-1, c)[:m]  # (M, C)
        return sel_nor, sel_abn

    def _compute_loss(self, feats, normal_scores, distances, bs, ncrops):
        half = bs // 2
        t = normal_scores.size(1)

        # NormalLoss: normal videos' score curve should be small (L2)
        loss_normal = torch.norm(normal_scores[:half], p=2, dim=1).mean()

        # MPP triplet per BN layer: anchor=running_mean, pos=normal feat (close),
        # neg=abnormal feat (far), mahalanobis distance, weighted [5, 20].
        loss_mpp = normal_scores.new_zeros(())
        for feat, bn, dist, wt in zip(
            feats, self.normal_head.bns, distances, self.config.w_triplet
        ):
            feat_bct = feat.view(bs, ncrops, feat.size(1), feat.size(2)).mean(
                1
            )  # (B,C,T)
            sel_nor, sel_abn = self._select(feat_bct, dist, half, t)
            if sel_nor.size(0) == 0:
                continue
            triplet = nn.TripletMarginWithDistanceLoss(
                margin=self.config.mpp_margin,
                distance_function=partial(_mahalanobis, var=bn.running_var),
            )
            anchor = bn.running_mean[None, :].expand(sel_nor.size(0), -1)
            loss_mpp = loss_mpp + wt * triplet(anchor, sel_nor, sel_abn)

        return self.config.w_normal * loss_normal + self.config.w_mpp * loss_mpp
