"""BN-WVAD: BatchNorm-based Weakly Supervised VAD (Zhou et al., 2023).

Faithful port of the official ``WSAD`` (cool-xuan/BN-WVAD, ``models/model.py`` +
``normal_head.py`` + ``translayer.py``) into this repo's HF style. Module/param
names mirror the official state dict so ``ckpts/*.pkl`` loads directly.

Architecture (official):
  - ``embedding`` (``Temporal``): ``Conv1d(1024->512, k=3) + ReLU``.
  - ``selfatt`` (shared dual-branch ``Transformer`` — same translayer as UR-DMU).
  - ``normal_head`` (``NormalHead``): conv/BN stack whose BatchNorm running stats
    define the normal distribution; anomaly = **DFM** (Mahalanobis distance of the
    pre-BN features to the BN running mean/var). Eval score = ``Σ distance ·
    normal_score`` (an unbounded rank-score; fine for ROC-AUC).

Train returns ``pre_normal_scores`` + selected normal/abnormal feats (MPP). Verified
vs the official checkpoint in ``scripts/verify_bn_wvad.py``.
Ref: https://github.com/cool-xuan/BN-WVAD — https://arxiv.org/abs/2311.15367
"""

from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from src.modules.translayer import Transformer
from src.registry import MODELS

from .configuration_bn_wvad import BNWVADConfig


@dataclass
class BNWVADVideoAnomalyDetectionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    scores: Optional[torch.FloatTensor] = None


class Temporal(nn.Module):
    def __init__(self, input_size: int, out_size: int):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(input_size, out_size, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        return x.permute(0, 2, 1)


class NormalHead(nn.Module):
    """Official ``NormalHead``: conv/BN stack; BN running stats = normality model."""

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

    def forward(self, x):  # x: (BN, C, T) -> [conv1_out, conv2_out, scores]
        outputs = []
        x = self.conv1(x)
        outputs.append(x)
        x = self.conv2(self.act(self.bn1(x)))
        outputs.append(x)
        x = self.sigmoid(self.conv3(self.act(self.bn2(x))))
        outputs.append(x)
        return outputs


class BNWVADPreTrainedModel(PreTrainedModel):
    config_class = BNWVADConfig
    base_model_prefix = "bn_wvad"

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if getattr(module, "weight", None) is not None:
                nn.init.xavier_uniform_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias, 0)

    @property
    def dummy_inputs(self):
        return {"video": torch.randn(2, 10, 32, self.config.feature_size)}


@MODELS.register("bn_wvad")
class BNWVADForVideoAnomalyDetection(BNWVADPreTrainedModel):
    def __init__(self, config: BNWVADConfig):
        super().__init__(config)
        h = config.hidden_size
        self.ratio_sample = config.ratio_sample
        self.ratio_batch = config.ratio_batch
        self.normal_head = NormalHead(h, config.ratios, config.kernel_sizes)
        self.embedding = Temporal(config.feature_size, h)
        self.selfatt = Transformer(h, config.num_layers, config.num_heads, 128, h, dropout=config.dropout_rate)
        self._force_split = False

    @property
    def force_split(self) -> bool:
        return self._force_split

    @force_split.setter
    def force_split(self, val: bool):
        self._force_split = val

    def get_normal_scores(self, x, ncrops=None):
        outputs = self.normal_head(x.permute(0, 2, 1))
        normal_scores = outputs[-1]
        xhs = outputs[:-1]
        if ncrops:
            b = normal_scores.shape[0] // ncrops
            normal_scores = normal_scores.view(b, ncrops, -1).mean(1)
        return xhs, normal_scores

    @staticmethod
    def get_mahalanobis_distance(feats, anchor, var, ncrops=None):
        # feats: (BN, C, T); anchor/var: (C,) -> (BN, T) [official, no eps]
        distance = torch.sqrt(
            torch.sum((feats - anchor[None, :, None]) ** 2 / var[None, :, None], dim=1)
        )
        if ncrops:
            bs = distance.shape[0] // ncrops
            distance = distance.view(bs, ncrops, -1).mean(1)
        return distance

    def forward(
        self,
        video: torch.FloatTensor,
        abnormal_labels: Optional[torch.FloatTensor] = None,
        normal_labels: Optional[torch.FloatTensor] = None,
    ) -> BNWVADVideoAnomalyDetectionOutput:
        x = video[..., : self.config.feature_size]
        if x.dim() == 4:
            b, n, t, d = x.size()
            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1

        x = self.embedding(x)
        x = self.selfatt(x)

        normal_feats, normal_scores = self.get_normal_scores(x, n)
        anchors = [bn.running_mean for bn in self.normal_head.bns]
        variances = [bn.running_var for bn in self.normal_head.bns]
        distances = [
            self.get_mahalanobis_distance(f, a, v, ncrops=n)
            for f, a, v in zip(normal_feats, anchors, variances)
        ]

        train = abnormal_labels is not None and normal_labels is not None
        if train:
            loss = self._compute_loss(normal_feats, normal_scores, distances, anchors, variances, b, n, t)
            # official train returns pre_normal_scores; expose distance·score as scores
            scores = (sum(distances) * normal_scores).unsqueeze(-1)
            return BNWVADVideoAnomalyDetectionOutput(loss=loss, scores=scores)

        distance_sum = sum(distances)
        return BNWVADVideoAnomalyDetectionOutput(
            loss=None, scores=(distance_sum * normal_scores).unsqueeze(-1)
        )

    # ---- training selection + losses (official pos_neg_select + MPP/Normal) ----
    def pos_neg_select(self, feats, distance, ncrops):
        # feats: (b*ncrops, c, t); distance: (b, t) [already crop-averaged]
        bsn, c, t = feats.shape
        b = bsn // ncrops
        select_num_sample = int(t * self.ratio_sample)
        select_num_batch = int(b // 2 * t * self.ratio_batch)
        feats = feats.view(b, ncrops, c, t).mean(1)  # (b, c, t)
        nor_distance = distance[: b // 2]
        nor_feats = feats[: b // 2].permute(0, 2, 1)
        abn_distance = distance[b // 2 :]
        abn_feats = feats[b // 2 :].permute(0, 2, 1)
        abn_distance_flatten = abn_distance.reshape(-1)
        abn_feats_flatten = abn_feats.reshape(-1, c)

        m_samp = torch.zeros_like(abn_distance, dtype=torch.bool)
        m_samp.scatter_(1, torch.topk(abn_distance, max(select_num_sample, 1), dim=-1)[1], True)
        m_batch = torch.zeros_like(abn_distance_flatten, dtype=torch.bool)
        m_batch.scatter_(0, torch.topk(abn_distance_flatten, max(select_num_batch, 1), dim=-1)[1], True)
        mask = m_batch | m_samp.reshape(-1)
        select_abn = abn_feats_flatten[mask]
        m = int(torch.sum(mask))

        k_nor = m // max(b // 2, 1) + 1
        idx_nor = torch.topk(nor_distance, min(k_nor, t), dim=-1)[1]
        select_nor = torch.gather(nor_feats, 1, idx_nor[..., None].expand(-1, -1, c))
        select_nor = select_nor.permute(1, 0, 2).reshape(-1, c)[:m]
        return select_nor, select_abn

    def _compute_loss(self, feats, normal_scores, distances, anchors, variances, b, n, t):
        loss_normal = torch.norm(normal_scores[: b // 2], p=2, dim=1).mean()
        loss_mpp = normal_scores.new_zeros(())
        for feat, dist, anchor, var, wt in zip(
            feats, distances, anchors, variances, self.config.w_triplet
        ):
            sel_nor, sel_abn = self.pos_neg_select(feat, dist, n)
            if sel_nor.size(0) == 0 or sel_abn.size(0) == 0:
                continue
            mse = sel_nor.new_zeros(())  # mahalanobis triplet (anchor=mean)
            triplet = nn.TripletMarginWithDistanceLoss(
                margin=self.config.mpp_margin,
                distance_function=partial(self._maha_pair, var=var),
            )
            anc = anchor[None, :].expand(min(sel_nor.size(0), sel_abn.size(0)), -1)
            mm = anc.size(0)
            loss_mpp = loss_mpp + wt * (triplet(anc, sel_nor[:mm], sel_abn[:mm]) + mse)
        return self.config.w_normal * loss_normal + self.config.w_mpp * loss_mpp

    @staticmethod
    def _maha_pair(a, b, var):
        return torch.sqrt(torch.sum((a - b) ** 2 / var, dim=-1) + 1e-12)
