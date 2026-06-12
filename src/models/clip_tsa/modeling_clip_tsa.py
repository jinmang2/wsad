"""CLIP-TSA: CLIP-Assisted Temporal Self-Attention for WSVAD (ICIP'23).

Slot mapping (WSAD_INTEGRATION_PLAN.md section 4):
  - slot 1 (features): CLIP ViT-B/16 frame features (512-d, text-aligned).
  - slot 2 (temporal encoder): **TSA** — Transformer self-attention over snippets
    (shared ``src.modules`` MHSA; eager or SDPA/flash via ``attn_impl``).
  - slot 4 (scoring head): RTFM-style top-k feature-magnitude MIL.
  - slot 5 (loss): RTFM loss + temporal smoothness + sparsity.

Honors the shared runner contract:
``forward(video, abnormal_labels=None, normal_labels=None) -> ModelOutput`` with
``.loss`` (None at inference) and ``.scores`` of shape ``(B, T, 1)``.

Reference: https://arxiv.org/abs/2212.05136
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from src.loss import RTFMLoss, SparsityLoss, TemporalSmoothnessLoss
from src.modules import TransformerEncoderLayer
from src.modules.mil import topk_magnitude_select
from src.registry import MODELS

from .configuration_clip_tsa import CLIPTSAConfig


@dataclass
class CLIPTSAVideoAnomalyDetectionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    abnormal_scores: Optional[torch.FloatTensor] = None
    normal_scores: Optional[torch.FloatTensor] = None
    abn_feamagnitude: Optional[torch.FloatTensor] = None
    nor_feamagnitude: Optional[torch.FloatTensor] = None
    scores: Optional[torch.FloatTensor] = None


class CLIPTSAPreTrainedModel(PreTrainedModel):
    config_class = CLIPTSAConfig
    base_model_prefix = "clip_tsa"

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
        return {"video": torch.randn(32, 10, self.config.num_segments, 512)}


@MODELS.register("clip_tsa")
class CLIPTSAForVideoAnomalyDetection(CLIPTSAPreTrainedModel):
    def __init__(self, config: CLIPTSAConfig):
        super().__init__(config)
        self.k = config.k
        f = config.feature_size

        self.encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    dim=f,
                    heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout_rate,
                    attn_impl=config.attn_impl,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(f)
        self.fc1 = nn.Linear(f, f)
        self.fc2 = nn.Linear(f, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
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
    ) -> CLIPTSAVideoAnomalyDetectionOutput:
        f = self.config.feature_size
        video = video[..., :f]
        bs, ncrops, t, _ = video.size()

        # temporal self-attention over snippets (per crop)
        x = video.reshape(bs * ncrops, t, f)
        for layer in self.encoder:
            x = layer(x)
        features = self.norm(x)

        scores = self.relu(self.fc1(features))
        scores = self.dropout(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.dropout(scores)
        scores = self.sigmoid(self.fc3(scores))  # (bs*ncrops, T, 1)
        scores = scores.view(bs, ncrops, -1).mean(dim=1).unsqueeze(2)  # (bs, T, 1)

        feat_mag = torch.norm(features, p=2, dim=2)
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

        abn_feamagnitude, score_abnormal = topk_magnitude_select(
            abn_mag,
            abnormal_features,
            abnormal_scores,
            self.k,
            ncrops,
            t,
            f,
            self.dropout,
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
