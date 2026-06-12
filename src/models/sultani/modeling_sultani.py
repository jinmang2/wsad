"""Sultani MIL baseline (CVPR'18) — the simplest WSVAD method.

Slot mapping (WSAD_INTEGRATION_PLAN.md section 4):
  - slot 2 (temporal encoder): none — crops are mean-pooled, segments scored
    independently.
  - slot 4 (scoring head): a 3-layer FC regressor (2048->512->32->1) producing a
    per-segment anomaly score.
  - slot 5 (loss): MIL ranking on bag-max scores + smoothness + sparsity.

Honors the shared runner contract:
``forward(video, abnormal_labels=None, normal_labels=None) -> ModelOutput`` with
``.loss`` (None at inference) and ``.scores`` of shape ``(B, T, 1)``.

Reference: https://arxiv.org/abs/1801.04264
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from src.loss import MILRankingLoss
from src.registry import MODELS

from .configuration_sultani import SultaniConfig


@dataclass
class SultaniVideoAnomalyDetectionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    scores: Optional[torch.FloatTensor] = None


class SultaniPreTrainedModel(PreTrainedModel):
    config_class = SultaniConfig
    base_model_prefix = "sultani"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    @property
    def dummy_inputs(self):
        return {"video": torch.randn(32, 10, 32, 2049)}


@MODELS.register("sultani")
class SultaniForVideoAnomalyDetection(SultaniPreTrainedModel):
    def __init__(self, config: SultaniConfig):
        super().__init__(config)
        self.regressor = nn.Sequential(
            nn.Linear(config.feature_size, config.hidden1),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden1, config.hidden2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden2, 1),
            nn.Sigmoid(),
        )
        self._force_split = False

    @property
    def force_split(self) -> bool:
        """Force the normal/abnormal split during evaluation (debugging)."""
        return self._force_split

    @force_split.setter
    def force_split(self, val: bool):
        self._force_split = val

    def forward(
        self,
        video: torch.FloatTensor,
        abnormal_labels: Optional[torch.FloatTensor] = None,
        normal_labels: Optional[torch.FloatTensor] = None,
    ) -> SultaniVideoAnomalyDetectionOutput:
        # video: (B, ncrops, T, C) -> drop magnitude channel, mean over crops.
        f = self.config.feature_size
        x = video[..., :f].mean(dim=1)  # (B, T, f)
        scores = self.regressor(x)  # (B, T, 1)

        loss = None
        if abnormal_labels is not None and normal_labels is not None:
            half = scores.size(0) // 2
            normal_scores = scores[:half, :, 0]  # (half, T)
            abnormal_scores = scores[half:, :, 0]
            loss = MILRankingLoss(
                lambda_smooth=self.config.lambda_smooth,
                lambda_sparse=self.config.lambda_sparse,
            )(normal_scores=normal_scores, abnormal_scores=abnormal_scores)

        return SultaniVideoAnomalyDetectionOutput(loss=loss, scores=scores)
