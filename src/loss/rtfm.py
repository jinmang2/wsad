"""RTFM loss (Tian et al., ICCV'21).

Robust Temporal Feature Magnitude learning separates normal vs. abnormal by
pushing the top-k feature magnitude of abnormal snippets above a margin while
keeping normal snippet magnitudes small. Classification is a standard BCE on the
top-k mean scores. Temporal smoothness / sparsity are added by the model on top
(see ``src.loss.base``).

Reference: https://arxiv.org/abs/2101.10030
"""

import torch
from torch import nn

from src.registry import LOSSES


@LOSSES.register("rtfm")
class RTFMLoss(nn.Module):
    """BCE classification + feature-magnitude separation.

    Args:
        alpha: weight on the magnitude-separation term.
        margin: target margin between abnormal and normal top-k magnitudes.
    """

    def __init__(self, alpha: float = 0.0001, margin: float = 100.0):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.criterion = nn.BCELoss()

    def forward(
        self,
        normal_scores: torch.Tensor,
        abnormal_scores: torch.Tensor,
        normal_labels: torch.Tensor,
        abnormal_labels: torch.Tensor,
        nor_feamagnitude: torch.Tensor,
        abn_feamagnitude: torch.Tensor,
    ) -> torch.Tensor:
        # --- classification term ---
        labels = torch.cat((normal_labels, abnormal_labels), dim=0)
        scores = torch.cat((normal_scores, abnormal_scores), dim=0).squeeze()
        loss_cls = self.criterion(scores, labels)

        # --- magnitude separation term ---
        # mean over the top-k selected snippets, then L2 over feature dim.
        abn_mag = torch.norm(torch.mean(abn_feamagnitude, dim=1), p=2, dim=1)
        nor_mag = torch.norm(torch.mean(nor_feamagnitude, dim=1), p=2, dim=1)
        # abnormal magnitude should exceed `margin`; normal should be small.
        loss_abn = torch.abs(self.margin - abn_mag)
        loss_nor = nor_mag
        loss_sep = torch.mean((loss_abn + loss_nor) ** 2)

        return loss_cls + self.alpha * loss_sep
