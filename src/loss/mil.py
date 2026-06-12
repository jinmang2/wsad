"""MIL ranking loss (Sultani et al., CVPR'18).

The weak label lives at the *bag* (video) level. For each bag take the maximum
segment score; an abnormal bag's max should outrank a normal bag's max by a
margin. Temporal smoothness and sparsity regularize the abnormal score curve.

Reference: "Real-world Anomaly Detection in Surveillance Videos",
https://arxiv.org/abs/1801.04264
"""

import torch
from torch import nn

from src.registry import LOSSES


@LOSSES.register("mil_ranking")
class MILRankingLoss(nn.Module):
    """Hinge ranking on bag-max scores + smoothness + sparsity (Sultani'18).

    Args:
        lambda_smooth: weight on temporal smoothness of the abnormal curve.
        lambda_sparse: weight on sparsity of the abnormal curve.
    """

    def __init__(self, lambda_smooth: float = 8e-5, lambda_sparse: float = 8e-5):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_sparse = lambda_sparse

    def forward(
        self,
        normal_scores: torch.Tensor,
        abnormal_scores: torch.Tensor,
    ) -> torch.Tensor:
        # scores: (B, T) per-segment anomaly scores in [0, 1]
        nor_max = normal_scores.max(dim=1).values  # (B,)
        abn_max = abnormal_scores.max(dim=1).values  # (B,)

        ranking = torch.relu(1.0 - abn_max + nor_max).mean()

        diffs = abnormal_scores[:, 1:] - abnormal_scores[:, :-1]
        smooth = (diffs**2).sum(dim=1).mean()
        sparse = abnormal_scores.sum(dim=1).mean()

        return ranking + self.lambda_smooth * smooth + self.lambda_sparse * sparse
