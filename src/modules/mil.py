"""Shared MIL scoring primitives (slot 4).

``topk_magnitude_select`` is the top-k feature-magnitude selection used by both
RTFM and CLIP-TSA: pick the ``k`` snippets with largest L2 feature magnitude
(with dropout on the selection mask), then gather those snippets' features (per
crop) and their mean score. Factoring it here keeps every magnitude-MIL method a
config, not a fork.
"""

from typing import Tuple

import torch
from torch import nn


def topk_magnitude_select(
    magnitudes: torch.Tensor,  # (n, T)
    features: torch.Tensor,  # (n*ncrops, T, f)
    scores: torch.Tensor,  # (n, T, 1)
    k: int,
    ncrops: int,
    t: int,
    f: int,
    dropout: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (selected_features, topk_mean_score).

    selected_features: (n*ncrops, k, f) gathered at the top-k magnitude indices.
    topk_mean_score:   (n, 1) mean score over the top-k snippets.
    """
    device = features.device
    n = magnitudes.shape[0]

    select_idx = dropout(torch.ones_like(magnitudes, device=device))
    mag_drop = magnitudes * select_idx
    idx = torch.topk(mag_drop, k, dim=1)[1]  # (n, k)

    idx_feat = idx.unsqueeze(2).expand([-1, -1, features.shape[2]])
    feats = features.view(n, ncrops, t, f).permute(1, 0, 2, 3)
    selected = torch.zeros(0, device=device)
    for crop_feat in feats:
        sel = torch.gather(crop_feat, 1, idx_feat)
        selected = torch.cat([selected, sel])

    idx_score = idx.unsqueeze(2).expand([-1, -1, scores.shape[2]])
    score = torch.mean(torch.gather(scores, 1, idx_score), dim=1)
    return selected, score
