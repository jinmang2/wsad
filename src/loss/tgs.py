"""Temporal Gaussian Splatting (TGS) loss — GS-MoE (ICCV'25).

Self-supervised refinement of the abnormal score curve: detect peaks in the
predicted anomaly scores, splat a Gaussian around each peak (width from the
local monotonic run, sigma from local score spread), and use the normalized sum
of Gaussians as a soft pseudo-label that BCE pulls the scores toward. This pulls
in subtle low-scoring snippets near confident peaks.

Paper: "Mixture of Experts Guided by Gaussian Splatters Matters" (arXiv 2508.06318).
Official code unreleased at implementation time — this follows the paper's
equations (peak prominence > 0.2 over ±2 neighbours; Gaussian splat Eq. 4;
pseudo-label Eq. 5; loss Eq. 6 = top-k-norm MIL + BCE(score, pseudo)). Peak
prominence is approximated in torch; documented as such.
"""

import torch
from torch import nn

from src.registry import LOSSES


def _find_peaks(s: torch.Tensor, prominence: float = 0.2, nbr: int = 2):
    """Local maxima of a 1-D score curve with a prominence threshold.

    A point is a peak if it is the max within ±``nbr`` and stands ``prominence``
    above the lowest neighbour in that window (torch approximation of
    scipy.signal.find_peaks prominence).
    """
    t = s.shape[0]
    peaks = []
    for i in range(t):
        lo, hi = max(0, i - nbr), min(t, i + nbr + 1)
        window = s[lo:hi]
        if s[i] >= window.max() and (s[i] - window.min()) > prominence:
            peaks.append(i)
    return peaks


def _gaussian_pseudo_label(s: torch.Tensor, prominence: float = 0.2) -> torch.Tensor:
    """Build the Gaussian-splatted pseudo-label for one abnormal score curve."""
    t = s.shape[0]
    device = s.device
    peaks = _find_peaks(s, prominence)
    if not peaks:
        return torch.zeros(t, device=device)

    idx = torch.arange(t, device=device).float()
    accum = torch.zeros(t, device=device)
    for p in peaks:
        # width: monotonic increasing run before, decreasing run after
        v1 = 0
        while p - v1 - 1 >= 0 and s[p - v1 - 1] <= s[p - v1]:
            v1 += 1
        v2 = 0
        while p + v2 + 1 < t and s[p + v2 + 1] <= s[p + v2]:
            v2 += 1
        w = max(min(v1, v2), 1)
        lo, hi = max(0, p - w), min(t, p + w + 1)
        sigma = s[lo:hi].std().clamp_min(1e-3)

        gauss = torch.exp(-((idx - p) ** 2) / (2 * sigma**2))
        # gating mask G: within width and score >= s_peak - sigma (Eq. 4)
        mask = torch.zeros(t, device=device)
        mask[lo:hi] = (s[lo:hi] >= (s[p] - sigma)).float()
        mask[p] = 1.0
        accum = accum + gauss * mask

    return accum.clamp(0, 1)


@LOSSES.register("tgs")
class TemporalGaussianSplattingLoss(nn.Module):
    """BCE between abnormal scores and their Gaussian-splatted pseudo-labels."""

    def __init__(self, prominence: float = 0.2):
        super().__init__()
        self.prominence = prominence

    def forward(self, abnormal_scores: torch.Tensor) -> torch.Tensor:
        # abnormal_scores: (B, T) in [0, 1]
        with torch.no_grad():
            targets = torch.stack(
                [
                    _gaussian_pseudo_label(abnormal_scores[i].detach(), self.prominence)
                    for i in range(abnormal_scores.shape[0])
                ]
            )
        return nn.functional.binary_cross_entropy(
            abnormal_scores.clamp(1e-6, 1 - 1e-6), targets
        )
