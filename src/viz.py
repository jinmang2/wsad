"""Qualitative figure for video anomaly detection (issue #21).

The figure every WSVAD paper plots: a per-frame/snippet anomaly-score curve with
the ground-truth anomaly region(s) shaded. This is the liveness proof for the
serving pipeline; the full qualitative suite (top-k thumbnails, PR/ROC, cross-
matrix panels) is Spec 3.

``matplotlib`` is imported lazily so importing this module is cheap and never
fails when plotting isn't needed.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

Interval = Tuple[int, int]


def _mask_to_intervals(mask: Sequence[int]) -> List[Interval]:
    """Contiguous True runs of a binary mask -> list of half-open [start, end)."""
    m = np.asarray(mask).astype(np.int8)
    if m.size == 0:
        return []
    padded = np.concatenate(([0], m, [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def _coerce_gt(gt, n: int) -> List[Interval]:
    """Accept a binary mask (len == n) or a list of (start, end) intervals."""
    if gt is None:
        return []
    arr = np.asarray(gt)
    if arr.ndim == 1 and arr.size == n and set(np.unique(arr).tolist()).issubset({0, 1}):
        return _mask_to_intervals(arr)
    return [(int(s), int(e)) for s, e in gt]


def plot_anomaly_scores(
    scores: Sequence[float],
    gt: Optional[Union[Sequence[int], Sequence[Interval]]] = None,
    *,
    threshold: Optional[float] = None,
    title: Optional[str] = None,
    xlabel: str = "snippet / frame index",
    ylabel: str = "anomaly score",
    legend: bool = True,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Plot a per-index anomaly-score curve with GT anomaly regions shaded.

    Args:
        scores: 1-D per-index anomaly scores.
        gt: a binary mask (``len == len(scores)``) or a list of ``(start, end)``
            half-open index intervals marking ground-truth anomalies.
        threshold: optional horizontal decision line.
        title: optional figure title.
        save_path: if set, save the figure (forces the headless Agg backend).
        show: if True, call ``plt.show()``.
    Returns:
        ``(fig, ax)``.
    """
    import matplotlib

    if save_path is not None and not show:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    scores = np.asarray(scores, dtype=float).ravel()
    x = np.arange(len(scores))

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(x, scores, color="#1f77b4", lw=1.5, label="anomaly score")

    for i, (s, e) in enumerate(_coerce_gt(gt, len(scores))):
        # light-orange GT region with dashed red boundaries (paper convention)
        ax.axvspan(s, e, color="#ff9e4a", alpha=0.35,
                   label="ground truth" if i == 0 else None)
        for boundary in (s, e):
            ax.axvline(boundary, color="#d62728", ls="--", lw=1.0)

    if threshold is not None:
        ax.axhline(threshold, color="gray", ls="--", lw=1, label="threshold")

    top = 1.05 if (scores.size and scores.max() <= 1) else (float(scores.max()) * 1.05 if scores.size else 1.0)
    ax.set_xlim(0, max(len(scores) - 1, 1))
    ax.set_ylim(0, top)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    return fig, ax
