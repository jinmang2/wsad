"""Per-(backbone, head) eval shaping for the comparison matrix (additive).

The core ``trainer.evaluate`` / ``inference.score_feature`` path is **I3D /
visual-head shaped** (test feature ``(T, ncrops, D)`` -> ``(1, ncrops, T, D)``,
no ``lengths``) and stays unchanged — design invariant: the original
training/offline-eval is never modified. But two real cases that path cannot
serve correctly:

  - **CLIP test layout is ``(ncrops=1, T, D)``** (transposed vs I3D's
    ``(T, ncrops, D)``), so the I3D ``permute(1,0,2)`` collapses ``T`` to 1.
  - **VadCLIP** consumes ``(B=windows, T<=256, D)`` + per-window ``lengths`` (the
    official ``process_split`` windowed eval); the generic 4-D path crashes
    (``DistanceAdj`` size mismatch) and is numerically wrong otherwise.

This module supplies the per-head input shaping the matrix + CLIP serving need,
dispatching on ``(backbone, head)``. Visual heads slice their own magnitude
channel, so an appended-magnitude feature is fine.
"""

from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_curve

# heads whose forward consumes (B, ncrops, T, D) and slices [..., :feature_size]
_VISUAL_LAYOUT_HEADS = {
    "mil",
    "sultani",
    "rtfm",
    "mgfn",
    "ur_dmu",
    "bn_wvad",
    "s3r",
    "gs_moe",
    "clip_tsa",
}
_VADCLIP_MAXLEN = 256
# heads whose O(T^2) full-length test attention OOMs an 8 GB GPU with all 10 crops
# at once -> score each crop separately + average (the official per-crop protocol).
_PERCROP_HEADS = {"ur_dmu", "bn_wvad"}


def _to_crops_layout(feat: np.ndarray, backbone: str) -> np.ndarray:
    """Shape a single video's cached feature to ``(1, ncrops, T, D)``.

    I3D test cache is ``(T, ncrops, D)``; CLIP cache is ``(ncrops=1, T, D)``.
    """
    feat = np.asarray(feat, dtype=np.float32)
    if feat.ndim == 2:  # (T, D) -> single crop
        feat = feat[None]  # (1, T, D) == (ncrops, T, D)
        return feat[None]  # (1, 1, T, D)
    if backbone == "i3d":  # (T, ncrops, D) -> (ncrops, T, D)
        feat = np.transpose(feat, (1, 0, 2))
    # CLIP: already (ncrops, T, D)
    return feat[None]  # (1, ncrops, T, D)


def _vadclip_lengths(length: int, maxlen: int = _VADCLIP_MAXLEN) -> torch.Tensor:
    """Per-window valid lengths for an official ``process_split`` (UCF eval)."""
    n = int(length / maxlen) + 1
    out = torch.zeros(n, dtype=torch.long)
    rem = length
    for j in range(n):
        if rem > maxlen:
            out[j] = maxlen
            rem -= maxlen
        else:
            out[j] = rem
    return out


def _vadclip_split(feat: np.ndarray, maxlen: int = _VADCLIP_MAXLEN):
    """``(T, D)`` -> ``(windows, maxlen, D)`` (last window zero-padded)."""
    t, d = feat.shape
    if t < maxlen:
        out = np.zeros((1, maxlen, d), dtype=np.float32)
        out[0, :t] = feat
        return out, t
    n = int(t / maxlen) + 1
    out = np.zeros((n, maxlen, d), dtype=np.float32)
    for i in range(n):
        chunk = feat[i * maxlen : i * maxlen + maxlen]
        out[i, : len(chunk)] = chunk
    return out, t


@torch.no_grad()
def score_video(
    model: torch.nn.Module,
    feature: np.ndarray,
    backbone: str,
    head: str,
    device: str = "cpu",
    frames_per_clip: int = 16,
) -> np.ndarray:
    """Per-frame anomaly scores for one video's cached feature.

    Returns a 1-D array of length ``T_snippets * frames_per_clip``.
    """
    feat = np.asarray(feature, dtype=np.float32)

    if head == "vadclip":
        if feat.ndim == 3:  # (ncrops=1, T, D) -> (T, D)
            feat = feat[0]
        split, length = _vadclip_split(feat)
        visual = torch.as_tensor(split, dtype=torch.float32, device=device)
        lengths = _vadclip_lengths(length).to(device)
        out = model(video=visual, lengths=lengths)
        # (windows, maxlen, 1) -> (windows*maxlen,) -> first `length` snippets
        prob = torch.sigmoid(out.binary_logits.reshape(-1))[:length]
        scores = prob.cpu().numpy()

    else:  # visual-layout heads + tpwng (all consume (B, ncrops, T, D); tpwng
        # means over the crop dim inside _encode, so the 4-D layout is correct)
        x = torch.as_tensor(_to_crops_layout(feat, backbone), device=device)
        if head in _PERCROP_HEADS:
            # UR-DMU/BN-WVAD: official eval scores each crop's FULL-LENGTH sequence
            # separately then averages the 10 crops (ucf_infer.py: per-crop forward,
            # mean over 10). Doing all crops in one forward is non-faithful AND ~10x
            # the peak memory of the O(T^2) distance/attention -> OOMs 8 GB. Per-crop
            # = faithful + fits.
            per = [model(video=x[:, c : c + 1]).scores.squeeze(0).squeeze(-1)
                   for c in range(x.shape[1])]
            scores = torch.stack(per).mean(0).float().cpu().numpy()
        else:
            out = model(video=x)
            scores = out.scores.squeeze(0).squeeze(-1).float().cpu().numpy()

    return np.repeat(scores, frames_per_clip)


def frame_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    is_abnormal: Optional[np.ndarray] = None,
    far_threshold: float = 0.5,
) -> Dict[str, float]:
    """Frame-level metrics from concatenated per-frame scores and labels.

    Shared by :func:`evaluate` and ``WSVADTrainer.evaluate`` so the two cannot drift.
    ``is_abnormal`` marks which frames came from anomalous *videos*; pass ``None`` to get
    only the two primary metrics.
    """
    fpr, tpr, _ = roc_curve(labels, preds)
    precision, recall, _ = precision_recall_curve(labels, preds)
    metrics = {"roc_auc": float(auc(fpr, tpr)), "pr_auc": float(auc(recall, precision))}
    if is_abnormal is None:
        return metrics

    abn_label, abn_pred = labels[is_abnormal], preds[is_abnormal]
    if abn_label.size and 0 < abn_label.sum() < abn_label.size:
        f, t, _ = roc_curve(abn_label, abn_pred)
        metrics["abnormal_auc"] = float(auc(f, t))

    normal_pred = preds[~is_abnormal]
    if normal_pred.size:
        metrics["far_normal"] = float((normal_pred > far_threshold).mean())
    return metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    test_dataset,
    backbone: str,
    head: str,
    device: str = "cpu",
    frames_per_clip: int = 16,
    far_threshold: float = 0.5,
) -> Dict[str, float]:
    """Frame-level metrics over the UCF-Crime test set.

    ``roc_auc`` is the primary number every paper reports, but it is a weak discriminator
    here: the test set is dominated by normal frames, so a model that merely separates
    normal *videos* from anomalous ones already scores well. The plan
    (WSAD_INTEGRATION_PLAN.md §8) asks for two secondary metrics that do not have that
    escape hatch, and this returns them alongside:

    - ``abnormal_auc`` — ROC-AUC computed **only over frames from anomalous videos**. Every
      video in the subset contains both normal and anomalous frames, so this measures
      temporal localization rather than video-level separability. It is always the harder
      number and it is where methods actually differ.
    - ``far_normal`` — the false-alarm rate on normal videos: the fraction of their frames
      scored above ``far_threshold``. A model can buy AUC with a globally raised score
      curve; this catches that.

    Returned as extra keys, so existing ``roc_auc``/``pr_auc`` numbers stay comparable.
    """
    model.eval()
    preds, labels, is_abnormal = [], [], []
    for i in range(len(test_dataset)):
        s = test_dataset[i]
        p = score_video(model, s["feature"], backbone, head, device, frames_per_clip)
        y = np.asarray(s["label"])
        n = min(len(p), len(y))
        preds.append(p[:n])
        labels.append(y[:n])
        is_abnormal.append(np.full(n, float(np.asarray(s["anomaly"]).item()) > 0.5))
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    is_abnormal = np.concatenate(is_abnormal)

    return frame_metrics(preds, labels, is_abnormal, far_threshold)
