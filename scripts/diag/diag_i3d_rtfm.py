"""Diagnose why our Gowtham-faithful extraction != RTFM's released Abuse001.

Hypotheses: (A) 10-crop ORDER differs, (B) RTFM used nonlocal weights,
(C) different extractor entirely. Extracts once per model (cached to /tmp) and
reports crop-matched cosine, crop-AVERAGED cosine (order-invariant), and the
snippet-0 crop permutation (best rtfm-crop per mine-crop).
"""

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.features.i3d_gowtham import build_model, extract_from_video  # noqa: E402

VIDEO = os.path.join(ROOT, ".reference", "I3D_Feature_Extraction_resnet",
                     "samplevideos", "Abuse001_x264.mp4")
RTFM = os.path.expanduser(
    "~/data/wsad/ucf_crime/_archive/UCF_Train_ten_crop_i3d/Abuse001_x264_i3d.npy")


def _cos(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _extract_cached(tag, use_nl):
    cache = f"/tmp/diag_i3d_{tag}.npy"
    if os.path.exists(cache):
        return np.load(cache)
    pth = os.path.join(ROOT, "pretrained", "i3d", f"i3d_{tag}_r50_kinetics.pth")
    m = build_model(pth, use_nl=use_nl, device="cuda" if torch.cuda.is_available() else "cpu")
    feat = extract_from_video(m, VIDEO)
    np.save(cache, feat)
    return feat


def analyze(tag, mine, rtfm):
    T = min(mine.shape[0], rtfm.shape[0])
    m, r = mine[:T], rtfm[:T]
    # crop-matched (order-sensitive)
    cm = np.mean([_cos(m[t].ravel(), r[t].ravel()) for t in range(T)])
    # crop-averaged (order-invariant)
    ma, ra = m.mean(1), r.mean(1)
    ca = np.mean([_cos(ma[t], ra[t]) for t in range(T)])
    print(f"[{tag}] T mine={mine.shape[0]} rtfm={rtfm.shape[0]} | "
          f"crop-matched cos={cm:.4f} | crop-AVG cos={ca:.4f}")
    # snippet-0 permutation: best rtfm crop for each mine crop
    M = np.array([[_cos(m[0, i], r[0, j]) for j in range(10)] for i in range(10)])
    best = M.argmax(1)
    print(f"        snippet0 mine-crop -> best rtfm-crop: {list(best)} "
          f"(diag={'identity' if list(best)==list(range(10)) else 'PERMUTED'}), "
          f"max-per-row mean={M.max(1).mean():.3f}")


def main():
    rtfm = np.load(RTFM)
    analyze("baseline", _extract_cached("baseline", False), rtfm)
    analyze("nonlocal", _extract_cached("nonlocal", True), rtfm)


if __name__ == "__main__":
    main()
