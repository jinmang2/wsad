"""Grid over I3D preprocessing hypotheses vs RTFM's released Abuse001 feature.

After verify_i3d_extract.py proved our extractor is bit-exact to Gowtham yet only
hits crop-avg cosine 0.74 (nonlocal) / 0.51 (baseline) vs RTFM — and crop-order,
temporal offset, and BGR were ruled out — this sweeps the remaining cheap
preprocessing knobs (resize dims x interpolation) to see if any closes the gap.
If none beats ~0.74, the evidence-based conclusion is that RTFM used a different
(unpublished) I3D checkpoint / extractor, not a preprocessing variant of Gowtham.

    uv run python scripts/grid_i3d_rtfm.py

Uses the non-local model (closest to RTFM). Each config re-extracts Abuse001
(~90 s); results cached to /tmp/grid_i3d_<tag>.npy.
"""

import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import src.features.i3d_gowtham as g  # noqa: E402

VIDEO = os.path.join(ROOT, ".reference", "I3D_Feature_Extraction_resnet",
                     "samplevideos", "Abuse001_x264.mp4")
RTFM = os.path.expanduser(
    "~/data/wsad/ucf_crime/_archive/UCF_Train_ten_crop_i3d/Abuse001_x264_i3d.npy")
PTH = os.path.join(ROOT, "pretrained", "i3d", "i3d_nonlocal_r50_kinetics.pth")

_INTERP = {
    "lanczos": getattr(Image, "Resampling", Image).LANCZOS,
    "bilinear": getattr(Image, "Resampling", Image).BILINEAR,
    "bicubic": getattr(Image, "Resampling", Image).BICUBIC,
}

# (tag, W, H, interp). The frame MUST be 256-tall: Gowtham's 10-crop coords
# (16:240, 58:282, -224:, ...) are hardcoded for a 256x340 frame, so resize is NOT
# a free variable — only the interpolation filter is. (340,256,lanczos) = faithful default.
CONFIGS = [
    ("w340h256_lanczos", 340, 256, "lanczos"),
    ("w340h256_bilinear", 340, 256, "bilinear"),
    ("w340h256_bicubic", 340, 256, "bicubic"),
]


def _make_load_frame(W, H, interp):
    rs = _INTERP[interp]

    def load_frame(frame_file):
        data = np.array(Image.open(frame_file).resize((W, H), rs)).astype(float)
        data = (data * 2 / 255) - 1
        return data

    return load_frame


def _cos(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def crop_avg_cos(feat, rtfm):
    T = min(len(feat), len(rtfm))
    a, b = feat[:T].mean(1), rtfm[:T].mean(1)
    return np.mean([_cos(a[t], b[t]) for t in range(T)])


def main():
    rtfm = np.load(RTFM)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = g.build_model(PTH, use_nl=True, device=dev)
    orig = g.load_frame
    results = []
    for tag, W, H, interp in CONFIGS:
        cache = f"/tmp/grid_i3d_{tag}.npy"
        if os.path.exists(cache):
            feat = np.load(cache)
        else:
            # H!=256 means the oversample crop coords (designed for 256-tall) shift;
            # keep Gowtham's crop logic, vary only the resize filter (the one free
            # knob — see CONFIGS: the 10-crop coords pin the frame to 256-tall).
            g.load_frame = _make_load_frame(W, H, interp)
            try:
                feat = g.extract_from_video(model, VIDEO, device=dev)
            except Exception as e:
                print(f"  [{tag}] extract failed: {e}")
                continue
            finally:
                g.load_frame = orig
            np.save(cache, feat)
        c = crop_avg_cos(feat, rtfm)
        results.append((tag, c))
        print(f"  [{tag}] crop-avg cos vs RTFM = {c:.4f}  shape={feat.shape}")

    print("\n=== summary (vs RTFM Abuse001, nonlocal) ===")
    for tag, c in sorted(results, key=lambda x: -x[1]):
        print(f"  {c:.4f}  {tag}")
    best = max(results, key=lambda x: x[1]) if results else ("none", 0)
    verdict = ("CLOSES gap" if best[1] > 0.9 else
               "does NOT close gap -> RTFM used a different/unpublished I3D checkpoint")
    print(f"\nbest = {best[0]} ({best[1]:.4f}) -> {verdict}")


if __name__ == "__main__":
    main()
