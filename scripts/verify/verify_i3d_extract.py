"""Verify the Gowtham/RTFM-faithful I3D extractor (src.features.i3d_gowtham).

Two checks, on Gowtham's bundled ``samplevideos/Abuse001_x264.mp4``:

  (1) PORT CORRECTNESS — our ``extract_from_frames_dir`` vs Gowtham's *original*
      ``extract_features.run()`` on the *same* ffmpeg frames + same model.
      Expect bit-exact (max|Δ| ~ 0): identical math, identical inputs.

  (2) LINEAGE / FAITHFULNESS — our output vs RTFM's released
      ``_archive/UCF_Train_ten_crop_i3d/Abuse001_x264_i3d.npy``. Expect a tiny
      gap only (different ffmpeg version / JPG bytes), cosine ~1.0, L2 scale ~22.

    conda run -n balaenoptera python scripts/verify_i3d_extract.py
"""

import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
GOWTHAM = os.path.join(ROOT, ".reference", "I3D_Feature_Extraction_resnet")
sys.path.insert(0, GOWTHAM)

from src.features.i3d_gowtham import (  # noqa: E402
    build_model, extract_from_frames_dir, ffmpeg_extract_frames,
)

VIDEO = os.path.join(GOWTHAM, "samplevideos", "Abuse001_x264.mp4")
PTH = os.path.join(ROOT, "pretrained", "i3d", "i3d_baseline_r50_kinetics.pth")
RTFM = os.path.expanduser(
    "~/data/wsad/ucf_crime/_archive/UCF_Train_ten_crop_i3d/Abuse001_x264_i3d.npy"
)


class _DictAdapter(nn.Module):
    """Gowtham's run() calls ``i3d({'frames': tensor})``; our model takes a tensor."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, batch):
        return self.m(batch["frames"])


def main():
    # Gowtham's original code uses Image.ANTIALIAS (removed in Pillow >=10); it was
    # a verbatim alias for LANCZOS, so restoring it runs the official code unchanged.
    from PIL import Image
    if not hasattr(Image, "ANTIALIAS"):
        Image.ANTIALIAS = Image.Resampling.LANCZOS

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(PTH, use_nl=False, device=dev)

    with tempfile.TemporaryDirectory() as tmp:
        ffmpeg_extract_frames(VIDEO, tmp)
        n_frames = len(os.listdir(tmp))
        print(f"frames: {n_frames}")

        mine = extract_from_frames_dir(model, tmp, frequency=16, batch_size=20,
                                       sample_mode="oversample", device=dev)

        from extract_features import run as gowtham_run  # original code, unmodified
        ref = gowtham_run(_DictAdapter(model), 16, tmp, 20, "oversample")

    print(f"mine  {mine.shape}  | gowtham {ref.shape}")
    d = np.abs(mine - ref)
    print(f"(1) PORT   max|Δ|={d.max():.3e} mean|Δ|={d.mean():.3e}  "
          f"-> {'BIT-EXACT ✓' if d.max() < 1e-4 else 'MISMATCH ✗'}")

    rtfm = np.load(RTFM)
    T = min(mine.shape[0], rtfm.shape[0])
    a, b = mine[:T].reshape(T, -1), rtfm[:T].reshape(T, -1)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    rel = np.abs(mine[:T] - rtfm[:T]).mean() / (np.abs(rtfm[:T]).mean() + 1e-9)
    print(f"(2) RTFM   mine T={mine.shape[0]} vs rtfm T={rtfm.shape[0]} (cmp {T})")
    print(f"           cosine mean={cos.mean():.4f} min={cos.min():.4f} | "
          f"rel-mean-diff={rel:.4f} | L2/snip mine={np.linalg.norm(a,axis=1).mean()/np.sqrt(10):.2f} "
          f"rtfm={np.linalg.norm(b,axis=1).mean()/np.sqrt(10):.2f}")


if __name__ == "__main__":
    main()
