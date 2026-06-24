"""Per-crop DeepMIL 1024-d train set — the FAITHFUL UR-DMU/BN-WVAD training layout.

The official UR-DMU/BN-WVAD ``dataset_loader`` lists each 10-crop file as a SEPARATE
training sample (UCF_Train.list has 1610 vids x 10 crops = 16100 rows; each ``np.load``
is one ``(T, 1024)`` crop, segmented to 200). Our stacked ``i3d_1024_seg200`` instead
keeps ``(10, 200, 1024)`` per video and the model crop-AVERAGES — i.e. 1610 samples
with a weaker per-video loss instead of 16100 independent per-crop losses. That 10x data
+ per-crop gradient is the main remaining from-scratch lever (the 8 GB batch cap only got
BN-WVAD to ~0.82 / UR-DMU ~0.81).

This writes ``features/i3d_1024_seg200_pc/`` with:
  train/ = 16100 per-crop ``<vid>__<c>_i3d.npy`` of shape ``(200, 1024)`` (model sees
           dim==3 -> n=1, one loss per crop, exactly official)
  test/  = symlink to the stacked ``i3d_1024_seg200/test`` (eval stays per-video 10-crop)

    PYTHONPATH=. WSAD_DATA=~/data/wsad python scripts/build_deepmil_1024_percrop.py
"""

import glob
import os

import numpy as np

from src.data.local import segment

ROOT = os.path.expanduser(os.environ.get("WSAD_DATA", "~/data/wsad"))
RAW = os.path.join(ROOT, "ucf_crime", "features", "i3d_1024_raw", "train_npy")
STACKED = os.path.join(ROOT, "ucf_crime", "features", "i3d_1024_seg200")
DST = os.path.join(ROOT, "ucf_crime", "features", "i3d_1024_seg200_pc")
SEG = 200


def main() -> None:
    out_tr = os.path.join(DST, "train")
    os.makedirs(out_tr, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW, "**", "*.npy"), recursive=True))
    n = 0
    for f in files:
        stem = os.path.basename(f)[:-4]  # <vid>_x264 or <vid>_x264__<c>
        dst = os.path.join(out_tr, stem + "_i3d.npy")
        if os.path.exists(dst):
            n += 1
            continue
        a = np.load(f).astype(np.float32)  # (T, 1024) one crop
        np.save(dst, segment(a, SEG))  # (200, 1024)
        n += 1
        if n % 2000 == 0:
            print(f"  {n}/{len(files)}", flush=True)
    print(f"train: wrote {n} per-crop files -> {out_tr}", flush=True)

    # test: reuse the stacked per-video 10-crop set (eval crop-averages)
    test_link = os.path.join(DST, "test")
    if not os.path.exists(test_link):
        os.symlink(os.path.join(STACKED, "test"), test_link)
    print(f"test: symlinked {test_link} -> {STACKED}/test", flush=True)
    print("DONE. feature_variant=i3d_1024_seg200_pc --feature-dim 1024 (single_crop train=per-crop sample)", flush=True)


if __name__ == "__main__":
    main()
