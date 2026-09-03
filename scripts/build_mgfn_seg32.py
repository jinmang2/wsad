"""Build a seg32 cache from the verified MGFN-authors' full-length I3D features.

`features/i3d_mgfn/` is the MGFN/RTFM-lineage 10-crop I3D (L2~22), full-length
`(T, 10, 2048)`, COMPLETE 1610/290 (byte-verified). Seg32 models (MGFN/RTFM/S3R/
Sultani) expect `(10, 32, 2048)` train; eval stays full-length `(T, 10, 2048)`.

This streams (low RAM) train -> 32-segment mean-pool -> `features/i3d_mgfn_seg32/
train/<name>.npy` (10, 32, 2048); test is symlinked to the full-length source.
Idempotent. Unlike the bad `build_correct_i3d.py` (mixed scales), train AND test
here are the SAME lineage (L2~22) -> a clean consistent pair for paper reproduction.

    PYTHONPATH=. WSAD_DATA=~/data/wsad python scripts/build_mgfn_seg32.py
"""

import os
import glob

import numpy as np

from src.data.local import segment

ROOT = os.path.expanduser(os.environ.get("WSAD_DATA", "~/data/wsad"))
SEG = int(os.environ.get("SEG", 32))  # 32 = MGFN/RTFM/S3R; 200 = UR-DMU/BN-WVAD
SRC = os.path.join(ROOT, "ucf_crime", "features", "i3d_mgfn")
DST = os.path.join(ROOT, "ucf_crime", "features", f"i3d_mgfn_seg{SEG}")


def seg32_crops(a: np.ndarray) -> np.ndarray:
    """``(T, 10, 2048)`` full-length -> ``(10, SEG, 2048)`` per-crop SEG-seg pool."""
    ac = np.transpose(a.astype(np.float32), (1, 0, 2))  # (10, T, 2048)
    return np.stack([segment(ac[c], SEG) for c in range(ac.shape[0])])  # (10, SEG, 2048)


def build_train() -> None:
    out = os.path.join(DST, "train")
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "train", "*.npy")))
    n = 0
    for f in files:
        dst = os.path.join(out, os.path.basename(f))
        if os.path.exists(dst):
            continue
        np.save(dst, seg32_crops(np.load(f)))
        n += 1
        if n % 200 == 0:
            print(f"  train {n}/{len(files)}", flush=True)
    print(f"train: wrote {n}, total {len(files)} -> {out}", flush=True)


def link_test() -> None:
    """Eval is full-length; point seg32 test at the full-length source (no copy)."""
    dst = os.path.join(DST, "test")
    src = os.path.join(SRC, "test")
    if os.path.islink(dst) or os.path.exists(dst):
        print(f"test: {dst} already present", flush=True)
        return
    os.symlink(src, dst)
    print(f"test: symlinked {dst} -> {src} (full-length, {len(os.listdir(src))} files)", flush=True)


if __name__ == "__main__":
    print(f"Building seg32 cache from {SRC} (L2~22 MGFN lineage)...", flush=True)
    build_train()
    link_test()
    print("DONE. Use with data.feature_variant=i3d_mgfn_seg32 (seg32 train, full test).", flush=True)
