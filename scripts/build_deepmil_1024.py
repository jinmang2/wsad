"""Stack DeepMIL UCF per-crop 1024-d I3D into per-video (10, seg, 1024) for UR-DMU/BN-WVAD.

The DeepMIL release (Roc-Ng, the 1024-d Carreira/pytorch-i3d features UR-DMU/BN-WVAD
actually use — NOT the 2048-d ResNet50-I3D of RTFM/MGFN) ships per-crop full-length
files: ``<vid>_x264.npy`` (crop 0) + ``<vid>_x264__1..9.npy`` (crops 1-9), each
``(T, 1024)``. This stacks the 10 crops -> ``(10, T, 1024)``, segments train to 200
(uniform, = UR-DMU's np.linspace "random_perturb"), keeps test full-length, and writes
``features/i3d_1024_seg200/{train,test}/<vid>_x264_i3d.npy``.

    PYTHONPATH=. WSAD_DATA=~/data/wsad python scripts/build_deepmil_1024.py
"""

import glob
import os

import numpy as np

from src.data.local import segment

ROOT = os.path.expanduser(os.environ.get("WSAD_DATA", "~/data/wsad"))
RAW = os.path.join(ROOT, "ucf_crime", "features", "i3d_1024_raw")
DST = os.path.join(ROOT, "ucf_crime", "features", "i3d_1024_seg200")
SEG = 200


def crop_files(stem_dir: str, stem: str):
    """The 10 per-crop files for one video, in crop order 0..9."""
    return [os.path.join(stem_dir, stem + sfx) for sfx in
            [".npy"] + [f"__{c}.npy" for c in range(1, 10)]]


def stack_video(stem_dir: str, stem: str) -> np.ndarray:
    """10 per-crop ``(T, 1024)`` -> ``(10, T, 1024)`` (min-clip T for safety)."""
    arrs = [np.load(f).astype(np.float32) for f in crop_files(stem_dir, stem)]
    t = min(a.shape[0] for a in arrs)
    return np.stack([a[:t] for a in arrs])  # (10, T, 1024)


def build(split: str, src_dir: str, do_seg: bool) -> None:
    out = os.path.join(DST, split)
    os.makedirs(out, exist_ok=True)
    # crop-0 files (no __c suffix) enumerate the videos; index stem->dir in ONE walk
    stem_dir = {os.path.basename(f)[:-4]: os.path.dirname(f)
                for f in glob.glob(os.path.join(src_dir, "**", "*_x264.npy"), recursive=True)}
    stems = sorted(stem_dir)
    n = 0
    for stem in stems:
        d = stem_dir[stem]
        dst = os.path.join(out, stem + "_i3d.npy")
        if os.path.exists(dst):
            continue
        v = stack_video(d, stem)  # (10, T, 1024)
        if do_seg:  # train: (10, 200, 1024) == i3d_mgfn train layout (crops, seg, D)
            v = np.stack([segment(v[c], SEG) for c in range(v.shape[0])])
        else:  # test: full-length, transpose to (T, 10, 1024) == i3d test layout
            v = np.transpose(v, (1, 0, 2))
        np.save(dst, v)
        n += 1
        if n % 200 == 0:
            print(f"  {split} {n}/{len(stems)}", flush=True)
    print(f"{split}: wrote {n}, total {len(stems)} -> {out}", flush=True)


if __name__ == "__main__":
    print(f"Building i3d_1024_seg200 from {RAW} (DeepMIL 1024-d, UR-DMU/BN-WVAD)...", flush=True)
    build("train", os.path.join(RAW, "train_npy"), do_seg=True)
    build("test", os.path.join(RAW, "test_npy"), do_seg=False)
    print("DONE. Use data.feature_variant=i3d_1024_seg200 + feature dim 1024.", flush=True)
