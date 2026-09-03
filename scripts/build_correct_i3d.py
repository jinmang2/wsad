"""Rebuild `features/i3d/{train,test}` with the CORRECT I3D features.

Root cause (verified, see memory eval-matrix-serving §5): `features/i3d/train.zip`
(10,32,2048) is a mismatched/corrupted extraction — the official MGFN ckpt scores
its abnormal videos LOWER than normal (5% vs 100% on test). The consistent train
features live in `_archive/UCF_Train_ten_crop_i3d` (T,10,2048, official MGFN 100%);
RTFM/MGFN train from scratch to paper AUC on them (0.8448 / 0.816).

This streams (low RAM):
  - train: `_archive/UCF_Train_ten_crop_i3d/*.npy` (T,10,2048) -> mean-pool to
    32 segments -> `features/i3d/train/<name>.npy` (10,32,2048)
  - test : `features/i3d/test.zip` (already correct) -> `features/i3d/test/<name>.npy`

Non-destructive: leaves the bad `train.zip` in place; the loader prefers the
populated npy dir (src/data/local.py:_load_i3d). Idempotent (skips existing).

    PYTHONPATH=. WSAD_DATA=~/data/wsad python scripts/build_correct_i3d.py
"""

import io
import os
import zipfile

import numpy as np

ROOT = os.path.expanduser(os.environ.get("WSAD_DATA", "~/data/wsad"))
DS = os.path.join(ROOT, "ucf_crime")
ARCH_TRAIN = os.path.join(DS, "_archive", "UCF_Train_ten_crop_i3d")
TEST_ZIP = os.path.join(DS, "features", "i3d", "test.zip")
OUT_TRAIN = os.path.join(DS, "features", "i3d", "train")
OUT_TEST = os.path.join(DS, "features", "i3d", "test")


def seg32(a):  # (10, T, 2048) -> (10, 32, 2048) uniform mean-pool over T
    T = a.shape[1]
    out = np.zeros((a.shape[0], 32, a.shape[2]), np.float32)
    e = np.linspace(0, T, 33, dtype=int)
    for i in range(32):
        lo, hi = e[i], e[i + 1]
        out[:, i] = a[:, lo:hi].mean(1) if hi > lo else a[:, min(lo, T - 1)]
    return out


def build_train():
    os.makedirs(OUT_TRAIN, exist_ok=True)
    files = sorted(f for f in os.listdir(ARCH_TRAIN) if f.endswith(".npy"))
    n = 0
    for f in files:
        dst = os.path.join(OUT_TRAIN, f)
        if os.path.exists(dst):
            continue
        a = np.load(os.path.join(ARCH_TRAIN, f)).astype(np.float32)  # (T, 10, 2048)
        a = np.transpose(a, (1, 0, 2))  # (10, T, 2048)
        np.save(dst, seg32(a))  # (10, 32, 2048)
        n += 1
        if n % 200 == 0:
            print(f"  train {n}/{len(files)}", flush=True)
    print(f"train: wrote {n}, total {len(files)} -> {OUT_TRAIN}", flush=True)


def build_test():
    os.makedirs(OUT_TEST, exist_ok=True)
    z = zipfile.ZipFile(TEST_ZIP)
    infos = [i for i in z.infolist() if i.filename.endswith(".npy")]
    n = 0
    for info in infos:
        name = info.filename.split("/")[-1]
        dst = os.path.join(OUT_TEST, name)
        if os.path.exists(dst):
            continue
        a = np.load(io.BytesIO(z.read(info)))  # (T, 10, 2048) — keep as-is (test layout)
        np.save(dst, a)
        n += 1
        if n % 100 == 0:
            print(f"  test {n}/{len(infos)}", flush=True)
    print(f"test: wrote {n}, total {len(infos)} -> {OUT_TEST}", flush=True)


if __name__ == "__main__":
    print("Building CORRECT i3d features (non-destructive; bad train.zip kept)...")
    build_train()
    build_test()
    print("DONE. Loader will now use features/i3d/{train,test}/*.npy (correct).")
