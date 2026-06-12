"""Local-first, backbone-aware feature loading (``~/data/wsad``).

One download per backbone, three CLIP contracts. Reuses :class:`FeatureDataset`
(so the trainer collate / runner contract are unchanged); only the array shaping
differs per backbone:

  - ``i3d``  : per-video ``.npy`` (10, 32, 2048) train / (T, 10, 2048) test, +mag.
  - ``clip`` : per-crop ``.npy`` (Tframes, 512); single-crop selected, then
    ``process_feat`` (len 256, VadCLIP) or uniform ``segment`` (32-seg), no mag.

See ``docs/DATA_LOCAL.md``. ``build_datasets_local`` returns the same
``(train_dict, test_dataset)`` shape as ``src.trainer.build_datasets``.
"""

import os
from typing import Dict, List, Optional

import numpy as np

from src.data.features import FeatureDataset


# ---- CLIP snippet processing (faithful to VadCLIP utils/tools.py) ----
def uniform_extract(feat: np.ndarray, length: int) -> np.ndarray:
    """Uniformly sample ``length`` rows from ``(T, D)`` (VadCLIP uniform_extract)."""
    r = np.linspace(0, len(feat) - 1, length, dtype=np.int64)
    return feat[r, :]


def pad(feat: np.ndarray, length: int) -> np.ndarray:
    if len(feat) >= length:
        return feat[:length]
    out = np.zeros((length, feat.shape[1]), dtype=feat.dtype)
    out[: len(feat)] = feat
    return out


def process_feat(feat: np.ndarray, length: int) -> np.ndarray:
    """VadCLIP train: uniform-extract if longer than ``length`` else pad."""
    return uniform_extract(feat, length) if len(feat) > length else pad(feat, length)


def segment(feat: np.ndarray, n_seg: int) -> np.ndarray:
    """Uniform mean-pool ``(T, D) -> (n_seg, D)`` (Sultani/RTFM 32-seg convention)."""
    out = np.zeros((n_seg, feat.shape[1]), dtype=np.float32)
    edges = np.linspace(0, len(feat), n_seg + 1, dtype=int)
    for i in range(n_seg):
        lo, hi = edges[i], edges[i + 1]
        out[i] = feat[lo:hi].mean(0) if hi > lo else feat[min(lo, len(feat) - 1)]
    return out


def _shape_clip(feat: np.ndarray, mode: str, length: Optional[int], n_seg: Optional[int]):
    """``(T, 512) -> (ncrops=1, L, 512)`` per the runner's CLIP contract."""
    if n_seg is not None:
        arr = segment(feat, n_seg)  # tpwng / clip_tsa: 32-seg
    elif mode == "train" and length is not None:
        arr = process_feat(feat, length)  # vadclip: len 256
    else:
        arr = feat.astype(np.float32)  # test full-length (vadclip handles any T)
    return arr[None, :, :]  # add the single-crop dim -> (1, L, 512)


def _crop0(fname: str) -> bool:
    """Keep only crop ``__0`` (official single-crop test; train aug uses one too)."""
    base = fname[:-4] if fname.endswith(".npy") else fname
    return ("__" not in base) or base.endswith("__0")


def _list_npy(d: str) -> List[str]:
    return sorted(f for f in os.listdir(d) if f.endswith(".npy")) if os.path.isdir(d) else []


def _load_clip(
    root: str, mode: str, length: Optional[int], n_seg: Optional[int], single_crop: bool
) -> Dict[str, np.ndarray]:
    d = os.path.join(root, "clip", mode)
    out = {}
    for f in _list_npy(d):
        if single_crop and not _crop0(f):
            continue
        feat = np.load(os.path.join(d, f))
        out[f] = _shape_clip(feat, mode, length, n_seg)
    if not out:
        raise FileNotFoundError(f"no CLIP .npy under {d} (see docs/DATA_LOCAL.md)")
    return out


def _load_i3d(root: str, mode: str) -> Dict[str, np.ndarray]:
    d = os.path.join(root, "i3d", mode)
    out = {f: np.load(os.path.join(d, f)) for f in _list_npy(d)}
    if not out:
        raise FileNotFoundError(f"no I3D .npy under {d} (see docs/DATA_LOCAL.md)")
    return out


def has_local(root: str, backbone: str, mode: str) -> bool:
    return bool(_list_npy(os.path.join(os.path.expanduser(root), backbone, mode)))


def build_datasets_local(data_cfg):
    """Build ``({normal, abnormal}, test)`` from ``~/data/wsad`` (local backbone)."""
    root = os.path.expanduser(getattr(data_cfg, "root", "~/data/wsad"))
    backbone = getattr(data_cfg, "backbone", "i3d")
    with_mag = backbone == "i3d"
    length = getattr(data_cfg, "clip_length", 256)
    n_seg = getattr(data_cfg, "segment", None)
    single_crop = getattr(data_cfg, "single_crop", True)

    def load(mode):
        if backbone == "clip":
            return _load_clip(root, mode, length, n_seg, single_crop)
        return _load_i3d(root, mode)

    train_vals = load("train")
    names = list(train_vals)
    normal = [n for n in names if "Normal" in n]
    abnormal = [n for n in names if "Normal" not in n]
    train = {
        "normal": FeatureDataset(
            normal, {f: train_vals[f] for f in normal}, with_magnitude=with_mag
        ),
        "abnormal": FeatureDataset(
            abnormal, {f: train_vals[f] for f in abnormal}, with_magnitude=with_mag
        ),
    }

    test_vals = load("test")
    gt = _load_ground_truth(data_cfg)
    test = FeatureDataset(
        list(test_vals), test_vals, labels=gt, with_magnitude=with_mag
    )
    return train, test


def _load_ground_truth(data_cfg):
    """Frame-level GT (backbone-agnostic); reuse the HF i3d ``ground_truth.json``."""
    import json

    from huggingface_hub import hf_hub_download

    from src.data.features import DEFAULT_FEATURE_HUB

    path = hf_hub_download(
        repo_id=DEFAULT_FEATURE_HUB,
        filename="ground_truth.json",
        repo_type="dataset",
        cache_dir=getattr(data_cfg, "cache_dir", None),
    )
    # keys may carry an extension/suffix; FeatureDataset looks up by exact fname,
    # so normalize both sides to the bare video id at lookup time is the caller's job.
    return json.load(open(path))
