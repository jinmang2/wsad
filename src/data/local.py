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
import zipfile
from typing import Dict, List, Optional, Tuple

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


# ---- layout resolvers: dataset-keyed `features/{i3d,clip}` + `annotations/`,
# with legacy fallback (top-level `clip/`, dataset-root `*.zip`/`ground_truth.json`).
def _features_root(root: str, data_cfg) -> str:
    return os.path.join(_dataset_root(root, data_cfg), "features")


def _clip_dir(root: str, data_cfg, mode: str) -> str:
    new = os.path.join(_features_root(root, data_cfg), "clip", mode)
    if os.path.isdir(new):
        return new
    return os.path.join(os.path.expanduser(root), "clip", mode)  # legacy top-level


def _i3d_variant(data_cfg) -> str:
    """Feature-extraction variant subdir under ``features/`` (default ``i3d``).

    Lets the comparison matrix select a specific I3D extraction —
    ``i3d_tushar`` (HF tushar-n), ``i3d_pyvideo`` (HF main/pytorchvideo),
    ``i3d_ours`` (our Gowtham re-extract) — each a self-consistent train+test set.

    ``i3d_mgfn`` = MGFN authors' 10-crop I3D (HKU OneDrive), full-length
    ``(T, 10, 2048)``, RTFM-family scale L2~22 — COMPLETE 1610/290 but raw
    pre-seg32 (needs T->32 to feed seg32 models). Do NOT mix with ``i3d``
    (DeepMIL scale ~2.5). See features/i3d_mgfn/PROVENANCE.md.
    """
    return getattr(data_cfg, "feature_variant", None) or "i3d"


def _i3d_npy_dir(root: str, data_cfg, mode: str) -> str:
    v = _i3d_variant(data_cfg)
    new = os.path.join(_features_root(root, data_cfg), v, mode)
    if os.path.isdir(new):
        return new
    return os.path.join(os.path.expanduser(root), v, mode)  # legacy top-level


def _load_clip(
    root: str,
    mode: str,
    length: Optional[int],
    n_seg: Optional[int],
    single_crop: bool,
    data_cfg=None,
) -> Dict[str, np.ndarray]:
    d = _clip_dir(root, data_cfg, mode)
    out = {}
    for f in _list_npy(d):
        if single_crop and not _crop0(f):
            continue
        feat = np.load(os.path.join(d, f))
        out[f] = _shape_clip(feat, mode, length, n_seg)
    if not out:
        raise FileNotFoundError(f"no CLIP .npy under {d} (see docs/DATA_LOCAL.md)")
    return out


# Variants whose **test** cache is stored crops-first, ``(ncrops, T, D)``, unlike the I3D
# test cache which is ``(T, ncrops, D)``. Everything downstream assumes the I3D convention
# for i3d-backbone variants, so these are transposed at load. Confirmed by inspecting the
# arrays, not assumed: i3d_1024_seg200 test is (88, 10, 1024) while languagebind_10crop
# test is (10, 88, 768) — the same video, opposite axis order.
_CROPS_FIRST_TEST_VARIANTS = {"languagebind_10crop"}


def _identity(x):  # open_func for lazy npy-dir loading (FeatureDataset does np.load)
    return x


def _load_i3d(root: str, mode: str, data_cfg=None) -> Dict[str, str]:
    """Return ``{name: path}`` (LAZY). Eager-loading all npy into RAM OOM-kills WSL
    for big variants (i3d_mgfn_seg200 train = 25 GB > 15 GB RAM -> session crash).
    FeatureDataset(open_func=_identity) does ``np.load(path)`` per __getitem__."""
    d = _i3d_npy_dir(root, data_cfg, mode)
    out = {f: os.path.join(d, f) for f in _list_npy(d)}
    if not out:
        raise FileNotFoundError(f"no I3D .npy under {d} (see docs/DATA_LOCAL.md)")
    return out


# ---- zip-direct I3D (read the MGFN-provided {train,test}.zip in place; "B") ----
def _dataset_root(root: str, data_cfg) -> str:
    """``<root>/<dataset_dir>`` (the dir holding the zips + gt + UCFClipFeatures)."""
    sub = getattr(data_cfg, "dataset_dir", "ucf_crime")
    root = os.path.expanduser(root)
    return os.path.join(root, sub) if sub else root


def _i3d_zip_path(root: str, data_cfg, mode: str) -> Optional[str]:
    # new: <dataset>/features/<variant>/<mode>.zip ; legacy: <dataset>/<mode>.zip
    new = os.path.join(_features_root(root, data_cfg), _i3d_variant(data_cfg), f"{mode}.zip")
    if os.path.exists(new):
        return new
    legacy = os.path.join(_dataset_root(root, data_cfg), f"{mode}.zip")
    return legacy if os.path.exists(legacy) else None


def _zip_feature_values(
    path: str, dynamic: bool
) -> Tuple[List[str], Dict, Optional[callable]]:
    """``(names, values, open_func)`` from a per-video ``.npy`` zip.

    ``dynamic`` keeps ``ZipInfo`` (lazy ``np.load`` via ``open_func``); otherwise
    eagerly loads arrays into memory.
    """
    z = zipfile.ZipFile(path)
    infos = [i for i in z.infolist() if not i.is_dir() and i.filename.endswith(".npy")]
    names = [i.filename.split("/")[-1] for i in infos]
    if dynamic:
        return names, {n: i for n, i in zip(names, infos)}, z.open
    return names, {n: np.load(z.open(i)) for n, i in zip(names, infos)}, None


def has_local(root: str, backbone: str, mode: str, data_cfg=None) -> bool:
    if backbone == "clip":
        d = _clip_dir(root, data_cfg, mode)
    elif backbone == "i3d":
        d = _i3d_npy_dir(root, data_cfg, mode)
    else:
        d = os.path.join(os.path.expanduser(root), backbone, mode)
    return bool(_list_npy(d))


def has_local_i3d_zip(root: str, data_cfg, mode: str = "train") -> bool:
    return _i3d_zip_path(root, data_cfg, mode) is not None


def _build_i3d_from_zip(root: str, data_cfg):
    """Build ``({normal, abnormal}, test)`` directly from the local I3D zips."""
    dynamic = bool(getattr(data_cfg, "dynamic_load", True))
    tr_zip = _i3d_zip_path(root, data_cfg, "train")
    te_zip = _i3d_zip_path(root, data_cfg, "test")
    if tr_zip is None or te_zip is None:
        raise FileNotFoundError(
            f"missing {'train' if tr_zip is None else 'test'}.zip under "
            f"{_dataset_root(root, data_cfg)} (see docs/DATA_LOCAL.md)"
        )

    names, vals, opn = _zip_feature_values(tr_zip, dynamic)
    normal = [n for n in names if "Normal" in n]
    abnormal = [n for n in names if "Normal" not in n]
    train = {
        "normal": FeatureDataset(
            normal, {f: vals[f] for f in normal}, open_func=opn, with_magnitude=True
        ),
        "abnormal": FeatureDataset(
            abnormal, {f: vals[f] for f in abnormal}, open_func=opn, with_magnitude=True
        ),
    }

    s_names, s_vals, s_opn = _zip_feature_values(te_zip, dynamic)
    gt = _align_gt(s_names, _load_ground_truth(data_cfg))
    test = FeatureDataset(
        s_names, s_vals, labels=gt, open_func=s_opn, with_magnitude=True
    )
    return train, test


def build_datasets_local(data_cfg):
    """Build ``({normal, abnormal}, test)`` from ``~/data/wsad`` (local backbone)."""
    root = os.path.expanduser(getattr(data_cfg, "root", "~/data/wsad"))
    backbone = getattr(data_cfg, "backbone", "i3d")
    # magnitude defaults to the I3D convention; the comparison matrix overrides it
    # so visual-magnitude heads (RTFM/MGFN/...) get a magnitude channel on CLIP too.
    with_mag = bool(getattr(data_cfg, "with_magnitude", backbone == "i3d"))

    # I3D: prefer per-video npy dirs (i3d/{train,test}); else read the local
    # MGFN {train,test}.zip in place ("B": zip-direct, no extraction/duplication).
    if backbone == "i3d" and not _list_npy(_i3d_npy_dir(root, data_cfg, "train")):
        if has_local_i3d_zip(root, data_cfg, "train"):
            return _build_i3d_from_zip(root, data_cfg)
    length = getattr(data_cfg, "clip_length", 256)
    n_seg = getattr(data_cfg, "segment", None)
    single_crop = getattr(data_cfg, "single_crop", True)
    # VadCLIP trains on ALL 10 crops as separate samples (official ucf_CLIP_rgb.csv =
    # 1610 vids x 10 crops = 16100 rows); test stays single-crop. 10x crop augmentation.
    train_all_crops = bool(getattr(data_cfg, "train_all_crops", False))

    def load(mode):
        if backbone == "clip":
            sc = single_crop and not (train_all_crops and mode == "train")
            return _load_clip(root, mode, length, n_seg, sc, data_cfg)
        return _load_i3d(root, mode, data_cfg)

    # i3d npy dirs are loaded LAZILY (values = paths, np.load per __getitem__) to avoid
    # OOM-killing WSL on big variants (seg200 = 25 GB); clip values are eager arrays.
    i3d_open = _identity if backbone == "i3d" else None
    train_vals = load("train")
    names = list(train_vals)
    normal = [n for n in names if "Normal" in n]
    abnormal = [n for n in names if "Normal" not in n]
    # Optional per-head crop sampling for TRAINING only (test always averages all crops,
    # which is the protocol every reported number was produced with). See FeatureDataset.
    crop_sampling = getattr(data_cfg, "train_crop_sampling", None)
    train = {
        "normal": FeatureDataset(
            normal, {f: train_vals[f] for f in normal}, open_func=i3d_open,
            with_magnitude=with_mag, crop_sampling=crop_sampling,
        ),
        "abnormal": FeatureDataset(
            abnormal, {f: train_vals[f] for f in abnormal}, open_func=i3d_open,
            with_magnitude=with_mag, crop_sampling=crop_sampling,
        ),
    }

    test_vals = load("test")
    gt = _align_gt(list(test_vals), _load_ground_truth(data_cfg))
    test = FeatureDataset(
        list(test_vals), test_vals, labels=gt, open_func=i3d_open, with_magnitude=with_mag,
        time_major=_i3d_variant(data_cfg) in _CROPS_FIRST_TEST_VARIANTS,
    )
    return train, test


_UCF_ID_ANCHOR = "_x264"


def _bare_vid(fname: str) -> str:
    """Backbone-agnostic video id: drop the extension, any backbone tag, and a ``__<crop>``.

    Every UCF-Crime id ends at ``_x264``, so anchoring there is robust to whatever suffix a
    new backbone appends — the previous version stripped only a literal ``_i3d``, which
    meant a newly added feature set (``Abuse028_x264_languagebind.npy``) failed to match its
    ground truth and every test video was silently dropped.

    ``Abuse028_x264_i3d.npy``, ``Abuse028_x264__0.npy`` and
    ``Abuse028_x264_languagebind.npy`` all -> ``Abuse028_x264``.
    """
    b = fname[:-4] if fname.endswith(".npy") else fname
    at = b.find(_UCF_ID_ANCHOR)
    if at != -1:
        return b[: at + len(_UCF_ID_ANCHOR)]
    if b.endswith("_i3d"):  # datasets without the _x264 anchor keep the old behaviour
        b = b[: -len("_i3d")]
    return b.split("__")[0]


def _align_gt(test_keys: List[str], gt: Dict[str, list]) -> Dict[str, list]:
    """Re-key ``gt`` to the dataset's exact test filenames via the bare video id.

    Resolves the I3D-keyed (``<Vid>_i3d.npy``) ground truth against CLIP per-crop
    filenames (``<Vid>_x264__0.npy``) — the documented CLIP gt-alignment gap.
    """
    gt_by_vid = {_bare_vid(k): v for k, v in gt.items()}
    return {k: gt_by_vid[_bare_vid(k)] for k in test_keys if _bare_vid(k) in gt_by_vid}


def _local_ground_truth_path(data_cfg) -> Optional[str]:
    """Explicit ``data.ground_truth``, else ``<root>/<dataset_dir>/ground_truth.json``."""
    gt = getattr(data_cfg, "ground_truth", None)
    if gt:
        gt = os.path.expanduser(str(gt))
        if os.path.exists(gt):
            return gt
    root = getattr(data_cfg, "root", "~/data/wsad")
    ds = _dataset_root(root, data_cfg)
    for cand in (
        os.path.join(ds, "annotations", "ground_truth.json"),  # new layout
        os.path.join(ds, "ground_truth.json"),  # legacy dataset-root
    ):
        if os.path.exists(cand):
            return cand
    return None


def _load_ground_truth(data_cfg):
    """Frame-level GT (backbone-agnostic). Prefer the LOCAL ``ground_truth.json``
    (verified 290/290 aligned with the I3D test zip; keys ``<Vid>_i3d.npy``);
    fall back to the HF i3d dataset only when no local file is present."""
    import json

    path = _local_ground_truth_path(data_cfg)
    if path is None:
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
