"""Cached-feature datasets (the training/eval hot path).

Two loaders, one output contract:
  - ``FeatureDataset`` — reads the per-video ``.npy`` members of a HF zip (the
    current I3D seg32 cache). Back-compat with the old ``src.dataset``.
  - ``ManifestFeatureDataset`` — reads per-video ``.npy`` by path from a
    :class:`~src.data.manifest.Manifest` (the modern, script-free path; this is
    how CLIP / VadCLIP-style caches are organized).

Both ``__getitem__`` return ``{feature, anomaly, event, class_id[, label]}``:
``event``/``class_id`` are the (free) fine-grained class for VadCLIP/GS-MoE; the
binary methods just use ``anomaly``. Backbone-agnostic — feature dim is whatever
the cached array has (I3D 2048, CLIP 512); models slice what they need.
"""

import json
import os
import zipfile
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset

from src.data.labels import anomaly_label, event_id, parse_event
from src.data.manifest import Manifest

DEFAULT_FEATURE_HUB = "jinmang2/ucf_crime_tencrop_i3d_seg32"
DEFAULT_FILENAMES = {"train": "train.zip", "test": "test.zip"}


def _add_magnitude(feature: np.ndarray) -> np.ndarray:
    """Append an L2-norm channel along the feature dim (2048 -> 2049)."""
    magnitude = np.linalg.norm(feature, axis=-1, keepdims=True)
    return np.concatenate((feature, magnitude), axis=-1)


def _sample_outputs(
    feature: np.ndarray, fname: str, with_magnitude: bool
) -> Dict[str, np.ndarray]:
    if with_magnitude:
        feature = _add_magnitude(feature)
    return {
        "feature": feature,
        "anomaly": np.array(0.0 if not _is_anom(fname) else 1.0, dtype=np.float32),
        "event": parse_event(fname),
        "class_id": np.array(event_id(fname), dtype=np.int64),
    }


def _is_anom(fname: str) -> bool:
    return anomaly_label(fname) == "Abnormal"


class FeatureDataset(Dataset):
    """Zip-backed cached features (current I3D seg32 cache).

    ``crop_sampling="random"`` returns **one uniformly-chosen crop** per access instead of
    all ten. That is what the official 10-crop training protocols actually do — PEL4VAD's
    ``train.list``, for instance, has 16100 entries for 1610 videos because every crop is
    its own training sample — whereas averaging the crops both departs from the recipe and
    smooths the features.

    It is also the difference between reading 8.2 MB and 0.82 MB per sample: the cached
    arrays are ``(ncrops, T, D)`` and C-contiguous, so a single crop is one contiguous
    slice, and memory-mapping means only that slice is faulted in. At batch 128 that is
    1.05 GB per step against 0.10 GB.

    Use it for **training only** — evaluation must keep averaging all ten crops, which is
    the protocol every reported number here was produced with.
    """

    def __init__(
        self,
        filenames: List[str],
        values: Dict[str, Union[zipfile.ZipInfo, np.ndarray]],
        labels: Optional[Dict[str, float]] = None,
        open_func: Optional[Callable] = None,
        with_magnitude: bool = True,
        crop_sampling: Optional[str] = None,
        time_major: bool = False,
    ):
        if crop_sampling not in (None, "random"):
            raise ValueError(f"crop_sampling must be None or 'random', got {crop_sampling!r}")
        self.filenames = filenames
        self.values = values
        self.labels = labels
        self.open_func = open_func
        self.with_magnitude = with_magnitude
        self.crop_sampling = crop_sampling
        # Cached test features are not stored consistently: the I3D test cache is
        # (T, ncrops, D) while some variants — LanguageBind, for one — keep (ncrops, T, D)
        # in both splits. Downstream code (`eval_matrix._to_crops_layout`) assumes the I3D
        # convention for anything loaded as an i3d variant, so a crops-first cache has to be
        # transposed on the way in or it silently reads T snippets as crops.
        self.time_major = time_major

    def __len__(self) -> int:
        return len(self.values)

    def open(self, value: Union[zipfile.ZipInfo, np.ndarray]) -> np.ndarray:
        if self.open_func is None:  # eager array already in RAM
            return self._pick_crop(value)
        if self.crop_sampling is None:
            return np.load(self.open_func(value))  # dynamic (lazy) load
        return self._read_one_crop(self.open_func(value))

    def _pick_crop(self, feature: np.ndarray) -> np.ndarray:
        """One random crop, kept 3-D as ``(1, T, D)`` so downstream shapes are unchanged."""
        if self.crop_sampling is None or feature.ndim != 3:
            return np.asarray(feature)
        crop = int(np.random.randint(feature.shape[0]))
        return np.array(feature[crop : crop + 1])

    def _read_one_crop(self, path: str) -> np.ndarray:
        """Read a single crop's bytes straight out of the ``.npy``, without the rest.

        Memory-mapping is not enough here: the kernel's readahead faults in most of an
        8 MB file anyway, which measured only a 19% saving. Seeking to the crop's offset
        and reading exactly its bytes gets the full 10x, because the arrays are
        ``(ncrops, T, D)`` and C-contiguous so a crop is one contiguous block.

        Falls back to a normal load for any layout this cannot address safely.
        """
        with open(path, "rb") as fh:
            major, minor = np.lib.format.read_magic(fh)
            reader = getattr(np.lib.format, f"read_array_header_{major}_{minor}", None)
            if reader is None:  # an .npy version this numpy cannot parse header-only
                fh.seek(0)
                return self._pick_crop(np.load(fh))
            shape, fortran, dtype = reader(fh)
            if fortran or len(shape) != 3 or dtype.hasobject:
                fh.seek(0)
                return self._pick_crop(np.load(fh))
            crop = int(np.random.randint(shape[0]))
            count = int(shape[1]) * int(shape[2])
            fh.seek(crop * count * dtype.itemsize, 1)
            data = np.fromfile(fh, dtype=dtype, count=count)
        return data.reshape(1, shape[1], shape[2])

    def get_filename(self, idx: int) -> str:
        return self.filenames[idx]

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        fname = self.get_filename(idx)
        feature = self.open(self.values[fname])
        if self.time_major and feature.ndim == 3:
            feature = np.ascontiguousarray(feature.transpose(1, 0, 2))  # (crops,T,D)->(T,crops,D)
        out = _sample_outputs(feature, fname, self.with_magnitude)
        if self.labels is not None:
            out["label"] = np.array(self.labels[fname], dtype=np.float32)
        return out


class ManifestFeatureDataset(Dataset):
    """Manifest-backed cached features (per-video ``.npy`` by path).

    The modern path: each record points at a ``.npy`` on disk; the class comes
    from the manifest. Use for CLIP / VideoMAE caches and any new backbone.
    """

    def __init__(
        self,
        manifest: Manifest,
        labels: Optional[Dict[str, list]] = None,
        with_magnitude: bool = False,
    ):
        self.records = list(manifest)
        self.labels = labels
        self.with_magnitude = with_magnitude

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        rec = self.records[idx]
        feature = np.load(rec.path)
        out = _sample_outputs(feature, rec.video_id, self.with_magnitude)
        if self.labels is not None:
            out["label"] = np.array(self.labels[rec.video_id], dtype=np.float32)
        return out


def _build_feature_dataset(
    filepath: str, mode: str, dynamic_load: bool
) -> Union[Dataset, Dict[str, Dataset]]:
    assert mode in ("train", "test")
    zipf = zipfile.ZipFile(filepath)

    filenames, values = [], {}
    for member in zipf.infolist():
        filename = member.filename.split("/")[-1]
        filenames.append(filename)
        values[filename] = member if dynamic_load else np.load(zipf.open(member))

    open_func = zipf.open if dynamic_load else None

    if mode == "test":
        gt_path = hf_hub_download(
            repo_id=DEFAULT_FEATURE_HUB,
            filename="ground_truth.json",
            repo_type="dataset",
            force_download=True,
        )
        gt = json.load(open(gt_path))
        return FeatureDataset(filenames, values, labels=gt, open_func=open_func)

    normal = [f for f in filenames if "Normal" in f]
    abnormal = [f for f in filenames if "Normal" not in f]
    return {
        "normal": FeatureDataset(
            normal, {f: values[f] for f in normal}, open_func=open_func
        ),
        "abnormal": FeatureDataset(
            abnormal, {f: values[f] for f in abnormal}, open_func=open_func
        ),
    }


def build_feature_dataset(
    mode: str = "train",
    local_path: Optional[str] = None,
    filename: Optional[str] = None,
    cache_dir: Optional[str] = None,
    revision: str = "main",
    dynamic_load: bool = True,
) -> Union[Dataset, Dict[str, Dataset]]:
    assert mode in ("train", "test")
    assert sum([local_path is None, filename is None]) != 1

    if local_path is None:
        filepath = hf_hub_download(
            repo_id=DEFAULT_FEATURE_HUB,
            filename=DEFAULT_FILENAMES[mode],
            cache_dir=cache_dir,
            revision=revision,
            repo_type="dataset",
        )
    else:
        filepath = os.path.join(local_path, filename)

    return _build_feature_dataset(filepath, mode, dynamic_load)
