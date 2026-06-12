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
    """Zip-backed cached features (current I3D seg32 cache)."""

    def __init__(
        self,
        filenames: List[str],
        values: Dict[str, Union[zipfile.ZipInfo, np.ndarray]],
        labels: Optional[Dict[str, float]] = None,
        open_func: Optional[Callable] = None,
        with_magnitude: bool = True,
    ):
        self.filenames = filenames
        self.values = values
        self.labels = labels
        self.open_func = open_func
        self.with_magnitude = with_magnitude

    def __len__(self) -> int:
        return len(self.values)

    def open(self, value: Union[zipfile.ZipInfo, np.ndarray]) -> np.ndarray:
        if self.open_func is None:
            return value
        return np.load(self.open_func(value))  # dynamic (lazy) load

    def get_filename(self, idx: int) -> str:
        return self.filenames[idx]

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        fname = self.get_filename(idx)
        feature = self.open(self.values[fname])
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
