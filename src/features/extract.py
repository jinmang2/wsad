"""Backbone-agnostic offline extraction core (issue #17).

One generic loop that runs any registered ``FeatureExtractor`` over a set of
videos and caches ``<stem>_<backbone>.npy``. The backbone-specific model/decord
work lives behind ``extractor.extract(path)``; this module stays dependency-light
(os, numpy) so it imports and unit-tests without decord / open_clip / a GPU.
"""

import os
from typing import Iterable, List, Mapping

import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional here

    def tqdm(x, **_):
        return x


def save_path_for(video_path: str, output_dir: str, backbone: str) -> str:
    """``<output_dir>/<video-stem>_<backbone>.npy`` (name-based cache, never a path)."""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(output_dir, f"{stem}_{backbone}.npy")


def run_extraction(
    samples: Iterable[Mapping],
    extractor,
    output_dir: str,
    backbone: str,
    *,
    skip_existing: bool = True,
    path_key: str = "video_path",
) -> List[str]:
    """Extract + cache features for each sample; return the saved paths.

    Args:
        samples: iterable of mappings, each holding a video path under ``path_key``.
        extractor: any object exposing ``extract(video_path) -> np.ndarray``.
        output_dir: cache directory (created if missing).
        backbone: name used for the ``<stem>_<backbone>.npy`` suffix.
        skip_existing: skip a video whose cache file already exists.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved: List[str] = []
    for sample in tqdm(samples):
        out = save_path_for(sample[path_key], output_dir, backbone)
        if skip_existing and os.path.exists(out):
            saved.append(out)
            continue
        feats = np.asarray(extractor.extract(sample[path_key]))
        np.save(out, feats)
        saved.append(out)
    return saved
