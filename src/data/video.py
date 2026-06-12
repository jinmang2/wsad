"""Raw-video dataset for offline feature extraction (decord + 10-crop).

Used only by the extractors (`scripts/extract_features.py`, `src/features/*`),
not at train time. Decodes a video into ``clip_length``-frame clips and applies
the 10-crop transform. Kept separate from the cached-feature loaders so the heavy
``decord`` dependency is only imported when actually extracting.
"""

from enum import Enum
from pathlib import Path
from typing import Union

import importlib

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.gtransforms import get_ten_crop_transforms


class BridgeType(Enum):
    PILLOW = "pillow"
    PYTORCH = "torch"


def is_decord_available() -> bool:
    return importlib.util.find_spec("decord") is not None


class TencropVideoFrameDataset(Dataset):
    def __init__(
        self,
        video_path: Union[str, Path],
        bridge_type: Union[str, BridgeType] = "pillow",
        clip_length: int = 16,
        **transform_kwargs,
    ):
        super().__init__()
        if not isinstance(bridge_type, BridgeType):
            bridge_type = BridgeType(bridge_type)
        self.bridge_type = bridge_type

        if not is_decord_available():
            raise ImportError("To support decoding videos, please install `decord`.")
        import decord

        bridge = "torch" if bridge_type == BridgeType.PYTORCH else "native"
        decord.bridge.set_bridge(bridge)

        self.video_reader = decord.VideoReader(uri=video_path)
        self.n_frames = len(self.video_reader)
        self.clip_length = clip_length
        self.transform = get_ten_crop_transforms(
            bridge_type=bridge_type.value,
            clip_length=clip_length,
            **transform_kwargs,
        )

    def __len__(self) -> int:
        return (self.n_frames - 1) // self.clip_length + 1

    def __getitem__(self, idx: int) -> torch.Tensor:
        start_idx, end_idx = idx * self.clip_length, (idx + 1) * self.clip_length
        indices = range(start_idx, min(self.n_frames, end_idx))
        clip = self.video_reader.get_batch(indices)
        if self.bridge_type == BridgeType.PILLOW:
            clip = list(map(Image.fromarray, clip.asnumpy()))
        return self.transform(clip)


# Alias: both spellings are used across the codebase / extractors.
TenCropVideoFrameDataset = TencropVideoFrameDataset
