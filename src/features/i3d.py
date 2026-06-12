"""I3D backbone adapter (the de-facto WSVAD standard, already in this repo).

Wraps the existing ``src.i3d`` model + ``src.dataset.TenCropVideoFrameDataset``
behind the ``FeatureExtractor`` ABC so the extraction script can dispatch on
``--backbone i3d`` uniformly with future backbones. Non-breaking: the legacy
monolithic path in ``scripts/extract_features.py`` keeps working unchanged.
"""

from typing import List, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.features.base import FeatureExtractor
from src.registry import FEATURE_EXTRACTORS


@FEATURE_EXTRACTORS.register("i3d")
class I3DFeatureExtractor(FeatureExtractor):
    name = "i3d"
    dim = 2048
    unit = "snippet"
    snippet_len = 16
    text_aligned = False
    modality = "visual"

    def __init__(
        self,
        model_name: str = "i3d_8x8_r50",
        device: str = "cuda",
        batch_size: int = 16,
    ):
        from src.i3d import build_i3d_feature_extractor

        self.device = device
        self.batch_size = batch_size
        self.model = build_i3d_feature_extractor(model_name=model_name)
        self.model.eval().to(device)

    @torch.no_grad()
    def extract(self, video: Union[str, List[Image.Image]]) -> np.ndarray:
        from src.data import TenCropVideoFrameDataset

        clips = TenCropVideoFrameDataset(video, frames_per_clip=self.snippet_len)
        loader = DataLoader(clips, batch_size=self.batch_size, shuffle=False)

        outputs = []
        for inputs in loader:
            # (B, ncrops, clip_len, C, H, W) -> (B, ncrops, C, clip_len, H, W)
            inputs = inputs.permute(0, 1, 3, 2, 4, 5)
            ncrops = inputs.shape[1]
            crops = [
                self.model(inputs[:, i].to(self.device)).detach().cpu().numpy()
                for i in range(ncrops)
            ]
            outputs.append(np.stack(crops, axis=1))
        # -> (T, ncrops, dim)
        return np.squeeze(np.vstack(outputs))
