"""VideoMAE / VideoMAEv2 ViT-B backbone — PLANNED (feature-ablation vs I3D).

A strong modern *visual* backbone (not text-aligned). Use it for a clean
feature-vs-method ablation on visual-only heads (RTFM, MGFN, Sultani, UR-DMU):
swap I3D -> VideoMAE-B, keep everything else fixed.

Differences vs I3D:
  - **unit = snippet** (16-frame clip) — same temporal unit as I3D, so the
    32-seg / full-T conventions and the loader carry over directly.
  - **dim = 768** (ViT-B). ViT-L = 1024 (tight on 6 GB), ViT-g = 1408 (won't fit).
  - **preprocess**: tubelet embedding, ImageNet-style normalize, 16x224x224 input.
    Use `transformers.VideoMAEModel` + `VideoMAEImageProcessor`; take the mean of
    patch tokens (or the pooled output) per clip.
  - **text_aligned = False** -> visual-only heads only (no VadCLIP text branch
    unless you add a learned projection / bridge into CLIP space).

Status: not implemented (deferred). Run after CLIP unlocks the VLM methods.
"""

from typing import List, Union

import numpy as np
from PIL import Image

from src.features.base import FeatureExtractor
from src.registry import FEATURE_EXTRACTORS


@FEATURE_EXTRACTORS.register("videomae")
class VideoMAEFeatureExtractor(FeatureExtractor):
    name = "videomae"
    dim = 768
    unit = "snippet"
    snippet_len = 16
    text_aligned = False
    modality = "visual"

    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = device

    def extract(self, video: Union[str, List[Image.Image]]) -> np.ndarray:
        raise NotImplementedError(
            "VideoMAE extractor is planned (see module docstring / "
            "docs/FEATURE_EXTRACTORS.md). Uses transformers.VideoMAEModel; deferred."
        )
