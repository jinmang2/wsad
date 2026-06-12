"""Feature-extractor abstraction (slot 1 of the unified framework).

Feature extraction is offline and inference-only: run once, cache ``[T, D]``
(or ``[T, ncrops, D]``) per video as ``.npy``, then train light heads on the
cache. This ABC standardizes the metadata that *gates which methods apply*
(see WSAD_INTEGRATION_PLAN.md section 7.2): the ``unit`` (snippet vs frame vs
audio), the output ``dim``, and crucially ``text_aligned`` — only text-aligned
backbones (CLIP, InternVideo2) unlock the VLM/text-branch methods.

Every backbone registers itself via ``@FEATURE_EXTRACTORS.register("name")`` so
``scripts/extract_features.py --backbone <name>`` can dispatch by string.
"""

from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
from PIL import Image


class FeatureExtractor(ABC):
    """Base class for offline feature backbones.

    Subclass attributes describe the backbone so downstream code (loader,
    method-applicability checks, manifest) stays backbone-agnostic:

    Attributes:
        name: registry key, also the cache-dir / filename suffix (``*_<name>.npy``).
        dim: output feature dimension (I3D 2048, CLIP-B/16 512, VideoMAE-B 768,
            VGGish 128).
        unit: temporal unit of one feature vector — ``"snippet"`` (a window of
            ``snippet_len`` frames, e.g. I3D/VideoMAE), ``"frame"`` (per-frame,
            e.g. CLIP), or ``"audio"`` (a fixed audio window, e.g. VGGish).
        snippet_len: frames per snippet (only meaningful when ``unit=="snippet"``).
        text_aligned: whether the feature space matches a CLIP-style text encoder
            (enables the VLM text branch). False for I3D/VideoMAE/VGGish.
        modality: ``"visual"`` or ``"audio"``.
    """

    name: str = "base"
    dim: int = 0
    unit: str = "snippet"
    snippet_len: int = 16
    text_aligned: bool = False
    modality: str = "visual"

    @abstractmethod
    def extract(self, video: Union[str, List[Image.Image]]) -> np.ndarray:
        """Extract cached features for one video.

        Args:
            video: a video path or a list of PIL frames.
        Returns:
            ``np.ndarray`` of shape ``[T, ncrops, dim]`` (10-crop visual
            backbones) or ``[T, dim]`` (single-crop / audio).
        """
        raise NotImplementedError

    def info(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "unit": self.unit,
            "snippet_len": self.snippet_len,
            "text_aligned": self.text_aligned,
            "modality": self.modality,
        }
