"""VGGish audio backbone — PLANNED (only for XD-Violence audio-visual methods).

Needed by the AV methods (HL-Net, MACIL-SD, AVadCLIP) on XD-Violence; irrelevant
to visual-only UCF-Crime work. Implement last (plan Phase P9).

Differences vs the visual backbones:
  - **modality = audio**: requires an audio pipeline (ffmpeg/torchaudio to PCM ->
    log-mel spectrogram), not video frame decode.
  - **unit = audio** window of 0.96 s; **dim = 128**.
  - **text_aligned = False**; pairs with a visual stream at the head/fusion stage.

Status: not implemented (deferred to the XD-Violence / AV phase).
"""

from typing import List, Union

import numpy as np
from PIL import Image

from src.features.base import FeatureExtractor
from src.registry import FEATURE_EXTRACTORS


@FEATURE_EXTRACTORS.register("vggish")
class VGGishFeatureExtractor(FeatureExtractor):
    name = "vggish"
    dim = 128
    unit = "audio"
    snippet_len = 0
    text_aligned = False
    modality = "audio"

    def __init__(self, device: str = "cpu"):
        self.device = device

    def extract(self, video: Union[str, List[Image.Image]]) -> np.ndarray:
        raise NotImplementedError(
            "VGGish extractor is planned for the XD-Violence / AV phase "
            "(see docs/FEATURE_EXTRACTORS.md)."
        )
