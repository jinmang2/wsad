"""CLIP ViT-B/16 backbone (highest-ROI extraction: unlocks the VLM methods).

CLIP features live in the **same space as the CLIP text encoder**, so they are
the only backbone here that unlocks the VLM/text-branch methods (CLIP-TSA,
VadCLIP, TPWNG, WSVAD-CLIP). I3D/VideoMAE are visually strong but NOT text-aligned.

Differences vs I3D:
  - **unit = frame** (per-frame), not 16-frame snippet. Use ``segment_to`` to mean-
    pool frames into T fixed segments (matches the 32-seg convention) or keep
    frame-level (``segment_to=None``) for newer methods.
  - **dim = 512**; **preprocess** = the CLIP image transform (224, CLIP mean/std).
  - **text_aligned = True**.

Heavy deps (``open_clip``, ``decord``) are imported lazily so this module always
imports; only ``extract()`` / ``__init__`` need them. Execution needs a GPU.
"""

from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from src.features.base import FeatureExtractor
from src.registry import FEATURE_EXTRACTORS


@FEATURE_EXTRACTORS.register("clip")
class CLIPFeatureExtractor(FeatureExtractor):
    name = "clip"
    dim = 512
    unit = "frame"
    snippet_len = 1
    text_aligned = True
    modality = "visual"

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "laion2b_s34b_b88k",
        device: str = "cuda",
        batch_size: int = 256,
        fp16: bool = True,
        segment_to: Optional[int] = 32,
    ):
        """Args:
        segment_to: if set, mean-pool the per-frame features into this many
            uniform segments (32-seg convention); if ``None``, keep frame-level.
        """
        import open_clip  # lazy

        self.device = device
        self.batch_size = batch_size
        self.fp16 = fp16 and device != "cpu"
        self.segment_to = segment_to

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval().to(device)
        if self.fp16:
            self.model.half()

    def _read_frames(self, video: Union[str, List[Image.Image]]) -> List[Image.Image]:
        if isinstance(video, str):
            import decord  # lazy

            vr = decord.VideoReader(uri=video)
            return [Image.fromarray(vr[i].asnumpy()) for i in range(len(vr))]
        return video

    @staticmethod
    def _segment(feats: np.ndarray, n_seg: int) -> np.ndarray:
        # feats: (n_frames, D) -> (n_seg, D) uniform mean-pool (Sultani convention)
        out = np.zeros((n_seg, feats.shape[1]), dtype=np.float32)
        r = np.linspace(0, len(feats), n_seg + 1, dtype=int)
        for i in range(n_seg):
            lo, hi = r[i], r[i + 1]
            out[i] = feats[lo:hi].mean(0) if hi > lo else feats[min(lo, len(feats) - 1)]
        return out

    @torch.no_grad()
    def extract(self, video: Union[str, List[Image.Image]]) -> np.ndarray:
        frames = self._read_frames(video)

        feats = []
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i : i + self.batch_size]
            tensor = torch.stack([self.preprocess(im) for im in batch]).to(self.device)
            if self.fp16:
                tensor = tensor.half()
            emb = self.model.encode_image(tensor)  # (b, 512)
            feats.append(emb.float().cpu().numpy())
        feats = np.concatenate(feats, axis=0)  # (n_frames, 512)

        if self.segment_to is not None:
            feats = self._segment(feats, self.segment_to)
        return feats
