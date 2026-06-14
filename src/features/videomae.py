"""VideoMAE / VideoMAEv2 ViT-B backbone — a strong modern *visual* backbone.

Use it for a clean feature-vs-method ablation on visual-only heads (RTFM, MGFN,
Sultani, UR-DMU): swap I3D -> VideoMAE-B, keep everything else fixed.

Differences vs I3D:
  - **unit = snippet** (16-frame clip) — same temporal unit as I3D, so the
    32-seg / full-T conventions and the loader carry over directly. We chunk the
    video into non-overlapping 16-frame snippets (matching I3D's frequency=16).
  - **dim = 768** (ViT-B). ViT-L = 1024 (tight on 8 GB), ViT-g = 1408 (won't fit).
  - **preprocess**: ``VideoMAEImageProcessor`` (tubelet embedding, Kinetics-style
    normalize, 16x224x224). Per clip we mean-pool the patch tokens of
    ``VideoMAEModel.last_hidden_state`` -> one 768-d vector.
  - **text_aligned = False** -> visual-only heads only.

VideoMAEv2: pass ``model_name`` of a HF checkpoint that loads via
``transformers.VideoMAEModel`` (e.g. a v2 ViT-B export). Checkpoints published
only as OpenGVLab native weights need their own loader; ``hidden_size`` is read
from the model config so ``dim`` adapts automatically.
"""

from typing import List, Optional, Union

import numpy as np
import torch
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
        batch_size: int = 8,
        fp16: bool = True,
        frequency: int = 16,
        segment_to: Optional[int] = None,
    ):
        """Args:
        frequency: stride between snippet starts (16 = non-overlapping, I3D-style).
        segment_to: if set, uniform mean-pool the per-snippet features to this many
            segments (32-seg convention); if ``None``, keep per-snippet (T, dim).
        """
        from transformers import VideoMAEImageProcessor, VideoMAEModel  # lazy (heavy)

        self.device = device
        self.batch_size = batch_size
        self.fp16 = fp16 and device != "cpu"
        self.frequency = frequency
        self.segment_to = segment_to

        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEModel.from_pretrained(model_name).eval().to(device)
        if self.fp16:
            self.model.half()
        self.dim = int(self.model.config.hidden_size)

    def _read_frames(self, video: Union[str, List[Image.Image]]) -> List[Image.Image]:
        if isinstance(video, str):
            import decord  # lazy

            vr = decord.VideoReader(uri=video)
            return [Image.fromarray(vr[i].asnumpy()) for i in range(len(vr))]
        return video

    @staticmethod
    def _segment(feats: np.ndarray, n_seg: int) -> np.ndarray:
        out = np.zeros((n_seg, feats.shape[1]), dtype=np.float32)
        r = np.linspace(0, len(feats), n_seg + 1, dtype=int)
        for i in range(n_seg):
            lo, hi = r[i], r[i + 1]
            out[i] = feats[lo:hi].mean(0) if hi > lo else feats[min(lo, len(feats) - 1)]
        return out

    @torch.no_grad()
    def extract(self, video: Union[str, List[Image.Image]]) -> np.ndarray:
        frames = self._read_frames(video)
        n, clip = len(frames), self.snippet_len
        if n < clip:
            frames = frames + [frames[-1]] * (clip - n)
            n = clip
        # non-overlapping 16-frame snippets (I3D frequency=16 convention)
        starts = list(range(0, n - clip + 1, self.frequency))
        clips = [frames[s : s + clip] for s in starts]

        feats = []
        for i in range(0, len(clips), self.batch_size):
            batch = clips[i : i + self.batch_size]
            # processor takes a list of videos (each = list of 16 frames)
            px = self.processor(batch, return_tensors="pt")["pixel_values"].to(self.device)
            if self.fp16:
                px = px.half()
            hidden = self.model(pixel_values=px).last_hidden_state  # (b, tokens, dim)
            feats.append(hidden.float().mean(dim=1).cpu().numpy())
        feats = np.concatenate(feats, axis=0)  # (T, dim)

        if self.segment_to is not None:
            feats = self._segment(feats, self.segment_to)
        return feats
