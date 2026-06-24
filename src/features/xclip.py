"""X-CLIP backbone — a **text-aligned VIDEO** CLIP (Microsoft, in HF Transformers).

Unlike `clip.py` (per-frame image CLIP), X-CLIP encodes a short video clip into one
512-d embedding that lives in the CLIP **text** space, with a cross-frame attention +
video-specific prompting. So it is the only *temporal* text-aligned backbone here →
it unlocks the VLM/text-branch heads (VadCLIP, CLIP-TSA, TPWNG) on motion-aware video
features instead of frame-averaged CLIP.

  - **unit = snippet** (a window of `snippet_len` frames; X-CLIP base samples 8).
  - **dim = 512**; **text_aligned = True**.
  - per snippet: `XCLIPModel.get_video_features` over `num_frames` sampled frames.

`model_name` examples: `microsoft/xclip-base-patch16` (8 frames),
`microsoft/xclip-base-patch32`, `microsoft/xclip-base-patch16-16-frames`.
`num_frames` is read from the model config so the frame sampling matches the checkpoint.
"""

from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from src.features.base import FeatureExtractor
from src.registry import FEATURE_EXTRACTORS


@FEATURE_EXTRACTORS.register("xclip")
class XCLIPFeatureExtractor(FeatureExtractor):
    name = "xclip"
    dim = 512
    unit = "snippet"
    snippet_len = 16
    text_aligned = True
    modality = "visual"

    def __init__(
        self,
        model_name: str = "microsoft/xclip-base-patch16",
        device: str = "cuda",
        batch_size: int = 8,
        fp16: bool = True,
        frequency: int = 16,
        segment_to: Optional[int] = None,
    ):
        from transformers import XCLIPModel, XCLIPProcessor  # lazy (heavy)

        self.device = device
        self.batch_size = batch_size
        self.fp16 = fp16 and device != "cpu"
        self.frequency = frequency
        self.segment_to = segment_to

        self.processor = XCLIPProcessor.from_pretrained(model_name)
        self.model = XCLIPModel.from_pretrained(model_name).eval().to(device)
        if self.fp16:
            self.model.half()
        self.dim = int(self.model.config.projection_dim)
        # frames the checkpoint expects per clip (8 or 16)
        self.num_frames = int(getattr(self.model.config.vision_config, "num_frames", 8))

    def _read_frames(self, video: Union[str, List[Image.Image]]) -> List[Image.Image]:
        if isinstance(video, str):
            import decord  # lazy

            vr = decord.VideoReader(uri=video)
            return [Image.fromarray(vr[i].asnumpy()) for i in range(len(vr))]
        return video

    @staticmethod
    def _sample(frames: List[Image.Image], k: int) -> List[Image.Image]:
        idx = np.linspace(0, len(frames) - 1, k).round().astype(int)
        return [frames[i] for i in idx]

    @staticmethod
    def _segment(feats: np.ndarray, n_seg: int) -> np.ndarray:
        out = np.zeros((n_seg, feats.shape[1]), dtype=np.float32)
        r = np.linspace(0, len(feats), n_seg + 1, dtype=int)
        for i in range(n_seg):
            lo, hi = r[i], r[i + 1]
            out[i] = feats[lo:hi].mean(0) if hi > lo else feats[min(lo, len(feats) - 1)]
        return out

    @torch.no_grad()
    def _encode(self, clips: List[List[Image.Image]]) -> np.ndarray:
        """One batch of 16-frame clips -> ``(b, 512)`` (X-CLIP video embedding)."""
        batch = [self._sample(c, self.num_frames) for c in clips]  # 16 -> num_frames
        px = self.processor(videos=batch, return_tensors="pt")["pixel_values"].to(self.device)
        if self.fp16:
            px = px.half()
        emb = self.model.get_video_features(pixel_values=px)
        if not torch.is_tensor(emb):  # transformers>=5 returns an output object
            emb = getattr(emb, "pooler_output", None)
            if emb is None:
                raise RuntimeError("XCLIP get_video_features returned no pooler_output")
        return emb.float().cpu().numpy()

    @torch.no_grad()
    def extract(self, video: Union[str, List[Image.Image]]) -> np.ndarray:
        if isinstance(video, str):
            feats = self.iter_snippets(video, self._encode)
        else:
            frames, clip = list(video), self.snippet_len
            if len(frames) < clip:
                frames += [frames[-1]] * (clip - len(frames))
            starts = list(range(0, len(frames) - clip + 1, self.frequency))
            clips = [frames[s : s + clip] for s in starts]
            feats = np.concatenate(
                [self._encode(clips[i : i + self.batch_size])
                 for i in range(0, len(clips), self.batch_size)], axis=0)

        if self.segment_to is not None:
            feats = self._segment(feats, self.segment_to)
        return feats
