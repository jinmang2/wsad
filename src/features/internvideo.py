"""InternVideo2 backbone — current-SOTA video encoder (OpenGVLab), via HF
`AutoModel(trust_remote_code=True)`.

⚠️ Size/feasibility: the public InternVideo2 checkpoints are large (Stage2 1B / 6B,
Chat 8B). The 1B vision tower in fp16 (~2 GB params + activations) is the only
realistic option on an 8 GB GPU and is still tight — extract with small `batch_size`
and `fp16`, or on a bigger GPU. This module imports cleanly (all heavy deps are lazy);
it only downloads/loads on first `extract()`. The exact `get_vid_fea` / preprocessing
interface differs per release, so **validate the output shape on a few videos (and the
forensic gate) before a full 1900-video run.**

`model_name` e.g. `OpenGVLab/InternVideo2-Stage2_1B-224p-f4`. `text_aligned=True`
(InternVideo2 is contrastively trained with text), so in principle it can also feed the
VLM heads — but the head text encoders are CLIP-space, so treat as visual unless you
re-align the text tower.
"""

from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from src.features.base import FeatureExtractor
from src.registry import FEATURE_EXTRACTORS

# ImageNet/CLIP-style normalization (InternVideo2 uses OpenAI-CLIP mean/std)
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


@FEATURE_EXTRACTORS.register("internvideo")
class InternVideoFeatureExtractor(FeatureExtractor):
    name = "internvideo"
    dim = 0  # read from the model on load
    unit = "snippet"
    snippet_len = 16
    text_aligned = True
    modality = "visual"

    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVideo2-Stage2_1B-224p-f4",
        device: str = "cuda",
        batch_size: int = 2,
        fp16: bool = True,
        frequency: int = 16,
        num_frames: int = 4,
        size: int = 224,
        segment_to: Optional[int] = None,
    ):
        from transformers import AutoModel  # lazy (heavy)

        self.device = device
        self.batch_size = batch_size
        self.fp16 = fp16 and device != "cpu"
        self.frequency = frequency
        self.num_frames = num_frames
        self.size = size
        self.segment_to = segment_to

        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True,
            torch_dtype=torch.float16 if self.fp16 else torch.float32,
        ).eval().to(device)
        # vid-feature method name varies across releases
        self._fea_fn = next(
            (getattr(self.model, n) for n in ("get_vid_feat", "get_vid_fea", "encode_vision")
             if hasattr(self.model, n)), None)
        if self._fea_fn is None:
            raise AttributeError(
                f"{model_name} exposes no get_vid_feat/get_vid_fea/encode_vision; "
                "inspect the model card and set the right call.")

    def _read_frames(self, video: Union[str, List[Image.Image]]) -> List[Image.Image]:
        if isinstance(video, str):
            import decord  # lazy

            vr = decord.VideoReader(uri=video)
            return [Image.fromarray(vr[i].asnumpy()) for i in range(len(vr))]
        return video

    def _clip_tensor(self, frames: List[Image.Image]) -> torch.Tensor:
        """`num_frames` uniformly sampled frames -> (T, 3, size, size) normalized."""
        import torchvision.transforms.functional as TF

        idx = np.linspace(0, len(frames) - 1, self.num_frames).round().astype(int)
        out = []
        for i in idx:
            im = frames[i].convert("RGB").resize((self.size, self.size))
            t = TF.to_tensor(im)
            out.append(TF.normalize(t, _MEAN, _STD))
        return torch.stack(out)  # (T, 3, H, W)

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
        """One batch of 16-frame clips -> ``(b, dim)`` via the InternVideo2 vid-feature fn."""
        batch = torch.stack([self._clip_tensor(c) for c in clips]).to(self.device)  # (b,T,3,H,W)
        if self.fp16:
            batch = batch.half()
        emb = self._fea_fn(batch)
        emb = emb[0] if isinstance(emb, (tuple, list)) else emb
        out = emb.float().reshape(emb.shape[0], -1).cpu().numpy()
        self.dim = out.shape[1]
        return out

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
