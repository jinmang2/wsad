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
        batch_size: int = 16,
        fp16: bool = True,
        frequency: int = 16,
        segment_to: Optional[int] = None,
        sample_to: Optional[int] = None,
    ):
        """Args:
        frequency: stride between snippet starts (16 = non-overlapping, I3D-style).
        segment_to: if set, uniform mean-pool the per-snippet features to this many
            segments (32-seg convention); if ``None``, keep per-snippet (T, dim).
        sample_to: FAST seg path — extract only this many uniformly-spaced snippets
            (one per segment) instead of all-then-pool; ~10x fewer forwards on long
            clips. Output is already ``(sample_to, dim)``; overrides ``segment_to``.
        """
        from transformers import VideoMAEImageProcessor, VideoMAEModel  # lazy (heavy)

        self.device = device
        self.batch_size = batch_size
        self.fp16 = fp16 and device != "cpu"
        self.frequency = frequency
        self.segment_to = segment_to
        self.sample_to = sample_to

        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEModel.from_pretrained(model_name).eval()
        self._patch_qkv_bias(model_name)  # before .half()/.to() — see method
        self.model = self.model.to(device)
        if self.fp16:
            self.model.half()
        self.dim = int(self.model.config.hidden_size)

    def _patch_qkv_bias(self, model_name: str) -> None:
        """Restore the checkpoint's timm-style attention bias dropped by the HF loader.

        MCG-NJU VideoMAE checkpoints store attention bias as ``q_bias`` / ``v_bias``
        (with ``k_bias`` ≡ 0, the original qkv-fused convention). transformers' newer
        VideoMAE port uses split ``query/key/value.bias`` Linear layers but does NOT
        convert the old keys, so ``query.bias`` / ``value.bias`` load as ZERO — silently
        discarding a large learned bias (||q_bias|| ≈ 17). That corrupts every attention
        score and thus the features. Map ``q_bias -> query.bias``, ``v_bias -> value.bias``
        (key.bias stays 0). No-op on transformers versions that already load correctly.
        """
        import torch

        enc = self.model.encoder.layer
        a0 = enc[0].attention.attention
        if a0.query.bias is None or a0.query.bias.abs().sum() > 0:
            return  # older transformers already mapped the bias (or arch has none)

        from huggingface_hub import hf_hub_download

        try:
            sd = torch.load(hf_hub_download(model_name, "pytorch_model.bin"),
                            map_location="cpu")
        except Exception:
            from safetensors.torch import load_file
            sd = load_file(hf_hub_download(model_name, "model.safetensors"))

        patched = 0
        with torch.no_grad():
            for i, layer in enumerate(enc):
                att = layer.attention.attention
                for src, dst in (("q_bias", att.query), ("v_bias", att.value)):
                    key = f"videomae.encoder.layer.{i}.attention.attention.{src}"
                    if key in sd and dst.bias is not None:
                        dst.bias.copy_(sd[key].to(dst.bias.dtype))
                        patched += 1
        if patched:
            print(f"[videomae] restored {patched} timm q/v-bias tensors the HF loader dropped")

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
    def _encode(self, clips: List[List[Image.Image]]) -> np.ndarray:
        """One batch of clips (each = 16 PIL frames) -> ``(b, dim)`` (token mean-pool)."""
        px = self.processor(clips, return_tensors="pt")["pixel_values"].to(self.device)
        if self.fp16:
            px = px.half()
        hidden = self.model(pixel_values=px).last_hidden_state  # (b, tokens, dim)
        return hidden.float().mean(dim=1).cpu().numpy()

    @torch.no_grad()
    def extract(self, video: Union[str, List[Image.Image]]) -> np.ndarray:
        if isinstance(video, str):  # RAM-bounded streaming from the file
            feats = self.iter_snippets(video, self._encode)
        else:  # in-memory frame list (tests / pre-decoded)
            frames, clip = list(video), self.snippet_len
            if len(frames) < clip:
                frames += [frames[-1]] * (clip - len(frames))
            starts = list(range(0, len(frames) - clip + 1, self.frequency))
            clips = [frames[s : s + clip] for s in starts]
            feats = np.concatenate(
                [self._encode(clips[i : i + self.batch_size])
                 for i in range(0, len(clips), self.batch_size)], axis=0)

        if self.segment_to is not None and not self.sample_to:
            feats = self._segment(feats, self.segment_to)
        return feats
