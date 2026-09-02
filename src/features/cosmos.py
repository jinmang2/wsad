"""NVIDIA Cosmos-Embed1 — a modern video-text embedder (ViT + Q-Former).

Why this backbone is worth a slot next to I3D/VideoMAE:

  - **Video-text aligned.** ``visual_proj`` lives in the same 768-d space as
    ``get_text_embeddings`` — so unlike VideoMAE it can feed the VLM heads
    (VadCLIP / CLIP-TSA / TPWNG), not just the visual ones.
  - **No UCF-Crime in its training data.** The base checkpoints are trained on
    Kinetics-400/600/700, robotics (AgiBot, BridgeV2, RoboNet, DROID, 1X) and AV
    footage. That matters because the sibling ``Cosmos-Embed1-448p-anomaly-detection``
    IS LoRA-finetuned on Vad-Reasoning, which *contains UCF-Crime, XD-Violence,
    ShanghaiTech and UBnormal* — evaluating that variant on our test split would be
    label leakage. **Use the base checkpoints here; the anomaly-tuned one is banned.**

Protocol choices (so the cache drops into the existing loaders unchanged):
  - **snippet grid = 16 frames**, identical to I3D/VideoMAE, so ``T`` and the 32-seg /
    seg200 conventions carry over. The model itself consumes 8 frames per clip, so we
    uniformly subsample 8 of the snippet's 16 — the snippet *span* is what has to match,
    not the internal frame count.
  - ``output="proj"`` (default) → the 768-d text-aligned projection. It is
    **L2-normalized by the model**, which means the feature magnitude is constant and
    carries no information — magnitude-based heads (RTFM, MGFN) lose their signal on it.
  - ``output="cls"`` → the Q-Former pooled tokens *before* projection and normalization.
    Not text-aligned, but magnitude-preserving, which is what the magnitude heads need.

So the head↔output mapping is deliberate: ``proj`` for the text/VLM heads and the
forensic screens, ``cls`` for RTFM/MGFN-style magnitude heads.
"""

from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from src.features.base import FeatureExtractor
from src.registry import FEATURE_EXTRACTORS

# Finetuned on Vad-Reasoning, which aggregates UCF-Crime/XD-Violence/ShanghaiTech/
# UBnormal — using it on our benchmark would leak test labels.
_LEAKY_CHECKPOINTS = ("anomaly-detection",)


@FEATURE_EXTRACTORS.register("cosmos")
class CosmosEmbed1FeatureExtractor(FeatureExtractor):
    name = "cosmos"
    dim = 768
    unit = "snippet"
    snippet_len = 16
    text_aligned = True
    modality = "visual"

    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Embed1-448p",
        device: str = "cuda",
        batch_size: int = 8,
        fp16: bool = True,
        frequency: int = 16,
        model_frames: int = 8,
        output: str = "proj",
        segment_to: Optional[int] = None,
        sample_to: Optional[int] = None,
        allow_leaky_checkpoint: bool = False,
    ):
        """Args:
        model_frames: frames the model consumes per snippet (8 per the processor config).
        output: ``"proj"`` (text-aligned, L2-normalized) or ``"cls"`` (Q-Former pooled,
            magnitude-preserving). See the module docstring for which heads want which.
        segment_to: uniform mean-pool to this many segments (32-seg convention).
        sample_to: FAST seg path — extract only this many uniformly-spaced snippets.
        allow_leaky_checkpoint: opt out of the UCF-Crime leakage guard (don't).
        """
        from transformers import AutoProcessor  # lazy (heavy)

        from src.features._hf_compat import install_v4_compat

        if not allow_leaky_checkpoint and any(m in model_name for m in _LEAKY_CHECKPOINTS):
            raise ValueError(
                f"{model_name} is finetuned on Vad-Reasoning, which contains UCF-Crime — "
                "evaluating it on the UCF-Crime test split is label leakage. Use "
                "nvidia/Cosmos-Embed1-448p (or -336p/-224p) instead."
            )
        if output not in ("proj", "cls"):
            raise ValueError(f"output must be 'proj' or 'cls', got {output!r}")

        install_v4_compat()  # the vendored Q-Former/ViT are written against transformers v4

        self.device = device
        self.batch_size = batch_size
        self.fp16 = fp16 and device != "cpu"
        self.frequency = frequency
        self.model_frames = model_frames
        self.output = output
        self.segment_to = segment_to
        self.sample_to = sample_to
        self.text_aligned = output == "proj"

        self.dtype = torch.float16 if self.fp16 else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = self._load_model(model_name).to(device, dtype=self.dtype).eval()

    @staticmethod
    def _load_model(model_name: str):
        """Construct on CPU, then ``load_state_dict`` — deliberately NOT ``from_pretrained``.

        ``AutoModel.from_pretrained`` builds the model on the **meta** device and then
        materializes only the tensors the checkpoint stores. That breaks this checkpoint in
        two silent ways:

        1. ``normalization_mean`` / ``normalization_std`` are ``persistent=False`` buffers
           holding the ImageNet constants, so they are absent from the checkpoint *by
           design* and stay uninitialized — values around 1e30 go into a division and every
           feature comes out NaN.
        2. The ViT position embedding is stored at its 224 px training grid (257 tokens) and
           is meant to be bicubic-interpolated to the configured resolution (1025 tokens at
           448 px) by a ``_register_load_state_dict_pre_hook`` in the vendored ViT. Direct
           assignment never fires that hook, so the load either raises or, with
           ``ignore_mismatched_sizes=True``, leaves the position embedding at zero.

        ``from_config`` runs the real ``__init__`` on CPU (constants intact) and
        ``load_state_dict`` goes through the interpolation hook, giving a load with **zero
        missing and zero unexpected keys**. It costs one transient CPU copy of the weights.
        """
        import json

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_config(config, trust_remote_code=True)

        try:
            index = json.load(open(hf_hub_download(model_name, "model.safetensors.index.json")))
            shards = sorted(set(index["weight_map"].values()))
        except Exception:
            shards = ["model.safetensors"]
        state = {}
        for shard in shards:
            state.update(load_file(hf_hub_download(model_name, shard)))

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"{model_name} loaded with missing={missing[:5]} unexpected={unexpected[:5]} — "
                "the checkpoint layout changed; features would be silently wrong."
            )
        return model

    @staticmethod
    def _segment(feats: np.ndarray, n_seg: int) -> np.ndarray:
        out = np.zeros((n_seg, feats.shape[1]), dtype=np.float32)
        r = np.linspace(0, len(feats), n_seg + 1, dtype=int)
        for i in range(n_seg):
            lo, hi = r[i], r[i + 1]
            out[i] = feats[lo:hi].mean(0) if hi > lo else feats[min(lo, len(feats) - 1)]
        return out

    def _to_model_frames(self, clip: List[Image.Image]) -> np.ndarray:
        """One ``snippet_len``-frame snippet -> ``(model_frames, 3, H, W)`` uint8."""
        idx = np.linspace(0, len(clip) - 1, self.model_frames).round().astype(int)
        return np.stack([np.asarray(clip[i], dtype=np.uint8).transpose(2, 0, 1) for i in idx])

    @torch.no_grad()
    def _encode(self, clips: List[List[Image.Image]]) -> np.ndarray:
        videos = torch.from_numpy(np.stack([self._to_model_frames(c) for c in clips]))
        inputs = self.processor(videos=videos).to(self.device, dtype=self.dtype)
        out = self.model.get_video_embeddings(**inputs)
        feats = out.visual_proj if self.output == "proj" else out.visual_cls_tokens
        feats = feats.float().cpu().numpy()
        self.dim = int(feats.shape[-1])
        return feats

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
