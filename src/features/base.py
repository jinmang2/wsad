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

    def iter_snippets(self, video_path: str, on_clips) -> np.ndarray:
        """RAM-bounded streaming extraction over a video file.

        Reads only ``batch_size * snippet_len`` frames at a time from decord
        (random-access ``get_batch``) instead of materializing every PIL frame —
        a long surveillance clip can be 100k+ frames (decode-all = tens of GB RAM,
        which has OOM-killed WSL before). ``on_clips(list_of_clips) -> (b, dim)``
        runs the backbone on one batch of clips (each clip = ``snippet_len`` PIL
        frames); results are concatenated to ``(T, dim)``.

        Subclasses with the same snippet grid (videomae/xclip/internvideo) call this
        with their per-batch encode hook; ``self.frequency``/``snippet_len``/
        ``batch_size`` must be set on the instance.
        """
        import decord  # lazy

        vr = decord.VideoReader(uri=video_path)
        n = len(vr)
        clip, freq, bs = self.snippet_len, self.frequency, self.batch_size
        sample_to = getattr(self, "sample_to", None)
        if sample_to:  # extract only `sample_to` uniformly-spaced snippets (fast seg path:
            # one snippet per segment instead of all-then-pool, ~10x fewer forwards on long
            # surveillance clips). Output is already (sample_to, dim) — no segment pooling.
            import numpy as _np
            starts = sorted(set(_np.linspace(0, max(n - clip, 0), sample_to).astype(int).tolist()))
        else:
            starts = list(range(0, max(n - clip + 1, 1), freq))
        feats = []
        for i in range(0, len(starts), bs):
            bstarts = starts[i : i + bs]
            lo, hi = bstarts[0], min(bstarts[-1] + clip, n)
            win = vr.get_batch(list(range(lo, hi))).asnumpy()  # ONE contiguous read/batch
            clips = []
            for s in bstarts:
                a = win[s - lo : s - lo + clip]
                frames = [Image.fromarray(f) for f in a]
                if len(frames) < clip:  # pad the tail snippet
                    frames += [frames[-1]] * (clip - len(frames))
                clips.append(frames)
            feats.append(np.asarray(on_clips(clips)))
        return np.concatenate(feats, axis=0)

    def info(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "unit": self.unit,
            "snippet_len": self.snippet_len,
            "text_aligned": self.text_aligned,
            "modality": self.modality,
        }
