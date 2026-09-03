"""HF-style end-to-end anomaly-detection pipeline (issue #20).

Bundles a feature backbone (the *processor*), a trained head, and post-process
into one serveable object: ``raw video -> per-frame anomaly scores``. No separate
offline extraction step at serve time — the extractor is absorbed, the way an HF
``pipeline`` absorbs its processor.

Internals stay decoupled (the matrix harness swaps backbone/head freely); this
factory is the bundled surface. Spec 1 implements ``mode="offline"`` (bidirectional,
exact = training); ``window``/``causal`` extend the same object in Spec 2.

Design notes:
- The backbone<->head text-alignment guard runs at construction (fail early).
- A fresh head config has its feature-dim field matched to the backbone dim, so
  ``pipeline("clip", "sultani")`` builds a 512-d head without manual config.
- ``from_features`` is the extraction-free entry (cached features / tests);
  ``__call__`` runs the full raw-video path.
"""

from typing import Optional, Union

import numpy as np
import torch

from src.compat import assert_compatible
from src.registry import FEATURE_EXTRACTORS, MODELS

# config field that carries the raw (pre-magnitude) feature dim, per head family
_DIM_FIELDS = ("feature_size", "channels", "visual_width")


def _set_feature_dim(config, dim: int) -> None:
    for f in _DIM_FIELDS:
        if hasattr(config, f):
            setattr(config, f, dim)


class AnomalyDetectionPipeline:
    """``raw video -> per-frame anomaly scores`` for a (backbone, head) pair."""

    def __init__(
        self,
        backbone: str,
        head: Union[str, "torch.nn.Module"],
        *,
        head_ckpt: Optional[str] = None,
        model: Optional["torch.nn.Module"] = None,
        device: str = "cpu",
        with_magnitude: bool = True,
        segment_to: Optional[int] = None,
        extractor=None,
        extractor_kwargs: Optional[dict] = None,
    ):
        self.backbone = backbone
        self.device = device
        self.with_magnitude = with_magnitude
        self.segment_to = segment_to
        self._extractor = extractor
        self._extractor_kwargs = extractor_kwargs or {}

        # fail early on a text-branch head + visual-only backbone
        if isinstance(head, str):
            assert_compatible(backbone, head)

        if model is not None:
            self.model = model
        elif isinstance(head, str):
            cls = MODELS.get(head)
            if head_ckpt is not None:
                self.model = cls.from_pretrained(head_ckpt)
            else:
                cfg = cls.config_class()
                dim = getattr(FEATURE_EXTRACTORS.get(backbone), "dim", None)
                if dim:
                    _set_feature_dim(cfg, dim)
                self.model = cls(cfg)
        else:  # a ready nn.Module head instance
            self.model = head
        self.model.eval().to(device)

    @property
    def extractor(self):
        if self._extractor is None:
            from src.features import build_extractor

            self._extractor = build_extractor(
                self.backbone, device=self.device, **self._extractor_kwargs
            )
        return self._extractor

    def _preprocess(self, feats: np.ndarray) -> torch.Tensor:
        """``(T, D)`` or ``(T, ncrops, D)`` -> ``(1, ncrops, T', D'+mag)`` tensor."""
        from src.data.features import _add_magnitude
        from src.data.local import segment

        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim == 2:  # single-crop (T, D)
            feats = feats[:, None, :]
        feats = np.transpose(feats, (1, 0, 2))  # (ncrops, T, D)

        crops = []
        for crop in feats:
            if self.segment_to is not None:
                crop = segment(crop, self.segment_to)
            if self.with_magnitude:
                crop = _add_magnitude(crop)
            crops.append(crop.astype(np.float32))
        arr = np.stack(crops, axis=0)  # (ncrops, T', D')
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def from_features(self, feats: np.ndarray) -> np.ndarray:
        """Run the head on already-extracted features -> ``(T',)`` scores."""
        x = self._preprocess(feats)
        out = self.model(video=x)
        return out.scores.squeeze(0).squeeze(-1).cpu().numpy()

    @torch.no_grad()
    def __call__(self, video) -> np.ndarray:
        """Full path: ``raw video -> (T',) per-index anomaly scores``."""
        return self.from_features(self.extractor.extract(video))

    def visualize(self, video_or_scores, gt=None, **kwargs):
        """Run (if needed) and plot the score curve with GT shading."""
        from src.viz import plot_anomaly_scores

        scores = (
            video_or_scores
            if isinstance(video_or_scores, np.ndarray)
            else self(video_or_scores)
        )
        return plot_anomaly_scores(scores, gt=gt, **kwargs)
