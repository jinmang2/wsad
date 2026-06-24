"""Offline feature backbones (slot 1).

Importing this package registers every backbone in ``FEATURE_EXTRACTORS``.
``i3d`` is implemented; ``clip`` / ``videomae`` / ``vggish`` are planned stubs
(see ``docs/FEATURE_EXTRACTORS.md``). Heavy deps (open_clip, torchaudio) are
imported lazily inside each extractor so this import never fails.
"""

from src.features.base import FeatureExtractor  # noqa
from src.registry import FEATURE_EXTRACTORS  # noqa

from . import clip, i3d, videomae, vggish, xclip, internvideo  # noqa  (trigger registration)


def build_extractor(name: str, **kwargs) -> FeatureExtractor:
    """Construct a registered feature extractor by name."""
    return FEATURE_EXTRACTORS.get(name)(**kwargs)
