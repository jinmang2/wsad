"""Backbone <-> head compatibility guard (issue #18).

Text-branch heads (VadCLIP, TPWNG) consume a CLIP-style text encoder, so they
only make sense on a **text-aligned** backbone (CLIP, InternVideo2). Visual-only
backbones (I3D, VideoMAE) have no matching text space. This guard turns an
otherwise-silent shape/space mismatch into a clear, early error — meant to be
called by the pipeline factory and the matrix harness *before* any training.

Both predicates read class attributes via the registries without instantiating
anything, so no heavy backbone deps (open_clip, transformers) are imported.
"""

import src.features  # noqa: F401  (registers FEATURE_EXTRACTORS entries)
import src.models  # noqa: F401  (registers MODELS entries)
from src.registry import FEATURE_EXTRACTORS, MODELS


def backbone_is_text_aligned(backbone: str) -> bool:
    """Whether a registered backbone produces text-aligned features."""
    return bool(getattr(FEATURE_EXTRACTORS.get(backbone), "text_aligned", False))


def head_requires_text_aligned(head: str) -> bool:
    """Whether a registered head needs a text-aligned backbone."""
    return bool(getattr(MODELS.get(head), "requires_text_aligned", False))


def is_compatible(backbone: str, head: str) -> bool:
    """True unless a text-branch head is paired with a visual-only backbone."""
    return not (head_requires_text_aligned(head) and not backbone_is_text_aligned(backbone))


def assert_compatible(backbone: str, head: str) -> None:
    """Raise ``ValueError`` if the head needs a text space the backbone lacks."""
    if not is_compatible(backbone, head):
        raise ValueError(
            f"head '{head}' requires a text-aligned backbone (e.g. clip, "
            f"internvideo2), but backbone '{backbone}' is visual-only. "
            f"Use a text-aligned backbone or a visual-only head."
        )
