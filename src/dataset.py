"""Back-compat shim — the data layer moved to ``src.data``.

Existing imports (`from src.dataset import build_feature_dataset`, etc.) keep
working. New code should import from ``src.data`` directly.
"""

from src.data.features import (  # noqa
    DEFAULT_FEATURE_HUB,
    DEFAULT_FILENAMES,
    FeatureDataset,
    ManifestFeatureDataset,
    build_feature_dataset,
)
from src.data.video import (  # noqa
    BridgeType,
    TenCropVideoFrameDataset,
    TencropVideoFrameDataset,
    is_decord_available,
)
