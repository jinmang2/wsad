"""Modular data layer.

- ``labels``   : UCF class taxonomy + filename->event/anomaly parsing.
- ``manifest`` : script-free dataset manifest (replaces the deprecated HF loader script).
- ``features`` : cached-feature datasets (zip-backed + manifest-backed).
- ``video``    : raw-video 10-crop dataset for offline extraction.
"""

from src.data.features import (  # noqa
    DEFAULT_FEATURE_HUB,
    DEFAULT_FILENAMES,
    FeatureDataset,
    ManifestFeatureDataset,
    build_feature_dataset,
)
from src.data.labels import (  # noqa
    CLASS_TO_ID,
    NUM_CLASSES,
    UCF_CLASSES,
    event_id,
    is_anomaly,
    parse_event,
)
from src.data.manifest import Manifest, Record  # noqa
from src.data.video import (  # noqa
    BridgeType,
    TenCropVideoFrameDataset,
    TencropVideoFrameDataset,
    is_decord_available,
)
