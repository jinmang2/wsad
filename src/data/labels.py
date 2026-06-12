"""UCF-Crime label taxonomy and filename parsing.

The 14 classes (index 0 = Normal) match the old HF dataset script
(`jinmang2/ucf_crime/ucf_crime.py`): one binary ``anomaly`` label (Normal vs
Abnormal) and one fine-grained ``event`` class. The class is encoded in the file
name (``Abuse028_x264``, ``RoadAccidents001_x264``, ``Normal_Videos_003_x264``),
so it is a **free, video-level** label — recovered here without any extra
annotation. This is what VadCLIP's MIL-Align and GS-MoE's class-experts use; the
binary-only methods (Sultani/RTFM/MGFN/UR-DMU) ignore it.
"""

import os
from typing import List

# index 0 == Normal; indices 1..13 == the 13 anomaly classes (paper order).
UCF_CLASSES: List[str] = [
    "Normal",
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
]
NUM_CLASSES = len(UCF_CLASSES)
CLASS_TO_ID = {c: i for i, c in enumerate(UCF_CLASSES)}
ANOMALY_CLASSES = UCF_CLASSES[1:]

_STRIP_SUFFIXES = ("_i3d", "_clip", "_videomae", "_x264", "_x", "_mp4")
_STRIP_EXTS = (".npy", ".mp4", ".avi", ".zip")


def _basename(name: str) -> str:
    base = os.path.basename(name)
    for ext in _STRIP_EXTS:
        if base.endswith(ext):
            base = base[: -len(ext)]
    return base


def parse_event(name: str) -> str:
    """Filename/path -> event class name (one of ``UCF_CLASSES``).

    >>> parse_event("Abuse028_x264_i3d.npy")
    'Abuse'
    >>> parse_event("RoadAccidents001_x264_i3d.npy")
    'RoadAccidents'
    >>> parse_event("Normal_Videos_003_x264_i3d.npy")
    'Normal'
    """
    base = _basename(name)
    if base.lower().startswith("normal"):
        return "Normal"
    for cls in ANOMALY_CLASSES:
        if base.startswith(cls):
            return cls
    # unknown prefix -> treat as Normal (and let the caller decide to warn/raise)
    return "Normal"


def event_id(name: str) -> int:
    """Filename/path -> class id in [0, 14)."""
    return CLASS_TO_ID[parse_event(name)]


def is_anomaly(name: str) -> bool:
    """True if the file is an anomaly (event != Normal)."""
    return parse_event(name) != "Normal"


def anomaly_label(name: str) -> str:
    """Binary label string, matching the old script's ``anomaly`` field."""
    return "Abnormal" if is_anomaly(name) else "Normal"
