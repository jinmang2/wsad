"""Offline tests for the data layer (label parsing + manifest); no downloads."""

import os
import tempfile

from src.data import UCF_CLASSES, Manifest, event_id, is_anomaly, parse_event
from src.data.manifest import Record


def test_event_parsing():
    cases = {
        "Abuse028_x264_i3d.npy": "Abuse",
        "RoadAccidents001_x264_i3d.npy": "RoadAccidents",
        "Normal_Videos_003_x264_i3d.npy": "Normal",
        "Shoplifting015_x264_clip.npy": "Shoplifting",
        "Shooting048_x264.mp4": "Shooting",
    }
    for name, expected in cases.items():
        assert parse_event(name) == expected, name
        assert is_anomaly(name) == (expected != "Normal")
        assert event_id(name) == UCF_CLASSES.index(expected)


def test_shooting_vs_shoplifting_prefix():
    # both start with "Sho" — full-prefix match must disambiguate
    assert parse_event("Shooting001_i3d.npy") == "Shooting"
    assert parse_event("Shoplifting001_i3d.npy") == "Shoplifting"


def test_manifest_from_filenames():
    names = ["Abuse028_x264_i3d.npy", "Normal_Videos_003_x264_i3d.npy"]
    m = Manifest.from_filenames(names, split="train")
    assert len(m) == 2
    assert m[0].event == "Abuse" and m[0].is_anomaly
    assert m[1].event == "Normal" and not m[1].is_anomaly
    assert m[0].class_id == UCF_CLASSES.index("Abuse")

    only_anom = m.filter(anomaly=True)
    assert len(only_anom) == 1 and only_anom[0].event == "Abuse"


def test_manifest_jsonl_roundtrip():
    m = Manifest(
        [Record("v1", "/x/Abuse001_i3d.npy", "Abuse", "Abnormal", "train", 10)]
    )
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.jsonl")
        m.to_jsonl(p)
        m2 = Manifest.from_jsonl(p)
    assert len(m2) == 1
    assert m2[0].event == "Abuse" and m2[0].class_id == UCF_CLASSES.index("Abuse")
