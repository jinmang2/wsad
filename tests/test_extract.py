"""Backbone-agnostic extraction core (issue #17).

Tests the generic loop with a fake extractor — no decord/GPU/models needed.
"""

import os

import numpy as np

from src.features.extract import run_extraction, save_path_for


class _FakeExtractor:
    def __init__(self):
        self.calls = 0

    def extract(self, path):
        self.calls += 1
        return np.zeros((4, 8), dtype=np.float32)


def test_save_path_for():
    assert (
        save_path_for("/v/Robbery001_x264.mp4", "/out", "clip")
        == "/out/Robbery001_x264_clip.npy"
    )


def test_run_extraction_saves(tmp_path):
    ex = _FakeExtractor()
    samples = [{"video_path": "/v/A.mp4"}, {"video_path": "/v/B.mp4"}]
    saved = run_extraction(samples, ex, str(tmp_path), "videomae")
    assert len(saved) == 2
    assert all(os.path.exists(p) for p in saved)
    assert saved[0].endswith("A_videomae.npy")
    assert np.load(saved[0]).shape == (4, 8)
    assert ex.calls == 2


def test_run_extraction_skips_existing(tmp_path):
    ex = _FakeExtractor()
    samples = [{"video_path": "/v/A.mp4"}]
    run_extraction(samples, ex, str(tmp_path), "i3d")
    run_extraction(samples, ex, str(tmp_path), "i3d")  # second call must skip
    assert ex.calls == 1
