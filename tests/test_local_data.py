"""Local backbone-aware loading + CLIP snippet processing (synthetic, offline)."""

import numpy as np

from src.data import local


def test_process_feat_long_and_short():
    long = np.random.randn(500, 512).astype(np.float32)
    assert local.process_feat(long, 256).shape == (256, 512)
    short = np.random.randn(40, 512).astype(np.float32)
    out = local.process_feat(short, 256)
    assert out.shape == (256, 512)
    assert np.allclose(out[40:], 0)  # padded tail


def test_segment_uniform_pool():
    feat = np.random.randn(100, 512).astype(np.float32)
    assert local.segment(feat, 32).shape == (32, 512)


def test_crop0_filter():
    assert local._crop0("Abuse001_x264__0.npy")
    assert not local._crop0("Abuse001_x264__5.npy")
    assert local._crop0("Abuse001_x264.npy")  # no crop suffix -> keep


def test_load_clip_single_crop_shapes(tmp_path):
    d = tmp_path / "clip" / "train"
    d.mkdir(parents=True)
    for crop in range(3):
        np.save(d / f"Abuse001_x264__{crop}.npy", np.random.randn(120, 512).astype(np.float32))
    np.save(d / "Normal_Videos_005_x264__0.npy", np.random.randn(80, 512).astype(np.float32))

    # vadclip contract: single-crop, len 256
    vals = local._load_clip(str(tmp_path), "train", length=256, n_seg=None, single_crop=True)
    assert set(vals) == {"Abuse001_x264__0.npy", "Normal_Videos_005_x264__0.npy"}  # __1/__2 dropped
    assert vals["Abuse001_x264__0.npy"].shape == (1, 256, 512)

    # tpwng/clip_tsa contract: single-crop, 32-seg
    seg = local._load_clip(str(tmp_path), "train", length=None, n_seg=32, single_crop=True)
    assert seg["Abuse001_x264__0.npy"].shape == (1, 32, 512)


def test_load_clip_test_full_length(tmp_path):
    d = tmp_path / "clip" / "test"
    d.mkdir(parents=True)
    np.save(d / "Abuse028_x264__0.npy", np.random.randn(73, 512).astype(np.float32))
    vals = local._load_clip(str(tmp_path), "test", length=256, n_seg=None, single_crop=True)
    assert vals["Abuse028_x264__0.npy"].shape == (1, 73, 512)  # full-length kept
