"""Single-crop training samples (`crop_sampling="random"`).

The official 10-crop protocols treat each crop as its own training sample — PEL4VAD's
`train.list` has 16100 entries for 1610 videos — so averaging the crops departs from the
recipe. Reading one crop is also 3x less disk per sample (measured 1.29 vs 3.83 MB).

The crop is read by seeking to its byte offset inside the `.npy` rather than loading the
array, so the tests below check **exactness** above all: a wrong offset would return
plausible-looking numbers from the wrong place in the file and nothing would complain.
"""

import numpy as np
import pytest

from src.data.features import FeatureDataset


def _write(tmp_path, array, name="Abuse001_x264_i3d.npy", **kw):
    path = tmp_path / name
    np.save(path, array, **kw)
    return {name: str(path)}, name


def _identity(x):
    return x


def test_sampled_crop_is_byte_exact_and_covers_every_crop(tmp_path):
    rng = np.random.default_rng(0)
    array = rng.standard_normal((10, 37, 64)).astype(np.float32)
    vals, name = _write(tmp_path, array)

    ds = FeatureDataset([name], vals, open_func=_identity, with_magnitude=False,
                        crop_sampling="random")
    seen = set()
    for seed in range(60):
        np.random.seed(seed)
        got = ds[0]["feature"]
        assert got.shape == (1, 37, 64)
        matches = [c for c in range(10) if np.array_equal(got[0], array[c])]
        assert len(matches) == 1, "read did not land exactly on one real crop"
        seen.add(matches[0])
    assert seen == set(range(10)), f"only drew crops {sorted(seen)}"


def test_default_still_returns_every_crop(tmp_path):
    array = np.random.default_rng(1).standard_normal((10, 12, 8)).astype(np.float32)
    vals, name = _write(tmp_path, array)
    ds = FeatureDataset([name], vals, open_func=_identity, with_magnitude=False)
    assert np.array_equal(ds[0]["feature"], array)


def test_fortran_order_falls_back_instead_of_reading_garbage(tmp_path):
    """The offset arithmetic assumes C order; a Fortran-ordered file must not be
    misread."""
    array = np.asfortranarray(np.random.default_rng(2).standard_normal((4, 9, 5)).astype(np.float32))
    vals, name = _write(tmp_path, array)
    ds = FeatureDataset([name], vals, open_func=_identity, with_magnitude=False,
                        crop_sampling="random")
    got = ds[0]["feature"]
    assert got.shape == (1, 9, 5)
    assert any(np.array_equal(got[0], array[c]) for c in range(4))


def test_non_3d_arrays_are_left_alone(tmp_path):
    array = np.random.default_rng(3).standard_normal((16, 32)).astype(np.float32)
    vals, name = _write(tmp_path, array)
    ds = FeatureDataset([name], vals, open_func=_identity, with_magnitude=False,
                        crop_sampling="random")
    assert np.array_equal(ds[0]["feature"], array)


def test_unknown_sampling_mode_is_rejected():
    with pytest.raises(ValueError, match="crop_sampling"):
        FeatureDataset([], {}, crop_sampling="middle")


def test_eager_arrays_also_honour_the_setting():
    """CLIP features are held in RAM, so there is no I/O win — but the sampling semantics
    must still match, or a head would train on different data depending on the backbone."""
    array = np.random.default_rng(4).standard_normal((10, 6, 3)).astype(np.float32)
    name = "Abuse002_x264_clip.npy"
    ds = FeatureDataset([name], {name: array}, with_magnitude=False, crop_sampling="random")
    got = ds[0]["feature"]
    assert got.shape == (1, 6, 3)
    assert any(np.array_equal(got[0], array[c]) for c in range(10))
