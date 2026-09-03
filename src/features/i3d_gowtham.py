"""Gowtham/RTFM-faithful I3D feature extraction (the standard WSVAD I3D features).

A faithful port of GowthamGottimukkala/I3D_Feature_Extraction_resnet
(`extract_features.py` + `main.py`): ffmpeg -> per-frame JPG -> PIL resize
340x256 (ANTIALIAS) -> ``(x*2/255)-1`` -> exact 10-crop oversample -> I3Res50 ->
``(T, 10, 2048)`` per video. This is the lineage that produced RTFM's
``UCF_*_ten_crop_i3d`` (feature L2/snippet ~22), verified by
``scripts/verify_i3d_extract.py``.

Distinct from :mod:`src.features.i3d` (the pytorchvideo ``i3d_8x8_r50`` path,
which uses Kinetics 0.45/0.225 normalization and torchvision TenCrop — a different
preprocessing, NOT scale-compatible with the MGFN/RTFM features).

Weights: convert the facebookresearch Caffe2 blobs first::

    python scripts/convert_i3d_caffe2.py pretrained/i3d/i3d_baseline_32x2_IN_pretrain_400k.pkl \\
        pretrained/i3d/i3d_baseline_r50_kinetics.pth --no-nl
"""

import os
import subprocess
import tempfile

import numpy as np
import torch
from PIL import Image

from src.i3d import I3Res50

# Pillow >=10 removed Image.ANTIALIAS (it was an alias for LANCZOS).
_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS", getattr(Image, "ANTIALIAS", 1))

CHUNK_SIZE = 16  # frames per snippet (Gowtham chunk_size, fixed)


def build_model(pretrained_path: str, use_nl: bool, device: str = "cuda") -> I3Res50:
    """``I3Res50(use_nl)`` with converted Caffe2 weights, eval mode (BN frozen)."""
    model = I3Res50(use_nl=use_nl)
    sd = torch.load(pretrained_path, map_location="cpu")
    model.load_state_dict(sd, strict=False)  # our model has no fc/drop head
    return model.eval().to(device)


def load_frame(frame_file: str) -> np.ndarray:
    """PIL open -> resize (W=340, H=256) ANTIALIAS -> ``(x*2/255)-1`` in [-1,1]."""
    data = Image.open(frame_file).resize((340, 256), _LANCZOS)
    data = np.array(data).astype(float)
    data = (data * 2 / 255) - 1
    assert data.max() <= 1.0 and data.min() >= -1.0
    return data


def load_rgb_batch(frames_dir, rgb_files, frame_indices) -> np.ndarray:
    batch = np.zeros(frame_indices.shape + (256, 340, 3))
    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):
            batch[i, j] = load_frame(os.path.join(frames_dir, rgb_files[frame_indices[i][j]]))
    return batch


def oversample_data(data: np.ndarray):
    """Exact Gowtham 10-crop: 5 spatial crops (of the 256x340 frame) + h-flips."""
    flip = np.array(data[:, :, :, ::-1, :])
    crops = lambda d: [
        np.array(d[:, :, :224, :224, :]),     # top-left
        np.array(d[:, :, :224, -224:, :]),    # top-right
        np.array(d[:, :, 16:240, 58:282, :]), # center
        np.array(d[:, :, -224:, :224, :]),    # bottom-left
        np.array(d[:, :, -224:, -224:, :]),   # bottom-right
    ]
    return crops(data) + crops(flip)


@torch.no_grad()
def extract_from_frames_dir(
    model, frames_dir, frequency=16, batch_size=20, sample_mode="oversample", device="cuda"
) -> np.ndarray:
    """Faithful port of Gowtham ``run()`` -> ``(num_chunks, 10 or 1, 2048)``."""
    assert sample_mode in ("oversample", "center_crop")

    def forward_batch(b_data):
        b_data = b_data.transpose([0, 4, 1, 2, 3])  # b,t,h,w,c -> b,c,t,h,w
        t = torch.from_numpy(b_data).to(device).float()
        return model(t).cpu().numpy()

    rgb_files = sorted(os.listdir(frames_dir), key=lambda f: int(os.path.splitext(f)[0]))
    frame_cnt = len(rgb_files)
    assert frame_cnt > CHUNK_SIZE
    clipped = ((frame_cnt - CHUNK_SIZE) // frequency) * frequency
    frame_indices = np.array(
        [[j for j in range(i * frequency, i * frequency + CHUNK_SIZE)]
         for i in range(clipped // frequency + 1)]
    )
    batch_num = int(np.ceil(frame_indices.shape[0] / batch_size))
    frame_indices = np.array_split(frame_indices, batch_num, axis=0)

    n_crop = 10 if sample_mode == "oversample" else 1
    full = [[] for _ in range(n_crop)]
    for bi in range(batch_num):
        batch_data = load_rgb_batch(frames_dir, rgb_files, frame_indices[bi])
        if sample_mode == "oversample":
            for i, crop in enumerate(oversample_data(batch_data)):
                assert crop.shape[-2] == 224 and crop.shape[-3] == 224
                full[i].append(forward_batch(crop))
        else:
            crop = batch_data[:, :, 16:240, 58:282, :]
            full[0].append(forward_batch(crop))

    full = [np.concatenate(c, axis=0) for c in full]
    full = np.concatenate([np.expand_dims(c, 0) for c in full], axis=0)
    full = full[:, :, :, 0, 0, 0]              # (n_crop, T, 2048)
    return full.transpose([1, 0, 2])           # (T, n_crop, 2048)


def ffmpeg_extract_frames(video: str, out_dir: str) -> None:
    """ffmpeg -> ``<out_dir>/%d.jpg`` (start 0, native fps) — exactly Gowtham/main.py."""
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-loglevel", "quiet", "-i", video, "-start_number", "0",
         os.path.join(out_dir, "%d.jpg")],
        check=True,
    )


def extract_from_video(
    model, video, frequency=16, batch_size=20, sample_mode="oversample", device="cuda"
) -> np.ndarray:
    """ffmpeg-extract frames to a temp dir, then run the faithful pipeline."""
    with tempfile.TemporaryDirectory() as tmp:
        ffmpeg_extract_frames(video, tmp)
        return extract_from_frames_dir(model, tmp, frequency, batch_size, sample_mode, device)
