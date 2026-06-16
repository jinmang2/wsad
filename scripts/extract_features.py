import argparse
import os
from typing import Union

import datasets
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from src.data import TenCropVideoFrameDataset
from src.i3d import build_i3d_feature_extractor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def load_feature_extractor(
    model_name: str = "i3d_8x8_r50", device: str = "cpu"
) -> torch.nn.Module:
    model = build_i3d_feature_extractor(model_name=model_name)
    model.eval()
    model.to(device)
    return model


def _extract_video_features(
    video_clips_dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    outputs = []
    dataloader = DataLoader(video_clips_dataset, batch_size=batch_size, shuffle=False)
    for inputs in dataloader:
        # Unlike Tushar-N's B which is `n_videos`, our B is `n_clips`.
        # inputs.shape: (n_clips / bsz, n_crops, clip_len, n_channel, width, height)
        # (B / bsz, 10, 16, 3, 224, 224) -> (B / bsz, 10, 3, 16, 224, 224)
        inputs = inputs.permute(0, 1, 3, 2, 4, 5)
        n_crops = inputs.shape[1]
        crops = [
            model(inputs[:, i].to(device)).detach().cpu().numpy()
            for i in range(n_crops)
        ]
        # Combine crops: [(B / bsz, 2048, 1, 1, 1)] * 10 -> (B / bsz, 10, 2048, 1, 1, 1)
        outputs.append(np.stack(crops, axis=1))

    # vstack: [(B / bsz, 10, 2048, 1, 1, 1)] * bsz -> (B, 10, 2048, 1, 1, 1)
    outputs = np.squeeze(np.vstack(outputs))  # squeeze: (B, 10, 2048)

    return outputs


def extract_features_from_video(
    dataset: Union[datasets.Dataset, datasets.DatasetDict],
    model: torch.nn.Module,
    device: torch.device,
    output_dir: str,
    batch_size: int = 16,
):
    if isinstance(dataset, datasets.DatasetDict):
        for mode, dset in dataset.items():
            subset_outpath = os.path.join(output_dir, mode)
            extract_features_from_video(dset, model, device, subset_outpath)
        return

    assert isinstance(dataset, datasets.Dataset), (
        "The type of dataset argument must be `datasets.Dataset` or"
        f"`datasets.DatasetDict`. Your input's type is {type(dataset)}."
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for sample in tqdm(dataset):
        # check existence
        file_name = sample["video_path"].split(os.sep)[-1].split(".")[0]
        save_path = os.path.join(output_dir, file_name + "_i3d.npy")

        if os.path.exists(save_path):
            continue

        # If the size of the video is larger than 1GB, divide it by the segment length.
        # After that, upload it to RAM and receive the tencrop result from the model.
        # Note That: The script below provides the result of converting bytes to killobytes.
        # https://huggingface.co/datasets/jinmang2/ucf_crime/blob/main/ucf_crime.py
        if sample["size"] > 1024**2:
            # The fps of the video in `ucf_crime` dataset is 30. Therefore, 3,000 video frames
            # are about 100 seconds long, which is a good video length for inference in colab pro+.
            # Among the transforms of `TenCropVideoFrameDataset`, `LoopPad` forcibly pads clips lower than
            # `frames_per_clip`(default: 16), so segment_length is designated as a multiple of 16.
            seg_len = 16 * 188  # 3,008
            import decord  # lazy: only the legacy big-video path needs it

            # read video frames
            vr = decord.VideoReader(uri=sample["video_path"])
            segments = []
            for seg in tqdm(range(len(vr) // seg_len + 1)):
                seg_folder = os.path.join(output_dir, file_name)

                if not os.path.exists(seg_folder):
                    os.makedirs(seg_folder)

                seg_save_path = os.path.join(seg_folder, file_name + f"_{seg}.npy")

                if os.path.exists(seg_save_path):
                    outputs = np.load(seg_save_path)
                else:
                    images = []
                    for i in [seg * seg_len + i for i in range(seg_len)]:
                        if i == len(vr):
                            break
                        arr = vr[i].asnumpy()
                        images.append(Image.fromarray(arr))
                    video_clips_dataset = TenCropVideoFrameDataset(images)
                    # inference
                    outputs = _extract_video_features(
                        video_clips_dataset, model, device, batch_size
                    )
                    np.save(seg_save_path, outputs)

                segments.append(outputs)
            outputs = np.vstack(segments)
        else:
            # read video frames
            video_clips_dataset = TenCropVideoFrameDataset(sample["video_path"])
            # inference
            outputs = _extract_video_features(
                video_clips_dataset, model, device, batch_size
            )

        # save
        np.save(save_path, outputs)


def segment_features(feat_output_dir: str, seg_output_dir: str, seg_length: int = 32):
    files = sorted(os.listdir(feat_output_dir))
    for file in tqdm(files):
        if not file.endswith(".npy"):
            continue

        savepath = os.path.join(seg_output_dir, file)
        if os.path.exists(savepath):
            continue

        # (nclips, 10, 2048) -> (10, nclips, 2048)
        features = np.load(os.path.join(feat_output_dir, file)).transpose(1, 0, 2)

        divided_features = []
        for f in features:
            new_feat = np.zeros((seg_length, f.shape[1])).astype(np.float32)
            r = np.linspace(0, len(f), seg_length + 1, dtype=int)
            for i in range(seg_length):
                if r[i] != r[i + 1]:
                    new_feat[i, :] = np.mean(f[r[i] : r[i + 1], :], 0)
                else:
                    new_feat[i, :] = f[r[i], :]
            divided_features.append(new_feat)
        divided_features = np.array(divided_features, dtype=np.float32)

        np.save(savepath, divided_features)


def main(args: argparse.Namespace):
    # Backbone-agnostic path (issue #17): dispatch to any registered extractor and
    # cache `<stem>_<backbone>.npy`. The legacy i3d_8x8_r50 path below is kept for
    # `--backbone` unset so existing reproduction stays byte-identical.
    if args.backbone:
        from src.features import build_extractor
        from src.features.extract import run_extraction

        extractor = build_extractor(args.backbone, device=args.device)
        dset = load_dataset(args.repo_id, config_name=args.config_name)
        feat_output_dir = os.path.join(args.output_dir, f"features_{args.backbone}")
        subsets = (
            dset.items()
            if isinstance(dset, datasets.DatasetDict)
            else [(None, dset)]
        )
        for mode, dsub in subsets:
            out_dir = (
                feat_output_dir if mode is None else os.path.join(feat_output_dir, mode)
            )
            run_extraction(dsub, extractor, out_dir, args.backbone)
        return

    feat_output_dir = os.path.join(args.output_dir, "anomaly_features")
    # anomaly = load_ucf_crime_dataset(args.repo_id, args.cache_dir, args.config_name)
    ucf_crime_anomaly_dset = load_dataset("jinmang2/ucf_crime", config_name="anomaly")
    model = load_feature_extractor(args.model_name, args.device)
    extract_features_from_video(
        ucf_crime_anomaly_dset, model, args.device, feat_output_dir
    )

    seg_output_dir = os.path.join(
        args.output_dir, f"segment_features_{args.seg_length}"
    )
    # Apply segments only for the train dataset
    segment_features(
        os.path.join(feat_output_dir, "train"), seg_output_dir, args.seg_length
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract video features and segment them."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="jinmang2/ucf_crime",
        help="HuggingFace dataset repository ID.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for the dataset.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for feature extractor."
    )
    parser.add_argument(
        "--config_name", type=str, default="anomaly", help="Dataset configuration name."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="i3d_8x8_r50",
        help="Legacy pytorchvideo I3D model name (used only when --backbone is unset).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Registered backbone to dispatch on (i3d, clip, videomae, ...). "
        "When set, uses the backbone-agnostic extractor and caches "
        "<stem>_<backbone>.npy; when unset, runs the legacy I3D path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for features.",
    )
    parser.add_argument(
        "--seg_length",
        type=int,
        default=32,
        help="Segment length for feature extraction.",
    )

    args = parser.parse_args()
    main(args)
