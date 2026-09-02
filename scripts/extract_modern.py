"""Disk/VRAM-conscious modern-backbone extraction from the UCF-Crime raw ZIPs.

The raw videos (~128 GB) live inside `raw/*.zip`, so we NEVER unzip them all. For
each video we stream ONE member out of its zip to a temp file, run a registered
`src.features` extractor (which batches snippets to bound VRAM), save the `.npy`,
and delete the temp — disk peak ≈ one video. Backbone-agnostic via the registry,
so `--backbone videomae|xclip|internvideo|clip` all dispatch here.

    # forensic GATE first: a small test sample, then run feature_forensics on it
    # (relative content/probe screen — confirm the set ranks above i3d before full extract)
    WSAD_DATA=~/data/wsad uv run python scripts/extract_modern.py \
        --backbone videomae --model-name MCG-NJU/videomae-base --split test --limit 20 \
        --out ~/data/wsad/ucf_crime/features/videomae_base_GATE
    # full run (all 1900): drop --limit, --split train then test
"""

import argparse
import os
import tempfile
import zipfile

import numpy as np

ROOT = os.path.expanduser(os.environ.get("WSAD_DATA", "~/data/wsad"))
RAW = os.path.join(ROOT, "ucf_crime", "raw")
SPLIT = os.path.join(RAW, "UCF_Crimes-Train-Test-Split")


def build_zip_index() -> dict:
    """{bare_video_id: (zip_path, member_name)} over every raw video zip (filelist only)."""
    idx = {}
    for z in sorted(os.listdir(RAW)):
        if not z.endswith(".zip"):
            continue
        zp = os.path.join(RAW, z)
        try:
            with zipfile.ZipFile(zp) as zf:
                for m in zf.namelist():
                    if m.endswith(".mp4"):
                        vid = os.path.splitext(os.path.basename(m))[0]  # <Vid>_x264
                        idx.setdefault(vid, (zp, m))
        except zipfile.BadZipFile:
            continue
    return idx


def split_video_ids(split: str) -> list:
    """Test ids from the temporal-annotation file; train ids = everything else in the zips."""
    test_txt = os.path.join(SPLIT, "Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
    test_ids = []
    if os.path.exists(test_txt):
        for line in open(test_txt):
            parts = line.split()
            if parts:
                test_ids.append(os.path.splitext(parts[0])[0])  # <Vid>_x264
    if split == "test":
        return test_ids
    # train = all videos in zips minus the test set
    allids = set(build_zip_index())
    return sorted(allids - set(test_ids))


def extract_one(extractor, zip_path: str, member: str, backbone: str) -> np.ndarray:
    """Stream one video out of its zip to a temp file, extract, delete temp."""
    with zipfile.ZipFile(zip_path) as zf, tempfile.NamedTemporaryFile(
        suffix=".mp4", delete=False
    ) as tmp:
        tmp.write(zf.read(member))
        tmp_path = tmp.name
    try:
        return np.asarray(extractor.extract(tmp_path))
    finally:
        os.unlink(tmp_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, help="videomae | xclip | internvideo | clip")
    ap.add_argument("--model-name", default=None, help="HF id (e.g. MCG-NJU/videomae-large)")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--limit", type=int, default=None, help="extract only the first N (forensic gate)")
    ap.add_argument("--segment-to", type=int, default=None, help="mean-pool to N segments (train) or keep full (test)")
    ap.add_argument("--sample-to", type=int, default=None, help="FAST: extract only N uniformly-spaced snippets/video (~10x fewer forwards); output is (N, dim)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from src.features import build_extractor

    kw = {"device": args.device}
    if args.model_name:
        kw["model_name"] = args.model_name
    if args.segment_to is not None:
        kw["segment_to"] = args.segment_to
    if args.sample_to is not None:
        kw["sample_to"] = args.sample_to
    extractor = build_extractor(args.backbone, **kw)

    idx = build_zip_index()
    ids = split_video_ids(args.split)
    ids = [v for v in ids if v in idx]
    if args.limit:
        # balanced gate sample: interleave anomaly + Normal
        anom = [v for v in ids if "Normal" not in v][: args.limit // 2]
        norm = [v for v in ids if "Normal" in v][: args.limit - len(anom)]
        ids = anom + norm
    os.makedirs(args.out, exist_ok=True)
    print(f"[extract] backbone={args.backbone} model={args.model_name} split={args.split} "
          f"n={len(ids)} dim={extractor.dim} -> {args.out}", flush=True)

    for i, vid in enumerate(ids):
        dst = os.path.join(args.out, f"{vid}_{args.backbone}.npy")
        if os.path.exists(dst):
            continue
        zp, member = idx[vid]
        feats = extract_one(extractor, zp, member, args.backbone)
        np.save(dst, feats)
        print(f"  [{i+1}/{len(ids)}] {vid} {feats.shape}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
