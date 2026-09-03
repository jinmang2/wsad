"""Extract Gowtham/RTFM-faithful I3D features for a set of raw videos.

Wraps :mod:`src.features.i3d_gowtham` (bit-exact to GowthamGottimukkala's
extractor, verified by ``scripts/verify_i3d_extract.py``). Produces one
``<Vid>_i3d.npy`` of shape ``(T, 10, 2048)`` per video.

    # baseline weights
    python scripts/extract_i3d_gowtham.py --videos '/path/*.mp4' --out OUTDIR
    # non-local weights (recommended; closer to RTFM-style features)
    python scripts/extract_i3d_gowtham.py --videos '/path/*.mp4' --out OUTDIR --nonlocal

Weights default to the converted ``pretrained/i3d/i3d_{baseline,nonlocal}_r50_kinetics.pth``
(run ``scripts/convert_i3d_caffe2.py`` first). Note: this faithful path writes per-frame
JPGs via ffmpeg then reads them (~37 s + I/O per video on an RTX2070S), so a full
UCF-Crime pass (~1900 videos) is an overnight job; parallelize across GPUs if available.
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.i3d_gowtham import build_model, extract_from_video

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="glob or dir of .mp4")
    ap.add_argument("--out", required=True, help="output dir for <Vid>_i3d.npy")
    ap.add_argument("--nonlocal", dest="nl", action="store_true")
    ap.add_argument("--weights", default=None, help="override .pth path")
    ap.add_argument("--frequency", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--sample-mode", default="oversample", choices=["oversample", "center_crop"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    variant = "nonlocal" if args.nl else "baseline"
    weights = args.weights or os.path.join(
        ROOT, "pretrained", "i3d", f"i3d_{variant}_r50_kinetics.pth"
    )
    if not os.path.exists(weights):
        raise SystemExit(f"missing weights {weights} (run scripts/convert_i3d_caffe2.py)")

    pat = args.videos
    videos = sorted(glob.glob(os.path.join(pat, "*.mp4") if os.path.isdir(pat) else pat))
    if not videos:
        raise SystemExit(f"no videos matched {pat}")
    os.makedirs(args.out, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(weights, use_nl=args.nl, device=dev)
    print(f"i3d-gowtham {variant} | {len(videos)} videos -> {args.out} (dev={dev})")

    for i, v in enumerate(videos):
        name = os.path.splitext(os.path.basename(v))[0]
        out = os.path.join(args.out, f"{name}_i3d.npy")
        if os.path.exists(out) and not args.overwrite:
            print(f"  [{i + 1}/{len(videos)}] skip {name} (exists)")
            continue
        t = time.time()
        feat = extract_from_video(
            model, v, args.frequency, args.batch_size, args.sample_mode, dev
        )
        np.save(out, feat)
        l2 = np.linalg.norm(feat.reshape(-1, feat.shape[-1]), axis=1).mean()  # per-2048-vec
        print(f"  [{i + 1}/{len(videos)}] {name} {feat.shape} "
              f"L2/snip={l2:.1f} {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
