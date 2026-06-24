"""Validate the feature-extraction lines (I3D tushar-n / nonlocal, CLIP) end-to-end.

Pulls one raw UCF-Crime video out of the ``raw/Anomaly-Videos-*.zip`` shards,
runs it through each extractor, and reports shape, per-snippet L2 scale, and wall
time so we know (a) the line is correct and (b) roughly how long a full extraction
takes. It also seg32-pools the I3D output to confirm it matches the MGFN train
layout ``(10, 32, 2048)`` and prints the scale gap vs the MGFN canonical (~2.5).

    python scripts/validate_extraction.py                 # i3d baseline, 1 video
    python scripts/validate_extraction.py --nonlocal      # i3d use_nl=True
    python scripts/validate_extraction.py --clip          # also time the CLIP line
    python scripts/validate_extraction.py --video /path/to.mp4

Findings (RTX2070S, 2026-06-14): tushar-n I3D ~39 s / 1795-frame video, L2/snip
~22 (= RTFM scale, ~10x the MGFN ~2.5 canonical). => fresh extractions are
self-consistent but NOT mixable with the MGFN zips; re-extract train+test together.
"""

import argparse
import os
import sys
import time
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW = os.path.expanduser(
    os.path.join(os.environ.get("WSAD_DATA", "~/data/wsad"), "ucf_crime", "raw")
)


def _extract_sample_video(dst="/tmp/wsad_extract_test") -> str:
    """Pull a small Abuse video out of Anomaly-Videos-Part-1.zip (once)."""
    os.makedirs(dst, exist_ok=True)
    member = "Anomaly-Videos-Part-1/Abuse/Abuse037_x264.mp4"
    out = os.path.join(dst, os.path.basename(member))
    if not os.path.exists(out):
        with zipfile.ZipFile(os.path.join(RAW, "Anomaly-Videos-Part-1.zip")) as z:
            with z.open(member) as src, open(out, "wb") as f:
                f.write(src.read())
    return out


def _scale(feat: np.ndarray) -> float:
    return float(np.linalg.norm(feat.reshape(-1, feat.shape[-1]), axis=1).mean())


def validate_i3d(video: str, use_nl: bool) -> None:
    import decord

    from src.data.local import segment
    from src.features.i3d import I3DFeatureExtractor

    name = "tushar-n-nonlocal" if use_nl else "tushar-n-baseline"
    # I3Res50(use_nl=...) shares the tushar-n checkpoint; nonlocal adds NL blocks.
    ext = I3DFeatureExtractor(model_name="tushar-n-baseline", device="cuda", batch_size=16)
    if use_nl:
        # NB: this loads the BASELINE checkpoint into a use_nl=True model, so the
        # non-local blocks stay randomly initialized. It is a timing/shape SMOKE
        # test only — NOT fidelity-grade. For real non-local features use the
        # converted weights via scripts/extract_i3d_gowtham.py --nonlocal.
        from src.i3d import I3Res50

        print("  [warn] nonlocal smoke: NL blocks random-init (not fidelity)", file=sys.stderr)
        sd = ext.model.state_dict()
        ext.model = I3Res50(use_nl=True).eval().to("cuda")
        ext.model.load_state_dict(sd, strict=False)

    n = len(decord.VideoReader(video))
    t = time.time()
    feat = ext.extract(video)  # (T, 10, 2048)
    dt = time.time() - t
    T, nc, dim = feat.shape
    seg = np.stack([segment(feat[:, c, :], 32) for c in range(nc)], 0)
    print(f"[I3D {name}] frames={n} -> {feat.shape} in {dt:.1f}s")
    print(f"            L2/snip={_scale(feat):.2f} (MGFN canonical ~2.5; RTFM ~22)")
    print(f"            seg32 crop-first {seg.shape} == MGFN train (10,32,2048)")


def validate_videomae(video: str) -> None:
    from src.features.videomae import VideoMAEFeatureExtractor

    ext = VideoMAEFeatureExtractor(device="cuda", segment_to=None)  # per-snippet (T,dim)
    t = time.time()
    feat = ext.extract(video)
    dt = time.time() - t
    print(f"[VideoMAE {ext.model.config._name_or_path if hasattr(ext.model.config,'_name_or_path') else 'base'}] "
          f"-> {feat.shape} {feat.dtype} in {dt:.1f}s | dim={ext.dim} "
          f"finite={np.isfinite(feat).all()} L2/snip={_scale(feat):.2f}")


def validate_clip(video: str) -> None:
    from src.features.clip import CLIPFeatureExtractor

    ext = CLIPFeatureExtractor(device="cuda", segment_to=None)  # frame-level (T,512)
    t = time.time()
    feat = ext.extract(video)
    dt = time.time() - t
    print(f"[CLIP ViT-B-16 laion2b] -> {feat.shape} {feat.dtype} in {dt:.1f}s")
    print(f"            L2/frame={_scale(feat):.2f}  (NOTE: laion2b space != VadCLIP's")
    print("            OpenAI-CLIP UCFClipFeatures; fresh != provided, retrain on it)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="raw .mp4 (default: sample from zip)")
    ap.add_argument("--nonlocal", dest="nonlocal_", action="store_true")
    ap.add_argument("--clip", action="store_true", help="also time the CLIP line")
    ap.add_argument("--videomae", action="store_true", help="also time the VideoMAE line")
    ap.add_argument("--skip-i3d", action="store_true", help="skip the I3D line")
    args = ap.parse_args()

    video = args.video or _extract_sample_video()
    print(f"video: {video}\n")
    if not args.skip_i3d:
        validate_i3d(video, use_nl=args.nonlocal_)
    if args.clip:
        validate_clip(video)
    if args.videomae:
        validate_videomae(video)


if __name__ == "__main__":
    main()
