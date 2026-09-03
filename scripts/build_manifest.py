"""Build the unified feature manifest for a dataset (the catalog every loader/tool reads).

One row per (backbone, split, video) with provenance + shape, read from the
``.npy`` headers only (no full array load), so it runs in seconds over the 9 GB
I3D zips. Writes both ``manifest.parquet`` (columnar, mmap-friendly) and
``manifest.jsonl`` (grep-able) under ``<dataset>/``.

Layout assumed (see docs/DATA_LOCAL.md):
    <dataset>/features/i3d/{train,test}.zip      # per-video <Vid>_i3d.npy
    <dataset>/features/clip/{train,test}/*.npy   # per-crop <Vid>__<c>.npy (symlinks ok)

    python scripts/build_manifest.py                       # dataset=ucf_crime
    python scripts/build_manifest.py --dataset aihub       # future dataset, same shape

Columns: dataset, backbone, split, video_id, label(normal|abnormal), path,
zip_member, n_crops, shape, dtype.
"""

import argparse
import json
import os
import zipfile
from collections import defaultdict

import numpy as np
from numpy.lib import format as npfmt

DATA_ROOT = os.path.expanduser(os.environ.get("WSAD_DATA", "~/data/wsad"))


def _npy_header(fobj):
    """(shape, dtype) from an open .npy file object, reading only the header."""
    major, minor = npfmt.read_magic(fobj)
    reader = getattr(npfmt, f"read_array_header_{major}_{minor}", None)
    if reader is None:  # very old/new .npy version
        reader = npfmt.read_array_header_1_0
    shape, _fortran, dtype = reader(fobj)
    return list(shape), str(dtype)


def _bare_vid(name: str) -> str:
    b = name[:-4] if name.endswith(".npy") else name
    if b.endswith("_i3d"):
        b = b[: -len("_i3d")]
    return b.split("__")[0]


def _label(video_id: str) -> str:
    return "normal" if "Normal" in video_id else "abnormal"


def _i3d_rows(dataset: str, ds_dir: str):
    rows = []
    for split in ("train", "test"):
        zp = os.path.join(ds_dir, "features", "i3d", f"{split}.zip")
        if not os.path.exists(zp):
            continue
        with zipfile.ZipFile(zp) as z:
            for info in z.infolist():
                if info.is_dir() or not info.filename.endswith(".npy"):
                    continue
                with z.open(info) as f:
                    shape, dtype = _npy_header(f)
                vid = _bare_vid(info.filename.split("/")[-1])
                rows.append(dict(
                    dataset=dataset, backbone="i3d", split=split, video_id=vid,
                    label=_label(vid), path=os.path.relpath(zp, DATA_ROOT),
                    zip_member=info.filename, n_crops=shape[0] if len(shape) == 3 else 1,
                    shape=shape, dtype=dtype,
                ))
    return rows


def _clip_rows(dataset: str, ds_dir: str):
    rows = []
    for split in ("train", "test"):
        d = os.path.join(ds_dir, "features", "clip", split)
        if not os.path.isdir(d):
            continue
        by_vid = defaultdict(list)
        for f in os.listdir(d):
            if f.endswith(".npy"):
                by_vid[_bare_vid(f)].append(f)
        for vid, files in by_vid.items():
            crop0 = sorted(files)[0]
            with open(os.path.join(d, crop0), "rb") as fobj:
                shape, dtype = _npy_header(fobj)
            rows.append(dict(
                dataset=dataset, backbone="clip", split=split, video_id=vid,
                label=_label(vid), path=os.path.relpath(d, DATA_ROOT),
                zip_member=None, n_crops=len(files), shape=shape, dtype=dtype,
            ))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ucf_crime")
    ap.add_argument("--data-root", default=DATA_ROOT)
    args = ap.parse_args()

    ds_dir = os.path.join(os.path.expanduser(args.data_root), args.dataset)
    rows = _i3d_rows(args.dataset, ds_dir) + _clip_rows(args.dataset, ds_dir)
    if not rows:
        raise SystemExit(f"no features found under {ds_dir}/features (see docs/DATA_LOCAL.md)")

    jsonl = os.path.join(ds_dir, "manifest.jsonl")
    with open(jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    import pandas as pd

    df = pd.DataFrame(rows)
    df["shape"] = df["shape"].apply(lambda s: ",".join(map(str, s)))  # parquet-friendly
    parquet = os.path.join(ds_dir, "manifest.parquet")
    df.to_parquet(parquet, index=False)

    # summary
    g = df.groupby(["backbone", "split"]).agg(
        n=("video_id", "count"),
        normal=("label", lambda s: (s == "normal").sum()),
        abnormal=("label", lambda s: (s == "abnormal").sum()),
    )
    print(g.to_string())
    print(f"\nwrote {parquet}\n      {jsonl}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
