"""Sort a flat VadCLIP ``UCFClipFeatures`` dump into ``clip/{train,test}``.

The official VadCLIP CLIP features ship as ``UCFClipFeatures/<Class>/<Vid>_x264__<crop>.npy``
with the train/test split encoded in ``list/ucf_CLIP_rgbtest.csv`` (not by folder).
This symlinks each ``.npy`` into ``<dst>/train`` or ``<dst>/test`` so the loader in
``src/data/local.py`` can walk them. See ``docs/DATA_LOCAL.md``.

    python scripts/prepare_clip_features.py \
        --src ~/data/wsad/ucf_crime/features/clip/_byclass \
        --dst ~/data/wsad/ucf_crime/features/clip
    # optional: --test-list list/ucf_CLIP_rgbtest.csv (else fetched from VadCLIP repo)
"""

import argparse
import os
import urllib.request

_TESTLIST_URL = (
    "https://raw.githubusercontent.com/nwpu-zxr/VadCLIP/main/list/ucf_CLIP_rgbtest.csv"
)


def _test_video_ids(test_list: str | None) -> set:
    if test_list and os.path.exists(test_list):
        lines = open(test_list).read().splitlines()
    else:
        lines = urllib.request.urlopen(_TESTLIST_URL).read().decode().splitlines()
    ids = set()
    for ln in lines[1:]:  # skip header "path,label"
        p = ln.split(",")[0].strip()
        if not p:
            continue
        base = os.path.basename(p)
        vid = base.split("__")[0]  # <Vid>_x264 (drop __crop.npy)
        ids.add(vid)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="UCFClipFeatures root (recursive .npy)")
    ap.add_argument("--dst", required=True, help="~/data/wsad/clip")
    ap.add_argument("--test-list", default=None, help="ucf_CLIP_rgbtest.csv (optional)")
    ap.add_argument("--copy", action="store_true", help="copy instead of symlink")
    args = ap.parse_args()

    src, dst = os.path.expanduser(args.src), os.path.expanduser(args.dst)
    test_ids = _test_video_ids(args.test_list)
    for sub in ("train", "test"):
        os.makedirs(os.path.join(dst, sub), exist_ok=True)

    n_train = n_test = 0
    for dirpath, _, files in os.walk(src):
        for f in files:
            if not f.endswith(".npy"):
                continue
            vid = f.split("__")[0]
            sub = "test" if vid in test_ids else "train"
            srcp = os.path.join(dirpath, f)
            dstp = os.path.join(dst, sub, f)
            if os.path.lexists(dstp):
                continue
            if args.copy:
                import shutil

                shutil.copy2(srcp, dstp)
            else:
                os.symlink(os.path.abspath(srcp), dstp)
            n_train += sub == "train"
            n_test += sub == "test"
    print(f"linked train={n_train} test={n_test} into {dst}")


if __name__ == "__main__":
    main()
