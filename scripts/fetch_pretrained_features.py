"""Fetch a published pre-extracted feature set and arrange it as a local variant.

Extraction is the binding constraint here — the single 8 GB card needs hours per backbone
(`docs/NEXT_PHASE.md`) — so a feature set someone has already extracted is worth far more
than the download costs. See `docs/FEATURE_SOURCES.md` for the survey behind the registry
below, including the leakage checks and the provenance caveats.

    # what is available, and what it would cost
    uv run python scripts/fetch_pretrained_features.py --list
    # download + arrange (resumable; re-run after any interruption)
    uv run python scripts/fetch_pretrained_features.py --source languagebind

Files land as `features/<variant>/{train,test}/<stem>_<backbone>.npy`, the same layout the
extractors write, so `run_matrix.py --variant <variant> --feature-dim <dim>` just works.
A PROVENANCE.md is written next to them recording where they came from and what is *not*
known about them — these are third-party arrays, and a number produced from them must carry
that caveat until a screen and a short head train confirm they behave.
"""

import argparse
import os
import shutil

SOURCES = {
    "languagebind": dict(
        repo="yukaneko55/UCF-Crime_features",
        subdir="Languagebind",
        suffix="_languagebind.npy",
        variant="languagebind_10crop",
        dim=768,
        layout="(10, T, 768) — 10-crop, 16-frame snippet grid, NOT L2-normalized",
        size_gb=26.4,
        leakage="clean — LanguageBind trains on VIDAL-10M, which contains no UCF-Crime",
        caveat="No model card, README or extraction script published; the exact checkpoint, "
               "crop protocol and stride are unverified. Screen before trusting a number.",
    ),
}


def _split_ids():
    from scripts.extract_modern import split_video_ids

    return {"train": set(split_video_ids("train")), "test": set(split_video_ids("test"))}


def _list() -> None:
    for name, s in SOURCES.items():
        print(f"{name}")
        print(f"  repo     {s['repo']} ({s['size_gb']} GB)")
        print(f"  variant  features/{s['variant']}/  dim={s['dim']}")
        print(f"  layout   {s['layout']}")
        print(f"  leakage  {s['leakage']}")
        print(f"  caveat   {s['caveat']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=sorted(SOURCES), help="which published set to fetch")
    ap.add_argument("--list", action="store_true", help="show the registry and exit")
    ap.add_argument("--root", default=os.environ.get("WSAD_DATA", "~/data/wsad"))
    ap.add_argument("--dataset", default="ucf_crime")
    ap.add_argument("--workers", type=int, default=4, help="parallel download workers")
    ap.add_argument("--link", action="store_true",
                    help="symlink out of the HF cache instead of copying (saves the 2nd copy, "
                         "but the variant breaks if the cache is cleared)")
    args = ap.parse_args()

    if args.list or not args.source:
        _list()
        return

    from huggingface_hub import snapshot_download

    src = SOURCES[args.source]
    out_root = os.path.join(os.path.expanduser(args.root), args.dataset, "features", src["variant"])
    splits = _split_ids()
    for split in splits:
        os.makedirs(os.path.join(out_root, split), exist_ok=True)

    print(f"[fetch] {src['repo']} -> {out_root}  ({src['size_gb']} GB, resumable)", flush=True)
    cached = snapshot_download(
        src["repo"],
        repo_type="dataset",
        allow_patterns=[f"{src['subdir']}/*"],
        max_workers=args.workers,
    )

    placed = {"train": 0, "test": 0}
    unknown = []
    source_dir = os.path.join(cached, src["subdir"])
    for name in sorted(os.listdir(source_dir)):
        if not name.endswith(src["suffix"]):
            continue
        stem = name[: -len(src["suffix"])]
        split = next((s for s, ids in splits.items() if stem in ids), None)
        if split is None:
            unknown.append(stem)
            continue
        dst = os.path.join(out_root, split, name)
        if not os.path.exists(dst):
            if args.link:
                os.symlink(os.path.join(source_dir, name), dst)
            else:
                shutil.copy2(os.path.join(source_dir, name), dst)
        placed[split] += 1

    with open(os.path.join(out_root, "PROVENANCE.md"), "w") as f:
        f.write(
            f"# {src['variant']}\n\n"
            f"Downloaded from `{src['repo']}` (`{src['subdir']}/`) by "
            f"`scripts/fetch_pretrained_features.py`.\n\n"
            f"- layout: {src['layout']}\n- dim: {src['dim']}\n- leakage: {src['leakage']}\n"
            f"- placed: train={placed['train']} test={placed['test']}\n\n"
            f"**Unverified third-party features.** {src['caveat']}\n"
        )

    print(f"[fetch] placed train={placed['train']} test={placed['test']}"
          + (f", {len(unknown)} ids not in either split: {unknown[:5]}" if unknown else ""))
    print(f"[fetch] next: uv run python scripts/diag/feature_forensics.py "
          f"--dir {out_root}/test --name {src['variant']}")


if __name__ == "__main__":
    main()
