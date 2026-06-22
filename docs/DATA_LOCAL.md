# Local data layout (`~/data/wsad`)

The data layer is **local-first with HF fallback** (`data.source: auto`). Features
are organized **per dataset** so a second dataset (e.g. AIHub) slots in as a sibling
without ambiguity. If a backbone is missing locally the loader falls back to the
Hugging Face cache.

```
~/data/wsad/                              # = data.root (override: WSAD_DATA env or data.root=...)
└── ucf_crime/                            # = data.dataset_dir (one dir per dataset)
    ├── raw/                              # source videos + official splits (Anomaly-Videos-*.zip, ...)
    ├── features/
    │   ├── i3d/
    │   │   ├── train.zip                 # MGFN canonical: 1610 × (10, 32, 2048)  seg32, crop-first
    │   │   └── test.zip                  #                 290 × (T, 10, 2048)    full-length
    │   └── clip/
    │       ├── _byclass/<Class>/*.npy    # VadCLIP UCFClipFeatures dump (per-crop (T,512) fp16)
    │       ├── train/  <Vid>_x264__<c>.npy   # symlinks into _byclass (official VadCLIP split)
    │       └── test/   <Vid>_x264__<c>.npy
    ├── annotations/
    │   └── ground_truth.json             # frame-level labels, keys <Vid>_i3d.npy (290, backbone-agnostic)
    ├── manifest.parquet / manifest.jsonl # unified catalog (built by scripts/build_manifest.py)
    └── _archive/                         # incompatible/superseded features (NOT used by the pipeline)
        ├── UCF_{Train,Test}_ten_crop_i3d/   # RTFM extraction — ~10x scale (L2~22), only 109/290 test
        └── UCF_test_feature.zip
```

The loader resolves this new layout first and falls back to the **legacy** layout
(top-level `i3d/`, `clip/`, dataset-root `{train,test}.zip` / `ground_truth.json`),
so older trees keep working. Path resolution lives in `src/data/local.py`
(`_features_root`, `_clip_dir`, `_i3d_npy_dir`, `_i3d_zip_path`, gt resolver).

## Feature provenance & scale (do NOT mix sources)

Magnitude-based methods (RTFM / MGFN / BN-WVAD) depend on feature L2 scale, so
mixing extractions silently breaks them. Verified per-snippet L2 norms:

| Source | I3D L2/snip | Notes |
|--------|-------------|-------|
| **`features/i3d/*.zip`** (canonical, DeepMIL/Roc-Ng) | **~2.5** | the working pair, 1610/290, what every runner expects (its PROVENANCE says DeepMIL; the "MGFN" label here was loose) |
| **`features/i3d_mgfn/{train,test}/`** (MGFN authors' OneDrive) | **~22** | full-length `(T,10,2048)`, **COMPLETE 1610/290** verified 2026-06-22, raw pre-seg32. RTFM-family scale. See its PROVENANCE.md |
| RTFM `_archive/...` | ~22 | different extraction, 109/290 test — quarantined, ignore (i3d_mgfn supersedes it) |
| our tushar-n extractor (`scripts/validate_extraction.py`) | ~22 | self-consistent w/ RTFM, **NOT** with the ~2.5 i3d/ |

> Two distinct lineages share the "MGFN" name loosely. The L2~2.5 pair in
> `features/i3d/` is DeepMIL (what the default runners use). The genuine
> MGFN-author distribution (HKU OneDrive) is L2~22 and lives in
> `features/i3d_mgfn/` — select with `data.feature_variant: i3d_mgfn`. Never mix
> the two scales across train/test.

→ A fresh extraction must re-do **both** train+test with one model; you cannot
reuse MGFN's test against tushar-n train. CLIP similarly: the provided
`UCFClipFeatures` are **OpenAI** CLIP ViT-B/16; our extractor defaults to
`laion2b` — a different space, so fresh CLIP features need a retrain, not a swap.

## Where each download goes

| Source | What | Put under |
|--------|------|-----------|
| DeepMIL UCF features (canonical) | I3D seg32 `train.zip` + full-length `test.zip` | `ucf_crime/features/i3d/` |
| MGFN authors' OneDrive 10-crop I3D | full-length `(T,10,2048)` per-video `.npy`, 1610/290 | `ucf_crime/features/i3d_mgfn/{train,test}/` (fetch via `scripts/onedrive_fetch/`) |
| VadCLIP `UCFClipFeatures` | CLIP per-crop `.npy` (`<Vid>_x264__0..9.npy`, `(T,512)`) | `ucf_crime/features/clip/_byclass/` then run prepare |

The VadCLIP dump is **not folder-split** — it ships `list/ucf_CLIP_rgbtest.csv`.
Sort the flat dump into `clip/{train,test}` symlinks:

```bash
python scripts/prepare_clip_features.py \
    --src ~/data/wsad/ucf_crime/features/clip/_byclass \
    --dst ~/data/wsad/ucf_crime/features/clip
```

## Unified manifest (the catalog)

`scripts/build_manifest.py` writes `manifest.parquet` + `manifest.jsonl` — one row
per `(backbone, split, video)` with `video_id, label, path, zip_member, n_crops,
shape, dtype` (read from `.npy` headers only, ~24 s over the 9 GB zips). Use it to
query/verify the data without loading arrays; it's the integration point for new
tooling and the AIHub dataset.

```bash
python scripts/build_manifest.py                 # dataset=ucf_crime (default)
python scripts/build_manifest.py --dataset aihub # future, same shape
```

## Per-runner CLIP post-processing (one download, three contracts)

The same `(T,512)` per-crop files feed all three CLIP models; the loader reshapes
per runner (set by the `data=` config you select):

| Runner | crops | train length | test | data config |
|--------|-------|--------------|------|-------------|
| `vadclip` | 1 (`__0`) | `process_feat` → 256 | full-length (T,1,512) | `data=clip_vadclip` |
| `tpwng`  | 1 (`__0`) | uniform → 32 seg | full-length (T,1,512) | `data=clip_seg` |
| `clip_tsa` | 1 (`__0`) | uniform → 32 seg | full-length (T,1,512) | `data=clip_seg` |

I3D models need no `data=` override (default `data=i3d`, zip-direct). Example:

```bash
python train.py runner=rtfm                         # I3D, local→HF auto
python train.py runner=vadclip data=clip_vadclip    # CLIP, single-crop, len 256
python train.py runner=tpwng   data=clip_seg        # CLIP, 32-seg
```

## Extracting features from raw video

`scripts/validate_extraction.py` runs one raw video end-to-end (decord → I3D /
CLIP) and reports shape, scale, and wall time. Validated on RTX2070S (2026-06-14):

| Line | time / video | full UCF (~1900) | output |
|------|-------------|------------------|--------|
| I3D tushar-n (`I3Res50`, self-contained, no pytorchvideo) | ~37 s | ~19.5 h | `(T,10,2048)` → seg32 `(10,32,2048)` |
| CLIP ViT-B/16 | ~7 s | ~3.7 h | `(T,512)` |

```bash
python scripts/validate_extraction.py            # i3d baseline, 1 sample video
python scripts/validate_extraction.py --nonlocal # I3Res50(use_nl=True)
python scripts/validate_extraction.py --clip     # also time the CLIP line
```

`pytorchvideo` is only needed for the alternative `i3d_8x8_r50` SlowFast variant
(imported lazily); the default tushar-n / nonlocal `I3Res50` path needs only
`decord` + `torch`.

## Notes

- Class label is parsed free from the filename (`Abuse001_…`), so VadCLIP CLASM /
  TPWNG prompts work without extra annotation.
- `ground_truth.json` is backbone-agnostic; `_align_gt`/`_bare_vid` re-key it from
  `<Vid>_i3d.npy` onto CLIP per-crop names.
- Faithful CLIP text branches (vadclip/tpwng) download OpenAI CLIP ViT-B/16 text
  weights once (open_clip `pretrained="openai"`) to match the **provided** features.
