# Local data layout (`~/data/wsad`)

The data layer is **local-first with HF fallback** (`data.source: auto`). Drop the
files you download into the tree below and the loader picks them up; if a backbone
dir is missing it falls back to the Hugging Face cache.

```
~/data/wsad/                      # = data.root (override: WSAD_DATA env or data.root=...)
├── i3d/                          # I3D 10-crop features (RTFM-style; rtfm/mgfn/ur_dmu/s3r/bn_wvad/mil/gs_moe)
│   ├── train/  <Vid>_i3d.npy     # (10, 32, 2048)  10-crop, 32-segment
│   └── test/   <Vid>_i3d.npy     # (T, 10, 2048)   full-length, 10-crop
│   # ground_truth.json comes from the HF dataset (jinmang2/ucf_crime_tencrop_i3d_seg32)
└── clip/                         # CLIP ViT-B/16 features (vadclip/clip_tsa/tpwng)
    ├── train/  <Vid>_x264__<crop>.npy   # (Tframes, 512)  per-crop, snippet-level (1 snippet = 16 frames)
    └── test/   <Vid>_x264__<crop>.npy
    # split = official VadCLIP test list; ground_truth.json reused from the i3d HF dataset (backbone-agnostic)
```

## Where each download goes

| Source | What | Put under |
|--------|------|-----------|
| RTFM Google Drive (`ucf_crime_tencrop_i3d`) | I3D 10-crop `.npy` per video | `i3d/train`, `i3d/test` (or skip — HF `ucf_crime_tencrop_i3d_seg32` is the fallback) |
| VadCLIP (Baidu/OneDrive `UCFClipFeatures`) | CLIP per-crop `.npy` (`<Vid>_x264__0..9.npy`, each `(T,512)`) | `clip/train`, `clip/test` |

The VadCLIP download is **not folder-split** — it ships `list/ucf_CLIP_rgb.csv`
(train) and `list/ucf_CLIP_rgbtest.csv` (test). Run the prepare script to sort the
flat `UCFClipFeatures/` dump into `clip/train` and `clip/test`:

```bash
python scripts/prepare_clip_features.py \
    --src /path/to/UCFClipFeatures --dst ~/data/wsad/clip   # symlinks by official test list
```

## Per-runner CLIP post-processing (one download, three contracts)

The same `(T,512)` per-crop files feed all three CLIP models; the loader reshapes
per runner (set by the `data=` config you select):

| Runner | crops | train length | test | data config |
|--------|-------|--------------|------|-------------|
| `vadclip` | 1 (`__0`) | `process_feat` → 256 | `process_split` → (n,256,512) | `data=clip_vadclip` |
| `tpwng`  | 1 (`__0`) | uniform → 32 seg | full-length (T,1,512) | `data=clip_seg` |
| `clip_tsa` | 1 (`__0`) or 10 | uniform → 32 seg | full-length (T,10/1,512) | `data=clip_seg` |

I3D models need no `data=` override (default `data=i3d`). Example:

```bash
python train.py runner=rtfm                         # I3D, local→HF auto
python train.py runner=vadclip data=clip_vadclip    # CLIP, single-crop, len 256
python train.py runner=tpwng   data=clip_seg        # CLIP, 32-seg
```

## Notes

- Class label is parsed free from the filename (`Abuse001_…` → class 7-style id),
  so VadCLIP's CLASM and TPWNG's prompts work without extra annotation.
- `ground_truth.json` (frame-level labels) is backbone-agnostic and reused across
  i3d/clip; the loader pulls it from the HF i3d dataset.
- Faithful CLIP text branches (vadclip/tpwng) download OpenAI CLIP ViT-B/16 text
  weights once (open_clip `pretrained="openai"`) to match the extracted features.
