# Data Pipeline

How features get from raw video to a training batch, the three HF datasets, the
axis/segmentation conventions every model depends on, and the RAM strategy for
the 7.8 GB box. (Hardware/feasibility numbers live in `docs/TRAINING.md`.)

## The three HF datasets (owner `jinmang2`)

| Dataset | Size | What it is | Role |
|---------|------|------------|------|
| `ucf_crime` | 103 GB | **raw videos** (zip parts) | extraction source only |
| `ucf-crime-tencrop-i3d` | 65.8 GB | **full-length** I3D feats `(nclips, 10, 2048)` per video — raw output of the extractor, before segmentation | variable-T methods; source for segmentation |
| `ucf_crime_tencrop_i3d_seg32` | 9.93 GB | **the training/eval set** (the loader's default) | train + test |

`ucf_crime_tencrop_i3d_seg32` is exactly what `scripts/extract_features.py`
produces: per-video I3D features, then uniformly **segmented to 32** for the train
split. It is `DEFAULT_FEATURE_HUB` in `src/dataset.py`. Contents:
- `train.zip` — **32-segment** features, `(10, 32, 2048)` per video (crops, T, D).
- `test.zip` — **full-length** features, `(nclips, 10, 2048)` per video (T, crops, D).
- `ground_truth.json` — frame-level test labels (built by `make_gt_ucf.py`).

> Note the **two layouts in one repo**: train is `(crops, T, D)`, test is
> `(T, crops, D)`. The loader + runner reconcile them (see below) — preserve this.

## Extraction pipeline (`scripts/extract_features.py`)

```
raw video ──decord──► frames
  └─ TenCropVideoFrameDataset: 16-frame clips × 10-crop (4 corners+center, ×2 flip)
        └─ I3D R50 ──► (nclips, 10, 2048)  saved as <video>_i3d.npy   [full-length repo]
              └─ segment_features: uniform mean-pool over time → (10, 32, 2048)  [seg32 train]
```
- **Snippet** = 16 frames → 1 feature. **10-crop** kept (RTFM/MGFN use all crops).
- **32-segment** = Sultani convention (every video → 32 segments). Test keeps full
  length so frame-level AUC aligns to real video length.
- I3D re-extraction is **unnecessary** (features already on the Hub). New backbones
  (CLIP/VideoMAE) plug into the same script via `--backbone` (see `docs/FEATURE_EXTRACTORS.md`).

## Loader (`src/dataset.py`)

`build_feature_dataset(mode)` pulls the zip from the Hub and builds a
`FeatureDataset`. Per item:
- **`add_magnitude`**: appends an L2-norm channel → **2048 → 2049**. MGFN consumes
  it (slices `[:2048]` features / `[2048:]` magnitude); RTFM/Sultani/UR-DMU/GS-MoE
  slice `[..., :feature_size]` and ignore it.
- **train**: split into `normal` / `abnormal` by `"Normal"` in the filename → a
  dict of two datasets.
- **test**: attaches `ground_truth.json` frame labels.
- **`dynamic_load`**: `True` = lazily `np.load` each member from the open zip per
  `__getitem__` (low RAM); `False` = decompress the whole zip into RAM up front
  (**OOMs on 7.8 GB** — train.zip is 4.2 GB). **Use `True` here.**

### Axis flow into the model (must stay consistent)
- train item `(10, 32, 2049)` → batched `(B, 10, 32, 2049)` = `(B, crops, T, D)`.
- test item `(nclips, 10, 2049)` → batched `(1, nclips, 10, 2049)`; the runner
  `permute(0, 2, 1, 3)` → `(1, 10, nclips, 2049)` = `(B, crops, T, D)`.
- Every model therefore receives `video = (B, crops, T, D)` and splits
  normal/abnormal at `B//2` (training concatenates the two loaders, **normal-first**).

## Class labels (extension needed for VadCLIP-align / GS-MoE routing)

The current loader yields only **binary** anomaly (0/1). VadCLIP's MIL-Align
(CLASM) and GS-MoE's class-expert routing need the **per-video anomaly class**
(13 UCF types: Abuse, Arrest, …, Vandalism). This needs **no new data** — the
class is encoded in the filename (`Abuse028_x264`, `Arrest001_x264`, …). Plan:
add a filename→class parser and emit `class_labels` from `FeatureDataset`; wire it
through the runner to the models that accept the `class_labels` kwarg.

## CLIP features (planned — unblocks CLIP-TSA / VadCLIP)

Not yet on the Hub. Use the VadCLIP authors' precomputed UCF CLIP ViT-B/16 (512-d)
— OneDrive link in `docs/REPRODUCTION.md` / project memory. Plan: re-upload as
`jinmang2/ucf_crime_clip_vitb16`, then a CLIP `configs/data/ucf_clip.yaml` with
`feature_size: 512`. Confirm the layout (10-crop? frame vs seg32?) on arrival and
adapt the loader if needed. Local extraction is avoided (103 GB raw + CPU decode).

## RAM strategy (7.8 GB system)

- `dynamic_load: true` is **mandatory** (zip stays on disk, one `.npy` read at a time).
- `num_workers`: keep low (2–4) — 4 CPU cores and tight RAM; each worker holds its
  own zip handle.
- **Eval risk**: test features are full-length, so a very long *normal* test video
  is a single large array (the longest run into hundreds of MB). `batch_size=1`
  eval handles it, but the largest few may still spike RAM — chunk them if needed.
- Keep everything on **native ext4** (`~`), never `/mnt/c` (9P = slow).
