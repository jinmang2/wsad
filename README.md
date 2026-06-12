# WSAD — Unified Weakly-Supervised Video Anomaly Detection

A **config-driven benchmark framework** for weakly-supervised video anomaly
detection (WSVAD) on UCF-Crime. Every method decomposes into the same slots —
`feature backbone × temporal encoder × scoring head × loss (× text/audio branch)`
— so each paper is a **config, not a fork**. Ten methods are implemented against a
single runner contract and trained with one explicit Accelerate loop.

Designed to run on a single **RTX 2060 (6 GB)**: features are extracted once and
cached, then light heads train on the cache (the backbone costs ~0 VRAM at train
time). See `WSAD_INTEGRATION_PLAN.md` for the full design.

## Implemented models

All registered in `src.registry.MODELS`, all honor the same contract
(`forward(video, abnormal_labels, normal_labels) -> .loss, .scores`):

| Runner | Method | Venue | Paradigm | Feature | Cross-checked vs official |
|--------|--------|-------|----------|---------|---------------------------|
| `mil` | Sultani-MIL | CVPR'18 | MIL ranking | I3D | — |
| `rtfm` | RTFM | ICCV'21 | feature magnitude | I3D | — |
| `mgfn` | MGFN | AAAI'23 | magnitude-contrastive | I3D | (pre-existing) |
| `ur_dmu` | UR-DMU | AAAI'23 | dual memory | I3D | — |
| `s3r` | S3R | ECCV'22 | dictionary / sparse | I3D | ✅ louisYen/S3R |
| `bn_wvad` | BN-WVAD | 2023 | BatchNorm-DFM | I3D | ✅ cool-xuan/BN-WVAD |
| `gs_moe` | GS-MoE | ICCV'25 | mixture-of-experts (SOTA ~91.6) | I3D | paper (code unreleased) |
| `clip_tsa` | CLIP-TSA | ICIP'23 | VLM | CLIP | — |
| `vadclip` | VadCLIP | AAAI'24 | VLM dual-branch | CLIP | ✅ nwpu-zxr/VadCLIP |
| `tpwng` | TPWNG | CVPR'24 | VLM + pseudo-label | CLIP | paper (code unreleased) |

I3D models train locally today; CLIP models (`clip_tsa`, `vadclip`, `tpwng`) need a
cached CLIP feature set. Per-paper slot maps and reproduction conditions:
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

## Quickstart

```bash
# train any model on the cached I3D features (Accelerate loop)
python train.py runner=rtfm
python train.py runner=gs_moe trainer.precision=16-mixed data.batch_size=32   # AMP on 6 GB

# evaluate / score a single video's cached feature
python -m src.inference --runner rtfm --checkpoint <ckpt> --device cuda
python -m src.inference --runner mgfn --feature path/to/video_i3d.npy
```

Override anything from the CLI (Hydra): `python train.py runner=ur_dmu data.batch_size=16
runner.model_config.attn_impl=sdpa trainer.max_epochs=100`.

## Architecture

```
raw video ──(offline, once)──► feature backbone ──► cache .npy
                                                       │ (train-time input)
                                                       ▼
   [ temporal encoder ] ─► [ scoring head ] ─► anomaly score s ∈ [T]
        (+ text / audio branch)                       │
                                                       ▼
                                          [ loss ] (+ self-training)
```

- **`src/registry.py`** — `MODELS / ENCODERS / HEADS / LOSSES / FEATURE_EXTRACTORS`
  registries; configs reference slots by string name.
- **`src/modules/`** — shared blocks: `attention` (MHSA with switchable
  eager/SDPA-FlashAttention kernels), `graph` (GCN for LGT-Adapter), `mil`
  (top-k magnitude), `compat` (pretrained-weight numerical-equivalence helpers).
- **`src/trainer.py`** — `WSVADTrainer` (Accelerate): explicit dual
  normal/abnormal loop, AMP, frame-level ROC/PR-AUC eval, checkpointing.

## Repository layout

```
train.py                 # Accelerate training entrypoint (Hydra)
src/
  registry.py            # slot registries
  trainer.py             # WSVADTrainer (Accelerate)
  inference.py           # pure-torch model build + AUC eval + single-video scoring
  data/                  # labels (14-class), manifest (script-free), feature & video datasets
  features/              # feature backbones: i3d, clip (+ videomae/vggish stubs)
  modules/               # attention, graph, mil, compat
  loss/                  # ranking, magnitude, mil, tgs, smoothness, sparsity
  models/                # mil, rtfm, mgfn, ur_dmu, s3r, bn_wvad, gs_moe, clip_tsa, vadclip, tpwng
  i3d.py, gtransforms.py # I3D backbone + video transforms (extraction)
configs/                 # Hydra: data / runner / trainer
scripts/                 # offline: extract_features.py, convert_official_to_hf.py
docs/                    # REPRODUCTION, DATA, TRAINING, FEATURE_EXTRACTORS, WEIGHTS_AND_OPTIMIZATION
tests/                   # offline contract + trainer tests (synthetic, no downloads)
```

## Data

Cached features are pulled from the Hugging Face Hub (owner `jinmang2`):

| Dataset | Content | Role |
|---------|---------|------|
| `ucf_crime` | raw videos (103 GB) | extraction source |
| `ucf-crime-tencrop-i3d` | full-length I3D 10-crop | variable-T |
| `ucf_crime_tencrop_i3d_seg32` | I3D 10-crop, 32-segment | **train/eval (default)** |

The data layer (`src/data/`) is manifest-driven and script-free (the deprecated HF
loader script is replaced). The fine-grained anomaly class is parsed for free from
the filename (`class_id`), enabling VadCLIP / GS-MoE class supervision. Storage,
RAM strategy, and the CLIP-feature plan: [`docs/DATA.md`](docs/DATA.md).

## Environment

Conda + CUDA (PyTorch, transformers, datasets, hydra-core, accelerate, einops,
scikit-learn). Training on cached features needs no video stack. **Feature
extraction** additionally needs `decord` (GPU build) + `pytorchvideo` (I3D) or
`open_clip` (CLIP) — see `docs/FEATURE_EXTRACTORS.md`. Install: `pip install -r
requirements.txt`.

## Docs

- [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) — per-paper slot maps, official-vs-ours notes, repro conditions
- [`docs/DATA.md`](docs/DATA.md) — data pipeline, the 3 datasets, axis conventions, storage/RAM
- [`docs/TRAINING.md`](docs/TRAINING.md) — Accelerate trainer + per-model RTX 2060 feasibility
- [`docs/FEATURE_EXTRACTORS.md`](docs/FEATURE_EXTRACTORS.md) — I3D → CLIP/VideoMAE/VGGish plan
- [`docs/WEIGHTS_AND_OPTIMIZATION.md`](docs/WEIGHTS_AND_OPTIMIZATION.md) — weight equivalence + SDPA/AMP/compile
