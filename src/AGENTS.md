<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-06-12 | Updated: 2026-06-12 -->

# src

## Purpose
The library core. Holds the Lightning training runner, the cached-feature
dataset, the (single) anomaly-detection model, loss functions, and the I3D
feature backbone used by the offline extractor. Everything the training loop
touches lives here.

## Key Files
| File | Description |
|------|-------------|
| `runner.py` | `VideoAnomalyDetectionRunner(pl.LightningModule)` — training/validation steps, optimizer, dataloaders, frame-level AUC computation, wandb plotting. The orchestration hub. |
| `dataset.py` | `build_feature_dataset` + `FeatureDataset` (cached `.npy`-in-`.zip` loader, appends magnitude channel, splits normal/abnormal) and `TenCropVideoFrameDataset` (raw-video → 10-crop clips, used only by the extractor). |
| `i3d.py` | I3D ResNet-50 backbone (`I3Res50`, `build_i3d_feature_extractor`). Loads weights from HF `jinmang2/test_video_fe`. Used **offline** for feature extraction, not at train time. |
| `gtransforms.py` | Group video transforms (`GroupResize`, `GroupTenCrop`, `ToTensorTenCrop`, `GroupStandardizationTenCrop`, `LoopPad`) consumed by `TenCropVideoFrameDataset`. |
| `__init__.py` | Empty package marker. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `loss/` | Loss functions: smoothness, sparsity, contrastive, MGFN (see `loss/AGENTS.md`) |
| `models/` | Model implementations: MGFN (done), RTFM (empty stub) (see `models/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- **The runner is the integration contract.** `training_step` concatenates normal+abnormal feature batches and calls `model(video=..., abnormal_labels=..., normal_labels=...)`; `validation_step` calls `model(video=...)` and reads `outputs.scores`. New models must honor this signature.
- **AUC is computed at frame level** in `on_validation_epoch_end`: per-clip scores are repeated `frames_per_clip` (16) times to align with frame-level ground truth, then `roc_curve`/`auc`. Don't change the repeat factor without matching the extraction stride.
- **Magnitude channel:** `FeatureDataset.add_magnitude` appends an L2-norm channel → feature dim 2048 becomes 2049. MGFN's amplifier slices `[:2048]` vs `[2048:]`. Any new head must expect 2049-dim cached features (or strip the channel).
- `dynamic_load` toggles lazy `.npy` reads from the zip vs eager load into RAM.

### Testing Requirements
- Smoke-test via `python run.py runner=mgfn` from repo root. Confirm `valid/rec_auc` appears in wandb and loss decreases.
- When adding a model, first verify the dataset/runner plumbing with a dummy forward (`MGFNPreTrainedModel.dummy_inputs` shows the expected `(32,10,32,2049)` shape).

### Common Patterns
- Cached features shaped `[T, 10, D]` (10-crop); crops are mean-reduced inside the model, not the loader.
- Models are HF `PreTrainedModel`s returning `ModelOutput` dataclasses.
- Plan target (`WSAD_INTEGRATION_PLAN.md` §5): this dir will gain `features/`, `encoders/`, `branches/`, `heads/`, `selftrain/` packages and a `models/registry.py` that assembles slots from config. New code should anticipate that decomposition.

## Dependencies

### Internal
- `runner.py` → `dataset.build_feature_dataset`.
- `models/mgfn` → `loss` (`MGFNLoss`, `SparsityLoss`, `TemporalSmoothnessLoss`).
- `dataset.py` → `gtransforms`.
- `scripts/extract_features.py` (sibling dir) → `i3d`, `dataset.TenCropVideoFrameDataset`.

### External
- `torch`, `lightning`, `transformers`, `numpy`, `scikit-learn`, `matplotlib`, `wandb`, `decord`, `pytorchvideo`, `huggingface_hub`, `einops`.

<!-- MANUAL: -->
