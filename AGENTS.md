<!-- Generated: 2026-06-12 | Updated: 2026-06-12 -->

# wsad — Weakly-Supervised Video Anomaly Detection

## Purpose
Research framework for **weakly-supervised video anomaly detection (WSVAD)** on
UCF-Crime. The current codebase trains anomaly-scoring heads on **pre-extracted,
cached I3D features** (10-crop, 32-segment) using PyTorch Lightning + Hydra +
HuggingFace `transformers`/`datasets`. The headline trick: feature extraction is
decoupled (offline, inference-only) from head training (online, light) so the
whole training loop fits in 6 GB VRAM (RTX 2060 target).

> **Major refactor in progress.** This branch (`feat/unified-wsvad-framework`,
> cut from `origin/13-run-test`) is being rebuilt toward the design in
> [`WSAD_INTEGRATION_PLAN.md`](WSAD_INTEGRATION_PLAN.md): a config-driven matrix of
> `feature_backbone × temporal_encoder × MIL_head × loss × {text,audio} branch`,
> where every paper becomes a config rather than a fork. Read that file before
> making structural changes. Sections below describe the **current** state; the
> plan describes the **target**.

## Current state (what actually exists)
- **Five working models** (all registered via `src/registry.py` `MODELS`, all
  honor the shared runner contract): `Sultani-MIL` (`runner=mil`), `RTFM`
  (`runner=rtfm`), `MGFN` (`runner=mgfn`), `CLIP-TSA` (`runner=clip_tsa`,
  needs CLIP feature cache), `UR-DMU` (`runner=ur_dmu`).
- **Slot/registry foundation** (`src/registry.py`: `MODELS`, `ENCODERS`, `HEADS`,
  `LOSSES`, `FEATURE_EXTRACTORS`) for the plan's config-driven matrix.
- **Shared modules** (`src/modules/`): `attention.py` (MHSA + Transformer block
  with switchable `eager`/`sdpa` kernels — FlashAttention via SDPA), `mil.py`
  (shared top-k magnitude selection), `compat.py` (pretrained-weight numerical-
  equivalence helpers).
- **Feature-backbone abstraction** (`src/features/`): `FeatureExtractor` ABC +
  registered `i3d` + `clip` (implemented) and `videomae`/`vggish` (planned stubs).
- **Unified inference/eval** in `src/inference.py` (pure torch, no Lightning);
  **contract tests** in `tests/` (18 passing: 5 models + attention equivalence).
- **I3D feature extractor** (`src/i3d.py`) + offline extraction script (`scripts/extract_features.py`).
- Training data is loaded as cached I3D features from HF Hub
  (`jinmang2/ucf_crime_tencrop_i3d_seg32`), never raw video at train time.
- Docs: `docs/REPRODUCTION.md` (per-paper slot maps + repro conditions),
  `docs/FEATURE_EXTRACTORS.md` (backbone plan), `docs/WEIGHTS_AND_OPTIMIZATION.md`
  (weight equivalence + SDPA/AMP/compile).

## Key Files
| File | Description |
|------|-------------|
| `run.py` | Hydra entrypoint. Instantiates model + Lightning runner + Trainer from config, calls `trainer.fit`. |
| `make_gt_ucf.py` | Builds frame-level ground-truth JSON for UCF-Crime test set from temporal annotations. One-shot script (hardcoded Colab Drive save path). |
| `WSAD_INTEGRATION_PLAN.md` | The design doc / method registry driving the rewrite. Source of truth for target architecture. |
| `requirements.txt` | pip deps (transformers, lightning, hydra-core, decord, pytorchvideo, etc.). |
| `README.md` | Conda/CUDA env setup incl. building `decord` with GPU support. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `src/` | All library code: dataset, runner, model, loss, I3D backbone (see `src/AGENTS.md`) |
| `configs/` | Hydra config tree: data / runner / trainer (see `configs/AGENTS.md`) |
| `scripts/` | Offline tooling: feature extraction, checkpoint conversion (see `scripts/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- **Entrypoint is `run.py` via Hydra**: run as `python run.py runner=mgfn` (selects `configs/runner/mgfn.yaml`). Override anything with dotted CLI args.
- The model contract is HF-style: a model is `_locate(model_class)(instantiate(model_config))`, must accept `video=...` (+ optional `*_labels`) and return a dataclass `ModelOutput` with a `.loss` and `.scores` field. The Lightning runner depends on this contract — preserve it when adding models.
- Training feeds **concatenated normal+abnormal batches** (`torch.cat` of the two dataloaders); the model splits them internally by `batch_size//2`. Keep `drop_last=True` on train loaders or the split breaks.
- When implementing the plan's slot/registry refactor, keep `src/i3d.py` import-compatible (existing configs reference it) — add a re-export shim if you move it.

### Testing Requirements
- No test suite exists yet. Validate by running `python run.py runner=mgfn` and confirming `valid/rec_auc` logs to wandb. MGFN should reproduce ~0.86–0.87 UCF-Crime AUC.
- Any refactor of RTFM/MGFN must match known-good AUC before proceeding (plan Phase P0).

### Common Patterns
- Hydra `defaults:` composition; `_target_` / `_locate` for instantiation.
- HF `PreTrainedModel` + `PretrainedConfig` for models; `ModelOutput` dataclasses for outputs.
- Features carry an appended **magnitude channel** (2048 → 2049) added in the dataset, consumed by magnitude-based heads.

## Dependencies

### External
- **PyTorch 2.5.1 / CUDA 12.1** — see `README.md` for the exact conda recipe (incl. source-built `decord` GPU).
- **lightning** — training loop (`pl.LightningModule`, `pl.Trainer`).
- **hydra-core / omegaconf** — config composition and CLI.
- **transformers** — `PreTrainedModel`/`PretrainedConfig`/`ModelOutput` base classes.
- **datasets / huggingface_hub** — pulls cached features and raw videos from `jinmang2/*` HF repos.
- **decord, pytorchvideo** — video decoding + I3D backbone (extraction only).
- **wandb** — experiment logging (loss curves, ROC, score-vs-label images).

### Data (HF Hub, owner `jinmang2`)
- `jinmang2/ucf_crime` — raw videos (for re-extraction).
- `jinmang2/ucf_crime_tencrop_i3d_seg32` — I3D 10-crop, 32-segment (default train/eval source).
- `jinmang2/ucf-crime-tencrop-i3d` — I3D 10-crop, full-length snippets.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
