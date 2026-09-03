<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-06-12 | Updated: 2026-06-12 -->

# src

## Purpose
The library core: the slot registry, the Accelerate trainer, inference/eval, the
modular data layer, shared neural modules, feature backbones, losses, and the ten
model packages. Everything the framework touches lives here.

## Key Files
| File | Description |
|------|-------------|
| `registry.py` | `MODELS / ENCODERS / HEADS / LOSSES / FEATURE_EXTRACTORS` registries. Slots register by string name; configs reference names. |
| `trainer.py` | `WSVADTrainer` (Accelerate) — explicit dual normal/abnormal loop, AMP, auto `class_labels`, per-epoch frame-level ROC/PR-AUC eval, checkpoint save. The training hub. |
| `inference.py` | Pure-torch model build (from runner config + ckpt), single-video scoring, frame-level AUC. No Lightning. |
| `i3d.py` | I3D ResNet-50 backbone (`build_i3d_feature_extractor`). Offline extraction only, not at train time. |
| `gtransforms.py` | Group video transforms consumed by `data/video.py` (extraction). |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `data/` | Modular data layer: `labels` (14-class taxonomy + filename parser), `manifest` (script-free), `features` (zip + manifest datasets), `video` (10-crop extraction). |
| `features/` | Feature backbones (slot 1): `i3d`+`clip` implemented, `videomae`/`vggish` stubs. |
| `modules/` | Shared blocks: `attention` (MHSA, eager/SDPA), `graph` (GCN), `mil` (top-k magnitude), `compat` (weight-equivalence). |
| `loss/` | Loss terms: ranking, magnitude (rtfm), mil, tgs, smoothness, sparsity, contrastive (see `loss/AGENTS.md`). |
| `models/` | The ten model packages (see `models/AGENTS.md`). |

## For AI Agents

### The runner contract (every model honors)
`forward(video, abnormal_labels=None, normal_labels=None[, class_labels]) ->`
`ModelOutput` with `.loss` (None at inference) and `.scores` of shape `(B, T, 1)`.
- Training concatenates the normal + abnormal loaders **normal-first**; the model
  splits at `B//2` (`force_split` / `self.training`).
- `video` is `(B, ncrops, T, C)`; cached features carry an appended **magnitude
  channel** (2048→2049). MGFN consumes it; others slice `[..., :feature_size]`.
- Models accepting `class_labels` (VadCLIP, GS-MoE) get a `(B,)` int class id
  (0=Normal, 1..13); the trainer passes it automatically.
- Eval AUC: per-clip `.scores` repeated `frames_per_clip` (16) frames, then
  `roc_curve`. BN-WVAD's score is an unbounded rank-score (fine for AUC).

### Testing
- Offline contract tests in `tests/` (synthetic, no downloads): `test_models.py`
  (forward/backward for all models), `test_trainer.py` (full train+eval loop),
  `test_attention.py`, `test_data.py`. Run with `uv run pytest`.
- New model: copy an existing package (`configuration_*.py`, `modeling_*.py`,
  `__init__.py`), register with `@MODELS.register("name")`, import it in
  `models/__init__.py`, add `configs/runner/<name>.yaml`, add to `MODEL_NAMES`.

### Common Patterns
- Models are HF `PreTrainedModel`s returning `ModelOutput` dataclasses.
- Reuse `src/modules` (attention/graph/mil) and `src/loss` rather than duplicating.

## Dependencies

### Internal
- `trainer.py` → `data`, `inference`, model `forward`.
- `models/*` → `loss`, `modules`, `registry`.
- `features/*` → `i3d`, `data.video`, `registry`.

### External
- `torch`, `accelerate`, `transformers`, `numpy`, `scikit-learn`, `huggingface_hub`,
  `einops`. Extraction only: `decord`, `pytorchvideo`, `open_clip`.

<!-- MANUAL: -->
