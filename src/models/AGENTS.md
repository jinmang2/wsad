<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-06-12 | Updated: 2026-06-12 -->

# models

## Purpose
Anomaly-detection model implementations, each packaged HuggingFace-style
(`PretrainedConfig` + `PreTrainedModel` + `ModelOutput`). Selected at runtime by
`configs/runner/*.yaml` via `model_class` / `model_config`.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | `from .mgfn import *` — re-exports the MGFN package. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `mgfn/` | **MGFN** (AAAI'23) Glance-and-Focus + magnitude-contrastive head (see `mgfn/AGENTS.md`) |
| `rtfm/` | **RTFM** (ICCV'21) MTN encoder + top-k feature-magnitude head — implemented; registered as `"rtfm"`. Analysis in `docs/REPRODUCTION.md`. |

## For AI Agents

### Working In This Directory
- **A model package must export** a `*Config(PretrainedConfig)` and a `*ForVideoAnomalyDetection(PreTrainedModel)` whose `forward(video, abnormal_labels=None, normal_labels=None)` returns a `ModelOutput` with `.loss` (None at inference) and `.scores` (`[B, T, 1]`). This is what `run.py` instantiates and what `src/runner.py` consumes.
- **Register every model** with `@MODELS.register("name")` from `src/registry.py` and import the package in `src/models/__init__.py` so the name resolves. RTFM/MGFN are the two reference implementations — copy their package layout (`configuration_*.py`, `modeling_*.py`, `__init__.py`) for new papers.
- New models slice off the appended magnitude channel themselves if they don't need it (RTFM uses `video[..., :feature_size]`); MGFN consumes it.
- Inputs are **2049-dim** cached features (2048 I3D + 1 magnitude), 10-crop, segmented to T=32 at train time.

### Testing Requirements
- Each model needs a `dummy_inputs` property (see MGFN: `torch.randn(32,10,32,2049)`) for shape-checking.
- End-to-end: add a runner config and run `python run.py runner=<name>`; confirm AUC matches the paper within tolerance before claiming completion.

### Common Patterns
- HF `PreTrainedModel` subclass with `config_class` and `base_model_prefix`.
- Training vs inference split governed by a `force_split` flag + `self.training` (see MGFN `magnitude_selection_and_score_prediction`).
- Loss assembled in `forward` from `src/loss` components when labels are present.

## Dependencies

### Internal
- `src/loss/` — all loss terms.
- `src/runner.py` — the consumer of model outputs.

### External
- `torch`, `transformers` (`PreTrainedModel`, `PretrainedConfig`, `ModelOutput`), `einops`.

<!-- MANUAL: -->
