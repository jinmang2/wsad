<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-06-12 | Updated: 2026-06-12 -->

# configs

## Purpose
Hydra configuration tree. `run.py` composes `default.yaml`, which pulls one
config from each of `data/`, `runner/`, and `trainer/` (plus trainer callbacks
and loggers). Selecting a method = selecting a `runner/*.yaml`. This is the
backbone of the plan's "every paper is a config, not a fork" goal.

## Key Files
| File | Description |
|------|-------------|
| `default.yaml` | Top-level Hydra `defaults:` list (data=default, runner=default, trainer=default, callbacks=all, logger=all) + `wandb_key`. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `data/` | Dataset/loader config. `default.yaml`: batch_size 16, frames_per_clip 16, `revision: tushar-n`, dynamic_load. `ucf.yaml` inherits default. |
| `runner/` | Model + optimizer selection. `default.yaml`: runner class + Adam (lr 1e-3, wd 5e-4). `mgfn.yaml`: `MGFNForVideoAnomalyDetection` + full `MGFNConfig`. |
| `trainer/` | `lightning.pytorch.Trainer` args (`default.yaml`: max_epochs 1000, precision 32-true, accelerator auto). |
| `trainer/callbacks/` | `all.yaml` composes `lrmonitor` + `model_checkpoint`. |
| `trainer/logger/` | `all.yaml` composes `wandb` (`WandbLogger`, project `anomaly_detection_on_video`, run name templated from data/runner choices + timestamp). |

## For AI Agents

### Working In This Directory
- **Add a method** = add `runner/<name>.yaml` with `defaults: [default]`, a `model_class` (dotted import path), and a `model_config` (`_target_` to the `PretrainedConfig` + its fields). Run with `python run.py runner=<name>`.
- `model_config` is `instantiate()`-d (needs `_target_`); `model_class` is `_locate()`-d (plain dotted string, no `_target_`). Don't mix the two conventions.
- Wandb run name uses `${hydra:runtime.choices.data}-${hydra:runtime.choices.runner}` — keep config filenames meaningful, they become experiment names.
- Override anything at the CLI: `python run.py runner=mgfn data.batch_size=8 trainer.cls.precision=16-mixed`. For 6 GB, prefer mixed precision.

### Testing Requirements
- `python run.py runner=<name> --cfg job` prints the composed config without running — use it to validate a new config resolves before training.

### Common Patterns
- Hydra `defaults: [default]` inheritance per leaf config.
- `_target_` for instantiable objects (configs, Trainer, logger, callbacks); dotted strings for classes resolved by `_locate`.

## Dependencies

### Internal
- `runner/*.yaml` reference classes in `src/models/*` and `src/runner.py`.
- `data/*.yaml` fields are consumed by `src/runner.setup` / dataloaders and `src/dataset.build_feature_dataset`.

### External
- `hydra-core`, `omegaconf`, `lightning` (Trainer/loggers/callbacks targets).

<!-- MANUAL: -->
