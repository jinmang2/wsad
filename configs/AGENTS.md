<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-06-12 | Updated: 2026-06-12 -->

# configs

## Purpose
Hydra configuration tree. `train.py` composes `default.yaml`, which pulls one
config from each of `data/`, `runner/`, and `trainer/`. Selecting a method =
selecting a `runner/*.yaml`. This is the backbone of "every paper is a config,
not a fork".

## Key Files
| File | Description |
|------|-------------|
| `default.yaml` | Top-level Hydra `defaults:` (data=default, runner=default, trainer=default). |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `data/` | Dataset/loader config: batch_size, frames_per_clip, `revision`, `dynamic_load`, cache_dir. |
| `runner/` | Model + optimizer selection. `default.yaml`: Adam (lr 1e-3, wd 5e-4). One `<method>.yaml` per model (`mil`, `rtfm`, `mgfn`, `ur_dmu`, `s3r`, `bn_wvad`, `gs_moe`, `clip_tsa`, `vadclip`, `tpwng`). |
| `trainer/` | Accelerate trainer fields: `max_epochs`, `precision` (`no` / `16-mixed` / `bf16`). Consumed by `train.py` → `src.trainer.WSVADTrainer`. |

## For AI Agents

### Working In This Directory
- **Add a method** = add `runner/<name>.yaml` with `defaults: [default]`, a
  `model_class` (dotted import path, `_locate`-d) and a `model_config` (`_target_`
  to the `PretrainedConfig` + fields, `instantiate`-d). Run `python train.py runner=<name>`.
- `model_config` needs `_target_`; `model_class` is a plain dotted string. Don't mix.
- Override anything at the CLI: `python train.py runner=mgfn data.batch_size=8
  trainer.precision=16-mixed runner.model_config.attn_impl=sdpa`. Prefer 16-mixed on 6 GB.

### Testing
- `python train.py runner=<name> --cfg job` prints the composed config without
  running — validate a new config resolves before training.

### Common Patterns
- Hydra `defaults: [default]` inheritance per leaf config.
- `_target_` for instantiable objects (the model config); dotted strings for the
  model class resolved by `_locate`.

## Dependencies

### Internal
- `runner/*.yaml` reference model classes in `src/models/*`.
- `data/*.yaml` fields feed `src/trainer.build_datasets` / `src.data.build_feature_dataset`.
- `trainer/default.yaml` fields feed `train.py` → `WSVADTrainer`.

### External
- `hydra-core`, `omegaconf`.

<!-- MANUAL: -->
