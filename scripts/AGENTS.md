<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-06-12 | Updated: 2026-06-12 -->

# scripts

## Purpose
Offline, run-once tooling that sits outside the training loop: I3D feature
extraction from raw video, and conversion of official checkpoints into this
repo's HF-style module layout. Nothing here is imported by `run.py`.

## Key Files
| File | Description |
|------|-------------|
| `extract_features.py` | CLI (`argparse`) that loads `jinmang2/ucf_crime` raw videos, runs the I3D extractor over 10-crop clips (`src.dataset.TenCropVideoFrameDataset` + `src.i3d`), saves per-video `*_i3d.npy`, then uniformly segments into 32-segment features. Handles >1 GB videos by chunking (`seg_len = 16*188`). This is the plan's "extract once, cache to .npy" enabler. |
| `convert_official_to_hf.py` | `convert(state_dict)` remaps official MGFN checkpoint keys (`stages.*`, `to_tokens`, `to_mag`, `to_logits`, `fc`) onto this repo's `backbone.amplifier.*` / `backbone.layers.*` naming. |

## For AI Agents

### Working In This Directory
- **Run from repo root** (`python scripts/extract_features.py ...`) — it imports `src.*`, so the repo root must be on `PYTHONPATH`.
- Extraction is **inference-only and GPU-light** but **time/disk heavy**; it caches and skips already-extracted files. Default device is `cuda`.
- The plan (§5) wants this generalized to `--backbone {i3d,clip,videomae,vggish}`. When adding backbones, keep the 10-crop / 32-segment conventions and the `*_<backbone>.npy` naming so `src/dataset.py` and `make_gt_ucf.py` stay compatible.
- `convert_official_to_hf.py` is tightly coupled to module names in `src/models/mgfn/modeling_mgfn.py` — update it whenever those modules are renamed.

### Testing Requirements
- Verify a single extracted `.npy` has shape `(nclips, 10, 2048)` (pre-segment) or `(32, ...)` after `segment_features`.
- For the converter: load an official checkpoint, run `convert`, then `MGFNForVideoAnomalyDetection.load_state_dict(...)` with `strict=False` and inspect missing/unexpected keys.

### Common Patterns
- `argparse` CLIs with sensible defaults pointing at `jinmang2/*` HF repos.
- Idempotent caching: skip work if the output file already exists.

## Dependencies

### Internal
- `src.i3d.build_i3d_feature_extractor`, `src.dataset.TenCropVideoFrameDataset`.

### External
- `torch`, `datasets`, `numpy`, `decord`, `PIL`, `tqdm`, `huggingface_hub`.

<!-- MANUAL: -->
