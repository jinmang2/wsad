# scripts/

Run everything with `WSAD_DATA=~/data/wsad uv run python scripts/<...>.py`.

## Top-level (core pipeline)
| script | purpose |
|---|---|
| `run_matrix.py` | `{feature-variant × head}` comparison matrix. Per-paper recipe in `HEADS`; trains via `WSVADTrainer.fit_steps`; per-head eval via `src.eval_matrix`; wandb-logged; resumable. See `docs/WSVAD_REPRO_RECIPES.md`. |
| `gpu_queue.sh` | Serial GPU reproduction queue (8 GB → one heavy run at a time): memory heads on 1024-d DeepMIL + the i3d 2048-d heads on **clean** `i3d_mgfn_seg32` + cosine. Run detached (`nohup setsid`). |
| `build_correct_i3d.py` | Build consistent I3D features (`features/i3d/{train,test}`) from a source extraction. |
| `extract_features.py` | Backbone-agnostic extraction dispatcher (`--backbone clip/i3d/...`). |
| `extract_i3d_gowtham.py` | Faithful Gowtham/Tushar-N I3D 10-crop extractor (raw video → `(T,10,2048)`). |
| `convert_official_to_hf.py` | Official ckpt → repo state_dict converter (imported by run_matrix + verify/). |
| `convert_i3d_caffe2.py` | I3D Caffe2 `.pkl` → torch `.pth`. |
| `prepare_clip_features.py` | Sort VadCLIP `UCFClipFeatures` into `clip/{train,test}`. |
| `build_manifest.py` | Build the unified dataset manifest. |
| `train_archive_probe.py` | Standalone trainer probe (train=_archive seg32, test=test.zip). Reproduced MGFN 0.8667 (lr 5e-5, batch 16, 12 ep, seed 0). |

## verify/ — fidelity checks (output-match vs official)
`verify_{rtfm,mgfn,vadclip,ur_dmu,bn_wvad,s3r,clip_tsa,checkpoints,i3d_extract}.py`,
`eval_{mgfn_local,vadclip_ucf}.py`. Confirm ported models match official weights/outputs.

## diag/ — diagnostics & superseded probes (kept for reference)
`stabilize.py` (per-step AUC probe), `smoke_train_{mgfn,vadclip}.py` (train-loop smoke),
`diag_i3d_rtfm.py`, `grid_i3d_rtfm.py`, `vadclip_oracle.py`, `validate_extraction.py`.
Function now largely covered by `run_matrix.py` + `WSVADTrainer.fit_steps`.
