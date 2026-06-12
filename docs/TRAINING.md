# Training: framework choice & RTX 2060 feasibility

Two questions: (1) is PyTorch Lightning still the right harness given the
normal+abnormal dual-loader pattern, and (2) which models actually train on a
6 GB RTX 2060 (often sharing the GPU with another job) + 7.8 GB system RAM.

> Status: the recommendation below is **implemented** — `src/trainer.py`
> (`WSVADTrainer`, Accelerate) + `train.py` (Hydra entrypoint). The legacy
> Lightning path (`run.py` / `src/runner.py`) is kept. Actual AUC training on the
> real feature cache is still pending (data handled in a separate session).

## Implemented trainer (`src/trainer.py`)

`WSVADTrainer` is the explicit Accelerate loop:
- **dual loader** = `zip(normal_loader, abnormal_loader)`; each step concatenates
  normal-first and the model splits at `B//2`.
- **AMP** via `Accelerator(mixed_precision=...)`; `train.py` maps the Hydra
  `trainer.cls.precision` (`16-mixed` → `fp16`) onto it.
- **class labels**: models whose `forward` accepts `class_labels` (VadCLIP,
  GS-MoE) automatically receive a `(B,)` int per-video class id (0=Normal, 1..13);
  both models were standardized to that single format.
- **eval**: frame-level ROC/PR-AUC each epoch via `src.inference.score_feature`
  (handles the test layout + BN-WVAD's unbounded rank-scores).
- **dataset-agnostic**: `fit(train_datasets, test_dataset, epochs)` takes datasets,
  so `tests/test_trainer.py` drives the full train+eval loop on **synthetic
  features offline** (verified for the plain and class_labels paths).

Run: `python train.py runner=rtfm` (Accelerate) — or the legacy
`python run.py runner=rtfm` (Lightning).

## 1. Training-framework review

### What the project needs
- **Dual dataloader**: each step concatenates a `normal` batch and an `abnormal`
  batch (normal-first); the model splits at `B//2`. This is the one non-standard
  requirement — the harness must feed two loaders per step.
- **Determinism / control**: weight-equivalence checks (`docs/WEIGHTS_AND_OPTIMIZATION.md`)
  and the eager/SDPA switch want an explicit, inspectable loop.
- **6 GB AMP**: fp16/bf16 mixed precision is mandatory; cached features mean the
  backbone costs ~0 VRAM, so only the light head is on-GPU.
- **Low RAM**: tight control over workers / prefetch.

### Options

| Harness | Dual-loader fit | Boilerplate | Control | 2026 status | Verdict |
|---------|-----------------|-------------|---------|-------------|---------|
| **Lightning** (current) | `train_dataloader` returns a tuple → `CombinedLoader`; works but is the historical friction point | low | medium (hooks hide the loop) | actively maintained, still common in research | OK, already works |
| **HF `Trainer`** | awkward — single-loader oriented; dual stream needs `get_train_dataloader` override + a collator hack | low | low (very opinionated) | very common, but for standard supervised | poor fit |
| **HF `Accelerate`** | trivial — you write the loop, just `accelerator.prepare(...)` each loader | low–med | high | common, lightweight | strong fit |
| **Lightning `Fabric`** | trivial — keep your loop, `fabric.setup` + `setup_dataloaders` | low–med | high | maintained alongside Lightning | strong fit |
| **Raw PyTorch + AMP** | trivial — `zip(normal, abnormal)` | medium (checkpoint/log by hand) | highest | n/a | fine, most effort |

### Recommendation
Lightning is **still actively used and the current `runner.py` works** — no need to
rip it out to make progress. But for *this* project (dual-loader is core,
determinism matters, RAM/VRAM are tight, models are tiny), the `LightningModule`
abstraction earns little. When we touch the training loop, **migrate to a lean
explicit loop on `Accelerate` (or Lightning `Fabric`)**: zip the two loaders
yourself, keep AMP + checkpoint + a thin logger, and drop the `CombinedLoader`
friction entirely. `Fabric` if we want to stay in the Lightning ecosystem;
`Accelerate` if we prefer the HF stack (we already depend on `transformers`).

Concretely, the migration is small: `src/runner.py`'s `training_step` /
`validation_step` / `*_dataloader` become ~80 lines of explicit loop. Defer until
we actually start training (per your call) — flagged, not done.

## 2. RTX 2060 (6 GB) trainability — per model

Memory math: cached features → backbone ~0 VRAM at train time. The on-GPU cost is
**Adam state** (fp32 ≈ 16 B/param: weight+grad+m+v) + activations (small at T=32) +
~0.8 GB CUDA context. AMP (`precision=16-mixed`) roughly halves weight/activation
memory. The user's *other* job often holds ~3.5 GB → assume **~2.6 GB free** when
shared, full 6 GB when not.

| Model | Params | Adam state (fp32) | Est. train VRAM (AMP) | Fits 6 GB alone | Fits w/ other job (~2.6 GB) | Notes |
|-------|--------|-------------------|-----------------------|-----------------|------------------------------|-------|
| Sultani (`mil`) | 1.07 M | 17 MB | ~1 GB | ✅ | ✅ | trivial |
| CLIP-TSA | 6.63 M | 106 MB | ~1.2 GB | ✅ | ✅ | needs CLIP feats |
| UR-DMU | 8.47 M | 135 MB | ~1.3 GB | ✅ | ✅ | |
| VadCLIP | 12.61 M | 202 MB | ~1.4 GB | ✅ | ✅ | needs CLIP feats + class labels |
| RTFM | 24.72 M | 396 MB | ~1.6 GB | ✅ | ✅ | |
| MGFN | 28.65 M | 458 MB | ~1.7 GB | ✅ | ✅ | |
| GS-MoE | 80.25 M | 1.28 GB | ~3–3.5 GB | ✅ | ⚠️ tight | 13 experts; reduce `hidden_size`/`num_experts`, or free the GPU |

**Verdict:** every model except GS-MoE is comfortable even while sharing the GPU.
GS-MoE fits alone but is tight (~3 GB) when the other job is running — shrink it
(`hidden_size=512` or fewer experts) or train it when the GPU is free.

### System RAM (7.8 GB), not VRAM
Often the *real* limit. `dynamic_load=true` is mandatory (see `docs/DATA.md`);
keep `num_workers` at 2–4. Eval of the longest full-length test videos can spike
RAM — chunk if needed.

### Training time (rough, 2060)
Tiny heads on cached seg32 (~1600 train videos, batch 32 → ~50 steps/epoch):
- Sultani/RTFM/MGFN/UR-DMU/CLIP-TSA/VadCLIP: **minutes to ~1 h** for 50–100 epochs.
- GS-MoE: **several hours** (13 experts run sequentially; paper reports ~3 h on an
  A4500, so expect longer on a 2060).

## 3. Future papers — 2060 outlook

| Candidate | Feature | Local train on 2060? | Note |
|-----------|---------|----------------------|------|
| TPWNG / WSVAD-CLIP / SCL | CLIP | ✅ (light heads) | need CLIP feature cache |
| MIST / MSL / ECU | I3D | ✅ but slower | iterative pseudo-label self-training |
| HL-Net / MACIL-SD | I3D + VGGish | ✅ compute | needs XD-Violence + audio pipeline (different dataset) |
| LAVAD / Holmes-VAD | VLM+LLM | ❌ | 7B+ model resident at inference — infeasible on 6 GB (API/quant only) |

Headline: the **6 GB box is fine for essentially every feature-based head** (the
extract-once / train-light-heads design is what makes this work). The only hard
blockers are training-free LLM methods, which are a different paradigm entirely.
