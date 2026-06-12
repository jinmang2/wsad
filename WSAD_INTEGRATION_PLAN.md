# WSAD Integrated Framework — Plan & Method Registry

> **Purpose of this file.** This is a self-contained design doc for extending the
> [`jinmang2/wsad`](https://github.com/jinmang2/wsad) repo into a *unified* weakly-supervised
> video anomaly detection (WSVAD) benchmark framework. It is written to be parsed by a coding
> agent: every method is tagged with its paradigm, feature backbone, modality, approx. UCF-Crime
> AUC, RTX-2060 feasibility, and integration target. Numbers are *approximate* and depend heavily
> on the feature backbone/crop — treat them as relative, not absolute.

---

## 0. TL;DR for an agent

- **Existing repo state:** I3D feature pipeline + RTFM + MGFN, hydra configs (`configs/{data,runner,trainer}`), `runner.py`, `dataset.py`, `make_gt_ucf.py`, `extract_features.py`, `src/i3d.py`, `src/models/{rtfm,mgfn}`, `src/loss/`.
- **Goal:** turn it into `feature_backbone × temporal_encoder × MIL_head × loss × {text_branch, audio_branch}` — a config-driven matrix so every method below is a config, not a fork.
- **Hardware:** single **RTX 2060 (6 GB)**, moving to **WSL2 + CUDA**. Strategy = *extract features once, cache to `.npy`, train light heads on cached features.* Everything stays in 6 GB this way.
- **Hard blocker:** training-free LLM methods (LAVAD / Holmes-VAD) need a 7B+ VLM/LLM resident at inference → **not feasible locally on 6 GB**. Use API or 4-bit quant, or defer.
- **What's new vs the user's collected list:** **GS-MoE (2025, ~91.6% SOTA)**, **SCL (2025, 88.47%)**, **WSVAD-CLIP (2025)**, plus the **VideoMAE feature backbone** and the **training-free LLM** category.

---

## 1. Context & constraints

| Item | Value |
|---|---|
| Primary dataset | **UCF-Crime** (frame-level AUC, weakly labeled) |
| Secondary | XD-Violence (audio-visual, AP), ShanghaiTech (AUC) — optional |
| GPU | RTX 2060, **6 GB VRAM** → AMP/fp16 mandatory, batch ≤ 32 |
| Env | WSL2 (Ubuntu) + CUDA, PyTorch, `timm`, `open_clip` |
| Core trick | **Decouple feature extraction (offline, inference-only) from head training (online, light).** This is what makes 6 GB viable for nearly every method here. |

### Data sources (already on HF Hub — owner: `jinmang2`)

| Dataset | Content | Use |
|---|---|---|
| [`jinmang2/ucf_crime`](https://huggingface.co/datasets/jinmang2/ucf_crime) | **raw videos** | P2: re-extract CLIP / VideoMAE features |
| [`jinmang2/ucf_crime_tencrop_i3d_seg32`](https://huggingface.co/datasets/jinmang2/ucf_crime_tencrop_i3d_seg32) | I3D 10-crop, **32-segment** | P0/P1: RTFM/MGFN reproduce (Sultani 32-seg convention) |
| [`jinmang2/ucf-crime-tencrop-i3d`](https://huggingface.co/datasets/jinmang2/ucf-crime-tencrop-i3d) | I3D 10-crop, full-length snippets | variable-T methods (UR-DMU etc.) |

→ **P0–P1 (RTFM/MGFN reproduce) can start immediately from the seg32 features — no extraction needed.** Only P2+ (CLIP/VideoMAE) requires running an extractor on the raw-video set. Load all three via `datasets.load_dataset(...)` / `huggingface_hub.snapshot_download`.

---

## 2. Method registry

**Paradigm legend:** `MIL` = multiple-instance ranking · `MAG` = feature-magnitude · `PL` = pseudo-label / self-training · `MEM` = memory units · `VLM` = CLIP/text-prompt · `MoE` = mixture-of-experts · `AV` = audio-visual · `GEN` = generative aug · `LLM` = training-free LLM.

**Modality:** V = visual-only · AV = audio+visual.

**2060 feasibility:** ★★★ trivial on cached feats · ★★ ok but heavier preprocessing · ✗ infeasible locally.

| # | Method | Year/Venue | Paradigm | Feature | Mod | ~UCF AUC | 2060 | In repo? | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **MIL (Sultani)** | CVPR'18 | MIL | C3D/I3D | V | ~77 | ★★★ | add | The baseline. Ranking loss + sparsity/smoothness. Implement first as sanity check. |
| 2 | **GCN (Noise Cleaner)** | CVPR'19 | PL | TSN/C3D | V | ~82 | ★★ | add | Graph label-noise cleaner → plug-and-play action classifier. |
| 3 | **Motion-Aware (M.A.)** | BMVC'19 | MIL | C3D+flow | V | ~79 | ★★ | add | Adds optical-flow motion attention. Flow extraction = extra preprocessing. |
| 4 | **HL-Net** | ECCV'20 | AV | I3D+VGGish | AV | (XD AP) | ★★ | add | XD-Violence dataset paper. Holistic + localized + audio. |
| 5 | **WSAL** | TIP'21 | MIL | I3D | V | ~85 | ★★★ | add | Localizing anomalies, mutual-guidance. |
| 6 | **RTFM** | ICCV'21 | MAG | I3D | V | 84.3 | ★★★ | ✅ done | Top-k feature-magnitude. Already implemented. |
| 7 | **MIST** | CVPR'21 | PL | C3D/I3D | V | 82.3 | ★★ | add | Multiple-instance self-training + sparse-continuous sampling. |
| 8 | **MACIL-SD** | MM'22 | AV | I3D+VGGish | AV | (XD AP) | ★★ | add | Modality-aware contrastive + self-distillation. |
| 9 | **MSL** | AAAI'22 | PL | VideoSwin | V | 85.6 | ★★ | add | Multi-sequence learning transformer, self-training. |
| 10 | **MGFN** | AAAI'23 | MAG | I3D | V | ~87 | ★★★ | ✅ done | Glance-and-focus + magnitude-contrastive. Already implemented. |
| 11 | **ECU** | CVPR'23 | PL | I3D | V | ~86 | ★★★ | add | Completeness + uncertainty of pseudo-labels. |
| 12 | **CLIP-TSA** | ICIP'23 | VLM | **CLIP** | V | 87.6 | ★★★ | add | **First VLM target.** CLIP visual feats + temporal self-attention + MIL. Simplest CLIP entry. |
| 13 | **HSN** | '23 | MIL | I3D | V | ~86 | ★★★ | add | Human-scene network + self-rectifying loss. |
| 14 | **UR-DMU** | AAAI'23 | MEM | I3D | V | 87.0 | ★★★ | add | Dual memory (normal/abnormal) + uncertainty regulation. Strong, popular baseline. |
| 15 | **UMIL** | CVPR'23 | MIL+VLM | CLIP | V | 86.8 | ★★★ | add | Unbiased MIL, debiases context. |
| 16 | **SSRL** | ECCV'22 | MIL | I3D | V | 87.4 | ★★★ | add | Scale-aware spatio-temporal relation. |
| 17 | **CLAV** | CVPR'23 | MIL | I3D | V | ~86.5 | ★★★ | add | Context-motion relational ("look around"). |
| 18 | **Dance-SA** | '21 | MIL | I3D | V | ~85 | ★★★ | add | CRF-as-self-attention view. |
| 19 | **TCN** | — | enc | any | V | — | ★★★ | add | Generic temporal-conv encoder; treat as a *temporal_encoder* option, not a full method. |
| 20 | **VadCLIP** ⭐ | AAAI'24 | VLM | **CLIP** | V | 88.0 | ★★★ | add | **Flagship VLM.** Dual-branch: visual binary + fine-grained language-image alignment + prompts. |
| 21 | **TPWNG** | CVPR'24 | VLM | CLIP | V | 87.8 | ★★★ | add | Text prompt + normality guidance. |
| 22 | **GS-MoE** ⭐ | 2025 | MoE | I3D/CLIP | V | **~91.6** | ★★ | add | **Current SOTA.** Mixture-of-experts guided by Gaussian splatters. Splatting preproc = heavier. |
| — | **SCL** | 2025 | VLM | CLIP | V | 88.5 | ★★★ | add* | Semantic consistency across temporal scales. (not in user list) |
| — | **WSVAD-CLIP** | 2025 | VLM | CLIP | V | 87.9 | ★★★ | add* | Temporally-aware + prompt learning. (not in user list) |
| — | **LAVAD** | CVPR'24 | LLM | VLM caption+LLM | V | ~80 | ✗ | defer | Training-free; needs BLIP/LLaVA + LLM resident. API/quant only on 6 GB. |
| — | **Holmes-VAD** | 2024 | LLM | VLM instruct | V | — | ✗ | defer | Instruction-tuned VLM; local infeasible. |

**Feature backbones to support** (offline extractors): `C3D`, `I3D` (✅ exists), `VideoSwin`, **`CLIP ViT-B/16`** (enables #12,15,20,21), **`VideoMAE`** (the user's `++`; ViT-B fine on 2060, ViT-L tight). Audio: `VGGish` for AV methods (#4,8).

---

## 3. Gap analysis — collected vs latest

- ✅ **Well covered (2018–2024):** MIL, GCN, M.A., HL-Net, WSAL, RTFM, MIST, MACIL-SD, MSL, MGFN, ECU, CLIP-TSA, HSN, UR-DMU, UMIL, VadCLIP, TPWNG, Dance-SA, CLAV, SSRL.
- 🆕 **Missing latest (2025):** **GS-MoE** (SOTA ~91.6), **SCL** (88.5), **WSVAD-CLIP** (87.9). Add these to stay current.
- 🧱 **Missing categories:**
  - **VideoMAE feature backbone** — strong modern features, drop-in upgrade over I3D.
  - **Training-free LLM** (LAVAD/Holmes-VAD) — a *different paradigm* worth a section, but **hardware-blocked locally**.
  - **Generative aug** (GV-VAD, 2025) — optional, advanced.
- 🎯 **Recommended "new models to actually run"** given 2060: **CLIP-TSA → VadCLIP → UR-DMU → GS-MoE**, all on cached CLIP/VideoMAE features.

---

## 4. Unified architecture

Every method above decomposes into the same 6 slots. Build the framework around these interfaces; each paper = a config selecting slots.

```
                 ┌─────────────────────────────────────────────┐
 raw video ──►   │ 1. FEATURE BACKBONE (offline, cached .npy)   │  C3D/I3D/VideoSwin/CLIP/VideoMAE
                 │    out: snippet feats X ∈ [T, D]            │  (+ VGGish audio for AV)
                 └─────────────────────────────────────────────┘
                              │ (load cached feats — this is the train-time input)
                              ▼
                 ┌─────────────────────────────────────────────┐
                 │ 2. TEMPORAL ENCODER                          │  GCN | Transformer/TSA | TCN |
                 │    X → H ∈ [T, D']                           │  glance-focus | memory units | MoE
                 └─────────────────────────────────────────────┘
                              ▼
   (optional)    ┌──────────────────────┐   ┌──────────────────────────────┐
   text prompts ─│ 3a. TEXT BRANCH (CLIP│   │ 3b. AUDIO BRANCH (VGGish)    │ (AV only)
                 │ text enc, frozen)    │   └──────────────────────────────┘
                 └──────────────────────┘
                              ▼
                 ┌─────────────────────────────────────────────┐
                 │ 4. MIL / SCORING HEAD                        │  top-k MIL | feature-magnitude |
                 │    H (+text/audio) → anomaly score s ∈ [T]  │  attention | memory-read | align
                 └─────────────────────────────────────────────┘
                              ▼
                 ┌─────────────────────────────────────────────┐
                 │ 5. LOSS                                      │  MIL ranking + sparsity/smoothness |
                 │                                              │  magnitude-contrastive | pseudo-label CE |
                 │                                              │  contrastive/self-distill | MIL-Align (VLM)
                 └─────────────────────────────────────────────┘
                              ▼
                 ┌─────────────────────────────────────────────┐
                 │ 6. (optional) PSEUDO-LABEL / SELF-TRAIN loop │  MIST, MSL, ECU, GCN
                 └─────────────────────────────────────────────┘
```

**Registry pattern.** Use a `@register("name")` decorator per slot so configs reference strings:

```yaml
# configs/runner/vadclip.yaml
backbone:  clip_vitb16          # selects cached feature dir
encoder:   { name: transformer, layers: 2, dim: 512 }
text_branch: { name: clip_text, prompts: learnable, n_ctx: 8 }
head:      { name: dual_branch_align }
loss:      [ mil_ce, mil_align, smooth, sparse ]
self_train: null
```

---

## 5. Repo integration (target layout)

```
src/
  features/
    base.py            # FeatureExtractor ABC: extract(video) -> [T, D]
    i3d.py             # (move existing src/i3d.py here)
    clip.py            # NEW: open_clip ViT-B/16 frame features
    videomae.py        # NEW: VideoMAE snippet features
    vggish.py          # NEW: audio (AV methods)
  encoders/
    __init__.py        # register: gcn, tcn, transformer_tsa, glance_focus, memory, moe
  branches/
    text_clip.py       # frozen CLIP text encoder + (learnable) prompts
    audio.py
  heads/
    mil_topk.py  magnitude.py  attention.py  memory_read.py  align.py
  loss/
    ranking.py  magnitude_contrastive.py  pseudo_label.py  self_distill.py  mil_align.py
  models/
    rtfm/  mgfn/                      # EXISTING
    registry.py                       # assembles slots from config
  selftrain/
    loop.py            # MIST/MSL/ECU/GCN pseudo-label iteration
  dataset.py  runner.py               # EXISTING — extend to load any cached backbone
extract_features.py    # add --backbone {i3d,clip,videomae,vggish}
configs/
  data/    ucf_i3d.yaml  ucf_clip.yaml  ucf_videomae.yaml  xd_av.yaml
  runner/  mil.yaml rtfm.yaml mgfn.yaml clip_tsa.yaml vadclip.yaml ur_dmu.yaml gs_moe.yaml ...
  trainer/ default.yaml
```

Backwards-compat: keep `src/i3d.py` re-export shim so existing RTFM/MGFN configs don't break.

---

## 6. Implementation roadmap (phased, 2060-aware)

| Phase | Deliverable | Methods unlocked | Risk |
|---|---|---|---|
| **P0** | Refactor existing RTFM/MGFN into the slot/registry abstraction; reproduce current AUC | RTFM, MGFN | low — pure refactor, must match old numbers |
| **P1** | `MIL (Sultani)` baseline + unified eval (`make_gt_ucf` reuse) | MIL | low |
| **P2** | **CLIP feature extractor** + cache UCF-Crime → `.npy` | (enabler) | med — extraction time, but inference-only |
| **P3** | **CLIP-TSA** (simplest VLM) on CLIP feats | CLIP-TSA | low |
| **P4** | **VadCLIP** ⭐ dual-branch + text prompts + MIL-Align loss | VadCLIP, TPWNG, SCL, WSVAD-CLIP | med — text branch + align loss |
| **P5** | **UR-DMU** memory-unit head | UR-DMU | low |
| **P6** | Self-training loop → MIST/MSL/ECU/GCN | 4 methods | med — iterative pseudo-labels |
| **P7** | **VideoMAE** backbone; re-run P1–P5 for feature ablation | all, new feats | med |
| **P8** | **GS-MoE** (SOTA) — MoE head; Gaussian-splat preproc | GS-MoE | high — splatting cost on 6 GB |
| **P9** | XD-Violence + audio branch → HL-Net, MACIL-SD | 2 AV methods | med |
| **P10** | *(stretch)* training-free LLM via API → LAVAD | LAVAD | blocked locally |

**Order rationale:** P0–P1 lock the abstraction against known-good numbers; P2–P5 deliver the headline "VLM upgrade" (the user's main interest) cheaply; later phases broaden coverage.

---

## 7. Feature backbone strategy (the 6 GB linchpin) — **expanded**

> This section is the open design question. Feature choice drives *both* accuracy and which methods
> are even applicable (text-aligned backbones unlock the VLM branch; others don't).

### 7.1 Backbone comparison

| Backbone | Pretrain | Unit | Dim | Text-aligned? | UCF role | 2060 extract |
|---|---|---|---|---|---|---|
| **C3D** | Sports1M | 16-frame clip | 4096 | ✗ | legacy (Sultani'18) | ★★★ |
| **I3D-R50** ⭐ | Kinetics-400 | 16-frame snippet | **1024** | ✗ | **de-facto standard** (RTFM/MGFN/UR-DMU) | ★★★ (already done) |
| **VideoSwin** | Kinetics | clip | 1024 | ✗ | MSL et al. | ★★ |
| **CLIP ViT-B/16** ⭐ | WIT (image-text) | **per-frame** | **512** | ✅ **yes** | unlocks VLM (VadCLIP/CLIP-TSA/TPWNG/WSVAD-CLIP) | ★★★ |
| **VideoMAE / v2 ViT-B** | K400/UnlabeledHybrid | 16-frame clip | **768** | ✗ | strong modern visual feats | ★★ (B ok, fp16) |
| VideoMAEv2 ViT-L/g | UnlabeledHybrid | clip | 1024 / 1408 | ✗ | SOTA visual, heavy | ✗/✗ (L tight, g no) |
| **InternVideo2** | multimodal | clip | 768+ | ✅ (has aligned text) | strongest ViFM; 1B/6B | ✗ locally (use precomputed/API) |
| VGGish (audio) | AudioSet | 0.96 s | 128 | ✗ | AV methods (XD: HL-Net, MACIL-SD, AVadCLIP) | ★★★ |

### 7.2 The key design insight — **text alignment gates the method set**

- **CLIP-family features live in the same space as the CLIP text encoder.** That is *why* VadCLIP / CLIP-TSA / TPWNG / WSVAD-CLIP use CLIP and not I3D — their "language branch" (prompts → text embeddings → image-text alignment / MIL-Align) only works on a text-aligned backbone.
- **VideoMAE / I3D / VideoSwin are visually stronger but NOT text-aligned.** On them you can run *visual-only* heads (RTFM, MGFN, UR-DMU, GS-MoE), but a VadCLIP-style alignment branch has no matching text space. To combine "VideoMAE visual power + text branch" you'd need a **bridge** (e.g., a learned projection into CLIP space, or use InternVideo2 which ships an aligned text encoder). Treat that as a research contribution, not a freebie.
- **Practical split:**
  - *VLM / text-branch methods* → extract **CLIP ViT-B/16** (and optionally ViT-L/14 if disk allows).
  - *Visual-only heads* → compare **I3D (have it) vs VideoMAE-B** as a clean feature-ablation.
  - *AV methods (XD-Violence)* → add **VGGish** audio.

### 7.3 Extraction conventions (match the existing features)

- **Snippet:** 16 consecutive frames → one feature vector (I3D/VideoMAE). CLIP is per-frame; aggregate frames→snippet by mean or keep frame-level.
- **10-crop:** 4 corners + center, ×2 flip → 10 views per snippet, shape `[T, 10, D]`. The user's I3D feats already use this. Mean over crops at train time, or keep for crop-robustness (RTFM uses all 10).
- **Segmentation — two conventions, support both:**
  - **32-segment** (Sultani/RTFM/MGFN): each video uniformly split into 32 segments → `[32, D]` (or `[32,10,D]`). Matches `ucf_crime_tencrop_i3d_seg32`.
  - **Full-length T snippets** (UR-DMU and newer): keep all snippets, sample/pad in the loader. Matches `ucf-crime-tencrop-i3d`.
- Record per-video metadata so heads stay backbone-agnostic.

### 7.4 Storage & manifest

- Extract **once**, store `[T, D]` (or `[T,10,D]`) per video as `.npy`/`.h5`. Train loads these → near-zero GPU for the backbone at train time.
- `features/manifest.json`: `video_id → {backbone → {path, T, D, crops, fps, segmented:32|full}}`. Configs reference a backbone *name*, never a path.
- Disk note: 10-crop CLIP for all 1900 UCF videos is non-trivial GB — start **single-crop CLIP** to validate P3/P4, add 10-crop only if it moves AUC.

### 7.5 Recommended extraction order

1. **(have it)** I3D seg32 + full → P0/P1.
2. **CLIP ViT-B/16, single-crop, frame-level** → unlocks P3 (CLIP-TSA) + P4 (VadCLIP). *Highest ROI extraction.*
3. **VideoMAE-B** → P7 feature-ablation vs I3D on visual-only heads.
4. **VGGish** (only if doing XD-Violence / AV in P9).
5. *(stretch)* CLIP ViT-L/14 or precomputed InternVideo2 feats if chasing SOTA.

---

## 8. Evaluation protocol

- **UCF-Crime:** frame-level **AUC** (primary). Reuse `make_gt_ucf.py`.
- Report **abnormal-subset AUC** and **FAR@normal** as secondary.
- **XD-Violence:** frame-level **AP**.
- **Headline output:** one matrix table `{I3D, CLIP, VideoMAE} × {all methods}` → makes the feature-vs-method contribution explicit (this is the framework's main selling point).

---

## 9. References

- VadCLIP — https://arxiv.org/abs/2308.11681 · CLIP-TSA — https://arxiv.org/abs/2212.05136
- UR-DMU — https://arxiv.org/abs/2302.05160 · RTFM — arXiv 2101.10030 · MGFN — arXiv 2211.15098
- GS-MoE (2025) — https://openreview.net/forum?id=rrdNQZRHEm
- LAVAD (CVPR'24, training-free) — https://lucazanella.github.io/lavad/
- WSVAD-CLIP (2025) — https://www.mdpi.com/2313-433x/11/10/354
- Sultani MIL (CVPR'18) — UCF-Crime dataset — https://www.crcv.ucf.edu/projects/real-world/
- **Feature backbones:** VideoMAEv2 — https://github.com/OpenGVLab/VideoMAEv2 · InternVideo2 (ECCV'24) — https://github.com/OpenGVLab/InternVideo
- **AVadCLIP** (audio-visual CLIP, 2025) — https://arxiv.org/html/2504.04495
- **Survey:** Video Anomaly Detection in 10 Years — https://arxiv.org/html/2405.19387v1

> **Disclaimer:** AUC values are approximate, aggregated from papers/leaderboards, and vary with
> feature backbone, crop, and split. Verify against each official repo before citing.