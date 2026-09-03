# Feature Extractor Research: architectures, training data, and pipeline cost

Companion to `RESEARCH_ROADMAP.md` and `FEATURE_EXTRACTORS.md`. Answers four
questions for the WSVAD feature-extractor → anomaly-detection pipeline:
1. What architecture does each backbone use?
2. What data was each trained on, and how much?
3. If we *re-build* this, how much training data do we actually need?
4. How far can the `extractor → detector` pipeline be optimized for production?

Grounded in this repo's extractors (`src/features/*.py`) + the 2024–25 literature.

---

## 1 + 2. Backbone architectures and their training data

| Backbone (repo) | Architecture | Pretrain data | Scale | Supervision | Output |
|---|---|---|---|---|---|
| **I3D** (`i3d_gowtham.py`) | Inflated 3D ConvNet — here the **I3D-ResNet50** (`I3Res50`) variant, 2D ResNet inflated to 3D + optional Non-Local blocks | **ImageNet** (2D inflate) → **Kinetics-400** | K400 ≈ **240k** train clips, 400 action classes, ~10s each | Supervised (action labels) | `(T, 10crop, 2048)` per 16-frame snippet |
| **CLIP ViT-B/16** (`clip.py`) | ViT-B/16 image encoder (12-layer transformer, 16×16 patches) contrastively aligned to a text encoder | OpenAI: **WIT-400M**; this repo uses open_clip **`laion2b_s34b_b88k`** = **LAION-2B** (2.3B EN image-text pairs, 34B samples seen) | 400M – 2B **image-text pairs** | Weak/contrastive (web alt-text) | `(frame, 512)` → mean-pool to 32 seg |
| **VideoMAE v1** (`videomae.py`, `MCG-NJU/videomae-base`) | ViT-B with **tubelet** embedding, masked-autoencoder pretrain (90% masking) | **Kinetics-400** (frames only, **no labels**) | ~240k videos, 800–1600 epochs | **Self-supervised** (reconstruction) | `(T, 768)` per 16-frame clip |
| **VideoMAEv2** (loader TODO) | Same MAE + **dual masking**, scaled to ViT-g (**1.0B params**) | **UnlabeledHybrid** = Kinetics + Something-Something + AVA + WebVid2M + self-collected IG | **~1.35M** videos | Self-supervised + label distill | `(T, 768/1024/1408)` |
| **InternVideo2** (frontier) | Progressive: masked modeling + crossmodal contrastive + next-token; video encoder up to **6B params** | Multimodal video-audio-speech caption corpus | billion-scale; 6B model = **256×A100 for ~18+14+3 days** | Multi-stage multimodal | text-aligned video features |
| **VGGish** (`vggish.py`, AV phase) | VGG-style CNN on log-mel spectrogram | AudioSet | ~2M audio clips | Supervised (audio tags) | `(T_audio, 128)` |

**The structural split that drives method choice** (from `FEATURE_EXTRACTORS.md`):
`text_aligned` is the single most important property. CLIP and InternVideo2 live
in a shared image/text space → they unlock the VLM/text-branch heads (CLIP-TSA,
VadCLIP, TPWNG, WSVAD-CLIP). I3D / VideoMAE are visually strong but have **no
text space** → visual-only heads only (Sultani, RTFM, MGFN, UR-DMU, GS-MoE).

**Performance context** (UCF-Crime frame AUC, from roadmap): I3D-only ceiling
~86; CLIP + multi-backbone fusion → ~87–88; the lever is the *backbone*, not the
head. One independent study even reports I3D (90%) > plain ViT (86%) on the same
LSTM head — i.e. a generic ViT is **not** automatically better than I3D; the win
comes from *video-pretrained* or *text-aligned* backbones, not transformers per se.

---

## 3. If we re-build this — how much data do we actually need?

**Key reframing: there are two completely different "training" budgets, and people
conflate them.**

### 3a. Training the *feature extractor* (the backbone) — you almost never do this
The backbones above cost **240k–1.35M videos** (or 400M–2B image-text pairs) and
**hundreds of GPU-days**. Reproducing one from scratch is a foundation-model
project, not a VAD project. **Don't.** The correct move (and what this repo does)
is to **download pretrained frozen weights** and run inference-only extraction.

If you *must* adapt a backbone to a new domain (e.g. CCTV/AIHub), the options in
ascending cost:
- **Zero training** — use frozen I3D/CLIP/VideoMAE as-is. Default. Works because
  Kinetics/LAION already cover human-activity + object semantics.
- **Linear probe / LoRA** on the backbone — hundreds–few thousand labeled clips.
- **Self-supervised continued pretrain** (VideoMAE-style, no labels) on your raw
  domain footage — tens of thousands of *unlabeled* clips is enough to shift the
  representation, since MAE needs no annotation. This is the realistic "rebuild"
  path for a new camera domain.
- **Full re-pretrain** — only if you are publishing a new foundation model.

### 3b. Training the *anomaly-detection head* (what WSVAD actually trains)
This is tiny by comparison and is **the entire point of the offline-feature
design**. The standard benchmark training sets:

| Dataset | Train videos | Labels | Notes |
|---|---|---|---|
| **UCF-Crime** | ~1,610 (800 normal + 810 anomaly) | **video-level only** (weak) | the standard WSVAD benchmark |
| **XD-Violence** | ~3,954 | video-level + audio | largest; AV |
| **ShanghaiTech (weakly)** | 437 | video-level | smaller |

So to stand up a working detector on a **new** domain you need on the order of
**~1–4k weakly-labeled videos** (just "this video contains an anomaly somewhere"
vs "normal") — *not* frame-level annotation. The features are cached once; the MIL
head trains from scratch in minutes-to-hours on a single 6 GB GPU (this repo
trains the light heads at batch 32 in AMP on an RTX 2060).

**Practical recipe for a new deployment:**
1. Frozen pretrained backbone (CLIP for text-branch methods, or I3D for the
   proven visual baseline). Optional unlabeled MAE continued-pretrain if the
   domain is very off-distribution.
2. Collect ~1–3k videos with **video-level** normal/anomaly tags only.
3. Extract features once → cache `.npy`.
4. Train a RTFM/UR-DMU/VadCLIP head on the cache.

---

## 4. Pipeline optimization: `extractor → detector` for production

### Where the cost actually is
The detector head is **negligible** (a few-layer MLP/attention over cached
features — runs at thousands of FPS). **~95%+ of wall-clock and FLOPs is the
feature extractor.** Optimization = optimize extraction + I/O; the head is free.

Worst offender in this repo's faithful I3D path (`i3d_gowtham.py`): it does
**ffmpeg → per-frame JPG → PIL → 10-crop** = 10× forward passes per snippet, plus
disk round-trips. Bit-exact for reproduction, **terrible** for production.

### Optimization levers (ROI-ordered, production)

| Lever | Effect | Cost / caveat |
|---|---|---|
| **Drop 10-crop → 1 center crop** | **~10× extraction speedup** | tiny AUC drop; the single biggest win |
| **In-memory decode (decord), no JPG dump** | removes ffmpeg+disk round-trip | already how `clip.py`/`videomae.py` read |
| **fp16 / AMP inference** | ~2× throughput, ½ VRAM | repo already does `.half()` |
| **Batch snippets across the clip** | saturates GPU | bounded by VRAM |
| **`torch.compile` + SDPA/Flash attention** | fuses kernels (`WEIGHTS_AND_OPTIMIZATION.md`) | ~1e-3 numeric drift — fine for prod, not for the bit-exact gate |
| **Sparse/strided sampling** (skip frames, larger stride) | linear speedup | coarser temporal resolution |
| **Smaller/faster backbone** (X3D, MobileNet-3D, CLIP ViT-B over ViT-L) | big | accuracy trade |
| **Cache features once, reuse across heads** | amortizes extraction to ~0 for experiments | the repo's whole design |
| **TensorRT / ONNX export, INT8** | edge-deployment throughput | engineering + calibration |

### Real-time numbers from the literature
- Offline SOTA methods ignore latency entirely; they assume features are
  pre-extracted. That is fine for batch/forensic use, **not** for live CCTV.
- The **Real-Time WSVAD** line (WACV 2024) reports **~23 FPS** end-to-end
  (drops to ~21 FPS adding frame-level inference) and an **86.9% AUC with a ~6.4s
  decision window** — i.e. real-time is achievable but you pay ~1–3 AUC points and
  must design the extractor + temporal window for streaming, not 10-crop offline.

### Practical deployment guidance
- **Forensic / batch** (search archived footage): keep accuracy-max config
  (10-crop, ViT-L/InternVideo2, fusion). Throughput doesn't matter; cache once.
- **Live / edge** (alerting): single-crop, fp16, lightweight backbone (X3D /
  CLIP-B / VideoMAE-B), strided sampling, sliding decision window. Target the
  ~20–30 FPS regime; accept ~86–87 AUC.
- **Hybrid (recommended)**: cheap streaming detector for first-pass alerts → on
  trigger, re-run the heavy accurate config on the flagged clip for confirmation.
  Gets live latency *and* offline accuracy where it counts.

---

## Sources
- RTFM (arXiv 2101.10030); DAKD (2406.02831); VAD survey (2405.10347);
  "Evolution of VAD: DNN→MLLM" (2507.21649) — per `RESEARCH_ROADMAP.md`.
- VideoMAE V2 (CVPR 2023, arXiv 2303.16727) — UnlabeledHybrid ~1.35M videos, ViT-g 1B.
- InternVideo2 (ECCV 2024, arXiv 2403.15377) — 6B encoder, 256×A100 multi-stage.
- CLIP/WIT-400M; LAION-2B (arXiv 2210.08402 / 2212.07143).
- Real-Time WSVAD (WACV 2024) — ~23 FPS, 86.9% AUC / 6.4s window.
- Comparative I3D vs ViT on LSTM head (IJISAE) — I3D 90% > ViT 86%.
