# Reproduction Tracker & Paper Analysis

Central log for the unified WSVAD framework (see `../WSAD_INTEGRATION_PLAN.md`).
Every method is decomposed into the 6 slots and matched to its paper's
experimental conditions so results are reproducible, not approximate.

## Status

| Method | Paradigm | Feature | Status | Target UCF AUC | Runner config |
|--------|----------|---------|--------|----------------|---------------|
| MIL (Sultani) | MIL | I3D | ✅ implemented (this branch) | ~0.77 | `runner=mil` |
| RTFM   | MAG      | I3D     | ✅ implemented (this branch) | ~0.843 | `runner=rtfm` |
| MGFN   | MAG      | I3D     | ✅ implemented | ~0.87 | `runner=mgfn` |
| CLIP-TSA | VLM | CLIP | ✅ model done (needs CLIP feature cache to train) | ~0.876 | `runner=clip_tsa` |
| UR-DMU | MEM | I3D | ✅ implemented (this branch) | ~0.87 | `runner=ur_dmu` |
| VadCLIP | VLM | CLIP | ✅ model done (needs CLIP cache + class labels for align) | ~0.88 | `runner=vadclip` |

> Feature-extractor extension plan (I3D → CLIP/VideoMAE/VGGish): `docs/FEATURE_EXTRACTORS.md`.
> Pretrained-weight equivalence + optimization (SDPA/flash, AMP): `docs/WEIGHTS_AND_OPTIMIZATION.md`.

## Common experimental setup (UCF-Crime, I3D)

| Item | Value | Source |
|------|-------|--------|
| Features | I3D 10-crop, **32-segment** at train time | `jinmang2/ucf_crime_tencrop_i3d_seg32` |
| Feature dim | 2048 (+1 appended magnitude → 2049 in loader) | `FeatureDataset.add_magnitude` |
| Train batch | 32 normal + 32 abnormal, concatenated normal-first | `src/runner.py` |
| Optimizer | Adam, lr 1e-3, weight_decay 5e-4 | `configs/runner/default.yaml` |
| Eval | frame-level ROC-AUC; clip score repeated ×16 frames | `src/runner.on_validation_epoch_end` |
| Precision | fp16/AMP recommended on RTX 2060 (`trainer.cls.precision=16-mixed`) | — |

### Data-layout conventions (must preserve)
- **train (seg32 hub):** `(ncrops, T, D)` → batched `(B, ncrops, T, D)`.
- **test (full hub):** `(T, ncrops, D)` → runner/inference permute to `(B, ncrops, T, D)`.
- Models receive `video=(B, ncrops, T, C)` and split normal/abnormal at `B//2`.

## Runner contract (every model must honor)
```
forward(video, abnormal_labels=None, normal_labels=None) -> ModelOutput
  .loss    : scalar (None at inference)
  .scores  : (B, T, 1) per-snippet anomaly score in [0, 1]
```
Training concatenates the normal+abnormal dataloaders; the model splits them
internally (see `force_split` / `self.training`). Keep `drop_last=True`.

---

## RTFM — Robust Temporal Feature Magnitude (ICCV'21)

arXiv 2101.10030 · paradigm: feature-magnitude (MAG)

### Slot mapping
| Slot | RTFM component | Code |
|------|----------------|------|
| 2. temporal encoder | **MTN**: pyramid dilated conv (d=1/2/4) + non-local global context, residual | `MTN`, `NonLocalBlock1D` |
| 4. scoring head | 3-layer MLP (2048→512→128→1) + **top-k magnitude** selection (k=3) | `RTFMForVideoAnomalyDetection` |
| 5. loss | BCE on top-k scores + magnitude separation + smoothness + sparsity | `RTFMLoss` + `src/loss/base` |

### Key idea
Abnormal snippets have **larger feature magnitude**. Select top-k snippets by
magnitude per video; push abnormal top-k magnitude above a margin (100) while
shrinking normal magnitude. This is robust to the weak (video-level) labels.

### Hyperparameters (`configs/runner/rtfm.yaml`)
`feature_size=2048, num_segments=32, k=3, dropout=0.7, margin=100, alpha=1e-4,`
`lambda_smooth=8e-4, lambda_sparse=8e-3`. Paper trains ~50 epochs, Adam lr 1e-3.

### Repro notes
- Cached features are 2049-dim (magnitude appended for MGFN). RTFM **slices off**
  the extra channel (`video[..., :2048]`) and computes its own magnitude on the
  MTN-encoded features — do not feed the appended channel into the MTN.
- `k = num_segments // 10 = 3` (paper convention).

---

## MGFN — Magnitude-Contrastive Glance-and-Focus (AAAI'23)

arXiv 2211.15098 · paradigm: feature-magnitude (MAG) · ✅ pre-existing

### Slot mapping
| Slot | MGFN component | Code |
|------|----------------|------|
| 1.5 amplifier | fuses feature + appended magnitude channel (eq.1–2) | `MGFNFeatureAmplifier` |
| 2. temporal encoder | Glance (global clip transformer) + Focus (local SAC) blocks | `GlanceBlock`, `FocusBlock` |
| 4. scoring head | LN + FC + top-k magnitude selection (k=3) | `MGFNForVideoAnomalyDetection` |
| 5. loss | BCE + magnitude contrastive + smoothness + sparsity | `MGFNLoss` + `src/loss/base` |

### Repro notes
- **Uses** the appended magnitude channel (unlike RTFM): amplifier slices
  `[:2048]` (features) and `[2048:]` (magnitude), fuses with `mag_ratio=0.1`.

---

## Sultani MIL — Real-world Anomaly Detection (CVPR'18)

arXiv 1801.04264 · paradigm: multiple-instance ranking (MIL) · ✅ implemented

### Slot mapping
| Slot | Sultani component | Code |
|------|-------------------|------|
| 2. temporal encoder | none — crops mean-pooled, segments scored independently | — |
| 4. scoring head | 3-layer FC regressor (2048→512→32→1, dropout 0.6) | `SultaniForVideoAnomalyDetection` |
| 5. loss | MIL ranking on bag-max + smoothness + sparsity | `MILRankingLoss` |

### Key idea
The label is only at the *bag* (video) level. Take the max segment score per
bag; an abnormal bag's max should outrank a normal bag's max by margin 1
(hinge). Smoothness + sparsity (λ=8e-5 each) regularize the abnormal curve.
The simplest possible WSVAD method — use it as the sanity check for the shared
dataset + eval pipeline.

### Hyperparameters (`configs/runner/mil.yaml`)
`feature_size=2048, hidden1=512, hidden2=32, dropout=0.6, λ_smooth=λ_sparse=8e-5`.
Paper uses Adagrad; Adam(lr 1e-3) is a fine drop-in on cached features. ~1.07M params.

### Repro notes
- Uses neither the appended magnitude channel nor crops individually: slices
  `[..., :2048]` and mean-pools the 10 crops before the regressor.

---

## CLIP-TSA — CLIP-Assisted Temporal Self-Attention (ICIP'23)

arXiv 2212.05136 · paradigm: VLM · ✅ model implemented (needs CLIP feature cache)

### Slot mapping
| Slot | Component | Code |
|------|-----------|------|
| 1. features | CLIP ViT-B/16 frame features (512-d, text-aligned) | `src/features/clip.py` |
| 2. temporal encoder | **TSA**: Transformer self-attention over snippets | `src.modules.TransformerEncoderLayer` |
| 4. scoring head | RTFM-style top-k feature-magnitude MIL | `src.modules.mil.topk_magnitude_select` |
| 5. loss | RTFM loss + smoothness + sparsity | `RTFMLoss` + `src/loss/base` |

### Repro notes
- Reuses the RTFM magnitude-MIL head/loss on a CLIP-encoded sequence — the only
  new piece is the self-attention encoder. `attn_impl=eager|sdpa` (see optimization doc).
- **Blocked on the CLIP feature cache** (`src/features/clip.py` implemented; run
  extraction once a GPU is free, then point `configs/data` at the 512-d cache).
- 6.63M params.

## UR-DMU — Dual Memory Units w/ Uncertainty Regulation (AAAI'23)

arXiv 2302.05160 · paradigm: memory (MEM) · ✅ implemented

### Slot mapping
| Slot | Component | Code |
|------|-----------|------|
| 2. temporal encoder | distance-biased Transformer self-attention (learnable Gaussian prior) | `DistanceAdj` + `TransformerEncoderLayer` |
| 4. scoring head | dual `MemoryUnit` banks (normal/abnormal) + variational uncertainty latent + MLP cls | `URDMUForVideoAnomalyDetection` |
| 5. loss | MIL BCE + memory-activation MIL + triplet + KL | `_compute_loss` |

### Key idea
Two learnable memory banks specialize to normal vs. abnormal patterns; attention
read over each bank augments the features. A variational latent (reparameterized
mu/logvar + KL) regularizes uncertainty. Memory-activation is supervised so the
abnormal bank fires on abnormal videos (and vice versa).

### Repro notes
- Crops are mean-pooled up front (memory operates per video); the official repo
  averages at score level — align this for exact-AUC reproduction.
- mem_size=60, top-k ratio T/16, triplet margin 1.0; loss weights in config.
  Verify ~0.87 against official hyperparameters when training. 8.47M params.

## VadCLIP — Adapting Vision-Language Models for WSVAD (AAAI'24)

arXiv 2308.11681 · paradigm: VLM · ✅ architecture implemented (binary path runnable now)

Cross-checked against the official repo `nwpu-zxr/VadCLIP` (`src/model.py`,
`src/utils/layers.py`, `src/ucf_train.py`, `src/ucf_option.py`).

### Slot mapping
| Slot | Component | Code | Official |
|------|-----------|------|----------|
| 1. features | CLIP ViT-B/16 frames (512-d) | `src/features/clip.py` | `clip.load("ViT-B/16")` |
| 2. encoder | LGT-Adapter: windowed temporal Transformer + sim-graph GCN + dist-graph GCN | `encode_video` | `temporal` + `gc1..gc4` |
| 3a. text | learnable class prompt table (+ optional CLIP encode) | `text_features`, `load_clip_text_features` | `encode_textprompt` (CoOp + frozen CLIP) |
| 4. head | binary branch (C) + visual-language alignment (A) | `classifier`, alignment block | `logits1`, `logits2` |
| 5. loss | CLAS2 binary MIL + text contrastive (+ CLASM MIL-Align when class labels) | `_compute_loss` | `CLAS2 + CLASM + loss3` |

### Official-vs-ours comparison (sanity check, not bit-exact)
**Matches the official design:**
- LGT-Adapter dual graph: similarity adjacency (cosine, threshold 0.7, softmax =
  `adj4`) + distance adjacency (`exp(-|i-j|/e)` = `DistanceAdj`), each a 2-layer
  GCN `512→256→256`, concat → `linear` (512). Faithful port in `src/modules/graph.py`.
- Windowed local attention mask (block-diagonal, window=8) on the temporal Transformer.
- Binary branch `classifier(vf + mlp2(vf))`; alignment uses binary scores to attend
  visual feats, adds to text features (+`mlp1`), cosine logits `/0.07`. Same flow.
- Losses: CLAS2 (sigmoid top-k `T/16+1` mean → BCE, abnormal=1) and the text
  contrastive (|cos(normal, abnormal_i)| averaged ×0.1) reproduced exactly; CLASM
  (top-k mean per class → softmax CE on normalized multi-hot) implemented.
- Hyperparameters from `ucf_option.py`: width 512, layers 2, head 1, window 8,
  prompt 10/10, num_class 14, AdamW lr 2e-5, MS-LR [4,8]×0.1, 10 epochs.

**Deliberate differences (documented, architecturally equivalent):**
- Temporal block = this repo's pre-norm `TransformerEncoderLayer` (eager/SDPA
  switchable) vs official post-norm QuickGELU `ResidualAttentionBlock`. Same
  capacity; not bit-exact (acceptable — we re-train, not load their checkpoint).
- Text features are a **learnable table** until CLIP-encoded (official encodes
  CoOp prompts through frozen CLIP). `load_clip_text_features()` fills them from
  CLIP when `open_clip` is available — deferred with the CLIP extractor.
- `.scores` (our eval contract) = `sigmoid(binary logits)` (C-branch). The
  A-branch alignment logits are returned for the headline alignment AUC; wiring
  the alignment-based frame score + per-video **class labels** (for CLASM) is the
  remaining data extension before full-AUC reproduction.

### Repro notes / blockers
- Needs the **CLIP feature cache** (512-d) — same blocker as CLIP-TSA.
- Needs a **class-label path** in the dataset (UCF folder name → one of 14
  classes) to activate CLASM; current `FeatureDataset` only yields binary labels.
- 12.61M params (excl. the frozen CLIP text encoder).

## Next-paper queue (analysis before implementation)

1. **MIL (Sultani, P1)** — simplest baseline: per-bag ranking loss + sparsity/
   smoothness on 32-seg I3D. Slot 2 = identity/FC, slot 4 = max-score MIL ranking.
   Good sanity check for the shared eval.
2. **CLIP-TSA (P3)** — first VLM target. Requires CLIP feature extractor
   (`scripts/extract_features.py --backbone clip`). Slot 2 = temporal self-attn.
3. **VadCLIP (P4)** — dual-branch (visual binary + language-image align). Adds the
   text branch (slot 3a) and MIL-Align loss; needs CLIP **text-aligned** features.

> For each new paper: fill a slot-mapping table, list the exact hyperparameters,
> add `configs/runner/<name>.yaml`, register the model, then verify forward/
> backward via `tests/test_models.py` before chasing the AUC number.
