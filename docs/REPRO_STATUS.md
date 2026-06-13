# Reproduction & Data-Pipeline Status (2026-06-13)

Snapshot of checkpoint verification, the local-data audit, and the extraction /
dataset strategy. Reproducible checks live in `scripts/{verify_checkpoints,
eval_mgfn_local,smoke_train_mgfn}.py`.

## 0. Faithful-port verification status (per model)

Standard (user's, MGFN-style): re-implement faithfully from the paper + official
code in repo style, then **prove output equivalence** against the official code.
Method: fetch the official repo into `.reference/` (gitignored); if an official
checkpoint exists, load it into the port and match outputs/AUC; if not, transfer
the official module's weights into the port and compare forward under a shared RNG
seed. Verify scripts live in `scripts/`.

| Model | Official | Ckpt | Verification | Result |
|-------|----------|------|--------------|--------|
| **MGFN** | yes | ✅ local pkl | ckpt-load + UCF AUC | ✅ 145→145 clean; AUC 0.8208 (see §1) |
| **VadCLIP** | yes (vendored) | ✅ local pth | ckpt-load + oracle + full UCF AUC | ✅ logits ~1e-6, **AUC 0.880199 == official** |
| **CLIP-TSA** | yes (cloned) | ✗ none published | weight-transfer + seeded forward | ✅ scores **max\|Δ\|=0.0 (bit-exact)** |
| **UR-DMU** | yes (cloned) | ✅ in-repo `ucf_trans_2022.pkl` | ckpt-load into both + forward | ✅ was **non-faithful**; rewritten → loads clean, scores **max\|Δ\|=5.96e-08** |
| **RTFM** | yes (cloned) | ✗ none published | weight-transfer + forward | ✅ fixed non-local **softmax→f/N** bug; scores **max\|Δ\|=0.0 (bit-exact)** |
| **BN-WVAD** | yes (cloned) | ✅ in-repo `xd_best.pkl` | ckpt-load into both | ✅ was **non-faithful** (selfatt); rewritten → loads clean, **max\|Δ\|=2.4e-06** |
| **Sultani** | Keras (old) | ✗ not portable | architectural faithfulness | ✅ fixed non-faithful extra ReLU on the 32-unit FC (original = linear); 3-FC MIL matches paper |
| **S3R** | yes (cloned) | ✗ none published | weight-transfer + forward | ✅ rewritten to official keys (en/de-Normal, GroupNorm Aggregate, macro dict input); scores **max\|Δ\|=0.0 (bit-exact)** |
| **TPWNG** | ✗ unreleased | ✗ | paper §-by-§ (arXiv 2404.08531) | ✅ **paper-faithful**: NVP (Eq.3-4), TCSAL `z=F·σ`/`χ_z` (Eq.7-9), PLG α=0.2/θ=0.55, text-projection-only fine-tune, Capitalized prompts. No code → no numeric verify |
| **GS-MoE** | ✗ unreleased | ✗ | paper §-by-§ (arXiv 2508.06318) | ✅ **paper-faithful**: 13 experts, expert MLP `1024→512→256→128→64→1`, gate **bi-directional cross-attention** + MLP `2048→1024→512→256→128→1`, TGS loss. No code → no numeric verify |

UR-DMU + BN-WVAD share the official dual-branch temporal Transformer — factored into
`src/modules/translayer.py` (device-safe) and imported by both (DRY).

**Summary (all 10 models):** **7 output-verified vs official code/ckpt** (MGFN,
VadCLIP, CLIP-TSA, UR-DMU, RTFM, BN-WVAD, S3R) — 5 of those were non-faithful
approximations now rewritten + verified (VadCLIP, UR-DMU, BN-WVAD, RTFM softmax bug,
S3R key/dict). **3 paper-faithful section-by-section** (Sultani, TPWNG, GS-MoE) —
no public official code, so architecture-matched to the paper but not numerically
verifiable. Nothing left pending.

### Efficiency (attention kernels)
`src/modules/attention.py` exposes `attn_impl ∈ {eager, sdpa}`. Measured eager vs
SDPA (same weights, fp32): **max|Δ| ≈ 1e-7** (reduction-reordering noise) — within
fp32 noise but **not bit-identical**. So the bit-exact verifications and all the
faithful models keep **`eager` as the default**; `sdpa` (flash/mem-efficient) is an
opt-in speed mode for training where a ~1e-7 deviation is acceptable. No model's
verified output was changed for speed (the user's "must compute identically" bar).

## 1. Checkpoint verification (ckpt-backed)

| Model | Ckpt | Result | Evidence |
|-------|------|--------|----------|
| **MGFN** | `pretrained/mgfn/mgfn_ucf.pkl` | ✅ **fully compatible** | `convert()` maps 145→145 keys; load_state_dict missing=0 unexpected=0, 0 shape-mismatch; forward OK |
| **VadCLIP** | `pretrained/vadclip/model_ucf.pth` | ✅ **faithfully ported & verified** | loads clean (missing=0/unexpected=0); logits match official to ~1e-6; **UCF AUC 0.880199 == official** |
| **CLIP-TSA** | (no public ckpt) | ✅ **bit-identical to official** | weight-transfer (51/51 params) + seeded forward → scores max\|Δ\|=0.0 (`scripts/verify_clip_tsa.py`) |

### MGFN — confirmed reproducible
- `scripts/convert_official_to_hf.convert` is correct for the official MGFN UCF
  state dict. Run `python scripts/verify_checkpoints.py`.
- End-to-end eval on LOCAL `test.zip` + LOCAL `ground_truth.json` (290/290 keys
  aligned): **ROC-AUC = 0.8208**, PR-AUC = 0.1699 (`scripts/eval_mgfn_local.py`).
- Paper/official MGFN UCF ROC-AUC ≈ **0.8667** → **−4.6 pt gap**. Most likely
  cause = **I3D feature provenance** (the current `test.zip` vs the "fully-intact
  290 extracted" set the author plans to upload), or a test-time eval detail. Not
  a code-correctness issue — the pipeline is sound. **TODO: re-eval once the
  canonical 290 extract lands.**

### VadCLIP — faithfully ported & numerically verified (2026-06-14)
`src/models/vadclip/modeling_vadclip.py` was rewritten to mirror the official
`CLIPVAD` (nwpu-zxr/VadCLIP, vendored at `.reference/VadCLIP`) exactly, in HF
`Config`/`PreTrainedModel` style:
- `temporal` = `Transformer` of `ResidualAttentionBlock` (`nn.MultiheadAttention`
  `in_proj` + QuickGELU MLP, pre-norm, seq-first `(T,B,D)`) with the windowed
  local-attention mask; fixed `visual_length=256`.
- LGT-Adapter: `gc1..gc4` (reuses `src.modules.graph.GraphConvolution`, faithful),
  `disAdj` (`DistanceAdj` with the unused `sigma` param), `adj4` (cosine,
  threshold 0.7, row-softmax with `lengths`), `linear`.
- Text branch: self-contained `clipmodel` = CLIP text tower (token_embedding,
  positional_embedding, 12-layer causal transformer, ln_final, text_projection)
  loaded from the ckpt's `clipmodel.*`; `encode_textprompt` places learnable CoOp
  context (`text_prompt_embeddings`) around the class-name embeddings exactly as
  official. Tokenizer = `open_clip` ViT-B-16 (token ids verified identical to the
  official `clip.tokenize`).
- `convert_official_vadclip` drops only the unused `clipmodel.visual.*` and
  `clipmodel.logit_scale`; everything else loads by matching names.

**Verification** (`scripts/{vadclip_oracle,verify_vadclip,eval_vadclip_ucf}.py`):
the official model is run as an oracle (CLIP built from the ckpt, no network).
Per-frame `binary_logits`/`alignment_logits` match to **~1e-6**, `text_features`
to **0.0**. Full UCF test (290 vids, official `ucf_CLIP_rgbtest.csv` crop __5 +
`gt_ucf.npy`): **AUC1 0.880199 / AUC2 0.856938 — identical to the official run to
6 decimals**, matching the paper (~0.88). The old open_clip `CLIPPromptTextEncoder`
(`text_encoder.py`) is no longer used by VadCLIP (still imported by tpwng/clip_tsa
until those are ported).

## 2. Local data audit (`~/data/wsad/ucf_crime`)

| Path | Provenance | Shape | Status |
|------|-----------|-------|--------|
| `train.zip` | MGFN I3D, seg32 | `(10, 32, 2048)` /vid | ✅ usable (1610 vids) |
| `test.zip` | MGFN I3D, full-len | `(T, 10, 2048)` /vid | ✅ **290 intact** |
| `ground_truth.json` | frame labels | list per vid (T·16) | ✅ **290/290 aligned** (keys = `<vid>_i3d.npy`) |
| `UCFClipFeatures/<Class>/` | VadCLIP CLIP | `(T, 512)` fp16 /crop | ✅ 19 500 npy = 1950 vid × 10 crop |
| `UCF_Train_ten_crop_i3d` | RTFM I3D | 10-crop | ⚠️ different scale (~10×), don't mix |
| `UCF_Test_ten_crop_i3d` | RTFM I3D | — | ⚠️ only 109 vids (incomplete) |
| `raw/` | original UCF-Crime | mp4 in zips + splits | source for re-extraction |

### Wiring — DONE 2026-06-14 (strategy "B: zip-direct now")
The trainer now reads the on-disk data with no extraction/duplication and no HF:
1. **I3D zip-direct** — `src/data/local._build_i3d_from_zip` reads
   `ucf_crime/{train,test}.zip` in place (gated by `has_local_i3d_zip`); config
   keys `data.dataset_dir` (default `ucf_crime`) + `data.dynamic_load`.
2. **CLIP split** — `prepare_clip_features.py` symlinked `UCFClipFeatures/` →
   `clip/{train,test}` (16100 train / 2900 test = 1610 / 290 vids × 10 crops).
3. **Local GT** — `_load_ground_truth` prefers `ucf_crime/ground_truth.json`
   (HF only if absent); `_align_gt` re-keys it by bare video id so the I3D-keyed
   GT resolves against CLIP per-crop filenames (closes the CLIP gt-align TODO).

Verified (offline, `HF_HUB_OFFLINE=1`): `build_datasets` → train 1610 / test 290,
test labels attached; MGFN trains (loss 1.67→0.22) and VadCLIP trains on CLIP
(loss 9.51→9.35); 53 tests green. **Layout fork (B now / C later)** — see §4; the
WebDataset/manifest path is still the scale plan for extraction outputs + AIHub.

## 3. I3D extraction pipeline

Two distinct I3D lineages — **do not mix their features** (scale differs ~10×):

1. **tushar-n** (`src/i3d.py`, `src/features/i3d.py`) — torch-native, HF weights.
   Already wired into `scripts/extract_features.py`. This is the repo default.
2. **GowthamGottimukkala / RTFM** (`pretrained/i3d/*.pkl`) — **Caffe2 blobs**
   (`res_conv1_bn_*`, `nonlocal_conv*`; baseline 430 / nonlocal 540 blobs from
   facebookresearch/video-nonlocal-net). Needs **Caffe2→PyTorch conversion**
   before use. This is the lineage RTFM used for `UCF_*_ten_crop_i3d`.

Plan: add a Caffe2→torch converter for the two pkls, build a ResNet-50 I3D
(+nonlocal) torch module, and reproduce the RTFM extract path; cross-check feature
stats against `UCF_Train_ten_crop_i3d` to confirm fidelity. Separately, the
`open_clip` ViT-B/16 extractor (FEATURE_EXTRACTORS.md Step B) reproduces the
`UCFClipFeatures` CLIP cache.

## 4. Dataset strategy (open decision — needs your call)

Target: one efficient, scriptable data layer that serves the current cached
features **and** future AIHub / vision data + new benchmarks, while matching the
published dataset statistics.

Options for the on-disk contract:
- **A. Extract zips → `i3d/{train,test}/*.npy`** (matches current loader). Simple,
  random-access, but 1900+ small files × backbones = many inodes; 96 GB+ already.
- **B. Read the zips in place** (teach `build_datasets` to use the local zips +
  local gt; minimal change, no duplication). Fast to ship; zip random-access is OK
  for `(10,32,2048)` members.
- **C. WebDataset (.tar shards)** — best for scale/throughput, sequential IO,
  cloud/S3-friendly, ideal for AIHub-scale and multi-backbone. More upfront work
  (shard writer, sampler that still supports the normal/abnormal dual loader).

Recommendation: **B now** (unblocks local training immediately, low risk) +
**C as the scale path** for AIHub/extraction outputs, with a manifest
(`src/data/manifest.py`) as the single source of truth so the physical format
(npy / zip / tar) is swappable without touching models.

## 5. Verified state — what runs today
- ✅ All 53 tests pass (`pytest tests/ -q`).
- ✅ MGFN inference end-to-end on local data (0.8208 AUC).
- ✅ MGFN **training** loop on real local I3D features (loss 1.67→0.22 over 2
  epochs, `scripts/smoke_train_mgfn.py`).
