# Feature Extractor Plan (slot 1)

How the offline backbones differ and how to extend `scripts/extract_features.py`
from I3D-only to a `--backbone {i3d,clip,videomae,vggish}` dispatcher. Extraction
is **inference-only and offline**: run once, cache `.npy`, train heads on the
cache. (See `WSAD_INTEGRATION_PLAN.md` §7.)

## Why this matters: text-alignment gates the method set
The single most important property is **`text_aligned`**. CLIP features live in
the same space as the CLIP text encoder, so only CLIP (and InternVideo2) unlock
the VLM/text-branch methods (CLIP-TSA, VadCLIP, TPWNG, WSVAD-CLIP). I3D/VideoMAE
are visually strong but have **no matching text space** — on them you can only run
visual-only heads (Sultani, RTFM, MGFN, UR-DMU, GS-MoE).

## Backbone differences

| Backbone | Status | dim | unit | snippet_len | text_aligned | preprocess | dep |
|----------|--------|-----|------|-------------|--------------|------------|-----|
| **i3d** | ✅ implemented | 2048 | snippet | 16 | ✗ | 10-crop video transform (`gtransforms`) | pytorchvideo |
| **clip** | ⬜ planned (next) | 512 | **frame** | 1 | **✅** | `open_clip` image transform (224, CLIP norm) | open_clip |
| **videomae** | ⬜ planned | 768 | snippet | 16 | ✗ | tubelet, ImageNet norm, 16×224×224 | transformers |
| **vggish** | ⬜ planned (AV phase) | 128 | **audio** | — | ✗ | PCM → log-mel spectrogram | torchaudio |

The three axes that actually change the code:
1. **unit** — snippet (I3D/VideoMAE: 16 frames → 1 vector) vs frame (CLIP:
   per-frame, aggregate to snippet by mean) vs audio window (VGGish).
2. **preprocess** — different transforms entirely (video 10-crop vs CLIP image vs
   VideoMAE tubelet vs mel-spectrogram). This is why each lives in its own file.
3. **dim** — 2048 / 512 / 768 / 128. Downstream stays backbone-agnostic because
   the loader appends the magnitude channel generically and configs carry
   `feature_size`; only model-specific code (e.g. MGFN's hardcoded 2048 split)
   needs the right dim.

What **stays the same** across all visual backbones:
- 10-crop convention (optional; start single-crop for CLIP).
- 32-segment vs full-T segmentation (`segment_features` is reusable as-is).
- Cache as `*_<backbone>.npy`; the loader/manifest reference a backbone *name*,
  never a path.

## Current code state
- `src/features/base.py` — `FeatureExtractor` ABC (metadata: `dim`, `unit`,
  `snippet_len`, `text_aligned`, `modality`) + `extract()`.
- `src/features/i3d.py` — ✅ adapter wrapping the existing `src.i3d` +
  `TenCropVideoFrameDataset`. Registered `"i3d"`.
- `src/features/{clip,videomae,vggish}.py` — registered stubs with full metadata
  and an implementation sketch in the docstring; `extract()` raises
  `NotImplementedError` until built.
- `src/features/build_extractor(name, **kw)` — name → instance.
- All register into `src.registry.FEATURE_EXTRACTORS` (lazy heavy-dep imports, so
  importing the package never fails when open_clip/torchaudio are absent).

## Plan to finish (when GPU is free)

### Step A — refactor the script to dispatch on backbone
`scripts/extract_features.py` currently hardcodes I3D. Change `main()` to:
```python
from src.features import build_extractor
extractor = build_extractor(args.backbone, device=args.device)
feats = extractor.extract(sample["video_path"])   # [T, ncrops, dim]
np.save(f"{name}_{args.backbone}.npy", feats)
```
Keep `segment_features(...)` unchanged (works for any dim). Add `--backbone`
(default `i3d`) and an output suffix from the backbone name. The legacy I3D path
stays byte-identical because the I3D adapter reuses the same model + dataset.

### Step B — implement CLIP (highest ROI: unlocks P3 CLIP-TSA, P4 VadCLIP)
- `open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")`.
- Decode frames (decord), apply CLIP `preprocess`, `model.encode_image` in fp16
  batches → `[n_frames, 512]`.
- Aggregate frames → snippet by mean to match the 32-seg convention; also keep a
  frame-level option for newer methods.
- **Start single-crop** to validate P3/P4; add 10-crop only if it moves AUC
  (10-crop CLIP over 1900 videos is non-trivial disk).
- Cache to a new HF dataset (mirror `jinmang2/ucf_crime_tencrop_i3d_seg32`).

### Step C — implement VideoMAE-B (feature ablation vs I3D)
- `transformers.VideoMAEModel` + `VideoMAEImageProcessor`, 16×224×224 clips,
  fp16. Mean of patch tokens (or pooled) per clip → `[T, 768]`.
- Re-run Sultani/RTFM/MGFN on VideoMAE feats for the `{I3D, VideoMAE} × methods`
  ablation (plan Phase P7).

### Step D — VGGish (only when starting XD-Violence / AV, plan P9)
- ffmpeg/torchaudio → log-mel → VGGish → `[T_audio, 128]`; align to visual T at
  the fusion head.

## Headline deliverable (plan §8)
One matrix table `{I3D, CLIP, VideoMAE} × {all methods}` — makes the
feature-vs-method contribution explicit. The backbone metadata + name-based cache
is what lets a single config swap the backbone without touching the heads.
