# WSVAD research roadmap (performance track)

Where we are and where the gains are, for UCF-Crime weakly-supervised video anomaly
detection. Grounded in the 2024–25 landscape + this repo's assets.

## Landscape (UCF-Crime frame-level AUC)

| Era | Approach | AUC | Note |
|-----|----------|-----|------|
| 2018 | Sultani MIL (C3D/I3D) | ~75–77 | the original baseline |
| 2021 | RTFM (I3D) | 84.3 | feature-magnitude MIL |
| 2022–23 | MGFN / S3R / UR-DMU (I3D) | 85–86 | I3D-feature ceiling |
| 2024 | DAKD, fusion (I3D+S3D+**CLIP**) | ~87 | **CLIP backbone + fusion is the lever** |
| 2023+ | VadCLIP / text-aligned (CLIP) | 86–88 | VLM branch, prompts |
| 2025 | MLLM / zero-shot reasoning | — | frontier; heavier, explainable |

**Takeaways:** (1) the I3D-only ceiling is ~86; (2) **CLIP features + multi-backbone
fusion** is the proven, cheap step up; (3) stronger visual backbones (VideoMAEv2,
InternVideo2) raise the visual ceiling; (4) the research frontier is VLM/MLLM.

## Assets in this repo (verified)

- **I3D (canonical)**: MGFN `features/i3d/*.zip` (L2~2.5). MGFN head local ROC-AUC
  ~0.82 (paper 0.867). 7 heads output-verified, 3 paper-faithful.
- **I3D (faithful extractor)**: `src/features/i3d_gowtham.py` bit-exact to Gowtham,
  baseline+nonlocal converted weights — re-extract any data (AIHub) self-consistently.
- **CLIP**: VadCLIP `features/clip/` (OpenAI ViT-B/16), VadCLIP head trains.
- **VideoMAE**: `src/features/videomae.py` (v1 ViT-B, (T,768)) — implemented, validated.

## Experimental ladder (ROI-ordered)

1. **Backbone ablation on existing heads (I3D vs CLIP)** — *no new extraction*.
   Train/eval the same head (Sultani/RTFM) on `data.backbone=i3d` vs `clip`, compare
   test AUC. Establishes the in-repo ablation harness + a baseline table. ← START HERE
2. **VideoMAE-B features for UCF + ablation** — needs a batch extraction entry from
   the raw zips (~3.7h on RTX2070S) then retrain a visual head. Measures v1 ViT-B vs I3D.
3. **VideoMAEv2 loader** — OpenGVLab checkpoints are NOT transformers-native; needs a
   custom loader (or a converted HF export). Higher visual ceiling than v1.
4. **Multi-backbone fusion (I3D + CLIP)** — concat/late-fuse features into one head;
   the landscape's best single step. Reuses assets we already have.
5. **VLM/MLLM frontier** — prompt/reasoning heads on CLIP-aligned features; longer-term.

## Tracked gaps / TODO

- [ ] Batch VideoMAE extraction entry from raw `Anomaly-Videos-*.zip` (mirror
      `extract_i3d_gowtham.py`; needs a zip→video read or an unzip step).
- [ ] VideoMAEv2 (OpenGVLab) loader — transformers `VideoMAEModel` won't load it as-is.
- [ ] Ablation harness: one entry that trains+evals a head on a chosen `data.backbone`
      and prints test ROC-AUC, for an apples-to-apples backbone table.

## Reference

See `docs/REPRO_STATUS.md` (faithful-port status, I3D lineages), `docs/DATA_LOCAL.md`
(layout, manifest). Landscape sources: RTFM (arXiv 2101.10030), DAKD (2406.02831),
VAD survey (2405.10347), "Evolution of VAD: DNN→MLLM" (2507.21649).
