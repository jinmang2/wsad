# WSVAD comparison framework — multi-backbone matrix, HF-style serving, streaming inference

**Status:** approved design (Spec 1 detailed; Specs 2–3 scoped)
**Date:** 2026-06-16
**Branch:** `feat/eval-matrix-serving`

## Goal (decisive)

Turn this repo from a *paper-reproduction* codebase into a **WSVAD comparison +
serving framework**:

1. Implement every WSVAD method as an HF-style, easy-to-serve model.
2. Run them on **real video input** end-to-end (no separate offline extraction
   step at serving time — the extractor is absorbed like an HF *processor*).
3. Produce **paper-style test figures** (where an anomaly was detected vs ground
   truth) and **real-time inference**.

AIHub CCTV is *one input source*, not the goal. The goal is the framework.

## Why this is a separate effort

Distinct role from `feat/data-pipeline` (which builds the extractors + data
layout this work consumes). Managed as its own branch + epic + per-component
issues + PR.

---

## Spec roadmap (dependency order)

| Spec | Scope | Deliverable |
|---|---|---|
| **Spec 1 — Foundation** (this doc, detailed) | backbone-agnostic extraction · dim-agnostic heads · matrix harness · HF-style pipeline (offline mode) | `{backbone × head}` AUC table + `raw video → scores + basic figure` |
| **Spec 2 — Streaming/Causal** | inference adapter (window/causal) · causal-finetune recipe · **LLM-inference-technique research track** · real-time benchmark | real-time inference + causal checkpoints |
| **Spec 3 — Qualitative suite** | full paper-style figures · localization metrics · cross-matrix comparison dashboard | qualitative-eval module |

Architectural invariant across all specs: **original bidirectional training +
offline eval is never modified** (source of truth for the matrix + accurate
figures). Streaming/causal is an *additive inference layer*, never a rewrite of
model definitions.

---

## Spec 1 — Foundation (detailed)

### Architecture

Three decoupled layers + a bundling layer on top:

```
[backbone registry] ─┐
                     ├─►  pipeline(backbone, head)  ──► raw video → per-frame scores → figure
[head registry]    ──┘         (HF-style bundle factory)
        ▲
[matrix harness] ── crosses both registries → {backbone × head} train/eval table
```

Internals stay **fully decoupled** (matrix needs swappability); the surface is a
**bundle** (HF-style serving). Both requirements satisfied by one factory.

### Components

**① Backbone-agnostic extraction**
- `scripts/extract_features.py` → `--backbone {i3d,clip,videomae,videomaev2,internvideo2}`
  dispatcher (roadmap step A). Reuse `build_extractor()`; cache `*_<backbone>.npy`;
  keep `segment_features` unchanged (dim-agnostic). Legacy I3D path stays byte-identical.
- **New loaders:** `internvideo2.py`, `videomaev2.py`. Both ship as OpenGVLab
  native weights (not transformers-native) → custom loader (prefer an HF export
  if one validates). `dim` read from model config so downstream adapts.
- Reuse existing dataset-keyed layout + unified manifest from `feat/data-pipeline`.

**② Dim-agnostic heads + compatibility guard**
- **Fix MGFN hardcoded 2048** (`src/models/mgfn/modeling_mgfn.py:83`,
  `x[:, :2048] / x[:, 2048:]`) → derive the magnitude split from
  `config.feature_size`. Other heads already read `feature_size` (verified by grep:
  Sultani, GS-MoE, BN-WVAD, VadCLIP).
- **text-align guard:** each head declares `requires_text_aligned: bool`. The
  pipeline factory checks it against `extractor.text_aligned` and rejects
  incompatible pairs (e.g. VadCLIP + VideoMAE) with a clear error. CLIP and
  InternVideo2 are text-aligned → pass both visual-only and VLM heads.

**③ Matrix harness**
- One entry (`scripts/run_matrix.py` or `run.py` extension): `(backbone, head,
  dataset)` → train + eval → log test ROC-AUC + artifacts.
- Auto-selects only compatible pairs (derived from ② guard) → emits the
  `{backbone × head}` table. This is the body of "compare all methods".

**④ HF-style pipeline (processor absorbed)**
- `AnomalyDetectionPipeline(backbone, head_ckpt)`: holds extractor (= HF
  processor role) + trained head + post-process. Input: raw video → output:
  per-frame anomaly scores (+ figure). No separate extraction step at serve time.
- Spec 1 implements **`mode="offline"`** only (bidirectional, exact = current).
  `window`/`causal` extend the *same* interface in Spec 2.

**⑤ Minimal figure (pipeline liveness)**
- One per-frame score curve + GT anomaly-region shading. Full suite is Spec 3;
  this is just visual proof the pipeline runs end-to-end.

### Data flow

`raw video → extractor(processor) → [T, dim] features → head(mode=offline) →
[T] per-frame scores → postprocess → score curve + figure`.
Training flow unchanged: `cached *_<backbone>.npy → head → ROC-AUC`.

### Error handling
- Incompatible `{backbone, head}` (text-align): explicit `ValueError` at factory
  time, never a silent shape error.
- Dim mismatch: `feature_size` is the single source; heads slice the appended
  magnitude channel relative to it; assert on load.
- Missing weights / loader: `strict=True` load after key remap (existing
  convention in `WEIGHTS_AND_OPTIMIZATION.md`).

### Testing
- Dim-agnostic forward smoke test per head with dummy features (2048/512/768).
- **MGFN numerical-equivalence regression** after the 2048 fix
  (`verify_numerical_equivalence`, eager/fp32/TF32-off) so reproduction stays
  bit-exact.
- Compatibility guard unit test (incompatible pair raises).
- Pipeline test: short sample video → assert score length == frame/snippet count.

### Repo changes
- **Modify:** `scripts/extract_features.py`, `src/models/mgfn/*` (2048),
  per-head `requires_text_aligned`.
- **New:** `src/features/internvideo2.py`, `src/features/videomaev2.py`,
  `src/pipeline.py` (bundle factory), `scripts/run_matrix.py`, minimal figure util.
- **Unchanged:** all original training entries + model definitions (additive only).

---

## Spec 2 preview — streaming/causal (research-driven)

Non-causality has two sources: **(A) training-time loss construction** (top-k,
32-seg norm, MIL max — exists only at train time, *not* an inference problem) and
**(B) inference-time temporal context** (RTFM MTN / MGFN glance-focus / UR-DMU /
GS-MoE global attention — each snippet score depends on future). The score head
is per-snippet; only context is bidirectional → conversion is *not* fatal.

Mitigation spectrum (zero-retrain → clean):
1. **Sliding window** (no retrain, all heads) — window W, emit last/center, slide.
2. **Causal mask + cheap finetune on cached features** — removes discrepancy;
   nearly free because features are cached.
3. **Stateful streaming** (KV-cache-like state + dilated-conv ring buffer) — true
   online, constant per-step cost; needs (2).

**Research track:** map LLM inference-optimization techniques (StreamingLLM
attention sinks, sliding-window attention, KV-cache, chunked prefill,
quantization) onto causal WSVAD — an under-explored space; documenting +
validating these is a genuine contribution. Train/inference discrepancy is real
and measured, not assumed.

---

## Sources
See `docs/FEATURE_EXTRACTOR_RESEARCH.md` for backbone/data/optimization research
and citations (RTFM, VideoMAEv2, InternVideo2, LAION, Real-Time WSVAD WACV 2024).
