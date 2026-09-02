# Session handoff — WSVAD comparison framework

Pick-up notes for continuing on another machine. **Single source of truth = PR #22**;
roadmap = Epic #16. No per-spec PRs, no new issues (track in PR #22). Delete this
file once the work lands.

- **Branch:** `feat/eval-matrix-serving` (based on `feat/data-pipeline`, PR base `dev`)
- **PR:** #22 (draft) · **Epic:** #16 · sub-issues #17–21 were **closed/consolidated** into PR #22
- **Last commit:** `857af70` (6 commits ahead of the data-pipeline base)

## Environment (reproduce on the new PC)
- **uv only — conda is gone.** `uv sync --group dev` recreates everything from
  `pyproject.toml` + `uv.lock` (Python 3.11, torch 2.11.0+cu130 from the PyTorch index).
- Interpreter: **`uv run python`**. No `PYTHONPATH=.` — the project installs editable,
  so `src.*`, `scripts.*` and `experiments.*` import directly.
- `decord`, `open_clip_torch`, `timm`, `transformers` are all pinned in the lock file.
  Still needed for runtime work: GPU + weights for InternVideo2 / VideoMAEv2.
- Tests: `uv run pytest -q` → **74 passed** (verified on the uv env, 2026-09-02).

## What is DONE (Spec 1 scaffold — code only, runtime-UNVERIFIED)
- `src/compat.py` — backbone↔head text-align guard (`assert_compatible`); flags
  `requires_text_aligned=True` on VadCLIP + TPWNG only.
- `src/models/mgfn/modeling_mgfn.py` — removed hardcoded `2048`; split now from
  `config.channels` (dim-agnostic). 2048 path verified byte-identical.
- `src/pipeline.py` — `AnomalyDetectionPipeline(backbone, head)`: extractor(processor)
  + head + postprocess; `from_features()` (extraction-free) + `visualize()`; auto-sets
  head feature-dim to backbone dim. **`mode="offline"` only.**
- `src/viz.py` — `plot_anomaly_scores()`: blue curve + light-orange GT band + dashed
  red boundaries (matches paper Fig.10); `legend` toggle.
- `src/features/extract.py` + `scripts/extract_features.py --backbone` — registry
  dispatcher caching `<stem>_<backbone>.npy`; `import decord` made lazy; legacy I3D
  path intact when `--backbone` unset.
- Tests: `tests/test_compat.py`, `tests/test_pipeline.py`, `tests/test_viz.py`,
  `tests/test_extract.py`.

## What is NOT done (next, in PR #22 checklist)
**Spec 1 finish (GPU):** real clip/videomae extraction on ≥1 video → raw-video
end-to-end through the pipeline (needs decord + a sample) → `{backbone×head}` matrix
with **real AUC** → InternVideo2/VideoMAEv2 loaders (OpenGVLab weights aren't
transformers-native; need a custom loader).
**Spec 2 (the real weight, NOT STARTED):** inference adapter with offline/sliding-
window/causal modes (same pipeline API); causal-finetune recipe on cached features;
LLM-inference-technique research track (StreamingLLM attention-sink, KV-cache, chunked
prefill → causal-VAD); real-time benchmark (FPS/AUC/latency).
**Spec 3 (NOT STARTED):** full qualitative suite + localization metrics + cross-matrix
dashboard.

## Design + research (read first)
- `docs/superpowers/specs/2026-06-16-wsvad-matrix-serving-design.md` (architecture,
  3-spec roadmap, invariant: original training/model defs unchanged; causal is additive)
- `docs/FEATURE_EXTRACTOR_RESEARCH.md` (backbones, data, optimization, Real-Time WSVAD)

## Open issues found in self-review (address during Spec 1 finish / Spec 2)
1. **[Med]** Pipeline defaults (`with_magnitude=True`, `segment_to=None`) suit visual
   magnitude heads; text heads (tpwng/clip_tsa) trained with 32-seg & no magnitude →
   serving/matrix config should carry **per-head preprocessing**. (Visual path OK.)
2. **[Med]** `pipeline._set_feature_dim` sets `visual_width` to backbone dim → VadCLIP
   on a non-512 text-aligned backbone (InternVideo2) would mismatch its CLIP-512 text
   tower. Fine for CLIP-512 today; handle when InternVideo2 lands.
3. **[Low]** `viz._coerce_gt` mask-vs-intervals heuristic can misread tiny inputs;
   prefer explicit `[(start,end)]` intervals.
4. **[Low]** `--backbone` path doesn't segment (legacy path does); add `--seg` if a
   head needs seg32 caches.

## Gotchas
- `gh pr edit --body` fails on this repo (Projects-classic GraphQL bug). Edit the PR
  body via REST: `gh api repos/jinmang2/wsad/pulls/22 -X PATCH -F body=@file.md`.
- ~~`scripts/extract_features.py` needs `PYTHONPATH=.`~~ — fixed by the editable uv install;
  `load_dataset(..., config_name=...)` kwarg kept as-was (verify against current `datasets`).

## Quick commands
```bash
# tests
uv run pytest -q
# backbone extraction (needs decord + GPU for clip/videomae)
uv run python scripts/extract_features.py --backbone clip
# figure smoke (see /tmp/gen_fig.py pattern): plot_anomaly_scores(scores, gt=[(s,e)], legend=False, save_path=...)
```
