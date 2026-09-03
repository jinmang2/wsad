# Session handoff — WSVAD comparison framework

Pick-up notes for continuing on another machine. Single source of truth = PR #22
(draft, base `dev`), roadmap = Epic #16; track work there rather than opening issues.

- **Branch:** `feat/eval-matrix-serving`
- **State as of 2026-09-02:** Spec 1 complete with real numbers; Spec 2 steps 1–4
  implemented and measured; Track A (better features) is gated on GPU hours, not code.

## Environment
- **uv only — conda is gone.** `uv sync --group dev` recreates everything from
  `pyproject.toml` + `uv.lock` (Python 3.11, torch 2.11.0+cu130 from the PyTorch index).
- Interpreter: **`uv run python`**. No `PYTHONPATH=.` — the project installs editable, so
  `src.*`, `scripts.*` and `experiments.*` import directly.
- Tests: `uv run pytest -q` → **106 passed**.
- Hardware reality that shapes every plan: one **RTX 2070 SUPER (8 GB)** on WSL, which
  reboots intermittently. Serialize GPU work; check `nvidia-smi` (utilisation and power,
  not just free VRAM) before starting anything. The network is sometimes a mobile hotspot,
  so a large download is its own budget even when the GPU is idle. **Long training runs and
  multi-GB downloads need the user's go-ahead each time.**

## Where the results stand
`docs/RESULTS_TABLE.md` — 11 heads reproduced against their papers on UCF-Crime. Best
reproduction 0.8654 (VadCLIP, paper 0.8801); UR-DMU's official checkpoint reproduces
**0.8697 exactly**, which is what proves the data, eval and ports are faithful. Still
`_pending_`: `s3r`, `tpwng`, and a real `pel4vad` run (only a 200-step smoke so far).

`docs/REPRO_FROMSCRATCH.md` VERDICT still holds: the remaining gap to paper is a
**feature-extraction ceiling**, not the training recipe.

## Spec 2 — efficient / causal / streaming (docs/SPEC2_EFFICIENCY.md)
Implemented: `WSAD_ATTN=window` (band SDPA + the decay as a depthwise convolution),
`WSAD_ATTN=causal` (past-only), `src/streaming.py` (`StreamingScorer`, bit-identical to
offline scoring in bounded memory), and `scripts/diag/bench_efficiency.py`.

Three findings worth not re-deriving:
1. **The official branch-2 is unstreamable in principle** — it divides by a whole-clip
   normalizer. The causal path substitutes the interior limit `(1+r)/(1-r)`; that is what
   makes streaming exact, and it is a design decision, not a detail.
2. **UR-DMU's learned attention genuinely uses long-range context.** Band W=64 costs
   2.2 pt AUC; the curve recovers monotonically to 0.8697 at full width.
3. **FlexAttention does not fix the windowed path's throughput** — 5.9x on branch-1 alone,
   but a wash-to-16%-worse at the layer level plus 68 s of compilation, because branch-1 is
   a minority of layer cost. Kept behind `WSAD_ATTN_KERNEL=flex`, default `sdpa`.

Streaming latency is **one snippet** (16 frames), not zero: attention is past-only but the
embedding `Conv1d` keeps `padding=1`. Reported, not hidden.

## Track A — better features (the actual accuracy lever)
- **VideoMAE-base is the best screened candidate** (content 0.7209 / probe 0.6767 on the
  40-video screen vs I3D's ~0.59 / ~0.54). Full extraction ≈ 3 h. `videomae_seg32` is
  partially extracted; `scripts/run_phase1_videomae.sh` is idempotent — just re-run it.
- **Cosmos-Embed1 is integrated but lost the screen** (224p: content 0.6620 / probe 0.6272)
  at ~2x VideoMAE's cost. Do not spend the budget on it. It stays as the only text-aligned
  video backbone besides CLIP.
- **LanguageBind is the next thing to screen and costs zero GPU** —
  `docs/FEATURE_SOURCES.md`, `scripts/fetch_pretrained_features.py --source languagebind`.
  26 GB download (ask first), 10-crop × T × 768, magnitude-preserving, text-aligned, no
  UCF-Crime in its training data.

## Two loader traps already paid for — expect more of them
Both cost a day and were silent (features came out wrong or NaN, nothing warned):
- `transformers` v5 does not convert MCG-NJU VideoMAE's timm `q_bias`/`v_bias`, so
  `query.bias`/`value.bias` load as **zero** (`src/features/videomae.py`).
- v5 builds on the meta device and materializes only checkpoint tensors, so
  `persistent=False` buffers stay uninitialized and non-checkpoint-shaped parameters get
  reinitialized (`src/features/cosmos.py` — construct with `from_config`, then
  `load_state_dict`).
**Always check a new backbone's load report for MISSING/UNEXPECTED keys and sanity-check
the feature magnitudes before extracting 1900 videos.**

## Next candidates (see docs/EXTENSION_CANDIDATES.md)
1. `pel4vad` full run (5000 steps) — the head is ported and smoke-trained. **Ask first.**
2. LanguageBind screen — download, `feature_forensics.py`, then a short head train.
   **Ask first.**
3. `STPrompt` (0.8808, also localizes → Spec 3) and `MTFL` (0.8978, ships Video Swin
   features, but no stated license).

## Gotchas
- `gh pr edit --body` fails on this repo (Projects-classic GraphQL bug). Use REST:
  `gh api repos/jinmang2/wsad/pulls/22 -X PATCH -F body=@file.md`.
- `pgrep -f "<pattern>"` matches this shell's own command line — filter it out or you will
  "find" a job that is not running, or kill your own shell.

## Quick commands
```bash
uv run pytest -q
uv run python scripts/run_matrix.py --heads pel4vad --variant i3d_1024_seg200 --feature-dim 1024
uv run python scripts/diag/bench_efficiency.py --attn eager,window:64,causal:64
uv run python scripts/diag/feature_forensics.py --dir <feature-dir> --name <label>
```
