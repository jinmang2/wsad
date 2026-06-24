# Next phase — feature extractor (primary, GPU) + Spec 2 efficiency (parallel, design)

Motivation (proven this session, see `REPRO_FROMSCRATCH.md` VERDICT): the accuracy gap to
paper is a **feature-extraction ceiling**, not training/recipe/eval. The 2017 I3D features
carry the anomaly signal only weakly and atemporally — `feature_forensics.py` (content/probe
screen): `i3d_mgfn` content-AUC 0.56 / linear-probe 0.51, while a VideoMAE gate sample reaches
0.71 / 0.62 on the *same* videos. So the highest-leverage move is **better features**.

⚠️ Gate caveat (corrected 2026-06-24): the forensic screen is a *relative* atemporal proxy, NOT
a ceiling — every per-snippet score (magnitude, content, probe) under-predicts the temporal
heads (i3d_mgfn probe 0.51 → head 0.83). Do NOT accept/reject a feature set on the oracle alone.
Use it to *rank* candidates cheaply; the real gate is a short head train (`run_matrix.py`).
The old "raw-magnitude AUC" headline was unsound (i3d_mgfn magnitude 0.46 yet head 0.83).

## Track A — modern feature extractor (PRIMARY, GPU-serial)

### Candidates (research 2026-06-24)
| extractor | why | access | note |
|---|---|---|---|
| **VideoMAE-v2 / VideoMAE-large** | strong self-sup video features, beats I3D | HF `MCG-NJU/videomae-large`; in Transformers | easiest pipeline |
| **OPear/videomae-large-finetuned-UCF-Crime** | already UCF-anomaly-adapted | HF | anomaly-tuned features = stronger magnitude signal? |
| **InternVideo2** | current SOTA video backbone | InternVideo repo | heaviest; check 8GB feasibility |
| **X-CLIP** | CLIP-aligned video → suits VLM heads | HF `microsoft/xclip-*` | pairs with vadclip/clip_tsa |
| pre-extracted UCF features | skips extraction compute | not found for these on UCF-Crime (only THUMOS/ANet via InternVideo) | likely must self-extract |

Recent SOTA WSVAD (RefineVAD'25, TRACES, GV-VAD, AnomalyCLIP) ~0.87+ — confirms modern
features + better heads are where gains are.

### Plan (each step gated)
1. **Pick + access** — start with VideoMAE-large (HF, lowest friction). Decide finetuned
   (OPear) vs raw based on the forensic gate.
2. **Forensic screen (NEW, our edge)** — extract a SMALL balanced sample, run `scripts/diag/
   feature_forensics.py --dir <sample>` on it. Does the set rank ABOVE `i3d_mgfn` on
   content/probe-AUC? If it can't beat I3D even on the cheap atemporal screen, skip it.
   If it ranks higher, proceed — but confirm with a short `run_matrix.py` head train before
   committing to the full 1900-video extraction (the screen ranks, it does not certify).
3. **Extract** — adapt `scripts/extract_i3d_gowtham.py` pattern to a VideoMAE extractor
   (10-crop or single-crop, snippet=16). Write `features/videomae_*/{train,test}` + PROVENANCE.
4. **Re-run matrix** (`run_matrix.py --variant videomae_*`) — measure lift vs i3d_mgfn.
5. **Head × feature mapping** — magnitude heads need magnitude-preserving features; VLM heads
   need CLIP-align; memory heads need temporal richness. Tabulate which extractor wins which
   head (novel contribution).

### Open questions to resolve first
- VideoMAE output dim/snippet protocol (vs I3D 2048-d/16-frame) → head `feature_size` config.
- 8 GB extraction feasibility (VideoMAE-large inference, batch 1) + time for 1900 videos.
- Does anomaly-finetuned (OPear) leak test labels? (UCF test videos must NOT be in its
  finetune set — verify before using, else it's leakage.)

## Track B — Spec 2 efficiency / causality / inference (PARALLEL, design+CPU)

Already have the toolkit (gradient checkpointing, `WSAD_ATTN=mem` SDPA attention, AMP). Spec 2
makes the memory-heavy heads (UR-DMU/BN-WVAD/CLIP-TSA) run on 8 GB + real-time:
- **Efficient temporal attention** — sliding-window / linear / StreamingLLM attention-sink to
  kill the O(T²) (already motivated by UR-DMU OOM); extend `src/modules/translayer.py` mem path.
- **Causal / streaming inference** — online scoring (no full-clip lookahead), KV-cache, chunked
  prefill → causal-VAD for live surveillance.
- **Benchmark** — FPS / latency / peak-mem / AUC trade-off table across heads.

Sequencing: GPU is 8 GB serial, so feature experiments take the GPU; Spec 2 runs as design +
CPU prototyping + the benchmark harness, then validates on whichever feature baseline wins.

## Immediate next step (proposed)
Resolve Track A "open questions" (VideoMAE protocol + OPear leakage check + 8 GB feasibility),
then run the **forensic gate** on a VideoMAE sample before committing to full extraction.
