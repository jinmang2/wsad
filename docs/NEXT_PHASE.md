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

---

## Measured 2026-09-02 (gate + cost, both open questions resolved)

**1. The q/v-bias fix (`14d4e04`) really did improve the features.** Forensic screen,
identical 40-video test subset, `scripts/diag/feature_forensics.py`:

| feature set | MAGNITUDE | CONTENT | PROBE |
|---|---|---|---|
| `videomae_GATEfix` (bias restored) | 0.5537 | **0.7209** | **0.6767** |
| `videomae_GATE` (bias silently zeroed) | 0.4991 | 0.7072 | 0.6152 |
| `i3d_1024_seg200` (same 40) | 0.4995 | 0.5993 | 0.5536 |
| `i3d_mgfn` (same 40) | 0.4937 | 0.5916 | 0.5392 |

VideoMAE ranks clearly above both I3D variants on every proxy, and the bias fix moved
the linear probe +0.061 (0.615 → 0.677) and pulled magnitude off dead chance
(0.499 → 0.554). Screen only *ranks* — the real gate is still a short head train — but
nothing here argues against extracting the full set. The stale `videomae_GATE`
features should be deleted.

**2. Full VideoMAE-base extraction costs ~2.5–3 h, not ~30 h.** Measured on the
RTX 2070 SUPER 8 GB with a warm cache (`uv run scripts/extract_modern.py`, marginal
rate from a two-point fit that removes ~20–40 s of process/model startup):

| split | protocol | marginal rate | projected |
|---|---|---|---|
| train (1610) | `--sample-to 32` | ~3.7 s/video | ~1.6 h |
| test (290) | full length | ~10.4 s/video | ~0.9 h |

The old "88 videos in 102 min" (~70 s/video) that made this look like a 30-hour job is
**~19x slower than the extractor measures in isolation**. The cause was not established —
the plausible candidates are GPU contention with a concurrent head train and the WSL host
dying partway through the window (the run left no per-video timestamps, so the wall time
may cover a long stall rather than 88 slow videos). Either way the extractor itself is
fine: with the GPU otherwise idle this is a single evening, so
`scripts/run_phase1_videomae.sh` is worth resuming as-is (it is idempotent/resumable).

## Cosmos-Embed1 — added as a backbone, cost measured (2026-09-02)

`src/features/cosmos.py` registers NVIDIA's Cosmos-Embed1 (`--backbone cosmos`). It is the
first **text-aligned** video backbone here besides CLIP, so it can feed the VLM heads, and
its base checkpoints contain no UCF-Crime (Kinetics + robotics + AV only). The
`-448p-anomaly-detection` variant IS finetuned on Vad-Reasoning (UCF-Crime, XD-Violence,
ShanghaiTech, UBnormal) and the extractor **refuses to load it** — same leakage trap as
the OPear VideoMAE variant flagged above.

Two output modes, because the head↔feature mapping actually differs:
`output="proj"` is the 768-d text-aligned projection, **L2-normalized by the model**
(measured L2 = 1.0000 ± 0.0003) — the magnitude channel is therefore constant and useless
to RTFM/MGFN. `output="cls"` is the Q-Former pooled vector before projection (L2 = 11.61 ±
0.02), magnitude-preserving, for those heads.

### Throughput (RTX 2070 SUPER 8 GB, fp16, measured end-to-end and per stage)

| backbone | snippets/s | notes |
|---|---|---|
| VideoMAE-base (ViT-B, 1 forward per 16-frame tubelet) | 38.9 | batch 16, 775 MB |
| Cosmos-Embed1-224p (ViT-g, 8 frame-forwards per snippet) | 5.6 | batch-independent, 2.6–3.4 GB |
| Cosmos-Embed1-448p | ~4x slower again (1025 vs 257 tokens) | 5.0 GB at batch 4 |

Stage breakdown at batch 8 (224p): model 1283 ms, processor 92 ms, PIL→numpy 12 ms — it is
**GPU-compute bound**, so there is no cheap preprocessing win to take. The cost is
structural: a 1B-param ViT-g run once per *frame* (8 per snippet) against VideoMAE-base's
86M ViT-B run once per *clip*.

Projected full UCF-Crime extraction: **~6 h at 224p** (train `--sample-to 32` ≈ 2.6 h +
full-length test ≈ 3.5 h) and **~24 h at 448p**. So 224p is an overnight job and 448p is
not practical on this card without a smaller variant or a second GPU.

### Where this leaves Track A
The ranking question is now cheap to answer (a 40-video 224p gate is ~30 min) but the
*commitment* question is expensive. Sequence accordingly: gate at 224p, and only spend the
448p budget if the 224p features clear i3d and VideoMAE on the screen.
