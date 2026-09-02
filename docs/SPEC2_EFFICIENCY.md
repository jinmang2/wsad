# Spec 2 — efficient / causal / streaming WSVAD (design)

Parallel design track (the user's "feature 먼저, 효율 병렬"). Goal: make the memory-heavy
heads (UR-DMU / BN-WVAD / CLIP-TSA) run on 8 GB **and in real time / causally**, with a
measured AUC-vs-efficiency trade-off. Builds on the toolkit already shipped this effort:
`WSAD_ATTN=mem` (SDPA + einsum, `src/modules/translayer.py:83-89`), gradient checkpointing
(`Transformer.gradient_checkpointing_enable`), fp16 AMP, `src/modules/amp.safe_bce`.

## Motivation
- The dual-branch temporal Transformer is **O(T²)** (`translayer.py:91` `q·kᵀ`; the decay
  `attn2` is `(n,n)`). At seg200 / full-length test it OOMs 8 GB (the whole reason `mem`
  and per-crop eval exist). For live surveillance T is effectively unbounded.
- Current scoring is **bidirectional + full-sequence** (`src/eval_matrix.py` scores the
  whole clip at once) → no online/causal operation, no bounded latency.

## Key structural lever (why this head is unusually amenable)
`translayer.Attention` is **dual-branch**:
- **branch-2** = a FIXED distance-decay `exp(-|i-j|/e)`, `e=e¹≈2.718` → it is already
  ~0 beyond `|i-j|≳15`. So restricting branch-2 to a local **window** is *near-lossless*
  (the tail it drops is numerically negligible). Only **branch-1** (learned `qkᵀ`) is
  approximated by windowing. This makes a sliding-window variant far cheaper in accuracy
  than for a generic full-attention Transformer.

## Design — opt-in attention variants (extend the `WSAD_ATTN` flag)
Add to `translayer.py` alongside `eager`/`mem` (same env-flag pattern, default unchanged
so verified ckpts stay bit-exact):
1. **`WSAD_ATTN=window` (sliding-window, band W≈64)** — O(T·W).
   - branch-1: `F.scaled_dot_product_attention(q,k,v, attn_mask=band)` (band/local mask).
   - branch-2: compute the decay only within the band (near-exact, see above).
   - Recommended default for long sequences; smallest accuracy hit.
2. **`WSAD_ATTN=linear` (linear attention, O(T))** — feature-map `φ(q)(φ(k)ᵀv)` for
   branch-1; branch-2 stays the (cheap, local) decay. Most aggressive; validate AUC drop.
3. **`WSAD_ATTN=sink` (StreamingLLM attention-sink)** — keep the first `s` "sink" tokens +
   a recent window; enables *unbounded* streaming without re-encoding history.

## Design — causal & streaming inference
- **Causal mask**: score each frame from past-only (triangular mask on branch-1; branch-2
  decay restricted to `j≤i`). Needed because surveillance is online. Expect a small AUC
  drop vs bidirectional; quantify it.
- **`StreamingScorer`** (new, `src/inference/streaming.py`): process a video in chunks of
  `C` frames, keep a KV-cache of the last `W` keys/values (+ `s` sink tokens), emit
  per-frame anomaly scores online. Peak memory = `O(W)`, latency = `C` frames. Wraps any
  head whose backbone is the translayer (UR-DMU/BN-WVAD); CLIP-TSA's TSA gets the same
  windowing.

## Benchmark (`scripts/diag/bench_efficiency.py`)
For each `(head ∈ {ur_dmu, bn_wvad, clip_tsa}) × (attn ∈ {eager, mem, window, linear, sink})
× (mode ∈ {bidir, causal-stream})`, measure on the UCF-Crime test set:
- **AUC** (frame-level ROC) — the quality axis.
- **FPS** (frames/sec), **per-chunk latency** (ms), **peak GPU mem** (MB), max T before OOM.
Emit a markdown table: the efficiency frontier (AUC vs FPS/mem). Headline target: UR-DMU/
BN-WVAD running full-length on 8 GB at real-time FPS with ≤1 pt AUC drop vs `eager`.

## Implementation order (small, verifiable steps)
1. `WSAD_ATTN=window` in `translayer.Attention.forward` (band SDPA + banded decay). Verify
   max|Δ| vs `eager` is tiny (branch-2 near-exact) and AUC drop ≤ ~0.5 pt on UR-DMU eval.
2. `bench_efficiency.py` (AUC + FPS + peak-mem table) — quantify `window`/`linear` vs
   `eager`/`mem`.
3. `causal` mask variant + AUC delta.
4. `StreamingScorer` + the streaming FPS/latency numbers.

## Validation
- Numerics: `window` vs `eager` max|Δ| (expect branch-2 ≈0, branch-1 = the approximation).
- Quality: AUC drop per variant on the existing UR-DMU/BN-WVAD test eval (reuse
  `src.eval_matrix.evaluate`).
- Efficiency: the FPS / peak-mem / max-T table; confirm full-length on 8 GB without per-crop
  splitting (which `mem`/per-crop currently work around). No change to the `eager` default,
  so all verified checkpoints and reproduced numbers are untouched.

---

## Step 1 + 2 RESULTS (implemented and measured 2026-09-02)

`WSAD_ATTN=window` is in `src/modules/translayer.py` (band SDPA over query blocks +
the decay as a depthwise 1-D convolution), with `WSAD_ATTN_WINDOW` (half-width, default
64) and `WSAD_ATTN_CHUNK` (query block, default 256). `scripts/diag/bench_efficiency.py`
produces both tables below; `tests/test_translayer_window.py` pins the numerics.

### Quality — UR-DMU official checkpoint, UCF-Crime test (290 videos, 1.11 M frames)

| attn | AUC | ΔAUC | AP | FPS | latency ms (mean/p95) | peak mem MB |
|---|---|---|---|---|---|---|
| `eager` | 0.8697 | +0.0000 | 0.3562 | 64244 | 60 / 84 | 2880 |
| `mem` | 0.8697 | -0.0000 | 0.3562 | 85457 | 45 / 63 | 912 |
| `window:64` | 0.8476 | **-0.0221** | 0.3353 | 39864 | 96 / 213 | 474 |
| `window:256` | 0.8611 | -0.0086 | 0.3539 | 51602 | 74 / 203 | 482 |
| `window:∞` (band ≥ T) | **0.8697** | **±0.0000** | 0.3562 | 35648 | 107 / 213 | **470** |

**The ≤0.5 pt target at W=64 was NOT met — the real cost is 2.2 pt.** The design's
structural claim held exactly as argued: branch-2 windowing is lossless (the `window:∞`
row reproduces the official 0.8697 / 0.3562 *bit-for-bit* on real data, and the unit test
bounds the dropped decay tail at <1e-6). The loss is entirely branch-1 — **UR-DMU's
learned attention genuinely uses long-range context**, and AUC recovers monotonically
with the band (0.8400 @32 → 0.8476 @64 → 0.8611 @256 → 0.8697 @full). That is a result
about the model, not a bug in the kernel.

Two things worth keeping from this table even though the headline target failed:
- **`window:∞` is strictly better than `mem` for eval**: identical AUC, 1.9x less memory
  (470 vs 912 MB) — because the dominant allocation in `eager` was never branch-1, it was
  the `(b, h, n, n)` decay repeat, and the convolution removes it entirely.
- At test-set lengths windowing is a **throughput loss** (40 k vs 64 k FPS): the masked
  SDPA path gives up the flash kernel, and O(T·W) does not pay for itself yet at T ≈ 200–4000.

### Cost vs sequence length (synthetic, 2-layer translayer, dim 512, RTX 2070 8 GB)

| T | `eager` ms / MB | `mem` ms / MB | `window:64` ms / MB |
|---|---|---|---|
| 512 | 2 / 44 | 2 / 33 | 9 / 31 |
| 2048 | 8 / 275 | 7 / 91 | 5 / 55 |
| 8192 | 94 / 3719 | 57 / 872 | 14 / 151 |
| 16384 | 8744 / 14584 | 218 / 3256 | 29 / 279 |
| 32768 | **OOM** | 9550 / 12632 | **58 / 535** |

This is where the window path earns its place. The crossover is **T ≈ 2000**; past it the
scaling is exactly linear (T 16384 → 32768 doubles window's cost 29 → 58 ms and 279 →
535 MB) while `eager` OOMs at 32 k and `mem` degrades to 9.5 s. At 32 k frames window is
**165x faster than `mem` and 24x lighter**.

### Verdict and revised sequencing
- Windowing is **not** a free accuracy-preserving speedup for the existing 290-video
  benchmark; do not switch the reported eval path to it. Keep `eager`/`mem` there.
- It *is* the enabler for the unbounded-T streaming case Spec 2 actually targets, which
  is the regime where `eager` cannot run at all.
- So the AUC-vs-W curve above becomes the honest headline for Spec 2: **the streaming
  budget buys accuracy back monotonically**, and the design question is where a live
  deployment wants to sit on it.
- Next (unchanged order): causal mask + its AUC delta, then `StreamingScorer` — whose
  value is now quantified in advance by the T ≥ 8192 rows.

## Step 3 + 4 — causal attention and `StreamingScorer` (implemented 2026-09-02)

`WSAD_ATTN=causal` (band + past-only, both branches) and `src/streaming.py` are in. The
AUC deltas still need the GPU and are pending; what is already settled is the *structure*,
and two things fell out of it that change the Spec 2 story.

### The official branch-2 is not streamable in principle
The bidirectional decay divides by `attn2.sum(-1)`, and that sum is a **whole-clip**
quantity: it depends on the sequence length and on how far `j` sits from either end. An
online scorer cannot compute it — the stream has not finished — and scoring the same frame
inside two differently-sized windows would give two different answers.

The causal backend therefore normalizes by the **interior limit** `(1+r)/(1-r) ≈ 5.4977`,
which the true normalizer approaches exponentially (`r^17 ≈ 2e-3`). This buys exact
position-independence, and its cost is a small deviation from the offline model confined to
the first and last ~20 snippets of a clip. This substitution is the reason streaming can be
exact at all, so it is a design decision, not an implementation detail.

### The eval path is exactly streamable, and the latency is one snippet
Walking UR-DMU's eval forward, every component is either pointwise in time (`Memory_Unit`
reads a learnable bank, never other frames; `encoder_mu`; `cls_head`) or has a finite
receptive field (`Temporal` = `Conv1d(k=3, pad=1)`; the translayer stack = `depth · W`).
So `StreamingScorer` reproduces offline scoring **bit-for-bit** from a bounded window
rather than approximating it — `receptive_field()` derives that window from the module tree
so a config change cannot silently make the two diverge.

The honest caveat: causal attention is past-only, but the `Conv1d` keeps `padding=1`, so a
score still depends on **one** future snippet. Streaming latency is one snippet (16 frames),
not zero. A truly zero-latency variant needs a causal convolution, which changes what the
pretrained weights mean — so this is reported rather than hidden.

Tests (`tests/test_translayer_window.py`, `tests/test_streaming.py`, 18 of them) pin: the
causal decay against a masked dense reference; that changing frame *k* leaves every output
before *k* bit-identical (with a bidirectional control that must fail that same check);
full-width causal against a dense causal reference; streaming == offline; chunk-size
invariance; and that the model never sees more than `left + chunk + right` frames.

### Still to measure (needs the GPU, currently held by the Cosmos gate)
`causal:W` rows in `bench_efficiency.py` — the AUC price of removing lookahead at matched
band width, and the streaming FPS / per-chunk latency at T ≥ 8192.

## FlexAttention — tried, measured, NOT adopted as the default (2026-09-02)

The windowed path loses throughput because a masked SDPA gives up the flash kernel, so the
obvious fix was `torch.nn.attention.flex_attention` (in torch 2.11): a block-sparse kernel
built for exactly this mask. It is implemented and available as `WSAD_ATTN_KERNEL=flex`,
with a fallback to the chunked path if it is unavailable or refuses a shape. It is **not**
the default, and the measurements are why.

### Isolated branch-1 — flex looks like a large win

| T | chunked SDPA | flex (compiled) | speedup |
|---|---|---|---|
| 512 | 0.29 ms | 0.19 ms | 1.51x |
| 8192 | 3.83 ms | 1.10 ms | 3.48x |
| 32768 | 17.80 ms | 3.02 ms | **5.89x** |
| 65536 | 28.20 ms | 7.55 ms | 3.73x |

Numerically identical (max|Δ| ≈ 2e-7) and marginally lighter. Two traps found on the way:
`flex_attention` **must** be wrapped in `torch.compile` — called eagerly it warns and
materializes the full (T, T) score matrix, discarding the entire point — and
`create_block_mask` needs `_compile=True`, or building the mask itself OOMs 8 GB at
T = 32768.

### Full translayer — the win evaporates

| T | kernel | cold (incl. compile) | steady state |
|---|---|---|---|
| 8192 | sdpa | 1374 ms | **18.43 ms** |
| 8192 | flex | **67852 ms** | 18.71 ms |
| 32768 | sdpa | 74 ms | **62.10 ms** |
| 32768 | flex | 1723 ms | 72.03 ms |

Steady state is a wash at 8 k and **16% worse** at 32 k, and the cold cost is
catastrophic — **68 seconds** of compilation at T = 8192. Amortizing it over many chunks
does not help, because there is nothing left to amortize into: the gain is gone by then.

The reason the isolated number does not survive is that branch-1 is a **minority of layer
cost**. The decay convolution, the FFN and the LayerNorms dominate, so a 5.9x on one part
of ~25% of the runtime cannot move the total, and flex's own overhead more than eats it.
That also re-frames the earlier throughput finding: the windowed path's regression is not
mainly a flash-kernel problem, so a better attention kernel was never going to fix it.

**Kept as an opt-in flag, not deleted**, because the isolated result may well hold on a
card with different kernel selection — but on this hardware the honest answer to "does
FlexAttention fix the windowed path?" is **no**, and the default stays `sdpa`.
