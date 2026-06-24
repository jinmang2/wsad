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
