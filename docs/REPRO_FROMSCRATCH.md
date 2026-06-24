# From-scratch reproduction — results, levers, and the 8 GB ceiling

Companion to `REPRO_STATUS.md` (which tracks **port fidelity** — every model loads its
official checkpoint and matches outputs). This doc tracks **from-scratch training** on
the clean cached features, on a single **RTX 2070 SUPER (8 GB, Turing sm_75)**.

Regenerate the table any time: `PYTHONPATH=. python experiments/results_table.py`.

## Results (UCF-Crime frame-level ROC-AUC, 290-video test)

| head | feature | ours (from-scratch) | paper | port verified |
|------|---------|--------------------:|------:|:--:|
| sultani | i3d_mgfn_seg32 | 0.8071 | 0.7541 (C3D) | arch |
| rtfm | i3d_mgfn_seg32 | 0.8341 | 0.8430 | ✅ bit-exact |
| mgfn | i3d_mgfn_seg32 | 0.8332 | 0.8667 | ✅ ckpt |
| s3r | i3d_mgfn_seg32 | _queue_ | 0.8599 | ✅ bit-exact |
| ur_dmu | i3d_1024_seg200 | 0.8145 | 0.8697 | ✅ ckpt 0.8697 exact |
| bn_wvad | i3d_1024_seg200 | 0.8233 | 0.8724 | ✅ ckpt |
| gs_moe | i3d_mgfn_seg32 | _queue_ | 0.9160 | paper-only |
| clip_tsa | clip | _queue_ | 0.8758 | ✅ bit-exact |
| vadclip | clip | 0.8654 | 0.8801 | ✅ AUC 0.8802 exact |
| tpwng | clip | _queue_ | 0.8779 | paper-only |

`_queue_` = serial GPU run in progress (`scripts/gpu_queue_pc.sh` + chained CLIP heads).
sultani > paper because the paper number is C3D-era; modern I3D features lift the
old MIL baseline.

## Feature-provenance facts (don't re-derive — see the VERDICT below)
- Two distinct 2048-d I3D extractions exist: `features/i3d` = **DeepMIL/Roc-Ng** (L2~2.5)
  and `i3d_mgfn` = **MGFN authors'** HKU OneDrive (L2~22, byte-verified). They are NOT
  interchangeable: DeepMIL has **inverted** anomaly-magnitude (breaks RTFM/MGFN/S3R →
  chance), so the magnitude heads must use **`i3d_mgfn`** (`i3d_mgfn_seg32` for seg32).
- The old `outputs/matrix/results.json` (feature=`None`, scoring ~0.85) trained on
  now-DELETED `features/i3d/{train,test}/` npy dirs that mixed train=`_archive` (L2~21,
  magnitude-correct) with test=DeepMIL (L2~2.6). It scored high because the TRAIN half had
  the correct magnitude direction (content features transfer) — an artifact, not a better
  set. Those numbers are not used here.
- UR-DMU/BN-WVAD use **1024-d** DeepMIL/Carreira I3D (`i3d_1024_seg200`, L2~8).

## The memory-head ceiling (UR-DMU / BN-WVAD): ~0.82 from scratch
The port is faithful (official ckpt = 0.8697 exact). From-scratch nonetheless caps
~0.82 after exhausting every training lever:

| lever | ur_dmu | bn_wvad | note |
|-------|-------:|--------:|------|
| batch 4 (old `HEAD_BATCH` cap) | 0.807 | 0.800 | batch 4 wrecks BN-WVAD's BatchNorm stats |
| batch 32 + toolkit + cosine | 0.8145 | **0.8233** | 8 GB batch ceiling |
| per-crop batch 64 (official layout) | 0.8114 | 0.8003 | faithful, but per-crop ↓ BN samples ⇒ hurts bn_wvad |

- **batch** (4→32) is the biggest single lever — BN-WVAD is literally BatchNorm-based.
  batch 64 (official) OOMs at seg200 even with the full toolkit (model body, not just
  attention). 8 GB caps batch at 32.
- **per-crop training** (official lists each of the 10 crops as a separate sample,
  16100 rows; we crop-average to 1610) was the prime suspect but **did not help** —
  ur_dmu flat, bn_wvad worse (its BatchNorm wants more flattened-crop samples, which the
  stacked `(10,200,1024)` layout gives). Built as `i3d_1024_seg200_pc` for the record.
- Remaining ~0.05 gap is **training dynamics / seed** (runs oscillate 0.77–0.82; the
  field reports best-of-N), not a structural bug. A seed sweep (official seed 2022) is
  the only untried lever and would likely add ≤0.02 — still short of 0.87.

## Training-scale toolkit (8 GB), all opt-in — `src/`
Verified the eager/fp32 path is unchanged (74 tests green; bit-exact/Δ checks):
- **gradient checkpointing** — `Transformer.gradient_checkpointing_enable()`, fwd/grad Δ=0.
- **mem-efficient attention** — `WSAD_ATTN=mem`: branch-1 SDPA + branch-2 einsum (drops
  the `(b,h,T,T)` decay/`dots` materialization), Δ=7e-8 vs eager.
- **fp16/bf16 AMP** — `--mixed-precision`; `src/modules/amp.safe_bce` makes the BCE-on-
  probability losses autocast-safe (no-op in fp32) across all 5 BCE heads.
- **gradient accumulation** — `--grad-accum` (note: does not enlarge BatchNorm's batch).

CLI: `scripts/run_matrix.py --batch N --grad-checkpoint --mixed-precision fp16 --grad-accum K`
(+ `WSAD_ATTN=mem`). To reproduce the paper memory-head numbers, run the official
`--batch 64` on a ≥24 GB GPU (no toolkit needed).

## VERDICT — the gap is a feature-extraction ceiling (ultragoal 2026-06-24)

A thorough investigation (`.omc/ultragoal/`) settled why every head sits ~0.02-0.05 below
paper. It is the **I3D feature-extraction provenance ceiling**, not training/recipe/eval:

1. **Eval sound** — UR-DMU ckpt → 0.8697 and VadCLIP ckpt → 0.8802 reproduce paper EXACTLY.
2. **Recipe ruled out** — RTFM (matching recipe) reproduces (−0.009). MGFN at the FULL
   official recipe (15000 steps, batch 16, cosine) = 0.8197 and peaked at **step 200** then
   overfit — 3× more training does not help.
3. **Feature ceiling proven** — the authors' OWN MGFN ckpt on the byte-verified MGFN features
   (`i3d_mgfn`) scores only **0.8346** (paper 0.8667); our from-scratch MGFN **0.8332 = 99.8%
   of that ckpt**. We reach what the features allow.
4. **Feature forensics** (`scripts/diag/feature_forensics.py`, rebuilt 2026-06-24) — the
   feature cache carries the anomaly signal only *weakly* and *atemporally*. Three relative
   proxies on the 290-vid test set (all crop-averaged, snippet-level):

   | set | MAGNITUDE-AUC | CONTENT-AUC | linear-PROBE-AUC |
   |-----|--------------:|------------:|-----------------:|
   | `i3d_mgfn` | 0.46 (near-chance) | 0.56 | 0.51 |
   | `i3d_1024_seg200` | 0.48 | 0.62 | 0.55 |
   | `videomae` (40-vid gate) | 0.50 | 0.71 | 0.62 |

   **Correction to the earlier verdict:** raw *magnitude*-AUC is NOT a feature-quality gate.
   `i3d_mgfn` magnitude-AUC is 0.46 (mildly inverted, near chance) yet RTFM/MGFN train to 0.83
   on it — magnitude neither predicts nor bounds trainable performance. The real distinction
   between `i3d_mgfn` and DeepMIL `features/i3d` for the magnitude heads is *scale/content*,
   not a clean "magnitude direction". Likewise every atemporal proxy here (incl. the supervised
   linear probe) under-predicts the temporal heads (probe 0.51 → head 0.83), because WSVAD is
   dominated by temporal modeling these per-snippet scores can't see.

   What the proxies DO show, consistently and apples-to-apples (same 40 videos): **modern
   features are more linearly separable** — `videomae` content/probe (0.71/0.62) clearly beats
   `i3d_mgfn` (0.56/0.51). So "better features" remains the right lever, but for the
   content-separability reason, and gated by a short head train, not by magnitude.

Bottom line: the reproductions are faithful and reach the ceiling the cached I3D features
permit. The forensic screen says modern backbones (VideoMAE) carry a stronger, more separable
anomaly signal — the next lever — but the only honest accept/reject gate is a short
temporal-head train on the new set, NOT a training-free oracle.

## Harness
- `experiments/reproduce.py` — single parametrized from-scratch harness (epoch recipes).
- `scripts/run_matrix.py` — {variant × head} matrix (step recipes, toolkit flags).
- `scripts/gpu_queue*.sh` — serial 8 GB queues (one heavy run at a time).
- `experiments/results_table.py` — regenerates the table above from result JSONs.
