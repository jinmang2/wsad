# UCF-Crime reproduction — ours vs paper (clean features)

Frame-level ROC-AUC on the 290-video test split. `*` = official checkpoint.

| head | feature | ours | paper | Δ | source |
|---|---|---|---|---|---|
| sultani | i3d_mgfn_seg32 | 0.8071 | 0.7541 | +0.0530 | `outputs/matrix_sultani_clean/results.json` |
| rtfm | i3d_mgfn_seg32 | 0.8341 | 0.8430 | -0.0089 | `outputs/matrix_rtfm_cosine/results.json` |
| mgfn | i3d_mgfn_seg32 | 0.8332 | 0.8667 | -0.0335 | `outputs/matrix_mgfn_freq/results.json` |
| s3r | i3d_mgfn_seg32 | _pending_ | 0.8599 | — | — |
| ur_dmu | i3d_1024_seg200 | 0.8145 | 0.8697 | -0.0552 | `outputs/matrix_urdmu_1024_b32cos/results.json` |
| bn_wvad | i3d_1024_seg200 | 0.8233 | 0.8724 | -0.0491 | `outputs/matrix_bnwvad_1024_b32full/results.json` |
| gs_moe | i3d_mgfn_seg32 | 0.7838 | 0.9160 | -0.1322 | `outputs/matrix_gs_moe_clean/results.json` |
| clip_tsa | clip | 0.8123 | 0.8758 | -0.0635 | `outputs/matrix_clip_tsa_clip/results.json` |
| vadclip | clip | 0.8654 | 0.8801 | -0.0147 | `experiments/runs/vadclip_scratch/seed234/result.json` |
| tpwng | clip | _pending_ | 0.8779 | — | — |
| pel4vad | i3d_1024_seg200 | _0.8281 (200-step smoke)_ | 0.8676 | — | `outputs/pel4vad_smoke.log` |

`pel4vad` is a **200-step smoke run**, not a reproduction — the official recipe is ~6290
steps (16100 crop-samples / batch 128 x 50 epochs). It is listed because the port is wired
and learning (loss 1.41 → 0.86, AUC 0.8256 → 0.8281 across the two evals, still rising),
which is what the smoke was for. Replace the row with a real number before comparing it to
anything, and note the crop-handling gap documented in `modeling_pel4vad.py`: the official
run trains on individual crops (10x samples + augmentation) where this port averages them.

### PEL4VAD divergence (2026-09-02) and its cause

The first full run diverged: AUC 0.8079 at step 300, then 0.5697 at both step 600 and 900,
with train loss climbing 1.01 → 3.46 → 5.42. Reproduced on real cached features and traced
to the **signed power normalization** in `TCA`. The official expression,
`sqrt(relu(x)) - sqrt(relu(-x))`, has derivative `0.5/sqrt(|x|)` — unbounded as an
activation approaches zero: measured 5e17 at x = 1e-36, and the run hit a total gradient
norm of **6.6e18**. Adam then absorbs that spike into its second moment and throttles the
affected parameters for thousands of steps, which is why the collapse was permanent rather
than a transient spike.

Fixed by writing it as `sign(x)·sqrt(|x| + 1e-6)` — the same function away from zero (values
differ by under 5e-7 for activations of order 1) with the derivative bounded at 500. Verified
on the same seed and step budget that previously exploded: gradient norm at step 850 goes
from 6.6e18 to 0.43, and the loss decreases monotonically 2.46 → 0.117 over 900 steps.

Worth noting for other ports: the official code carries the same unbounded expression. It is
plausible that averaging 10 crops (see the crop-handling note above) produces smoother
activations that land near zero more often than the official single-crop inputs do, which
would make this repo hit a landmine the original rarely steps on.
