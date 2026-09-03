# Next-session plan (kickoff)

Trigger: say **"핸드오프대로 시작"** (or "NEXT_SESSION 진행"). I auto-load this via the
`next-session-plan` memory; this file is the human-readable copy.

## Where we are (reproduction baseline, all committed)
| model | from-scratch | paper | port/eval verified |
|---|---|---|---|
| VadCLIP | 0.8654 | 0.8801 | ✅ |
| RTFM | 0.834 | 0.843 | ✅ |
| MGFN | 0.833 | 0.8667 | ✅ |
| S3R | 0.829 | 0.8599 | ✅ (learnable-dict cap) |
| UR-DMU | 0.797 | 0.8697 | ✅ official ckpt = 0.8697 EXACT on i3d_1024 |

Infra done: lazy i3d load (RAM-OOM fix), per-crop eval (8GB GPU-OOM fix), DeepMIL
**1024-d** features (`i3d_1024_seg200`, UR-DMU/BN-WVAD) + `--feature-dim`, dual-report
(best + mean±std), cosine LR decay, `--eval-start/--eval-every`. 2048-d = `i3d_mgfn_seg200`.

## Kick off 3 tracks at once — RESOURCE-AWARE (the "터질 수도 있으니 알아서 계획")
**Only ONE GPU-heavy run at a time** (8GB; 3 parallel GPU = OOM). CPU/doc track runs
in parallel. Use nohup + lazy load + stream logs (3 past session-crashes were 25GB
eager-load RAM-OOM — now fixed; keep it that way). Serialize the GPU queue:

1. **[GPU, serial] Finish reproducing remaining heads** on the right features:
   BN-WVAD (i3d_1024_seg200 + per-crop, --feature-dim 1024), Sultani / GS-MoE (2048
   i3d_mgfn_seg32), CLIP-TSA / TPWNG (CLIP). dual-report each.
2. **[GPU, serial] UR-DMU/BN-WVAD from-scratch training tuning** — close ~0.80→0.87.
   The gap is the TRAINING loop (ckpt = 0.8697 exact proves data/eval/port faithful):
   attack overfitting/early-peak — eval-every-10 (official) to catch the early peak,
   early-stop, lr/wd, official 3000-step recipe. (NOT features — already correct.)
3. **[CPU/doc, parallel] Consolidation + README results table**: collapse
   experiments/repro_vadclip.py + repro_i3d.py + build_* into ONE parametrized
   reproduction harness (head/variant/recipe/dim-driven) — the user's anti-bloat
   ask; delete superseded probes; write the paper-comparison results table into README.

## Then Spec 2 (causality / real-time / efficiency) — INCLUDED, not deferred
Motivated by the UR-DMU O(T²) OOM: efficient temporal attention (sliding-window /
linear / streaming) so memory-heavy methods (UR-DMU/BN-WVAD/CLIP-TSA) run on commodity
8GB + real-time. See HANDOFF.md Spec-2 notes (StreamingLLM attention-sink, KV-cache,
chunked prefill, causal-VAD, FPS/AUC/latency benchmark). Spec 1-2 (new extractors like
VideoMAE) stays DEFERRED.

## Don't re-derive (already settled this session)
- UR-DMU/BN-WVAD = 1024-d I3D (Carreira/pytorch-i3d, DeepMIL) ≠ 2048-d ResNet50-I3D
  (RTFM/MGFN/S3R). Confirmed from ckpt (Conv1d 1024→512).
- random_perturb = np.linspace (no augmentation; no-op).
- The UR-DMU gap is training, not feature/eval/port (official ckpt = 0.8697 exact).
