# WSVAD Reproduction — verified per-method recipes & findings

Source of truth for training/eval reproduction. Recipes verified against vendored
official code in `.reference/<repo>` (sciomc multi-agent audit, 2026-06-17). Feed
`scripts/run_matrix.py` (`HEADS` dict) + `src/trainer.py` (`WSVADTrainer.fit_steps`).

## Key findings (corrected understanding)

1. **"step-based" is NOT a property of MIL.** It is the iteration-counting convention
   in the official repos: `--lr '[0.001]*15000'` is an `eval()`-d list of 15000 LRs →
   15000 gradient steps. MIL's only essential is a balanced normal+abnormal pairing per
   update. 15000 steps ≈ ~600 reshuffled epochs on UCF (~810 abn + 800 nor, batch 32).
2. **No Gaussian smoothing anywhere.** RTFM/S3R/UR-DMU/BN-WVAD/CLIP-TSA/VadCLIP all do
   only `np.repeat(score, 16)` (snippet→frame) then sklearn AUC. Do NOT add smoothing.
3. **Best-checkpoint on test AUC is the standard** (every repo evals during training and
   keeps the best). Not leakage — it's the published protocol. (BN-WVAD selects on AP.)
4. **The reproduction gap is FEATURE PROVENANCE, not recipe.** On consistent Tushar-N
   I3D features, RTFM tops ~0.80–0.82 regardless of weight_decay (0.0005 vs official
   0.005 made ~no difference). The paper's 0.843 is on the authors' exact I3D (`mix_5c`)
   extraction. → faithful reproduction requires controlling feature extraction
   end-to-end (re-extract train+test with one verified extractor). [confirmed empirically]
5. **weight_decay paper-vs-code mismatches** (use the CODE value — it produced the
   reported number): RTFM/S3R/CLIP-TSA code = 0.005 (paper text often says 0.0005);
   UR-DMU/BN-WVAD = 5e-5. Our configs originally inherited a flat 0.0005 (wrong).

## Per-method recipe (UCF-Crime, I3D unless noted)

| method | steps/epochs | batch (nor+abn) | num_seg train | lr | wd | schedule | best-ckpt | reported | source |
|---|---|---|---|---|---|---|---|---|---|
| RTFM | 15000 steps | 32+32 | 32 | 1e-3 const | **0.005** | none | AUC (eval from step 200) | 0.8430 | .reference/RTFM main.py:41 |
| S3R | 15000 steps | 32+32 | 32 | 1e-3 const | **0.005** | none, **eval warmup 5000** | AUC | 0.8599 | .reference/S3R trainval:135 |
| MGFN | ~5000 steps | 8+8 (×10crop) | 32 | 1e-3 | 5e-4 | none | AUC | 0.8667 (ours ckpt 0.8208) | paper (no .ref) |
| UR-DMU | 3000 steps | 64+64 | **200** (test 32) | 1e-4 const | **5e-5** | none | AUC | 0.8697 | .reference/UR-DMU ucf_main:55 |
| BN-WVAD | 1000 steps | 64+64 | **200** | 1e-4 const | **5e-5** | **grad_clip 1.0** | **AP (not AUC)** | 0.8275 | .reference/BN-WVAD train.py:16 |
| Sultani (MIL) | ~30 ep | 30+30 | 32 | 1e-3 (Adam; paper Adagrad 0.01) | 5e-4 | none | AUC | 0.75 (C3D)/~0.83 (I3D) | paper (no .ref) |
| CLIP-TSA | 4000 steps | 16+16 | 32 | 1e-3 const | **0.005** | none | AUC (eval from 150) | ~0.87 | .reference/CLIP-TSA main:122 |
| VadCLIP | 10 epochs | 64+64 | 256 (window) | 2e-5 | 0.01 (AdamW) | MultiStepLR [4,8] γ0.1 | **ROC1 = sigmoid(logits1)** | 0.8801 | .reference/VadCLIP ucf_train:47 |
| GS-MoE | ~30–50 ep [est] | 64+64 | 200(paper)/32(ours) | 1e-4 [est] | 5e-4 | none | AUC | 0.916 | paper (no .ref) |

Notes: UR-DMU/BN-WVAD num_seg=200 (our i3d features are 32-seg → mismatch; also i3d-OOM
on 8 GB GPU). VadCLIP eval uses `sigmoid(logits1)`=ROC1 (0.880); `1-softmax(logits2)[:,0]`
=ROC2 is ~0.857 (don't use). S3R uses an offline OMP dictionary (ours is learnable → caps
AUC). MGFN/Sultani/GS-MoE have no vendored reference (recipe from paper/estimate).

## Eval protocol (all methods)
Per-snippet score → (10-crop mean for I3D test) → `np.repeat(score, 16)` to frame level →
concatenate all test frames → `sklearn.roc_auc_score(gt, preds)`. No post-processing.
Our impl: `src/eval_matrix.py` (per-head shaping; vadclip windowed; visual heads 4-D crop layout).

## Hardware constraints (this box: RTX 2070 8 GB GPU + 15 GB WSL RAM)
- i3d features: lazy zip load + `num_workers=0` (zip handle not shareable across workers →
  BadZipFile). Eager-loading 1610 npy = ~12.8 GB RAM → WSL crash. ONE i3d job at a time.
- NEVER CPU-eval-fallback for big heads (clip_tsa full-length test attention >20 GB → WSL OOM kill).
- i3d UR-DMU/BN-WVAD/CLIP-TSA eval OOMs on 8 GB (full-length test attention) → CLIP versions OK.
