"""Faithful VadCLIP (AAAI'24) from-scratch reproduction on UCF-Crime CLIP features.

Goal: confirm our verified VadCLIP *port* (bit-exact to official under weight load,
ROC1=0.8801) also reproduces the paper number when trained FROM SCRATCH with the
official recipe — distinct from the eval-only `--official` matrix cell.

Official recipe (.reference/VadCLIP/src/ucf_train.py + ucf_option.py, verified):
  AdamW(lr=2e-5)  [AdamW default weight_decay=0.01]
  MultiStepLR(milestones=[4, 8], gamma=0.1)
  max_epoch=10, batch_size=64 (per normal/abnormal loader -> 128 total), seed=234
  windowed visual_length=256, per-window feat_lengths, eval = sigmoid(logits1)=ROC1.

This drives the canonical `src.trainer.WSVADTrainer` (now with optimizer_name/
scheduler knobs) + the per-head windowed eval in `src.eval_matrix.evaluate`.
Reproducible: fixed seed, recipe in-code, results + per-epoch log written under
`experiments/runs/vadclip_scratch/`.

    PYTHONPATH=. WSAD_DATA=~/data/wsad conda run -n balaenoptera \
        python experiments/repro_vadclip.py

KNOWN RISK (open): our training loss path does not yet thread per-window
`feat_lengths` into CLAS2/CLASM masking + length-based top-k. If from-scratch
ROC1 lands well below 0.88, that masking is the prime suspect to fix next.
"""

import json
import os
import random
import time

import numpy as np
import torch

import src.models  # noqa: F401  (register heads)
from scripts.run_matrix import _build_model, _data_cfg, HEADS
from src.eval_matrix import evaluate
from src.models.vadclip.modeling_vadclip import load_pretrained_clip_text
from src.trainer import WSVADTrainer, build_datasets

SEED = 234
MAX_EPOCH = int(os.environ.get("REPRO_EPOCHS", 10))  # override for smoke (REPRO_EPOCHS=1)
BATCH = 64          # per loader (normal / abnormal); official batch_size
LR = 2e-5
WEIGHT_DECAY = 0.01  # AdamW default (official passes no weight_decay)
MILESTONES = [4, 8]
GAMMA = 0.1
OUT = "experiments/runs/vadclip_scratch"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    spec = HEADS["vadclip"]
    data_cfg = _data_cfg("clip", spec)
    train_sets, test_set = build_datasets(data_cfg)
    model, _, _ = _build_model("vadclip", "clip", device)
    # from-scratch == random head + FROZEN pretrained CLIP text tower (official freezes
    # CLIP). Without this the text-alignment branch is random -> ROC stuck at chance.
    n_clip = load_pretrained_clip_text(model)
    print(f"[init] loaded + froze {n_clip} pretrained CLIP text-tower tensors", flush=True)

    n_train = len(train_sets["normal"]) + len(train_sets["abnormal"])
    print(f"[data] train={n_train} (normal {len(train_sets['normal'])} + "
          f"abnormal {len(train_sets['abnormal'])})  test={len(test_set)}", flush=True)
    print(f"[recipe] AdamW lr={LR} wd={WEIGHT_DECAY} MultiStepLR{MILESTONES}x{GAMMA} "
          f"epochs={MAX_EPOCH} batch={BATCH}(+{BATCH}) seed={SEED}", flush=True)

    log = []

    def eval_fn(m):
        res = evaluate(m, test_set, backbone="clip", head="vadclip", device=device)
        log.append({"roc_auc": res["roc_auc"], "pr_auc": res["pr_auc"]})
        json.dump(log, open(f"{OUT}/epoch_log.json", "w"), indent=2)
        return res

    trainer = WSVADTrainer(
        model=model, learning_rate=LR, weight_decay=WEIGHT_DECAY, batch_size=BATCH,
        num_workers=int(data_cfg.num_workers), frames_per_clip=16, mixed_precision="no",
        optimizer_name="adamw", scheduler_milestones=MILESTONES, scheduler_gamma=GAMMA,
    )
    best = trainer.fit(train_sets, epochs=MAX_EPOCH, eval_fn=eval_fn,
                       select_metric="roc_auc")

    result = {
        "head": "vadclip", "backbone": "clip", "source": "from-scratch",
        "recipe": {"optimizer": "adamw", "lr": LR, "weight_decay": WEIGHT_DECAY,
                   "milestones": MILESTONES, "gamma": GAMMA, "epochs": MAX_EPOCH,
                   "batch_total": 2 * BATCH, "seed": SEED},
        "best_roc_auc": round(best["roc_auc"], 4), "best_pr_auc": round(best["pr_auc"], 4),
        "best_epoch": best["best_epoch"], "last_roc_auc": round(best["last"], 4),
        "paper_roc_auc": 0.8801, "official_ckpt_roc_auc": 0.8770,
        "n_test": len(test_set), "seconds": round(time.time() - t0, 1),
    }
    json.dump(result, open(f"{OUT}/result.json", "w"), indent=2)
    print("\n=== VadCLIP from-scratch result ===", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    print(f"\nwrote {OUT}/result.json", flush=True)


if __name__ == "__main__":
    main()
