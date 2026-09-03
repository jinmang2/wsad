"""From-scratch reproduction for seg32 I3D heads (MGFN/RTFM/S3R/Sultani).

Uses the PROVEN epoch-based recipe (scripts/diag/train_archive_probe.py reached
MGFN 0.8667 / RTFM 0.8448): Adam lr 5e-5, batch 16, ~12-15 epochs, best-checkpoint
on test AUC — NOT run_matrix's unverified lr 1e-3 step-based recipe, which diverges
to chance (0.52). Trains on the clean MGFN-lineage features (norm ~22):

    features/i3d_mgfn_seg32/  (train seg32 (10,32,2048) + test full-length (T,10,2048),
    built from the byte-verified MGFN-authors' distribution by build_mgfn_seg32.py).

    WSAD_DATA=~/data/wsad uv run \
        python experiments/repro_i3d.py            # HEAD=mgfn default
    HEAD=rtfm LR=5e-5 EPOCHS=15 python experiments/repro_i3d.py
"""

import json
import os
import random
import time

import numpy as np
import torch
from omegaconf import OmegaConf

import src.models  # noqa: F401  (register heads)
from scripts.run_matrix import _build_model
from src.eval_matrix import evaluate
from src.trainer import WSVADTrainer, build_datasets

HEAD = os.environ.get("HEAD", "mgfn")
VARIANT = os.environ.get("VARIANT", "i3d_mgfn_seg32")
LR = float(os.environ.get("LR", 5e-5))
EPOCHS = int(os.environ.get("EPOCHS", 15))
BATCH = int(os.environ.get("BATCH", 16))
SEED = int(os.environ.get("SEED", 0))
OUT = f"experiments/runs/{HEAD}_{VARIANT}/seed{SEED}"

PAPER = {"mgfn": 0.8667, "rtfm": 0.8430, "s3r": 0.8599, "sultani": 0.83}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    os.makedirs(OUT, exist_ok=True)
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    data_cfg = OmegaConf.create({
        "root": os.environ.get("WSAD_DATA", "~/data/wsad"), "source": "local",
        "backbone": "i3d", "dataset_dir": "ucf_crime", "feature_variant": VARIANT,
        "ground_truth": None, "dynamic_load": False, "batch_size": 1,
        "frames_per_clip": 16, "num_workers": 4, "clip_length": None, "segment": None,
        "single_crop": True, "with_magnitude": True,
    })
    train_sets, test_set = build_datasets(data_cfg)
    model, _, wd = _build_model(HEAD, "i3d", device)
    n_train = len(train_sets["normal"]) + len(train_sets["abnormal"])
    print(f"[data] variant={VARIANT} train={n_train} test={len(test_set)}", flush=True)
    print(f"[recipe] {HEAD} Adam lr={LR} wd={wd} batch={BATCH}(+{BATCH}) epochs={EPOCHS} seed={SEED}", flush=True)

    log = []

    def eval_fn(m):
        res = evaluate(m, test_set, backbone="i3d", head=HEAD, device=device)
        log.append({"roc_auc": res["roc_auc"], "pr_auc": res["pr_auc"]})
        json.dump(log, open(f"{OUT}/epoch_log.json", "w"), indent=2)
        return res

    trainer = WSVADTrainer(model=model, learning_rate=LR, weight_decay=wd, batch_size=BATCH,
                           num_workers=4, frames_per_clip=16, mixed_precision="no")
    best = trainer.fit(train_sets, epochs=EPOCHS, eval_fn=eval_fn, select_metric="roc_auc")

    result = {
        "head": HEAD, "variant": VARIANT, "source": "from-scratch",
        "recipe": {"optimizer": "adam", "lr": LR, "wd": wd, "batch_total": 2 * BATCH,
                   "epochs": EPOCHS, "seed": SEED},
        "best_roc_auc": round(best["roc_auc"], 4), "best_pr_auc": round(best["pr_auc"], 4),
        "best_epoch": best["best_epoch"], "last_roc_auc": round(best["last"], 4),
        "paper_roc_auc": PAPER.get(HEAD), "n_test": len(test_set),
        "seconds": round(time.time() - t0, 1),
    }
    json.dump(result, open(f"{OUT}/result.json", "w"), indent=2)
    print("\n=== result ===\n" + json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
