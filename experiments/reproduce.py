"""Single parametrized from-scratch reproduction harness for the WSVAD heads.

Replaces ``experiments/repro_i3d.py`` + ``experiments/repro_vadclip.py``: one
``RECIPES`` registry holds the VERIFIED from-scratch recipe per head and a single
``main()`` drives the canonical :class:`src.trainer.WSVADTrainer.fit` + the per-head
:func:`src.eval_matrix.evaluate`. The recipes are the ones that actually reproduce
(epoch-based, Adam lr 5e-5 for I3D; the official AdamW/MultiStepLR for VadCLIP) —
NOT ``run_matrix``'s generic step-based lr 1e-3, which diverges to chance (0.52) on
these heads. ``run_matrix.py`` stays the {variant × head} *matrix* orchestrator and
owns the step-based official recipes for the memory heads (UR-DMU / BN-WVAD).

    HEAD=mgfn    python experiments/reproduce.py        # I3D 2048-d seg32
    HEAD=rtfm    python experiments/reproduce.py
    HEAD=s3r     python experiments/reproduce.py
    HEAD=vadclip python experiments/reproduce.py        # CLIP 512-d, AdamW MultiStepLR

Every field is env-overridable for sweeps: VARIANT, DIM, LR, EPOCHS, BATCH, SEED.

NOTE (feature provenance, open): the I3D recipe reaches MGFN ~0.81 on the
``i3d_mgfn_seg32`` train set (the byte-verified MGFN distribution, REPRO_STATUS §2),
while ``scripts/diag/train_archive_probe.py`` reached 0.8667 training on the
``_archive/UCF_Train_ten_crop_i3d`` set — i.e. the residual MGFN gap tracks the
*train-feature lineage*, not the recipe. Override VARIANT to compare.
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

# Verified from-scratch recipe per head. ``wd=None`` -> use the runner-config weight
# decay. ``milestones`` -> MultiStepLR (VadCLIP); ``freeze_clip``/``all_crops`` are
# VadCLIP-only knobs. ``paper`` is the published UCF-Crime frame-level ROC-AUC.
RECIPES = {
    "mgfn":    dict(backbone="i3d",  variant="i3d_mgfn_seg32", dim=2048, optimizer="adam",  lr=5e-5, wd=None, epochs=15, batch=16, seed=0,   paper=0.8667),
    "rtfm":    dict(backbone="i3d",  variant="i3d_mgfn_seg32", dim=2048, optimizer="adam",  lr=5e-5, wd=None, epochs=15, batch=16, seed=0,   paper=0.8430),
    "s3r":     dict(backbone="i3d",  variant="i3d_mgfn_seg32", dim=2048, optimizer="adam",  lr=5e-5, wd=None, epochs=15, batch=16, seed=0,   paper=0.8599),
    "sultani": dict(backbone="i3d",  variant="i3d_mgfn_seg32", dim=2048, optimizer="adam",  lr=5e-5, wd=None, epochs=15, batch=16, seed=0,   paper=0.8300),
    "vadclip": dict(backbone="clip", variant=None,             dim=512,  optimizer="adamw", lr=2e-5, wd=0.01, epochs=10, batch=64, seed=234,
                    milestones=[4, 8], gamma=0.1, eval_every=10, freeze_clip=True, all_crops=True, paper=0.8801, official_ckpt=0.8770),
}


def _env(key, cast, default):
    v = os.environ.get(key)
    return cast(v) if v is not None else default


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _data_cfg(rec: dict):
    """Loader config matching the two original scripts (eager npy dirs, single-crop)."""
    backbone = rec["backbone"]
    cfg = OmegaConf.create({
        "root": os.environ.get("WSAD_DATA", "~/data/wsad"), "source": "local",
        "backbone": backbone, "dataset_dir": "ucf_crime",
        "feature_variant": rec["variant"], "ground_truth": None,
        "dynamic_load": False, "batch_size": 1, "frames_per_clip": 16,
        "num_workers": 4, "clip_length": 256 if backbone == "clip" else None,
        "segment": None, "single_crop": True, "with_magnitude": True,
    })
    if backbone == "clip" and rec.get("all_crops"):
        cfg.train_all_crops = True  # VadCLIP trains all 10 crops as separate rows
    return cfg


def main() -> None:
    head = os.environ.get("HEAD", "mgfn")
    if head not in RECIPES:
        raise SystemExit(f"HEAD={head} not in RECIPES {list(RECIPES)}; UR-DMU/BN-WVAD use run_matrix.py")
    rec = dict(RECIPES[head])
    # env overrides
    rec["variant"] = os.environ.get("VARIANT", rec["variant"])
    rec["dim"] = _env("DIM", int, rec["dim"])
    rec["lr"] = _env("LR", float, rec["lr"])
    rec["epochs"] = _env("EPOCHS", int, rec["epochs"])
    rec["batch"] = _env("BATCH", int, rec["batch"])
    rec["seed"] = _env("SEED", int, rec["seed"])

    out = f"experiments/runs/{head}_{rec['variant'] or rec['backbone']}/seed{rec['seed']}"
    os.makedirs(out, exist_ok=True)
    set_seed(rec["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    data_cfg = _data_cfg(rec)
    train_sets, test_set = build_datasets(data_cfg)
    model, _, cfg_wd = _build_model(head, rec["backbone"], device, dim=rec["dim"])
    wd = rec["wd"] if rec["wd"] is not None else cfg_wd

    n_clip = None
    if rec.get("freeze_clip"):  # VadCLIP: random head + FROZEN pretrained CLIP text tower
        from src.models.vadclip.modeling_vadclip import load_pretrained_clip_text
        n_clip = load_pretrained_clip_text(model)

    n_train = len(train_sets["normal"]) + len(train_sets["abnormal"])
    print(f"[data] head={head} backbone={rec['backbone']} variant={rec['variant']} "
          f"dim={rec['dim']} train={n_train} test={len(test_set)}", flush=True)
    sched = f" MultiStepLR{rec['milestones']}x{rec['gamma']}" if rec.get("milestones") else ""
    clip_msg = f" +froze {n_clip} CLIP text tensors" if n_clip else ""
    print(f"[recipe] {rec['optimizer']} lr={rec['lr']} wd={wd}{sched} "
          f"epochs={rec['epochs']} batch={rec['batch']}(+{rec['batch']}) seed={rec['seed']}{clip_msg}", flush=True)

    log = []

    def eval_fn(m):
        res = evaluate(m, test_set, backbone=rec["backbone"], head=head, device=device)
        log.append({"roc_auc": res["roc_auc"], "pr_auc": res["pr_auc"]})
        json.dump(log, open(f"{out}/epoch_log.json", "w"), indent=2)
        return res

    trainer = WSVADTrainer(
        model=model, learning_rate=rec["lr"], weight_decay=wd, batch_size=rec["batch"],
        num_workers=int(data_cfg.num_workers), frames_per_clip=16, mixed_precision="no",
        optimizer_name=rec["optimizer"],
        scheduler_milestones=rec.get("milestones"), scheduler_gamma=rec.get("gamma", 0.1),
    )
    best = trainer.fit(train_sets, epochs=rec["epochs"], eval_fn=eval_fn,
                       select_metric="roc_auc", eval_every=rec.get("eval_every"))

    result = {
        "head": head, "backbone": rec["backbone"], "variant": rec["variant"],
        "dim": rec["dim"], "source": "from-scratch",
        "recipe": {"optimizer": rec["optimizer"], "lr": rec["lr"], "wd": wd,
                   "milestones": rec.get("milestones"), "gamma": rec.get("gamma"),
                   "epochs": rec["epochs"], "batch_total": 2 * rec["batch"], "seed": rec["seed"]},
        "best_roc_auc": round(best["roc_auc"], 4), "best_pr_auc": round(best["pr_auc"], 4),
        "best_epoch": best["best_epoch"], "last_roc_auc": round(best["last"], 4),
        "paper_roc_auc": rec["paper"], "official_ckpt_roc_auc": rec.get("official_ckpt"),
        "n_test": len(test_set), "seconds": round(time.time() - t0, 1),
    }
    json.dump(result, open(f"{out}/result.json", "w"), indent=2)
    print("\n=== result ===\n" + json.dumps(result, indent=2), flush=True)
    print(f"\nwrote {out}/result.json", flush=True)


if __name__ == "__main__":
    main()
