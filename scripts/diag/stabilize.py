"""Training-reproduction probe: train one head, log per-epoch ROC-AUC to a FLUSHED
file so progress is monitorable despite ``conda run`` stdout buffering (issue #6).

Uses the correct per-head eval adapter (:mod:`src.eval_matrix`) every epoch, on
``test_dataset=None`` training (no double eval). Writes ``<out>`` as JSONL, one
line per epoch, flushed + fsynced, plus a final summary line.

    PYTHONPATH=. WSAD_DATA=~/data/wsad python scripts/stabilize.py \
        --head rtfm --backbone i3d --lr 5e-5 --epochs 20 --out /tmp/rtfm.jsonl
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from hydra import compose, initialize
from hydra.utils import _locate, instantiate
from omegaconf import OmegaConf

import src.models  # noqa: F401
from src.eval_matrix import evaluate
from src.trainer import WSVADTrainer, build_datasets

BACKBONE_DIM = {"i3d": 2048, "clip": 512}
_DIM_FIELDS = ("feature_size", "channels", "visual_width")

# per-head CLIP preprocessing (I3D zip is pre-segmented to 32 + magnitude already)
_SPEC = {
    "rtfm": dict(runner="rtfm", mag=True, seg=32, length=None),
    "mgfn": dict(runner="mgfn", mag=True, seg=32, length=None),
    "sultani": dict(runner="mil", mag=True, seg=32, length=None),
    "mil": dict(runner="mil", mag=True, seg=32, length=None),
}


def _set_dim(cfg, dim):
    for f in _DIM_FIELDS:
        if hasattr(cfg, f):
            setattr(cfg, f, dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--backbone", default="i3d")
    ap.add_argument("--lr", type=float, default=None, help="override runner lr")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/stabilize.jsonl")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = _SPEC[args.head]

    data_cfg = OmegaConf.create({
        "root": os.environ.get("WSAD_DATA", "~/data/wsad"),
        "source": "local", "backbone": args.backbone, "dataset_dir": "ucf_crime",
        "ground_truth": None, "dynamic_load": False, "batch_size": 1,
        "frames_per_clip": 16, "num_workers": 4,
        "clip_length": spec["length"] if args.backbone == "clip" else None,
        "segment": spec["seg"] if args.backbone == "clip" else None,
        "single_crop": True,
        "with_magnitude": spec["mag"] if args.backbone == "clip" else True,
    })
    train_sets, test_set = build_datasets(data_cfg)

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="default", overrides=[f"runner={spec['runner']}"])
    model_cfg = instantiate(cfg.runner.model_config)
    _set_dim(model_cfg, BACKBONE_DIM[args.backbone])
    model = _locate(cfg.runner.model_class)(model_cfg).to(device)
    lr = args.lr if args.lr is not None else float(cfg.runner.optimizer.learning_rate)
    wd = float(cfg.runner.optimizer.weight_decay)

    # prepare ONCE, then loop epochs manually (mirrors WSVADTrainer.fit body)
    import inspect
    from torch.utils.data import DataLoader
    from src.trainer import _collate_train

    accepts_class = "class_labels" in inspect.signature(model.forward).parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loaders = {k: DataLoader(train_sets[k], batch_size=args.batch_size, shuffle=True,
                             drop_last=True, num_workers=4, collate_fn=_collate_train)
               for k in ("normal", "abnormal")}

    log = open(args.out, "w")
    def emit(d):
        log.write(json.dumps(d) + "\n"); log.flush(); os.fsync(log.fileno())
        print(d, flush=True)

    emit({"event": "start", "head": args.head, "backbone": args.backbone,
          "lr": lr, "wd": wd, "epochs": args.epochs, "batch": args.batch_size,
          "n_test": len(test_set)})

    aucs = []
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        total, nb_steps = 0.0, 0
        for nb, ab in zip(loaders["normal"], loaders["abnormal"]):
            feature = torch.cat([nb["feature"], ab["feature"]], dim=0).to(device)
            kwargs = {}
            if accepts_class:
                kwargs["class_labels"] = torch.cat([nb["class_id"], ab["class_id"]], dim=0).to(device)
            out = model(video=feature, abnormal_labels=ab["anomaly"].to(device),
                        normal_labels=nb["anomaly"].to(device), **kwargs)
            optimizer.zero_grad()
            out.loss.backward()
            optimizer.step()
            total += out.loss.item(); nb_steps += 1
        m = evaluate(model, test_set, backbone=args.backbone, head=args.head, device=device)
        aucs.append(m["roc_auc"])
        emit({"epoch": ep, "train_loss": round(total / max(nb_steps, 1), 4),
              "roc_auc": round(m["roc_auc"], 4), "pr_auc": round(m["pr_auc"], 4),
              "sec": round(time.time() - t0, 1)})

    a = np.array(aucs)
    emit({"event": "summary", "max": round(float(a.max()), 4),
          "last": round(float(a[-1]), 4),
          "last5_mean": round(float(a[-5:].mean()), 4),
          "last5_std": round(float(a[-5:].std()), 4)})
    log.close()


if __name__ == "__main__":
    main()
