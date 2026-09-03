"""Validate from-scratch training on the CORRECT train features (_archive).

Root cause (verified): `features/i3d/train.zip` (10,32,2048) is MISMATCHED with
`test.zip` — the official MGFN ckpt scores its abnormal videos LOWER than normal
(5% vs 100% on test). The `_archive/UCF_Train_ten_crop_i3d` (T,10,2048) features
ARE consistent (official MGFN 100%). This script trains a head from scratch on the
_archive train features (pooled to 32-seg) and evals on test.zip, logging per-epoch
AUC to a flushed JSONL — to prove training reproduces once the features are right.

    PYTHONPATH=. WSAD_DATA=~/data/wsad python scripts/train_archive_probe.py \
        --head rtfm --lr 5e-5 --epochs 15 --out /tmp/rtfm_archive.jsonl
"""

import argparse
import glob
import io
import json
import os
import time
import zipfile

import numpy as np
import torch
from hydra import compose, initialize
from hydra.utils import _locate, instantiate
from torch.utils.data import DataLoader

import src.models  # noqa: F401
from src.data.features import FeatureDataset
from src.data.local import _align_gt, _load_ground_truth
from src.eval_matrix import evaluate
from src.trainer import _collate_train

BACKBONE_DIM = 2048
_DIM_FIELDS = ("feature_size", "channels", "visual_width")
_SPEC = {"rtfm": "rtfm", "mgfn": "mgfn", "sultani": "mil", "mil": "mil"}


def _seg32(a):  # (10, T, 2048) -> (10, 32, 2048) uniform mean-pool over T
    T = a.shape[1]
    out = np.zeros((a.shape[0], 32, a.shape[2]), np.float32)
    e = np.linspace(0, T, 33, dtype=int)
    for i in range(32):
        lo, hi = e[i], e[i + 1]
        out[:, i] = a[:, lo:hi].mean(1) if hi > lo else a[:, min(lo, T - 1)]
    return out


def _load_archive_train(root):
    d = os.path.join(root, "ucf_crime", "_archive", "UCF_Train_ten_crop_i3d")
    fs = sorted(glob.glob(d + "/*.npy"))
    vals = {}
    for f in fs:
        a = np.load(f).astype(np.float32)  # (T, 10, 2048)
        a = np.transpose(a, (1, 0, 2))  # (10, T, 2048)
        vals[os.path.basename(f)] = _seg32(a)  # (10, 32, 2048)
    return vals


def _load_test_zip(root):
    zp = os.path.join(root, "ucf_crime", "features", "i3d", "test.zip")
    z = zipfile.ZipFile(zp)
    infos = [i for i in z.infolist() if i.filename.endswith(".npy")]
    names = [i.filename.split("/")[-1] for i in infos]
    vals = {n: np.load(io.BytesIO(z.read(i))) for n, i in zip(names, infos)}
    return names, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="rtfm")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/train_archive.jsonl")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = os.path.expanduser(os.environ.get("WSAD_DATA", "~/data/wsad"))

    log = open(args.out, "w")
    def emit(d):
        log.write(json.dumps(d) + "\n"); log.flush(); os.fsync(log.fileno())
        print(d, flush=True)

    tr = _load_archive_train(root)
    names = list(tr)
    normal = [n for n in names if "Normal" in n]
    abnormal = [n for n in names if "Normal" not in n]
    train_sets = {
        "normal": FeatureDataset(normal, {n: tr[n] for n in normal}, with_magnitude=True),
        "abnormal": FeatureDataset(abnormal, {n: tr[n] for n in abnormal}, with_magnitude=True),
    }
    te_names, te_vals = _load_test_zip(root)
    from types import SimpleNamespace
    gt = _align_gt(te_names, _load_ground_truth(SimpleNamespace(root=root, dataset_dir="ucf_crime", ground_truth=None)))
    test_set = FeatureDataset(te_names, te_vals, labels=gt, with_magnitude=True)
    emit({"event": "start", "head": args.head, "lr": args.lr, "n_train_nor": len(normal),
          "n_train_abn": len(abnormal), "n_test": len(test_set)})

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="default", overrides=[f"runner={_SPEC[args.head]}"])
    model_cfg = instantiate(cfg.runner.model_config)
    for f in _DIM_FIELDS:
        if hasattr(model_cfg, f):
            setattr(model_cfg, f, BACKBONE_DIM)
    model = _locate(cfg.runner.model_class)(model_cfg).to(device)
    wd = float(cfg.runner.optimizer.weight_decay)

    import inspect
    accepts_class = "class_labels" in inspect.signature(model.forward).parameters
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=wd)
    loaders = {k: DataLoader(train_sets[k], batch_size=args.batch_size, shuffle=True,
                             drop_last=True, num_workers=4, collate_fn=_collate_train)
               for k in ("normal", "abnormal")}

    aucs = []
    for ep in range(args.epochs):
        t0 = time.time(); model.train(); total = 0.0; n = 0
        for nb, ab in zip(loaders["normal"], loaders["abnormal"]):
            feat = torch.cat([nb["feature"], ab["feature"]], 0).to(device)
            kw = {}
            if accepts_class:
                kw["class_labels"] = torch.cat([nb["class_id"], ab["class_id"]], 0).to(device)
            out = model(video=feat, abnormal_labels=ab["anomaly"].to(device),
                        normal_labels=nb["anomaly"].to(device), **kw)
            opt.zero_grad(); out.loss.backward(); opt.step()
            total += out.loss.item(); n += 1
        m = evaluate(model, test_set, backbone="i3d", head=args.head, device=device)
        aucs.append(m["roc_auc"])
        emit({"epoch": ep, "train_loss": round(total / max(n, 1), 4),
              "roc_auc": round(m["roc_auc"], 4), "pr_auc": round(m["pr_auc"], 4),
              "sec": round(time.time() - t0, 1)})
    a = np.array(aucs)
    emit({"event": "summary", "max": round(float(a.max()), 4), "last": round(float(a[-1]), 4),
          "last5_mean": round(float(a[-5:].mean()), 4), "last5_std": round(float(a[-5:].std()), 4)})
    log.close()


if __name__ == "__main__":
    main()
