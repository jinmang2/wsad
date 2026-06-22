"""``{backbone × head}`` comparison matrix — train + eval, real ROC-AUC (Spec 1 ③).

Crosses the backbone registry (I3D 2048-d, CLIP 512-d) with the head registry,
auto-selecting only compatible pairs (text-branch heads need a text-aligned
backbone — :mod:`src.compat`). Each pair is trained on the cached local features
and evaluated with the per-head eval adapter (:mod:`src.eval_matrix`), then logged
to a JSON + Markdown table.

The original training loop (:class:`src.trainer.WSVADTrainer`) and model defs are
unchanged; per-epoch eval is skipped during training (``test_dataset=None``) and a
single correct eval is run at the end — the matrix only *orchestrates*.

Examples
--------
    # quick smoke of two solid pairs (few epochs)
    PYTHONPATH=. python scripts/run_matrix.py --heads rtfm,vadclip --quick
    # full matrix
    PYTHONPATH=. python scripts/run_matrix.py --epochs 50
    # paper-faithful cells from official checkpoints (no training)
    PYTHONPATH=. python scripts/run_matrix.py --official-only
"""

import argparse
import json
import os
import time
import traceback
from typing import Optional

import torch
from hydra import compose, initialize
from hydra.utils import _locate, instantiate
from omegaconf import OmegaConf

import src.models  # noqa: F401  (register heads)
from src.compat import is_compatible
from src.eval_matrix import evaluate
from src.trainer import WSVADTrainer, build_datasets

BACKBONE_DIM = {"i3d": 2048, "clip": 512}
_DIM_FIELDS = ("feature_size", "channels", "visual_width")

# (backbone, head) -> train batch size override; i3d (2048d) attention / memory-bank
# heads OOM at the default batch on an 8GB GPU. Smaller batch is the only change.
HEAD_BATCH = {("i3d", "ur_dmu"): 4, ("i3d", "bn_wvad"): 4}

# head -> training/data recipe. ``runner`` is the configs/runner/<name>.yaml.
# ``text`` heads need a text-aligned backbone; ``magnitude``/``segment``/``length``
# pick the per-head CLIP preprocessing (I3D zip is pre-segmented to 32 + magnitude).
# PER-PAPER TRAINING (these MIL methods train by ITERATIONS, not epochs):
#   ``steps`` = training iterations, ``batch`` = videos per loader (total = 2*batch
#   = normal+abnormal), ``lr`` = paper learning rate.
#   [ref] = from vendored .reference/<repo>; [est] = best-estimate, verify vs paper.
#   wd/eval_start/grad_clip/ckpt verified vs official code (sciomc audit 2026-06-17).
HEADS = {
    "sultani": dict(runner="mil", text=False, magnitude=True, segment=32, length=None, steps=5000, batch=30, lr=1e-3, wd=5e-4),    # [est] Sultani CVPR'18
    "rtfm": dict(runner="rtfm", text=False, magnitude=True, segment=32, length=None, steps=15000, batch=32, lr=1e-3, wd=5e-3, eval_start=200),  # [ref] RTFM main.py:41 wd=0.005
    "mgfn": dict(runner="mgfn", text=False, magnitude=True, segment=32, length=None, steps=5000, batch=8, lr=1e-3, wd=5e-4),       # [est] MGFN (paper 0.8667)
    "ur_dmu": dict(runner="ur_dmu", text=False, magnitude=True, segment=32, length=None, steps=3000, batch=64, lr=1e-4, wd=5e-5),  # [ref] UR-DMU wd=5e-5 (i3d-OOM 8GB)
    "bn_wvad": dict(runner="bn_wvad", text=False, magnitude=True, segment=32, length=None, steps=1000, batch=64, lr=1e-4, wd=5e-5, grad_clip=1.0, ckpt="pr_auc"),  # [ref] BN-WVAD best on AP
    "s3r": dict(runner="s3r", text=False, magnitude=True, segment=32, length=None, steps=15000, batch=32, lr=1e-3, wd=5e-3, eval_start=5000),  # [ref] S3R wd=0.005, 5000 warmup
    "gs_moe": dict(runner="gs_moe", text=False, magnitude=True, segment=32, length=None, steps=5000, batch=64, lr=1e-4, wd=5e-4),  # [est] GS-MoE
    "clip_tsa": dict(runner="clip_tsa", text=False, magnitude=False, segment=32, length=None, steps=4000, batch=16, lr=1e-3, wd=5e-3),  # [ref] CLIP-TSA main.py:122 wd=0.005
    "vadclip": dict(runner="vadclip", text=True, magnitude=False, segment=None, length=256, steps=3000, batch=64, lr=2e-5, wd=1e-2),    # [ref] VadCLIP AdamW (official ckpt used)
    "tpwng": dict(runner="tpwng", text=True, magnitude=False, segment=32, length=None, steps=3000, batch=16, lr=1e-3, wd=5e-3),    # [est] TPWNG (paper-only)
}

# paper-faithful official head checkpoints present locally (eval-only).
OFFICIAL_CKPT = {
    ("clip", "vadclip"): "pretrained/vadclip/model_ucf.pth",
    ("i3d", "mgfn"): "pretrained/mgfn/mgfn_ucf.pkl",
}


def _set_dim(cfg, dim: int) -> None:
    for f in _DIM_FIELDS:
        if hasattr(cfg, f):
            setattr(cfg, f, dim)


def _data_cfg(backbone: str, spec: dict, variant: str = "i3d"):
    dim_seg = spec["segment"] if backbone == "clip" else None  # I3D zip already 32-seg
    return OmegaConf.create(
        {
            "root": os.environ.get("WSAD_DATA", "~/data/wsad"),
            "source": "local",
            "backbone": backbone,
            "dataset_dir": "ucf_crime",
            "feature_variant": variant,  # which I3D extraction (i3d_tushar/i3d_pyvideo/i3d_ours)
            "ground_truth": None,
            # i3d: lazy zip load (avoids eager-loading 1610 npy ~12.8 GB into the
            # 15 GB WSL RAM, which crashes WSL). clip uses npy dirs (unaffected).
            "dynamic_load": backbone == "i3d",
            "batch_size": 1,
            "frames_per_clip": 16,
            # i3d lazy-zip: the ZipFile handle isn't shareable across DataLoader
            # worker processes (BadZipFile), so load in-process. clip = picklable npy.
            "num_workers": 0 if backbone == "i3d" else 4,
            "clip_length": spec["length"],
            "segment": dim_seg,
            "single_crop": True,
            "with_magnitude": spec["magnitude"] if backbone == "clip" else True,
        }
    )


def _build_model(head: str, backbone: str, device: str):
    spec = HEADS[head]
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="default", overrides=[f"runner={spec['runner']}"])
    model_cfg = instantiate(cfg.runner.model_config)
    _set_dim(model_cfg, BACKBONE_DIM[backbone])
    model = _locate(cfg.runner.model_class)(model_cfg)
    lr = float(cfg.runner.optimizer.learning_rate)
    wd = float(cfg.runner.optimizer.weight_decay)
    return model.to(device), lr, wd


def _official_converter(head: str):
    """Return the official-checkpoint -> repo-state converter for a head."""
    if head == "vadclip":
        from src.models.vadclip.modeling_vadclip import convert_official_vadclip

        return convert_official_vadclip
    if head == "mgfn":
        from scripts.convert_official_to_hf import convert

        return convert
    return None


def _load_official(model, path: str, head: str):
    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    conv = _official_converter(head)
    if conv is not None:
        state = conv(state)
    clean = {k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(clean or state, strict=False)
    if missing or unexpected:
        print(f"  [official load] missing={len(missing)} unexpected={len(unexpected)}")
    return model


def run_pair(
    backbone: str,
    head: str,
    device: str,
    official: bool,
    variant: str = "i3d",
    wandb_proj: Optional[str] = None,
    steps_override: Optional[int] = None,
    batch_override: Optional[int] = None,
    lr_override: Optional[float] = None,
    lr_decay: Optional[str] = None,
    eval_every: Optional[int] = None,
    eval_start_override: Optional[int] = None,
) -> dict:
    spec = HEADS[head]
    t0 = time.time()
    feat_tag = variant if backbone == "i3d" else backbone
    data_cfg = _data_cfg(backbone, spec, variant)
    train_sets, test_set = build_datasets(data_cfg)
    model, cfg_lr, wd = _build_model(head, backbone, device)
    # per-paper recipe (HEADS[head]) with optional CLI overrides
    max_steps = steps_override or spec.get("steps", 3000)
    lr = lr_override or spec.get("lr") or cfg_lr
    eval_interval = eval_every or max(max_steps // 30, 50)

    if official and (backbone, head) in OFFICIAL_CKPT:
        _load_official(model, OFFICIAL_CKPT[(backbone, head)], head)
        metrics = evaluate(model, test_set, backbone=backbone, head=head, device=device)
        return {
            "backbone": backbone, "head": head, "feature": feat_tag, "dim": BACKBONE_DIM[backbone],
            "steps": 0, "source": "official-ckpt",
            "roc_auc": round(metrics["roc_auc"], 4), "pr_auc": round(metrics["pr_auc"], 4),
            "best_step": -1, "n_test": len(test_set), "seconds": round(time.time() - t0, 1),
        }

    # thin orchestration: per-paper recipe (HEADS) -> the canonical WSVADTrainer.fit_steps.
    from src.trainer import WSVADTrainer

    bs = HEAD_BATCH.get((backbone, head)) or batch_override or spec.get("batch", 16)
    wd_v = spec.get("wd", wd)
    grad_clip = spec.get("grad_clip")
    ckpt_metric = spec.get("ckpt", "roc_auc")  # BN-WVAD selects on AP (pr_auc)
    eval_start = eval_start_override if eval_start_override is not None else spec.get("eval_start", 0)

    run = None
    if wandb_proj:
        import wandb
        run = wandb.init(project=wandb_proj, name=f"{feat_tag}-{head}", reinit=True,
                         config={"feature": feat_tag, "backbone": backbone, "head": head,
                                 "lr": lr, "wd": wd_v, "batch": 2 * bs, "steps": max_steps,
                                 "grad_clip": grad_clip, "ckpt_metric": ckpt_metric,
                                 "n_train": len(train_sets["normal"]) + len(train_sets["abnormal"]),
                                 "n_test": len(test_set)})

    def eval_fn(m):  # per-head eval; GPU-only (no CPU fallback -> clip_tsa >20GiB kills WSL)
        try:
            res = evaluate(m, test_set, backbone=backbone, head=head, device=device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise
        if run:
            run.log({"roc_auc": res["roc_auc"], "pr_auc": res["pr_auc"]})
        return res

    trainer = WSVADTrainer(model=model, learning_rate=lr, weight_decay=wd_v, batch_size=bs,
                           num_workers=int(data_cfg.num_workers), frames_per_clip=16,
                           mixed_precision="no", grad_clip_norm=grad_clip)
    best = trainer.fit_steps(train_sets, max_steps=max_steps, eval_fn=eval_fn,
                             eval_interval=eval_interval, eval_start=eval_start,
                             select_metric=ckpt_metric, lr_decay=lr_decay)
    if run:
        run.summary.update({"best_roc_auc": best["roc_auc"], "best_pr_auc": best["pr_auc"],
                            "best_step": best["best_step"], "last_auc": best["last"]})
        run.finish()
    return {
        "backbone": backbone, "head": head, "feature": feat_tag, "dim": BACKBONE_DIM[backbone],
        "steps": max_steps, "batch": 2 * bs, "lr": lr, "wd": wd_v, "source": "trained",
        "roc_auc": round(best["roc_auc"], 4), "pr_auc": round(best["pr_auc"], 4),
        "best_step": best["best_step"], "last_auc": round(best["last"], 4),
        "auc_mean": best.get("auc_mean"), "auc_std": best.get("auc_std"), "lr_decay": lr_decay,
        "n_test": len(test_set), "seconds": round(time.time() - t0, 1),
    }


def write_table(results: list, out_md: str) -> None:
    backbones = sorted({r["backbone"] for r in results if "roc_auc" in r})
    heads = [h for h in HEADS if any(r["head"] == h for r in results)]
    cell = {(r["backbone"], r["head"]): r for r in results}
    lines = ["# {backbone × head} ROC-AUC matrix", "", "Frame-level ROC-AUC on UCF-Crime test (290 videos).", ""]
    lines.append("| head \\ backbone | " + " | ".join(f"{b} ({BACKBONE_DIM[b]}d)" for b in backbones) + " |")
    lines.append("|" + "---|" * (len(backbones) + 1))
    for h in heads:
        row = [h]
        for b in backbones:
            r = cell.get((b, h))
            if r is None:
                row.append("·")
            elif "error" in r:
                row.append("err")
            else:
                tag = "*" if r["source"] == "official-ckpt" else ""
                row.append(f"{r['roc_auc']:.4f}{tag}")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "`*` = official checkpoint (eval-only, paper-faithful); others trained from scratch.", ""]
    with open(out_md, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="{backbone × head} comparison matrix")
    ap.add_argument("--backbones", default="i3d,clip")
    ap.add_argument("--heads", default=",".join(HEADS))
    ap.add_argument("--steps", type=int, default=None, help="override per-paper step count (HEADS recipe)")
    ap.add_argument("--batch", type=int, default=None, help="override per-loader batch (total=2x)")
    ap.add_argument("--quick", action="store_true", help="200 steps (smoke)")
    ap.add_argument("--official", action="store_true", help="use official ckpt where available")
    ap.add_argument("--official-only", action="store_true", help="only eval official-ckpt pairs")
    ap.add_argument("--out", default="outputs/matrix")
    ap.add_argument("--force", action="store_true", help="recompute pairs already in results")
    ap.add_argument("--variant", default="i3d", help="i3d feature variant: i3d_tushar/i3d_pyvideo/i3d_ours")
    ap.add_argument("--wandb", default=None, help="wandb project name (enables per-cell logging)")
    ap.add_argument("--lr", type=float, default=None, help="override runner lr (e.g. 5e-5)")
    ap.add_argument("--lr-decay", default=None, choices=[None, "cosine"], help="per-step LR decay (stabilizes constant-lr divergence)")
    ap.add_argument("--eval-every", type=int, default=None, help="override eval interval in steps (MGFN overfits early -> eval often to catch the peak)")
    ap.add_argument("--eval-start", type=int, default=None, help="override step to start evaluating (default = per-head recipe)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    steps_override = 200 if args.quick else args.steps
    os.makedirs(args.out, exist_ok=True)
    json_path, md_path = f"{args.out}/results.json", f"{args.out}/matrix.md"

    results = []
    done = set()
    if os.path.exists(json_path) and not args.force:
        results = json.load(open(json_path))
        # re-run failed cells; only treat successful pairs as done
        done = {(r["backbone"], r["head"]) for r in results if "error" not in r}

    backbones = args.backbones.split(",")
    heads = args.heads.split(",")
    for backbone in backbones:
        for head in heads:
            if head not in HEADS:
                continue
            if not is_compatible(backbone, head):
                continue
            if args.official_only and (backbone, head) not in OFFICIAL_CKPT:
                continue
            if (backbone, head) in done:
                print(f"[skip] {backbone}×{head} (already in results)")
                continue
            print(f"\n=== {backbone} × {head} (feature={args.variant if backbone=='i3d' else backbone}, steps={steps_override or HEADS[head].get('steps')}) ===")
            try:
                r = run_pair(backbone, head, device,
                             args.official or args.official_only, variant=args.variant,
                             wandb_proj=args.wandb, steps_override=steps_override,
                             batch_override=args.batch, lr_override=args.lr,
                             lr_decay=args.lr_decay, eval_every=args.eval_every,
                             eval_start_override=args.eval_start)
                print("  ->", r)
            except Exception as e:
                r = {"backbone": backbone, "head": head, "error": str(e)[:300]}
                print("  !! FAILED:", str(e)[:200])
                traceback.print_exc()
            results = [x for x in results if not (x["backbone"] == backbone and x["head"] == head)]
            results.append(r)
            json.dump(results, open(json_path, "w"), indent=2)
            write_table(results, md_path)

    print(f"\nwrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
