"""Efficiency frontier for the dual-branch temporal heads (Spec 2, step 2).

Measures **quality vs cost** for each attention backend on the real UCF-Crime test
set, using an official checkpoint so the AUC is the paper-faithful one and any drop
is attributable to the backend alone (no training variance in the way):

    AUC / AP  — frame-level, identical protocol to ``src.eval_matrix.evaluate``
    FPS       — frames scored per second (the whole test set, GPU-side)
    latency   — per-video wall time, mean and p95
    peak mem  — ``torch.cuda.max_memory_allocated`` over the pass

Backends are given as ``--attn eager,mem,window:64`` — a ``window:W`` entry sets
``WSAD_ATTN=window`` with half-width ``W``. ``eager`` is the reference row; every
other row reports its Δ against it.

    uv run python scripts/diag/bench_efficiency.py \
        --head ur_dmu --variant i3d_1024_seg200 --dim 1024 \
        --ckpt .reference/UR-DMU/models/ucf_trans_2022.pkl \
        --attn eager,mem,window:32,window:64,window:128
"""

import argparse
import json
import os
import statistics
import time

import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from scripts.run_matrix import _build_model, _data_cfg, _load_official, HEADS
from src.eval_matrix import score_video
from src.modules import translayer
from src.trainer import build_datasets


def _parse_backend(spec: str):
    """``"window:64"`` -> ``("window", 64)``; ``"eager"`` -> ``("eager", None)``."""
    name, _, arg = spec.partition(":")
    return name, int(arg) if arg else None


def _bench_one(model, test_set, backbone, head, device, backend):
    """One full test-set pass under ``backend``; returns metrics + cost."""
    name, window = _parse_backend(backend)
    translayer._ATTN_IMPL = name
    if window is not None:
        translayer._ATTN_WINDOW = window

    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    preds, labels, latencies, frames = [], [], [], 0
    for i in range(len(test_set)):
        s = test_set[i]
        t0 = time.perf_counter()
        p = score_video(model, s["feature"], backbone, head, device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

        y = np.asarray(s["label"])
        n = min(len(p), len(y))
        preds.append(p[:n])
        labels.append(y[:n])
        frames += n

    preds, labels = np.concatenate(preds), np.concatenate(labels)
    fpr, tpr, _ = roc_curve(labels, preds)
    precision, recall, _ = precision_recall_curve(labels, preds)
    total = sum(latencies)
    return {
        "attn": backend,
        "roc_auc": float(auc(fpr, tpr)),
        "pr_auc": float(auc(recall, precision)),
        "seconds": total,
        "fps": frames / total,
        "latency_ms": 1000 * statistics.mean(latencies),
        "latency_p95_ms": 1000 * sorted(latencies)[int(0.95 * len(latencies))],
        "peak_mem_mb": (torch.cuda.max_memory_allocated() / 2**20) if device.startswith("cuda") else float("nan"),
        "n_videos": len(test_set),
        "n_frames": frames,
    }


def _table(rows):
    ref = rows[0]
    out = [
        f"# Efficiency frontier — {rows[0]['head']} @ {rows[0]['feature']} "
        f"({ref['n_videos']} videos, {ref['n_frames']} frames)",
        "",
        "| attn | AUC | ΔAUC | AP | FPS | latency ms (mean/p95) | peak mem MB |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        d = r["roc_auc"] - ref["roc_auc"]
        out.append(
            f"| `{r['attn']}` | {r['roc_auc']:.4f} | {d:+.4f} | {r['pr_auc']:.4f} | "
            f"{r['fps']:.0f} | {r['latency_ms']:.0f} / {r['latency_p95_ms']:.0f} | {r['peak_mem_mb']:.0f} |"
        )
    out += ["", "`eager` is the reference row; ΔAUC is measured against it."]
    return "\n".join(out)


def _scaling(backends, lengths, device, dim=512, heads=4, dim_head=64, depth=2):
    """Cost vs sequence length on synthetic input — where does O(T·W) overtake O(T²)?

    The test set tops out around a few thousand snippets, so the asymptotic argument for
    windowing has to be shown on longer T. OOM is recorded, not fatal: that is the number
    that matters (the largest T a backend can score at all).
    """
    from src.modules.translayer import Transformer

    net = Transformer(dim, depth, heads, dim_head, dim * 2).to(device).eval()
    rows = []
    for n in lengths:
        for backend in backends:
            name, window = _parse_backend(backend)
            translayer._ATTN_IMPL = name
            if window is not None:
                translayer._ATTN_WINDOW = window
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            try:
                x = torch.randn(1, n, dim, device=device)
                with torch.no_grad():  # warm up the kernels, then time a clean pass
                    net(x)
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    net(x)
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                mem = (torch.cuda.max_memory_allocated() / 2**20) if device.startswith("cuda") else float("nan")
                rows.append({"T": n, "attn": backend, "ms": 1000 * dt, "peak_mem_mb": mem, "oom": False})
                print(f"  T={n:<7} {backend:<14} {1000 * dt:8.1f} ms  {mem:8.0f} MB")
            except torch.cuda.OutOfMemoryError:
                rows.append({"T": n, "attn": backend, "ms": float("nan"), "peak_mem_mb": float("nan"), "oom": True})
                print(f"  T={n:<7} {backend:<14}      OOM")
                torch.cuda.empty_cache()
    return rows


def _scaling_table(rows, backends):
    lengths = sorted({r["T"] for r in rows})
    by = {(r["T"], r["attn"]): r for r in rows}
    out = ["# Cost vs sequence length (synthetic, 2-layer translayer, dim 512)", "",
           "| T | " + " | ".join(f"`{b}` ms / MB" for b in backends) + " |",
           "|---|" + "---|" * len(backends)]
    for n in lengths:
        cells = []
        for b in backends:
            r = by[(n, b)]
            cells.append("OOM" if r["oom"] else f"{r['ms']:.0f} / {r['peak_mem_mb']:.0f}")
        out.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="ur_dmu", choices=sorted(HEADS))
    ap.add_argument("--backbone", default="i3d")
    ap.add_argument("--variant", default="i3d_1024_seg200")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--ckpt", default=".reference/UR-DMU/models/ucf_trans_2022.pkl")
    ap.add_argument("--attn", default="eager,mem,window:32,window:64,window:128")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="outputs/bench_efficiency")
    ap.add_argument("--scaling", default=None,
                    help="comma-separated sequence lengths — run the synthetic cost-vs-T sweep instead of the test-set eval")
    args = ap.parse_args()

    backends = [b.strip() for b in args.attn.split(",")]
    if args.scaling:
        lengths = [int(v) for v in args.scaling.split(",")]
        rows = _scaling(backends, lengths, args.device)
        os.makedirs(args.out, exist_ok=True)
        json.dump(rows, open(f"{args.out}/scaling.json", "w"), indent=2)
        table = _scaling_table(rows, backends)
        open(f"{args.out}/scaling.md", "w").write(table + "\n")
        print("\n" + table)
        return

    spec = HEADS[args.head]
    _, test_set = build_datasets(_data_cfg(args.backbone, spec, args.variant))
    model, _, _ = _build_model(args.head, args.backbone, args.device, dim=args.dim)
    _load_official(model, args.ckpt, args.head)
    model.eval()

    feat = args.variant if args.backbone == "i3d" else args.backbone
    rows = []
    for backend in backends:
        r = _bench_one(model, test_set, args.backbone, args.head, args.device, backend)
        r.update(head=args.head, feature=feat)
        print(
            f"  {r['attn']:<12} AUC={r['roc_auc']:.4f} AP={r['pr_auc']:.4f} "
            f"FPS={r['fps']:.0f} lat={r['latency_ms']:.0f}ms peak={r['peak_mem_mb']:.0f}MB"
        )
        rows.append(r)

    os.makedirs(args.out, exist_ok=True)
    json.dump(rows, open(f"{args.out}/results.json", "w"), indent=2)
    table = _table(rows)
    open(f"{args.out}/table.md", "w").write(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
