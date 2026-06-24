"""Aggregate every reproduction result into one paper-comparison table.

Scans the matrix JSONs under ``outputs/`` and the from-scratch ``result.json`` files
under ``experiments/runs/`` for the BEST clean-feature number per head, and prints a
markdown table (ours vs paper). Clean = the byte-verified MGFN-lineage I3D
(``i3d_mgfn_seg32``, 2048-d) / DeepMIL (``i3d_1024_seg200``, 1024-d) / CLIP — NOT the
contaminated ``features/i3d`` the old ``outputs/matrix/results.json`` (feature=None) used.

    PYTHONPATH=. python experiments/results_table.py            # print
    PYTHONPATH=. python experiments/results_table.py --write README_TABLE.md
"""

import argparse
import glob
import json
import os

PAPER = {  # published UCF-Crime frame-level ROC-AUC
    "sultani": 0.7541, "rtfm": 0.8430, "mgfn": 0.8667, "s3r": 0.8599,
    "ur_dmu": 0.8697, "bn_wvad": 0.8724, "gs_moe": 0.9160,
    "clip_tsa": 0.8758, "vadclip": 0.8801, "tpwng": 0.8779,
}
ORDER = ["sultani", "rtfm", "mgfn", "s3r", "ur_dmu", "bn_wvad", "gs_moe", "clip_tsa", "vadclip", "tpwng"]
# feature lineage each head reproduces on (clean sets only)
FEATURE = {h: "i3d_1024_seg200" for h in ("ur_dmu", "bn_wvad")}
FEATURE.update({h: "clip" for h in ("clip_tsa", "vadclip", "tpwng")})
FEATURE.update({h: "i3d_mgfn_seg32" for h in ("sultani", "rtfm", "mgfn", "s3r", "gs_moe")})

# treat these matrix dirs as authoritative clean results; skip the contaminated default.
CONTAMINATED = {"outputs/matrix"}  # feature=None == features/i3d (MISMATCHED with test)


# the ONLY feature lineage that counts as a faithful reproduction per head; anything
# else (contaminated features/i3d=None, the 2048-d i3d_mgfn_seg200 for memory heads, the
# i3d_pyvideo/tushar probes) is rejected so a stray weak run can't masquerade as the result.
ACCEPT = {h: {"i3d_1024_seg200"} for h in ("ur_dmu", "bn_wvad")}
ACCEPT.update({h: {"clip", None} for h in ("clip_tsa", "vadclip", "tpwng")})
ACCEPT.update({h: {"i3d_mgfn_seg32"} for h in ("sultani", "rtfm", "mgfn", "s3r", "gs_moe")})


def collect() -> dict:
    """head -> best record on the head's canonical clean feature (max roc_auc)."""
    best = {}
    def consider(rec, src):
        h = rec.get("head")
        roc = rec.get("roc_auc", rec.get("best_roc_auc"))
        if not h or roc is None:
            return
        feat = rec.get("feature") or rec.get("variant")
        if feat not in ACCEPT.get(h, set()):  # reject non-canonical / contaminated features
            return
        if h not in best or roc > best[h]["roc"]:
            best[h] = {"roc": roc, "pr": rec.get("pr_auc", rec.get("best_pr_auc")),
                       "feature": feat or "clip", "source": rec.get("source", "trained"), "path": src}

    for jp in glob.glob("outputs/**/results.json", recursive=True):
        if os.path.dirname(jp) in CONTAMINATED:
            continue
        try:
            for rec in json.load(open(jp)):
                consider(rec, jp)
        except Exception:
            continue
    for jp in glob.glob("experiments/runs/**/result.json", recursive=True):
        try:
            consider(json.load(open(jp)), jp)
        except Exception:
            continue
    return best


def render(best: dict) -> str:
    lines = ["# UCF-Crime reproduction — ours vs paper (clean features)", "",
             "Frame-level ROC-AUC on the 290-video test split. `*` = official checkpoint.", "",
             "| head | feature | ours | paper | Δ | source |",
             "|---|---|---|---|---|---|"]
    for h in ORDER:
        paper = PAPER.get(h)
        r = best.get(h)
        if r is None:
            lines.append(f"| {h} | {FEATURE.get(h,'?')} | _pending_ | {paper:.4f} | — | — |")
            continue
        tag = "*" if r["source"] == "official-ckpt" else ""
        delta = f"{r['roc'] - paper:+.4f}" if paper else "—"
        lines.append(f"| {h} | {r['feature']} | {r['roc']:.4f}{tag} | {paper:.4f} | {delta} | `{r['path']}` |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", default=None, help="write the table to this markdown file")
    args = ap.parse_args()
    table = render(collect())
    print(table)
    if args.write:
        with open(args.write, "w") as f:
            f.write(table)
        print(f"wrote {args.write}")


if __name__ == "__main__":
    main()
