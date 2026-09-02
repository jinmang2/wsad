"""Feature-forensics — *relative* discriminability screen for a feature set.

WHY THIS EXISTS (and what the old ``magnitude_forensics.py`` got wrong)
----------------------------------------------------------------------
The repro VERDICT once leaned on a single proxy — the frame-level ROC-AUC of the
**raw feature L2 magnitude** — as if it were the feature-quality *ceiling* and a
go/no-go *gate* for new extractors. That premise is unsound, and this module
exists to replace it. Measured facts (``scripts/diag``, 290-vid UCF test):

* ``i3d_mgfn`` raw-magnitude AUC = **0.45** (mildly *inverted*, near chance) yet
  RTFM/MGFN train to **0.83** on it. So magnitude-AUC does NOT predict, let alone
  upper-bound, trainable performance — it must not be used as an accept/reject gate.
* Every *atemporal* proxy here (magnitude, content-distance, even a supervised
  linear probe) UNDER-predicts the temporal heads, because WSVAD performance is
  dominated by temporal modeling (RTFM multi-scale, MGFN attention, MIL ranking)
  that no per-snippet score captures. ``i3d_mgfn`` linear-probe ≈ 0.51 → head 0.83.

So treat the numbers below as a **fast relative screen**, not a ceiling: they tell
you whether a set carries *any* linearly-accessible anomaly signal and how two sets
*rank*, cheaply, before spending a GPU. They do NOT tell you the final head AUC.
The only faithful gate is a short temporal-head train (e.g. ``run_matrix.py`` on a
sample). Validated ranking that DID hold downstream-ward (same 40-vid subset):
``videomae`` probe 0.615 > ``i3d_1024`` 0.554 > ``i3d_mgfn`` 0.539.

Three proxies, all snippet-level, crop-averaged, reported as ``|AUC-0.5|`` too
(an inverted 0.45 still carries 0.05 of signal — sign is learnable):
  - **MAGNITUDE**  snippet L2-norm (norm-then-crop-mean = RTFM-faithful).
  - **CONTENT**    L2-distance to the NORMAL-video centroid (content separability).
  - **PROBE**      5-fold GroupKFold(by video) logistic-regression AUC — the
                   strongest atemporal proxy; "how linearly separable is anomaly".

Run with `uv run python` (needs scikit-learn).
"""

import glob
import io
import json
import os
import zipfile
from typing import Iterator, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = os.path.expanduser("~/data/wsad/ucf_crime")
GT = json.load(open(os.path.join(ROOT, "annotations", "ground_truth.json")))
FPC = 16  # frames per snippet (I3D/VideoMAE stride 16; GT is per-frame)


def _bare(name: str) -> str:
    """Bare UCF video id: everything before ``_x264`` (robust to any backbone suffix)."""
    return os.path.basename(name).split("_x264")[0]


def _gt_for(name: str):
    b = _bare(name)
    for k in GT:
        if _bare(k) == b:
            return np.asarray(GT[k], dtype=np.int8)
    return None


def _snip(feat: np.ndarray) -> np.ndarray:
    """(T, ncrops, D) or (T, D) -> per-snippet (T, D) by crop-mean."""
    return feat.mean(axis=1) if feat.ndim == 3 else feat


def _mag(feat: np.ndarray) -> np.ndarray:
    """RTFM-faithful snippet magnitude: per-crop L2 norm, then crop-mean -> (T,).

    NB: norm-then-mean (not crop-mean-then-norm); the old module did the weaker
    crop-mean-then-norm which understates the magnitude signal by Jensen.
    """
    pc = np.linalg.norm(feat, axis=-1)            # (T, ncrops) or (T,)
    return pc.mean(axis=1) if pc.ndim == 2 else pc


def _snip_labels(gt: np.ndarray, T: int) -> np.ndarray:
    """Snippet label = any anomalous frame in its 16-frame window."""
    return np.array([gt[i * FPC:(i + 1) * FPC].max() if i * FPC < len(gt) else 0
                     for i in range(T)], dtype=np.int8)


def _iter_zip(path) -> Iterator[Tuple[str, np.ndarray]]:
    z = zipfile.ZipFile(path)
    for i in z.infolist():
        if i.filename.endswith(".npy"):
            yield i.filename.split("/")[-1], np.load(io.BytesIO(z.read(i)))


def _iter_dir(d) -> Iterator[Tuple[str, np.ndarray]]:
    for f in sorted(glob.glob(os.path.join(d, "**", "*.npy"), recursive=True)):
        yield os.path.basename(f), np.load(f)


def _fmt(auc: float) -> str:
    return f"{auc:.4f}(|dev|{abs(auc - 0.5):.3f})"


def screen(name, it, run_probe=True):
    """Compute the three relative-discriminability proxies for one feature set.

    Returns a dict (also prints a line). ``it`` yields ``(filename, feat)`` where
    ``feat`` is ``(T, ncrops, D)`` or ``(T, D)``.
    """
    vids = []  # (vid, snip_feats (T,D), mag (T,), gt (frames,))
    for vid, feat in it:
        g = _gt_for(vid)
        if g is None:
            continue
        feat = feat.astype(np.float32)
        vids.append((vid, _snip(feat), _mag(feat), g))
    if not vids:
        print(f"  {name}: no aligned videos"); return None

    # normal-video centroid (content reference); fall back to all if no Normal in sample
    nor = [f for v, f, _, _ in vids if "Normal" in v]
    centroid = (np.concatenate(nor, 0) if nor
                else np.concatenate([f for _, f, _, _ in vids], 0)).mean(0)

    mag_s, con_s, allg = [], [], []
    X, y, grp = [], [], []
    for gi, (vid, f, mag, g) in enumerate(vids):
        L = len(np.repeat(mag, FPC))
        mag_s.append(np.repeat(mag, FPC)[:len(g)])
        con_s.append(np.repeat(np.linalg.norm(f - centroid, axis=-1), FPC)[:len(g)])
        allg.append(g[:L])
        X.append(f); y.append(_snip_labels(g, f.shape[0])); grp.append(np.full(f.shape[0], gi))
    g = np.concatenate(allg)
    if len(set(g.tolist())) < 2:
        print(f"  {name}: single-class GT"); return None
    mauc = roc_auc_score(g, np.concatenate(mag_s))
    cauc = roc_auc_score(g, np.concatenate(con_s))

    probe = float("nan")
    if run_probe:
        X = np.concatenate(X); y = np.concatenate(y); grp = np.concatenate(grp)
        Xn = (X - X.mean(0)) / (X.std(0) + 1e-6)
        aucs = []
        for tr, te in GroupKFold(5).split(Xn, y, grp):
            if y[tr].sum() == 0 or y[te].sum() in (0, len(y[te])):
                continue
            clf = LogisticRegression(max_iter=500).fit(Xn[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.decision_function(Xn[te])))
        probe = float(np.mean(aucs)) if aucs else float("nan")

    print(f"  {name:30} n={len(vids):3} | MAG={_fmt(mauc)} | CONTENT={_fmt(cauc)} "
          f"| PROBE={probe:.4f} (atemporal — NOT the head ceiling)")
    return {"name": name, "n": len(vids), "magnitude_auc": mauc,
            "content_auc": cauc, "probe_auc": probe}


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=None, help="screen an arbitrary feature dir (gate use)")
    ap.add_argument("--name", default=None, help="label for --dir")
    ap.add_argument("--no-probe", action="store_true", help="skip the (slower) linear probe")
    args = ap.parse_args()
    print(f"GT: {len(GT)} videos")
    print("RELATIVE atemporal screen (higher = more linearly-accessible signal). "
          "These do NOT predict temporal-head AUC — gate with a short head train.\n")
    if args.dir:
        screen(args.name or os.path.basename(args.dir.rstrip('/')),
               _iter_dir(args.dir), run_probe=not args.no_probe)
        return
    F = os.path.join(ROOT, "features")
    tz = os.path.join(F, "i3d", "test.zip")
    if os.path.exists(tz):
        screen("i3d test.zip (DeepMIL)", _iter_zip(tz), run_probe=not args.no_probe)
    for v in ("i3d_mgfn", "i3d_1024_seg200"):
        d = os.path.join(F, v, "test")
        if os.path.isdir(d):
            screen(f"{v}/test", _iter_dir(d), run_probe=not args.no_probe)


if __name__ == "__main__":
    main()
