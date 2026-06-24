"""End-to-end MGFN eval on LOCAL test.zip + LOCAL ground_truth.json.

Validates: official ckpt -> convert -> load -> data loader -> gt alignment ->
frame-level ROC-AUC. Bypasses the HF force_download in build_feature_dataset.
"""

import json
import os
import zipfile

import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_curve

import src.models  # noqa: F401
from scripts.convert_official_to_hf import convert
from src.data.features import FeatureDataset
from src.models.mgfn.configuration_mgfn import MGFNConfig
from src.models.mgfn.modeling_mgfn import MGFNForVideoAnomalyDetection

ROOT = os.path.join(os.path.expanduser(os.environ.get("WSAD_DATA", "~/data/wsad")), "ucf_crime")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

# --- model ---
model = MGFNForVideoAnomalyDetection(MGFNConfig())
official = torch.load("pretrained/mgfn/mgfn_ucf.pkl", map_location="cpu", weights_only=False)
m, u = model.load_state_dict(convert(official), strict=False)
assert not m and not u, (len(m), len(u))
model = model.to(DEVICE).eval()

# --- data (local, dynamic load) — resolve the new features/i3d + annotations
# layout (legacy dataset-root fallback) via the loader's own resolvers ---
from types import SimpleNamespace

from src.data.local import _i3d_zip_path, _local_ground_truth_path

_cfg = SimpleNamespace(root=os.path.dirname(ROOT), dataset_dir="ucf_crime", ground_truth=None)
z = zipfile.ZipFile(_i3d_zip_path(os.path.dirname(ROOT), _cfg, "test"))
infos = [i for i in z.infolist() if not i.is_dir()]
filenames = [i.filename.split("/")[-1] for i in infos]
values = {fn: info for fn, info in zip(filenames, infos)}
gt = json.load(open(_local_ground_truth_path(_cfg)))
ds = FeatureDataset(filenames, values, labels=gt, open_func=z.open, with_magnitude=True)
print("test videos:", len(ds))


@torch.no_grad()
def score(feature):
    x = torch.as_tensor(feature, dtype=torch.float32, device=DEVICE)
    x = x.permute(1, 0, 2).unsqueeze(0)  # (T,crop,D)->(1,crop,T,D)
    out = model(video=x)
    s = out.scores.squeeze(0).squeeze(-1).cpu().numpy()  # (T,)
    return np.repeat(s, 16)


all_p, all_l = [], []
for i in range(len(ds)):
    s = ds[i]
    p = score(s["feature"])
    l = np.asarray(s["label"])
    n = min(len(p), len(l))
    all_p.append(p[:n])
    all_l.append(l[:n])

preds = np.concatenate(all_p)
labels = np.concatenate(all_l)
fpr, tpr, _ = roc_curve(labels, preds)
roc = auc(fpr, tpr)
pre, rec, _ = precision_recall_curve(labels, preds)
prauc = auc(rec, pre)
print(f"\nframes evaluated: {len(preds)}  positives: {int(labels.sum())}")
print(f"MGFN UCF  ROC-AUC = {roc:.4f}   PR-AUC = {prauc:.4f}")
print("paper/official MGFN UCF ROC-AUC ~ 0.8667")
