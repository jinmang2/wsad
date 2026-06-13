"""Definitive VadCLIP equivalence: run the OFFICIAL CLIPVAD and the repo PORT over
the full UCF-Crime test set (official ucf_CLIP_rgbtest.csv, crop __5) with the
official gt_ucf.npy, using the official ucf_test.py protocol. Reports AUC1/AUC2
for both and the worst per-frame divergence.

Run from .reference/VadCLIP/src as cwd:
  cd .reference/VadCLIP/src && PYTHONPATH=.:/home/jinmang2/wsad \
    conda run -n balaenoptera python /home/jinmang2/wsad/scripts/eval_vadclip_ucf.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

REF = "/home/jinmang2/wsad/.reference/VadCLIP/src"
ROOT = "/home/jinmang2/wsad"
sys.path.insert(0, REF)

import clip.clip as clipmod  # noqa: E402
import utils.tools as tools  # noqa: E402
from clip.model import build_model  # noqa: E402

sys.path.insert(0, ROOT)
from src.models.vadclip.configuration_vadclip import VadCLIPConfig  # noqa: E402
from src.models.vadclip.modeling_vadclip import (  # noqa: E402
    VadCLIPForVideoAnomalyDetection,
    convert_official_vadclip,
)

CKPT = f"{ROOT}/pretrained/vadclip/model_ucf.pth"
TESTCSV = f"{REF}/../list/ucf_CLIP_rgbtest.csv"
GT = f"{REF}/../list/gt_ucf.npy"
CLIPDIR = "/home/jinmang2/data/wsad/ucf_crime/UCFClipFeatures"
MAXLEN = 256
device = "cuda"

label_map = ["Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary",
             "Explosion", "Fighting", "RoadAccidents", "Robbery", "Shooting",
             "Shoplifting", "Stealing", "Vandalism"]

# --- official model (CLIP built from ckpt, no network) ---
ckpt = torch.load(CKPT, map_location="cpu")
clip_sd = {k[len("clipmodel."):]: v for k, v in ckpt.items() if k.startswith("clipmodel.")}
clip_model = build_model(clip_sd).to(device).eval()
clipmod.load = lambda name, dev: (clip_model.to(dev), None)
import model as off  # noqa: E402

off.clip.load = clipmod.load
off_model = off.CLIPVAD(14, 512, MAXLEN, 512, 1, 2, 8, 10, 10, device)
off_model.load_state_dict(ckpt)
off_model.to(device).eval()

# --- repo port ---
port = VadCLIPForVideoAnomalyDetection(VadCLIPConfig())
port.load_state_dict(convert_official_vadclip(ckpt), strict=False)
port.to(device).eval()

df = pd.read_csv(TESTCSV)
gt = np.load(GT)


def resolve(path):
    base = os.path.basename(path)
    cls = base.split("__")[0].split("_x264")[0]
    # find the class subdir (UCFClipFeatures/<Class>/<base>)
    for d in os.listdir(CLIPDIR):
        cand = os.path.join(CLIPDIR, d, base)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(base)


def lengths_for(length):
    out = torch.zeros(int(length / MAXLEN) + 1)
    rem = length
    for j in range(int(length / MAXLEN) + 1):
        if j == 0 and rem < MAXLEN:
            out[j] = rem
        elif j == 0 and rem > MAXLEN:
            out[j] = MAXLEN; rem -= MAXLEN
        elif rem > MAXLEN:
            out[j] = MAXLEN; rem -= MAXLEN
        else:
            out[j] = rem
    return out.to(int)


@torch.no_grad()
def both(vid_path):
    feat = np.load(vid_path).astype(np.float32)
    split, clip_length = tools.process_split(feat, MAXLEN)
    visual = torch.tensor(split)
    length = int(clip_length)
    if length < MAXLEN:
        visual = visual.unsqueeze(0)
    visual = visual.to(device)
    lengths = lengths_for(length)
    pad_mask = tools.get_batch_mask(lengths, MAXLEN).to(device)

    _, ol1, ol2 = off_model(visual, pad_mask, label_map, lengths)
    pout = port(video=visual, lengths=lengths)
    pl1, pl2 = pout.binary_logits, pout.alignment_logits

    def flat(x):
        return x.reshape(x.shape[0] * x.shape[1], x.shape[2])

    ol1, ol2, pl1, pl2 = flat(ol1), flat(ol2), flat(pl1), flat(pl2)
    diff = max(float((ol1[:length] - pl1[:length]).abs().max()),
              float((ol2[:length] - pl2[:length]).abs().max()))
    # official prob1 (binary), prob2 (alignment)

    def probs(l1, l2):
        p1 = torch.sigmoid(l1[:length].squeeze(-1))
        p2 = 1 - l2[:length].softmax(dim=-1)[:, 0].squeeze(-1)
        return p1.cpu().numpy(), p2.cpu().numpy()

    return probs(ol1, ol2), probs(pl1, pl2), diff


off_ap1, off_ap2, port_ap1, port_ap2 = [], [], [], []
worst = 0.0
for i in range(len(df)):
    p, label = df.loc[i]["path"], df.loc[i]["label"]
    (o1, o2), (q1, q2), d = both(resolve(p))
    off_ap1.append(o1); off_ap2.append(o2); port_ap1.append(q1); port_ap2.append(q2)
    worst = max(worst, d)

off_ap1 = np.repeat(np.concatenate(off_ap1), 16)
off_ap2 = np.repeat(np.concatenate(off_ap2), 16)
port_ap1 = np.repeat(np.concatenate(port_ap1), 16)
port_ap2 = np.repeat(np.concatenate(port_ap2), 16)

print(f"\ntest videos: {len(df)}   gt frames: {len(gt)}   pred frames: {len(off_ap1)}")
print(f"worst per-frame |Δ| (official vs port): {worst:.2e}")
print("\n            AUC1(binary)   AP1       AUC2(align)   AP2")
for name, a1, a2 in [("official", off_ap1, off_ap2), ("port    ", port_ap1, port_ap2)]:
    print(f"  {name}  {roc_auc_score(gt, a1):.6f}     {average_precision_score(gt, a1):.6f}  "
          f"{roc_auc_score(gt, a2):.6f}     {average_precision_score(gt, a2):.6f}")
print("\npaper VadCLIP UCF AUC ~ 0.88 (binary)")
