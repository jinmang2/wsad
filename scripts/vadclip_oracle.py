"""Run the OFFICIAL CLIPVAD (.reference/VadCLIP) with model_ucf.pth to produce
reference outputs (text_features_ori, logits1, logits2) on real local CLIP
features. CLIP is built from the ckpt's clipmodel.* weights (no network).

Replicates ucf_test.py's per-video length/padding bookkeeping exactly.
"""

import sys

import numpy as np
import torch

REF = "/home/jinmang2/wsad/.reference/VadCLIP/src"
sys.path.insert(0, REF)

import clip.clip as clipmod  # noqa: E402
from clip.model import build_model  # noqa: E402

CKPT = "/home/jinmang2/wsad/pretrained/vadclip/model_ucf.pth"
VIDS = ["Abuse028_x264", "RoadAccidents133_x264", "Normal_Videos_867_x264"]
MAXLEN = 256
device = "cuda"

ckpt = torch.load(CKPT, map_location="cpu")
clip_sd = {k[len("clipmodel."):]: v for k, v in ckpt.items() if k.startswith("clipmodel.")}
clip_model = build_model(clip_sd).to(device).eval()


def fake_load(name, dev):
    return clip_model.to(dev), None


clipmod.load = fake_load
import model as off  # noqa: E402

off.clip.load = fake_load

m = off.CLIPVAD(14, 512, MAXLEN, 512, 1, 2, 8, 10, 10, device)
missing, unexpected = m.load_state_dict(ckpt, strict=False)
print("official load: missing=%d unexpected=%d" % (len(missing), len(unexpected)))
m.to(device).eval()

label_map = ["Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary",
             "Explosion", "Fighting", "RoadAccidents", "Robbery", "Shooting",
             "Shoplifting", "Stealing", "Vandalism"]

import utils.tools as tools  # noqa: E402


def run_one(vid):
    feat = np.load(f"/home/jinmang2/data/wsad/clip/test/{vid}__0.npy").astype(np.float32)
    split, clip_length = tools.process_split(feat, MAXLEN)
    visual = torch.tensor(split)
    length = int(clip_length)
    len_cur = length
    if len_cur < MAXLEN:
        visual = visual.unsqueeze(0)
    visual = visual.to(device)

    lengths = torch.zeros(int(length / MAXLEN) + 1)
    rem = length
    for j in range(int(length / MAXLEN) + 1):
        if j == 0 and rem < MAXLEN:
            lengths[j] = rem
        elif j == 0 and rem > MAXLEN:
            lengths[j] = MAXLEN
            rem -= MAXLEN
        elif rem > MAXLEN:
            lengths[j] = MAXLEN
            rem -= MAXLEN
        else:
            lengths[j] = rem
    lengths = lengths.to(int)
    pad_mask = tools.get_batch_mask(lengths, MAXLEN).to(device)

    with torch.no_grad():
        tf, l1, l2 = m(visual, pad_mask, label_map, lengths)
    return dict(
        vid=vid, raw_T=feat.shape[0], split=tuple(visual.shape),
        text=tf.float().cpu().numpy(), logits1=l1.float().cpu().numpy(),
        logits2=l2.float().cpu().numpy(),
    )


out = {}
for v in VIDS:
    r = run_one(v)
    out[v] = r
    print(f"{v}: rawT={r['raw_T']} split={r['split']} "
          f"text{r['text'].shape} l1{r['logits1'].shape} l2{r['logits2'].shape} "
          f"l1.mean={r['logits1'].mean():.5f} l2.mean={r['logits2'].mean():.5f}")

np.savez("/home/jinmang2/wsad/.reference/vadclip_oracle.npz",
         **{f"{v}__{k}": out[v][k] for v in VIDS for k in ("text", "logits1", "logits2")})
print("saved oracle -> .reference/vadclip_oracle.npz")
