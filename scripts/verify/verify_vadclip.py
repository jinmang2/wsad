"""Verify the faithful VadCLIP port matches the official CLIPVAD numerically.

Loads pretrained/vadclip/model_ucf.pth into the repo port and compares
binary_logits / alignment_logits / text_features against the official oracle
(.reference/vadclip_oracle.npz, produced by scripts/vadclip_oracle.py) on the
same inputs.

Run: uv run python scripts/verify_vadclip.py
"""

import sys

import numpy as np
import torch

import src.models  # noqa: F401
from src.models.vadclip.configuration_vadclip import VadCLIPConfig
from src.models.vadclip.modeling_vadclip import (
    VadCLIPForVideoAnomalyDetection,
    convert_official_vadclip,
)

import os  # noqa: E402

ROOT = os.environ.get("WSAD_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.expanduser(os.environ.get("WSAD_DATA", "~/data/wsad"))
REF = f"{ROOT}/.reference/VadCLIP/src"
sys.path.insert(0, REF)
import utils.tools as tools  # noqa: E402  (official process_split / mask)

CKPT = f"{ROOT}/pretrained/vadclip/model_ucf.pth"
ORACLE = f"{ROOT}/.reference/vadclip_oracle.npz"
VIDS = ["Abuse028_x264", "RoadAccidents133_x264", "Normal_Videos_867_x264"]
MAXLEN = 256
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- build port + load official weights ---
model = VadCLIPForVideoAnomalyDetection(VadCLIPConfig())
ckpt = torch.load(CKPT, map_location="cpu")
converted = convert_official_vadclip(ckpt)
missing, unexpected = model.load_state_dict(converted, strict=False)
print(f"port load: missing={len(missing)} unexpected={len(unexpected)}")
if missing:
    print("  missing:", missing[:8])
if unexpected:
    print("  unexpected:", unexpected[:8])
model = model.to(device).eval()

oracle = np.load(ORACLE)


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
def run(vid):
    feat = np.load(f"{DATA}/clip/test/{vid}__0.npy").astype(np.float32)
    split, clip_length = tools.process_split(feat, MAXLEN)
    visual = torch.tensor(split)
    if int(clip_length) < MAXLEN:
        visual = visual.unsqueeze(0)
    visual = visual.to(device)
    lengths = lengths_for(int(clip_length))
    out = model(video=visual, lengths=lengths)
    return out


def cmp(name, ours, ref):
    ours = np.asarray(ours, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    d = np.abs(ours - ref)
    print(f"    {name:9} shape{ours.shape} max|Δ|={d.max():.2e} mean|Δ|={d.mean():.2e}")
    return d.max()


print("\n=== numerical comparison vs official oracle ===")
worst_l1 = worst_l2 = worst_tx = 0.0
for vid in VIDS:
    out = run(vid)
    print(f"\n{vid}:")
    worst_l1 = max(worst_l1, cmp("logits1", out.binary_logits.cpu().numpy(), oracle[f"{vid}__logits1"]))
    worst_l2 = max(worst_l2, cmp("logits2", out.alignment_logits.cpu().numpy(), oracle[f"{vid}__logits2"]))
    worst_tx = max(worst_tx, cmp("text", out.text_features.cpu().numpy(), oracle[f"{vid}__text"]))

print("\n=== verdict ===")
print(f"binary_logits (fp32, text-independent) worst max|Δ| = {worst_l1:.2e}  (expect <1e-3)")
print(f"text_features (official runs CLIP fp16) worst max|Δ| = {worst_tx:.2e}")
print(f"alignment_logits                        worst max|Δ| = {worst_l2:.2e}")
ok = worst_l1 < 1e-3
print("BINARY BRANCH MATCHES OFFICIAL:", ok)
