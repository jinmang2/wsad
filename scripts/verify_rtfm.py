"""Verify the RTFM port matches the official Model (tianyu0207/RTFM).

No official ckpt is shipped, so equivalence is shown by transferring the official
``Model`` weights into the repo port (official ``Aggregate.*`` -> port ``mtn.*``)
and comparing eval-mode forward (dropout off -> deterministic).

Run from .reference/RTFM as cwd:
  cd .reference/RTFM && PYTHONPATH=.:<repo> \
    conda run -n balaenoptera python <repo>/scripts/verify_rtfm.py
"""

import os
import sys

import torch

ROOT = os.environ.get("WSAD_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/.reference/RTFM")
sys.path.insert(0, ROOT)

from model import Model as OffModel  # noqa: E402

from src.models.rtfm.configuration_rtfm import RTFMConfig  # noqa: E402
from src.models.rtfm.modeling_rtfm import RTFMForVideoAnomalyDetection  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
F, T, N = 2048, 32, 10

off = OffModel(n_features=F, batch_size=1).to(device).eval()
port = RTFMForVideoAnomalyDetection(RTFMConfig(feature_size=F)).to(device).eval()

# transfer official -> port: rename Aggregate.* -> mtn.*
osd = off.state_dict()
psd = port.state_dict()
mapped, unmapped = {}, []
for k, v in osd.items():
    pk = k.replace("Aggregate.", "mtn.")
    if pk in psd and psd[pk].shape == v.shape:
        mapped[pk] = v
    else:
        unmapped.append(k)
miss = [k for k in psd if k not in mapped]
print(f"transferred {len(mapped)}/{len(psd)} port params; official-unmapped={unmapped[:4]}; port-missing={miss[:4]}")
port.load_state_dict(mapped, strict=False)

x = torch.randn(1, N, T, F, device=device)
with torch.no_grad():
    off_scores = off(x)[6]            # index 6 = per-frame scores (bs,T,1)
    port_scores = port(video=x).scores

d = (off_scores - port_scores).abs()
print(f"\nscores shape: official {tuple(off_scores.shape)} port {tuple(port_scores.shape)}")
print(f"max|Δ|={d.max():.2e}  mean|Δ|={d.mean():.2e}")
print("RTFM PORT MATCHES OFFICIAL:", bool(d.max() < 1e-4))
print("  off [:5]:", off_scores.flatten()[:5].tolist())
print("  port[:5]:", port_scores.flatten()[:5].tolist())
