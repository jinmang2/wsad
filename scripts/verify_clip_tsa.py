"""Verify the CLIP-TSA port matches the official Model architecturally + numerically.

No official checkpoint is published (the repo ships datasets only), so equivalence
is shown by transferring identical weights from the official ``Model``
(.reference/CLIP-TSA) into the repo port and comparing forward outputs under a
shared RNG seed (PerturbedTopK draws Gaussian noise; same seed -> same noise).

Run from .reference/CLIP-TSA as cwd:
  cd .reference/CLIP-TSA && PYTHONPATH=.:/home/jinmang2/wsad \
    conda run -n balaenoptera python /home/jinmang2/wsad/scripts/verify_clip_tsa.py
"""

import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = "/home/jinmang2/wsad"
sys.path.insert(0, f"{ROOT}/.reference/CLIP-TSA")
sys.path.insert(0, ROOT)

import model as off  # noqa: E402  (sets default tensor type to cuda — keep it; the
# official forward relies on bare torch.zeros(...) landing on cuda)

from src.models.clip_tsa.configuration_clip_tsa import CLIPTSAConfig  # noqa: E402
from src.models.clip_tsa.modeling_clip_tsa import (  # noqa: E402
    CLIPTSAForVideoAnomalyDetection,
)

device = "cuda"
F, T = 512, 32
args = SimpleNamespace(visual="vit", gpu="0")

# official: k=0.95 (UCF), num_samples=100, apply_HA=True
off_model = off.Model(F, batch_size=1, k=0.95, num_samples=100, apply_HA=True, args=args)
off_model = off_model.to(device).eval()

port = CLIPTSAForVideoAnomalyDetection(
    CLIPTSAConfig(feature_size=F, num_segments=T, topk_ratio=0.95, num_samples=100)
).to(device).eval()

# --- transfer official weights -> port (Aggregate->aggregate; drop unused mlp.*) ---
osd = off_model.state_dict()
psd = port.state_dict()
mapped = {}
for k, v in osd.items():
    if k.startswith("mlp."):
        continue  # CLIP path (f==512) never uses the I3D reduction MLP
    pk = k.replace("Aggregate.", "aggregate.")
    if pk in psd and psd[pk].shape == v.shape:
        mapped[pk] = v
miss = [k for k in psd if k not in mapped]
print(f"transferred {len(mapped)}/{len(psd)} port params; unmapped: {miss[:6]}")
port.load_state_dict(mapped, strict=False)

# --- seeded forward comparison ---
x = torch.randn(1, 1, T, F, device=device)

torch.manual_seed(0)
o = off_model(x)  # tuple; idx6=scores, idx9=feat_magnitudes
off_scores, off_mag = o[6], o[9]

torch.manual_seed(0)
p = port(video=x)
port_scores, port_mag = p.scores, None

ds = (off_scores - port_scores).abs()
print(f"\nscores      shape {tuple(port_scores.shape)}  max|Δ|={ds.max():.2e}  mean|Δ|={ds.mean():.2e}")
# magnitudes: official feat_magnitudes (bs,T) vs port (recompute not exposed) — compare scores primarily
print("scores match (official vs port):", bool(ds.max() < 1e-4))
print(f"  example off[:5]={off_scores.flatten()[:5].tolist()}")
print(f"  example port[:5]={port_scores.flatten()[:5].tolist()}")
