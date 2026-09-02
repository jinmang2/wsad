"""Verify the UR-DMU port matches the official WSAD using the official checkpoint.

Loads .reference/UR-DMU/models/ucf_trans_2022.pkl into BOTH the official ``WSAD``
and the repo port, then compares the eval-path frame scores on identical random
1024-d input (the eval path is deterministic — no variational sampling).

Run from .reference/UR-DMU as cwd:
  cd .reference/UR-DMU && PYTHONPATH=.:<repo> \
    uv run python <repo>/scripts/verify_ur_dmu.py
"""

import sys
import types

import torch

# stub optional deps the official repo imports (ipdb in translayer, visdom in utils)
for name in ("ipdb", "visdom"):
    if name not in sys.modules:
        try:
            __import__(name)
        except Exception:
            sys.modules[name] = types.ModuleType(name)
            sys.modules[name].set_trace = lambda *a, **k: None
            sys.modules[name].Visdom = object

import os

ROOT = os.environ.get("WSAD_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = f"{ROOT}/.reference/UR-DMU"
sys.path.insert(0, REF)
sys.path.insert(0, ROOT)

from model import WSAD  # noqa: E402  (official)

from src.models.ur_dmu.configuration_ur_dmu import URDMUConfig  # noqa: E402
from src.models.ur_dmu.modeling_ur_dmu import (  # noqa: E402
    URDMUForVideoAnomalyDetection,
)

CKPT = f"{REF}/models/ucf_trans_2022.pkl"
device = "cuda"
D, T, N = 1024, 32, 10

ckpt = torch.load(CKPT, map_location="cpu")

off = WSAD(input_size=D, flag="Test", a_nums=60, n_nums=60)
off.load_state_dict(ckpt)
off = off.to(device).eval()
off.flag = "Test"

port = URDMUForVideoAnomalyDetection(URDMUConfig(feature_size=D))
miss, unexp = port.load_state_dict(ckpt, strict=False)
print(f"port load: missing={len(miss)} unexpected={len(unexp)}")
if miss:
    print("  missing:", miss[:8])
if unexp:
    print("  unexpected:", unexp[:8])
port = port.to(device).eval()

torch.manual_seed(0)
x = torch.randn(1, N, T, D, device=device)

with torch.no_grad():
    off_frame = off(x)["frame"]            # (1, T)
    port_out = port(video=x).scores         # (1, T, 1)

port_frame = port_out.squeeze(-1)
d = (off_frame - port_frame).abs()
print(f"\nframe scores shape: official {tuple(off_frame.shape)} port {tuple(port_frame.shape)}")
print(f"max|Δ|={d.max():.2e}  mean|Δ|={d.mean():.2e}")
print("UR-DMU PORT MATCHES OFFICIAL:", bool(d.max() < 1e-4))
print("  off [:5]:", off_frame.flatten()[:5].tolist())
print("  port[:5]:", port_frame.flatten()[:5].tolist())
