"""Verify the BN-WVAD port matches the official WSAD using the official checkpoint.

Loads .reference/BN-WVAD/ckpts/xd_best.pkl into BOTH the official ``WSAD`` and the
repo port, compares eval-path scores (distance_sum * normal_scores) on identical
random 1024-d input (eval = BN running stats, deterministic).

Run from .reference/BN-WVAD as cwd:
  cd .reference/BN-WVAD && PYTHONPATH=.:<repo> \
    uv run python <repo>/scripts/verify_bn_wvad.py
"""

import sys
import types
from types import SimpleNamespace

import torch

for name in ("ipdb", "visdom", "wandb"):
    if name not in sys.modules:
        try:
            __import__(name)
        except Exception:
            sys.modules[name] = types.ModuleType(name)
            sys.modules[name].set_trace = lambda *a, **k: None
            sys.modules[name].Visdom = object

import os

ROOT = os.environ.get("WSAD_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/.reference/BN-WVAD")
sys.path.insert(0, ROOT)

from models.model import WSAD  # noqa: E402  (official)

from src.models.bn_wvad.configuration_bn_wvad import BNWVADConfig  # noqa: E402
from src.models.bn_wvad.modeling_bn_wvad import (  # noqa: E402
    BNWVADForVideoAnomalyDetection,
)

CKPT = f"{ROOT}/.reference/BN-WVAD/ckpts/xd_best.pkl"
device = "cuda"
D, T, N = 1024, 32, 10
args = SimpleNamespace(ratio_sample=0.2, ratio_batch=0.4, ratios=[16, 32], kernel_sizes=[1, 1, 1])

ckpt = torch.load(CKPT, map_location="cpu")
ckpt = ckpt.get("model", ckpt) if isinstance(ckpt, dict) and "model" in ckpt else ckpt

off = WSAD(input_size=D, flag="Test", args=args)
off.load_state_dict(ckpt)
off = off.to(device).eval()
off.flag = "Test"

port = BNWVADForVideoAnomalyDetection(BNWVADConfig(feature_size=D))
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
    off_s = off(x)                      # (1, T)
    port_s = port(video=x).scores.squeeze(-1)  # (1, T)

d = (off_s - port_s).abs()
print(f"\nscores shape: official {tuple(off_s.shape)} port {tuple(port_s.shape)}")
print(f"max|Δ|={d.max():.2e}  mean|Δ|={d.mean():.2e}")
print("BN-WVAD PORT MATCHES OFFICIAL:", bool(d.max() < 1e-4))
print("  off [:5]:", off_s.flatten()[:5].tolist())
print("  port[:5]:", port_s.flatten()[:5].tolist())
