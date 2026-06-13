"""Verify the S3R port matches the official S3R (louisYen/S3R) by weight transfer.

No official checkpoint is shipped, so equivalence is shown by transferring the
official ``S3R`` weights into the repo port (names match 1:1) and comparing the
``video_scores`` on identical random video + macro-dictionary inputs (eval, dropout
off -> deterministic).

Run from .reference/S3R as cwd:
  cd .reference/S3R && PYTHONPATH=.:/home/jinmang2/wsad \
    conda run -n balaenoptera python /home/jinmang2/wsad/scripts/verify_s3r.py
"""

import sys

import torch

ROOT = "/home/jinmang2/wsad"
sys.path.insert(0, f"{ROOT}/.reference/S3R")
sys.path.insert(0, ROOT)

from anomaly.models.detectors.detector import S3R as OffS3R  # noqa: E402

from src.models.s3r.configuration_s3r import S3RConfig  # noqa: E402
from src.models.s3r.modeling_s3r import S3RForVideoAnomalyDetection  # noqa: E402

device = "cuda"
C, T, N, S = 2048, 32, 10, 64

off = OffS3R(dim=C, batch_size=1, quantize_size=T, dropout=0.7, modality="univ-task")
off = off.to(device).eval()

port = S3RForVideoAnomalyDetection(S3RConfig(feature_size=C, dict_size=S)).to(device).eval()

# transfer official -> port (names match 1:1; our extra `dictionary` has no source)
osd, psd = off.state_dict(), port.state_dict()
mapped = {k: v for k, v in osd.items() if k in psd and psd[k].shape == v.shape}
off_only = [k for k in osd if k not in mapped]
port_missing = [k for k in psd if k not in mapped]
print(f"transferred {len(mapped)}/{len(psd)} port params")
print(f"  official-only (unused, e.g. projections): {off_only[:6]}")
print(f"  port-only (runner fallback): {port_missing[:6]}")
port.load_state_dict(mapped, strict=False)

torch.manual_seed(0)
video = torch.randn(1, N, T, C, device=device)
macro = torch.randn(1, S, C, device=device)

with torch.no_grad():
    off_out = off(video, macro)["video_scores"]      # (1, T, 1)
    port_out = port(video=video, macro=macro).scores   # (1, T, 1)

d = (off_out - port_out).abs()
print(f"\nvideo_scores shape: official {tuple(off_out.shape)} port {tuple(port_out.shape)}")
print(f"max|Δ|={d.max():.2e}  mean|Δ|={d.mean():.2e}")
print("S3R PORT MATCHES OFFICIAL:", bool(d.max() < 1e-4))
print("  off [:5]:", off_out.flatten()[:5].tolist())
print("  port[:5]:", port_out.flatten()[:5].tolist())
