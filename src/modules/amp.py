"""Autocast-safe loss helpers for mixed-precision (fp16/bf16) training.

``F.binary_cross_entropy`` / ``nn.BCELoss`` operate on probabilities and are
explicitly **unsafe to autocast** (PyTorch raises). The WSVAD heads compute MIL/
memory losses as BCE on sigmoid outputs (faithful to the official code), so AMP
training would crash. ``safe_bce`` computes the BCE in fp32 with autocast disabled.

In fp32 (non-AMP) training/eval this is a *no-op*: autocast is already off and the
inputs are already fp32, so ``.float()`` returns the same tensor and the result is
numerically identical to ``F.binary_cross_entropy`` — the verified-checkpoint paths
are unchanged. It only fixes the fp16/bf16 path.
"""

import torch
import torch.nn.functional as F


def safe_bce(inp: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    with torch.autocast(device_type=inp.device.type, enabled=False):
        return F.binary_cross_entropy(inp.float(), target.float(), **kwargs)
