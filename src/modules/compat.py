"""Pretrained-weight compatibility helpers.

``verify_numerical_equivalence`` is the gate for "our re-implementation matches
the official one": run both models on a shared input in an exact configuration
(eager attention, fp32, eval, TF32 off) and assert the outputs agree to a tight
tolerance. See ``docs/WEIGHTS_AND_OPTIMIZATION.md``.
"""

from contextlib import contextmanager
from typing import Callable, Dict, Tuple

import torch
from torch import nn


@contextmanager
def exact_math():
    """Disable TF32 / cuDNN nondeterminism for a bit-exact comparison."""
    prev_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_tf32_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32_matmul
        torch.backends.cudnn.allow_tf32 = prev_tf32_cudnn


@torch.no_grad()
def verify_numerical_equivalence(
    ours: nn.Module,
    reference: Callable[[torch.Tensor], torch.Tensor],
    sample_input: torch.Tensor,
    our_output_key: str = "scores",
    atol: float = 1e-5,
) -> Tuple[bool, float]:
    """Compare our model's output against a reference on the same input.

    Args:
        ours: our model; called as ``ours(video=sample_input)`` and the
            ``our_output_key`` field of the ModelOutput is compared.
        reference: a callable returning the reference tensor for ``sample_input``
            (e.g. the official model's forward).
        sample_input: shared input tensor.
        atol: pass threshold on the max absolute difference.

    Returns:
        (passed, max_abs_diff).
    """
    ours.eval()
    with exact_math():
        ours_out = getattr(ours(video=sample_input), our_output_key)
        ref_out = reference(sample_input)
    max_diff = (ours_out - ref_out).abs().max().item()
    return max_diff < atol, max_diff


def remap_state_dict(
    state_dict: Dict[str, torch.Tensor],
    rename: Callable[[str], str],
) -> Dict[str, torch.Tensor]:
    """Apply a key-rename function to every entry (official -> our naming).

    Keys for which ``rename`` returns ``None`` are dropped (e.g. official-only
    buffers). Use with ``load_state_dict(..., strict=True)`` afterwards so any
    leftover mismatch is loud.
    """
    out: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        new_key = rename(key)
        if new_key is not None:
            out[new_key] = tensor
    return out
