"""Shims for `trust_remote_code` models written against transformers v4.

Some published checkpoints ship their own modeling code (Cosmos-Embed1 vendors a
BLIP-2-style Q-Former and EVA ViT) that was written for transformers 4.x. Downgrading
is not an option — the pinned 5.10.2 is what every reproduced number in
`docs/RESULTS_TABLE.md` was produced with — so restore what v5 expects instead.

Each shim is installed only if absent, so it becomes a no-op once upstream catches up.
"""

from types import MappingProxyType

import torch

# Read-only on purpose: this is a *fallback* for models that never ran `post_init`, and a
# shared mutable dict would let one model's tied-weight bookkeeping leak into another's.
# Anything that tries to write to it should fail loudly rather than corrupt state.
_EMPTY_TIED_KEYS = MappingProxyType({})


def _find_pruneable_heads_and_indices(heads, n_heads: int, head_size: int, already_pruned_heads):
    """transformers v4 ``pytorch_utils.find_pruneable_heads_and_indices``.

    Only reachable through ``PreTrainedModel.prune_heads``, which feature extraction
    never calls — it exists purely so the vendored Q-Former module imports.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads, index


def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers: int):
    """transformers v4 ``ModuleUtilsMixin._convert_head_mask_to_5d``."""
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
    return head_mask.to(dtype=self.dtype)


def _get_head_mask(self, head_mask, num_hidden_layers: int, is_attention_chunked: bool = False):
    """transformers v4 ``ModuleUtilsMixin.get_head_mask``.

    Every BERT-lineage encoder calls this on each forward. v5 dropped it, so a vendored
    v4 encoder crashes at inference (not at load) without this.
    """
    if head_mask is not None:
        head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        if is_attention_chunked is True:
            head_mask = head_mask.unsqueeze(-1)
    else:
        head_mask = [None] * num_hidden_layers
    return head_mask


def install_v4_compat() -> None:
    """Make transformers v5 able to load v4-era ``trust_remote_code`` models."""
    from transformers import pytorch_utils
    from transformers import modeling_utils
    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
        pytorch_utils.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices

    # v5 sets `all_tied_weights_keys` in `post_init()` and then reads it during loading.
    # A v4-era `__init__` that never calls `post_init` leaves it undefined, and the load
    # dies with an unhelpful AttributeError deep inside `_finalize_model_loading`.
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = _EMPTY_TIED_KEYS

    mixin = modeling_utils.ModuleUtilsMixin
    if not hasattr(mixin, "get_head_mask"):
        mixin._convert_head_mask_to_5d = _convert_head_mask_to_5d
        mixin.get_head_mask = _get_head_mask

