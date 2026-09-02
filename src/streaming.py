"""Online, bounded-memory scoring for the temporal WSVAD heads (Spec 2, step 4).

The offline path (`src.eval_matrix.score_video`) hands the model a whole video at once.
That is fine for a 290-video benchmark and impossible for live surveillance, where T grows
without bound and a score is wanted as each frame arrives.

The heads built on `src.modules.translayer` (UR-DMU, BN-WVAD) turn out to be exactly
streamable, because in eval every component is either per-timestep or has a **finite
temporal receptive field**:

  - `Temporal` — `Conv1d(kernel_size=3, padding=1)`, i.e. ±1 snippet;
  - `selfatt` — the translayer stack: with `WSAD_ATTN=window` each layer reaches ±W, and
    with `WSAD_ATTN=causal` it reaches only backwards, so `depth` layers reach `depth·W`;
  - `Memory_Unit`, `encoder_mu`, `cls_head` — pointwise in time (the memory read attends
    over a learnable bank, never over other frames).

So a chunk of scores can be reproduced **bit-for-bit** from a bounded window of input, and
this scorer does exactly that rather than approximating. `receptive_field` derives the
window from the model instead of hard-coding it, so a config change cannot silently make
the streaming output diverge from the offline one.

One caveat worth stating plainly: under `WSAD_ATTN=causal` the attention is past-only, but
the `Conv1d` still has `padding=1`, so a frame's score depends on **one** future snippet.
Streaming latency is therefore one snippet (16 frames), not zero. Making it truly zero
would require a causal convolution, which changes the weights' meaning and would break
checkpoint compatibility — so it is reported, not hidden.
"""

from typing import Iterable, Iterator, Optional, Tuple

import torch
from torch import nn

from src.modules.translayer import Transformer


def receptive_field(model: nn.Module, window: Optional[int] = None) -> Tuple[int, int]:
    """``(left, right)`` snippet context a score at time ``i`` depends on.

    Derived by walking the module tree: every translayer stack contributes
    ``depth · window`` (backwards only when the attention backend is causal) and every
    temporal ``Conv1d`` contributes its symmetric padding.
    """
    from src.modules import translayer

    window = translayer._ATTN_WINDOW if window is None else window
    causal = translayer._ATTN_IMPL == "causal"
    banded = translayer._ATTN_IMPL in ("window", "causal")

    left = right = 0
    for module in model.modules():
        if isinstance(module, Transformer):
            if not banded:
                raise ValueError(
                    "streaming needs a bounded attention span — set WSAD_ATTN=causal "
                    "(or window); the default 'eager'/'mem' path attends to the whole clip."
                )
            reach = len(module.layers) * window
            left += reach
            right += 0 if causal else reach
        elif isinstance(module, nn.Conv1d):
            pad = module.padding[0] if isinstance(module.padding, tuple) else int(module.padding)
            left += pad
            right += pad
    return left, right


class StreamingScorer:
    """Score a video in chunks, reading a bounded context around each chunk.

    Peak memory depends on ``left + chunk + right``, not on the video length, so a stream
    of any duration runs in constant memory. The output is identical to scoring the whole
    sequence at once with the same attention backend — boundary behaviour matches too,
    because the first and last chunks are given the true sequence ends and therefore see
    the same convolution zero-padding the offline path does.
    """

    def __init__(
        self,
        model: nn.Module,
        chunk: int = 256,
        device: str = "cpu",
        left: Optional[int] = None,
        right: Optional[int] = None,
    ):
        self.model = model.eval()
        self.chunk = chunk
        self.device = device
        derived_left, derived_right = receptive_field(model)
        self.left = derived_left if left is None else left
        self.right = derived_right if right is None else right

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(ncrops, T, D)`` -> ``(T,)``, averaging crops the way the official eval does."""
        per_crop = [
            self.model(video=x[None, c : c + 1]).scores.squeeze(0).squeeze(-1)
            for c in range(x.shape[0])
        ]
        return torch.stack(per_crop).mean(0)

    @torch.no_grad()
    def score(self, feature: torch.Tensor) -> torch.Tensor:
        """Per-snippet scores for a full ``(ncrops, T, D)`` (or ``(T, D)``) feature."""
        feat = feature if feature.dim() == 3 else feature[None]
        feat = feat.to(self.device)
        total = feat.shape[1]

        out = []
        for start in range(0, total, self.chunk):
            end = min(start + self.chunk, total)
            lo, hi = max(0, start - self.left), min(total, end + self.right)
            scores = self._forward(feat[:, lo:hi])
            out.append(scores[start - lo : end - lo])
        return torch.cat(out)

    @torch.no_grad()
    def stream(self, blocks: Iterable[torch.Tensor]) -> Iterator[torch.Tensor]:
        """Consume ``(ncrops, C, D)`` blocks as they arrive; yield each block's scores.

        Keeps only the trailing ``left`` snippets between calls. A block's scores are
        emitted once its ``right`` lookahead is available, so with ``right > 0`` the
        yielded scores lag the input by that many snippets; the final block is flushed
        when the stream ends.
        """
        history: Optional[torch.Tensor] = None
        pending: Optional[torch.Tensor] = None  # block whose lookahead has not arrived

        def emit(block: torch.Tensor, tail: Optional[torch.Tensor], last: bool):
            nonlocal history
            context = history if history is not None else block[:, :0]
            window = torch.cat([context, block] + ([tail] if tail is not None else []), dim=1)
            scores = self._forward(window)
            lo = context.shape[1]
            out = scores[lo : lo + block.shape[1]]
            keep = torch.cat([context, block], dim=1)
            history = keep[:, -self.left :] if self.left else keep[:, :0]
            return out

        for block in blocks:
            block = (block if block.dim() == 3 else block[None]).to(self.device)
            if pending is not None:
                yield emit(pending, block[:, : self.right] if self.right else None, False)
            pending = block
        if pending is not None:
            yield emit(pending, None, True)
