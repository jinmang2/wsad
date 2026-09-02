"""PEL4VAD — Prompt-Enhanced Learning for weakly-supervised VAD (TIP 2024).

Faithful port of `yujiangpu20/PEL4VAD` (MIT), which reports **0.8676** on UCF-Crime.
Two things make it worth a slot in the matrix:

  - **It runs on features we already have.** The official config is `feat_dim = 1024`,
    i.e. the DeepMIL I3D that UR-DMU / BN-WVAD use (`i3d_1024_seg200`). No new extraction.
  - **Its temporal aggregation (TCA) is a different mechanism from every head here.**
    A single attention map is used twice — once globally, once masked to a local window of
    `win_size` — and the two are mixed by a *learned* scalar `sigmoid(alpha)`, on top of a
    *learnable* distance decay `exp(-|w·d² - b|)`. Compare UR-DMU's dual branch, where the
    decay is fixed and the two branches are concatenated rather than interpolated.

Slot mapping (WSAD_INTEGRATION_PLAN.md §4): slot 2 = `XEncoder`/`TCA`; slot 4 = the causal
`t_step` classifier convolution; slot 5 = top-k MIL BCE + the prompt-alignment KL term.

Deviations from the official code, all deliberate:
  - **Device-safe.** The original hard-codes `.cuda()` in `DistanceAdj` and the window mask
    and builds the distance matrix through `scipy.spatial.distance.pdist`; both are done in
    torch on the input's device here, so the model runs on CPU (and in the tests).
  - **`text_proj`.** The official prompt loss works only because `feature_size // 2` happens
    to equal the CLIP text width (1024 // 2 == 512). On any other feature dim that matmul
    is a shape error, so a projection is inserted when the dims do not already agree —
    which is what lets this head run on the 2048-d and 768-d variants too.
  - **The prompt term is optional.** It needs per-video class ids; without them the head
    trains on the MIL term alone rather than failing.

**Known fidelity gap — crop handling at training time.** The official `list/ucf/train.list`
has 16100 entries for 1610 videos: each of the 10 crops is a *separate training sample*, so
the official run sees 10x the samples and gets crop-level augmentation for free (its test
loader then averages the 10 crops per video). This repo's loader hands every head
`(B, ncrops, T, D)` and this port averages the crops, matching what `gs_moe` and the other
single-crop heads here do — consistent across the matrix, but **not** the official recipe.
If a full run lands short of the paper's 0.8676, this is the first thing to suspect, and
sampling one random crop per video during training is the cheap experiment.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from src.modules.amp import safe_bce
from src.registry import MODELS

from .configuration_pel4vad import PEL4VADConfig

# Vendored from PEL4VAD (MIT): CLIP text embeddings of the 14 UCF-Crime class prompts.
# Its `prompt_extract/list/ucf_label.list` order — normal event, abuse, arrest, arson,
# assault, burglary, explosion, fighting, road accidents, robbery, shooting, shoplifting,
# stealing, vandalism — is index-for-index identical to `src.data.labels.UCF_CLASSES`, so
# a class id indexes the prompt table directly. Verified, not assumed.
_PROMPT_FILE = "ucf_prompt.npy"


@dataclass
class PEL4VADVideoAnomalyDetectionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    scores: Optional[torch.FloatTensor] = None


class DistanceAdj(nn.Module):
    """Learnable temporal distance prior ``exp(-|w·d² - b|)`` added to the attention map.

    ``w`` and ``b`` are parameters initialised from the config, so unlike UR-DMU's fixed
    ``exp(-|i-j|/e)`` the model can widen or narrow its own locality during training.
    """

    def __init__(self, gamma: float, bias: float):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([gamma]))
        self.b = nn.Parameter(torch.tensor([bias]))

    def forward(self, batch_size: int, seq_len: int, device) -> torch.Tensor:
        idx = torch.arange(seq_len, device=device, dtype=torch.float32)
        dist = (idx.view(-1, 1) - idx.view(1, -1)).abs()  # cityblock, as in the original
        adj = torch.exp(-torch.abs(self.w * dist**2 - self.b))
        return adj.unsqueeze(0).expand(batch_size, -1, -1)


class TCA(nn.Module):
    """Temporal Context Aggregation: one attention map read globally and locally.

    The global map keeps the distance prior; the local map is the *same* map masked to the
    window. Mixing them with a learned ``sigmoid(alpha)`` lets the head slide between
    long-range and local aggregation instead of committing to either.
    """

    def __init__(self, d_model: int, dim_k: int, dim_v: int, n_heads: int, norm: bool):
        super().__init__()
        self.dim_k, self.dim_v, self.n_heads, self.norm = dim_k, dim_v, n_heads, norm
        self.q = nn.Linear(d_model, dim_k)
        self.k = nn.Linear(d_model, dim_k)
        self.v = nn.Linear(d_model, dim_v)
        self.o = nn.Linear(dim_v, d_model)
        self.norm_fact = 1 / (dim_k**0.5)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, mask: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q(x).view(-1, b, t, self.dim_k // self.n_heads)
        k = self.k(x).view(-1, b, t, self.dim_k // self.n_heads)
        v = self.v(x).view(-1, b, t, self.dim_v // self.n_heads)

        g_map = torch.matmul(q, k.permute(0, 1, 3, 2)) * self.norm_fact + adj
        l_map = g_map.masked_fill(mask == 0, -1e9)

        glb = torch.matmul(g_map.softmax(-1), v).view(b, t, -1)
        lcl = torch.matmul(l_map.softmax(-1), v).view(b, t, -1)

        alpha = torch.sigmoid(self.alpha)
        out = alpha * glb + (1 - alpha) * lcl
        if self.norm:  # signed power norm then L2, as in the official code
            out = torch.sqrt(F.relu(out)) - torch.sqrt(F.relu(-out))
            out = F.normalize(out)
        return self.o(out).view(-1, t, x.shape[2])


class XEncoder(nn.Module):
    def __init__(self, cfg: PEL4VADConfig):
        super().__init__()
        d = cfg.feature_size
        self.n_heads = cfg.num_heads
        self.win_size = cfg.win_size
        self.self_attn = TCA(d, cfg.hidden_size, cfg.hidden_size, cfg.num_heads, cfg.norm)
        self.linear1 = nn.Conv1d(d, d // 2, kernel_size=1)
        self.linear2 = nn.Conv1d(d // 2, cfg.out_dim, kernel_size=1)
        self.dropout1 = nn.Dropout(cfg.dropout_rate)
        self.dropout2 = nn.Dropout(cfg.dropout_rate)
        self.norm = nn.LayerNorm(d)
        self.loc_adj = DistanceAdj(cfg.gamma, cfg.bias)

    def _window_mask(self, t: int, device) -> torch.Tensor:
        """The official mask: index ``j - w//2 + k`` **clamped** to the sequence.

        Clamping (rather than dropping) means rows near the ends attend to the boundary
        frame repeatedly, so their effective window is narrower. Reproduced as-is.
        """
        m = torch.zeros(t, t, device=device)
        w = self.win_size
        cols = torch.arange(w, device=device) - w // 2
        for j in range(t):
            m[j, (cols + j).clamp(0, t - 1)] = 1.0
        return m

    def forward(self, x: torch.Tensor):
        b, t, _ = x.shape
        adj = self.loc_adj(b, t, x.device)
        mask = self._window_mask(t, x.device)

        x = x + self.self_attn(x, mask, adj)
        x = self.norm(x).permute(0, 2, 1)
        x_v = self.dropout1(F.gelu(self.linear1(x)))  # (B, d/2, T) — the prompt-space feature
        x_e = self.dropout2(F.gelu(self.linear2(x_v)))  # (B, out_dim, T)
        return x_e, x_v


class PEL4VADPreTrainedModel(PreTrainedModel):
    config_class = PEL4VADConfig
    base_model_prefix = "pel4vad"

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)) and getattr(module, "weight", None) is not None:
            nn.init.xavier_uniform_(module.weight)


@MODELS.register("pel4vad")
class PEL4VADForVideoAnomalyDetection(PEL4VADPreTrainedModel):
    def __init__(self, config: PEL4VADConfig):
        super().__init__(config)
        self.encoder = XEncoder(config)
        self.classifier = nn.Conv1d(config.out_dim, 1, config.t_step, padding=0)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / config.temp))

        prompt_width = config.feature_size // 2
        self.text_proj = (
            nn.Identity() if prompt_width == config.prompt_dim
            else nn.Linear(prompt_width, config.prompt_dim)
        )
        self.register_buffer(
            "class_prompts", torch.zeros(config.num_classes, config.prompt_dim), persistent=True
        )
        self.post_init()
        self._load_class_prompts()

    def _load_class_prompts(self) -> None:
        """Load the vendored CLIP prompt table, if it is the expected shape."""
        import os

        path = os.path.join(os.path.dirname(__file__), _PROMPT_FILE)
        if not os.path.exists(path):
            return
        table = torch.from_numpy(np.load(path)).float()
        if table.shape == self.class_prompts.shape:
            with torch.no_grad():
                self.class_prompts.copy_(table)

    def forward(
        self,
        video: torch.FloatTensor,
        abnormal_labels: Optional[torch.FloatTensor] = None,
        normal_labels: Optional[torch.FloatTensor] = None,
        class_labels: Optional[torch.FloatTensor] = None,
    ) -> PEL4VADVideoAnomalyDetectionOutput:
        x = video[..., : self.config.feature_size]
        if x.dim() == 4:  # (B, ncrops, T, D) -> mean over crops (single-crop head)
            x = x.mean(dim=1)

        x_e, x_v = self.encoder(x)
        # left-pad only: a score at t sees [t - t_step + 1, t], never the future
        logits = torch.sigmoid(self.classifier(F.pad(x_e, (self.config.t_step - 1, 0))))
        scores = logits.permute(0, 2, 1)  # (B, T, 1)

        loss = None
        if abnormal_labels is not None and normal_labels is not None:
            loss = self._compute_loss(scores, x_v, class_labels)
        return PEL4VADVideoAnomalyDetectionOutput(loss=loss, scores=scores)

    @staticmethod
    def _topk_mil(frame: torch.Tensor, is_normal: torch.Tensor) -> torch.Tensor:
        """Official ``CLAS2``: k=1 for normal videos, k = T//16 + 1 for abnormal ones."""
        t = frame.shape[1]
        k_abn = max(t // 16 + 1, 1)
        top_normal = frame.topk(1, dim=1).values.mean(1)
        top_abnormal = frame.topk(min(k_abn, t), dim=1).values.mean(1)
        return torch.where(is_normal, top_normal, top_abnormal)

    def _compute_loss(self, scores, x_v, class_labels) -> torch.Tensor:
        b = scores.size(0)
        half = b // 2
        device = scores.device
        frame = scores.squeeze(-1)

        y = torch.cat([torch.zeros(half, device=device), torch.ones(b - half, device=device)])
        vid = self._topk_mil(frame, y == 0).clamp(1e-6, 1 - 1e-6)
        loss = safe_bce(vid, y)

        if class_labels is not None and self.config.lamda > 0:
            loss = loss + self.config.lamda * self._prompt_loss(scores, x_v, class_labels)
        return loss

    def _prompt_loss(self, scores, x_v, class_labels, scale: float = 10.0) -> torch.Tensor:
        """Official PEL term: align class-attentive video features with class prompts.

        Anomaly-weighted and normal-weighted pooling give a foreground and a background
        vector per video; each is matched to its prompt (the video's class, or `Normal`),
        and a KL divergence pulls same-class pairs together across the batch.
        """
        device = scores.device
        cls = class_labels.long().to(device)

        feat = x_v.permute(0, 2, 1)  # (B, T, d/2)
        feat = self.text_proj(feat)

        abn_w = F.normalize(((scale * scores).exp() - 1), p=1, dim=1)
        nor_w = F.normalize(((scale * (1.0 - scores)).exp() - 1), p=1, dim=1)
        fg = torch.bmm(abn_w.permute(0, 2, 1), feat).squeeze(1)  # (B, d)
        bg = torch.bmm(nor_w.permute(0, 2, 1), feat).squeeze(1)

        prompts = self.class_prompts.to(device)
        is_abn = cls != 0
        video_feat = torch.cat([fg, bg[is_abn]], dim=0)
        video_cls = torch.cat([cls, torch.zeros(int(is_abn.sum()), device=device, dtype=cls.dtype)])
        token_feat = prompts[video_cls]

        v = F.normalize(video_feat, dim=-1)
        t = F.normalize(token_feat, dim=-1)
        v2t = self.logit_scale.exp() * v @ t.t()

        same = (video_cls.view(-1, 1) == video_cls.view(1, -1)).float()
        target = F.softmax(same * 10, dim=1)
        pred = F.log_softmax(v2t, dim=1)
        if not torch.isfinite(pred).all():
            return torch.zeros((), device=device)
        return F.kl_div(pred, target, reduction="batchmean")
