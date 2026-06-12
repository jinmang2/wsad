"""TPWNG: Text Prompt with Normality Guidance (CVPR'24).

Slot mapping (WSAD_INTEGRATION_PLAN.md section 4):
  - slot 1 (features): CLIP ViT-B/16 frame features (512-d, text-aligned).
  - slot 2 (temporal encoder): **TCSAL** — a transformer with a learnable
    per-position soft-mask that adapts each query's attention span.
  - slot 3a (text branch): learnable class prompts (CoOp-style); class 0 == Normal.
  - slot 4 (scoring head): frame↔text similarity → anomaly score from the fused
    normal/anomaly similarity (PLG).
  - slot 5 (loss): ranking + distributional-inconsistency (DIL) + BCE to pseudo-
    labels + sparsity + smoothness.
  - slot 6 (self-training): pseudo-labels are generated from the model's own
    similarities each step (the first PL method here).

Honors the runner contract: ``forward(video, abnormal_labels, normal_labels) ->
ModelOutput`` with ``.loss`` and ``.scores`` (B, T, 1).

Official code unreleased — paper-based reimplementation (arXiv 2404.08531).
Faithful: CLIP-grounded CoOp text prompt (l=8, only the CLIP text projection is
fine-tuned, towers frozen), TCSAL soft-mask span (χ_z, R=256), Normality Visual
Prompt (NVP) refining the normal-event text from normal frames, PLG fusion
(α=0.2, θ=0.55), and L_rank + L_dil + L_cl + λ1·sparsity + λ2·smoothness
(λ1=0.1, λ2=0.01). ``use_clip_text=False`` falls back to a learnable text table
(offline); ``use_nvp=False`` disables NVP. NVP guides the training losses /
pseudo-labels; the eval ``scores`` come from the base text–frame similarity.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from src.loss import SparsityLoss, TemporalSmoothnessLoss
from src.modules import TransformerEncoderLayer
from src.registry import MODELS

from .configuration_tpwng import TPWNGConfig


@dataclass
class TPWNGVideoAnomalyDetectionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    scores: Optional[torch.FloatTensor] = None
    similarity: Optional[torch.FloatTensor] = None  # (B, T, num_class)


class TCSAL(nn.Module):
    """Temporal transformer with a learnable per-position soft-mask span.

    Each query position predicts a span ``z`` (in [0, R]); the additive attention
    bias is ``log(clamp((R + z - |i-j|)/R, 0, 1))`` — a soft local window of radius
    ~z with an R-wide ramp (paper's χ_z). Composes with the eager/SDPA kernels.
    """

    def __init__(self, dim, layers, heads, span_R, dropout, attn_impl):
        super().__init__()
        self.span_R = span_R
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    dim, heads=heads, dropout=dropout, attn_impl=attn_impl
                )
                for _ in range(layers)
            ]
        )
        self.span = nn.Linear(dim, 1)  # z = R * sigmoid(linear(x))

    def _soft_mask_bias(self, x):
        b, t, _ = x.shape
        z = self.span_R * torch.sigmoid(self.span(x)).squeeze(-1)  # (B, T) per query
        idx = torch.arange(t, device=x.device).float()
        dist = (idx[:, None] - idx[None, :]).abs()  # (T, T)
        # clamp((R + z - dist)/R, 0, 1) per query row -> (B, T, T)
        m = ((self.span_R + z[:, :, None] - dist[None]) / self.span_R).clamp(0, 1)
        return torch.log(m + 1e-6).unsqueeze(1)  # (B, 1, T, T) additive bias

    def forward(self, x):
        bias = self._soft_mask_bias(x)
        for layer in self.layers:
            x = layer(x, attn_bias=bias)
        return x


class NormalityVisualPrompt(nn.Module):
    """NVP: enhance the normal-event text embedding from normal video frames.

    Paper: ``Q_i = softmax(X_n · T_n) · X_n``; ``T̄_n = FFN(T_n ∪ Q_i) + T_n``.
    A text-guided aggregation of normal frames refines the normal-text prototype,
    which (via the distributional-inconsistency loss) sharpens anomaly separation.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )

    def forward(self, normal_text: torch.Tensor, normal_frames: torch.Tensor):
        # normal_text: (d,)  normal_frames: (N, d) pooled normal frames
        s = normal_frames @ normal_text  # (N,)
        w = torch.softmax(s, dim=0)
        q = (w.unsqueeze(-1) * normal_frames).sum(0)  # (d,) text-guided aggregate
        enhanced = self.ffn(torch.cat([normal_text, q], dim=-1)) + normal_text
        return enhanced  # (d,)


class TPWNGPreTrainedModel(PreTrainedModel):
    config_class = TPWNGConfig
    base_model_prefix = "tpwng"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    @property
    def dummy_inputs(self):
        return {"video": torch.randn(8, 1, 32, 512)}


@MODELS.register("tpwng")
class TPWNGForVideoAnomalyDetection(TPWNGPreTrainedModel):
    def __init__(self, config: TPWNGConfig):
        super().__init__(config)
        d = config.embed_dim
        self.proj = nn.Linear(config.feature_size, d)
        self.tcsal = TCSAL(
            d,
            config.tcsal_layers,
            config.tcsal_heads,
            config.span_R,
            dropout=0.0,
            attn_impl=config.attn_impl,
        )
        # text branch: faithful = CLIP-grounded CoOp prompt (l=n_ctx, prefix-only;
        # only the CLIP text projection fine-tunes); legacy = learnable table.
        self.use_clip_text = config.use_clip_text
        if config.use_clip_text:
            from src.models.vadclip.text_encoder import CLIPPromptTextEncoder

            self.text_encoder = CLIPPromptTextEncoder(
                model_name=config.clip_model_name,
                pretrained=config.clip_pretrained,
                embed_dim=d,
                prompt_prefix=config.n_ctx,
                prompt_postfix=0,
                unfreeze_projection=True,
            )
            self.text_features = None
        else:
            self.text_encoder = None
            self.text_features = nn.Parameter(torch.empty(config.num_class, d))
            nn.init.normal_(self.text_features, std=0.01)

        self.use_nvp = config.use_nvp
        self.nvp = NormalityVisualPrompt(d) if config.use_nvp else None

        self.sparse = SparsityLoss(config.lambda_sparse)
        self.smooth = TemporalSmoothnessLoss(config.lambda_smooth)
        self._force_split = False

    def _text(self) -> torch.Tensor:
        if self.text_encoder is not None:
            return self.text_encoder(self.config.class_names)  # (C, d)
        return self.text_features

    @property
    def force_split(self) -> bool:
        return self._force_split

    @force_split.setter
    def force_split(self, val: bool):
        self._force_split = val

    def _encode(self, video):
        f = self.config.feature_size
        x = video[..., :f].mean(dim=1)  # (B, T, f) mean over crops
        h = self.tcsal(self.proj(x))  # (B, T, d)
        return F.normalize(h, dim=-1)  # (B, T, d) unit frames

    def forward(
        self,
        video: torch.FloatTensor,
        abnormal_labels: Optional[torch.FloatTensor] = None,
        normal_labels: Optional[torch.FloatTensor] = None,
    ) -> TPWNGVideoAnomalyDetectionOutput:
        hn = self._encode(video)  # (B, T, d)
        tf = F.normalize(self._text(), dim=-1)  # (C, d)
        sim = hn @ tf.t()  # (B, T, C) cosine similarity
        s_nn = sim[..., 0]  # frame↔normal-text similarity (B, T)
        s_an = sim[..., 1:].max(dim=-1).values  # best frame↔anomaly-text (B, T)

        # PLG fusion -> eval anomaly score in [0, 1] (base text, no NVP at test)
        a = self.config.alpha
        psi = a * s_an + (1 - a) * (1 - s_nn)  # (B, T)
        score = psi.clamp(0, 1).unsqueeze(-1)  # (B, T, 1)

        loss = None
        if abnormal_labels is not None and normal_labels is not None:
            loss = self._compute_loss(hn, tf, sim, s_nn, s_an)

        return TPWNGVideoAnomalyDetectionOutput(loss=loss, scores=score, similarity=sim)

    def _compute_loss(self, hn, tf, sim, s_nn, s_an):
        bs, t = s_nn.shape
        half = bs // 2
        a = self.config.alpha

        # NVP: refine the normal-event text from the batch's normal frames, then
        # recompute the normal-text similarity used to guide DIL / pseudo-labels.
        if self.nvp is not None and half > 0:
            normal_frames = hn[:half].reshape(-1, hn.shape[-1])
            enhanced = self.nvp(tf[0], normal_frames)
            enhanced = F.normalize(enhanced, dim=-1)
            s_nn = hn @ enhanced  # (B, T) normality-guided

        # PLG with the (NVP-guided) normal similarity -> pseudo-labels
        psi = a * s_an + (1 - a) * (1 - s_nn)
        score_g = psi.clamp(0, 1)
        gamma = (psi.detach() > self.config.theta).float()
        loss_cl = F.binary_cross_entropy(score_g.clamp(1e-6, 1 - 1e-6), gamma)

        # ranking: anomaly-sim higher in abnormal than normal; normal-sim high on normal
        an_max = s_an.max(dim=1).values  # (B,)
        loss_rank = torch.relu(1.0 - an_max[half:] + an_max[:half]).mean()
        loss_rank = loss_rank + torch.relu(1.0 - s_nn[:half].max(dim=1).values).mean()

        # distributional inconsistency: decorrelate normal vs anomaly similarity curves
        sn = F.normalize(s_nn, dim=1)
        sa = F.normalize(s_an, dim=1)
        loss_dil = (sn * sa).sum(dim=1).mean()  # cosine, pushed toward 0

        # sparsity + smoothness on the anomaly score curve
        score = score_g.unsqueeze(-1)
        loss_reg = self.sparse(score[half:].reshape(-1)) + self.smooth(score)

        return loss_cl + loss_rank + loss_dil + loss_reg
