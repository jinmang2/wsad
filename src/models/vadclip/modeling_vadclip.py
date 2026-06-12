"""VadCLIP: Adapting Vision-Language Models for WSVAD (AAAI'24).

Faithful adaptation of the official ``CLIPVAD`` (nwpu-zxr/VadCLIP) to this repo's
runner contract. Slot mapping (WSAD_INTEGRATION_PLAN.md section 4):
  - slot 1 (features): CLIP ViT-B/16 frame features (512-d, text-aligned).
  - slot 2 (temporal encoder): **LGT-Adapter** — a windowed-local temporal
    Transformer + two graph-conv branches (similarity graph = global/semantic,
    distance graph = local/temporal), concatenated.
  - slot 3a (text branch): **faithful** ``encode_textprompt`` — a frozen CLIP text
    tower (OpenAI ViT-B/16) with learnable CoOp context tokens (prompt_prefix=10 /
    postfix=10) around each class name, encoded every forward (see
    ``text_encoder.CLIPPromptTextEncoder``). ``use_clip_text=False`` falls back to a
    free learnable table for offline/no-download runs.
  - slot 4 (scoring head): binary branch (C) -> per-frame anomaly logits, and the
    visual-language alignment branch (A) -> per-class logits (MIL-Align).
  - slot 5 (loss): binary MIL BCE (CLAS2) + text contrastive (loss3); the
    multi-class MIL-Align CE (CLASM, loss2) activates when per-video class labels
    are supplied (a data extension — current cache only has binary labels).

``.scores`` (runner/eval contract) = ``sigmoid(binary logits)``; the alignment
logits are returned alongside for the A-branch.

Differences vs official (architecturally equivalent, not bit-exact): the temporal
block is this repo's pre-norm ``TransformerEncoderLayer`` (eager/SDPA switchable)
rather than the official post-norm QuickGELU ``ResidualAttentionBlock``. The text
branch, LGT-Adapter (adj4 + DistanceAdj), losses (CLAS2/CLASM/text-contrastive),
and the (B, 1, 256, 512) input contract now match the official ``CLIPVAD``.

Ref: https://github.com/nwpu-zxr/VadCLIP/blob/main/src/model.py
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from src.modules import TransformerEncoderLayer
from src.modules.graph import GraphConvolution, distance_adj, similarity_adj
from src.registry import MODELS

from .configuration_vadclip import VadCLIPConfig


@dataclass
class VadCLIPVideoAnomalyDetectionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    scores: Optional[torch.FloatTensor] = None
    binary_logits: Optional[torch.FloatTensor] = None
    alignment_logits: Optional[torch.FloatTensor] = None


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


def _mlp(width: int) -> nn.Sequential:
    return nn.Sequential(
        OrderedDict(
            [
                ("c_fc", nn.Linear(width, width * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(width * 4, width)),
            ]
        )
    )


class VadCLIPPreTrainedModel(PreTrainedModel):
    config_class = VadCLIPConfig
    base_model_prefix = "vadclip"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.01)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    @property
    def dummy_inputs(self):
        return {"video": torch.randn(8, 1, self.config.visual_length, 512)}


@MODELS.register("vadclip")
class VadCLIPForVideoAnomalyDetection(VadCLIPPreTrainedModel):
    def __init__(self, config: VadCLIPConfig):
        super().__init__(config)
        vw = config.visual_width
        half = vw // 2

        # slot 2: LGT-Adapter
        self.temporal = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    dim=vw,
                    heads=config.visual_head,
                    dropout=config.dropout_rate,
                    attn_impl=config.attn_impl,
                )
                for _ in range(config.visual_layers)
            ]
        )
        self.frame_position_embeddings = nn.Embedding(config.visual_length, vw)
        self.gc1 = GraphConvolution(vw, half, residual=True)  # similarity graph
        self.gc2 = GraphConvolution(half, half, residual=True)
        self.gc3 = GraphConvolution(vw, half, residual=True)  # distance graph
        self.gc4 = GraphConvolution(half, half, residual=True)
        self.linear = nn.Linear(vw, vw)
        self.gelu = QuickGELU()

        # slot 4: binary (C) branch
        self.mlp2 = _mlp(vw)
        self.classifier = nn.Linear(vw, 1)

        # slot 3a/4: alignment (A) branch
        self.mlp1 = _mlp(vw)
        # text branch: faithful = frozen CLIP text tower + learnable CoOp context
        # (encode_textprompt); legacy = a free learnable table (offline/no CLIP).
        self.use_clip_text = config.use_clip_text
        self.text_encoder = None
        # legacy table is a plain Parameter (untouched by HF _init_weights)
        self.text_features = (
            None
            if config.use_clip_text
            else nn.Parameter(torch.empty(config.num_class, config.embed_dim))
        )
        if self.text_features is not None:
            nn.init.normal_(self.text_features, std=0.01)

        self._force_split = False
        self.post_init()

        # build the frozen CLIP text tower AFTER post_init so HF _init_weights does
        # NOT re-initialize its pretrained weights.
        if config.use_clip_text:
            from .text_encoder import CLIPPromptTextEncoder

            self.text_encoder = CLIPPromptTextEncoder(
                model_name=config.clip_model_name,
                pretrained=config.clip_pretrained,
                embed_dim=config.embed_dim,
                prompt_prefix=config.prompt_prefix,
                prompt_postfix=config.prompt_postfix,
            )

    def _text_features(self) -> torch.Tensor:
        """``(num_class, embed_dim)`` class text features (CLIP-encoded or table)."""
        if self.text_encoder is not None:
            return self.text_encoder(self.config.class_names)
        return self.text_features

    @property
    def force_split(self) -> bool:
        return self._force_split

    @force_split.setter
    def force_split(self, val: bool):
        self._force_split = val

    def _window_bias(self, t: int, device) -> torch.Tensor:
        """Block-diagonal local-attention mask (0 inside window, -inf outside)."""
        w = self.config.attn_window
        mask = torch.full((t, t), float("-inf"), device=device)
        for i in range((t + w - 1) // w):
            lo, hi = i * w, min((i + 1) * w, t)
            mask[lo:hi, lo:hi] = 0.0
        return mask.view(1, 1, t, t)

    def _position_embeddings(self, t: int, device) -> torch.Tensor:
        """Frame position embeddings for length ``t`` (interpolated if t > 256)."""
        vlen = self.config.visual_length
        if t <= vlen:
            return self.frame_position_embeddings(torch.arange(t, device=device))
        # eval on full-length clips: linearly interpolate the learned table to t
        table = self.frame_position_embeddings.weight.t().unsqueeze(0)  # (1, vw, vlen)
        out = F.interpolate(table, size=t, mode="linear", align_corners=False)
        return out.squeeze(0).t()  # (t, vw)

    def encode_video(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, vw)
        b, t, _ = x.size()
        h = x + self._position_embeddings(t, x.device).unsqueeze(0)

        bias = self._window_bias(t, x.device)
        for layer in self.temporal:
            h = layer(h, attn_bias=bias)

        sim = similarity_adj(h)
        dis = distance_adj(t, b, x.device)
        x1 = self.gelu(self.gc2(self.gelu(self.gc1(h, sim)), sim))
        x2 = self.gelu(self.gc4(self.gelu(self.gc3(h, dis)), dis))
        out = self.linear(torch.cat((x1, x2), dim=2))
        return out

    def forward(
        self,
        video: torch.FloatTensor,
        abnormal_labels: Optional[torch.FloatTensor] = None,
        normal_labels: Optional[torch.FloatTensor] = None,
        class_labels: Optional[torch.FloatTensor] = None,
    ) -> VadCLIPVideoAnomalyDetectionOutput:
        f = self.config.feature_size
        x = video[..., :f].mean(
            dim=1
        )  # (B, T, f) — mean over crops (CLIP single-stream)

        vf = self.encode_video(x)  # (B, T, vw)
        binary_logits = self.classifier(vf + self.mlp2(vf))  # (B, T, 1)

        # visual-language alignment (A branch) — official forward() math
        text_features_ori = self._text_features()  # (C, embed) CLIP-encoded or table
        attn = binary_logits.permute(0, 2, 1) @ vf  # (B, 1, vw)
        attn = attn / (attn.norm(dim=-1, keepdim=True) + 1e-12)
        tf = text_features_ori.unsqueeze(0).expand(x.size(0), -1, -1)  # (B, C, embed)
        tf = tf + attn.expand(-1, tf.size(1), -1)
        tf = tf + self.mlp1(tf)
        vf_n = vf / (vf.norm(dim=-1, keepdim=True) + 1e-12)
        tf_n = tf / (tf.norm(dim=-1, keepdim=True) + 1e-12)
        alignment_logits = vf_n @ tf_n.permute(0, 2, 1) / 0.07  # (B, T, C)

        scores = torch.sigmoid(binary_logits)  # (B, T, 1) — eval/runner contract

        loss = None
        if abnormal_labels is not None and normal_labels is not None:
            loss = self._compute_loss(
                binary_logits, alignment_logits, text_features_ori, class_labels
            )

        return VadCLIPVideoAnomalyDetectionOutput(
            loss=loss,
            scores=scores,
            binary_logits=binary_logits,
            alignment_logits=alignment_logits,
        )

    # ---- losses (official CLAS2 / CLASM / text contrastive) ----
    def _topk_k(self, t: int) -> int:
        return max(t // 16 + 1, 1)

    def _compute_loss(
        self, binary_logits, alignment_logits, text_features_ori, class_labels
    ):
        bs, t, _ = binary_logits.size()
        half = bs // 2
        device = binary_logits.device
        k = self._topk_k(t)

        # loss1 = CLAS2: binary MIL BCE (abnormal=1, normal=0; normal-first batch)
        y = torch.cat(
            [torch.zeros(half, device=device), torch.ones(half, device=device)]
        )
        probs = torch.sigmoid(binary_logits).squeeze(-1)  # (B, T)
        inst = torch.stack([torch.topk(probs[i], k)[0].mean() for i in range(bs)])
        loss1 = F.binary_cross_entropy(inst.clamp(1e-6, 1 - 1e-6), y)

        # loss3 = text-feature contrastive (Normal vs each abnormal class)
        tf = text_features_ori / (
            text_features_ori.norm(dim=-1, keepdim=True) + 1e-12
        )
        normal = tf[0]
        loss3 = sum(torch.abs(normal @ tf[j]) for j in range(1, tf.size(0)))
        loss3 = loss3 / max(tf.size(0) - 1, 1) * self.config.w_text

        loss = loss1 + loss3

        # loss2 = CLASM: multi-class MIL-Align CE (needs per-video class labels)
        if class_labels is not None:
            # class_labels: (B,) int (0=Normal, 1..13 anomaly) -> multi-hot
            labels = F.one_hot(
                class_labels.long().to(device), self.config.num_class
            ).float()
            labels = labels / labels.sum(dim=1, keepdim=True).clamp_min(1e-6)
            inst_logits = torch.stack(
                [
                    torch.topk(alignment_logits[i], k, dim=0)[0].mean(0)
                    for i in range(bs)
                ]
            )  # (B, C)
            loss2 = -torch.mean(
                torch.sum(labels * F.log_softmax(inst_logits, dim=1), dim=1)
            )
            loss = loss + loss2

        return loss

    @torch.no_grad()
    def load_clip_text_features(self, class_names=None, device: str = "cuda"):
        """Legacy-table only: CLIP-init the free ``text_features`` table.

        No-op when ``use_clip_text=True`` (the faithful path already encodes the
        prompts through the frozen CLIP text tower every forward via
        :class:`CLIPPromptTextEncoder`). Only useful as a one-time init for the
        ``use_clip_text=False`` fallback table.
        """
        if self.text_encoder is not None or self.text_features is None:
            return  # faithful path: nothing to fill
        import open_clip  # lazy

        names = class_names or self.config.class_names
        model = open_clip.create_model(
            self.config.clip_model_name, pretrained=self.config.clip_pretrained
        )
        model.eval().to(device)
        tokenizer = open_clip.get_tokenizer(self.config.clip_model_name)
        tokens = tokenizer([f"a video of {name}" for name in names]).to(device)
        feats = model.encode_text(tokens).float()
        assert feats.shape == self.text_features.shape
        self.text_features.data.copy_(feats.to(self.text_features.device))
