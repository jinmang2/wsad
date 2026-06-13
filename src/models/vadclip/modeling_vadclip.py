"""VadCLIP: Adapting Vision-Language Models for WSVAD (AAAI'24).

Faithful port of the official ``CLIPVAD`` (nwpu-zxr/VadCLIP, ``src/model.py``) into
this repo's HF ``PreTrainedModel``/``Config`` style. The module/parameter names
mirror the official state dict so ``pretrained/vadclip/model_ucf.pth`` loads
directly (``convert_official_vadclip`` drops only the unused ``clipmodel.visual.*``
and ``clipmodel.logit_scale`` keys).

Structure (official):
  - ``temporal``  : ``Transformer`` of post-/pre-norm ``ResidualAttentionBlock``
    (``nn.MultiheadAttention`` + QuickGELU MLP), seq-first ``(T,B,D)``, fixed
    ``visual_length`` with a windowed local-attention mask (``attn_window``).
  - ``gc1..gc4`` + ``disAdj`` : LGT-Adapter — similarity graph (``adj4``: cosine,
    threshold 0.7, row-softmax) and distance graph (``exp(-|i-j|/e)``).
  - ``linear`` fuses the two graph branches.
  - ``mlp2``/``classifier`` : binary branch (C) -> per-frame anomaly logits.
  - ``clipmodel`` (frozen CLIP text tower) + ``text_prompt_embeddings`` (learnable
    CoOp context) + ``encode_textprompt`` -> class text features; ``mlp1`` + the
    visual-language alignment branch (A) -> per-class logits.

Verified numerically against the official code (see ``scripts/verify_vadclip``):
the binary branch (``binary_logits``, fp32, text-independent) matches the official
``logits1`` to ~1e-4; the alignment branch matches ``logits2`` to the fp16 CLIP
text-tower precision (the official runs ``clipmodel`` in half).

``.scores`` (runner/eval contract) = ``sigmoid(binary logits)``.

Ref: https://github.com/nwpu-zxr/VadCLIP/blob/main/src/model.py
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from src.modules.graph import GraphConvolution
from src.registry import MODELS

from .configuration_vadclip import VadCLIPConfig


@dataclass
class VadCLIPVideoAnomalyDetectionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    scores: Optional[torch.FloatTensor] = None
    binary_logits: Optional[torch.FloatTensor] = None
    alignment_logits: Optional[torch.FloatTensor] = None
    text_features: Optional[torch.FloatTensor] = None


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """LayerNorm that casts to fp32 then back (official CLIP/VadCLIP behaviour)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.dtype
        return super().forward(x.float()).to(orig)


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


class ResidualAttentionBlock(nn.Module):
    """Official CLIP/VadCLIP residual block: ``nn.MultiheadAttention`` (combined
    ``in_proj``) + QuickGELU MLP, pre-norm, operating seq-first ``(L, N, E)``."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = _mlp(d_model)
        self.ln_2 = LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        mask = attn_mask.to(dtype=x.dtype, device=x.device) if attn_mask is not None else None
        h = self.ln_1(x)  # official: ln_1 once, shared as q=k=v
        x = x + self.attn(h, h, h, need_weights=False, attn_mask=mask)[0]
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.resblocks = nn.ModuleList(
            [ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x, attn_mask)
        return x


class DistanceAdj(nn.Module):
    """Distance graph ``exp(-|i-j|/e)`` (official ``DistanceAdj``). ``sigma`` is an
    unused parameter kept to absorb the official ``disAdj.sigma`` checkpoint key."""

    def __init__(self):
        super().__init__()
        self.sigma = nn.Parameter(torch.empty(1))
        self.sigma.data.fill_(0.1)

    def forward(self, batch_size: int, t: int, device) -> torch.Tensor:
        arith = torch.arange(t, device=device).float()
        dist = (arith[:, None] - arith[None, :]).abs()
        adj = torch.exp(-dist / torch.exp(torch.tensor(1.0, device=device)))
        return adj.unsqueeze(0).repeat(batch_size, 1, 1)


class CLIPTextTower(nn.Module):
    """Frozen CLIP text tower (OpenAI ViT-B/16). Loads from ``clipmodel.*`` keys.

    Reproduces the official ``encode_token`` / ``encode_text(embeddings, tokens)``.
    """

    def __init__(self, vocab: int, ctx: int, width: int, layers: int, heads: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab, width)
        self.positional_embedding = nn.Parameter(torch.empty(ctx, width))
        self.transformer = Transformer(width, layers, heads)
        self.ln_final = LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, width))
        mask = torch.empty(ctx, ctx).fill_(float("-inf")).triu_(1)  # causal
        self.register_buffer("attn_mask", mask, persistent=False)
        # CLIP-style init so the tower is valid even before loading clipmodel.*
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_projection, std=width**-0.5)

    def encode_token(self, token: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(token)

    def encode_text(self, emb: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        x = emb + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), token.argmax(dim=-1)] @ self.text_projection
        return x


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

    @property
    def dummy_inputs(self):
        return {"video": torch.randn(2, 1, self.config.visual_length, 512)}


@MODELS.register("vadclip")
class VadCLIPForVideoAnomalyDetection(VadCLIPPreTrainedModel):
    def __init__(self, config: VadCLIPConfig):
        super().__init__(config)
        vw = config.visual_width
        half = vw // 2

        # slot 2: LGT-Adapter — temporal Transformer + similarity/distance graphs
        self.temporal = Transformer(vw, config.visual_layers, config.visual_head)
        self.gc1 = GraphConvolution(vw, half, residual=True)
        self.gc2 = GraphConvolution(half, half, residual=True)
        self.gc3 = GraphConvolution(vw, half, residual=True)
        self.gc4 = GraphConvolution(half, half, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(vw, vw)
        self.gelu = QuickGELU()

        # slot 4: binary (C) and alignment (A) heads
        self.mlp1 = _mlp(vw)
        self.mlp2 = _mlp(vw)
        self.classifier = nn.Linear(vw, 1)

        # slot 3a: frozen CLIP text tower + learnable CoOp context
        self.clipmodel = CLIPTextTower(
            config.clip_vocab_size,
            config.clip_context_length,
            config.clip_text_width,
            config.clip_text_layers,
            config.clip_text_heads,
        )
        self.frame_position_embeddings = nn.Embedding(config.visual_length, vw)
        self.text_prompt_embeddings = nn.Embedding(config.clip_context_length, config.embed_dim)

        # windowed local-attention mask for the temporal branch (non-persistent)
        self.register_buffer(
            "temporal_mask",
            self._build_window_mask(config.visual_length, config.attn_window),
            persistent=False,
        )

        self._force_split = False
        self._tokenizer = None
        self.post_init()
        # keep CLIP text tower frozen (official freezes clipmodel)
        for p in self.clipmodel.parameters():
            p.requires_grad = False

    # ---- masks ----
    @staticmethod
    def _build_window_mask(visual_length: int, attn_window: int) -> torch.Tensor:
        """Block-diagonal windowed mask (official ``build_attention_mask``)."""
        mask = torch.empty(visual_length, visual_length).fill_(float("-inf"))
        n = visual_length // attn_window
        for i in range(n):
            lo = i * attn_window
            hi = min((i + 1) * attn_window, visual_length)
            if (i + 1) * attn_window < visual_length:
                mask[lo:hi, lo:hi] = 0
            else:
                mask[lo:visual_length, lo:visual_length] = 0
        return mask

    @property
    def force_split(self) -> bool:
        return self._force_split

    @force_split.setter
    def force_split(self, val: bool):
        self._force_split = val

    # ---- graphs ----
    def adj4(self, x: torch.Tensor, seq_len) -> torch.Tensor:
        """Cosine-similarity adjacency, threshold 0.7, row-softmax (official ``adj4``)."""
        x2 = x.matmul(x.permute(0, 2, 1))
        norm = torch.norm(x, p=2, dim=2, keepdim=True)
        x2 = x2 / (norm.matmul(norm.permute(0, 2, 1)) + 1e-20)
        out = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                a = F.threshold(x2[i], 0.7, 0)
                out[i] = F.softmax(a, dim=1)
        else:
            for i in range(len(seq_len)):
                L = int(seq_len[i])
                a = F.threshold(x2[i, :L, :L], 0.7, 0)
                out[i, :L, :L] = F.softmax(a, dim=1)
        return out

    def encode_video(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        # x: (B, visual_length, D)
        b, t, _ = x.shape
        pos_ids = torch.arange(self.config.visual_length, device=x.device)
        pos = self.frame_position_embeddings(pos_ids).unsqueeze(0).expand(b, -1, -1)
        h = (x + pos).permute(1, 0, 2)  # (T, B, D)
        h = self.temporal(h, self.temporal_mask)
        h = h.permute(1, 0, 2)  # (B, T, D)

        adj = self.adj4(h, lengths)
        dis = self.disAdj(b, t, x.device)
        x1 = self.gelu(self.gc2(self.gelu(self.gc1(h, adj)), adj))
        x2 = self.gelu(self.gc4(self.gelu(self.gc3(h, dis)), dis))
        return self.linear(torch.cat((x1, x2), dim=2))

    # ---- text ----
    def _tokenize(self, names: List[str]) -> torch.Tensor:
        if self._tokenizer is None:
            import open_clip  # lazy

            self._tokenizer = open_clip.get_tokenizer(self.config.clip_tokenizer_name)
        device = self.text_prompt_embeddings.weight.device
        return self._tokenizer(names).to(device)

    def encode_textprompt(self) -> torch.Tensor:
        names = self.config.class_names
        device = self.text_prompt_embeddings.weight.device
        word_tokens = self._tokenize(names)  # (C, 77)
        word_embedding = self.clipmodel.encode_token(word_tokens)  # (C, 77, D)
        ctx = self.text_prompt_embeddings(
            torch.arange(self.config.clip_context_length, device=device)
        )
        text_embeddings = ctx.unsqueeze(0).repeat(len(names), 1, 1)
        text_tokens = torch.zeros(len(names), self.config.clip_context_length, device=device)
        pre, post = self.config.prompt_prefix, self.config.prompt_postfix
        for i in range(len(names)):
            ind = int(torch.argmax(word_tokens[i], dim=-1))
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, pre + 1 : pre + ind] = word_embedding[i, 1:ind]
            text_embeddings[i, pre + ind + post] = word_embedding[i, ind]
            text_tokens[i, pre + ind + post] = word_tokens[i, ind]
        return self.clipmodel.encode_text(text_embeddings, text_tokens)

    def forward(
        self,
        video: torch.FloatTensor,
        abnormal_labels: Optional[torch.FloatTensor] = None,
        normal_labels: Optional[torch.FloatTensor] = None,
        class_labels: Optional[torch.FloatTensor] = None,
        lengths=None,
    ) -> VadCLIPVideoAnomalyDetectionOutput:
        x = video
        if x.dim() == 4:  # (B, ncrops, T, D) -> single-stream (mean over crops)
            x = x[..., : self.config.feature_size].mean(dim=1)

        vf = self.encode_video(x, lengths)  # (B, T, D)
        logits1 = self.classifier(vf + self.mlp2(vf))  # (B, T, 1)

        # visual-language alignment (A) — official forward() math (no eps in norms)
        text_features_ori = self.encode_textprompt()  # (C, D)
        attn = logits1.permute(0, 2, 1) @ vf  # (B, 1, D)
        attn = attn / attn.norm(dim=-1, keepdim=True)
        attn = attn.expand(x.shape[0], text_features_ori.shape[0], attn.shape[2])
        tf = text_features_ori.unsqueeze(0).expand(x.shape[0], -1, -1)
        tf = tf + attn
        tf = tf + self.mlp1(tf)
        vf_n = vf / vf.norm(dim=-1, keepdim=True)
        tf_n = tf / tf.norm(dim=-1, keepdim=True)
        logits2 = vf_n @ tf_n.permute(0, 2, 1).type(vf_n.dtype) / 0.07  # (B, T, C)

        scores = torch.sigmoid(logits1)
        loss = None
        if abnormal_labels is not None and normal_labels is not None:
            loss = self._compute_loss(logits1, logits2, text_features_ori, class_labels)

        return VadCLIPVideoAnomalyDetectionOutput(
            loss=loss,
            scores=scores,
            binary_logits=logits1,
            alignment_logits=logits2,
            text_features=text_features_ori,
        )

    # ---- losses (official CLAS2 / CLASM / text contrastive) ----
    def _topk_k(self, t: int) -> int:
        return max(t // 16 + 1, 1)

    def _compute_loss(self, binary_logits, alignment_logits, text_features_ori, class_labels):
        bs, t, _ = binary_logits.size()
        half = bs // 2
        device = binary_logits.device
        k = self._topk_k(t)

        y = torch.cat([torch.zeros(half, device=device), torch.ones(half, device=device)])
        probs = torch.sigmoid(binary_logits).squeeze(-1)
        inst = torch.stack([torch.topk(probs[i], k)[0].mean() for i in range(bs)])
        loss1 = F.binary_cross_entropy(inst.clamp(1e-6, 1 - 1e-6), y)

        tf = text_features_ori / (text_features_ori.norm(dim=-1, keepdim=True) + 1e-12)
        normal = tf[0]
        loss3 = sum(torch.abs(normal @ tf[j]) for j in range(1, tf.size(0)))
        loss3 = loss3 / max(tf.size(0) - 1, 1) * self.config.w_text
        loss = loss1 + loss3

        if class_labels is not None:
            labels = F.one_hot(class_labels.long().to(device), self.config.num_class).float()
            labels = labels / labels.sum(dim=1, keepdim=True).clamp_min(1e-6)
            inst_logits = torch.stack(
                [torch.topk(alignment_logits[i], k, dim=0)[0].mean(0) for i in range(bs)]
            )
            loss2 = -torch.mean(torch.sum(labels * F.log_softmax(inst_logits, dim=1), dim=1))
            loss = loss + loss2
        return loss


def convert_official_vadclip(state_dict: dict) -> "OrderedDict":
    """Map the official ``CLIPVAD`` state dict onto this port.

    Names already match (temporal/gc/disAdj/linear/mlp/classifier/clipmodel/
    frame_position_embeddings/text_prompt_embeddings); only the unused CLIP visual
    tower (``clipmodel.visual.*``) and ``clipmodel.logit_scale`` are dropped.
    """
    out = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("clipmodel.visual.") or k == "clipmodel.logit_scale":
            continue
        out[k] = v
    return out
