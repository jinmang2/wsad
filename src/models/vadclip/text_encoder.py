"""CLIP text encoder with learnable CoOp prompt context (VadCLIP ``encode_textprompt``).

Faithful port of the official ``CLIPVAD.encode_textprompt`` (nwpu-zxr/VadCLIP,
``src/model.py``) onto an ``open_clip`` backbone. The official forks OpenAI CLIP
with custom ``encode_token`` / ``encode_text(embeddings, tokens)`` methods; we
reproduce the same math against the public ``open_clip`` text tower:

  tokens = tokenize(class_name)                       # (C, 77)
  word_emb = token_embedding(tokens)                  # frozen CLIP token embeddings
  ctx = text_prompt_embeddings(arange(77))            # LEARNABLE CoOp context
  build prompt: [ BOS | ctx[:prefix] | class words | ctx | EOS@(prefix+len+postfix) ]
  x = transformer(prompt + positional, attn_mask) -> ln_final -> EOS pool @ proj

The CLIP weights are frozen; only ``text_prompt_embeddings`` learns. Use the same
CLIP variant the visual features were extracted with (UCF VadCLIP features =
OpenAI ViT-B/16) so text and frame features share one space — hence the default
``pretrained="openai"``.
"""

from typing import List, Optional

import torch
from torch import nn


class CLIPPromptTextEncoder(nn.Module):
    """Frozen CLIP text tower + learnable CoOp context (prompt_prefix/postfix)."""

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: Optional[str] = "openai",
        embed_dim: int = 512,
        prompt_prefix: int = 10,
        prompt_postfix: int = 10,
        unfreeze_projection: bool = False,
    ):
        super().__init__()
        import open_clip  # lazy: only needed for the faithful text path

        clip_model = open_clip.create_model(model_name, pretrained=pretrained)
        # frozen CLIP text tower pieces (the visual tower is dropped)
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = nn.Parameter(
            clip_model.positional_embedding.data.clone()
        )
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        text_projection = clip_model.text_projection
        if isinstance(text_projection, nn.Linear):
            self.text_projection = text_projection
            self._proj_is_linear = True
        else:
            self.text_projection = nn.Parameter(text_projection.data.clone())
            self._proj_is_linear = False
        self.register_buffer("attn_mask", clip_model.attn_mask, persistent=False)
        for p in self.parameters():
            p.requires_grad = False
        # TPWNG fine-tunes ONLY the final text projection (image+text towers frozen)
        if unfreeze_projection:
            if self._proj_is_linear:
                for p in self.text_projection.parameters():
                    p.requires_grad = True
            else:
                self.text_projection.requires_grad = True

        self.context_length = self.positional_embedding.shape[0]  # 77
        self.embed_dim = embed_dim
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix

        # the ONLY learnable text parameter (CoOp context), official std 0.01
        self.text_prompt_embeddings = nn.Embedding(self.context_length, embed_dim)
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)

        self._tokenizer = open_clip.get_tokenizer(model_name)

    def _tokenize(self, class_names: List[str]) -> torch.Tensor:
        return self._tokenizer(class_names).to(self.positional_embedding.device)

    def forward(self, class_names: List[str]) -> torch.Tensor:
        """Encode each class prompt -> ``(num_class, embed_dim)`` text features."""
        device = self.positional_embedding.device
        word_tokens = self._tokenize(class_names)  # (C, 77)
        c = word_tokens.shape[0]
        word_embedding = self.token_embedding(word_tokens)  # (C, 77, d) frozen

        ctx = self.text_prompt_embeddings(
            torch.arange(self.context_length, device=device)
        )
        text_embeddings = ctx.unsqueeze(0).repeat(c, 1, 1)  # (C, 77, d) learnable
        eot_pos = torch.zeros(c, dtype=torch.long, device=device)

        pre, post = self.prompt_prefix, self.prompt_postfix
        for i in range(c):
            ind = int(torch.argmax(word_tokens[i], dim=-1))  # EOS position = 1+len
            text_embeddings[i, 0] = word_embedding[i, 0]  # BOS
            text_embeddings[i, pre + 1 : pre + ind] = word_embedding[i, 1:ind]
            text_embeddings[i, pre + ind + post] = word_embedding[i, ind]  # EOS
            eot_pos[i] = pre + ind + post

        x = text_embeddings + self.positional_embedding
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)
        x = x[torch.arange(c, device=device), eot_pos]  # EOS pool
        if self._proj_is_linear:
            x = self.text_projection(x)
        else:
            x = x @ self.text_projection
        return x  # (C, embed_dim)
