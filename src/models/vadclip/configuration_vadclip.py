from transformers.configuration_utils import PretrainedConfig


class VadCLIPConfig(PretrainedConfig):
    """Configuration for VadCLIP (Wu et al., AAAI'24).

    Faithful to the official ``CLIPVAD`` (nwpu-zxr/VadCLIP, ``src/model.py`` +
    ``src/ucf_option.py``). Dual-branch over CLIP ViT-B/16 frame features: a
    coarse binary branch (C) and a fine visual-language alignment branch (A), with
    an LGT-Adapter (windowed temporal Transformer + similarity/distance graph
    convs) and a learnable-CoOp text branch run through a **frozen CLIP text
    tower** (``clipmodel``) every forward.

    Defaults match official UCF (``ucf_option.py``): visual_length 256,
    visual_width 512, visual_head 1, visual_layers 2, attn_window 8,
    prompt_prefix/postfix 10, classes 14.

    Args:
        feature_size / embed_dim: CLIP ViT-B/16 dim (512).
        visual_width / visual_layers / visual_head: temporal Transformer.
        visual_length: fixed padded length for the temporal branch + frame
            position embeddings (256). Inputs are padded/split to this length.
        attn_window: local attention window for the temporal Transformer.
        num_class: anomaly classes incl. "Normal" (UCF: 14).
        attn_impl: ``"eager"`` (exact) or ``"sdpa"`` (flash/fused, not bit-exact).
        w_text: weight on the text-feature contrastive (loss3, paper: 0.1).
        prompt_prefix / prompt_postfix: learnable CoOp context tokens before/after
            the class name in the 77-token prompt (paper: 10 / 10).
        clip_text_*: the frozen CLIP text tower architecture (OpenAI ViT-B/16:
            vocab 49408, context 77, width 512, 12 layers, 8 heads). Its weights
            load from the official ``clipmodel.*`` checkpoint keys.
        class_names: per-class prompt names — **official capitalized label_map
            values**, since the BPE tokenization is case-sensitive (``"Normal"`` ≠
            ``"normal"``); changing case changes the text features.
    """

    def __init__(
        self,
        feature_size: int = 512,
        embed_dim: int = 512,
        visual_width: int = 512,
        visual_layers: int = 2,
        visual_head: int = 1,
        visual_length: int = 256,
        attn_window: int = 8,
        num_class: int = 14,
        dropout_rate: float = 0.0,
        attn_impl: str = "eager",
        w_text: float = 0.1,
        prompt_prefix: int = 10,
        prompt_postfix: int = 10,
        # frozen CLIP text tower (OpenAI ViT-B/16)
        clip_vocab_size: int = 49408,
        clip_context_length: int = 77,
        clip_text_width: int = 512,
        clip_text_layers: int = 12,
        clip_text_heads: int = 8,
        clip_tokenizer_name: str = "ViT-B-16",
        class_names: list = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_size = feature_size
        self.embed_dim = embed_dim
        self.visual_width = visual_width
        self.visual_layers = visual_layers
        self.visual_head = visual_head
        self.visual_length = visual_length
        self.attn_window = attn_window
        self.num_class = num_class
        self.dropout_rate = dropout_rate
        self.attn_impl = attn_impl
        self.w_text = w_text
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.clip_vocab_size = clip_vocab_size
        self.clip_context_length = clip_context_length
        self.clip_text_width = clip_text_width
        self.clip_text_layers = clip_text_layers
        self.clip_text_heads = clip_text_heads
        self.clip_tokenizer_name = clip_tokenizer_name
        # official UCF label_map (ucf_test.py), Normal-first; class 0 == Normal.
        # Capitalized to match the official case-sensitive BPE tokenization.
        self.class_names = class_names or [
            "Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary",
            "Explosion", "Fighting", "RoadAccidents", "Robbery", "Shooting",
            "Shoplifting", "Stealing", "Vandalism",
        ]
