from transformers.configuration_utils import PretrainedConfig


class VadCLIPConfig(PretrainedConfig):
    """Configuration for VadCLIP (Wu et al., AAAI'24).

    Dual-branch: a coarse-grained binary branch (C) and a fine-grained
    visual-language alignment branch (A) over CLIP features, with an LGT-Adapter
    (local windowed Transformer + global/local graph convs).

    Defaults match the official UCF setup (src/ucf_option.py).

    Args:
        feature_size: CLIP ViT-B/16 dim (512). Magnitude channel sliced off.
        embed_dim: CLIP text/visual embedding dim (512).
        visual_width / visual_layers / visual_head: temporal Transformer.
        visual_length: max padded sequence length (frame position embeddings).
        attn_window: local attention window for the temporal Transformer.
        num_class: anomaly classes incl. "Normal" (UCF: 14).
        attn_impl: ``"eager"`` (exact) or ``"sdpa"`` (flash/fused).
        w_text: weight on the text-feature contrastive (loss3, paper: 0.1).
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
