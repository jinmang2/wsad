from transformers.configuration_utils import PretrainedConfig


class CLIPTSAConfig(PretrainedConfig):
    """Configuration for CLIP-TSA (Joo et al., ICIP'23).

    CLIP-TSA = CLIP frame features + Temporal Self-Attention encoder + an
    RTFM-style top-k feature-magnitude MIL head.

    Args:
        feature_size: CLIP ViT-B/16 dim (512). Appended magnitude channel sliced.
        num_segments: temporal length T at train time.
        num_layers / num_heads: TSA encoder depth / heads.
        mlp_ratio: FFN expansion in the encoder.
        k: top-k snippets selected by magnitude.
        dropout_rate: dropout in encoder + scoring MLP + top-k selection.
        attn_impl: ``"eager"`` (bit-exact, default) or ``"sdpa"`` (flash/fused).
        margin / alpha / lambda_*: forwarded to the RTFM-style losses.
    """

    def __init__(
        self,
        feature_size: int = 512,
        num_segments: int = 32,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        k: int = 3,
        dropout_rate: float = 0.5,
        attn_impl: str = "eager",
        margin: float = 100.0,
        alpha: float = 0.0001,
        lambda_smooth: float = 8e-4,
        lambda_sparse: float = 8e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_size = feature_size
        self.num_segments = num_segments
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.k = k
        self.dropout_rate = dropout_rate
        self.attn_impl = attn_impl
        self.margin = margin
        self.alpha = alpha
        self.lambda_smooth = lambda_smooth
        self.lambda_sparse = lambda_sparse
