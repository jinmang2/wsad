from transformers.configuration_utils import PretrainedConfig


class TPWNGConfig(PretrainedConfig):
    """Configuration for TPWNG (Yang et al., CVPR'24).

    Text Prompt with Normality Guidance: CLIP frame features aligned to learnable
    class-prompt text, a soft-mask temporal transformer (TCSAL), and pseudo-label
    self-training (the first PL / self-training method in this framework).

    Official code unreleased; defaults follow the paper (arXiv 2404.08531).

    Args:
        feature_size / embed_dim: CLIP ViT-B/16 dim (512).
        num_class: classes incl. Normal (UCF = 14); class 0 == Normal.
        n_ctx: learnable prompt context length (paper l=8).
        tcsal_layers / tcsal_heads: temporal transformer depth / heads (4 / 4).
        span_R: soft-mask ramp width R (paper 256).
        alpha: PLG fusion weight (paper 0.2).
        theta: pseudo-label threshold (paper UCF 0.55).
        lambda_sparse / lambda_smooth: regularizer weights (paper 0.1 / 0.01).
        attn_impl: "eager" (exact) or "sdpa" (flash/fused).
    """

    def __init__(
        self,
        feature_size: int = 512,
        embed_dim: int = 512,
        num_class: int = 14,
        n_ctx: int = 8,
        tcsal_layers: int = 4,
        tcsal_heads: int = 4,
        span_R: int = 256,
        alpha: float = 0.2,
        theta: float = 0.55,
        lambda_sparse: float = 0.1,
        lambda_smooth: float = 0.01,
        attn_impl: str = "eager",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_size = feature_size
        self.embed_dim = embed_dim
        self.num_class = num_class
        self.n_ctx = n_ctx
        self.tcsal_layers = tcsal_layers
        self.tcsal_heads = tcsal_heads
        self.span_R = span_R
        self.alpha = alpha
        self.theta = theta
        self.lambda_sparse = lambda_sparse
        self.lambda_smooth = lambda_smooth
        self.attn_impl = attn_impl
