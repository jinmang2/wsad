from transformers.configuration_utils import PretrainedConfig


class URDMUConfig(PretrainedConfig):
    """Configuration for UR-DMU (Zhou et al., AAAI'23).

    Dual Memory Units (normal / abnormal) with Uncertainty Regulation.

    Args:
        feature_size: input snippet feature dim (I3D = 2048). Magnitude channel sliced.
        hidden_size: embedding / encoder width (paper: 512).
        num_layers / num_heads: distance-biased self-attention encoder.
        mem_size: slots per memory bank (paper: 60 each).
        topk_ratio: fraction of T used for the top-k MIL mean (score = mean of top T/ratio).
        dropout_rate / attn_impl: dropout; ``"eager"`` (exact) or ``"sdpa"`` (flash).
        margin: triplet margin for normal/abnormal memory separation.
        w_mem / w_triplet / w_kl: loss weights for memory-MIL, triplet, KL terms.
    """

    def __init__(
        self,
        feature_size: int = 2048,
        hidden_size: int = 512,
        num_layers: int = 2,
        num_heads: int = 4,
        mem_size: int = 60,
        topk_ratio: int = 16,
        dropout_rate: float = 0.5,
        attn_impl: str = "eager",
        margin: float = 1.0,
        w_mem: float = 1.0,
        w_triplet: float = 0.1,
        w_kl: float = 0.001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mem_size = mem_size
        self.topk_ratio = topk_ratio
        self.dropout_rate = dropout_rate
        self.attn_impl = attn_impl
        self.margin = margin
        self.w_mem = w_mem
        self.w_triplet = w_triplet
        self.w_kl = w_kl
