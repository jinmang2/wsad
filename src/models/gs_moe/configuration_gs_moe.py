from transformers.configuration_utils import PretrainedConfig


class GSMoEConfig(PretrainedConfig):
    """Configuration for GS-MoE (ICCV'25).

    Mixture of class-specialized experts + a gate, trained with Temporal Gaussian
    Splatting (TGS). Official code unreleased at implementation time; defaults
    follow the paper (arXiv 2508.06318): I3D features, 13 UCF anomaly experts,
    AdamW, batch 128 (64+64), T=200 (we use the cache's T, e.g. 32).

    Args:
        feature_size: input snippet feature dim (our I3D cache = 2048; paper = 1024).
        hidden_size: expert/gate working width (paper expert width 1024).
        num_experts: class-specialized experts (UCF anomaly classes = 13).
        expert_heads / gate_heads: attention heads for expert / gate transformer.
        dropout_rate / attn_impl: dropout; "eager" (exact) or "sdpa" (flash).
        k_ratio: top-k MIL uses mean of top T/k_ratio snippets.
        tgs_prominence: peak prominence threshold for the TGS pseudo-labels.
        lambda_smooth / lambda_sparse / w_tgs: loss weights.
    """

    def __init__(
        self,
        feature_size: int = 2048,
        hidden_size: int = 1024,
        num_experts: int = 13,
        expert_heads: int = 2,
        gate_heads: int = 4,
        dropout_rate: float = 0.1,
        attn_impl: str = "eager",
        k_ratio: int = 16,
        tgs_prominence: float = 0.2,
        lambda_smooth: float = 8e-4,
        lambda_sparse: float = 8e-3,
        w_tgs: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.expert_heads = expert_heads
        self.gate_heads = gate_heads
        self.dropout_rate = dropout_rate
        self.attn_impl = attn_impl
        self.k_ratio = k_ratio
        self.tgs_prominence = tgs_prominence
        self.lambda_smooth = lambda_smooth
        self.lambda_sparse = lambda_sparse
        self.w_tgs = w_tgs
