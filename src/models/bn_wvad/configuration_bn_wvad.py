from transformers.configuration_utils import PretrainedConfig


class BNWVADConfig(PretrainedConfig):
    """Configuration for BN-WVAD (Zhou et al., 2023).

    BatchNorm-based criterion: abnormal snippets are outliers w.r.t. the running
    statistics of a NormalHead's BatchNorm layers. The anomaly score is the
    Divergence-of-Feature-from-Mean (DFM, a Mahalanobis distance to the BN
    running mean/var) times a learned normal score.

    Defaults follow the official cool-xuan/BN-WVAD (`options.py`).

    Args:
        feature_size: I3D dim (UCF 2048). Magnitude channel sliced off.
        hidden_size: embedding / self-attention width (512).
        num_layers / num_heads: temporal self-attention.
        ratios / kernel_sizes: NormalHead channel reductions / conv kernels.
        ratio_sample / ratio_batch: top-DFM selection ratios (per-video / per-batch).
        mpp_margin: triplet margin (1.0); w_triplet: per-BN-layer triplet weights.
        w_normal / w_mpp: normal-loss / MPP-loss weights.
    """

    def __init__(
        self,
        feature_size: int = 1024,
        hidden_size: int = 512,
        num_layers: int = 2,
        num_heads: int = 4,
        ratios=(16, 32),
        kernel_sizes=(1, 1, 1),
        ratio_sample: float = 0.2,
        ratio_batch: float = 0.4,
        dropout_rate: float = 0.0,
        mpp_margin: float = 1.0,
        w_triplet=(5.0, 20.0),
        w_normal: float = 1.0,
        w_mpp: float = 1.0,
        attn_impl: str = "eager",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ratios = list(ratios)
        self.kernel_sizes = list(kernel_sizes)
        self.ratio_sample = ratio_sample
        self.ratio_batch = ratio_batch
        self.dropout_rate = dropout_rate
        self.mpp_margin = mpp_margin
        self.w_triplet = list(w_triplet)
        self.w_normal = w_normal
        self.w_mpp = w_mpp
        self.attn_impl = attn_impl
