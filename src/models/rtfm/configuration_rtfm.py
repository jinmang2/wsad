from transformers.configuration_utils import PretrainedConfig


class RTFMConfig(PretrainedConfig):
    """Configuration for RTFM (Tian et al., ICCV'21).

    Args:
        feature_size: input snippet feature dim (I3D = 2048). If cached features
            carry an appended magnitude channel (2049), the extra channel is
            sliced off using this value.
        num_segments: temporal length T at train time (Sultani 32-segment).
        k: top-k snippets selected by magnitude (paper: num_segments // 10 = 3).
        dropout_rate: dropout applied in the scoring MLP and top-k selection.
        margin / alpha: forwarded to the RTFM loss.
        lambda_smooth / lambda_sparse: temporal-smoothness / sparsity weights.
    """

    def __init__(
        self,
        feature_size: int = 2048,
        num_segments: int = 32,
        k: int = 3,
        dropout_rate: float = 0.7,
        margin: float = 100.0,
        alpha: float = 0.0001,
        lambda_smooth: float = 8e-4,
        lambda_sparse: float = 8e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_size = feature_size
        self.num_segments = num_segments
        self.k = k
        self.dropout_rate = dropout_rate
        self.margin = margin
        self.alpha = alpha
        self.lambda_smooth = lambda_smooth
        self.lambda_sparse = lambda_sparse
