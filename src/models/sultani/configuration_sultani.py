from transformers.configuration_utils import PretrainedConfig


class SultaniConfig(PretrainedConfig):
    """Configuration for the Sultani MIL baseline (CVPR'18).

    Args:
        feature_size: input snippet feature dim (I3D = 2048). Any appended
            magnitude channel is sliced off using this value.
        hidden1 / hidden2: regressor MLP widths (paper: 512, 32).
        dropout_rate: dropout in the regressor (paper: 0.6).
        lambda_smooth / lambda_sparse: MIL regularizer weights (paper: 8e-5).
    """

    def __init__(
        self,
        feature_size: int = 2048,
        hidden1: int = 512,
        hidden2: int = 32,
        dropout_rate: float = 0.6,
        lambda_smooth: float = 8e-5,
        lambda_sparse: float = 8e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_size = feature_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.dropout_rate = dropout_rate
        self.lambda_smooth = lambda_smooth
        self.lambda_sparse = lambda_sparse
