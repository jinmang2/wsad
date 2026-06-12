from transformers.configuration_utils import PretrainedConfig


class S3RConfig(PretrainedConfig):
    """Configuration for S3R (Wu et al., ECCV'22).

    Self-supervised Sparse Representation: a normal-event **dictionary** is read
    by ``enNormal`` to reconstruct a normal-pattern stream, then ``deNormal``
    channel-filters it out of the video stream to highlight anomalies.

    Adaptation: the official dictionary is precomputed offline via dictionary
    learning on normal features; here it is a **learnable parameter** trained
    end-to-end (documented in the model). Cross-checked vs louisYen/S3R.

    Args:
        feature_size: I3D dim (2048). Magnitude channel sliced off.
        dict_size: number of normal-event dictionary slots (learnable here).
        reduction: Aggregate (MTN) channel reduction (paper 4).
        denormal_reduction: deNormal channel-gate reduction (paper 16).
        k: top-k snippets by magnitude (quantize_size // 10 = 3).
        dropout_rate: dropout in embeddings + classifier + top-k.
        margin / alpha: RTFM-style magnitude-separation loss.
        lambda_smooth / lambda_sparse / w_macro: loss weights.
    """

    def __init__(
        self,
        feature_size: int = 2048,
        dict_size: int = 64,
        reduction: int = 4,
        denormal_reduction: int = 16,
        k: int = 3,
        dropout_rate: float = 0.7,
        margin: float = 100.0,
        alpha: float = 0.0001,
        lambda_smooth: float = 8e-4,
        lambda_sparse: float = 8e-3,
        w_macro: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_size = feature_size
        self.dict_size = dict_size
        self.reduction = reduction
        self.denormal_reduction = denormal_reduction
        self.k = k
        self.dropout_rate = dropout_rate
        self.margin = margin
        self.alpha = alpha
        self.lambda_smooth = lambda_smooth
        self.lambda_sparse = lambda_sparse
        self.w_macro = w_macro
