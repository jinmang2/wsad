"""Config for the PEL4VAD port (Prompt-Enhanced Learning, TIP 2024)."""

from transformers import PretrainedConfig


class PEL4VADConfig(PretrainedConfig):
    """Defaults are the official UCF-Crime settings (`configs.py`, `dataset in ['ucf']`).

    Args:
        feature_size: input feature dim. The official code is written for the **1024-d**
            DeepMIL I3D (`cfg.feat_dim = 1024`), which is our `i3d_1024_seg200` variant.
        hidden_size / out_dim / num_heads: TCA and encoder widths (`hid_dim`, `out_dim`,
            `head_num`).
        win_size: width of the TCA local-attention window (UCF: 9).
        gamma / bias: initial values of the learnable distance-decay parameters
            ``exp(-|w·d² - b|)`` (UCF: 0.6 / 0.2).
        norm: apply the power+L2 normalization to the fused TCA output (UCF: True).
        t_step: temporal width of the causal classifier convolution (UCF: 9).
        temp: initial CLIP-style logit temperature (UCF: 0.09).
        lamda: weight of the prompt-alignment KL term (UCF: 1).
        prompt_dim: dim of the class-prompt embeddings (CLIP text = 512).
        num_classes: 14 = Normal + 13 UCF-Crime anomaly classes.
    """

    model_type = "pel4vad"

    def __init__(
        self,
        feature_size: int = 1024,
        hidden_size: int = 128,
        out_dim: int = 300,
        num_heads: int = 1,
        win_size: int = 9,
        gamma: float = 0.6,
        bias: float = 0.2,
        norm: bool = True,
        t_step: int = 9,
        dropout_rate: float = 0.1,
        temp: float = 0.09,
        lamda: float = 1.0,
        prompt_dim: int = 512,
        num_classes: int = 14,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.win_size = win_size
        self.gamma = gamma
        self.bias = bias
        self.norm = norm
        self.t_step = t_step
        self.dropout_rate = dropout_rate
        self.temp = temp
        self.lamda = lamda
        self.prompt_dim = prompt_dim
        self.num_classes = num_classes
