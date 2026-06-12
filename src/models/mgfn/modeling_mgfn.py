from dataclasses import dataclass
from typing import Optional

import torch
from einops import rearrange
from torch import einsum, nn
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput

from ...loss import MGFNLoss, SparsityLoss, TemporalSmoothnessLoss
from ...registry import MODELS
from .configuration_mgfn import MGFNConfig


@dataclass
class MGFNModelOutput(ModelOutput):
    outputs: torch.FloatTensor = None


@dataclass
class MGFNVideoAnomalyDetectionOutput(ModelOutput):
    loss: torch.FloatTensor = None
    abnormal_scores: torch.FloatTensor = None
    normal_scores: torch.FloatTensor = None
    a_feat_magnitude: torch.FloatTensor = None
    n_feat_magnitude: torch.FloatTensor = None
    scores: torch.FloatTensor = None


class MGFNLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class MGFNFeedForward(nn.Module):
    def __init__(self, dim: int, repe: int = 4, dropout: float = 0.0):
        super().__init__()
        self.layer_norm = MGFNLayerNorm(dim)
        self.in_conv = nn.Conv1d(dim, dim * repe, 1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_conv = nn.Conv1d(dim * repe, dim, 1)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.in_conv(x)
        x = self.gelu(x)
        x = self.dropout(x)
        out = self.out_conv(x)
        return out


class MGFNFeatureAmplifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        init_dim = config.dims[0]
        self.mag_ratio = config.mag_ratio
        self.to_tokens = nn.Conv1d(
            config.channels,
            init_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.to_mag = nn.Conv1d(1, init_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # bs: batch_size
        # ncrops: number of crops in each video clip,
        #        `P` in paper. (Sultani, Chen, and Shah 2018)
        # t: clips
        # c: feature dimension, `C` in paper.
        bs, ncrops, t, c = x.size()
        x = x.view(bs * ncrops, t, c).permute(0, 2, 1)
        x_f, x_m = x[:, :2048, :], x[:, 2048:, :]
        x_f = self.to_tokens(x_f)
        x_m = self.to_mag(x_m)  # eq (1)
        x_f = x_f + self.mag_ratio * x_m  # eq (2)
        return x_f


class GlanceAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * heads
        self.norm = MGFNLayerNorm(dim)
        # construct a video clip-level transformer (VCT)
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(inner_dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)
        shape, h = x.shape, self.heads
        x = rearrange(x, "b c ... -> b c (...)")
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h d) n -> b h n d", h=h), (q, k, v))
        q = q * self.scale
        # correlate the different temporal clips
        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        # softmax normalization
        attn = sim.softmax(dim=-1)
        # weighted average of all clips in the long video containing
        # both normal and abnormal (if exists) ones
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b (h d) n", h=h)
        out = self.to_out(out)
        return out.view(*shape)


class GlanceBlock(nn.Module):
    """MGFN component to learn the global correlatoin among clips."""

    def __init__(self, config, dim: int, heads: int):
        super().__init__()
        # shortcut convolution
        self.scc = nn.Conv1d(dim, dim, 3, padding=1)
        # video clip-level transformer
        self.attention = GlanceAttention(
            dim=dim,
            heads=heads,
            dim_head=config.dim_head,
        )
        # additional feed-forward network
        # To further improve the model's representation capability.
        self.ffn = MGFNFeedForward(dim, repe=config.ff_repe, dropout=config.dropout)

    def forward(self, x):
        x = self.scc(x) + x
        x = self.attention(x) + x
        x = self.ffn(x) + x
        return x


class FocusAttention(nn.Module):
    """
    MHRAs (multi-head relation aggregators):
          Like self-attention, SAC allows each channel to get access
        to the nearby channels to learn the channel-wise correlation
        without any learnable weights
    """

    def __init__(self, dim: int, heads: int, dim_head: int, local_aggr_kernel: int):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = nn.BatchNorm1d(dim)
        self.to_v = nn.Conv1d(dim, inner_dim, 1, bias=False)
        self.rel_pos = nn.Conv1d(
            heads,
            heads,
            local_aggr_kernel,
            padding=local_aggr_kernel // 2,
            groups=heads,
        )
        self.to_out = nn.Conv1d(inner_dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)  # (b*crop,c,t)
        b, c, *_, h = *x.shape, self.heads
        v = self.to_v(x)  # (b*crop,c,t)
        v = rearrange(v, "b (c h) ... -> (b c) h ...", h=h)  # (b*ten*64,c/64,32)
        out = self.rel_pos(v)
        out = rearrange(out, "(b c) h ... -> b (c h) ...", b=b)
        return self.to_out(out)


class FocusBlock(nn.Module):
    """MGFN component to enhance the feature learning in each video clip."""

    def __init__(self, config, dim, heads):
        super().__init__()
        # shortcut convolution
        self.scc = nn.Conv1d(dim, dim, 3, padding=1)
        # self-attentional convolution (SAC)
        self.attention = FocusAttention(
            dim=dim,
            heads=heads,
            dim_head=config.dim_head,
            local_aggr_kernel=config.local_aggr_kernel,
        )
        # additional feed-forward network
        # To further improve the model's representation capability.
        self.ffn = MGFNFeedForward(dim, repe=config.ff_repe, dropout=config.dropout)

    def forward(self, x):
        x = self.scc(x) + x
        x = self.attention(x) + x
        x = self.ffn(x) + x
        return x


class MGFNIntermediate(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer_norm = MGFNLayerNorm(in_dim)
        self.conv = nn.Conv1d(in_dim, out_dim, 1, stride=1)

    def forward(self, x):
        x = self.layer_norm(x)
        return self.conv(x)


class MGFNPreTrainedModel(PreTrainedModel):
    config_class = MGFNConfig
    base_model_prefix = "backbone"

    def _init_weights(self, module):
        return NotImplementedError

    @property
    def dummy_inputs(self):
        # @TODO: fixed tensor
        # (batch_size, n_crops, n_clips, feature_dim)
        # ncrops == 10
        return torch.randn(32, 10, 32, 2049)


class MGFNModel(MGFNPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        depths = config.depths
        mgfn_types = config.mgfn_types

        self.amplifier = MGFNFeatureAmplifier(config)

        layers = []
        for ind, (depth, mgfn_type) in enumerate(zip(depths, mgfn_types)):
            stage_dim = config.dims[ind]
            heads = stage_dim // config.dim_head

            if mgfn_type == "gb":
                block_cls = GlanceBlock
            elif mgfn_type == "fb":
                block_cls = FocusBlock
            else:
                raise AttributeError(
                    "The type of mgfn block must be either `gb` or `fb`."
                )

            blocks = []
            for _ in range(depth):
                block = block_cls(config, dim=stage_dim, heads=heads)
                blocks.append(block)

            if ind != len(depths) - 1:
                intermediate = MGFNIntermediate(stage_dim, config.dims[ind + 1])
                blocks.append(intermediate)

            layers.append(nn.Sequential(*blocks))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> MGFNModelOutput:
        x = self.amplifier(x)
        out = self.layers(x)
        return MGFNModelOutput(outputs=out)


@MODELS.register("mgfn")
class MGFNForVideoAnomalyDetection(MGFNPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.k = config.k
        last_dim = config.dims[-1]

        self.backbone = MGFNModel(config)

        self.layer_norm = nn.LayerNorm(last_dim)
        self.fc = nn.Linear(last_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self._force_split = False
        self.dropout = nn.Dropout(config.dropout_rate)

    @property
    def force_split(self) -> bool:
        """
        Whether or not to forcibly separate inputs
        from abnormal and normal in the evaluation phase.
        """
        return self._force_split

    @force_split.setter
    def force_split(self, val: bool):
        self._force_split = val

    def magnitude_selection_and_score_prediction(
        self,
        features,
        scores,
        batch_size,
        ncrops,
    ):
        # features.shape == (bsz * ncrops, T, f)
        device = features.device
        _, t, f = features.size()

        # feat_magnitudes.shape : (bsz * ncrops, T, f) -> (bsz, T)
        feat_magnitudes = torch.norm(features, p=2, dim=2)
        feat_magnitudes = feat_magnitudes.view(batch_size, ncrops, -1).mean(dim=1)

        # scores.shape : (bsz * ncrops, T, 1) -> (bsz, T, 1)
        scores = scores.view(batch_size, ncrops, -1).mean(dim=1)
        scores = scores.unsqueeze(dim=2)

        # In the training loop, normal and abnormal are concated and inputted.
        # Since `batch_size` is doubled, slicing is performed here.
        # Make sure to enable data loader's `drop_last` attribute when training.
        if self.force_split or self.training:
            normal_features = features[0 : batch_size // 2 * ncrops]
            abnormal_features = features[batch_size // 2 * ncrops :]

            normal_scores = scores[0 : batch_size // 2]
            abnormal_scores = scores[batch_size // 2 :]

            n_feat_magnitudes = feat_magnitudes[0 : batch_size // 2]
            a_feat_magnitudes = feat_magnitudes[batch_size // 2 :]
        # for inference
        else:
            normal_features = abnormal_features = features
            normal_scores = abnormal_scores = scores
            n_feat_magnitudes = a_feat_magnitudes = feat_magnitudes

        n_size = n_feat_magnitudes.shape[0]

        def magnitude_selection(feat_magnitudes, features):
            select_idx = torch.ones_like(feat_magnitudes, device=device)
            select_idx = self.dropout(select_idx)

            feat_magnitudes_drop = feat_magnitudes * select_idx
            idx = torch.topk(feat_magnitudes_drop, self.k, dim=1)[1]
            idx_feat = idx.unsqueeze(dim=2).expand([-1, -1, features.shape[2]])

            features = features.view(n_size, ncrops, t, f)
            features = features.permute(1, 0, 2, 3)

            total_select_feature = torch.zeros(0, device=device)
            for feature in features:
                feat_select = torch.gather(feature, 1, idx_feat)
                total_select_feature = torch.cat([total_select_feature, feat_select])

            return idx, total_select_feature

        def score_prediction(idx, scores):
            idx_score = idx.unsqueeze(dim=2).expand([-1, -1, scores.shape[2]])
            score = torch.mean(torch.gather(scores, 1, idx_score), dim=1)
            return score

        idx_abn, abn_feamagnitude = magnitude_selection(
            a_feat_magnitudes, abnormal_features
        )
        score_abnormal = score_prediction(idx_abn, abnormal_scores)

        idx_normal, nor_feamagnitude = magnitude_selection(
            n_feat_magnitudes, normal_features
        )
        score_normal = score_prediction(idx_normal, normal_scores)

        return score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores

    def forward(
        self,
        video: torch.FloatTensor,
        abnormal_labels: Optional[torch.FloatTensor] = None,
        normal_labels: Optional[torch.FloatTensor] = None,
    ) -> MGFNVideoAnomalyDetectionOutput:
        # video.shape == (
        #     batch_size,       -> Half were `normal` samples, and half were `abnormal` samples.
        #     number of crops,  -> 10, since using `TenCrop` in feature extraction
        #     segment_length,   -> T, the number used in original repository (Is equal to n_clips)
        #                          In training phase, 32.
        #     feature_dimension -> 2049, feature dimension(2048) + magnitude(1)
        # )
        bs, ncrops = video.size()[:2]
        # x_f.shape == (bsz * ncrops, last_hidden_dim, nclips)
        x_f = self.backbone(video).outputs
        x_f = x_f.permute(0, 2, 1)
        # x.shape == (bsz * ncrops, nclips, last_hidden_dim)
        x = self.layer_norm(x_f)
        # scores.shape == (bsz * ncrops, nclips, 1)
        scores = self.sigmoid(self.fc(x))

        (
            abnormal_scores,  # (bsz // 2, 1)
            normal_scores,  # (bsz // 2, 1)
            a_feat_magnitude,  # (bsz // 2 * ncrops, topk, last_hidden_dim)
            n_feat_magnitude,  # (bsz // 2 * ncrops, topk, last_hidden_dim)
            scores,  # (bsz, nclips, 1)
        ) = self.magnitude_selection_and_score_prediction(x, scores, bs, ncrops)

        loss = None
        if abnormal_labels is not None and normal_labels is not None:
            loss_smooth = TemporalSmoothnessLoss()(scores)
            loss_sparsity = SparsityLoss()(scores[: bs // 2, :, :].view(-1))
            loss_mgfn = MGFNLoss()(
                abnormal_scores=abnormal_scores,
                normal_scores=normal_scores,
                abnormal_labels=abnormal_labels,
                normal_labels=normal_labels,
                a_feat_magnitude=a_feat_magnitude,
                n_feat_magnitude=n_feat_magnitude,
            )
            loss = loss_mgfn + loss_smooth + loss_sparsity

        return MGFNVideoAnomalyDetectionOutput(
            loss=loss,
            abnormal_scores=abnormal_scores,
            normal_scores=normal_scores,
            a_feat_magnitude=a_feat_magnitude,
            n_feat_magnitude=n_feat_magnitude,
            scores=scores,
        )
