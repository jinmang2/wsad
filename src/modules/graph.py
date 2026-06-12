"""Graph primitives for the LGT-Adapter (VadCLIP and graph-based methods).

Faithful (batched) port of VadCLIP's ``GraphConvolution`` + the two adjacency
builders it uses:
  - ``similarity_adj``: cosine-similarity graph, thresholded at 0.7 then softmax
    (VadCLIP ``adj4``) — the *global/semantic* branch.
  - ``distance_adj``: ``exp(-|i-j| / e)`` temporal-distance graph (VadCLIP
    ``DistanceAdj``) — the *local/temporal* branch.

Ref: https://github.com/nwpu-zxr/VadCLIP/blob/main/src/utils/layers.py
"""

import torch
import torch.nn.functional as F
from torch import nn


class GraphConvolution(nn.Module):
    """Batched GCN layer (Kipf & Welling) with optional residual.

    residual: ``False`` -> none; ``in==out`` -> identity; else a Conv1d(k=5)
    projection (matches VadCLIP).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        residual: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0.1)

        self.use_residual = residual
        if not residual:
            self.residual = None
        elif in_features == out_features:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv1d(
                in_features, out_features, kernel_size=5, padding=2
            )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in), adj: (B, T, T)
        support = x.matmul(self.weight)  # (B, T, out)
        output = adj.matmul(support)
        if self.bias is not None:
            output = output + self.bias

        if not self.use_residual:
            return output
        if self.in_features != self.out_features:
            res = self.residual(x.permute(0, 2, 1)).permute(0, 2, 1)
            return output + res
        return output + self.residual(x)


def similarity_adj(x: torch.Tensor, threshold: float = 0.7) -> torch.Tensor:
    """Cosine-similarity adjacency, thresholded then row-softmaxed (VadCLIP adj4)."""
    sim = x.matmul(x.permute(0, 2, 1))  # (B, T, T)
    norm = torch.norm(x, p=2, dim=2, keepdim=True)
    sim = sim / (norm.matmul(norm.permute(0, 2, 1)) + 1e-20)
    sim = F.threshold(sim, threshold, 0.0)
    return F.softmax(sim, dim=-1)


def distance_adj(t: int, batch_size: int, device) -> torch.Tensor:
    """Temporal-distance adjacency ``exp(-|i-j| / e)`` (VadCLIP DistanceAdj)."""
    arith = torch.arange(t, device=device).float()
    dist = (arith[:, None] - arith[None, :]).abs()
    adj = torch.exp(-dist / torch.exp(torch.tensor(1.0, device=device)))
    return adj.unsqueeze(0).expand(batch_size, t, t)
