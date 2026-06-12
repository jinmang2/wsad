"""Perturbed Top-K HardAttention — the actual "TSA" of CLIP-TSA (ICIP'23).

Faithful port of the official ``utils/hard_attention.py`` (joos2010kj/CLIP-TSA).
This *is* the paper's Temporal Self-Attention: a differentiable top-k snippet
selector (perturbed-maximum trick, Cordonnier et al. / Berthet et al.). Scores per
snippet -> Gaussian-perturbed top-k indicator -> soft-aggregate selected snippets.

Earlier this repo approximated "TSA" with a plain Transformer; that dropped the
defining mechanism. This restores it.
"""

import torch
import torch.nn as nn


class PerturbedTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 1000, sigma: float = 0.05):
        # x: (b, d) snippet scores; k: number of snippets to select
        b, d = x.shape
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d), device=x.device)
        perturbed_x = x[:, None, :] + noise * sigma  # (b, num_samples, d)

        k = max(1, min(int(k), d))
        topk = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices = torch.sort(topk.indices, dim=-1).values  # (b, num_samples, k)

        perturbed_output = torch.nn.functional.one_hot(indices, num_classes=d).float()
        indicators = perturbed_output.mean(dim=1)  # (b, k, d)

        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma
        ctx.save_for_backward(perturbed_output, noise)
        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None] * 4)
        perturbed_output, noise = ctx.saved_tensors
        grad_expected = torch.einsum("bnkd,bne->bkde", perturbed_output, noise)
        grad_expected = grad_expected / (ctx.num_samples * ctx.sigma)
        grad_input = torch.einsum("bkde,bke->bd", grad_expected, grad_output)
        return (grad_input,) + tuple([None] * 3)


class _ScorerMLP(nn.Module):
    def __init__(self, input_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))


class HardAttention(nn.Module):
    """Score snippets, select a differentiable top-k, soft-aggregate (official)."""

    def __init__(self, k: float = 1.0, num_samples: int = 100, input_dim: int = 512):
        super().__init__()
        self.scorer = _ScorerMLP(input_dim)
        self.k_ratio = k  # selected count = round(k_ratio * T)
        self.num_samples = num_samples

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (b, t, d)
        scores = self.scorer(inputs)  # (b, t, 1)
        b, t, _ = scores.shape
        k = max(1, int(self.k_ratio * t)) if self.k_ratio <= 1 else int(self.k_ratio)
        topk = PerturbedTopKFunction.apply(
            scores.squeeze(-1), k, self.num_samples, 0.05
        )  # (b, k, t)
        out = topk.unsqueeze(-1) * inputs.unsqueeze(1)  # (b, k, t, d)
        out = torch.sum(out, dim=1)  # (b, t, d)
        return out
