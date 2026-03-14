from __future__ import annotations

import torch


def kl_unit_gaussian(mu: torch.Tensor, log_sigma: torch.Tensor, sigma: torch.Tensor | None = None) -> torch.Tensor:
    if sigma is None:
        sigma = torch.exp(log_sigma)
    return -0.5 * (1.0 + 2.0 * log_sigma - mu.square() - sigma.square())


def kl_gaussian_gaussian(
    mu1: torch.Tensor,
    logsig1: torch.Tensor,
    sig1: torch.Tensor,
    mu2: torch.Tensor,
    logsig2: torch.Tensor,
    sig2: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (2.0 * logsig2 - 2.0 * logsig1 + (sig1.square() + (mu1 - mu2).square()) / sig2.square() - 1.0)


def stochastic_binarize(images: torch.Tensor) -> torch.Tensor:
    return torch.bernoulli(images)
