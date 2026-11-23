"""Shared pieces of the jointly trained experiments: the design row, the encoder pass and the count objective."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def state(heads: Tensor, x: Tensor, codes: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Count loss, its gradient in the heads, the logits and the log evidence of each bag."""
    logits = (heads @ x.T).reshape(len(heads), len(codes), 4)
    scores = torch.einsum("hbi,bci->hbc", logits, codes).masked_fill(~valid, -torch.inf)
    log_bag = torch.logsumexp(scores, dim=2) - torch.nn.functional.softplus(logits).sum(dim=2)
    targets = torch.einsum("hbc,bci->hbi", torch.softmax(scores, dim=2), codes)
    gradient = ((torch.sigmoid(logits) - targets) / len(codes)).reshape(len(heads), -1) @ x
    return -log_bag.mean(dim=1), gradient, logits.flatten(1), log_bag


def normalized_design(raw: Tensor) -> Tensor:
    values = raw.double()
    norms = torch.linalg.vector_norm(values, dim=1, keepdim=True)
    # A learned encoder can output zero; preserve it instead of dividing by zero.
    values = values / torch.where(norms > 0, norms, torch.ones_like(norms))
    return torch.cat((values, torch.ones_like(values[:, :1])), dim=1) / math.sqrt(2)


def encode(encoder: nn.Module, images: Tensor) -> Tensor:
    with torch.no_grad():
        return normalized_design(torch.cat([encoder(batch) for batch in images.split(128)]))
