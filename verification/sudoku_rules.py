"""Legal 4x4 Sudoku grids and the exact posterior over them."""

from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor


@lru_cache(maxsize=1)
def complete_grids() -> NDArray[np.int64]:
    """The 288 legal grids, each as 16 class indices in row-major order."""
    rows = tuple(itertools.permutations(range(4)))
    prefixes: list[tuple[tuple[int, ...], ...]] = [()]
    for depth in range(4):
        children = []
        for prefix in prefixes:
            for row in rows:
                if any(row[col] in [old[col] for old in prefix] for col in range(4)):
                    continue
                if depth % 2 and any(set(row[k:k + 2]) & set(prefix[-1][k:k + 2])
                                     for k in (0, 2)):
                    continue
                children.append(prefix + (row,))
        prefixes = children
    grids = np.asarray(prefixes, dtype=np.int64).reshape(-1, 16)
    if grids.shape != (288, 16):
        raise AssertionError(f"Unexpected number of legal grids: {grids.shape}")
    return grids


def legal_grids(hard: Tensor) -> Tensor:
    """Whether each row of 16 predicted classes obeys the row, column and block rules."""
    grids = hard.reshape(-1, 4, 4)
    target = torch.arange(4, device=hard.device)
    row_ok = (grids.sort(dim=2).values == target).all(dim=(1, 2))
    col_ok = (grids.sort(dim=1).values == target[None, :, None]).all(dim=(1, 2))
    blocks = grids.reshape(-1, 2, 2, 2, 2).permute(0, 1, 3, 2, 4).reshape(-1, 4, 4)
    block_ok = (blocks.sort(dim=2).values == target).all(dim=(1, 2))
    return row_ok & col_ok & block_ok


def candidates(revealed: Tensor, has_revealed_cell: Tensor) -> tuple[Tensor, Tensor]:
    """The legal grids of every board, and which of them agree with its revealed top-left class."""
    grids = torch.tensor(complete_grids(), device=revealed.device)
    codes = grids[None].expand(len(revealed), 288, 16)
    return codes, (grids[None, :, 0] == revealed[:, None]) | ~has_revealed_cell[:, None]


def posterior(log_p: Tensor, codes: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Posterior marginals of every cell (the soft labels) and the log evidence of every board.

    `log_p` holds the log class probabilities of the 16 cells of each board. The posterior is over the valid
    grids of the board, with a uniform prior and unit temperature.
    """
    batch, count, positions = codes.shape
    selected = log_p[:, None].expand(batch, count, positions, 4).gather(3, codes[:, :, :, None])
    scores = selected.squeeze(-1).sum(dim=2).masked_fill(~valid, -torch.inf)
    log_evidence = scores.logsumexp(dim=1)
    weights = scores.softmax(dim=1)
    q = torch.zeros_like(log_p).scatter_add(
        2, codes.transpose(1, 2), weights[:, None, :].expand(batch, positions, count))
    return q, log_evidence


def implementation_check(device: str = "cpu") -> dict[str, float]:
    """The gradient of the negative log evidence equals that of the cross-entropy against the soft labels."""
    rng = np.random.default_rng(2026110977)
    logits = torch.tensor(rng.normal(size=(64, 16, 4)), device=device, dtype=torch.float64, requires_grad=True)
    revealed = torch.arange(64, device=device) % 4
    has_revealed_cell = torch.arange(64, device=device) % 3 > 0
    codes, valid = candidates(revealed, has_revealed_cell)
    if not legal_grids(codes[0]).all() or sorted(valid.sum(dim=1).unique().tolist()) != [72, 288]:
        raise AssertionError("Unexpected candidate grids")
    log_p = logits.log_softmax(dim=2)
    q, log_evidence = posterior(log_p, codes, valid)
    exact = torch.autograd.grad(-log_evidence.sum(), logits, retain_graph=True)[0]
    soft = torch.autograd.grad(-(q.detach() * log_p).sum(), logits)[0]
    result = {"soft_label_gradient_error": float((exact - soft).abs().max()),
              "soft_label_normalization_error": float((q.sum(dim=2) - 1).abs().max())}
    if max(result.values()) > 1e-12:
        raise AssertionError(result)
    return result
