"""Metric interventions used by the selection/capture experiment."""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
Array = NDArray[np.float64]


def metric_product(residual: Array, slow: Array, alpha: float) -> Array:
    return residual - (1.0 - alpha) * slow * (slow * residual).sum(axis=1)[:, None]


def adverse_direction(y: Array, residual: Array, anchor: Array) -> Array:
    """Minimize the weakest signed coordinate's drift over one-slow Grams."""
    coordinate = (anchor * y).argmin(axis=1)
    unit = np.zeros_like(y)
    unit[np.arange(len(y)), coordinate] = anchor[np.arange(len(y)), coordinate]
    norm = np.linalg.norm(residual, axis=1, keepdims=True)
    direction = unit + residual / np.maximum(norm, np.finfo(float).tiny)
    length = np.linalg.norm(direction, axis=1, keepdims=True)
    degenerate = length[:, 0] < 1e-12
    direction[degenerate] = 0.0
    direction[np.flatnonzero(degenerate), (coordinate[degenerate] + 1) % y.shape[1]] = 1.0
    return direction / np.linalg.norm(direction, axis=1, keepdims=True)
