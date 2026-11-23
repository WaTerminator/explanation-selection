#!/usr/bin/env python3
"""Finite algebra checks for exact capture in a rank-deficient representation."""

from __future__ import annotations

import json
from itertools import product

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, logsumexp


Array = NDArray[np.float64]


def evidence_terms(y: Array, bits: Array) -> tuple[float, Array, Array]:
    """Enumerate the global code set and differentiate its log evidence."""
    scores = bits @ y
    log_evidence = float(logsumexp(scores) - np.logaddexp(0, y).sum())
    posterior = np.exp(scores - logsumexp(scores))
    marginals = posterior @ bits
    # Differentiate conditional slices to avoid subtraction near saturation.
    residual = np.empty(len(y))
    for i in range(len(y)):
        others = np.arange(len(y)) != i
        products = np.exp(bits[:, others] @ y[others]
                          - np.logaddexp(0, y[others]).sum())
        a = products[bits[:, i] == 1].sum()
        b = products[bits[:, i] == 0].sum()
        residual[i] = 2 * expit(y[i]) * expit(-y[i]) * (a - b) / np.exp(log_evidence)
    covariance = (bits.T * posterior) @ bits - np.outer(marginals, marginals)
    hessian = 2 * covariance - 2 * np.diag(expit(y) * expit(-y))
    return log_evidence, residual, hessian


def main() -> None:
    n, dimension, steps = 8, 3, 256
    all_bits = np.array(list(product((0.0, 1.0), repeat=n)))
    minimum_energy_ratio = float("inf")
    maximum_hessian_norm = 0.0
    for seed in range(12):
        rng = np.random.default_rng(20260906 + seed)
        x = rng.normal(size=(n, dimension))
        witness = rng.normal(size=dimension)
        witness *= np.log(4 * n) / np.min(np.abs(x @ witness))
        z = np.sign(x @ witness)
        counts = ((z.reshape(2, 4) + 1) / 2).sum(axis=1)
        bits = all_bits[np.all(all_bits.reshape(-1, 2, 4).sum(axis=2) == counts, axis=1)]
        assert len(bits) >= 2, "Synthetic candidate set must be nontrivial"
        assert np.linalg.matrix_rank(x) == dimension < n
        kappa = 2 * np.linalg.norm(x, ord=2) ** 2
        eta = 3.2 / (kappa * 5)
        beta = 1 - eta * kappa * 5 / 4
        log_c, _, _ = evidence_terms(x @ witness, bits)
        c = np.exp(log_c)
        assert c > 0.5
        mu = np.min(np.abs(x @ witness))
        d_c = 2 * c**2 * (2 * c - 1)
        energy_constant = 0.5 * mu**2 / np.dot(witness, witness)
        gamma = eta * beta * energy_constant * d_c**2 / 4
        initial_v = -2 * log_c
        w = witness.copy()
        for t in range(steps):
            y = x @ w
            log_e, residual, hessian = evidence_terms(y, bits)
            radius = np.exp(-z * y).sum()
            v = -2 * log_e
            assert np.array_equal(np.sign(y), z)
            assert log_e >= log_c - 2e-12
            assert np.all(z * residual >= d_c * np.exp(-z * y) - 2e-12)
            assert 2 * c * radius / n <= v + 2e-12 <= 2 * radius + 4e-12
            assert radius <= n / (2 * c) / (1 / initial_v + gamma * t) + 2e-12
            maximum_hessian_norm = max(maximum_hessian_norm, float(np.linalg.norm(hessian, ord=2)))
            assert np.linalg.norm(hessian, ord=2) <= 2.5 + 2e-12
            assert np.max(np.abs(hessian[:4, 4:])) < 2e-12
            gradient = x.T @ residual
            diagonal = 0.5 + 1.5 * expit(gradient)
            direction = diagonal * gradient
            energy = float(gradient @ direction)
            lower = energy_constant * np.linalg.norm(residual, ord=1) ** 2
            assert energy >= lower - 2e-12
            minimum_energy_ratio = min(minimum_energy_ratio, energy / lower)
            for fraction in (0.125, 0.5, 1.0):
                later_log_e, _, _ = evidence_terms(x @ (w + fraction * eta * direction), bits)
                assert 2 * (later_log_e - log_e) >= fraction * eta * beta * energy - 2e-12
            w = w + eta * direction
    print(json.dumps({"systems": 12, "dimension": dimension, "images": n,
                      "bag_size": 4, "hessian_bound": 2.5, "segment_step_quantity": 3.2,
                      "updates_per_system": steps, "passed": True,
                      "minimum_parameter_energy_ratio": minimum_energy_ratio,
                      "maximum_observed_hessian_norm": maximum_hessian_norm,
                      "scope": "finite algebra checks; not a proof or image-learning experiment"}, indent=2))


if __name__ == "__main__":
    main()
