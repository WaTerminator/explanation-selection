"""Train a genuinely rank-deficient CIFAR count learner in parameter space."""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import product
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, logsumexp

from run_cifar_llp import allocate, candidates, digest, quantities

Array = NDArray[np.float64]
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/restricted.json"


def projection(features: Array) -> tuple[Array, Array, Array]:
    normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
    center = normalized.mean(axis=0)
    centered = normalized - center
    values, basis = np.linalg.eigh(centered.T @ centered / len(centered))
    values, basis = values[-32:][::-1], basis[:, -32:][:, ::-1]
    signs = np.sign(basis[np.abs(basis).argmax(axis=0), np.arange(32)])
    basis = basis * signs
    if values[-1] <= 0 or not np.isfinite(basis).all():
        raise ValueError("Fixed training-pool projection has invalid retained variance")
    return center, basis, np.sqrt(values)


def design(features: Array, center: Array, basis: Array, scale: Array) -> Array:
    normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
    projected = (normalized - center) @ basis / scale
    return np.column_stack((projected, np.ones(len(projected)))) / np.sqrt(33.0)


def adverse_diagonal(x: Array, weights: Array, gradient: Array, anchor: Array) -> Array:
    logits = np.einsum("dnp,dhmp->dhmn", x, weights)
    weakest = np.argmin(anchor * logits, axis=-1)
    rows = x[np.arange(len(x))[:, None, None], weakest]
    signs = np.take_along_axis(anchor, weakest[..., None], axis=-1)
    contribution = signs * rows * gradient
    return np.where(contribution >= 0.0, 0.5, 2.0)


def implementation_check() -> dict[str, float]:
    """Check native posterior, direct parameter updates, and diagonal adversary."""
    rng = np.random.default_rng(2026091401)
    x = rng.normal(size=(1, 8, 3))
    weights = rng.normal(size=(1, 1, 1, 3))
    counts = np.array([1, 3])
    codes, valid = candidates(counts)
    y = np.einsum("dnp,dhmp->dhmn", x, weights)
    state = quantities(y, codes, valid)
    words = np.array([v for v in product((0.0, 1.0), repeat=8)
                      if sum(v[:4]) == 1 and sum(v[4:]) == 3])
    score = words @ y.ravel() - np.logaddexp(0.0, y).sum()
    log_evidence = logsumexp(score)
    residual = 2.0 * (np.exp(score - log_evidence) @ words - expit(y.ravel()))
    gradient = np.einsum("dhmn,dnp->dhmp", state.residual, x)
    diagonal = np.array([0.5, 1.0, 2.0])
    next_w = weights + 0.01 * diagonal * gradient
    sample_change = 0.01 * (x[0] * diagonal) @ x[0].T @ residual
    anchor = np.where(y > 0, 1.0, -1.0)
    chosen = adverse_diagonal(x, weights, gradient, anchor).ravel()
    index = int(np.argmin(anchor * y))
    contribution = anchor.ravel()[index] * x[0, index] * gradient.ravel()
    vertex_minimum = min(np.array(v) @ contribution for v in product((0.5, 2.0), repeat=3))
    result = {
        "global_enumeration_residual_error": float(np.max(np.abs(residual - state.residual.ravel()))),
        "global_enumeration_evidence_error": float(abs(log_evidence - state.log_bag_evidence.sum())),
        "direct_parameter_update_error": float(np.max(np.abs((np.einsum("dnp,dhmp->dhmn", x, next_w) - y).ravel() - sample_change))),
        "adverse_diagonal_vertex_optimality_error": float(abs(chosen @ contribution - vertex_minimum)),
    }
    if max(result.values()) > 1e-10 or np.linalg.matrix_rank(x[0] @ x[0].T) != 3:
        raise AssertionError(result)
    return result


def run(features_path: Path, protocol_path: Path, output: Path, phase: str) -> None:
    protocol: dict[str, Any] = json.loads(protocol_path.read_text())
    if output.exists():
        raise FileExistsError(f"Output must be a fresh directory: {output}")
    if digest(features_path) != protocol["feature_sha256"]:
        raise ValueError("Feature cache identity differs from the fixed protocol")
    output.mkdir(parents=True)
    start = time.monotonic()
    source_hash = digest(Path(__file__))
    data = np.load(features_path)
    center, basis, scale = projection(data["train_features"].astype(np.float64))
    pool_x = design(data["train_features"].astype(np.float64), center, basis, scale)
    test_x = design(data["test_features"].astype(np.float64), center, basis, scale)
    original_counts = np.array([1, 2, 3, 1, 2, 3])
    excluded = np.concatenate(allocate(data["train_labels"], original_counts, 9, 2026090601))
    eligible = np.setdiff1d(np.arange(len(pool_x)), excluded)
    counts = np.array(protocol["bag_counts"], dtype=np.int64)
    allocations = [eligible[indices] for indices in allocate(data["train_labels"][eligible], counts, 6, protocol["data_seed"])]
    draws = np.array(protocol[f"{phase}_draws"], dtype=np.int64)
    selected = np.stack([allocations[draw] for draw in draws])
    x = pool_x[selected]
    heads, dimension = protocol["heads_per_draw"], protocol["parameter_dimension"]
    singular = np.linalg.svd(x, compute_uv=False)
    if x.shape[1:] != (96, 33) or np.any(singular[:, -1] <= 1e-10 * singular[:, 0]):
        raise ValueError("The fixed N=96, d=33 design is invalid; no replacement is allowed")
    eta = protocol["step_fraction"] * 4 / (protocol["metric_max"] * singular[:, 0] ** 2 * 5)
    metrics = np.stack((np.ones(dimension), np.geomspace(0.5, 2, dimension), np.geomspace(2, 0.5, dimension)))
    initial = np.stack([np.stack([np.random.default_rng(protocol["head_seed_base"] + 100 * int(draw) + head)
                                 .uniform(-1 / np.sqrt(dimension), 1 / np.sqrt(dimension), dimension)
                                 for head in range(heads)]) for draw in draws])
    weights = np.repeat(initial[:, :, None, :], 3, axis=2)
    codes, valid = candidates(counts)
    horizon = protocol["baseline_horizon"]
    stored = np.lib.format.open_memmap(output / "weights.npy", mode="w+", dtype=np.float64,
                                      shape=(horizon + 1, *weights.shape))
    capture_steps = np.full(weights.shape[:-1], -1, dtype=np.int64)
    capture_weights = np.full_like(weights, np.nan)
    previous = np.full(weights.shape[:-1], -np.inf)
    smallest_increment = 0.0
    for step in range(horizon + 1):
        stored[step] = weights
        logits = np.einsum("dnp,dhmp->dhmn", x, weights)
        state = quantities(logits, codes, valid)
        evidence = state.log_bag_evidence.sum(axis=-1)
        if not np.isfinite(weights).all() or not np.isfinite(evidence).all():
            raise FloatingPointError(f"Nonfinite baseline state at step {step}")
        smallest_increment = min(smallest_increment, float((evidence - previous).min()))
        entered = (capture_steps < 0) & (evidence > -np.log(2.0))
        capture_steps[entered] = step
        capture_weights[entered] = weights[entered]
        previous = evidence
        if step < horizon:
            gradient = np.einsum("dhmn,dnp->dhmp", state.residual, x)
            weights = weights + eta[:, None, None, None] * metrics * gradient
    stored.flush()
    captured = capture_steps >= 0
    print(f"{phase}: {capture_steps.size} direct-parameter baselines, {captured.sum()} captures", flush=True)
    current = np.repeat(np.where(captured[..., None], capture_weights, 0.0)[..., None, :], 3, axis=-2)
    anchor = np.where(np.einsum("dnp,dhmp->dhmn", x, np.where(captured[..., None], capture_weights, 0.0)) > 0, 1.0, -1.0)
    continuation_horizon = protocol["continuation_horizon"]
    continued = np.lib.format.open_memmap(output / "continuation_weights.npy", mode="w+", dtype=np.float64,
                                         shape=(continuation_horizon + 1, *current.shape))
    for step in range(continuation_horizon + 1):
        continued[step] = np.where(captured[..., None, None], current, np.nan)
        if step == continuation_horizon:
            break
        logits = np.einsum("dnp,dhmcp->dhmcn", x, current)
        state = quantities(logits, codes, valid)
        gradient = np.einsum("dhmcn,dnp->dhmcp", state.residual, x)
        changes = np.stack((gradient[..., 0, :],
                            metrics[[2, 2, 1]] * gradient[..., 1, :],
                            adverse_diagonal(x, current[..., 2, :], gradient[..., 2, :], anchor) * gradient[..., 2, :]), axis=-2)
        current = current + eta[:, None, None, None, None] * changes
        if not np.isfinite(current).all():
            raise FloatingPointError(f"Nonfinite continuation at step {step + 1}")
    continued.flush()
    np.savez(output / "model.npz", draw_ids=draws, training_design=x, test_design=test_x,
             train_indices=data["train_indices"][selected], train_labels=(data["train_labels"][selected] == 3),
             test_indices=data["test_indices"], test_labels=(data["test_labels"] == 3),
             excluded_original_indices=data["train_indices"][excluded],
             projection_center=center, projection_basis=basis, projection_scale=scale,
             parameter_diagonals=metrics, initial_weights=initial, eta=eta,
             bag_counts=counts, capture_steps=capture_steps, singular_values=singular)
    if smallest_increment < -1e-10 or source_hash != digest(Path(__file__)):
        raise AssertionError("Baseline ascent or unchanged execution source check failed")
    record = {"phase": phase, "protocol_sha256": digest(protocol_path),
              "feature_sha256": digest(features_path), "source_sha256": source_hash,
              "posterior_source_sha256": digest(ROOT / "verification/run_cifar_llp.py"),
              "baseline_trajectories": int(capture_steps.size), "captured_baselines": int(captured.sum()),
              "executed_continuations": int(captured.sum()) * 3,
              "untriggered_continuation_slots": int((~captured).sum()) * 3,
              "baseline_min_log_evidence_increment": smallest_increment,
              "implementation_checks": implementation_check(), "runtime_seconds": time.monotonic() - start,
              "files": {name: digest(output / name) for name in ("model.npz", "weights.npy", "continuation_weights.npy")}}
    (output / "run_record.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: record[key] for key in ("phase", "baseline_trajectories", "captured_baselines", "runtime_seconds")}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase", choices=("pilot", "evaluation"))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(implementation_check(), indent=2))
    else:
        if args.features is None or args.output is None or args.phase is None:
            parser.error("--features, --output and --phase are required")
        run(args.features, args.protocol, args.output, args.phase)


if __name__ == "__main__":
    main()
