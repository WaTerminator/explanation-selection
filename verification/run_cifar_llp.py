"""Exact count-posterior learning and capture in a shared frozen image head."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from itertools import product
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.linalg import helmert
from scipy.special import expit, logsumexp

from run_evidence_capture import adverse_direction, metric_product


Array = NDArray[np.float64]
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/selection.json"


class CountState(NamedTuple):
    residual: Array
    log_bag_evidence: Array


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def candidates(counts: NDArray[np.int64]) -> tuple[Array, NDArray[np.bool_]]:
    words = np.array(list(product((0.0, 1.0), repeat=4)))
    blocks = [words[words.sum(axis=1) == count] for count in counts]
    width = max(map(len, blocks))
    codes = np.zeros((len(counts), width, 4))
    valid = np.zeros((len(counts), width), dtype=bool)
    for bag, block in enumerate(blocks):
        codes[bag, :len(block)] = block
        valid[bag, :len(block)] = True
    return codes, valid


def quantities(y: Array, codes: Array, valid: NDArray[np.bool_]) -> CountState:
    grouped = y.reshape(*y.shape[:-1], len(codes), 4)
    scores = np.einsum("...bi,bci->...bc", grouped, codes)
    scores -= np.logaddexp(0.0, grouped).sum(axis=-1)[..., None]
    scores = np.where(valid, scores, -np.inf)
    log_bag = logsumexp(scores, axis=-1)
    posterior = np.exp(scores - log_bag[..., None])
    target = np.einsum("...bc,bci->...bi", posterior, codes).reshape(y.shape)
    return CountState(2.0 * (target - expit(y)), log_bag)


def design(features: NDArray[np.float32]) -> Array:
    values = features.astype(np.float64)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norm == 0) or not np.isfinite(values).all():
        raise ValueError("Feature rows must be finite and nonzero")
    return np.column_stack((values / norm, np.ones(len(values)))) / np.sqrt(2.0)


def metric_family(x: Array, draw: int, seed: int) -> tuple[Array, Array, dict[str, float]]:
    n, width = x.shape
    bags = n // 4
    contrasts = np.kron(np.eye(bags), helmert(4, full=False).T)
    constant = np.kron(np.eye(bags), np.ones((4, 1)) / 2.0)
    u, singular, vt = np.linalg.svd(x, full_matrices=False)
    if singular[-1] / singular[0] <= 1e-8:
        raise ValueError(f"Draw {draw}: full-row-rank condition fails; no replacement allowed")
    inverse = (vt.T / singular) @ u.T
    raw = x @ x.T
    _, feature_basis = np.linalg.eigh(contrasts.T @ raw @ contrasts)
    feature_basis = feature_basis[:, ::-1]
    spectrum = np.geomspace(4.0, 0.25, n - bags)
    rng = np.random.default_rng(seed + draw)
    bases = [feature_basis, feature_basis[:, ::-1]]
    bases += [np.linalg.qr(rng.normal(size=(n - bags, n - bags)))[0] for _ in range(2)]
    matrices = [contrasts @ ((q * spectrum) @ q.T) @ contrasts.T + constant @ constant.T
                for q in bases]
    matrices += [np.eye(n) * (spectrum.sum() + bags) / n,
                 raw * (4.0 / np.linalg.eigvalsh(raw)[-1])]
    metrics = np.stack(matrices)
    expected = np.sort(np.r_[spectrum, np.ones(bags)])
    full_error = max(float(np.max(np.abs(np.linalg.eigvalsh(k) - expected)))
                     for k in metrics[:4])
    active_error = max(float(np.max(np.abs(np.linalg.eigvalsh(contrasts.T @ k @ contrasts)
                                          - spectrum[::-1]))) for k in metrics[:4])
    pinv_error = float(np.max(np.abs(x @ inverse - np.eye(n))))
    if max(full_error, active_error, pinv_error) > 1e-9:
        raise AssertionError("Metric spectrum or full-rank inverse reconstruction failed")
    checks = {"feature_sigma_min": float(singular[-1]),
              "feature_sigma_max": float(singular[0]),
              "raw_condition_number": float((singular[0] / singular[-1]) ** 2),
              "raw_scaled_lambda_min": float(np.linalg.eigvalsh(metrics[-1])[0]),
              "full_spectrum_max_error": full_error, "active_spectrum_max_error": active_error,
              "right_inverse_max_error": pinv_error, "feature_width": float(width)}
    return metrics, inverse, checks


def allocate(labels: NDArray[np.int64], counts: NDArray[np.int64], draws: int,
             seed: int) -> list[NDArray[np.int64]]:
    rng = np.random.default_rng(seed)
    positive = rng.permutation(np.flatnonzero(labels == 3))
    negative = rng.permutation(np.flatnonzero(labels == 2))
    pos_offset = neg_offset = 0
    allocations = []
    for _ in range(draws):
        parts = []
        for count in counts:
            positive_part = positive[pos_offset:pos_offset + count]
            negative_part = negative[neg_offset:neg_offset + 4 - count]
            parts.append(rng.permutation(np.r_[positive_part, negative_part]))
            pos_offset += int(count)
            neg_offset += 4 - int(count)
        allocations.append(np.concatenate(parts))
    if len(np.unique(np.concatenate(allocations))) != 4 * len(counts) * draws:
        raise AssertionError("Training draws must use disjoint image rows")
    return allocations


def implementation_check() -> dict[str, float]:
    """Check one changed path: exact posterior, loss factor and shared-head update."""
    rng = np.random.default_rng(2026090602)
    x = rng.normal(size=(8, 13))
    w = rng.normal(size=13) * 0.1
    counts = np.array([1, 3])
    codes, valid = candidates(counts)
    y = x @ w
    state = quantities(y, codes, valid)
    words = np.array(list(product((0.0, 1.0), repeat=8)))
    words = words[(words[:, :4].sum(1) == 1) & (words[:, 4:].sum(1) == 3)]
    score = words @ y - np.logaddexp(0.0, y).sum()
    global_log = logsumexp(score)
    global_target = np.exp(score - global_log) @ words
    residual_error = float(np.max(np.abs(2.0 * (global_target - expit(y)) - state.residual)))
    xt = torch.tensor(x, dtype=torch.float64)
    wt = torch.tensor(w, dtype=torch.float64, requires_grad=True)
    target = torch.tensor(expit(y) + state.residual / 2.0, dtype=torch.float64)
    loss = 2.0 * torch.nn.functional.binary_cross_entropy_with_logits(xt @ wt, target,
                                                                    reduction="sum")
    loss.backward()
    gradient_error = float(np.max(np.abs(wt.grad.numpy() + x.T @ state.residual)))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    k = (q * np.geomspace(0.25, 4.0, 8)) @ q.T
    inverse = np.linalg.pinv(x)
    m = inverse @ k @ inverse.T + np.eye(13) - inverse @ x
    next_w = w - 0.032 * m @ wt.grad.numpy()
    update_error = float(np.max(np.abs(x @ next_w - y - 0.032 * k @ state.residual)))
    result = {"factorized_global_log_evidence_error": float(abs(global_log - state.log_bag_evidence.sum())),
              "factorized_global_residual_error": residual_error,
              "detached_gradient_max_error": gradient_error,
              "shared_head_one_step_max_error": update_error}
    if max(result.values()) > 1e-10:
        raise AssertionError(result)
    return result


def run(features_path: Path, protocol_path: Path, output: Path, phase: str) -> None:
    protocol = json.loads(protocol_path.read_text())
    if output.exists():
        raise FileExistsError(f"Output must be a fresh directory: {output}")
    output.mkdir(parents=True)
    start = time.monotonic()
    source_hash = digest(Path(__file__))
    checks: dict[str, object] = {"implementation": implementation_check()}
    data = np.load(features_path)
    counts = np.array(protocol["bag_counts"], dtype=np.int64)
    selected = protocol["pilot_draws"] if phase == "pilot" else protocol["evaluation_draws"]
    draws = np.array(selected, dtype=np.int64)
    allocations = allocate(data["train_labels"], counts, 9, protocol["data_seed"])
    x = np.stack([design(data["train_features"][allocations[draw]]) for draw in draws])
    test_x = design(data["test_features"])
    labels = np.stack([(data["train_labels"][allocations[draw]] == 3).astype(np.int64)
                       for draw in draws])
    if not np.array_equal(labels.reshape(len(draws), -1, 4).sum(axis=2),
                          np.broadcast_to(counts, (len(draws), len(counts)))):
        raise AssertionError("Constructed bags disagree with the published counts")
    families = [metric_family(values, int(draw), 2026090800) for draw, values in zip(draws, x)]
    metrics = np.stack([family[0] for family in families])
    inverse = np.stack([family[1] for family in families])
    checks["draws"] = {str(draw): family[2] for draw, family in zip(draws, families)}
    heads = protocol["heads_per_draw"]
    width = x.shape[2]
    weights = np.stack([np.stack([np.random.default_rng(protocol["head_seed_base"] + 100 * int(draw) + head)
                                 .uniform(-1 / np.sqrt(width), 1 / np.sqrt(width), width)
                                 for head in range(heads)]) for draw in draws])
    initial = np.einsum("dnp,dhp->dhn", x, weights)
    y = np.repeat(initial[:, :, None, :], metrics.shape[1], axis=2)
    eta = float(protocol["eta"])
    if eta * 4 * (x.shape[1] + 1) >= 4:
        raise AssertionError("The strict global ascent-step bound fails")
    codes, valid = candidates(counts)
    horizon = protocol["baseline_horizon"]
    logits = np.empty((horizon + 1, *y.shape))
    capture_steps = np.full(y.shape[:-1], -1, dtype=np.int64)
    capture_y = np.full_like(y, np.nan)
    lowest_ascent = 0.0
    previous_evidence = np.full(y.shape[:-1], -np.inf)
    for step in range(horizon + 1):
        logits[step] = y
        state = quantities(y, codes, valid)
        evidence = state.log_bag_evidence.sum(axis=-1)
        if not np.isfinite(y).all() or not np.isfinite(evidence).all():
            raise FloatingPointError(f"Nonfinite baseline state at step {step}")
        lowest_ascent = min(lowest_ascent, float((evidence - previous_evidence).min()))
        entered = (capture_steps < 0) & (evidence > -np.log(2.0))
        capture_steps[entered] = step
        capture_y[entered] = y[entered]
        previous_evidence = evidence
        if step < horizon:
            y = y + eta * np.einsum("dmij,dhmj->dhmi", metrics, state.residual)
    checks["baseline_min_log_evidence_increment"] = lowest_ascent
    if lowest_ascent < -1e-10:
        raise AssertionError("Baseline violates the checked ascent bound")
    print(f"{phase}: {capture_steps.size} baselines finished; {(capture_steps >= 0).sum()} captured",
          flush=True)

    captured = capture_steps >= 0
    start_y = np.where(captured[..., None], capture_y, 0.0)
    current = np.repeat(start_y[..., None, :], 3, axis=-2)
    anchor = np.where(start_y > 0.0, 1.0, -1.0)
    flat_start = start_y.reshape(-1, x.shape[1])
    fixed_slow = adverse_direction(flat_start, quantities(flat_start, codes, valid).residual,
                                   anchor.reshape(flat_start.shape)).reshape(start_y.shape)
    other_indices = [1, 0, 3, 2, 0, 0]
    other_metrics = metrics[:, other_indices]
    continuation_horizon = protocol["continuation_horizon"]
    continuations = np.empty((continuation_horizon + 1, *current.shape))
    for step in range(continuation_horizon + 1):
        continuations[step] = np.where(captured[..., None, None], current, np.nan)
        if step == continuation_horizon:
            break
        residual = quantities(current, codes, valid).residual
        other_change = np.einsum("dmij,dhmj->dhmi", other_metrics, residual[..., 0, :])
        fixed_change = 4.0 * metric_product(residual[..., 1, :].reshape(-1, x.shape[1]),
                                            fixed_slow.reshape(flat_start.shape), 0.25 / 4.0)
        adaptive_y = current[..., 2, :].reshape(flat_start.shape)
        adaptive_r = residual[..., 2, :].reshape(flat_start.shape)
        adaptive_slow = adverse_direction(adaptive_y, adaptive_r, anchor.reshape(flat_start.shape))
        adaptive_change = 4.0 * metric_product(adaptive_r, adaptive_slow, 0.25 / 4.0)
        changes = np.stack((other_change, fixed_change.reshape(start_y.shape),
                            adaptive_change.reshape(start_y.shape)), axis=-2)
        current += eta * changes
        if not np.isfinite(current).all():
            raise FloatingPointError(f"Nonfinite continuation at step {step + 1}")

    np.savez(output / "states.npz", draw_ids=draws,
             train_indices=np.stack([data["train_indices"][allocations[draw]] for draw in draws]),
             train_labels=labels, training_design=x, test_design=test_x,
             test_labels=(data["test_labels"] == 3).astype(np.int64), bag_counts=counts,
             metric_names=np.array(protocol["metric_names"]), metrics=metrics, pseudoinverse=inverse,
             initial_weights=weights, logits=logits, continuation_logits=continuations,
             capture_steps=capture_steps, eta=np.array(eta))
    record = {"phase": phase, "protocol_sha256": digest(protocol_path),
              "feature_sha256": digest(features_path), "source_sha256": source_hash,
              "source_unchanged_during_execution": source_hash == digest(Path(__file__)),
              "reused_adverse_source_sha256": digest(ROOT / "verification/run_evidence_capture.py"),
              "checks": checks, "baseline_trajectories": int(capture_steps.size),
              "captured_baselines": int(captured.sum()), "continuation_trajectories": int(captured.sum()) * 3,
              "runtime_seconds": time.monotonic() - start,
              "states_sha256": digest(output / "states.npz")}
    if not record["source_unchanged_during_execution"]:
        raise AssertionError("Execution source changed during the run")
    (output / "run_record.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: record[key] for key in ("phase", "baseline_trajectories", "captured_baselines",
                                                   "continuation_trajectories", "runtime_seconds")}))


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
            parser.error("--features, --output and --phase are required for a full run")
        run(args.features, args.protocol, args.output, args.phase)


if __name__ == "__main__":
    main()
