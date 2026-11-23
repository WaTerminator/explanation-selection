"""Check analytic dense-query minima and their local ordinary-GD continuations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import torch
from scipy.special import expit

from run_cifar_identifiable_traps import (
    Array, Indices, Queries, json_write, loss_and_logit_gradient, query_weights,
)


class Protocol(TypedDict):
    draws: list[int]
    rhos: list[float]
    steps: int
    save_every: int
    perturbation_logit_rms: float
    perturbation_seed: int
    branches: list[str]


class Center(TypedDict):
    draw: int
    rho: float
    correct_probability: float
    loss: float
    gradient_inf: float
    logit_realization_error: float
    logit_hessian_min: float
    effective_hessian_min: float
    effective_hessian_max: float
    jacobian_radius: float
    step_size: float
    nullity: int
    formula_hessian_error: float
    implementation_gradient_error: float


class Observation(TypedDict):
    draw: int
    rho: float
    branch: str
    step: int
    loss: float
    loss_excess: float
    gradient_norm: float
    logit_rms_distance: float
    probability_rms_distance: float
    effective_parameter_distance: float
    quadratic_energy: float
    nullspace_drift: float
    concept_accuracy: float
    reasoning_accuracy: float


def explicit_loss(z: torch.Tensor, pairs: torch.Tensor, counts: torch.Tensor,
                  weights: torch.Tensor) -> torch.Tensor:
    """Independent probability-product objective for the full query support."""
    p = torch.sigmoid(z)
    a, b = pairs.T
    likelihood = torch.stack(((1 - p[a]) * (1 - p[b]),
                              (1 - p[a]) * p[b] + p[a] * (1 - p[b]), p[a] * p[b]))
    evidence = likelihood[counts, torch.arange(len(a))]
    return -(weights * torch.log(evidence)).sum()


def center_geometry(x: Array, labels: Indices, queries: Queries, rho: float,
                    eta: float, draw: int) -> tuple[Array, Array, Array, Array, Center]:
    n, d = x.shape
    mixed = queries.counts == 1
    adjacency = np.zeros((n, n))
    a, b = queries.pairs[mixed].T
    np.add.at(adjacency, (a, b), 1)
    np.add.at(adjacency, (b, a), 1)
    degree = adjacency.sum(axis=1)
    assert np.all(degree == degree[0]) and degree[0] == 8
    assert np.all(labels[a] != labels[b])
    assert np.all(np.bincount(queries.pairs[~mixed].ravel(), minlength=n) == 1)
    np.testing.assert_array_equal(labels[queries.pairs].sum(axis=1), queries.counts)
    u, singular, vt = np.linalg.svd(x, full_matrices=False)
    assert np.linalg.matrix_rank(x) == n
    np.testing.assert_allclose(eta, n / (1.5 * singular[0] ** 2), atol=1e-15, rtol=1e-12)
    delta = math.sqrt(rho * rho + 4 * rho - 4)
    s = (2 + rho + delta) / 4
    q, evidence = 1 - s, s * s + (1 - s) ** 2
    sign = 2 * labels - 1
    target = sign * math.log(q / s)
    head = vt.T @ ((u.T @ target) / singular)
    z = torch.tensor(x @ head, dtype=torch.float64, requires_grad=True)
    pairs = torch.tensor(queries.pairs)
    counts = torch.tensor(queries.counts)
    weights = torch.tensor(query_weights(queries, np.array([rho]))[0])
    loss = explicit_loss(z, pairs, counts, weights)
    gradient = torch.autograd.grad(loss, z)[0].detach().numpy()
    hessian = torch.autograd.functional.hessian(
        lambda values: explicit_loss(values, pairs, counts, weights), z,
    ).detach().numpy()
    hp = 2 * delta / (n * q * evidence) * np.eye(n)
    hp += rho / (mixed.sum() * evidence ** 2) * (np.diag(degree) - adjacency)
    predicted = (s * q) ** 2 * sign[:, None] * hp * sign[None, :]
    formula_error = float(np.max(np.abs(predicted - hessian)))
    np.testing.assert_allclose(hessian, predicted, atol=1e-12, rtol=1e-9)
    effective = singular[:, None] * (u.T @ hessian @ u) * singular[None, :]
    eigenvalues, vectors = np.linalg.eigh(effective)
    assert eigenvalues[0] > 0 and np.max(np.abs(1 - eta * eigenvalues)) < 1
    native_loss, native_gradient = loss_and_logit_gradient(z.detach()[None], pairs,
                                                          counts, weights[None])
    np.testing.assert_allclose(native_loss[0], loss.detach(), atol=1e-12)
    gradient_error = float(np.max(np.abs(native_gradient[0].numpy() - gradient)))
    assert gradient_error < 1e-11 and np.max(np.abs(x.T @ gradient)) < 1e-10
    report: Center = {
        "draw": draw, "rho": rho, "correct_probability": q, "loss": float(loss.detach()),
        "gradient_inf": float(np.max(np.abs(x.T @ gradient))),
        "logit_realization_error": float(np.max(np.abs(x @ head - target))),
        "logit_hessian_min": float(np.linalg.eigvalsh(hessian)[0]),
        "effective_hessian_min": float(eigenvalues[0]), "effective_hessian_max": float(eigenvalues[-1]),
        "jacobian_radius": float(np.max(np.abs(1 - eta * eigenvalues))),
        "step_size": eta, "nullity": d - n, "formula_hessian_error": formula_error,
        "implementation_gradient_error": gradient_error,
    }
    return head, hessian, vt, vectors, report


def initial_branches(x: Array, head: Array, vt: Array, vectors: Array,
                     amplitude: float, rng: np.random.Generator) -> Array:
    weak, strong = vt.T @ vectors[:, 0], vt.T @ vectors[:, -1]
    weak *= np.sign(weak[np.argmax(np.abs(weak))])
    strong *= np.sign(strong[np.argmax(np.abs(strong))])
    parameter = rng.normal(size=x.shape[1])
    logits = np.linalg.lstsq(x, rng.normal(size=x.shape[0]), rcond=None)[0]
    directions = [weak, strong, parameter, logits]
    initial = [head.copy()]
    for direction in directions:
        perturbation = direction * amplitude / np.sqrt(np.mean((x @ direction) ** 2))
        initial.append(head + perturbation)
    return np.stack(initial)


def run(root: Path, protocol_path: Path, output: Path, device: str) -> None:
    if output.exists():
        raise FileExistsError(output)
    protocol = cast(Protocol, json.loads(protocol_path.read_text(encoding="utf-8")))
    assert protocol["branches"] == ["center", "weak_curvature", "strong_curvature", "random_parameter", "random_logit"]
    assert len(protocol["draws"]) * len(protocol["rhos"]) * len(protocol["branches"]) <= 30
    output.mkdir(parents=True)
    shutil.copy2(protocol_path, output / "protocol.json")
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    records: list[Observation] = []
    centers: list[Center] = []
    for draw in protocol["draws"]:
        with np.load(root / f"draw_{draw}" / "states.npz", allow_pickle=False) as state:
            x, labels = state["training_design"], state["train_labels"]
            queries = Queries(state["pairs"], state["counts"])
            eta = float(state["step_size"])
        heads, geometries, reports = [], [], []
        for k, rho in enumerate(protocol["rhos"]):
            head, hessian, vt, vectors, report = center_geometry(x, labels, queries, rho, eta, draw)
            rng = np.random.default_rng(protocol["perturbation_seed"] + 100 * draw + k)
            heads.append(initial_branches(x, head, vt, vectors, protocol["perturbation_logit_rms"], rng))
            geometries.append(hessian)
            reports.append(report)
            centers.append(report)
        initial = np.concatenate(heads)
        count = len(protocol["branches"])
        center_heads = np.repeat(np.stack([h[0] for h in heads]), count, axis=0)
        center_logits = center_heads @ x.T
        rhos = np.repeat(protocol["rhos"], count)
        weights = query_weights(queries, rhos)
        initial_null = initial - (initial @ vt.T) @ vt
        x_t = torch.tensor(x, device=device)
        w = torch.tensor(initial, device=device)
        pairs = torch.tensor(queries.pairs, device=device)
        counts = torch.tensor(queries.counts, device=device)
        weights_t = torch.tensor(weights, device=device)
        snapshots: list[Array] = []
        for step in range(protocol["steps"] + 1):
            z = w @ x_t.T
            loss, gz = loss_and_logit_gradient(z, pairs, counts, weights_t)
            gradient = gz @ x_t
            if step % protocol["save_every"] == 0 or step == protocol["steps"]:
                current, logits = w.cpu().numpy(), z.cpu().numpy()
                loss_np, grad_np = loss.cpu().numpy(), gradient.cpu().numpy()
                if not np.isfinite(current).all():
                    raise FloatingPointError(f"Nonfinite head at draw {draw}, step {step}")
                snapshots.append(current.copy())
                for index, values in enumerate(current):
                    group, variant = divmod(index, count)
                    dz = logits[index] - center_logits[index]
                    row_delta = vt @ (values - center_heads[index])
                    null = values - (values @ vt.T) @ vt
                    hard = logits[index] > 0
                    reasoning = hard[queries.pairs].sum(axis=1) == queries.counts
                    records.append({
                        "draw": draw, "rho": float(rhos[index]), "branch": protocol["branches"][variant],
                        "step": step, "loss": float(loss_np[index]),
                        "loss_excess": float(loss_np[index] - reports[group]["loss"]),
                        "gradient_norm": float(np.linalg.norm(grad_np[index])),
                        "logit_rms_distance": float(np.sqrt(np.mean(dz * dz))),
                        "probability_rms_distance": float(np.sqrt(np.mean((expit(logits[index]) - expit(center_logits[index])) ** 2))),
                        "effective_parameter_distance": float(np.linalg.norm(row_delta)),
                        "quadratic_energy": float(dz @ geometries[group] @ dz / 2),
                        "nullspace_drift": float(np.linalg.norm(null - initial_null[index])),
                        "concept_accuracy": float(np.mean(hard == labels)),
                        "reasoning_accuracy": float(weights[index] @ reasoning),
                    })
                if step % 10000 == 0:
                    print(json.dumps({"draw": draw, "step": step, "maximum_gradient": float(np.linalg.norm(grad_np, axis=1).max())}), flush=True)
            if step < protocol["steps"]:
                w = w - eta * gradient
        np.savez_compressed(output / f"draw_{draw}.npz", heads=np.stack(snapshots),
                            initial_heads=initial, center_heads=center_heads,
                            training_design=x, labels=labels, pairs=queries.pairs, counts=queries.counts,
                            weights=weights, rhos=rhos, hessians=np.stack(geometries), row_basis=vt,
                            step_size=eta)
    with (output / "curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    endpoints = [row for row in records if row["step"] == protocol["steps"]]
    ratios: list[dict[str, float | int | str]] = []
    for endpoint in endpoints:
        if endpoint["branch"] == "center":
            continue
        first = next(row for row in records if row["step"] == 0 and
                     (row["draw"], row["rho"], row["branch"]) == (endpoint["draw"], endpoint["rho"], endpoint["branch"]))
        ratio: dict[str, float | int | str] = {key: endpoint[key] for key in ("draw", "rho", "branch")}
        for metric in ("logit_rms_distance", "probability_rms_distance", "effective_parameter_distance", "quadratic_energy", "loss_excess", "gradient_norm"):
            ratio[metric] = float(endpoint[metric]) / float(first[metric])
        ratios.append(ratio)
    result = {"centers": centers, "endpoints": endpoints, "contraction_ratios": ratios,
              "branches": len(endpoints), "saved_records": len(records),
              "maximum_nullspace_drift": max(row["nullspace_drift"] for row in records),
              "torch": torch.__version__, "device": device}
    json_write(output / "result.json", result)
    print(json.dumps({"finished": True, "branches": len(endpoints), "saved_records": len(records)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    run(args.root, args.protocol, args.output, args.device)


if __name__ == "__main__":
    main()
