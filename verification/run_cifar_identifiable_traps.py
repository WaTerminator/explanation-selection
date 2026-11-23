"""Train ordinary visual heads on identifiable binary count queries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.special import expit
from torch import Tensor


Array = NDArray[np.float64]
Indices = NDArray[np.int64]


class Protocol(TypedDict):
    feature_sha256: str
    data_seed: int
    head_seed: int
    pair_seed: int
    test_pair_seed: int
    images_per_draw: int
    pilot_draws: list[int]
    evaluation_draws: list[int]
    pilot_heads: int
    evaluation_heads: int
    mixed_degree: int
    rhos: list[float]
    steps: int
    save_every: int


@dataclass(frozen=True)
class Queries:
    pairs: Indices
    counts: Indices


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()


def json_write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def make_queries(labels: Indices, degree: int, seed: int) -> Queries:
    """Keep the support fixed and every image's degree equal within each group."""
    rng = np.random.default_rng(seed)
    negative = rng.permutation(np.flatnonzero(labels == 0))
    positive = rng.permutation(np.flatnonzero(labels == 1))
    n = len(negative)
    if n != len(positive) or n % 2 or degree < 2 or degree >= n:
        raise ValueError("Pairing requires equal even class sizes and 2 <= degree < class size")
    homogeneous = np.concatenate((negative.reshape(-1, 2), positive.reshape(-1, 2)))
    offsets = np.r_[0, 1, rng.choice(np.arange(2, n), degree - 2, replace=False)]
    mixed = np.concatenate([np.column_stack((negative, np.roll(positive, int(offset)))) for offset in offsets])
    pairs = np.concatenate((homogeneous, mixed))
    swap = rng.random(len(pairs)) < 0.5
    pairs[swap] = pairs[swap, ::-1]
    counts = labels[pairs].sum(axis=1).astype(np.int64)
    inferred = np.full(len(labels), -1, dtype=np.int64)
    for count in (0, 2):
        inferred[pairs[counts == count].ravel()] = count // 2
    if not np.array_equal(inferred, labels):
        raise AssertionError("Homogeneous count constraints do not uniquely fix every image label")
    for mask, expected in ((counts == 1, degree), (counts != 1, 1)):
        if not np.all(np.bincount(pairs[mask].ravel(), minlength=len(labels)) == expected):
            raise AssertionError("Image exposure is not balanced")
    return Queries(pairs, counts)


def fast_gradient(logits: Tensor, neighbors: Tensor, targets: Tensor, rhos: Tensor) -> Tensor:
    """Logit gradient of the weighted query loss for the regular mixed-pair graph."""
    n, degree = neighbors.shape
    predictions = torch.sigmoid(logits)
    conditional = torch.sigmoid(logits[:, :, None] - logits[:, neighbors]).sum(dim=2)
    return (2 * (1 - rhos) / n) * (predictions - targets) + (2 * rhos / (n * degree)) * (
        degree * predictions - conditional)


def query_arrays(queries: Queries, n: int) -> tuple[NDArray[np.int64], Array]:
    """Neighbors of every input in the mixed-pair graph and the label fixed by its same-class query."""
    adjacent: list[list[int]] = [[] for _ in range(n)]
    targets = np.full(n, np.nan)
    for (left, right), count in zip(queries.pairs, queries.counts):
        if count == 1:
            adjacent[int(left)].append(int(right))
            adjacent[int(right)].append(int(left))
        else:
            targets[[left, right]] = count / 2
    if len({len(row) for row in adjacent}) != 1 or not np.isfinite(targets).all():
        raise ValueError("Expected a regular mixed-pair graph and a same-class query for every input")
    return np.asarray(adjacent, dtype=np.int64), targets


def query_weights(queries: Queries, rhos: Array) -> Array:
    mixed = queries.counts == 1
    return np.where(mixed[None, :], rhos[:, None] / mixed.sum(),
                    (1 - rhos[:, None]) / (~mixed).sum())


def sparse_queries(queries: Queries, pairs_per_class: int, labels: Indices) -> Queries:
    selected = np.concatenate([np.flatnonzero(queries.counts == count)[:pairs_per_class]
                               for count in (0, 2)] + [np.flatnonzero(queries.counts == 1)])
    restricted = Queries(queries.pairs[selected], queries.counts[selected])
    known = np.full(len(labels), -1, dtype=np.int64)
    for count in (0, 2):
        known[restricted.pairs[restricted.counts == count].ravel()] = count // 2
    for _ in range(len(labels)):
        previous = known.copy()
        for (a, b), count in zip(restricted.pairs, restricted.counts):
            if known[a] >= 0:
                implied = count - known[a]
                if implied not in (0, 1) or (known[b] >= 0 and known[b] != implied):
                    raise AssertionError("Inconsistent count graph")
                known[b] = implied
            if known[b] >= 0:
                known[a] = count - known[b]
        if np.array_equal(previous, known):
            break
    if not np.array_equal(known, labels):
        raise AssertionError("Sparse count graph does not identify all image labels")
    return restricted


def loss_and_logit_gradient(logits: Tensor, pairs: Tensor, counts: Tensor,
                            weights: Tensor) -> tuple[Tensor, Tensor]:
    left, right = logits[:, pairs[:, 0]], logits[:, pairs[:, 1]]
    log_normalizer = torch.nn.functional.softplus(left) + torch.nn.functional.softplus(right)
    legal_score = torch.where(counts == 0, 0.0,
                             torch.where(counts == 2, left + right, torch.logaddexp(left, right)))
    loss = ((log_normalizer - legal_score) * weights).sum(dim=1)
    target_left = torch.where(counts == 1, torch.sigmoid(left - right), counts / 2)
    target_right = torch.where(counts == 1, 1 - target_left, counts / 2)
    gradient = torch.zeros_like(logits)
    gradient.scatter_add_(1, pairs[:, 0].expand(len(logits), -1), weights * (torch.sigmoid(left) - target_left))
    gradient.scatter_add_(1, pairs[:, 1].expand(len(logits), -1), weights * (torch.sigmoid(right) - target_right))
    return loss, gradient


def implementation_check(device: str) -> dict[str, float]:
    """Verify exact per-query abduction, detached BCE and the direct head gradient."""
    rng = np.random.default_rng(231)
    labels = np.repeat([0, 1], 4).astype(np.int64)
    queries = make_queries(labels, 2, 17)
    x = torch.tensor(rng.normal(size=(8, 5)), device=device, dtype=torch.float64)
    w = torch.tensor(rng.normal(size=(2, 5)), device=device, dtype=torch.float64, requires_grad=True)
    pairs = torch.tensor(queries.pairs, device=device)
    counts = torch.tensor(queries.counts, device=device, dtype=torch.float64)
    weights = torch.tensor(query_weights(queries, np.array([0.5, 0.9])), device=device)
    logits = w @ x.T
    loss, grad_z = loss_and_logit_gradient(logits, pairs, counts, weights)
    grad_auto = torch.autograd.grad(loss.sum(), w, retain_graph=True)[0]
    enumeration = torch.zeros_like(loss)
    detached = torch.zeros_like(loss)
    for j, (a, b) in enumerate(queries.pairs):
        code = torch.tensor([(u, v) for u in (0, 1) for v in (0, 1) if u + v == queries.counts[j]],
                            device=device, dtype=torch.float64)
        z = logits[:, [a, b]]
        scores = z @ code.T - torch.nn.functional.softplus(z).sum(dim=1, keepdim=True)
        log_evidence = torch.logsumexp(scores, dim=1)
        target = (torch.softmax(scores, dim=1) @ code).detach()
        enumeration -= weights[:, j] * log_evidence
        detached += weights[:, j] * torch.nn.functional.binary_cross_entropy_with_logits(z, target, reduction="none").sum(dim=1)
    grad_bce = torch.autograd.grad(detached.sum(), w)[0]
    result = {"enumerated_loss_error": float((loss - enumeration).abs().max()),
              "direct_head_gradient_error": float((grad_auto - grad_z @ x).abs().max()),
              "detached_target_gradient_error": float((grad_bce - grad_auto).abs().max())}
    if max(result.values()) > 1e-11:
        raise AssertionError(result)
    return result


def test_queries(labels: Indices, seed: int) -> Queries:
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == label) for label in (0, 1)]
    pairs = np.concatenate([np.column_stack((rng.choice(groups[a], 1000), rng.choice(groups[b], 1000)))
                            for a in (0, 1) for b in (0, 1)])
    return Queries(pairs, labels[pairs].sum(axis=1).astype(np.int64))


def metrics(logits: Array, labels: Indices, queries: Queries) -> dict[str, Array]:
    predicted = logits > 0
    result = {"concept_acc": (predicted == labels).mean(axis=1),
              "confidence": np.maximum(expit(logits), expit(-logits)).mean(axis=1)}
    correct = predicted[:, queries.pairs].sum(axis=2) == queries.counts
    for count in (0, 1, 2):
        result[f"count{count}_acc"] = correct[:, queries.counts == count].mean(axis=1)
    result["balanced_pair_acc"] = (result["count0_acc"] + 2 * result["count1_acc"] + result["count2_acc"]) / 4
    return result


def logit_hessian(logits: Array, queries: Queries, weights: Array) -> Array:
    p = expit(logits)
    exposure = np.bincount(queries.pairs.ravel(), weights=np.repeat(weights, 2), minlength=len(logits))
    hessian = np.diag(exposure * p * (1 - p))
    mixed = queries.counts == 1
    a, b = queries.pairs[mixed].T
    q = expit(logits[a] - logits[b])
    covariance = weights[mixed] * q * (1 - q)
    np.add.at(hessian, (a, a), -covariance)
    np.add.at(hessian, (b, b), -covariance)
    np.add.at(hessian, (a, b), covariance)
    np.add.at(hessian, (b, a), covariance)
    return hessian


def run(features_path: Path, protocol_path: Path, output: Path, phase: str, device: str) -> None:
    if output.exists():
        raise FileExistsError(output)
    protocol = cast(Protocol, json.loads(protocol_path.read_text(encoding="utf-8")))
    if digest(features_path) != protocol["feature_sha256"]:
        raise ValueError("Feature identity differs from the protocol")
    output.mkdir(parents=True)
    checks = implementation_check(device)
    json_write(output / "protocol.json", protocol)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = np.load(features_path)
    raw = data["train_features"].astype(np.float64)
    pool = np.column_stack((raw, np.ones(len(raw))))
    y_pool = (data["train_labels"] == 3).astype(np.int64)
    x_test = np.column_stack((data["test_features"].astype(np.float64), np.ones(len(data["test_features"]))))
    y_test = (data["test_labels"] == 3).astype(np.int64)
    queries_test = test_queries(y_test, protocol["test_pair_seed"])
    rng = np.random.default_rng(protocol["data_seed"])
    class_rows = [rng.permutation(np.flatnonzero(y_pool == label)) for label in (0, 1)]
    count = protocol["images_per_draw"] // 2
    draws = protocol["pilot_draws"] if phase == "pilot" else protocol["evaluation_draws"]
    heads = protocol["pilot_heads"] if phase == "pilot" else protocol["evaluation_heads"]
    rhos = np.repeat(protocol["rhos"], heads)
    x_test_t = torch.tensor(x_test, device=device)
    rows: list[dict[str, float | int | str]] = []
    start = time.monotonic()
    for draw in draws:
        directory = output / f"draw_{draw}"
        directory.mkdir()
        selection = np.concatenate([group[draw * count:(draw + 1) * count] for group in class_rows])
        x_np, labels = pool[selection], y_pool[selection]
        queries = make_queries(labels, protocol["mixed_degree"], protocol["pair_seed"] + draw)
        if "homogeneous_pairs_per_class" in protocol:
            queries = sparse_queries(queries, int(protocol["homogeneous_pairs_per_class"]), labels)
        weights_np = query_weights(queries, rhos)
        x = torch.tensor(x_np, device=device)
        pairs = torch.tensor(queries.pairs, device=device)
        counts = torch.tensor(queries.counts, device=device, dtype=torch.float64)
        weights = torch.tensor(weights_np, device=device)
        singular = np.linalg.svd(x_np, compute_uv=False)
        if "homogeneous_pairs_per_class" in protocol:
            exposures = [np.bincount(queries.pairs.ravel(), weights=np.repeat(weight, 2), minlength=len(labels))
                         for weight in query_weights(queries, np.array(protocol["rhos"]))]
            bound = 0.75 * max(np.linalg.svd(x_np * np.sqrt(exposure)[:, None], compute_uv=False)[0] ** 2
                               for exposure in exposures)
            step_size = 1 / bound
        else:
            step_size = len(x_np) / (1.5 * singular[0] ** 2)
        witness = np.linalg.lstsq(x_np, 2 * labels - 1, rcond=None)[0]
        realization_margin = float(((2 * labels - 1) * (x_np @ witness)).min())
        initial = np.stack([np.random.default_rng(protocol["head_seed"] + 100 * draw + h)
                            .uniform(-1 / math.sqrt(512), 1 / math.sqrt(512), 513) for h in range(heads)])
        w = torch.tensor(np.tile(initial, (len(protocol["rhos"]), 1)), device=device)
        supervised = torch.zeros((1, x.shape[1]), device=device, dtype=torch.float64)
        labels_t = torch.tensor(labels, device=device, dtype=torch.float64)
        snapshots: list[Array] = []
        supervised_snapshots: list[Array] = []
        curve: list[dict[str, float | int]] = []
        prior = np.full(len(w), np.inf)
        maximum_increase = 0.0
        for step in range(protocol["steps"] + 1):
            logits = w @ x.T
            loss, grad_z = loss_and_logit_gradient(logits, pairs, counts, weights)
            gradient = grad_z @ x
            sup_logits = supervised @ x.T
            sup_gradient = (torch.sigmoid(sup_logits) - labels_t) @ x / len(x)
            if step % protocol["save_every"] == 0 or step == protocol["steps"]:
                loss_np = loss.cpu().numpy()
                if not np.isfinite(loss_np).all():
                    raise FloatingPointError(f"Nonfinite loss at draw {draw}, step {step}")
                maximum_increase = max(maximum_increase, float((loss_np - prior).max()))
                prior = loss_np.copy()
                snapshots.append(w.cpu().numpy().copy())
                supervised_snapshots.append(supervised.cpu().numpy().copy())
                train = metrics(logits.cpu().numpy(), labels, queries)
                test = metrics((w @ x_test_t.T).cpu().numpy(), y_test, queries_test)
                norms = torch.linalg.vector_norm(gradient, dim=1).cpu().numpy()
                for index in range(len(w)):
                    row: dict[str, float | int] = {"step": step, "run": index, "rho": float(rhos[index]),
                                                   "head": index % heads, "loss": float(loss_np[index]),
                                                   "gradient_norm": float(norms[index])}
                    row.update({f"train_{key}": float(value[index]) for key, value in train.items()})
                    row.update({f"test_{key}": float(value[index]) for key, value in test.items()})
                    curve.append(row)
                if step % (protocol["save_every"] * 10) == 0 or step == protocol["steps"]:
                    print(json.dumps({"phase": phase, "draw": draw, "step": step,
                                      "train_acc_range": [float(train["concept_acc"].min()), float(train["concept_acc"].max())],
                                      "loss_range": [float(loss_np.min()), float(loss_np.max())],
                                      "seconds": round(time.monotonic() - start, 1)}), flush=True)
            if step < protocol["steps"]:
                w = w - step_size * gradient
                supervised = supervised - step_size * sup_gradient
        if maximum_increase > 1e-9:
            raise AssertionError(f"Loss increased by {maximum_increase}")
        np.savez_compressed(directory / "states.npz", heads=np.stack(snapshots), supervised_heads=np.stack(supervised_snapshots),
                            initial_heads=initial, training_design=x_np, test_design=x_test,
                            train_labels=labels, test_labels=y_test, training_indices=data["train_indices"][selection],
                            test_indices=data["test_indices"], pairs=queries.pairs, counts=queries.counts,
                            test_pairs=queries_test.pairs, test_counts=queries_test.counts, rhos=rhos,
                            bag_weights=weights_np, step_size=step_size, realization_witness=witness)
        with (directory / "curves.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(curve[0]))
            writer.writeheader()
            writer.writerows(curve)
        final = curve[-len(w):]
        z_final = (w @ x.T).cpu().numpy()
        before = snapshots[-min(5, len(snapshots))] @ x_np.T
        for index, row in enumerate(final):
            hessian = logit_hessian(z_final[index], queries, weights_np[index])
            row.update({"draw": draw, "phase": phase,
                        "minimum_logit_hessian_eigenvalue": float(np.linalg.eigvalsh(hessian)[0]),
                        "final_window_label_changes": int(((before[index] > 0) != (z_final[index] > 0)).sum()),
                        "realization_margin": realization_margin})
            rows.append(row)
        sup_train = metrics(sup_logits.cpu().numpy(), labels, queries)
        sup_test = metrics((supervised @ x_test_t.T).cpu().numpy(), y_test, queries_test)
        json_write(directory / "supervised.json", {
            "train": {key: float(value[0]) for key, value in sup_train.items()},
            "test": {key: float(value[0]) for key, value in sup_test.items()},
            "realization_margin": realization_margin, "design_rank": int(np.linalg.matrix_rank(x_np)),
            "step_size": step_size, "maximum_saved_loss_increase": maximum_increase})
    with (output / "endpoints.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_write(output / "run_record.json", {"phase": phase, "protocol_sha256": digest(protocol_path),
               "runner_sha256": digest(Path(__file__)), "feature_sha256": digest(features_path),
               "checks": checks, "runs": len(rows), "torch": torch.__version__, "numpy": np.__version__,
               "device": device, "seconds": time.monotonic() - start})
    print(json.dumps({"finished": phase, "runs": len(rows), "seconds": round(time.monotonic() - start, 1)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "evaluation"), required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    run(args.features, args.protocol, args.output, args.phase, args.device)


if __name__ == "__main__":
    main()
