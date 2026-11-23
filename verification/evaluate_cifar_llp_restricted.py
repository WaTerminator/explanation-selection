"""Independent count-DP evaluation of complete restricted-head parameter paths.

The training runner is not imported. Complete codes are Boolean arrays and
96-bit byte strings, never single uint64 values.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from numpy.typing import NDArray

from evaluate_cifar_llp import count_log_distribution, digest, require, select_counts, write_csv, write_json

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Metrics:
    log_evidence: Array
    mean_bag_evidence: Array
    min_image_probability: Array
    bits: NDArray[np.bool_]
    legal: NDArray[np.bool_]
    fitted: NDArray[np.bool_]
    residual: Array


def metrics(logits: Array, counts: NDArray[np.int64]) -> Metrics:
    bags = logits.reshape(*logits.shape[:-1], len(counts), 4)
    log_one = -np.logaddexp(0.0, -bags)
    log_zero = -np.logaddexp(0.0, bags)
    bag_log = select_counts(count_log_distribution(log_one, log_zero), counts)
    unary = np.empty_like(bags)
    for index in range(4):
        others = [coordinate for coordinate in range(4) if coordinate != index]
        excluded = count_log_distribution(log_one[..., others], log_zero[..., others])
        unary[..., index] = np.exp(log_one[..., index] + select_counts(excluded, counts - 1) - bag_log)
    bits = logits > 0.0
    legal = (bits.reshape(bags.shape).sum(axis=-1) == counts).all(axis=-1) & (logits != 0).all(axis=-1)
    mean_bag = np.exp(bag_log).mean(axis=-1)
    confidence = np.exp(-np.logaddexp(0.0, -np.abs(logits))).min(axis=-1)
    return Metrics(bag_log.sum(axis=-1), mean_bag, confidence, bits, legal,
                   legal & (mean_bag >= 0.95) & (confidence >= 0.90),
                   2 * (unary - np.exp(log_one)).reshape(logits.shape))


def full_code(bits: NDArray[np.bool_]) -> str:
    return np.packbits(bits, bitorder="little").tobytes().hex()


def first_true(values: NDArray[np.bool_]) -> int | None:
    indices = np.flatnonzero(values)
    return int(indices[0]) if len(indices) else None


def check_path(weights: Array, x: Array, counts: NDArray[np.int64], eta: float,
               diagonal: Array | None, anchor: Array | None) -> tuple[Metrics, float, float]:
    size = len(weights)
    log_evidence = np.empty(size)
    mean_bag = np.empty(size)
    confidence = np.empty(size)
    bits = np.empty((size, x.shape[0]), dtype=bool)
    legal, fitted = np.empty(size, dtype=bool), np.empty(size, dtype=bool)
    residual_error, energy_error = 0.0, 0.0
    witness = weights[0]
    witness_logits = x @ witness
    captured_evidence = float(np.exp(metrics(witness_logits[None], counts).log_evidence[0]))
    for start in range(0, size, 256):
        end = min(start + 256, size)
        current = np.asarray(weights[start:end])
        require(np.isfinite(current).all(), f"Nonfinite parameter state in batch {start}:{end}")
        y = current @ x.T
        require(np.isfinite(y).all(), f"Nonfinite logits in batch {start}:{end}")
        measured = metrics(y, counts)
        require(np.isfinite(measured.log_evidence).all() and np.isfinite(measured.residual).all(),
                f"Nonfinite independent count quantities in batch {start}:{end}")
        log_evidence[start:end] = measured.log_evidence
        mean_bag[start:end] = measured.mean_bag_evidence
        confidence[start:end] = measured.min_image_probability
        bits[start:end] = measured.bits
        legal[start:end], fitted[start:end] = measured.legal, measured.fitted
        gradient = measured.residual @ x
        if diagonal is None:
            require(anchor is not None, "Adaptive continuation requires a captured anchor")
            weakest = np.argmin(anchor * y, axis=-1)
            contribution = anchor[weakest, None] * x[weakest] * gradient
            current_diagonal = np.where(contribution >= 0, 0.5, 2.0)
        else:
            current_diagonal = diagonal
        predicted = current + eta * current_diagonal * gradient
        count = min(end, size - 1) - start
        if count > 0:
            residual_error = max(residual_error, float(np.max(np.abs(predicted[:count] - weights[start + 1:start + count + 1]))))
        if anchor is not None:
            dc = 2 * captured_evidence ** 2 * (2 * captured_evidence - 1)
            mu = float(np.min(np.abs(witness_logits)))
            radius = np.exp(-anchor * y).sum(axis=-1)
            lower = 0.5 * mu ** 2 / float(witness @ witness) * dc ** 2 * radius ** 2
            actual = (current_diagonal * gradient ** 2).sum(axis=-1)
            energy_error = max(energy_error, float(np.maximum(lower - actual, 0).max()))
    require(residual_error <= 1e-9, f"Direct parameter path differs from independent DP: {residual_error}")
    require(energy_error <= 1e-10, f"Captured witness energy bound fails: {energy_error}")
    return Metrics(log_evidence, mean_bag, confidence, bits, legal, fitted, np.empty((0,))), residual_error, energy_error


def evaluate(input_path: Path, features_path: Path, protocol_path: Path, output: Path) -> None:
    require(not output.exists(), f"Evaluation output must be fresh: {output}")
    protocol: dict[str, Any] = json.loads(protocol_path.read_text())
    record = json.loads((input_path / "run_record.json").read_text())
    require(record["protocol_sha256"] == digest(protocol_path), "Protocol identity mismatch")
    require(digest(features_path) == protocol["feature_sha256"], "Feature identity mismatch")
    for name, expected in record["files"].items():
        require(digest(input_path / name) == expected, f"Saved artifact changed: {name}")
    raw = np.load(input_path / "model.npz")
    weights = np.load(input_path / "weights.npy", mmap_mode="r")
    continued = np.load(input_path / "continuation_weights.npy", mmap_mode="r")
    features = np.load(features_path)
    draws, names = raw["draw_ids"], protocol["metric_names"]
    require(draws.tolist() == protocol[f"{record['phase']}_draws"], "Unexpected draw allocation")
    require(weights.shape == (20001, len(draws), 4, 3, 33), "Unexpected prefix state shape")
    require(continued.shape == (4001, len(draws), 4, 3, 3, 33), "Unexpected continuation state shape")
    require(len(np.unique(raw["train_indices"])) == len(draws) * 96, "Training draws overlap")
    require(not np.isin(raw["train_indices"], raw["excluded_original_indices"]).any(), "Original cohort images reused")
    pool = features["train_features"].astype(np.float64)
    normalized = pool / np.linalg.norm(pool, axis=1, keepdims=True)
    center, basis, scale = raw["projection_center"], raw["projection_basis"], raw["projection_scale"]
    covariance = (normalized - center).T @ (normalized - center) / len(pool)
    projection_error = max(float(np.max(np.abs(center - normalized.mean(axis=0)))),
                           float(np.max(np.abs(basis.T @ basis - np.eye(32)))),
                           float(np.max(np.abs(covariance @ basis - basis * scale ** 2))),
                           float(np.max(np.abs(scale ** 2 - np.linalg.eigvalsh(covariance)[-32:][::-1]))))
    indices = np.searchsorted(features["train_indices"], raw["train_indices"])
    require(np.array_equal(features["train_indices"][indices], raw["train_indices"]), "Invalid official training IDs")
    rebuilt = np.concatenate((((normalized[indices] - center) @ basis / scale), np.ones((*indices.shape, 1))), axis=-1) / np.sqrt(33)
    test_features = features["test_features"].astype(np.float64)
    test_projected = (test_features / np.linalg.norm(test_features, axis=1, keepdims=True) - center) @ basis / scale
    rebuilt_test = np.column_stack((test_projected, np.ones(len(test_projected)))) / np.sqrt(33)
    design_error = max(float(np.max(np.abs(rebuilt - raw["training_design"]))),
                       float(np.max(np.abs(rebuilt_test - raw["test_design"]))))
    require(max(projection_error, design_error) < 1e-10, "Unlabeled fixed projection or design reconstruction fails")
    require(np.array_equal(raw["train_labels"], features["train_labels"][indices] == 3), "Train semantic labels mismatch")
    require(np.array_equal(raw["test_labels"], features["test_labels"] == 3), "Test labels mismatch")
    counts = raw["bag_counts"]
    require(np.array_equal(counts, protocol["bag_counts"]), "Count registry mismatch")
    require((raw["train_labels"].reshape(len(draws), 24, 4).sum(axis=-1) == counts).all(), "Bag count supervision mismatch")
    expected_metrics = np.stack((np.ones(33), np.geomspace(0.5, 2, 33), np.geomspace(2, 0.5, 33)))
    require(np.max(np.abs(expected_metrics - raw["parameter_diagonals"])) < 1e-15, "Parameter metric rule differs")
    endpoints: list[dict[str, Any]] = []
    continuations: list[dict[str, Any]] = []
    model_checks: list[dict[str, Any]] = []
    maximum_path_error = maximum_energy_error = 0.0
    minimum_increment = 0.0
    for di, draw in enumerate(draws):
        x, eta = raw["training_design"][di], float(raw["eta"][di])
        singular = np.linalg.svd(x, compute_uv=False)
        rank = int(np.linalg.matrix_rank(x))
        quantity = eta * 2 * singular[0] ** 2 * 5
        require(rank == 33 < len(x) and abs(quantity - 3.2) < 1e-10, "Rank-deficient model or step bound fails")
        model_checks.append({"draw": int(draw), "N": 96, "d": 33, "rank": rank,
                             "sample_nullity": 96 - rank, "feature_condition": float(singular[0] / singular[-1]),
                             "eta": eta, "segment_step_quantity": float(quantity)})
        for head in range(4):
            expected_initial = np.random.default_rng(protocol["head_seed_base"] + 100 * int(draw) + head).uniform(-1 / np.sqrt(33), 1 / np.sqrt(33), 33)
            require(np.max(np.abs(weights[0, di, head] - expected_initial)) < 1e-15, "Ordinary shared head differs from fixed seed")
            for mi, name in enumerate(names):
                path = weights[:, di, head, mi]
                measured, error, _ = check_path(path, x, counts, eta, expected_metrics[mi], None)
                maximum_path_error = max(maximum_path_error, error)
                minimum_increment = min(minimum_increment, float(np.diff(measured.log_evidence).min()))
                capture = first_true(measured.log_evidence > -np.log(2))
                require((-1 if capture is None else capture) == raw["capture_steps"][di, head, mi], "Independent capture time differs")
                if capture is not None:
                    require(measured.legal[capture:].all() and
                            (measured.bits[capture:] == measured.bits[capture]).all(),
                            "Baseline rewrites or loses legality of its captured complete code")
                stable = bool((measured.bits[-101:] == measured.bits[-1]).all())
                endpoint: dict[str, Any] = {"draw": int(draw), "head": head, "metric": name,
                    "well_fitted": bool(measured.fitted[-1]), "resolved": bool(measured.fitted[-1] and stable),
                    "capture_step": capture, "first_fit_step": first_true(measured.fitted),
                    "final_global_evidence": float(np.exp(measured.log_evidence[-1])),
                    "final_mean_bag_evidence": float(measured.mean_bag_evidence[-1]),
                    "final_min_image_probability": float(measured.min_image_probability[-1]),
                    "final_code_hex96": full_code(measured.bits[-1]),
                    "final_train_accuracy": float((measured.bits[-1] == raw["train_labels"][di]).mean()),
                    "final_test_accuracy": float(((raw["test_design"] @ path[-1] > 0) == raw["test_labels"]).mean()),
                    "capture_train_accuracy": None, "capture_test_accuracy": None}
                if capture is not None:
                    endpoint["capture_train_accuracy"] = float((measured.bits[capture] == raw["train_labels"][di]).mean())
                    endpoint["capture_test_accuracy"] = float(((raw["test_design"] @ path[capture] > 0) == raw["test_labels"]).mean())
                endpoints.append(endpoint)
                for ci, continuation_name in enumerate(protocol["continuations"]):
                    continuation = continued[:, di, head, mi, ci]
                    row: dict[str, Any] = {"draw": int(draw), "head": head, "prefix_metric": name,
                        "continuation": continuation_name, "triggered": capture is not None,
                        "sign_changes": None, "final_global_evidence": None, "final_train_accuracy": None,
                        "final_test_accuracy": None}
                    if capture is None:
                        require(np.isnan(continuation).all(), "Untriggered slots must remain NaN")
                    else:
                        require(np.isfinite(continuation).all(), "Executed continuation has nonfinite weights")
                        require(np.max(np.abs(continuation[0] - path[capture])) < 1e-15, "Switch reset or mismatch")
                        anchor = np.where(measured.bits[capture], 1.0, -1.0)
                        diagonal = expected_metrics[0 if ci == 0 else (2 if mi < 2 else 1)] if ci < 2 else None
                        cm, error, energy_error = check_path(continuation, x, counts, eta, diagonal, anchor)
                        maximum_path_error = max(maximum_path_error, error)
                        maximum_energy_error = max(maximum_energy_error, energy_error)
                        minimum_increment = min(minimum_increment, float(np.diff(cm.log_evidence).min()))
                        row.update(sign_changes=int((cm.bits != measured.bits[capture]).any(axis=-1).sum()),
                                   final_global_evidence=float(np.exp(cm.log_evidence[-1])),
                                   final_train_accuracy=float((cm.bits[-1] == raw["train_labels"][di]).mean()),
                                   final_test_accuracy=float(((raw["test_design"] @ continuation[-1] > 0) == raw["test_labels"]).mean()))
                    continuations.append(row)
    require(minimum_increment >= -1e-10, f"Global count log-evidence decreases: {minimum_increment}")
    flipped = sum(row["sign_changes"] > 0 for row in continuations if row["triggered"])
    require(flipped == 0, "Observed captured-code rewrite in a prescribed continuation")
    captures = [row for row in endpoints if row["capture_step"] is not None]
    summary = {"version": protocol["version"], "phase": record["phase"],
        "protocol_sha256": digest(protocol_path), "baseline_denominator": len(endpoints),
        "captured": len(captures), "well_fitted": sum(row["well_fitted"] for row in endpoints), "resolved": sum(row["resolved"] for row in endpoints),
        "continuation_slots": len(continuations), "executed_continuations": len(captures) * 3,
        "untriggered_continuation_slots": len(continuations) - len(captures) * 3, "continuations_with_sign_change": flipped,
        "model_checks": model_checks, "projection_error": projection_error, "design_error": design_error,
        "independent_parameter_path_max_error": maximum_path_error,
        "captured_witness_energy_max_violation": maximum_energy_error,
        "minimum_log_evidence_increment": minimum_increment,
        "per_metric": {name: {"denominator": sum(row["metric"] == name for row in endpoints),
            "captured": sum(row["metric"] == name for row in captures),
            "well_fitted": sum(row["metric"] == name and row["well_fitted"] for row in endpoints),
            "resolved": sum(row["metric"] == name and row["resolved"] for row in endpoints),
            "capture_step_median": float(np.median([row["capture_step"] for row in captures if row["metric"] == name])) if any(row["metric"] == name for row in captures) else None,
            "test_accuracy_median": float(np.median([row["final_test_accuracy"] for row in endpoints if row["metric"] == name]))} for name in names}}
    output.mkdir(parents=True)
    write_csv(output / "endpoints.csv", endpoints)
    write_csv(output / "continuations.csv", continuations)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def implementation_check() -> dict[str, float | str]:
    """Check a 96-bit code and a short independently evaluated parameter path."""
    generator = np.random.default_rng(2026091402)
    x = generator.normal(size=(8, 3))
    counts = np.array([1, 3], dtype=np.int64)
    initial = generator.normal(size=3) * 0.1
    diagonal = np.array([0.5, 1.0, 2.0])
    words = np.array([word for word in product((0.0, 1.0), repeat=8)
                      if sum(word[:4]) == 1 and sum(word[4:]) == 3])
    path = [initial]
    for _ in range(16):
        y = x @ path[-1]
        likelihood = np.exp(words @ y - np.logaddexp(0.0, y).sum())
        unary = likelihood @ words / likelihood.sum()
        residual = 2 * (unary - np.exp(-np.logaddexp(0.0, -y)))
        path.append(path[-1] + 0.01 * diagonal * (x.T @ residual))
    _, error, _ = check_path(np.stack(path), x, counts, 0.01, diagonal, None)
    bits = np.zeros(96, dtype=bool)
    before = full_code(bits)
    bits[-1] = True
    require(full_code(bits) != before and len(full_code(bits)) == 24, "Complete-code packing loses bit 95")
    corrupted = np.stack(path)
    corrupted[4, 0] = np.nan
    try:
        check_path(corrupted, x, counts, 0.01, diagonal, None)
    except ValueError as exception:
        require("Nonfinite parameter state" in str(exception), "Unexpected corruption rejection")
    else:
        raise AssertionError("A nonfinite intermediate parameter state was accepted")
    return {"parameter_path_max_error": error, "complete_96_bit_code": "PASS", "nonfinite_state_rejection": "PASS"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(implementation_check(), indent=2))
    else:
        if any(value is None for value in (args.input, args.features, args.protocol, args.output)):
            parser.error("--input, --features, --protocol and --output are required")
        evaluate(args.input, args.features, args.protocol, args.output)


if __name__ == "__main__":
    main()
