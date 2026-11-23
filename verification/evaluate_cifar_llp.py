#!/usr/bin/env python3
"""Independently evaluate frozen CIFAR-LLP states with Bernoulli-count DP.

No training runner is imported. Probabilities and unary posteriors are computed
by count dynamic programming; configuration enumeration is used only in a small
implementation check. Test accuracy never enters the continuation decision.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
VERSION = "cifar_llp_selection_capture_v1"
NUMERICAL_TOLERANCE = 1e-8
EVIDENCE_TOLERANCE = 1e-10
FIT_BAG_MEAN = 0.95
IMAGE_THRESHOLD = 0.90
POSTERIOR_THRESHOLD = 0.90
CONFIRMATION_UPDATES = 100


@dataclass(frozen=True)
class StateMetrics:
    bag_log_evidence: FloatArray
    log_evidence: FloatArray
    mean_bag_evidence: FloatArray
    min_image_probability: FloatArray
    conditional_map_probability: FloatArray
    conditional_hard_probability: FloatArray
    hard_codes: NDArray[np.uint64]
    map_codes: NDArray[np.uint64]
    legal: BoolArray
    well_fitted: BoolArray
    residual: FloatArray


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                          for key, value in row.items()} for row in rows)


def count_log_distribution(log_one: FloatArray, log_zero: FloatArray) -> FloatArray:
    """Log probabilities for every Bernoulli count, with variables on last axis."""
    result = np.full((*log_one.shape[:-1], log_one.shape[-1] + 1), -np.inf)
    result[..., 0] = 0.0
    for index in range(log_one.shape[-1]):
        updated = result + log_zero[..., index, None]
        updated[..., 1:] = np.logaddexp(updated[..., 1:], result[..., :-1] + log_one[..., index, None])
        result = updated
    return result


def select_counts(distribution: FloatArray, counts: NDArray[np.int64]) -> FloatArray:
    indices = np.broadcast_to(counts, distribution.shape[:-1])[..., None]
    return np.take_along_axis(distribution, indices, axis=-1)[..., 0]


def pack_codes(bits: BoolArray) -> NDArray[np.uint64]:
    powers = np.left_shift(np.uint64(1), np.arange(bits.shape[-1], dtype=np.uint64))
    return (bits.astype(np.uint64) * powers).sum(axis=-1, dtype=np.uint64)


def code_bits(code: int, size: int) -> list[int]:
    return [(code >> index) & 1 for index in range(size)]


def evaluate_states(logits: FloatArray, counts: NDArray[np.int64], bag_size: int) -> StateMetrics:
    require(bool(np.isfinite(logits).all()), "Nonfinite evaluated logits")
    bags = logits.reshape(*logits.shape[:-1], len(counts), bag_size)
    log_one = -np.logaddexp(0.0, -bags)
    log_zero = -np.logaddexp(0.0, bags)
    bag_log_evidence = select_counts(count_log_distribution(log_one, log_zero), counts)
    unary = np.empty_like(bags)
    for coordinate in range(bag_size):
        others = [index for index in range(bag_size) if index != coordinate]
        excluded = count_log_distribution(log_one[..., others], log_zero[..., others])
        unary[..., coordinate] = np.exp(log_one[..., coordinate]
                                       + select_counts(excluded, counts - 1) - bag_log_evidence)
    residual = 2.0 * (unary - np.exp(log_one))
    hard = bags > 0.0
    legal = (hard.sum(axis=-1) == counts).all(axis=-1) & (bags != 0.0).all(axis=(-2, -1))
    map_bits = np.zeros_like(hard)
    for bag, count in enumerate(counts):
        order = np.argsort(bags[..., bag, :], axis=-1, kind="stable")
        np.put_along_axis(map_bits[..., bag, :], order[..., -int(count):], True, axis=-1)
    log_evidence = bag_log_evidence.sum(axis=-1)
    map_log_likelihood = np.where(map_bits, log_one, log_zero).sum(axis=(-2, -1))
    hard_log_likelihood = np.maximum(log_one, log_zero).sum(axis=(-2, -1))
    map_probability = np.exp(map_log_likelihood - log_evidence)
    hard_probability = np.where(legal, np.exp(hard_log_likelihood - log_evidence), 0.0)
    mean_bag = np.exp(bag_log_evidence).mean(axis=-1)
    min_image = np.exp(-np.logaddexp(0.0, -np.abs(logits))).min(axis=-1)
    return StateMetrics(
        bag_log_evidence, log_evidence, mean_bag, min_image, map_probability, hard_probability,
        pack_codes(hard.reshape(logits.shape)), pack_codes(map_bits.reshape(logits.shape)), legal,
        legal & (mean_bag >= FIT_BAG_MEAN) & (min_image >= IMAGE_THRESHOLD), residual.reshape(logits.shape),
    )


def first_true(values: BoolArray) -> int | None:
    selected = np.flatnonzero(values)
    return int(selected[0]) if len(selected) else None


def confirmation_mask(codes: NDArray[np.uint64], fitted: BoolArray) -> BoolArray:
    last_change = np.maximum.accumulate(np.r_[0, np.where(codes[1:] != codes[:-1], np.arange(1, len(codes)), 0)])
    return fitted & (np.arange(len(codes)) - last_change >= CONFIRMATION_UPDATES)


def summarize_values(values: list[float]) -> dict[str, int | float | None]:
    return {"denominator": len(values), "minimum": min(values) if values else None,
            "median": float(np.median(values)) if values else None, "maximum": max(values) if values else None}


def implementation_check() -> dict[str, float | str]:
    generator = np.random.default_rng(2026090619)
    counts = np.array([1, 2], dtype=np.int64)
    values = generator.normal(size=(5, 8))
    evaluated = evaluate_states(values, counts, 4)
    candidates = np.array([bits for bits in itertools.product((0, 1), repeat=8)
                           if sum(bits[:4]) == 1 and sum(bits[4:]) == 2], dtype=np.float64)
    one = -np.logaddexp(0.0, -values)
    zero = -np.logaddexp(0.0, values)
    likelihoods = np.exp(np.einsum("ni,ci->nc", one, candidates)
                         + np.einsum("ni,ci->nc", zero, 1.0 - candidates))
    evidence = likelihoods.sum(axis=-1)
    unary = likelihoods @ candidates / evidence[:, None]
    checks = {
        "dp_global_evidence_max_absolute_error": float(np.max(np.abs(np.exp(evaluated.log_evidence) - evidence))),
        "dp_unary_max_absolute_error": float(np.max(np.abs(evaluated.residual / 2 + np.exp(one) - unary))),
        "dp_map_probability_max_absolute_error": float(np.max(np.abs(evaluated.conditional_map_probability - likelihoods.max(axis=-1) / evidence))),
    }
    head = values[0].copy()
    fixed_unary = unary[0]
    derivative = np.empty_like(head)
    step = 1e-6
    for index in range(len(head)):
        direction = np.eye(len(head))[index] * step
        loss_plus = 2 * (np.logaddexp(0.0, head + direction) - fixed_unary * (head + direction)).sum()
        loss_minus = 2 * (np.logaddexp(0.0, head - direction) - fixed_unary * (head - direction)).sum()
        derivative[index] = (loss_plus - loss_minus) / (2 * step)
    checks["twice_sum_detached_bce_gradient_error"] = float(np.max(np.abs(derivative + evaluated.residual[0])))
    design = generator.normal(size=(8, 13))
    pinv = np.linalg.pinv(design)
    basis, _ = np.linalg.qr(generator.normal(size=(8, 8)))
    gram = (basis * np.linspace(0.25, 4.0, 8)) @ basis.T
    metric = pinv @ gram @ pinv.T + np.eye(13) - pinv @ design
    weight = generator.normal(size=13)
    residual = evaluate_states(design @ weight, counts, 4).residual
    head_change = 0.01 * metric @ design.T @ residual
    checks["shared_head_one_step_parity_error"] = float(np.max(np.abs(design @ head_change - 0.01 * gram @ residual)))
    require(all(np.isfinite(value) and value < NUMERICAL_TOLERANCE for value in checks.values()),
            f"Independent implementation check failed: {checks}")
    return {"status": "PASS", **checks}


def active_basis(bag_count: int, bag_size: int) -> tuple[FloatArray, FloatArray]:
    contrasts = np.zeros((bag_count * bag_size, bag_count * (bag_size - 1)))
    constants = np.zeros((bag_count * bag_size, bag_count))
    for bag in range(bag_count):
        start = bag * bag_size
        constants[start:start + bag_size, bag] = 1 / np.sqrt(bag_size)
        for size in range(1, bag_size):
            column = bag * (bag_size - 1) + size - 1
            contrasts[start:start + size, column] = 1 / np.sqrt(size * (size + 1))
            contrasts[start + size, column] = -size / np.sqrt(size * (size + 1))
    return contrasts, constants


def verify_model(raw: dict[str, NDArray[Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    draws, heads, names = raw["draw_ids"].tolist(), protocol["heads_per_draw"], protocol["metric_names"]
    size, dimension = protocol["training_images"], 513
    expected = {
        "train_indices": (len(draws), size), "train_labels": (len(draws), size),
        "training_design": (len(draws), size, dimension), "test_design": (2000, dimension), "test_labels": (2000,),
        "metrics": (len(draws), len(names), size, size), "pseudoinverse": (len(draws), dimension, size),
        "initial_weights": (len(draws), heads, dimension),
        "logits": (protocol["baseline_horizon"] + 1, len(draws), heads, len(names), size),
        "continuation_logits": (protocol["continuation_horizon"] + 1, len(draws), heads, len(names), 3, size),
        "capture_steps": (len(draws), heads, len(names)),
    }
    for key, shape in expected.items():
        require(raw[key].shape == shape, f"Invalid {key} shape: {raw[key].shape}; expected {shape}")
        if key != "continuation_logits":
            require(bool(np.isfinite(raw[key]).all()), f"Nonfinite input: {key}")
    require(raw["metric_names"].tolist() == names, "Metric registry mismatch")
    require(raw["bag_counts"].tolist() == protocol["bag_counts"], "Bag count registry mismatch")
    require(draws in (protocol["pilot_draws"], protocol["evaluation_draws"]), "Unexpected phase draw registry")
    require(np.unique(raw["train_indices"]).size == len(draws) * size, "Training images overlap across draws")
    require(bool(np.isin(raw["train_labels"], [0, 1]).all()) and bool(np.isin(raw["test_labels"], [0, 1]).all()), "Labels must be fixed cat-positive 0/1")
    require(bool((raw["train_labels"].reshape(len(draws), -1, protocol["bag_size"]).sum(axis=-1) == raw["bag_counts"]).all()), "Observed counts disagree with bag labels")
    require(abs(float(raw["eta"]) - protocol["eta"]) <= 1e-15, "Step size differs from protocol")
    require(protocol["eta"] * protocol["kappa_max"] * (size + 1) < 4.0, "Capture step bound fails")
    contrasts, constants = active_basis(len(raw["bag_counts"]), protocol["bag_size"])
    spectrum = np.geomspace(4.0, 0.25, contrasts.shape[1])
    full_spectrum = np.sort(np.r_[spectrum, np.ones(constants.shape[1])])
    checks: list[dict[str, Any]] = []
    for draw_index, draw in enumerate(draws):
        design, pinv, metrics = raw["training_design"][draw_index], raw["pseudoinverse"][draw_index], raw["metrics"][draw_index]
        u, singular, vt = np.linalg.svd(design, full_matrices=False)
        require(singular[-1] / singular[0] > 1e-8, f"Training design fails full row rank check: draw {draw}")
        independent_pinv = (vt.T / singular) @ u.T
        eigenvalues = np.linalg.eigvalsh(metrics)
        errors = {
            "pseudoinverse_relative_error": float(np.linalg.norm(pinv - independent_pinv) / np.linalg.norm(independent_pinv)),
            "sample_identity_error": float(np.max(np.abs(design @ pinv - np.eye(size)))),
            "metric_symmetry_error": float(np.max(np.abs(metrics - metrics.transpose(0, 2, 1)))),
            "full_isospectral_error": 0.0, "active_isospectral_error": 0.0, "active_complement_error": 0.0,
        }
        for name in protocol["isospectral_metric_names"]:
            index = names.index(name)
            errors["full_isospectral_error"] = max(errors["full_isospectral_error"], float(np.max(np.abs(eigenvalues[index] - full_spectrum))))
            errors["active_isospectral_error"] = max(errors["active_isospectral_error"], float(np.max(np.abs(np.linalg.eigvalsh(contrasts.T @ metrics[index] @ contrasts) - np.sort(spectrum)))))
            errors["active_complement_error"] = max(errors["active_complement_error"], float(np.max(np.abs(constants.T @ metrics[index] @ constants - np.eye(len(raw["bag_counts"]))))), float(np.max(np.abs(contrasts.T @ metrics[index] @ constants))))
        scalar = full_spectrum.mean()
        errors["isotropic_reference_error"] = float(np.max(np.abs(metrics[names.index("isotropic")] - scalar * np.eye(size))))
        natural = design @ design.T
        errors["raw_reference_error"] = float(np.max(np.abs(metrics[names.index("raw_scaled")] - 4 * natural / np.linalg.eigvalsh(natural)[-1])))
        y0 = raw["initial_weights"][draw_index] @ design.T
        errors["initial_shared_logit_error"] = float(np.max(np.abs(raw["logits"][0, draw_index] - y0[:, None])))
        require(all(np.isfinite(value) and value <= NUMERICAL_TOLERANCE for value in errors.values()), f"Model consistency failure draw {draw}: {errors}")
        require(bool((eigenvalues[:, 0] > 0).all()) and float(eigenvalues.max()) <= 4 + NUMERICAL_TOLERANCE, f"SPD/upper spectral bound fails: draw {draw}")
        controlled = [index for index, name in enumerate(names) if name != "raw_scaled"]
        require(float(eigenvalues[controlled, 0].min()) >= 0.25 - NUMERICAL_TOLERANCE, f"Controlled lower bound fails: draw {draw}")
        checks.append({"draw": draw, "rank": size, "singular_value_ratio": float(singular[-1] / singular[0]),
                       "metric_min_eigenvalues": dict(zip(names, eigenvalues[:, 0].tolist())),
                       "metric_max_eigenvalues": dict(zip(names, eigenvalues[:, -1].tolist())), **errors})
    return {"passed": True, "tolerance": NUMERICAL_TOLERANCE, "draws": checks,
            "ascent_quantity": protocol["eta"] * protocol["kappa_max"] * (size + 1)}


def baseline_evaluation(raw: dict[str, NDArray[Any]], protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, FloatArray], dict[str, Any]]:
    logits = raw["logits"]
    shape = logits.shape[:-1]
    saved = {name: np.empty(shape, dtype=np.float64) for name in
             ("log_evidence", "mean_bag_evidence", "min_image_probability", "conditional_map_probability", "conditional_hard_probability")}
    saved["bag_log_evidence"] = np.empty((*shape, len(raw["bag_counts"])), dtype=np.float64)
    saved["hard_codes"] = np.empty(shape, dtype=np.uint64)
    saved["map_codes"] = np.empty(shape, dtype=np.uint64)
    saved["legal"] = np.empty(shape, dtype=bool)
    saved["well_fitted"] = np.empty(shape, dtype=bool)
    saved["empirically_confirmed"] = np.empty(shape, dtype=bool)
    rows: list[dict[str, Any]] = []
    test_bags = np.random.default_rng(protocol["test_bag_seed"]).permutation(len(raw["test_labels"])).reshape(-1, protocol["bag_size"])
    test_counts = raw["test_labels"][test_bags].sum(axis=1)
    eta, size = float(raw["eta"]), protocol["training_images"]
    maximum_update_error = maximum_reconstruction_error = maximum_log_evidence_drop = 0.0
    for di, draw in enumerate(raw["draw_ids"].tolist()):
        design, pinv = raw["training_design"][di], raw["pseudoinverse"][di]
        projector = design @ pinv
        for head in range(protocol["heads_per_draw"]):
            initial_weight = raw["initial_weights"][di, head]
            y0 = design @ initial_weight
            for mi, name in enumerate(protocol["metric_names"]):
                y = logits[:, di, head, mi]
                result = evaluate_states(y, raw["bag_counts"], protocol["bag_size"])
                confirmed = confirmation_mask(result.hard_codes, result.well_fitted)
                for key in saved:
                    saved[key][:, di, head, mi] = confirmed if key == "empirically_confirmed" else getattr(result, key)
                delta = eta * result.residual[:-1] @ raw["metrics"][di, mi].T
                update_error = float(np.max(np.abs(np.diff(y, axis=0) - delta)))
                reconstructed_y = y0 + (y - y0) @ projector.T
                reconstruction_error = float(np.max(np.abs(reconstructed_y - y)))
                evidence_drop = float(max(0.0, -np.diff(result.log_evidence).min()))
                maximum_update_error = max(maximum_update_error, update_error)
                maximum_reconstruction_error = max(maximum_reconstruction_error, reconstruction_error)
                maximum_log_evidence_drop = max(maximum_log_evidence_drop, evidence_drop)
                capture = first_true(result.log_evidence > -np.log(2.0))
                require((-1 if capture is None else capture) == int(raw["capture_steps"][di, head, mi]),
                        f"Independent capture time mismatch: draw={draw}, head={head}, metric={name}")
                fitted = first_true(result.well_fitted)
                lead = (fitted - capture) / fitted if capture is not None and fitted is not None and fitted > 0 else None
                changes = np.flatnonzero(result.hard_codes[1:] != result.hard_codes[:-1]) + 1
                onset = int(changes[-1]) if len(changes) else 0
                final_code = int(result.hard_codes[-1])
                final_bits = np.array(code_bits(final_code, size), dtype=np.int64)
                # Keep the ordinary head's null-space component when reconstructing W.
                final_weight = initial_weight + pinv @ (y[-1] - y0)
                final_train_reconstruction = float(np.max(np.abs(design @ final_weight - y[-1])))
                maximum_reconstruction_error = max(maximum_reconstruction_error, final_train_reconstruction)
                test_bits = raw["test_design"] @ final_weight > 0.0
                row = {"draw": draw, "head": head, "metric": name, "resolved": bool(confirmed[-1]),
                       "final_code": final_bits.tolist(), "final_code_integer": final_code,
                       "final_legal": bool(result.legal[-1]), "final_well_fitted": bool(result.well_fitted[-1]),
                       "final_log_evidence": float(result.log_evidence[-1]), "final_mean_bag_evidence": float(result.mean_bag_evidence[-1]),
                       "final_min_image_probability": float(result.min_image_probability[-1]),
                       "final_conditional_map_probability": float(result.conditional_map_probability[-1]),
                       "final_conditional_hard_probability": float(result.conditional_hard_probability[-1]),
                       "T_capture": capture, "T_fit": fitted, "T_image90": first_true(result.min_image_probability >= IMAGE_THRESHOLD),
                       "T_q90": first_true(result.conditional_map_probability >= POSTERIOR_THRESHOLD),
                       "T_confirm100": first_true(confirmed), "hindsight_final_code_stability_onset": onset,
                       "hindsight_last_sign_change": int(changes[-1]) if len(changes) else None,
                       "number_of_code_changes": len(changes), "primary_fractional_lead": lead,
                       "train_accuracy": float((final_bits == raw["train_labels"][di]).mean()),
                       "test_accuracy": float((test_bits == raw["test_labels"]).mean()),
                       "test_count_accuracy": float((test_bits[test_bags].sum(axis=1) == test_counts).mean()),
                       "update_max_absolute_error": update_error, "reconstruction_max_absolute_error": max(reconstruction_error, final_train_reconstruction),
                       "largest_log_evidence_decrease": evidence_drop}
                rows.append(row)
        print(json.dumps({"evaluated_draw": draw, "trajectories": len(rows)}), flush=True)
    checks = {"update_max_absolute_error": maximum_update_error,
              "head_reconstruction_max_absolute_error": maximum_reconstruction_error,
              "largest_log_evidence_decrease": maximum_log_evidence_drop}
    require(maximum_update_error <= NUMERICAL_TOLERANCE and maximum_reconstruction_error <= NUMERICAL_TOLERANCE,
            f"Trajectory/model replay check failed: {checks}")
    require(maximum_log_evidence_drop <= EVIDENCE_TOLERANCE, f"Baseline evidence decreased: {checks}")
    return rows, saved, {"passed": True, **checks}


def continuation_evaluation(raw: dict[str, NDArray[Any]], protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, FloatArray]]:
    continuations = raw["continuation_logits"]
    saved = {"continuation_log_evidence": np.full(continuations.shape[:-1], np.nan),
             "continuation_anchor_flip": np.zeros(continuations.shape[:-1], dtype=bool)}
    rows: list[dict[str, Any]] = []
    for di, draw in enumerate(raw["draw_ids"].tolist()):
        for head in range(protocol["heads_per_draw"]):
            for mi, name in enumerate(protocol["metric_names"]):
                capture = int(raw["capture_steps"][di, head, mi])
                for ci, continuation in enumerate(protocol["continuations_at_first_capture"]):
                    y = continuations[:, di, head, mi, ci]
                    common = {"draw": draw, "head": head, "prefix_metric": name, "continuation": continuation, "T_capture": capture if capture >= 0 else None}
                    if capture < 0:
                        require(bool(np.isnan(y).all()), f"Uncaptured branch contains states: {common}")
                        rows.append({**common, "status": "not_captured", "ever_anchor_flip": None, "first_anchor_flip_offset": None,
                                     "start_log_evidence": None, "final_log_evidence": None, "largest_log_evidence_decrease": None,
                                     "largest_evidence_decrease": None, "switch_continuity_error": None, "final_min_anchor_margin": None,
                                     "update_max_absolute_error": None})
                        continue
                    require(bool(np.isfinite(y).all()), f"Captured branch incomplete: {common}")
                    anchor_state = raw["logits"][capture, di, head, mi]
                    continuity_error = float(np.max(np.abs(y[0] - anchor_state)))
                    require(continuity_error <= NUMERICAL_TOLERANCE, f"Continuation switch is discontinuous: {common}")
                    result = evaluate_states(y, raw["bag_counts"], protocol["bag_size"])
                    anchor = np.where(anchor_state > 0.0, 1.0, -1.0)
                    flips = (y * anchor <= 0.0).any(axis=-1)
                    residual = result.residual[:-1]
                    if continuation == "other_fixed_isospectral_direction":
                        other_index = [1, 0, 3, 2, 0, 0][mi]
                        response = residual @ raw["metrics"][di, other_index].T
                    else:
                        at = slice(0, 1) if continuation == "fixed_adverse_one_slow" else slice(None, -1)
                        selected_residual = result.residual[at]
                        lengths = np.linalg.norm(selected_residual, axis=-1, keepdims=True)
                        require(bool((lengths > 0).all()), f"Zero adverse-direction residual: {common}")
                        weakest = (y[at] * anchor).argmin(axis=-1)
                        unit = selected_residual / lengths
                        unit[np.arange(len(unit)), weakest] += anchor[weakest]
                        norms = np.linalg.norm(unit, axis=-1, keepdims=True)
                        require(bool((norms > 0).all()), f"Undefined adverse direction: {common}")
                        unit /= norms
                        response = 4.0 * residual - 3.75 * unit * (unit * residual).sum(axis=-1, keepdims=True)
                    update_error = float(np.max(np.abs(np.diff(y, axis=0) - float(raw["eta"]) * response)))
                    require(update_error <= NUMERICAL_TOLERANCE, f"Prescribed continuation update mismatch: {common}: {update_error}")
                    saved["continuation_log_evidence"][:, di, head, mi, ci] = result.log_evidence
                    saved["continuation_anchor_flip"][:, di, head, mi, ci] = flips
                    rows.append({**common, "status": "evaluated", "ever_anchor_flip": bool(flips.any()),
                                 "first_anchor_flip_offset": first_true(flips), "start_log_evidence": float(result.log_evidence[0]),
                                 "final_log_evidence": float(result.log_evidence[-1]),
                                 "largest_log_evidence_decrease": float(max(0.0, -np.diff(result.log_evidence).min())),
                                 "largest_evidence_decrease": float(max(0.0, -np.diff(np.exp(result.log_evidence)).min())),
                                 "switch_continuity_error": continuity_error, "final_min_anchor_margin": float((y[-1] * anchor).min()),
                                 "update_max_absolute_error": update_error})
    return rows, saved


def paired_rows(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    lookup = {(row["draw"], row["head"], row["metric"]): row for row in rows}
    pairs = []
    for draw, head in sorted({(row["draw"], row["head"]) for row in rows}):
        for left_name, right_name in itertools.combinations(protocol["isospectral_metric_names"], 2):
            left, right = lookup[draw, head, left_name], lookup[draw, head, right_name]
            distance = sum(a != b for a, b in zip(left["final_code"], right["final_code"]))
            both = left["resolved"] and right["resolved"]
            pairs.append({"draw": draw, "head": head, "left_metric": left_name, "right_metric": right_name,
                          "left_resolved": left["resolved"], "right_resolved": right["resolved"], "both_resolved": both,
                          "final_hamming_distance": distance, "complement": distance == protocol["training_images"],
                          "changed_noncomplement": 0 < distance < protocol["training_images"],
                          "both_resolved_changed_noncomplement": both and 0 < distance < protocol["training_images"]})
    return pairs


def summarize(rows: list[dict[str, Any]], pairs: list[dict[str, Any]], continuations: list[dict[str, Any]],
              protocol: dict[str, Any], model: dict[str, Any], numerical: dict[str, Any]) -> dict[str, Any]:
    draws = sorted({row["draw"] for row in rows})
    iso = [row for row in rows if row["metric"] in protocol["isospectral_metric_names"]]
    eligible_leads = [row for row in iso if row["resolved"] and row["T_capture"] is not None and row["primary_fractional_lead"] is not None]
    lead_values = [row["primary_fractional_lead"] for row in eligible_leads]
    per_draw_lead = {str(draw): summarize_values([row["primary_fractional_lead"] for row in eligible_leads if row["draw"] == draw]) for draw in draws}
    per_metric = {}
    for name in protocol["metric_names"]:
        selected = [row for row in rows if row["metric"] == name]
        per_metric[name] = {"denominator": len(selected), "resolved": sum(row["resolved"] for row in selected),
                            "captured": sum(row["T_capture"] is not None for row in selected),
                            "final_code_counts": dict(Counter(str(row["final_code_integer"]) for row in selected)),
                            "train_accuracy": summarize_values([row["train_accuracy"] for row in selected]),
                            "test_accuracy": summarize_values([row["test_accuracy"] for row in selected])}
    qualifying_pairs = [row for row in pairs if row["both_resolved_changed_noncomplement"]]
    qualifying_draws = sorted({row["draw"] for row in qualifying_pairs})
    captured = sum(row["T_capture"] is not None for row in iso)
    observed_continuations = [row for row in continuations if row["status"] == "evaluated"]
    flips = sum(row["ever_anchor_flip"] for row in observed_continuations)
    decreases = sum(row["largest_log_evidence_decrease"] > EVIDENCE_TOLERANCE for row in observed_continuations)
    continuation_pass = flips == 0 and decreases == 0
    positive_draws = [draw for draw, values in per_draw_lead.items() if values["median"] is not None and values["median"] > 0]
    pilot = draws == protocol["pilot_draws"]
    return {
        "protocol_version": protocol["version"], "phase": "pilot" if pilot else "evaluation",
        "draws": draws, "trajectory_count": len(rows), "isospectral_trajectory_count": len(iso), "per_metric": per_metric,
        "pairs": {"all_denominator": len(pairs), "both_resolved": sum(row["both_resolved"] for row in pairs),
                  "changed_noncomplement_all": sum(row["changed_noncomplement"] for row in pairs),
                  "both_resolved_changed_noncomplement": len(qualifying_pairs), "qualifying_draws": qualifying_draws},
        "capture_and_lead": {"capture_denominator": len(iso), "captured": captured, "resolved_captured_leads": summarize_values(lead_values),
                             "per_draw_lead": per_draw_lead, "positive_median_lead_draws": positive_draws,
                             "resolved_captured_with_undefined_lead": sum(row["resolved"] and row["T_capture"] is not None and row["primary_fractional_lead"] is None for row in iso)},
        "continuations": {"prescribed_rows": len(continuations), "evaluated": len(observed_continuations),
                          "not_captured": len(continuations) - len(observed_continuations), "ever_anchor_flip_count": flips,
                          "log_evidence_decrease_count": decreases, "monotonicity_tolerance": EVIDENCE_TOLERANCE,
                          "update_max_absolute_error": max((row["update_max_absolute_error"] for row in observed_continuations), default=0.0),
                          "passed": continuation_pass},
        "model_checks": model, "trajectory_checks": numerical,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    script_sha = digest(Path(__file__))
    require(not args.output.exists(), f"Output already exists: {args.output}")
    protocol = json.loads(args.protocol.read_text())
    require(protocol["version"] == VERSION, "Unsupported frozen protocol version")
    self_check = implementation_check()
    with np.load(args.input, allow_pickle=False) as loaded:
        raw = {name: loaded[name] for name in loaded.files}
    model = verify_model(raw, protocol)
    rows, metrics, numerical = baseline_evaluation(raw, protocol)
    continuations, continuation_metrics = continuation_evaluation(raw, protocol)
    pairs = paired_rows(rows, protocol)
    expected_pairs = len(raw["draw_ids"]) * protocol["heads_per_draw"] * 6
    require(len(pairs) == expected_pairs, "Prespecified paired denominator mismatch")
    summary = summarize(rows, pairs, continuations, protocol, model, numerical)
    summary.update({"input_sha256": digest(args.input), "protocol_sha256": digest(args.protocol), "evaluator_sha256": script_sha,
                    "implementation_check": self_check,
                    "scope": "Frozen-state independent evaluation; no training or endpoint-dependent configuration selection",
                    "timing_definitions": {"T_fit": "First legal state with mean bag E >= .95 and minimum image modal probability >= .90; no waiting window",
                                           "T_q90": "First maximum conditional COMPLETE-code posterior >= .90; defined even when hard code is illegal",
                                           "T_confirm100": "First well-fitted state whose preceding 100 updates retain the same full code",
                                           "hindsight_final_code_stability_onset": "Step of last full hard-code change, or zero if unchanged",
                                           "primary_fractional_lead": "(T_fit-T_capture)/T_fit; null if either time is missing or T_fit=0; no positive-only filtering"},
                    "test_weight_reconstruction": "W_final = W0 + Xplus (y_final-X W0); the ordinary initial null-space component is retained",
                    "label_convention": "cat=1, bird=0; no label swap", "baseline_saved_states_per_trajectory": len(raw["logits"]),
                    "continuation_saved_states_per_trajectory": len(raw["continuation_logits"])})
    args.output.mkdir(parents=True)
    write_csv(args.output / "endpoints.csv", rows)
    write_csv(args.output / "pairs.csv", pairs)
    write_csv(args.output / "continuations.csv", continuations)
    np.savez_compressed(args.output / "state_metrics.npz", **metrics, **continuation_metrics)
    write_json(args.output / "endpoints.json", rows)
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"phase": summary["phase"], "trajectories": len(rows), "pairs": summary["pairs"]}), flush=True)


if __name__ == "__main__":
    main()
