"""Repeat the six-bag selection experiment at one step size and record when capture occurs.

The training sets, initial heads and isospectral Grams are those of the evaluation sets of
protocols/selection.json. One call handles one step size, given as a fraction of the global step bound, and
trains all 96 runs for the same cumulative step. At the first capture of every run, it applies the three Gram
changes of Appendix G.3 and records whether a label changes.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from evaluate_cifar_llp import pack_codes
from run_cifar_identifiable_traps import json_write
from run_cifar_llp import allocate, candidates, design, metric_family, quantities
from run_evidence_capture import adverse_direction, metric_product


Array = NDArray[np.float64]


def fitted(logits: Array, log_bag: Array, counts: NDArray[np.int64]) -> NDArray[np.bool_]:
    legal = ((logits > 0).reshape(len(logits), -1, 4).sum(2) == counts).all(1)
    return legal & (np.exp(log_bag).mean(1) >= .95) & (expit(np.abs(logits)).min(1) >= .9)


def selection_runs(features_path: Path, selection: dict) -> tuple[NDArray[np.int64], NDArray[np.int64], Array, Array]:
    """Counts, training-set ids, initial logits and Grams of the evaluation runs of the selection experiment."""
    counts = np.array(selection["bag_counts"], dtype=np.int64)
    sets = np.array(selection["evaluation_draws"], dtype=np.int64)
    data = np.load(features_path)
    allocations = allocate(data["train_labels"], counts, 9, selection["data_seed"])
    x = np.stack([design(data["train_features"][allocations[index]]) for index in sets])
    # The four isospectral Grams of each training set, built as in run_cifar_llp.py.
    metrics = np.stack([metric_family(values, int(index), 2026090800)[0] for index, values in zip(sets, x)])[:, :4]
    width = x.shape[2]
    weights = np.stack([np.stack([np.random.default_rng(selection["head_seed_base"] + 100 * int(index) + head)
                                 .uniform(-1 / np.sqrt(width), 1 / np.sqrt(width), width)
                                 for head in range(selection["heads_per_draw"])]) for index in sets])
    return counts, sets, np.einsum("dnp,dhp->dhn", x, weights), metrics


def run(protocol_path: Path, selection_path: Path, features_path: Path, fraction: float, output: Path) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if fraction not in protocol["fractions_of_the_step_bound"]:
        raise ValueError("The protocol does not list this fraction of the step bound")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    counts, sets, initial, metrics = selection_runs(features_path, json.loads(selection_path.read_text(encoding="utf-8")))
    # Run index = 16 * training set + 4 * head + Gram.
    y = np.repeat(initial[:, :, None], 4, axis=2).reshape(-1, initial.shape[-1])
    grams = np.repeat(metrics[:, None], initial.shape[1], axis=1).reshape(-1, y.shape[1], y.shape[1])
    other = np.repeat(metrics[:, None, [1, 0, 3, 2]], initial.shape[1], axis=1).reshape(grams.shape)
    n = y.shape[1]
    eta = fraction * protocol["step_bound"]
    np.testing.assert_allclose(protocol["step_bound"], 4 / (4 * (n + 1)), rtol=0, atol=1e-14)
    eigenvalues = np.linalg.eigvalsh(grams)
    if eigenvalues.min() < .25 - 1e-10 or eigenvalues.max() > 4 + 1e-10 or eta * 4 * (n + 1) >= 4:
        raise AssertionError("The spectral bounds or the strict step bound fail")
    steps = math.ceil(protocol["training_time"] / eta)
    codes, valid = candidates(counts)
    history = np.lib.format.open_memmap(output / "baseline.npy", mode="w+", dtype=np.float64,
                                       shape=(steps + 1, *y.shape))
    capture = np.full(len(y), -1, dtype=np.int64)
    first_fit = capture.copy()
    last_change = np.zeros(len(y), dtype=np.int64)
    previous_code = pack_codes(y > 0)
    previous_evidence = np.full(len(y), -np.inf)
    minimum_increment = 0.0
    for step in range(steps + 1):
        history[step] = y
        state = quantities(y, codes, valid)
        evidence = state.log_bag_evidence.sum(1)
        hard = pack_codes(y > 0)
        last_change[hard != previous_code] = step
        capture[(capture < 0) & (evidence > -np.log(2))] = step
        first_fit[(first_fit < 0) & fitted(y, state.log_bag_evidence, counts)] = step
        minimum_increment = min(minimum_increment, float((evidence - previous_evidence).min()))
        if not np.isfinite(y).all() or minimum_increment < -1e-10:
            raise AssertionError(f"Invalid or descending evidence at update {step}")
        previous_code, previous_evidence = hard, evidence
        if step < steps:
            y += eta * np.einsum("bij,bj->bi", grams, state.residual)
        if step % 4000 == 0 or step == steps:
            print(json.dumps({"phase": "training", "fraction": fraction, "update": step,
                              "captured": int((capture >= 0).sum())}), flush=True)
    history.flush()

    # Three Gram changes at the first capture of every captured run.
    eligible = np.flatnonzero(capture > 0)
    base_indices = np.repeat(eligible, 3)
    intervention = np.tile(np.arange(3), len(eligible))
    current = np.repeat(history[capture[eligible], eligible][:, None], 3, axis=1).reshape(-1, n)
    start_y = current.copy()
    anchor = np.where(current > 0, 1.0, -1.0)
    fixed = adverse_direction(current, quantities(current, codes, valid).residual, anchor)
    other_grams = other[base_indices]
    horizon = math.ceil(protocol["continuation_time"] / eta)
    code_history = np.empty((horizon + 1, len(current)), dtype=np.uint32)
    evidence_history = np.empty((horizon + 1, len(current)))
    first_flip = np.full(len(current), -1, dtype=np.int64)
    start_code = pack_codes(current > 0)
    for step in range(horizon + 1):
        state = quantities(current, codes, valid)
        code_history[step] = pack_codes(current > 0)
        evidence_history[step] = state.log_bag_evidence.sum(1)
        first_flip[(first_flip < 0) & (code_history[step] != start_code)] = step
        if not np.isfinite(current).all():
            raise FloatingPointError(f"Nonfinite continuation at update {step}")
        if step and np.min(evidence_history[step] - evidence_history[step - 1]) < -1e-10:
            raise AssertionError(f"The evidence of a continuation decreased at update {step}")
        if step < horizon:
            change = np.empty_like(current)
            opposite, one_fixed, adaptive = (intervention == i for i in range(3))
            change[opposite] = np.einsum("bij,bj->bi", other_grams[opposite], state.residual[opposite])
            change[one_fixed] = 4 * metric_product(state.residual[one_fixed], fixed[one_fixed], .25 / 4)
            slow = adverse_direction(current[adaptive], state.residual[adaptive], anchor[adaptive])
            change[adaptive] = 4 * metric_product(state.residual[adaptive], slow, .25 / 4)
            current += eta * change
        if step % 4000 == 0 or step == horizon:
            print(json.dumps({"phase": "continuation", "fraction": fraction, "update": step,
                              "label_changes": int((first_flip >= 0).sum())}), flush=True)
    np.savez_compressed(output / "continuations.npz", initial=start_y, final=current, codes=code_history,
                        log_evidence=evidence_history, first_flip=first_flip, baseline_index=base_indices,
                        intervention=intervention)
    np.savez(output / "summary.npz", counts=counts, training_sets=sets, capture=capture,
             first_fit=first_fit, last_change=last_change, grams=grams, eta=eta,
             final_fitted=fitted(y, quantities(y, codes, valid).log_bag_evidence, counts))
    json_write(output / "completed.json", {"fraction_of_the_step_bound": fraction, "step_size": eta, "updates": steps,
               "continuation_updates": horizon, "runs": len(y), "captured": len(eligible),
               "continuations": len(current), "continuations_changing_a_label": int((first_flip >= 0).sum()),
               "minimum_log_evidence_increment": minimum_increment, "seconds": time.monotonic() - started})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection-protocol", type=Path, required=True, help="protocols/selection.json")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--fraction", type=float, required=True, help="step size as a fraction of the step bound")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol, args.selection_protocol, args.features, args.fraction, args.output)


if __name__ == "__main__":
    main()
