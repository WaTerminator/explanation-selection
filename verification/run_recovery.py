"""Recover the correct concepts from wrong states by temporarily lowering the weight of the mixed queries.

The source is the output of run_neighborhood_entry.py: wrong states inside the contracting neighborhood of the
finite wrong state. From every such state, two trainings run for the same number of updates. One keeps the
original weight rho. The other trains with a lower weight and, at regular checks, restores the original weight
as soon as all observed counts are correct and the original loss is below 2 log(2) / n (Appendix F.2). The rule
uses the queries, the observed counts and the loss only, not the concept labels.
A CUDA device is required: the update is replayed from a CUDA graph.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from numpy.typing import NDArray

from run_cifar_identifiable_traps import fast_gradient, query_arrays, Queries, json_write, loss_and_logit_gradient, query_weights


def rule_met(losses: NDArray[np.float64], violations: NDArray[np.int64], n: int) -> NDArray[np.bool_]:
    """All observed counts correct and original loss below 2 log(2) / n."""
    if n <= 0 or losses.shape != violations.shape or not np.isfinite(losses).all():
        raise ValueError("The rule needs finite losses and matching count violations")
    return (violations == 0) & (losses < 2 * np.log(2) / n)


def check_rule() -> None:
    threshold = 2 * np.log(2) / 256
    losses = np.array([threshold / 2, threshold / 2, threshold, threshold * 2])
    np.testing.assert_array_equal(rule_met(losses, np.array([0, 1, 0, 0]), 256),
                                  np.array([True, False, False, False]))


def run(protocol_path: Path, source: Path, output: Path, device: str) -> None:
    settings = json.loads(protocol_path.read_text(encoding="utf-8"))["recovery"]
    block, interval, maximum = settings["updates_per_block"], settings["check_every"], settings["updates"]
    rho, low = settings["original_rho"], settings["low_rho"]
    if interval % block or maximum % interval or not 0 < low < rho < 1:
        raise ValueError("Invalid intervals or weights")
    check_rule()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    with np.load(source / "states.npz", allow_pickle=False) as data:
        x, labels = data["training_design"], data["labels"]
        queries = Queries(data["pairs"], data["counts"])
        original_weights, eta = data["weights"], float(data["eta"])
        heads, start_heads, start = data["heads"], data["final_heads"], data["logits"][-1]
        center = data["center_logits"]
        test = {name: data[name] for name in ("test_design", "test_labels", "test_pairs", "test_counts")}
    geometry = json.loads((source / "geometry.json").read_text(encoding="utf-8"))
    u, singular, vt = np.linalg.svd(x, full_matrices=False)
    pseudo = (vt.T / singular) @ u.T
    # Every start lies inside the contracting neighborhood of the wrong state.
    assert np.all(np.linalg.norm((start - center) @ u / singular, axis=1) < geometry["radius"])
    np.testing.assert_allclose(start_heads @ x.T, start, atol=1e-10)
    step_ratio = eta * singular[0] ** 2 / (2 * len(x))
    if not 0 < step_ratio < 2:
        raise ValueError(f"The step-size condition fails: {step_ratio}")
    neighbors, targets = query_arrays(queries, len(x))
    np.testing.assert_allclose(original_weights, query_weights(queries, np.array([rho]))[0], atol=1e-15)
    # Rows 0..N-1 keep the original weight; rows N..2N-1 start with the low weight.
    count = len(heads)
    training = ["unchanged_gd"] * count + ["temporary_reweighting_with_stopping_rule"] * count
    reweighted = np.arange(2 * count) >= count
    initial = np.concatenate((start, start))
    initial_heads = np.concatenate((start_heads, start_heads))
    logits = torch.tensor(initial, device=device)
    rhos = torch.tensor(np.where(reweighted, low, rho)[:, None], dtype=torch.float64, device=device)
    gram, design = torch.tensor(x @ x.T, device=device), torch.tensor(x, device=device)
    neighbors_t, targets_t = torch.tensor(neighbors, device=device), torch.tensor(targets, device=device)
    pairs, counts = torch.tensor(queries.pairs, device=device), torch.tensor(queries.counts, device=device)
    weights = torch.tensor(original_weights[None], device=device)
    gradient_errors: list[float] = []

    def gradient_check() -> None:
        """The fast gradient must equal the per-query posterior gradient at the current weights."""
        current_weights = query_weights(queries, rhos[:, 0].cpu().numpy())
        _, direct = loss_and_logit_gradient(logits, pairs, counts, torch.tensor(current_weights, device=device))
        fast = fast_gradient(logits, neighbors_t, targets_t, rhos)
        torch.testing.assert_close(fast, direct, atol=1e-13, rtol=1e-12)
        gradient_errors.append(float((fast - direct).abs().max()))

    gradient_check()

    def step() -> None:
        logits.add_(fast_gradient(logits, neighbors_t, targets_t, rhos) @ gram, alpha=-eta)

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    logits.copy_(torch.tensor(initial, device=device))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(block):
            step()
    logits.copy_(torch.tensor(initial, device=device))
    expected = logits.clone()
    for _ in range(block):
        expected.add_(fast_gradient(expected, neighbors_t, targets_t, rhos) @ gram, alpha=-eta)
    graph.replay()
    torch.testing.assert_close(logits, expected, atol=1e-12, rtol=1e-12)
    logits.copy_(torch.tensor(initial, device=device))
    np.testing.assert_array_equal(logits.cpu().numpy(), initial)
    snapshots: list[NDArray[np.float64]] = []
    curves: list[dict] = []
    restored = np.full(2 * count, -1, dtype=np.int64)
    for current in range(0, maximum + 1, interval):
        values = logits.cpu().numpy().copy()
        if not np.isfinite(values).all():
            raise FloatingPointError(f"Nonfinite state at update {current}")
        loss, gradient = loss_and_logit_gradient(logits, pairs, counts, weights)
        losses = loss.cpu().numpy()
        violations = ((values > 0)[:, queries.pairs].sum(axis=2) != queries.counts).sum(axis=1)
        selected = reweighted & (restored < 0) & rule_met(losses, violations, len(x))
        if selected.any():
            restored[selected] = current
            rhos[torch.tensor(selected, device=device)] = rho
            gradient_check()
            print(json.dumps({"update": current, "weight_restored_for_heads": heads[selected[count:]].tolist()}), flush=True)
        norm = torch.linalg.vector_norm(gradient @ design, dim=1).cpu().numpy()
        rows = [{"training": training[i], "head": int(heads[i % count]), "update": current,
                 "original_loss": float(losses[i]), "violated_counts": int(violations[i]),
                 "concept_accuracy": float(((values[i] > 0) == labels).mean()), "gradient_norm": float(norm[i]),
                 "update_at_which_the_original_weight_is_restored": int(restored[i])} for i in range(2 * count)]
        snapshots.append(values)
        curves.extend(rows)
        if current % 50000 == 0 or current == maximum:
            print(json.dumps({"update": current, "all_counts_correct": int((violations == 0).sum()),
                              "weight_restored": int((restored >= 0).sum()), "seconds": time.monotonic() - started}), flush=True)
        if current == maximum:
            break
        for _ in range(interval // block):
            graph.replay()
    final_heads = initial_heads + (values - initial) @ pseudo.T
    np.testing.assert_allclose(final_heads @ x.T, values, atol=1e-10)
    np.savez_compressed(output / "states.npz", logits=np.stack(snapshots), updates=np.arange(0, maximum + 1, interval),
                        restored=restored, reweighted=reweighted, heads=np.concatenate((heads, heads)),
                        final_heads=final_heads, training_design=x, labels=labels, pairs=queries.pairs,
                        counts=queries.counts, original_weights=original_weights, eta=eta,
                        test_logits=final_heads @ test["test_design"].T, **test)
    json_write(output / "curves.json", curves)
    json_write(output / "endpoints.json", rows)
    json_write(output / "completed.json", {"updates": maximum, "starts": count, "step_size_ratio": float(step_ratio),
               "largest_gradient_difference": max(gradient_errors), "seconds": time.monotonic() - started})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="output of run_neighborhood_entry.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.protocol, args.source, args.output, args.device)


if __name__ == "__main__":
    main()
