"""Continue the wrong random-initialization runs until they enter the contracting neighborhood of the wrong state.

The source is the output of run_random_initialization.py for gradient descent under the condition named by the
protocol. Every run that ended with all training concepts wrong is continued with the same objective and step
size. The neighborhood is the ball around the finite wrong state of Appendix F.3 in which gradient descent
provably contracts toward that state, so a run that enters it started in the basin of the wrong state.
A CUDA device is required: the update is replayed from a CUDA graph.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from scipy.special import expit

from run_cifar_identifiable_minima import explicit_loss
from run_cifar_identifiable_traps import fast_gradient, query_arrays, Queries, json_write, loss_and_logit_gradient


def run(protocol_path: Path, source: Path, output: Path, device: str, max_updates: int | None) -> None:
    settings = json.loads(protocol_path.read_text(encoding="utf-8"))["neighborhood_entry"]
    record = json.loads((source / "run_record.json").read_text(encoding="utf-8"))
    if record["condition"]["name"] != settings["condition"] or record["optimizer"] != "gd":
        raise ValueError("The source must be the gradient-descent run of the condition named by the protocol")
    block = settings["updates_per_block"]
    interval = settings["save_every"]
    maximum = settings["maximum_additional_updates"] if max_updates is None else min(settings["maximum_additional_updates"], max_updates)
    if interval % block or maximum % interval:
        raise ValueError("The saving interval and the budget must be multiples of the block")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    rho = record["rho"]
    with np.load(source / "states.npz", allow_pickle=False) as saved:
        x, labels = saved["training_design"], saved["labels"]
        queries = Queries(saved["pairs"], saved["counts"])
        weights_np = saved["weights"]
        eta, source_updates = float(saved["eta"]), int(saved["final_update"])
        final_logits, initial, final_heads = saved["logits"][-1], saved["initial_heads"], saved["final_heads"]
        test = {name: saved[name] for name in ("test_design", "test_labels", "test_pairs", "test_counts")}
    wrong = np.all((final_logits > 0) != labels, axis=1)
    if not np.all(wrong | np.all((final_logits > 0) == labels, axis=1)):
        raise ValueError("A source run is neither entirely correct nor entirely wrong")
    heads = np.flatnonzero(wrong)
    start_values, original_heads, source_w = final_logits[heads].copy(), initial[heads].copy(), final_heads[heads].copy()
    neighbors, targets = query_arrays(queries, len(x))
    np.testing.assert_array_equal(targets, labels)
    np.testing.assert_array_equal(labels[queries.pairs].sum(axis=1), queries.counts)
    n = len(x)
    mixed = queries.counts == 1
    assert np.all(np.bincount(queries.pairs[~mixed].ravel(), minlength=n) == 1)
    np.testing.assert_allclose(weights_np[mixed], rho / mixed.sum(), rtol=1e-14)
    np.testing.assert_allclose(weights_np[~mixed], (1-rho) / (~mixed).sum(), rtol=1e-14)
    u, singular, vt = np.linalg.svd(x, full_matrices=False)
    assert np.linalg.matrix_rank(x) == n
    # The finite wrong state: every correct label has probability v.
    delta = np.sqrt(rho*rho + 4*rho - 4)
    v = (2-rho-delta) / 4
    magnitude = -np.log(v / (1-v))
    center = -(2*labels-1) * magnitude
    # Radius of the ball in which every wrong margin stays at least b, and curvature bounds inside it.
    b = settings["minimum_wrong_margin"]
    radius = (magnitude-b) / np.linalg.norm(x, axis=1).max()
    t = np.exp(-b)
    ratio_bound = t * (1+t)**2 / (1+t*t)**2
    minimum_variance = expit(2*magnitude-b) * expit(-(2*magnitude-b))
    lower_logit = (2/n) * minimum_variance * (1-2*rho*ratio_bound)
    lower_effective = singular[-1]**2 * lower_logit
    upper_effective = singular[0]**2 / (2*n)
    assert radius > 0 and lower_effective > 0 and 0 < eta * upper_effective < 1
    center_t = torch.tensor(center, dtype=torch.float64, requires_grad=True)
    pairs_cpu, counts_cpu = torch.tensor(queries.pairs), torch.tensor(queries.counts)
    weights_cpu = torch.tensor(weights_np)
    center_loss_t = explicit_loss(center_t, pairs_cpu, counts_cpu, weights_cpu)
    center_gradient = torch.autograd.grad(center_loss_t, center_t)[0].numpy()
    center_hessian = torch.autograd.functional.hessian(
        lambda z: explicit_loss(z, pairs_cpu, counts_cpu, weights_cpu), center_t,
    ).numpy()
    assert np.max(np.abs(center_gradient @ x)) < 1e-12
    center_effective = singular[:, None] * (u.T @ center_hessian @ u) * singular[None, :]
    center_eigenvalues = np.linalg.eigvalsh(center_effective)
    assert center_eigenvalues[0] > 0 and np.max(np.abs(1-eta*center_eigenvalues)) < 1
    pseudo = (vt.T / singular) @ u.T
    target_heads = original_heads + (center-original_heads @ x.T) @ pseudo.T
    np.testing.assert_allclose(target_heads @ x.T, np.broadcast_to(center, start_values.shape), atol=1e-11)
    gram = torch.tensor(x @ x.T, device=device)
    x_t = torch.tensor(x, device=device)
    neighbors_t, targets_t = torch.tensor(neighbors, device=device), torch.tensor(targets, device=device)
    rhos_t = torch.full((len(heads), 1), rho, dtype=torch.float64, device=device)
    logits = torch.tensor(start_values, device=device)
    pairs_t, counts_t = torch.tensor(queries.pairs, device=device), torch.tensor(queries.counts, device=device)
    weights_t = torch.tensor(weights_np[None], device=device)
    _, reference_gradient = loss_and_logit_gradient(logits, pairs_t, counts_t, weights_t)
    direct = fast_gradient(logits, neighbors_t, targets_t, rhos_t)
    torch.testing.assert_close(direct, reference_gradient, atol=1e-14, rtol=1e-10)
    expected = (torch.tensor(source_w, device=device) - eta * direct @ x_t) @ x_t.T
    torch.testing.assert_close(logits - eta * direct @ gram, expected, atol=1e-10, rtol=1e-10)

    def step() -> None:
        logits.add_(fast_gradient(logits, neighbors_t, targets_t, rhos_t) @ gram, alpha=-eta)

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    logits.copy_(torch.tensor(start_values, device=device))
    for _ in range(block):
        step()
    block_reference = logits.clone()
    logits.copy_(torch.tensor(start_values, device=device))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(block):
            step()
    logits.copy_(torch.tensor(start_values, device=device))
    graph.replay()
    torch.testing.assert_close(logits, block_reference, atol=1e-12, rtol=1e-12)
    logits.copy_(torch.tensor(start_values, device=device))
    center_loss = float(center_loss_t.detach())
    json_write(output / "geometry.json", {
        "training_set": record["training_set"], "rho": rho, "step_size": eta,
        "correct_probability_at_wrong_state": float(v), "wrong_logit_magnitude": float(magnitude),
        "loss_at_wrong_state": center_loss, "radius": float(radius), "minimum_wrong_margin": b,
        "curvature_lower_bound_in_ball": float(lower_effective), "curvature_upper_bound": float(upper_effective),
        "contraction_factor_in_ball": float(1-eta*lower_effective), "source_updates": source_updates})
    started = time.monotonic()
    snapshots, steps = [], []
    first_entry = np.full(len(heads), -1, dtype=np.int64)
    stopping_fraction = settings["stop_within_fraction_of_radius"]
    with (output / "curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer: csv.DictWriter | None = None
        for added in range(0, maximum+1, interval):
            values = logits.cpu().numpy().copy()
            if not np.isfinite(values).all():
                raise FloatingPointError(f"Nonfinite continuation after {added} additional updates")
            difference = values-center
            # Distance to the wrong state in the parameter coordinates that the features can move.
            distances = np.linalg.norm(difference @ u / singular, axis=1)
            first_entry[(first_entry < 0) & (distances < radius)] = added
            loss, gradient = loss_and_logit_gradient(logits, pairs_t, counts_t, weights_t)
            gradient_norms = torch.linalg.vector_norm(gradient @ x_t, dim=1).cpu().numpy()
            losses = loss.cpu().numpy()
            accuracies = ((values > 0) == labels).mean(axis=1)
            observations = [{"head": int(head), "additional_updates": added, "total_updates": source_updates+added,
                             "distance": float(distances[i]), "distance_over_radius": float(distances[i]/radius),
                             "logit_rms_distance": float(np.sqrt(np.mean(difference[i]**2))),
                             "parameter_gradient_norm": float(gradient_norms[i]), "loss_excess": float(losses[i]-center_loss),
                             "concept_accuracy": float(accuracies[i])} for i, head in enumerate(heads)]
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(observations[0]))
                writer.writeheader()
            writer.writerows(observations)
            handle.flush()
            snapshots.append(values)
            steps.append(added)
            if added % (5*interval) == 0 or np.all(distances <= stopping_fraction*radius):
                print(json.dumps({"additional_updates": added, "inside": int((distances < radius).sum()), "runs": len(heads),
                                  "largest_distance_over_radius": float(np.max(distances/radius)),
                                  "seconds": round(time.monotonic()-started, 3)}), flush=True)
            if np.all(distances <= stopping_fraction*radius) or added == maximum:
                break
            for _ in range(interval // block):
                graph.replay()
    final = source_w + (values-start_values) @ pseudo.T
    np.testing.assert_allclose(final @ x.T, values, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(final-target_heads, axis=1), distances, atol=1e-10, rtol=1e-10)
    np.savez_compressed(output / "states.npz", logits=np.stack(snapshots), additional_updates=np.array(steps),
                        heads=heads, final_heads=final, initial_heads=original_heads, center_logits=center,
                        training_design=x, labels=labels, pairs=queries.pairs, counts=queries.counts,
                        weights=weights_np, eta=eta, first_entry=first_entry, source_updates=source_updates, **test)
    json_write(output / "endpoints.json", observations)
    json_write(output / "completed.json", {"seconds": time.monotonic()-started, "additional_updates": added,
               "runs": len(heads), "all_inside": bool(np.all(distances < radius))})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="output of run_random_initialization.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-updates", type=int, help="stop early for a quick check")
    args = parser.parse_args()
    run(args.protocol, args.source, args.output, args.device, args.max_updates)


if __name__ == "__main__":
    main()
