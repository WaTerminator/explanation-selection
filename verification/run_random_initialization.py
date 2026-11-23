"""Train randomly initialized heads on the identifiable query system.

One call trains the 32 random heads of one training set under one condition (a weight rho of the mixed-pair
loss, or a restriction of the same-class queries to few inputs) with full-batch gradient descent or with SGD
over sampled queries. No concept label is used to initialize or to train the heads. A CUDA device is required:
the update is replayed from a CUDA graph.
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
from torch import Tensor

from run_cifar_identifiable_traps import (
    json_write, loss_and_logit_gradient, make_queries, metrics, query_weights, sparse_queries, test_queries,
)


def full_gradient(logits: Tensor, neighbors: Tensor, anchor_targets: Tensor,
                  anchor_weights: Tensor, mixed_weight: float) -> Tensor:
    p = torch.sigmoid(logits)
    conditional = torch.sigmoid(logits[:, :, None] - logits[:, neighbors]).sum(dim=2)
    return anchor_weights * (p - anchor_targets) + mixed_weight * (neighbors.shape[1] * p - conditional)


def sampled_gradient(logits: Tensor, pairs: Tensor, counts: Tensor,
                     selected: Tensor) -> Tensor:
    rows = torch.arange(len(logits), device=logits.device)[:, None]
    locations = pairs[selected]
    left, right = logits[rows, locations[:, :, 0]], logits[rows, locations[:, :, 1]]
    c = counts[selected]
    target_left = torch.where(c == 1, torch.sigmoid(left - right), c / 2)
    target_right = torch.where(c == 1, 1 - target_left, c / 2)
    # A dense reduction keeps the sampled updates deterministic inside CUDA graphs.
    positions = torch.arange(logits.shape[1], device=logits.device)
    left_mask = locations[:, :, 0, None] == positions
    right_mask = locations[:, :, 1, None] == positions
    terms = ((torch.sigmoid(left) - target_left)[:, :, None] * left_mask
             + (torch.sigmoid(right) - target_right)[:, :, None] * right_mask)
    return terms.mean(dim=1)


def run(protocol_path: Path, features_path: Path, training_set: int, condition_name: str, optimizer: str,
        device: str, output: Path, max_updates: int | None) -> None:
    settings = json.loads(protocol_path.read_text(encoding="utf-8"))["random_initialization"]
    condition = next(c for c in settings["conditions"] if c["name"] == condition_name)
    if training_set not in settings["training_sets"]:
        raise ValueError("The protocol does not list this training set")
    head_count = settings["heads"]
    updates = settings["updates"] if max_updates is None else min(settings["updates"], max_updates)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(settings["batch_seed"] + training_set)
    with np.load(features_path, allow_pickle=False) as data:
        labels_pool = (data["train_labels"] == 3).astype(np.int64)
        rng = np.random.default_rng(settings["data_seed"])
        half = settings["inputs"] // 2
        allocations = [rng.permutation(np.flatnonzero(labels_pool == label)) for label in (0, 1)]
        selection = np.concatenate([ids[training_set * half:(training_set + 1) * half] for ids in allocations])
        labels = labels_pool[selection]
        x = np.column_stack((data["train_features"][selection].astype(np.float64), np.ones(len(labels))))
        test_x = np.column_stack((data["test_features"].astype(np.float64), np.ones(len(data["test_labels"]))))
        test_labels = (data["test_labels"] == 3).astype(np.int64)
        input_ids = data["train_indices"][selection].copy()
    dense = make_queries(labels, settings["mixed_degree"], settings["pair_seed"] + training_set)
    # Keep the same-class queries of the first identified_inputs / 4 pairs of each class, and every mixed pair.
    queries = sparse_queries(dense, condition["identified_inputs"] // 4, labels)
    mixed = queries.counts == 1
    rho = float(mixed.mean()) if condition["rho"] is None else condition["rho"]
    weights_np = query_weights(queries, np.array([rho]))[0]
    n, d = x.shape
    rng = np.random.default_rng(settings["head_seed"] + training_set)
    initial_heads = rng.uniform(-1 / np.sqrt(d - 1), 1 / np.sqrt(d - 1), size=(head_count, d))
    initial_logits = initial_heads @ x.T
    gram_np = x @ x.T
    eta = float(1 / (1.5 * np.max(np.sum(x * x, axis=1))))
    adjacent: list[list[int]] = [[] for _ in range(n)]
    targets = np.zeros(n)
    anchor_weights = np.zeros(n)
    for (left, right), count, weight in zip(queries.pairs, queries.counts, weights_np):
        if count == 1:
            adjacent[int(left)].append(int(right))
            adjacent[int(right)].append(int(left))
        else:
            targets[[left, right]] = count / 2
            anchor_weights[[left, right]] = weight
    neighbors = torch.tensor(adjacent, device=device)
    target_t = torch.tensor(targets, device=device)
    anchor_t = torch.tensor(anchor_weights, device=device)
    mixed_weight = rho / int(mixed.sum())
    x_t = torch.tensor(x, device=device)
    gram = torch.tensor(gram_np, device=device)
    logits = torch.tensor(initial_logits, device=device)
    pairs = torch.tensor(queries.pairs, device=device)
    counts = torch.tensor(queries.counts, device=device)
    weights = torch.tensor(weights_np[None, :], device=device)
    cdf = torch.tensor(np.cumsum(weights_np), device=device)
    cdf[-1] = 1
    batch_size = settings["queries_per_sgd_update"]
    _, reference = loss_and_logit_gradient(logits, pairs, counts, weights)
    direct = full_gradient(logits, neighbors, target_t, anchor_t, mixed_weight)
    torch.testing.assert_close(direct, reference, atol=1e-13, rtol=1e-12)
    # Enumerating the queries checks the expectation of the sampled gradient.
    expected = torch.zeros_like(logits)
    for q in range(len(queries.counts)):
        indices = torch.full((head_count, 1), q, device=device, dtype=torch.int64)
        expected += weights_np[q] * sampled_gradient(logits, pairs, counts, indices)
    torch.testing.assert_close(expected, reference, atol=1e-13, rtol=1e-12)
    # A sampled logit update must equal the corresponding parameter update.
    chosen = torch.randint(len(queries.counts), (head_count, batch_size), device=device)
    g = sampled_gradient(logits, pairs, counts, chosen)
    parameter_step = torch.tensor(initial_heads, device=device) - eta * (g @ x_t)
    torch.testing.assert_close(logits - eta * (g @ gram), parameter_step @ x_t.T, atol=1e-11, rtol=1e-11)
    previous = logits > 0
    last_change = torch.zeros(head_count, device=device, dtype=torch.int64)
    step_number = torch.zeros((), device=device, dtype=torch.int64)
    selected = torch.zeros((head_count, batch_size), device=device, dtype=torch.int64)

    def step() -> None:
        if optimizer == "gd":
            gradient = full_gradient(logits, neighbors, target_t, anchor_t, mixed_weight)
        else:
            uniforms = torch.rand((head_count, batch_size), device=device, dtype=torch.float64)
            selected.copy_(torch.searchsorted(cdf, uniforms))
            gradient = sampled_gradient(logits, pairs, counts, selected)
        logits.add_(gradient @ gram, alpha=-eta)
        step_number.add_(1)
        now = logits > 0
        last_change.copy_(torch.where((now != previous).any(dim=1), step_number, last_change))
        previous.copy_(now)

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    if optimizer == "sgd":
        graph.replay()
        first = selected.clone()
        graph.replay()
        if torch.equal(first, selected):
            raise AssertionError("The replayed update reused its sampled queries")
    logits.copy_(torch.tensor(initial_logits, device=device))
    previous.copy_(logits > 0)
    step_number.zero_()
    last_change.zero_()
    if int(np.linalg.matrix_rank(x)) != n:
        raise ValueError("The features do not have full row rank")
    pseudo = np.linalg.pinv(x)
    test_q = test_queries(test_labels, settings["test_pair_seed"])
    json_write(output / "run_record.json", {
        "training_set": training_set, "condition": condition, "optimizer": optimizer, "rho": rho, "step_size": eta,
        "identified_inputs": int(np.count_nonzero(anchor_weights)), "queries": len(queries.counts), "heads": head_count,
        "updates": updates, "queries_per_update": len(queries.counts) if optimizer == "gd" else batch_size,
        "torch": torch.__version__, "device": device})
    start = time.monotonic()
    snapshots = []
    rows: list[dict] = []
    with (output / "curves.csv").open("w", newline="") as handle:
        writer: csv.DictWriter | None = None
        for iteration in range(updates + 1):
            if iteration % settings["save_every"] == 0 or iteration == updates:
                values = logits.cpu().numpy().copy()
                if not np.isfinite(values).all():
                    raise FloatingPointError(f"Nonfinite logits at update {iteration}")
                heads = initial_heads + (values - initial_logits) @ pseudo.T
                np.testing.assert_allclose(heads @ x.T, values, atol=1e-10, rtol=1e-10)
                loss, gradient = loss_and_logit_gradient(logits, pairs, counts, weights)
                norms = torch.linalg.vector_norm(gradient @ x_t, dim=1).cpu().numpy()
                loss_np = loss.cpu().numpy()
                hard = values > 0
                correct = hard[:, queries.pairs].sum(axis=2) == queries.counts
                train_acc = (hard == labels).mean(axis=1)
                test = metrics(heads @ test_x.T, test_labels, test_q)
                changed = last_change.cpu().numpy()
                rows = [{"training_set": training_set, "condition": condition_name, "optimizer": optimizer, "head": i,
                         "update": iteration, "loss": float(loss_np[i]), "gradient_norm": float(norms[i]),
                         "concept_accuracy": float(train_acc[i]), "violated_queries": int((~correct[i]).sum()),
                         "test_concept_accuracy": float(test["concept_acc"][i]),
                         "last_label_change": int(changed[i])} for i in range(head_count)]
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                snapshots.append(values)
                if iteration % (10 * settings["save_every"]) == 0 or iteration == updates:
                    print(json.dumps({"training_set": training_set, "condition": condition_name, "optimizer": optimizer,
                                      "update": iteration, "all_correct": int((train_acc == 1).sum()),
                                      "all_wrong": int((train_acc == 0).sum()),
                                      "seconds": round(time.monotonic() - start, 1)}), flush=True)
            if iteration < updates:
                graph.replay()
    np.savez_compressed(output / "states.npz", initial_heads=initial_heads, final_heads=heads,
                        logits=np.stack(snapshots), training_design=x, test_design=test_x,
                        labels=labels, test_labels=test_labels, input_ids=input_ids,
                        pairs=queries.pairs, counts=queries.counts, weights=weights_np,
                        test_pairs=test_q.pairs, test_counts=test_q.counts, eta=eta,
                        last_label_change=changed, final_update=updates)
    json_write(output / "endpoints.json", rows)
    json_write(output / "completed.json", {"seconds": time.monotonic() - start, "runs": head_count, "updates": updates})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--training-set", type=int, required=True)
    parser.add_argument("--condition", required=True, help="name of a condition of the protocol")
    parser.add_argument("--optimizer", choices=("gd", "sgd"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-updates", type=int, help="stop early for a quick check")
    args = parser.parse_args()
    run(args.protocol, args.features, args.training_set, args.condition, args.optimizer, args.device, args.output,
        args.max_updates)


if __name__ == "__main__":
    main()
