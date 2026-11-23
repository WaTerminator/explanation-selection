"""Train the classifier head with SGD and Adam on frozen features, under bag-count supervision.

One call handles one training set at one size and trains all initial heads with both optimizers. The heads,
the data and the minibatch order are shared by the two optimizers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from joint_training_common import state
from run_cifar_identifiable_traps import json_write
from run_cifar_llp import allocate, candidates, design


def train(initial: NDArray[np.float64], x: Tensor, codes: Tensor, valid: Tensor,
          counts: NDArray[np.int64], labels: NDArray[np.int64], test_x: Tensor,
          test_labels: NDArray[np.int64], test_bags: NDArray[np.int64],
          method: str, rate: float, epochs: int, save_every: int, bags_per_minibatch: int,
          batch_seed: int, training_set: int, output: Path) -> list[dict]:
    w = torch.nn.Parameter(torch.tensor(initial, device=x.device))
    optimizer = (torch.optim.Adam([w], lr=rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
                 if method == "adam" else torch.optim.SGD([w], lr=rate, momentum=0, weight_decay=0))
    bag_count = len(codes)
    rng = np.random.default_rng(batch_seed + training_set)
    batch_bags = min(bags_per_minibatch, bag_count)
    label_history: list[NDArray[np.bool_]] = []
    rows: list[dict] = []
    updates = 0
    with (output / f"{method}-curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer: csv.DictWriter | None = None
        for epoch in range(epochs + 1):
            measure = epoch % save_every == 0 or epoch >= epochs - 100
            if measure:
                with torch.no_grad():
                    loss, _, logits, log_bag = state(w, x, codes, valid)
                    loss_np = loss.cpu().numpy()
                    z = logits.cpu().numpy()
                    log_bag_np = log_bag.cpu().numpy()
                    confidence = torch.sigmoid(logits.abs()).amin(dim=1).cpu().numpy()
                    test_hard = ((w @ test_x.T) > 0).cpu().numpy()
                finite = np.isfinite(z).all(axis=1) & np.isfinite(loss_np)
                hard = z > 0
                legal = (hard.reshape(len(w), bag_count, 4).sum(axis=2) == counts).all(axis=1)
                mean_evidence = np.exp(log_bag_np).mean(axis=1)
                fitted = finite & legal & (mean_evidence >= 0.95) & (confidence >= 0.90)
                if epoch >= epochs - 100:
                    label_history.append(hard.copy())
                stable = np.zeros(len(w), dtype=bool)
                if len(label_history) == 101:
                    stable = np.all(np.stack(label_history) == hard[None], axis=(0, 2))
                test_counts = test_labels[test_bags].sum(axis=1)
                test_reasoning = (test_hard[:, test_bags].sum(axis=2) == test_counts).mean(axis=1)
                rows = []
                for head in range(len(w)):
                    good = bool(finite[head])
                    rows.append({
                        "training_set": training_set, "training_inputs": len(x), "optimizer": method, "step_size": rate,
                        "head": head, "epoch": epoch, "updates": updates,
                        "loss": float(loss_np[head]) if good else None,
                        "satisfies_all_counts": bool(legal[head] and good), "fitted": bool(fitted[head]),
                        "log_global_evidence": float(log_bag_np[head].sum()) if good else None,
                        "mean_bag_evidence": float(mean_evidence[head]) if good else None,
                        "minimum_confidence": float(confidence[head]) if good else None,
                        "train_accuracy": float(np.mean(hard[head] == labels)) if good else None,
                        "test_accuracy": float(np.mean(test_hard[head] == test_labels)) if good else None,
                        "test_count_accuracy": float(test_reasoning[head]) if good else None,
                        "stable_last_100_epochs": bool(stable[head]),
                    })
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                if epoch % (5 * save_every) == 0 or epoch == epochs:
                    print(json.dumps({"optimizer": method, "epoch": epoch, "fitted": int(fitted.sum()),
                                      "loss": [row["loss"] for row in rows]}), flush=True)
                if not finite.all():
                    break
            if epoch == epochs:
                break
            optimizer.param_groups[0]["lr"] = rate * (0.01 + 0.99 * (1 + math.cos(math.pi * epoch / epochs)) / 2)
            permutation = rng.permutation(bag_count)
            for begin in range(0, bag_count, batch_bags):
                bag_ids = torch.tensor(permutation[begin:begin + batch_bags], device=x.device)
                image_ids = (4 * bag_ids[:, None] + torch.arange(4, device=x.device)).flatten()
                with torch.no_grad():
                    _, gradient, _, _ = state(w, x[image_ids], codes[bag_ids], valid[bag_ids])
                w.grad = gradient
                optimizer.step()
                updates += 1
    np.savez_compressed(output / f"{method}-state.npz", initial_heads=initial,
                        final_heads=w.detach().cpu().numpy(), training_design=x.cpu().numpy(),
                        labels=labels, counts=counts, test_bags=test_bags, final_epoch=epoch)
    json_write(output / f"{method}-endpoints.json", rows)
    return rows


def run(protocol_path: Path, features_path: Path, training_set: int, size: int, output: Path, device: str) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if str(size) not in protocol["sizes"] or training_set not in protocol["training_sets"]:
        raise ValueError("The protocol does not list this size or training set")
    settings = protocol["sizes"][str(size)]
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    with np.load(features_path, allow_pickle=False) as features:
        full_counts = np.resize(np.array([1, 2, 3], dtype=np.int64), protocol["allocation_size"] // 4)
        allocation = allocate(features["train_labels"], full_counts, protocol["allocation_draws"],
                              protocol["data_seed"])[training_set][:size]
        x_np = design(features["train_features"][allocation])
        labels = (features["train_labels"][allocation] == 3).astype(np.int64)
        test_np = design(features["test_features"])
        test_labels = (features["test_labels"] == 3).astype(np.int64)
    counts = full_counts[:size // 4]
    np.testing.assert_array_equal(labels.reshape(-1, 4).sum(axis=1), counts)
    codes_np, valid_np = candidates(counts)
    initial = np.stack([np.random.default_rng(protocol["head_seed"] + 100 * training_set + head)
                        .uniform(-1 / np.sqrt(x_np.shape[1]), 1 / np.sqrt(x_np.shape[1]), x_np.shape[1])
                        for head in range(protocol["heads"])])
    test_bags = np.random.default_rng(protocol["test_bag_seed"]).permutation(len(test_labels)).reshape(-1, 4)
    x, test_x = torch.tensor(x_np, device=device), torch.tensor(test_np, device=device)
    codes, valid = torch.tensor(codes_np, device=device), torch.tensor(valid_np, device=device)
    started = time.monotonic()
    endpoints: list[dict] = []
    for method, rate in settings["step_sizes"].items():
        endpoints.extend(train(initial, x, codes, valid, counts, labels, test_x, test_labels, test_bags, method, rate,
                               settings["epochs"], settings["save_every"], protocol["bags_per_minibatch"],
                               protocol["batch_seed"], training_set, output))
    json_write(output / "completed.json", {"training_set": training_set, "training_inputs": size,
                                          "runs": len(endpoints), "seconds": time.monotonic() - started})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--training-set", type=int, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.protocol, args.features, args.training_set, args.size, args.output, args.device)


if __name__ == "__main__":
    main()
