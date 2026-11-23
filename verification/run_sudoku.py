"""Train a ResNet-18 on 4x4 Sudoku boards whose cells are CIFAR-10 images of four classes.

One call trains one model on one training set. With `--supervision rules` the supervision consists of the Sudoku
rules and, on the chosen fraction of the boards, the class of the top-left cell; training minimizes the
cross-entropy against the exact posterior over the legal completions of each board. With `--supervision labels`
the same network is trained on the true class of every cell, with the same boards and minibatches; the
manuscript uses this as the reference.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torchvision
from torch import Tensor, nn
from torchvision.datasets import CIFAR10

from run_cifar_identifiable_traps import json_write
from sudoku_rules import candidates, complete_grids, legal_grids, posterior

PERMUTATIONS = [list(p) for p in itertools.permutations(range(4))]


def class_orders(labels: np.ndarray, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.permutation(np.flatnonzero(labels == label)) for label in range(4)]


def build_boards(protocol: dict, data_root: Path, training_set: int | None, device: str):
    """Images, true grids and CIFAR-10 indices of one collection of boards; `None` gives the test boards."""
    test = training_set is None
    raw = CIFAR10(str(data_root), train=not test, download=False)
    labels = np.asarray(raw.targets)
    boards = protocol["test_boards"] if test else protocol["boards"]
    per_class = 4 * boards
    if test:
        pools = [order[:per_class] for order in class_orders(labels, protocol["test_split_seed"])]
        seed = protocol["test_board_seed"]
    else:
        block = protocol["images_per_class_and_set"]
        start = block * training_set
        pools = [order[start:start + block][:per_class] for order in class_orders(labels, protocol["split_seed"])]
        seed = protocol["board_seed"] + training_set
    if any(len(pool) != per_class for pool in pools):
        raise ValueError("Image pool too small")
    rng = np.random.default_rng(seed)
    grids = complete_grids()[rng.integers(0, 288, size=boards)]
    ids = np.empty((boards, 16), dtype=np.int64)
    for label in range(4):
        ids[grids == label] = rng.permutation(pools[label])
    if len(np.unique(ids)) != ids.size:
        raise AssertionError("An image is repeated within a board collection")
    data = torch.tensor(raw.data[ids.flatten()]).permute(0, 3, 1, 2).to(device, torch.float32) / 255
    mean = torch.tensor(protocol["normalization"][0], device=device).view(1, 3, 1, 1)
    std = torch.tensor(protocol["normalization"][1], device=device).view(1, 3, 1, 1)
    truth = torch.tensor(grids, device=device)
    assert legal_grids(truth).all()
    return (data - mean) / std, truth, ids


def revealed_boards(truth: Tensor, fraction: float, seed: int) -> Tensor:
    """Choose the boards that reveal their top-left class, cycling over the four classes."""
    boards = len(truth)
    wanted = boards if fraction >= 1 else max(4, int(round(fraction * boards)))
    rng = np.random.default_rng(seed)
    first = truth[:, 0].cpu().numpy()
    by_class = [rng.permutation(np.flatnonzero(first == label)).tolist() for label in range(4)]
    chosen: list[int] = []
    while len(chosen) < wanted:
        for label in range(4):
            if by_class[label] and len(chosen) < wanted:
                chosen.append(by_class[label].pop())
    mask = torch.zeros(boards, dtype=torch.bool, device=truth.device)
    mask[torch.tensor(chosen, device=truth.device)] = True
    return mask


def make_model(size: int, device: str) -> nn.Module:
    network = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    network.fc = nn.Linear(512, 4)
    return nn.Sequential(nn.Upsample(size=(size, size), mode="bilinear", align_corners=False), network).to(device)


@torch.no_grad()
def full_logits(model: nn.Module, images: Tensor) -> Tensor:
    model.eval()
    return torch.cat([model(part) for part in images.split(512)]).reshape(-1, 16, 4)


def summarize(logits: Tensor, truth: Tensor, has_revealed_cell: Tensor) -> dict[str, object]:
    hard = logits.argmax(dim=2)
    codes, valid = candidates(truth[:, 0], has_revealed_cell)
    _, log_evidence = posterior(logits.double().log_softmax(dim=2), codes, valid)
    legal = legal_grids(hard)
    revealed_ok = (hard[:, 0] == truth[:, 0]) | ~has_revealed_cell
    perms = torch.tensor(PERMUTATIONS, device=hard.device)
    agreement = (perms[:, hard] == truth[None]).double().mean(dim=(1, 2))
    best = int(agreement.argmax())
    return {"concept_accuracy": float((hard == truth).double().mean()),
            "boards_satisfying_the_rules": float(legal.double().mean()),
            "revealed_cells_correct": (float((hard[:, 0] == truth[:, 0])[has_revealed_cell].double().mean())
                                       if has_revealed_cell.any() else None),
            "boards_satisfying_all_constraints": float((legal & revealed_ok).double().mean()),
            "boards_with_evidence_above_half": float((log_evidence > -np.log(2)).double().mean()),
            "mean_negative_log_evidence": float(-log_evidence.mean()),
            "best_class_permutation": PERMUTATIONS[best],
            "agreement_under_best_class_permutation": float(agreement[best])}


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, images: Tensor, truth: Tensor,
               has_revealed_cell: Tensor, supervision: str) -> None:
    model.train()
    log_p = model(images).reshape(-1, 16, 4).log_softmax(dim=2)
    if supervision == "labels":
        targets = torch.zeros_like(log_p).scatter_(2, truth[:, :, None], 1.0)
    else:
        targets, _ = posterior(log_p.detach(), *candidates(truth[:, 0], has_revealed_cell))
    loss = -(targets * log_p).sum(dim=2).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def run(protocol_path: Path, data_root: Path, training_set: int, seed: int, fraction: float, supervision: str,
        output: Path, device: str, max_epochs: int | None) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (training_set not in protocol["training_sets"] or seed not in protocol["seeds"]
            or fraction not in protocol["fractions_of_boards_revealing_a_cell"]):
        raise ValueError("The protocol does not list this training set, seed or fraction")
    epochs = protocol["epochs"] if max_epochs is None else min(protocol["epochs"], max_epochs)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = make_model(protocol["input_size"], device)
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol["step_size"])
    images, truth, ids = build_boards(protocol, data_root, training_set, device)
    has_revealed_cell = revealed_boards(truth, fraction, protocol["clue_seed"] + training_set)
    test_images, test_truth, test_ids = build_boards(protocol, data_root, None, device)
    none_revealed = torch.zeros(len(test_truth), dtype=torch.bool, device=device)
    np.savez_compressed(output / "data.npz", train_ids=ids, train_labels=truth.cpu().numpy(),
                        has_revealed_cell=has_revealed_cell.cpu().numpy(), test_ids=test_ids,
                        test_labels=test_truth.cpu().numpy())
    json_write(output / "run_record.json", {"supervision": supervision, "fraction_of_boards_revealing_a_cell": fraction,
               "training_set": training_set, "seed": seed, "boards_revealing_a_cell": int(has_revealed_cell.sum()),
               "step_size": protocol["step_size"], "device": device, "torch": torch.__version__})
    rng = np.random.default_rng(protocol["batch_order_seed"] + 1000 * training_set + seed)
    boards, batch = len(truth), protocol["boards_per_minibatch"]
    recorded = set(protocol["recorded_epochs"])
    started = time.monotonic()
    with (output / "curves.jsonl").open("w", encoding="utf-8") as stream:
        for epoch in range(epochs + 1):
            if epoch in recorded or epoch == epochs:
                train_logits, test_logits = full_logits(model, images), full_logits(model, test_images)
                record = {"epoch": epoch, "train": summarize(train_logits, truth, has_revealed_cell),
                          "test": summarize(test_logits, test_truth, none_revealed), "seconds": time.monotonic() - started}
                stream.write(json.dumps(record) + "\n")
                stream.flush()
            if epoch == epochs:
                np.savez_compressed(output / "predictions.npz", train_logits=train_logits.cpu().numpy(),
                                    test_logits=test_logits.cpu().numpy())
                json_write(output / "endpoints.json", record)
                break
            order = rng.permutation(boards)
            for begin in range(0, boards, batch):
                chosen = torch.tensor(order[begin:begin + batch], device=device)
                cells = (chosen[:, None] * 16 + torch.arange(16, device=device)).flatten()
                train_step(model, optimizer, images[cells], truth[chosen], has_revealed_cell[chosen], supervision)
    json_write(output / "completed.json", {"seconds": time.monotonic() - started, "epochs": epochs})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="directory that holds the torchvision CIFAR-10 files")
    parser.add_argument("--training-set", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fraction", type=float, required=True, help="fraction of the boards that reveal their top-left class")
    parser.add_argument("--supervision", choices=("rules", "labels"), default="rules")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-epochs", type=int, help="stop early for a quick check")
    args = parser.parse_args()
    run(args.protocol, args.data_root, args.training_set, args.seed, args.fraction, args.supervision, args.output,
        args.device, args.max_epochs)


if __name__ == "__main__":
    main()
