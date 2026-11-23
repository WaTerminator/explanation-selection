"""Train a ResNet-18 and its classifier head jointly with SGD or Adam.

One call trains one model from one initial network (a training set and an initial head). With
`--supervision counts` the only supervision is the number of cats in each four-image bag. With
`--supervision labels` the same network is trained on the correct concept labels; the manuscript uses this
as the reference (trained with Adam). Data, initial network, minibatch order and budget are identical for
both kinds of supervision and for both optimizers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet18_Weights, resnet18

from joint_training_common import encode, normalized_design, state
from run_cifar_identifiable_traps import json_write
from run_cifar_llp import allocate, candidates, design


def run(protocol_path: Path, features: Path, data_root: Path, training_set: int, head: int, optimizer_name: str,
        supervision: str, output: Path, device: str, max_epochs: int | None, rate_setting: str) -> None:
    protocol = json.loads(protocol_path.read_text())
    if training_set not in protocol["training_sets"] or head not in protocol["heads"]:
        raise ValueError("The protocol does not list this training set or head")
    size, scale, epochs = protocol["training_inputs"], float(protocol["logit_scale"]), protocol["epochs"]
    rate_key = "original_step_sizes" if rate_setting == "original" else "step_sizes"
    rates = protocol[rate_key][optimizer_name]
    head_rate, encoder_rate = rates["head"], rates["encoder"]
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    torch.manual_seed(2026110902)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    encoder.fc = nn.Identity()
    encoder.eval().requires_grad_(True).to(device)
    buffers = {name: value.detach().clone() for name, value in encoder.named_buffers()}
    counts = np.resize(np.array([1, 2, 3], dtype=np.int64), protocol["allocation_size"] // 4)
    with np.load(features, allow_pickle=False) as data:
        ids = allocate(data["train_labels"], counts, protocol["allocation_draws"], protocol["data_seed"])[training_set][:size]
        counts = counts[:size // 4]
        train_ids, test_ids = data["train_indices"][ids], data["test_indices"]
        labels = (data["train_labels"][ids] == 3).astype(np.int64)
        test_labels = (data["test_labels"] == 3).astype(np.int64)
        cached = design(data["train_features"][ids])
    np.testing.assert_array_equal(labels.reshape(-1, 4).sum(axis=1), counts)
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    train_data = CIFAR10(str(data_root), train=True, transform=transform, download=False)
    test_data = CIFAR10(str(data_root), train=False, transform=transform, download=False)
    train_images = torch.stack([train_data[int(i)][0] for i in train_ids]).to(device)
    test_images = torch.stack([test_data[int(i)][0] for i in test_ids]).to(device)
    initial_x = encode(encoder, train_images)
    initial_test_x = encode(encoder, test_images)
    # The encoder must reproduce the supplied frozen features before any update.
    np.testing.assert_allclose(initial_x.cpu().numpy(), cached, atol=2e-6, rtol=2e-5)
    initial = np.random.default_rng(protocol["head_seed"] + 100*training_set + head).uniform(
        -1/math.sqrt(initial_x.shape[1]), 1/math.sqrt(initial_x.shape[1]), initial_x.shape[1])
    # Logits are scale * v @ x. The optimizers update v, and every saved head is scale * v, in logit units.
    w = nn.Parameter(torch.tensor(initial / scale, device=device))

    def logit_weights() -> torch.Tensor:
        return scale * w

    true_labels = torch.tensor(labels, dtype=w.dtype, device=device)
    initial_test_logits = (initial_test_x @ logit_weights().detach()).cpu().numpy()
    codes_np, valid_np = candidates(counts)
    codes, valid = torch.tensor(codes_np, device=device), torch.tensor(valid_np, device=device)
    groups: list[dict[str, object]] = [{"params": [w], "lr": head_rate},
                                       {"params": list(encoder.parameters()), "lr": encoder_rate}]
    if optimizer_name == "adam":
        settings = protocol["adam"]
        groups[0]["eps"] = settings["epsilon_head"]
        optimizer = torch.optim.Adam(groups, betas=tuple(settings["betas"]), eps=settings["epsilon_encoder"],
                                     weight_decay=settings["weight_decay"])
    else:
        settings = protocol["sgd"]
        optimizer = torch.optim.SGD(groups, momentum=settings["momentum"], weight_decay=settings["weight_decay"])
    rng = np.random.default_rng(protocol["batch_seed"] + training_set)
    test_bags = np.random.default_rng(protocol["test_bag_seed"]).permutation(len(test_ids)).reshape(-1, 4)
    np.savez_compressed(output / "initial.npz", head=initial, training_design=initial_x.cpu().numpy(),
                        train_indices=train_ids, test_indices=test_ids, labels=labels, test_labels=test_labels,
                        counts=counts, test_bags=test_bags, test_logits=initial_test_logits)
    json_write(output / "run_record.json", {"training_set": training_set, "head": head, "optimizer": optimizer_name,
               "supervision": supervision, "training_inputs": size, "head_step_size": head_rate,
               "encoder_step_size": encoder_rate, "logit_scale": scale, "torch": torch.__version__, "device": device})
    budget = epochs if max_epochs is None else min(epochs, max_epochs)
    history = np.empty((budget+1, len(train_ids)))
    curves: list[dict] = []
    updates = 0
    for epoch in range(budget+1):
        with torch.no_grad():
            features_now = initial_x if epoch == 0 else encode(encoder, train_images)
            loss, _, logits, log_bag = state(logit_weights()[None], features_now, codes, valid)
            values = logits.cpu().numpy()[0]
            history[epoch] = values
            confidence = float(torch.sigmoid(logits.abs()).min())
            evidence = float(log_bag.exp().mean())
            legal = bool(np.all((values > 0).reshape(-1, 4).sum(axis=1) == counts))
            if not np.isfinite(values).all() or not torch.isfinite(loss).all():
                raise FloatingPointError(f"Nonfinite state at epoch {epoch}")
            row = {"epoch": epoch, "count_nll": float(loss), "satisfies_all_counts": legal,
                   "log_global_evidence": float(log_bag.sum()), "mean_bag_evidence": evidence,
                   "minimum_confidence": confidence, "seconds": time.monotonic()-started}
            curves.append(row)
        if epoch % 5 == 0 or epoch == budget:
            json_write(output / "curves.json", curves)
            print(json.dumps(row), flush=True)
        if epoch == budget:
            break
        factor = .01 + .99*(1+math.cos(math.pi*epoch/epochs))/2
        optimizer.param_groups[0]["lr"] = head_rate*factor
        optimizer.param_groups[1]["lr"] = encoder_rate*factor
        for batch in np.array_split(rng.permutation(len(counts)), len(counts)//protocol["bags_per_minibatch"]):
            positions = (batch[:, None]*4+np.arange(4)).ravel()
            x = normalized_design(encoder(train_images[positions]))
            nll, gradient, z, _ = state(logit_weights()[None], x, codes[batch], valid[batch])
            if supervision == "labels":
                targets = true_labels[positions][None]
                nll = torch.nn.functional.binary_cross_entropy_with_logits(z, targets, reduction="sum") / len(batch)
            if updates == 0:
                # The count loss must have the gradient of the cross-entropy against the detached posterior marginals.
                grouped = z.reshape(1, len(batch), 4)
                if supervision == "counts":
                    with torch.no_grad():
                        scores = torch.einsum("hbi,bci->hbc", grouped, codes[batch]).masked_fill(~valid[batch], -torch.inf)
                        targets = torch.einsum("hbc,bci->hbi", scores.softmax(dim=2), codes[batch])
                    surrogate = torch.nn.functional.binary_cross_entropy_with_logits(grouped, targets, reduction="sum")/len(batch)
                else:
                    surrogate = (torch.nn.functional.softplus(z) - targets*z).sum() / len(batch)
                    gradient = (torch.sigmoid(z) - targets) @ x / len(batch)
                soft_grad = torch.autograd.grad(surrogate, (w, encoder.conv1.weight), retain_graph=True)
                exact_grad = torch.autograd.grad(nll.sum(), (w, encoder.conv1.weight), retain_graph=True)
                # Compare head gradients in logit units, as returned by `state`.
                soft_grad, exact_grad = [(pair[0] / scale, pair[1]) for pair in (soft_grad, exact_grad)]
                for a, b in zip(soft_grad, exact_grad):
                    torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-5)
                torch.testing.assert_close(exact_grad[0], gradient[0], atol=1e-10, rtol=1e-10)
                if not torch.isfinite(exact_grad[1]).all() or float(exact_grad[1].norm()) == 0:
                    raise AssertionError("No finite nonzero encoder gradient")
            optimizer.zero_grad(set_to_none=True)
            nll.sum().backward()
            optimizer.step()
            updates += 1
    with torch.no_grad():
        final_test_x = encode(encoder, test_images)
        test_logits = (final_test_x @ logit_weights()).cpu().numpy()
    if not np.isfinite(test_logits).all():
        raise FloatingPointError("Nonfinite final test predictions")
    for name, value in encoder.named_buffers():
        torch.testing.assert_close(value, buffers[name], atol=0, rtol=0)
    np.savez_compressed(output / "states.npz", logits=history, final_head=logit_weights().detach().cpu().numpy(),
                        final_design=features_now.cpu().numpy(), test_logits=test_logits)
    # The optimizer state refers to the updated parameter, head / logit_scale.
    torch.save({"encoder": encoder.state_dict(), "head": logit_weights().detach(), "logit_scale": scale,
                "optimizer": optimizer.state_dict(), "batch_rng": rng.bit_generator.state, "epoch": budget},
               output / "checkpoint.pt")
    json_write(output / "completed.json", {"training_set": training_set, "head": head, "optimizer": optimizer_name,
               "supervision": supervision, "epochs": budget, "updates": updates, "final": row,
               "relative_feature_change": float(torch.linalg.vector_norm(features_now-initial_x)/torch.linalg.vector_norm(initial_x)),
               "seconds": time.monotonic()-started})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="directory that holds the torchvision CIFAR-10 files")
    parser.add_argument("--training-set", type=int, required=True)
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), required=True)
    parser.add_argument("--supervision", choices=("counts", "labels"), default="counts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rates", choices=("original", "expanded"), default="expanded")
    parser.add_argument("--max-epochs", type=int, help="stop early for a quick check; the schedule is unchanged")
    args = parser.parse_args()
    run(args.protocol, args.features, args.data_root, args.training_set, args.head, args.optimizer,
        args.supervision, args.output, args.device, args.max_epochs, args.rates)


if __name__ == "__main__":
    main()
