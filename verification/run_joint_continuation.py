"""Continue a jointly trained model and measure changes in training labels during minibatch updates.

The source is the output directory of run_joint_training.py for a count-supervised model whose evidence (the
probability that all training counts hold) exceeds one half. The continuation keeps the optimizer state and the
minibatch order of the source, restarts the initial step sizes and holds them constant, and keeps the encoder
and the head trainable. After every update, the test compares the predictions on all training inputs before and
after the update.
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
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet18_Weights, resnet18

from joint_training_common import encode, normalized_design, state
from run_cifar_identifiable_traps import json_write
from run_cifar_llp import candidates, quantities


def run(protocol_path: Path, source: Path, data_root: Path, output: Path, device: str) -> None:
    protocol = json.loads(protocol_path.read_text())
    steps, bags_per_minibatch = protocol["continuation"]["updates"], protocol["bags_per_minibatch"]
    record = json.loads((source / "run_record.json").read_text())
    done = json.loads((source / "completed.json").read_text())
    if record["supervision"] != "counts" or done["epochs"] != protocol["epochs"]:
        raise ValueError("The source must be a fully trained count-supervised model")
    if done["final"]["log_global_evidence"] <= -np.log(2):
        raise ValueError("The evidence of the source does not exceed one half")
    optimizer_name, started = record["optimizer"], time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    with np.load(source / "initial.npz", allow_pickle=False) as data:
        initial = {name: data[name] for name in data.files}
    with np.load(source / "states.npz", allow_pickle=False) as data:
        expected, source_x = data["logits"][-1], data["final_design"]
    checkpoint = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=False)
    scale = float(checkpoint["logit_scale"])
    encoder = resnet18(weights=None)
    encoder.fc = nn.Identity()
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval().requires_grad_(True).to(device)
    buffers = {name: value.detach().clone() for name, value in encoder.named_buffers()}
    # The optimizers update v = head / scale, as during training.
    w = nn.Parameter(checkpoint["head"].to(device) / scale)
    groups = [{"params": [w]}, {"params": list(encoder.parameters())}]
    optimizer = torch.optim.SGD(groups, lr=1.) if optimizer_name == "sgd" else torch.optim.Adam(groups, lr=1.)
    optimizer.load_state_dict(checkpoint["optimizer"])
    rates = protocol["step_sizes"][optimizer_name]
    for group, rate in zip(optimizer.param_groups, (rates["head"], rates["encoder"])):
        group["lr"] = rate
    rng = np.random.default_rng()
    rng.bit_generator.state = checkpoint["batch_rng"]
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    training = CIFAR10(str(data_root), train=True, transform=transform, download=False)
    images = torch.stack([training[int(i)][0] for i in initial["train_indices"]]).to(device)
    counts = initial["counts"]
    codes_np, valid_np = candidates(counts)
    codes, valid = torch.tensor(codes_np, device=device), torch.tensor(valid_np, device=device)
    with torch.no_grad():
        x = encode(encoder, images)
        current = (x @ (scale * w)).cpu().numpy()
    np.testing.assert_allclose(current, expected, atol=1e-8, rtol=1e-10)
    np.testing.assert_allclose(x.cpu().numpy(), source_x, atol=1e-9, rtol=1e-9)
    if float(quantities(current, codes_np, valid_np).log_bag_evidence.sum()) <= -np.log(2):
        raise ValueError("The reloaded model does not exceed evidence one half")
    history, rows = [current], []
    batches = len(codes) // bags_per_minibatch
    for step in range(steps):
        if step % batches == 0:
            order = rng.permutation(len(codes))
        bag_ids = order[(step % batches) * bags_per_minibatch:(step % batches + 1) * bags_per_minibatch]
        ids = (bag_ids[:, None] * 4 + np.arange(4)).reshape(-1)
        design = normalized_design(encoder(images[ids]))
        loss, _, _, _ = state((scale * w)[None], design, codes[bag_ids], valid[bag_ids])
        optimizer.zero_grad(set_to_none=True)
        loss.sum().backward()
        optimizer.step()
        with torch.no_grad():
            current_x = encode(encoder, images)
            current = (current_x @ (scale * w)).cpu().numpy()
        if not np.isfinite(current).all():
            raise FloatingPointError(f"Nonfinite predictions after update {step + 1}")
        rows.append({"update": step + 1, "labels_changed_by_this_update": int(np.count_nonzero((current > 0) != (history[-1] > 0))),
                     "labels_differing_from_the_source": int(np.count_nonzero((current > 0) != (history[0] > 0))),
                     "relative_feature_change": float(np.linalg.norm(current_x.cpu().numpy() - source_x) / np.linalg.norm(source_x))})
        history.append(current)
        if (step + 1) % 32 == 0:
            json_write(output / "curves.json", rows)
            print(json.dumps(rows[-1]), flush=True)
    for name, value in encoder.named_buffers():
        torch.testing.assert_close(value, buffers[name], atol=0, rtol=0)
    np.savez_compressed(output / "states.npz", logits=np.stack(history))
    json_write(output / "curves.json", rows)
    json_write(output / "completed.json", {"source": str(source), "optimizer": optimizer_name, "updates": steps,
               "updates_changing_a_training_label": sum(row["labels_changed_by_this_update"] > 0 for row in rows),
               "seconds": time.monotonic() - started})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="output directory of run_joint_training.py")
    parser.add_argument("--data-root", type=Path, required=True, help="directory that holds the torchvision CIFAR-10 files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.protocol, args.source, args.data_root, args.output, args.device)


if __name__ == "__main__":
    main()
