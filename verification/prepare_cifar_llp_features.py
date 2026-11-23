#!/usr/bin/env python3
"""Cache fixed ImageNet ResNet18 features for the CIFAR-10 bird/cat task.

This performs inference only. Labels select the two published task classes and
are retained for bag construction/evaluation; they never fit or select an
encoder. Features are the unnormalized 512-dimensional pre-classifier output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
import torchvision
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet18_Weights, resnet18


COUNT_TASK_REFERENCE = "https://github.com/barucden/mlellp/tree/829b2aa045b3b5fa0a39894e21919205cb9eccec"


class SplitMetadata(TypedDict):
    count: int
    class_counts: dict[str, int]
    official_split_size: int
    feature_norm_min: float
    feature_norm_max: float


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode_split(
    dataset: CIFAR10, model: nn.Module, device: torch.device, batch_size: int
) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.int64], SplitMetadata]:
    """Preserve official split order while encoding only bird and cat images."""
    labels = np.asarray(dataset.targets, dtype=np.int64)
    indices = np.flatnonzero(np.isin(labels, [2, 3])).astype(np.int64)
    selected_labels = labels[indices]
    loader = DataLoader(
        Subset(dataset, indices.tolist()), batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=device.type == "cuda",
    )
    features: list[torch.Tensor] = []
    observed_labels: list[torch.Tensor] = []
    with torch.inference_mode():
        for images, batch_labels in loader:
            features.append(model(images.to(device, non_blocking=True)).cpu())
            observed_labels.append(batch_labels)
    values = torch.cat(features).numpy()
    if not np.array_equal(torch.cat(observed_labels).numpy(), selected_labels):
        raise RuntimeError("Feature batches changed the official source row order")
    if values.dtype != np.float32 or values.shape != (len(indices), 512):
        raise RuntimeError(f"Unexpected feature array: {values.shape}, {values.dtype}")
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    if not np.isfinite(values).all() or np.any(norms == 0):
        raise RuntimeError("Feature cache contains non-finite or zero feature rows")
    counts = {str(label): int(np.sum(selected_labels == label)) for label in (2, 3)}
    expected = 5000 if dataset.train else 1000
    if counts != {"2": expected, "3": expected}:
        raise RuntimeError(f"Unexpected CIFAR-10 bird/cat class counts: {counts}")
    metadata: SplitMetadata = {
        "count": len(indices), "class_counts": counts,
        "official_split_size": len(dataset),
        "feature_norm_min": float(norms.min()), "feature_norm_max": float(norms.max()),
    }
    return values, indices, selected_labels, metadata


def prepare_features(
    data_root: Path, output_path: Path, metadata_path: Path, device_name: str,
    batch_size: int,
) -> None:
    """Encode both official splits using the fixed ImageNet weights."""
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError("Choose fresh feature and metadata output paths")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    started = time.perf_counter()
    source_hash = sha256(Path(__file__))
    weights = ResNet18_Weights.IMAGENET1K_V1
    device = torch.device(device_name)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = resnet18(weights=weights)
    weight_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    model.fc = nn.Identity()
    model.requires_grad_(False).eval().to(device)
    transform = weights.transforms()
    arrays: dict[str, NDArray[np.float32] | NDArray[np.int64]] = {}
    split_metadata: dict[str, SplitMetadata] = {}
    for split, train in (("train", True), ("test", False)):
        dataset = CIFAR10(str(data_root), train=train, transform=transform, download=False)
        values, indices, labels, split_metadata[split] = encode_split(
            dataset, model, device, batch_size
        )
        arrays.update({f"{split}_features": values, f"{split}_indices": indices,
                       f"{split}_labels": labels})
        print(json.dumps({"split": split, **split_metadata[split]}), flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)
    metadata = {
        "format_version": "cifar_llp_resnet18_features_v1",
        "feature_file": str(output_path.resolve()),
        "feature_sha256": sha256(output_path),
        "data_root": str(data_root.resolve()),
        "dataset": "torchvision CIFAR10 official train/test; download=False",
        "classes": {"2": "bird", "3": "cat"},
        "source_indices": "Zero-based within each official split, in ascending official order",
        "features": "Unnormalized float32 512-dimensional ResNet18 output before fc; no bias appended",
        "model": "torchvision.models.resnet18",
        "weights": "ResNet18_Weights.IMAGENET1K_V1",
        "weight_path": str(weight_path), "weight_url": weights.url,
        "weight_sha256": sha256(weight_path),
        "transform": {
            "input": "PIL RGB image from torchvision CIFAR10",
            "resize_shorter_edge": 256, "center_crop": [224, 224],
            "interpolation": "bilinear", "antialias": transform.antialias,
            "scale": "uint8 to float32 in [0,1] after spatial transforms",
            "mean": transform.mean, "std": transform.std,
        },
        "encoder_training": False, "augmentation": False,
        "feature_normalization": False, "mixed_precision": False,
        "label_use": "Select bird/cat task and retain original labels; no model fitting or model selection",
        "splits": split_metadata,
        "environment": {
            "python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "cuda": torch.version.cuda,
            "device": device_name,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "batch_size": batch_size, "data_loader_workers": 4,
            "cudnn_deterministic": True, "tf32": False,
        },
        "preparation_script": "verification/prepare_cifar_llp_features.py",
        "preparation_script_sha256_at_start": source_hash,
        "count_task_reference": COUNT_TASK_REFERENCE,
        "runtime_seconds": time.perf_counter() - started,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(output_path), "metadata": str(metadata_path),
                      "runtime_seconds": metadata["runtime_seconds"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    prepare_features(args.data_root, args.output, args.metadata, args.device, args.batch_size)


if __name__ == "__main__":
    main()
