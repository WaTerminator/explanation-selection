"""Tabulate the jointly trained models: per-run results, per-pair label differences and continuations.

The input directory holds, for every training set s and head h, the outputs of run_joint_training.py in
set_s/sgd_hh, set_s/adam_hh (count supervision) and set_s/reference_hh (correct concept labels), and the
outputs of run_joint_continuation.py in set_s/sgd_hh_continued and set_s/adam_hh_continued where they exist.
Outputs of run_frozen_optimizers.py in frozen_encoder/set_s/n<size> are tabulated as well.
The tables have the columns of results/joint_training/*.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measured(folder: Path) -> dict:
    with np.load(folder / "initial.npz") as initial, np.load(folder / "states.npz") as states:
        labels, counts = initial["labels"], initial["counts"]
        test_labels, bags = initial["test_labels"], initial["test_bags"]
        code, test = states["logits"][-1] > 0, states["test_logits"] > 0
    final = json.loads((folder / "completed.json").read_text())["final"]
    return {"code": code, "satisfies_all_counts": bool(np.all(code.reshape(-1, 4).sum(axis=1) == counts)),
            "mislabeled_training_images": int(np.count_nonzero(code != labels)),
            "evidence": float(np.exp(final["log_global_evidence"])),
            "test_concept_accuracy": float(np.mean(test == test_labels)),
            "test_count_accuracy": float(np.mean(test[bags].sum(axis=1) == test_labels[bags].sum(axis=1)))}


def frozen_encoder(root: Path) -> list[dict]:
    """SGD against Adam on the head with the frozen encoder: one row per training set, size and head."""
    rows = []
    for folder in sorted(root.glob("set_*/n*"), key=lambda path: (int(path.parent.name.split("_")[1]), int(path.name[1:]))):
        models = {}
        for optimizer in ("sgd", "adam"):
            endpoints = json.loads((folder / f"{optimizer}-endpoints.json").read_text())
            with np.load(folder / f"{optimizer}-state.npz") as state:
                models[optimizer] = (endpoints, state["final_heads"] @ state["training_design"].T > 0)
        for head in range(len(models["sgd"][0])):
            sgd, adam = models["sgd"][0][head], models["adam"][0][head]
            rows.append({"training_inputs": int(folder.name[1:]), "training_set": int(folder.parent.name.split("_")[1]), "head": head,
                         "both_fit": bool(sgd["fitted"] and adam["fitted"]),
                         "training_labels_on_which_the_models_differ": int(np.count_nonzero(models["sgd"][1][head] != models["adam"][1][head])),
                         "both_evidences_above_one_half": bool(min(sgd["log_global_evidence"], adam["log_global_evidence"]) > -np.log(2))})
    return rows


def compare_pair(training_set: int, head: int, models: dict) -> dict:
    sgd, adam = models["sgd"], models["adam"]
    return {
        "training_set": training_set, "head": head,
        "both_satisfy_all_counts": sgd["satisfies_all_counts"] and adam["satisfies_all_counts"],
        "training_labels_on_which_the_models_differ": int(np.count_nonzero(sgd["code"] != adam["code"])),
        "test_concept_gap_pp": round(100 * abs(sgd["test_concept_accuracy"] - adam["test_concept_accuracy"]), 8),
        "test_count_gap_pp": round(100 * abs(sgd["test_count_accuracy"] - adam["test_count_accuracy"]), 8),
    }


def evaluate(root: Path, output: Path, rate_setting: str) -> None:
    runs, pairs, continued = [], [], []
    for folder in sorted(root.glob("set_*")):
        training_set = int(folder.name.split("_")[1])
        for sgd_folder in sorted(folder.glob("sgd_h[0-9]*")):
            if not sgd_folder.name.removeprefix("sgd_h").isdigit():
                continue
            head = int(sgd_folder.name.removeprefix("sgd_h"))
            models = {optimizer: measured(folder / f"{optimizer}_h{head}") for optimizer in ("sgd", "adam")}
            pairs.append(compare_pair(training_set, head, models))
            if rate_setting == "original":
                continue
            reference = measured(folder / f"reference_h{head}")
            for optimizer, model in models.items():
                runs.append({"training_set": training_set, "head": head, "optimizer": optimizer,
                             "satisfies_all_counts": model["satisfies_all_counts"],
                             "mislabeled_training_images": model["mislabeled_training_images"],
                             "evidence": f"{model['evidence']:.6f}",
                             "test_concept_accuracy": f"{model['test_concept_accuracy']:.4f}",
                             "test_count_accuracy": f"{model['test_count_accuracy']:.4f}",
                             "reference_test_concept_accuracy": f"{reference['test_concept_accuracy']:.4f}",
                             "reference_test_count_accuracy": f"{reference['test_count_accuracy']:.4f}"})
                done = folder / f"{optimizer}_h{head}_continued" / "completed.json"
                if done.is_file():
                    record = json.loads(done.read_text())
                    continued.append({"training_set": training_set, "head": head, "optimizer": optimizer,
                                      "evidence_before_continuation": f"{model['evidence']:.6f}", "updates": record["updates"],
                                      "updates_changing_a_training_label": record["updates_changing_a_training_label"]})
    if not pairs:
        raise ValueError(f"No complete SGD/Adam pair under {root}")
    if rate_setting == "original":
        write_csv(output / "original_pairs.csv", pairs)
        return
    write_csv(output / "runs.csv", runs)
    write_csv(output / "pairs.csv", pairs)
    if continued:
        write_csv(output / "continuations.csv", continued)
    legal = [r for r in runs if r["satisfies_all_counts"]]
    summary = {"initial_networks": len(pairs)}
    for optimizer in ("sgd", "adam"):
        rows = [r for r in legal if r["optimizer"] == optimizer]
        if rows:
            summary[optimizer] = {
                "runs_satisfying_all_counts": len(rows),
                "mislabeled_training_images_percent": float(100 * np.mean([r["mislabeled_training_images"] for r in rows]) / 1024),
                "test_concept_accuracy_percent": float(100 * np.mean([float(r["test_concept_accuracy"]) for r in rows])),
                "lost_test_concept_points": float(100 * np.mean([float(r["reference_test_concept_accuracy"]) - float(r["test_concept_accuracy"]) for r in rows])),
                "test_count_accuracy_percent": float(100 * np.mean([float(r["test_count_accuracy"]) for r in rows])),
                "lost_test_count_points": float(100 * np.mean([float(r["reference_test_count_accuracy"]) - float(r["test_count_accuracy"]) for r in rows]))}
    common = [p["training_labels_on_which_the_models_differ"] for p in pairs if p["both_satisfy_all_counts"]]
    if common:
        summary["pairs_in_which_both_satisfy_all_counts"] = len(common)
        summary["labels_on_which_they_differ_min_median_max"] = [int(min(common)), float(np.median(common)), int(max(common))]
    if len(legal) > 2:
        summary["correlation_mislabeled_vs_lost_concept_accuracy"] = float(np.corrcoef(
            [r["mislabeled_training_images"] for r in legal],
            [float(r["reference_test_concept_accuracy"]) - float(r["test_concept_accuracy"]) for r in legal])[0, 1])
    if continued:
        summary["continuations"] = {"models": len(continued), "updates": sum(r["updates"] for r in continued),
                                    "updates_changing_a_training_label": sum(r["updates_changing_a_training_label"] for r in continued)}
    frozen = frozen_encoder(root / "frozen_encoder")
    if frozen:
        write_csv(output / "frozen_encoder.csv", frozen)
        summary["frozen_encoder"] = {
            str(size): {"pairs": len(rows), "pairs_in_which_both_fit": sum(r["both_fit"] for r in rows),
                        "pairs_selecting_different_labels": sum(r["training_labels_on_which_the_models_differ"] > 0 for r in rows),
                        "median_labels_on_which_they_differ": float(np.median([r["training_labels_on_which_the_models_differ"] for r in rows])),
                        "pairs_with_both_evidences_above_one_half": sum(r["both_evidences_above_one_half"] for r in rows)}
            for size in sorted({r["training_inputs"] for r in frozen})
            for rows in [[r for r in frozen if r["training_inputs"] == size]]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rates", choices=("expanded", "original"), default="expanded")
    args = parser.parse_args()
    evaluate(args.input, args.output, args.rates)
