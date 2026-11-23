"""Recompute the Sudoku results from the saved predictions, independently of the training code.

The evaluator enumerates legal grids independently and compares the outcomes with the saved predictions. The table has the columns of results/sudoku/runs.csv.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

ROWS = np.array(list(itertools.permutations(range(4))))
PERMUTATIONS = np.array(list(itertools.permutations(range(4))))


def legal(grid: np.ndarray) -> bool:
    g = grid.reshape(4, 4)
    blocks = [g[r:r + 2, c:c + 2].ravel() for r in (0, 2) for c in (0, 2)]
    return all(len(set(x)) == 4 for x in list(g) + list(g.T) + blocks)


def enumerate_grids() -> np.ndarray:
    found = [np.concatenate(rows) for rows in itertools.product(ROWS, repeat=4) if legal(np.concatenate(rows))]
    grids = np.array(found)
    assert grids.shape == (288, 16)
    return grids


GRIDS = enumerate_grids()


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def posterior(log_p: np.ndarray, revealed: np.ndarray, has_revealed_cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact posterior marginals and log evidence of every board, with a uniform prior and unit temperature."""
    scores = np.stack([log_p[:, np.arange(16), grid].sum(axis=1) for grid in GRIDS], axis=1)
    valid = (GRIDS[None, :, 0] == revealed[:, None]) | ~has_revealed_cell[:, None]
    scores = np.where(valid, scores, -np.inf)
    top = scores.max(axis=1, keepdims=True)
    log_evidence = top[:, 0] + np.log(np.exp(scores - top).sum(axis=1))
    weights = np.exp(scores - log_evidence[:, None])
    q = np.zeros_like(log_p)
    for index, grid in enumerate(GRIDS):
        q[:, np.arange(16), grid] += weights[:, index:index + 1]
    return q, log_evidence


def describe(logits: np.ndarray, truth: np.ndarray, has_revealed_cell: np.ndarray) -> dict[str, object]:
    hard = logits.argmax(axis=2)
    _, log_evidence = posterior(log_softmax(logits), truth[:, 0], has_revealed_cell)
    rules = np.array([legal(g) for g in hard])
    revealed_ok = (hard[:, 0] == truth[:, 0]) | ~has_revealed_cell
    return {"concept_accuracy": float((hard == truth).mean()),
            "boards_satisfying_the_rules": float(rules.mean()),
            "boards_satisfying_all_constraints": float((rules & revealed_ok).mean()),
            "boards_with_evidence_above_half": float((log_evidence > -np.log(2)).mean()),
            "revealed_cells_all_correct": bool((hard[:, 0] == truth[:, 0])[has_revealed_cell].all()),
            # Largest share of the cells on boards without a revealed cell that one class permutation explains.
            "cells_following_one_class_permutation": (max(float((perm[hard[~has_revealed_cell]] == truth[~has_revealed_cell]).mean())
                                                          for perm in PERMUTATIONS) if (~has_revealed_cell).any() else None)}


def evaluate(root: Path, output: Path) -> None:
    rows, worst = [], 0.0
    for folder in sorted(path.parent for path in root.rglob("run_record.json")):
        if not (folder / "completed.json").is_file():
            raise ValueError(f"Incomplete run: {folder}")
        record = json.loads((folder / "run_record.json").read_text())
        with np.load(folder / "data.npz") as data, np.load(folder / "predictions.npz") as predictions:
            truth, has_revealed_cell, test_truth = data["train_labels"], data["has_revealed_cell"], data["test_labels"]
            train = describe(predictions["train_logits"], truth, has_revealed_cell)
            test = describe(predictions["test_logits"], test_truth, np.zeros(len(test_truth), dtype=bool))
            assert len(np.unique(data["train_ids"])) == data["train_ids"].size
            assert not has_revealed_cell.any() or set(truth[has_revealed_cell, 0]) == {0, 1, 2, 3}
        saved = json.loads((folder / "endpoints.json").read_text())
        for key in ("concept_accuracy", "boards_satisfying_the_rules", "boards_satisfying_all_constraints", "boards_with_evidence_above_half"):
            worst = max(worst, abs(train[key] - saved["train"][key]))
        rules = record["supervision"] == "rules"
        one_permutation = train["cells_following_one_class_permutation"]
        rows.append({"supervision": "sudoku_rules" if rules else "true_labels",
                     "fraction_of_boards_revealing_a_cell": record["fraction_of_boards_revealing_a_cell"],
                     "training_set": record["training_set"], "seed": record["seed"],
                     "learns_the_concepts": train["concept_accuracy"] >= 0.99,
                     "training_boards_satisfying_all_constraints": f"{train['boards_satisfying_all_constraints']:.4f}",
                     "training_concept_accuracy": f"{train['concept_accuracy']:.4f}",
                     "test_concept_accuracy": f"{test['concept_accuracy']:.4f}",
                     "revealed_cells_all_correct": train["revealed_cells_all_correct"],
                     "cells_following_one_class_permutation": "" if one_permutation is None else f"{one_permutation:.4f}"})
    if not rows:
        raise ValueError(f"No run under {root}")
    rows.sort(key=lambda r: (r["supervision"], r["fraction_of_boards_revealing_a_cell"], r["training_set"], r["seed"]))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, object] = {"runs": len(rows), "largest_difference_from_training_record": worst,
                                  "sudoku_rules": []}
    under_rules = [r for r in rows if r["supervision"] == "sudoku_rules"]
    for fraction in sorted({r["fraction_of_boards_revealing_a_cell"] for r in under_rules}, reverse=True):
        cell = [r for r in under_rules if r["fraction_of_boards_revealing_a_cell"] == fraction]
        others = [r for r in cell if not r["learns_the_concepts"]]
        entry: dict[str, object] = {
            "fraction_of_boards_revealing_a_cell": fraction, "runs": len(cell),
            "runs_learning_the_concepts": sum(r["learns_the_concepts"] for r in cell),
            "median_test_concept_accuracy": float(np.median([float(r["test_concept_accuracy"]) for r in cell]))}
        if others:
            accuracy = [float(r["training_concept_accuracy"]) for r in others]
            permuted = [float(r["cells_following_one_class_permutation"]) for r in others if r["cells_following_one_class_permutation"]]
            entry["other_runs"] = {
                "runs": len(others),
                "training_boards_satisfying_all_constraints_min": min(float(r["training_boards_satisfying_all_constraints"]) for r in others),
                "training_concept_accuracy_min_max": [min(accuracy), max(accuracy)],
                "revealed_cells_all_correct": all(r["revealed_cells_all_correct"] for r in others),
                "cells_following_one_class_permutation_min": min(permuted) if permuted else None}
        summary["sudoku_rules"].append(entry)
    reference = [r for r in rows if r["supervision"] == "true_labels"]
    if reference:
        summary["true_labels"] = {"runs": len(reference), "runs_learning_the_concepts": sum(r["learns_the_concepts"] for r in reference),
                                  "median_test_concept_accuracy": float(np.median([float(r["test_concept_accuracy"]) for r in reference]))}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.input, args.output)
