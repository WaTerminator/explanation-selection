"""Recompute the natural-access results from the saved states, independently of the training code.

The input directory holds, for every training set s, the outputs of
  run_random_initialization.py  in random_initialization/set_s/<condition>/<gd|sgd>,
  run_neighborhood_entry.py     in neighborhood_entry/set_s,
  run_recovery.py               in recovery/set_s.
Parts that are absent are skipped. The query loss and its gradient are implemented again here with NumPy.
The tables have the columns of results/natural_access/*.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular
from scipy.special import expit


Array = NDArray[np.float64]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def objective(logits: Array, pairs: NDArray[np.int64], counts: NDArray[np.int64], weights: Array) -> tuple[Array, Array]:
    """Weighted negative log evidence of the pair counts and its logit gradient; weights has one row per state."""
    p = expit(logits)
    left, right = p[:, pairs[:, 0]], p[:, pairs[:, 1]]
    mixed_evidence = left*(1-right) + (1-left)*right
    evidence = np.where(counts == 0, (1-left)*(1-right), np.where(counts == 2, left*right, mixed_evidence))
    qleft = np.where(counts == 1, left*(1-right)/mixed_evidence, counts/2)
    qright = np.where(counts == 1, (1-left)*right/mixed_evidence, counts/2)
    gradient = np.zeros_like(logits)
    rows = np.arange(len(logits))[:, None]
    np.add.at(gradient, (rows, pairs[:, 0]), weights*(left-qleft))
    np.add.at(gradient, (rows, pairs[:, 1]), weights*(right-qright))
    return -(weights*np.log(evidence)).sum(axis=1), gradient


def weights_for(counts: NDArray[np.int64], rhos: Array) -> Array:
    mixed = counts == 1
    return np.where(mixed[None, :], rhos[:, None] / mixed.sum(), (1 - rhos[:, None]) / (~mixed).sum())


def sets_under(folder: Path) -> list[tuple[int, Path]]:
    return sorted((int(path.name.split("_")[1]), path) for path in folder.glob("set_*"))


def random_initialization(root: Path, settings: dict) -> tuple[list[dict], dict]:
    rows = []
    for training_set, folder in sets_under(root):
        for condition in settings["conditions"]:
            for optimizer in ("gd", "sgd"):
                run = folder / condition["name"] / optimizer
                if not (run / "completed.json").is_file():
                    continue
                record = json.loads((run / "run_record.json").read_text(encoding="utf-8"))
                with np.load(run / "states.npz", allow_pickle=False) as saved:
                    x, labels = saved["training_design"], saved["labels"]
                    final, last_change = saved["final_heads"] @ x.T, saved["last_label_change"]
                    np.testing.assert_allclose(final, saved["logits"][-1], atol=1e-10, rtol=1e-10)
                    updates = int(saved["final_update"])
                accuracy = ((final > 0) == labels).mean(axis=1)
                if updates != settings["updates"] or not np.all((accuracy == 0) | (accuracy == 1)) or not np.all(last_change <= updates - 20000):
                    raise AssertionError(f"A run is unfinished, partly wrong or still changing labels: {run}")
                for head in range(len(accuracy)):
                    rows.append({"identified_inputs": condition["identified_inputs"],
                                 "rho": f"{record['rho']:.2f}" if condition["rho"] is not None else f"{record['rho']:.4f}",
                                 "optimizer": optimizer, "training_set": training_set, "head": head,
                                 "all_training_concepts": "wrong" if accuracy[head] == 0 else "correct"})
    table: dict[str, dict[str, str]] = {}
    for condition in settings["conditions"]:
        rho = condition["rho"]
        if rho is None:
            mixed = settings["inputs"] * settings["mixed_degree"]
            rho_key = f"{mixed / (mixed + condition['identified_inputs']):.4f}"
        else:
            rho_key = f"{rho:.2f}"
        for optimizer in ("gd", "sgd"):
            cell = [r for r in rows if r["optimizer"] == optimizer and r["identified_inputs"] == condition["identified_inputs"]
                    and r["rho"] == rho_key]
            if cell:
                table.setdefault(condition["name"], {})[optimizer] = f"{sum(r['all_training_concepts'] == 'wrong' for r in cell)} of {len(cell)} wrong"
    return rows, {"runs": len(rows), "runs_with_all_training_concepts_wrong": table}


def neighborhood_entry(root: Path, settings: dict, output: Path) -> tuple[list[dict], dict]:
    rows, traces = [], []
    for training_set, folder in sets_under(root):
        geometry = json.loads((folder / "geometry.json").read_text(encoding="utf-8"))
        with np.load(folder / "states.npz", allow_pickle=False) as saved:
            x, labels, logits = saved["training_design"], saved["labels"], saved["logits"]
            added, heads, source_updates = saved["additional_updates"], saved["heads"], int(saved["source_updates"])
            rho, n = geometry["rho"], len(labels)
            # Finite wrong state, radius and curvature bounds, from the formulas of Appendix F.3.
            v = (2-rho-np.sqrt(rho*rho+4*rho-4))/4
            magnitude = np.log((1-v)/v)
            center = (1-2*labels)*magnitude
            np.testing.assert_allclose(saved["center_logits"], center, atol=1e-14)
            b = settings["minimum_wrong_margin"]
            radius = (magnitude-b)/np.sqrt(np.max(np.sum(x*x, axis=1)))
            eigenvalues = np.linalg.eigvalsh(x @ x.T)
            t = np.exp(-b)
            pbound = 1/(1+np.exp(2*magnitude-b))
            lower = eigenvalues[0]*(2/n)*pbound*(1-pbound)*(1-2*rho*t*(1+t)**2/(1+t*t)**2)
            upper = eigenvalues[-1]/(2*n)
            assert lower > 0 and 0 < float(saved["eta"])*upper < 1
            np.testing.assert_allclose([radius, lower, upper], [geometry["radius"], geometry["curvature_lower_bound_in_ball"],
                                                               geometry["curvature_upper_bound"]], rtol=1e-10)
            chol = np.linalg.cholesky(x @ x.T)
            first = np.full(len(heads), -1, dtype=np.int64)
            previous: Array | None = None
            for steps, z in zip(added, logits):
                distances = np.linalg.norm(solve_triangular(chol, (z-center).T, lower=True), axis=0)
                for head, distance in zip(heads[first < 0], distances[first < 0]):
                    traces.append({"training_set": training_set, "head": int(head),
                                   "total_updates": int(source_updates + steps), "distance_over_radius": float(distance / radius)})
                first[(first < 0) & (distances <= radius)] = steps
                assert np.all((z > 0) != labels)
                if previous is not None:
                    inside = previous < radius
                    assert np.all(distances[inside] <= previous[inside]+1e-11)
                previous = distances
            np.testing.assert_array_equal(first, saved["first_entry"])
        for head, entry in zip(heads, first):
            rows.append({"training_set": training_set, "head": int(head), "enters_the_contracting_neighborhood": bool(entry >= 0),
                         "total_updates_at_entry": int(source_updates+entry) if entry >= 0 else ""})
    if traces:
        write_csv(output / "entry_traces.csv", traces)
    entered = [r["total_updates_at_entry"] for r in rows if r["enters_the_contracting_neighborhood"]]
    return rows, {"runs": len(rows), "runs_entering": len(entered),
                  "total_updates_at_entry_min_max": [min(entered), max(entered)] if entered else None}


def recovery(root: Path, settings: dict) -> tuple[list[dict], dict]:
    rows = []
    for training_set, folder in sets_under(root):
        with np.load(folder / "states.npz", allow_pickle=False) as saved:
            logits, updates, labels = saved["logits"], saved["updates"], saved["labels"]
            restored, reweighted, heads = saved["restored"], saved["reweighted"], saved["heads"]
            pairs, counts, weights = saved["pairs"], saved["counts"], saved["original_weights"]
            test_hard = saved["final_heads"] @ saved["test_design"].T > 0
            test_labels, test_pairs, test_counts = saved["test_labels"], saved["test_pairs"], saved["test_counts"]
        n = len(labels)
        np.testing.assert_allclose(weights, weights_for(counts, np.array([settings["original_rho"]]))[0], atol=1e-15)
        losses = np.stack([objective(z, pairs, counts, weights[None])[0] for z in logits])
        violations = ((logits > 0)[:, :, pairs].sum(axis=3) != counts).sum(axis=2)
        concepts_correct = np.all((logits > 0) == labels, axis=2)
        met = (violations == 0) & (losses < 2*np.log(2)/n)
        for index in range(logits.shape[1]):
            if reweighted[index]:
                # The weight is restored at the first check at which the rule holds, and all concepts stay correct.
                first = int(updates[np.argmax(met[:, index])]) if met[:, index].any() else -1
                if first != restored[index]:
                    raise AssertionError(f"The rule is first met at {first}, not at {restored[index]}: {folder}")
                if first >= 0 and not concepts_correct[updates >= first, index].all():
                    raise AssertionError(f"A concept is wrong after the weight is restored: {folder}")
            elif restored[index] != -1:
                raise AssertionError("The unchanged training restored a weight")
            rows.append({"training_set": training_set, "head": int(heads[index]),
                         "training": "temporary_reweighting_with_stopping_rule" if reweighted[index] else "unchanged_gd",
                         "total_updates": int(updates[-1]),
                         "update_at_which_the_original_weight_is_restored": int(restored[index]) if reweighted[index] else "",
                         "all_training_concepts_correct": bool(concepts_correct[-1, index]),
                         "test_concept_accuracy": f"{np.mean(test_hard[index] == test_labels):.4f}",
                         "test_count_accuracy": f"{np.mean(test_hard[index][test_pairs].sum(axis=1) == test_counts):.4f}"})
    summary = {}
    for training in ("unchanged_gd", "temporary_reweighting_with_stopping_rule"):
        part = [r for r in rows if r["training"] == training]
        summary[training] = {"runs": len(part), "runs_with_all_training_concepts_correct": sum(r["all_training_concepts_correct"] for r in part),
                             "median_test_concept_accuracy": float(np.median([float(r["test_concept_accuracy"]) for r in part])),
                             "median_test_count_accuracy": float(np.median([float(r["test_count_accuracy"]) for r in part]))}
    switches = [r["update_at_which_the_original_weight_is_restored"] for r in rows if r["training"] != "unchanged_gd"]
    summary["weight_restored_after_updates_min_max"] = [min(switches), max(switches)]
    return rows, summary


def evaluate(root: Path, protocol_path: Path, output: Path) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result: dict[str, object] = {}
    for name, function, table in (("random_initialization", random_initialization, "random_initialization.csv"),
                                  ("recovery", recovery, "recovery.csv")):
        if (root / name).is_dir():
            rows, result[name] = function(root / name, protocol[name])
            if rows:
                write_csv(output / table, rows)
    if (root / "neighborhood_entry").is_dir():
        rows, result["neighborhood_entry"] = neighborhood_entry(root / "neighborhood_entry", protocol["neighborhood_entry"], output)
        if rows:
            write_csv(output / "neighborhood_entry.csv", rows)
    if not result:
        raise ValueError(f"No experiment output under {root}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.input, args.protocol, args.output)
