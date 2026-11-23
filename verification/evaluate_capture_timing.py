"""Recompute the capture-timing results from the saved trajectories, independently of the training code.

The input directory holds the outputs of run_capture_timing.py in step_fraction_<f> for every step size and, in
reversibility, the output of search_label_changing_continuations.py. Capture, convergence and label changes are
recomputed from every saved state with the count dynamic programming of evaluate_cifar_llp.py, and every
label-changing continuation is replayed with explicitly reconstructed Grams. The tables have the columns of
results/capture_timing/*.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from evaluate_cifar_llp import evaluate_states, pack_codes


Array = NDArray[np.float64]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def timing(root: Path, fractions: list[float]) -> tuple[list[dict], dict]:
    rows, summary, final_codes = [], {}, {}
    for fraction in fractions:
        folder = root / f"step_fraction_{fraction}"
        with np.load(folder / "summary.npz", allow_pickle=False) as saved:
            counts, eta, sets = saved["counts"], float(saved["eta"]), saved["training_sets"]
            recorded_capture, recorded_fit, recorded_last = saved["capture"], saved["first_fit"], saved["last_change"]
            grams = saved["grams"]
        history = np.load(folder / "baseline.npy", mmap_mode="r", allow_pickle=False)
        capture = np.full(history.shape[1], -1, dtype=np.int64)
        fit, last = capture.copy(), np.zeros_like(capture)
        previous = pack_codes(history[0] > 0)
        update_error = 0.0
        for begin in range(0, len(history), 128):
            block = np.asarray(history[begin:begin + 128])
            states = evaluate_states(block, counts, 4)
            for events, found in ((states.log_evidence > -np.log(2), capture), (states.well_fitted, fit)):
                selected = (found < 0) & events.any(axis=0)
                found[selected] = begin + events[:, selected].argmax(axis=0)
            changes = states.hard_codes != np.concatenate((previous[None], states.hard_codes[:-1]))
            last = np.maximum(last, np.where(changes, np.arange(begin, begin + len(block))[:, None], 0).max(0))
            previous = states.hard_codes[-1]
            if begin < len(history) - 1:
                # Check one update of every block of 128 against the residual of the dynamic programming.
                prediction = block[0] + eta * np.einsum("bij,bj->bi", grams, states.residual[0])
                update_error = max(update_error, float(np.max(np.abs(prediction - history[begin + 1]))))
        np.testing.assert_array_equal(capture, recorded_capture)
        np.testing.assert_array_equal(fit, recorded_fit)
        np.testing.assert_array_equal(last, recorded_last)
        if update_error > 1e-10:
            raise AssertionError(f"The saved updates differ from the independent residual at fraction {fraction}")
        final_codes[fraction] = previous.copy()
        with np.load(folder / "continuations.npz", allow_pickle=False) as saved:
            initial, code_history = saved["initial"], saved["codes"]
            base, first_flip = saved["baseline_index"], saved["first_flip"]
        np.testing.assert_allclose(initial, history[capture[base], base], atol=0, rtol=0)
        start = evaluate_states(initial, counts, 4)
        if not (start.log_evidence > -np.log(2)).all():
            raise AssertionError("A continuation starts below the capture threshold")
        np.testing.assert_array_equal(code_history[0], start.hard_codes)
        flips = code_history != code_history[0]
        np.testing.assert_array_equal(np.where(flips.any(0), flips.argmax(0), -1), first_flip)
        for index in range(len(capture)):
            rows.append({"step_size_fraction_of_bound": f"{fraction:g}", "step_size": f"{eta:g}", "training_set": int(sets[index // 16]),
                         "run_in_training_set": index % 16, "updates_until_capture": int(capture[index]),
                         "updates_until_convergence": int(fit[index]),
                         "interventions_after_capture": int(np.count_nonzero(base == index)),
                         "interventions_changing_a_label": int(np.count_nonzero(first_flip[base == index] >= 0))})
        both = (capture >= 0) & (fit >= 0)
        summary[f"{fraction:g}"] = {
            "step_size": eta, "runs": len(capture), "captured": int((capture >= 0).sum()), "converged": int((fit >= 0).sum()),
            "median_updates_until_capture": float(np.median(capture[capture >= 0])),
            "step_size_times_median_updates_until_capture": eta * float(np.median(capture[capture >= 0])),
            "step_size_times_median_updates_until_convergence": eta * float(np.median(fit[fit >= 0])),
            "median_capture_as_fraction_of_convergence": float(np.median(capture[both] / fit[both])),
            "interventions_after_capture": len(base), "interventions_changing_a_label": int((first_flip >= 0).sum()),
            "largest_difference_from_independent_update": update_error}
    reference = final_codes.get(0.8)
    if reference is not None:
        for fraction, codes in final_codes.items():
            summary[f"{fraction:g}"]["final_assignments_differing_from_step_fraction_0.8"] = int((codes != reference).sum())
    return rows, summary


def replay(initial: Array, counts: NDArray[np.int64], targets: NDArray[np.int64],
           controls: Array | None, horizon: int, block: int, eta: float) -> tuple[NDArray[np.int64], float]:
    """Replay continuations with explicit Grams K = 4I - 3.75 uu' and return the first label-changing update."""
    current = initial.copy()
    anchor = np.sign(initial)
    signs = anchor[np.arange(len(initial)), targets]
    first = np.full(len(initial), -1, dtype=np.int64)
    active = np.arange(len(initial))
    minimum_increment = float("inf")
    for step in range(1, horizon + 1):
        if not len(active):
            break
        old = evaluate_states(current[active], counts, 4)
        residual = old.residual
        norm = np.linalg.norm(residual, axis=1, keepdims=True)
        unit_r = residual / np.maximum(norm, np.finfo(float).tiny)
        if controls is None:
            unit = np.zeros_like(residual)
            unit[np.arange(len(active)), targets[active]] = signs[active]
            raw = unit_r + unit
        else:
            control = controls[(step - 1) // block, active]
            control_norm = np.linalg.norm(control, axis=1, keepdims=True)
            if np.any(control_norm < 1e-12):
                raise ValueError("A saved control has no well-defined direction")
            unit_v = control / control_norm
            raw = unit_r - unit_v
        length = np.linalg.norm(raw, axis=1, keepdims=True)
        slow = raw / np.maximum(length, 1e-300)
        degenerate = length[:, 0] < 1e-12
        slow[degenerate] = 0.
        if controls is None:
            slow[np.flatnonzero(degenerate), (targets[active][degenerate] + 1) % initial.shape[1]] = 1.
        # The zero vector realizes K = 4I; otherwise u is a unit vector.
        np.testing.assert_allclose(np.sum(slow * slow, axis=1)[~degenerate], 1., atol=1e-13)
        action = 4 * residual - 3.75 * slow * (slow * residual).sum(1, keepdims=True)
        if controls is not None:
            np.testing.assert_allclose(action, 2.125 * residual + 1.875 * norm * unit_v, atol=1e-10)
        if step == 1:
            gram = 4 * np.eye(initial.shape[1]) - 3.75 * np.outer(slow[0], slow[0])
            values = np.linalg.eigvalsh(gram)
            assert values.min() >= .25 - 1e-12 and values.max() <= 4 + 1e-12
        new = current[active] + eta * action
        after = evaluate_states(new, counts, 4)
        increment = float(np.min(after.log_evidence - old.log_evidence))
        minimum_increment = min(minimum_increment, increment)
        assert increment >= -1e-10
        current[active] = new
        flipped = np.min(anchor[active] * new, axis=1) < -1e-8
        first[active[flipped]] = step
        active = active[~flipped]
    if np.any(first < 0):
        raise AssertionError(f"{int((first < 0).sum())} saved continuations do not change a label when replayed")
    return first, minimum_increment


def reversibility(root: Path, source: Path) -> tuple[list[dict], dict]:
    with np.load(source / "summary.npz", allow_pickle=False) as data:
        capture, last, counts, grams = (data[k] for k in ("capture", "last_change", "counts", "grams"))
        eta = float(data["eta"])
    history = np.load(source / "baseline.npy", mmap_mode="r", allow_pickle=False)
    width = history.shape[-1]
    assert eta * 4 * (width + 1) < 4
    eigenvalues = np.linalg.eigvalsh(grams)
    assert eigenvalues.min() >= .25 - 1e-12 and eigenvalues.max() <= 4 + 1e-12
    index = np.arange(len(capture))
    before = evaluate_states(history[capture - 1, index], counts, 4)
    crossed = evaluate_states(history[capture, index], counts, 4)
    assert np.all(before.log_evidence <= -np.log(2)) and np.all(crossed.log_evidence > -np.log(2))
    hard = np.asarray(history) > 0
    changed = np.any(hard[1:] != hard[:-1], axis=-1)
    np.testing.assert_array_equal(np.where(changed, np.arange(1, len(history))[:, None], 0).max(0), last)
    del hard, changed
    witnesses: list[tuple[int, int, str]] = []
    minimum_increment = float("inf")
    searches = []
    with np.load(root / "greedy/continuations.npz", allow_pickle=False) as data:
        success = np.flatnonzero(data["first_flip"] > 0)
        indices, times, targets = (data[k][success] for k in ("source_index", "source_time", "targets"))
        initial = np.asarray(history[times, indices])
        np.testing.assert_array_equal(initial, data["initial"][success])
        assert np.all(evaluate_states(initial, counts, 4).legal)
        steps, increment = replay(initial, counts, targets, None, int(data["first_flip"][success].max()), 1, eta)
        np.testing.assert_array_equal(steps, data["first_flip"][success])
        minimum_increment = min(minimum_increment, increment)
        witnesses.extend((int(i), int(t), "greedy") for i, t in zip(indices, times))
        searches.append({"search": "greedy", "replayed_label_changing_continuations": len(success)})
    for stage in ("optimized", "refined"):
        for search in json.loads((root / stage / "completed.json").read_text())["searches"]:
            with np.load(root / stage / f"{search['tag']}.npz", allow_pickle=False) as data:
                success = np.flatnonzero(data["minimum_signed_margin"] < -1e-8)
                searches.append({"search": f"{stage}/{search['tag']}", "replayed_label_changing_continuations": len(success)})
                if not len(success):
                    continue
                indices, times, targets = (data[k][success] for k in ("source_index", "source_time", "targets"))
                initial = np.asarray(history[times, indices])
                assert np.all(evaluate_states(initial, counts, 4).legal)
                _, increment = replay(initial, counts, targets, data["controls"][:, success],
                                      int(search["horizon"]), int(search["block_updates"]), eta)
                minimum_increment = min(minimum_increment, increment)
                witnesses.extend((int(i), int(t), f"{stage}/{search['tag']}") for i, t in zip(indices, times))
    rows = []
    for run in range(len(capture)):
        starts = [start for source_run, start, _ in witnesses if source_run == run]
        if not starts:
            raise AssertionError(f"No replayed label-changing continuation for run {run}")
        rows.append({"run": run, "updates_until_capture": int(capture[run]),
                     "start_of_latest_label_changing_continuation": max(starts),
                     "updates_from_that_start_until_capture": int(capture[run] - max(starts))})
    delays = np.array([r["updates_from_that_start_until_capture"] for r in rows])
    relative = delays / np.array([r["updates_until_capture"] for r in rows])
    summary = {"runs": len(rows), "replayed_label_changing_continuations": len(witnesses),
               "delay_min_median_max": [int(delays.min()), float(np.median(delays)), int(delays.max())],
               "median_delay_as_fraction_of_updates_until_capture": float(np.median(relative)),
               "minimum_log_evidence_increment_of_replayed_updates": minimum_increment, "searches": searches}
    return rows, summary


def evaluate(root: Path, protocol_path: Path, output: Path) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    rows, summary = timing(root, protocol["fractions_of_the_step_bound"])
    write_csv(output / "runs.csv", rows)
    result: dict[str, object] = {"timing": summary}
    if (root / "reversibility").is_dir():
        fraction = protocol["reversibility"]["fraction_of_the_step_bound"]
        witness_rows, result["reversibility"] = reversibility(root / "reversibility", root / f"step_fraction_{fraction}")
        write_csv(output / "reversibility.csv", witness_rows)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.input, args.protocol, args.output)
