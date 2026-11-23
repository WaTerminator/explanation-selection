"""Run the experiments and checks accompanying the manuscript."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "verification"))

FROZEN_ENCODER = ("selection", "restricted", "recovery", "minima", "local")
ADDITIONAL_EXPERIMENTS = ("joint_training", "capture_timing", "natural_access", "sudoku")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def run(script: str, arguments: list[str]) -> None:
    subprocess.run([sys.executable, str(ROOT / "verification" / script), *map(str, arguments)],
                   cwd=ROOT, check=True)


def protocol(name: str) -> dict:
    return json.loads((ROOT / "protocols" / f"{name}.json").read_text(encoding="utf-8"))


def check_reference() -> None:
    expected_feature = "41d6314d5719c20bb0e31e1a153f8ca7a46b91556e82b11345f4341adfc418f1"
    require(digest(ROOT / "data/features.npz") == expected_feature, "Feature identity mismatch")

    folder = ROOT / "results/selection"
    endpoints, pairs = rows(folder / "endpoints.csv"), rows(folder / "pairs.csv")
    continued = [r for r in rows(folder / "continuations.csv") if r["T_capture"]]
    require(len(endpoints) == 144 and len(pairs) == 144 and all(r["both_resolved_changed_noncomplement"] == "True" for r in pairs),
            "Matched-spectrum comparisons across six sets and four initial heads")
    require(len(continued) == 360 and all(r["ever_anchor_flip"] == "False" for r in continued), "Capture preservation")
    raw = [r for r in endpoints if r["metric"] == "raw_scaled"]
    require(len(raw) == 24 and all(not r["T_capture"] for r in raw), "Uncaptured reference")
    lookup = {(r["draw"], r["head"], r["metric"]): r for r in endpoints}
    for metric, expected in (("test_accuracy", 11.225), ("test_count_accuracy", 5.0)):
        gaps = []
        for pair in pairs:
            group = pair["draw"], pair["head"]
            left = float(lookup[(*group, pair["left_metric"])][metric])
            right = float(lookup[(*group, pair["right_metric"])][metric])
            gaps.append(100 * abs(left - right))
        require(math.isclose(statistics.median(gaps), expected), f"Paired test gap in {metric}")
    folder = ROOT / "results/restricted"
    endpoints, continued = rows(folder / "endpoints.csv"), rows(folder / "continuations.csv")
    require(len(endpoints) == 48 and all(int(r["capture_step"]) >= 0 for r in endpoints), "Restricted captures")
    require(len(continued) == 144 and all(r["triggered"] == "True" and int(r["sign_changes"]) == 0 for r in continued),
            "Restricted continuations")
    ordinary = rows(ROOT / "results/recovery/endpoints.csv")
    require(len(ordinary) == 144 and all(float(r["train_concept_acc"]) == 1 for r in ordinary), "Random-head recovery")
    minima = json.loads((ROOT / "results/minima/result.json").read_text(encoding="utf-8"))
    require(len(minima["centers"]) == 6 and all(r["effective_hessian_min"] > 0 for r in minima["centers"]),
            "Stationary-center curvature")
    require(len(minima["contraction_ratios"]) == 24 and all(0 < r["logit_rms_distance"] < 1 for r in minima["contraction_ratios"]),
            "Local contraction")
    require(len(minima["endpoints"]) == 30 and all(r["concept_accuracy"] == 0 for r in minima["endpoints"]),
            "Wrong-label persistence near the centers")
    local = rows(ROOT / "results/local/local_stability.csv")
    compared = [r for r in local if r["claim_test"] == "1"]
    escapes = [r for r in compared if r["theory_stable"] == "1" and r["outcome"] == "escaped"]
    require((len(local), len(compared), len(escapes)) == (14580, 9720, 97),
            "Finite-radius comparison and retained escapes")
    check_joint_training()
    check_capture_timing()
    check_natural_access()
    check_sudoku()
    print("Feature cache and reference results: passed", flush=True)


def check_joint_training() -> None:
    folder = ROOT / "results/joint_training"
    runs, pairs = rows(folder / "runs.csv"), rows(folder / "pairs.csv")
    legal = [r for r in runs if r["satisfies_all_counts"] == "True"]
    by = {optimizer: [r for r in legal if r["optimizer"] == optimizer] for optimizer in ("sgd", "adam")}
    require((len(runs), len(by["sgd"]), len(by["adam"])) == (64, 28, 32), "Joint training: runs satisfying all counts")
    common = [int(p["training_labels_on_which_the_models_differ"]) for p in pairs if p["both_satisfy_all_counts"] == "True"]
    require((len(common), min(common), statistics.median(common), max(common)) == (28, 48, 90, 200),
            "Joint training: labels on which SGD and Adam differ")
    mislabeled = {o: 100 * statistics.mean(int(r["mislabeled_training_images"]) for r in by[o]) / 1024 for o in by}
    lost = {o: 100 * statistics.mean(float(r["reference_test_concept_accuracy"]) - float(r["test_concept_accuracy"]) for r in by[o]) for o in by}
    require((f"{mislabeled['sgd']:.2f}", f"{mislabeled['adam']:.2f}", f"{lost['sgd']:.1f}", f"{lost['adam']:.1f}")
            == ("10.61", "7.72", "10.2", "7.5"), "Joint training: mislabeled training images and lost accuracy")
    require(all(float(r["reference_test_concept_accuracy"]) > float(r["test_concept_accuracy"]) for r in legal),
            "Joint training: every reference is more accurate")
    continued = rows(folder / "continuations.csv")
    require(len(continued) == 49 and sum(int(r["updates"]) for r in continued) == 12544
            and all(int(r["updates"]) == 256 and r["updates_changing_a_training_label"] == "0" for r in continued),
            "Joint training: continued models keep their training labels")
    for filename, expected in (("original_pairs.csv", (27, 30, 0.35, 1.40)), ("pairs.csv", (28, 90, 4.075, 7.50))):
        cohort = rows(folder / filename)
        fitted = [r for r in cohort if r["both_satisfy_all_counts"] == "True"]
        keys = {(r["training_set"], r["head"]) for r in cohort}
        require(len(cohort) == 32 and keys == {(r["training_set"], r["head"]) for r in pairs}, "Joint-training pair identities")
        require(all(int(r["training_labels_on_which_the_models_differ"]) > 0 for r in fitted), "Joint-training selections")
        medians = [statistics.median(float(r[column]) for r in fitted) for column in
                   ("training_labels_on_which_the_models_differ", "test_concept_gap_pp", "test_count_gap_pp")]
        require(len(fitted) == expected[0] and all(math.isclose(x, y) for x, y in zip(medians, expected[1:])),
                f"Joint-training rate comparison in {filename}")
    frozen = rows(folder / "frozen_encoder.csv")
    require(len(frozen) == 64 and all(r["both_fit"] == "True" and int(r["training_labels_on_which_the_models_differ"]) > 0 for r in frozen),
            "Frozen encoder: SGD and Adam fit and select different labels")


def check_capture_timing() -> None:
    folder = ROOT / "results/capture_timing"
    runs = rows(folder / "runs.csv")
    medians = [statistics.median(int(r["updates_until_capture"]) for r in runs if r["step_size_fraction_of_bound"] == fraction)
               for fraction in ("0.2", "0.5", "0.8", "0.95")]
    require(len(runs) == 384 and all(int(r["updates_until_capture"]) > 0 and int(r["updates_until_convergence"]) > 0 for r in runs)
            and medians == [3608, 1446.5, 906, 763.5], "Capture timing: captured runs and median capture times")
    require(sum(int(r["interventions_after_capture"]) for r in runs) == 1152
            and all(r["interventions_changing_a_label"] == "0" for r in runs), "Capture timing: interventions after capture")
    delays = [int(r["updates_from_that_start_until_capture"]) for r in rows(folder / "reversibility.csv")]
    require((len(delays), min(delays), statistics.median(delays), max(delays)) == (96, 161, 240, 322),
            "Capture timing: delay between the latest label-changing continuation and capture")


def check_natural_access() -> None:
    folder = ROOT / "results/natural_access"
    random_runs = rows(folder / "random_initialization.csv")

    def wrong(optimizer: str, identified: str, rho: str | None = None) -> int:
        return sum(r["all_training_concepts"] == "wrong" for r in random_runs if r["optimizer"] == optimizer
                   and r["identified_inputs"] == identified and (rho is None or r["rho"] == rho))

    weights = ("0.84", "0.86", "0.88", "0.90", "0.92", "0.94", "0.96", "0.98", "0.99")
    require([wrong("gd", "256", rho) for rho in weights] == [0, 0, 0, 0, 0, 2, 7, 25, 42]
            and [wrong("sgd", "256", rho) for rho in weights] == [0, 0, 0, 0, 0, 2, 6, 26, 42],
            "Random initialization: entirely wrong runs by weight")
    require([wrong("gd", "32"), wrong("gd", "4"), wrong("sgd", "32"), wrong("sgd", "4")] == [37, 49, 36, 48],
            "Random initialization: entirely wrong runs with few identified inputs")
    require(wrong("gd", "256", "0.8889") == wrong("sgd", "256", "0.8889") == 0
            and len([r for r in random_runs if r["rho"] == "0.8889"]) == 192, "Equal query weights with full coverage")
    entry = rows(folder / "neighborhood_entry.csv")
    require(len(entry) == 25 and all(r["enters_the_contracting_neighborhood"] == "True" for r in entry),
            "Neighborhood entry: every wrong run enters")
    recovered = rows(folder / "recovery.csv")
    correct = {training: sum(r["all_training_concepts_correct"] == "True" for r in recovered if r["training"] == training)
               for training in ("unchanged_gd", "temporary_reweighting_with_stopping_rule")}
    require(len(recovered) == 50 and correct == {"unchanged_gd": 0, "temporary_reweighting_with_stopping_rule": 25},
            "Recovery: unchanged training against temporary reweighting")


def check_sudoku() -> None:
    runs = rows(ROOT / "results/sudoku/runs.csv")
    rules = {fraction: [r for r in runs if r["supervision"] == "sudoku_rules" and r["fraction_of_boards_revealing_a_cell"] == fraction]
             for fraction in ("1.0", "0.2", "0.02")}
    require([len(rules[f]) for f in rules] == [8, 8, 8]
            and [sum(r["learns_the_concepts"] == "True" for r in rules[f]) for f in rules] == [8, 2, 1],
            "Sudoku: runs learning the concepts")
    others = [r for f in ("0.2", "0.02") for r in rules[f] if r["learns_the_concepts"] == "False"]
    require(len(others) == 13 and all(float(r["training_boards_satisfying_all_constraints"]) == 1 and r["revealed_cells_all_correct"] == "True"
                                      and float(r["cells_following_one_class_permutation"]) >= 0.997 for r in others),
            "Sudoku: the other runs satisfy every constraint with permuted classes")


def check() -> None:
    check_reference()
    run("run_cifar_llp.py", ["--self-check"])
    run("run_cifar_llp_restricted.py", ["--self-check"])
    run("evaluate_cifar_llp_restricted.py", ["--self-check"])
    run("verify_fixed_linear_capture.py", [])
    from run_cifar_identifiable_traps import implementation_check
    print(json.dumps(implementation_check("cpu"), indent=2))
    import sudoku_rules
    print(json.dumps(sudoku_rules.implementation_check("cpu"), indent=2))
    from search_label_changing_continuations import implementation_check as search_check
    search_check("cpu")
    print("Posterior, gradient, and parameter-update checks: passed", flush=True)


def run_count_experiment(name: str, output: Path) -> None:
    folder = output / name
    require(not folder.exists(), f"Choose a fresh output directory: {folder}")
    features = str(ROOT / "data/features.npz")
    protocol_path = str(ROOT / "protocols" / f"{name}.json")
    suffix = "_restricted" if name == "restricted" else ""
    run(f"run_cifar_llp{suffix}.py", ["--features", features, "--protocol", protocol_path,
        "--phase", "evaluation", "--output", folder])
    arguments = ["--input", folder if suffix else folder / "states.npz",
                 "--protocol", protocol_path, "--output", folder / "evaluation"]
    if suffix:
        arguments += ["--features", features]
    run(f"evaluate_cifar_llp{suffix}.py", arguments)


def run_recovery(output: Path, device: str) -> None:
    folder = output / "recovery"
    require(not folder.exists(), f"Choose a fresh output directory: {folder}")
    protocol_path = ROOT / "protocols/recovery.json"
    run("run_cifar_identifiable_traps.py", ["--features", ROOT / "data/features.npz", "--protocol", protocol_path,
        "--phase", "evaluation", "--output", folder, "--device", device])


def run_local(output: Path, quick: bool) -> None:
    from run_local_stability import run_local_experiment, run_rate_check
    folder = output / "local"
    folder.mkdir(parents=True, exist_ok=False)
    summary = {"quick": quick, "local": run_local_experiment(folder, quick),
               "rate": run_rate_check(folder, quick)}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def chosen_sets(listed: list[int], requested: list[int] | None) -> list[int]:
    require(requested is None or set(requested) <= set(listed), f"Training sets must be among {listed}")
    return listed if requested is None else requested


def run_joint_training(output: Path, device: str, data_root: Path, training_sets: list[int] | None, rate_setting: str) -> None:
    """SGD, Adam and the reference from every initial network, the continuations, and the frozen-encoder comparison."""
    folder = output / ("joint_training_original" if rate_setting == "original" else "joint_training")
    path, frozen = ROOT / "protocols/joint_training.json", ROOT / "protocols/frozen_optimizers.json"
    settings = protocol("joint_training")
    features = ROOT / "data/features.npz"
    branches = [("sgd", "counts", "sgd"), ("adam", "counts", "adam")]
    if rate_setting == "expanded":
        branches.append(("adam", "labels", "reference"))
    for training_set in chosen_sets(settings["training_sets"], training_sets):
        require(not (folder / f"set_{training_set}").exists(), f"Choose a fresh output directory: {folder / f'set_{training_set}'}")
        for head in settings["heads"]:
            for optimizer, supervision, name in branches:
                run("run_joint_training.py", ["--protocol", path, "--features", features, "--data-root", data_root,
                    "--training-set", training_set, "--head", head, "--optimizer", optimizer, "--supervision", supervision,
                    "--output", folder / f"set_{training_set}" / f"{name}_h{head}", "--device", device, "--rates", rate_setting])
            if rate_setting == "original":
                continue
            for optimizer in ("sgd", "adam"):
                source = folder / f"set_{training_set}" / f"{optimizer}_h{head}"
                final = json.loads((source / "completed.json").read_text(encoding="utf-8"))["final"]
                # Only models whose evidence exceeds one half are continued.
                if final["log_global_evidence"] > -math.log(2):
                    run("run_joint_continuation.py", ["--protocol", path, "--source", source, "--data-root", data_root,
                        "--output", folder / f"set_{training_set}" / f"{optimizer}_h{head}_continued", "--device", device])
        if rate_setting == "original":
            continue
        for size in protocol("frozen_optimizers")["sizes"]:
            run("run_frozen_optimizers.py", ["--protocol", frozen, "--features", features, "--training-set", training_set,
                "--size", size, "--output", folder / "frozen_encoder" / f"set_{training_set}" / f"n{size}", "--device", device])
    run("evaluate_joint_training.py", ["--input", folder, "--output", folder / "evaluation", "--rates", rate_setting])


def run_capture_timing(output: Path, device: str) -> None:
    folder = output / "capture_timing"
    require(not folder.exists(), f"Choose a fresh output directory: {folder}")
    path = ROOT / "protocols/capture_timing.json"
    settings = protocol("capture_timing")
    for fraction in settings["fractions_of_the_step_bound"]:
        run("run_capture_timing.py", ["--protocol", path, "--selection-protocol", ROOT / "protocols/selection.json",
            "--features", ROOT / "data/features.npz", "--fraction", fraction, "--output", folder / f"step_fraction_{fraction}"])
    run("search_label_changing_continuations.py", ["--protocol", path,
        "--source", folder / f"step_fraction_{settings['reversibility']['fraction_of_the_step_bound']}",
        "--output", folder / "reversibility", "--device", device])
    run("evaluate_capture_timing.py", ["--input", folder, "--protocol", path, "--output", folder / "evaluation"])


def run_natural_access(output: Path, device: str, training_sets: list[int] | None) -> None:
    folder = output / "natural_access"
    path = ROOT / "protocols/natural_access.json"
    settings = protocol("natural_access")
    entry_condition = settings["neighborhood_entry"]["condition"]
    for training_set in chosen_sets(settings["random_initialization"]["training_sets"], training_sets):
        name = f"set_{training_set}"
        require(not (folder / "random_initialization" / name).exists(), f"Choose a fresh output directory: {folder / name}")
        for condition in settings["random_initialization"]["conditions"]:
            for optimizer in ("gd", "sgd"):
                run("run_random_initialization.py", ["--protocol", path, "--features", ROOT / "data/features.npz",
                    "--training-set", training_set, "--condition", condition["name"], "--optimizer", optimizer,
                    "--output", folder / "random_initialization" / name / condition["name"] / optimizer, "--device", device])
        run("run_neighborhood_entry.py", ["--protocol", path,
            "--source", folder / "random_initialization" / name / entry_condition / "gd",
            "--output", folder / "neighborhood_entry" / name, "--device", device])
        run("run_recovery.py", ["--protocol", path, "--source", folder / "neighborhood_entry" / name,
            "--output", folder / "recovery" / name, "--device", device])
    run("evaluate_natural_access.py", ["--input", folder, "--protocol", path, "--output", folder / "evaluation"])


def run_sudoku(output: Path, device: str, data_root: Path, training_sets: list[int] | None) -> None:
    folder = output / "sudoku"
    path = ROOT / "protocols/sudoku.json"
    settings = protocol("sudoku")
    fractions = settings["fractions_of_boards_revealing_a_cell"]
    for training_set in chosen_sets(settings["training_sets"], training_sets):
        require(not (folder / f"set_{training_set}").exists(), f"Choose a fresh output directory: {folder / f'set_{training_set}'}")
        for seed in settings["seeds"]:
            # The reference on the true labels shares the boards and minibatches of the runs with every cell revealed.
            for supervision, fraction in [("rules", fraction) for fraction in fractions] + [("labels", max(fractions))]:
                run("run_sudoku.py", ["--protocol", path, "--data-root", data_root, "--training-set", training_set,
                    "--seed", seed, "--fraction", fraction, "--supervision", supervision,
                    "--output", folder / f"set_{training_set}" / f"{supervision}_f{fraction}_seed{seed}", "--device", device])
    run("evaluate_sudoku.py", ["--input", folder, "--output", folder / "evaluation"])


def figures(output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import plot_cifar_llp as selection
    from plot_identifiable_reachability import render_main_figure, render_local_contraction_figure
    from plot_joint_training import plot as plot_joint_training
    from plot_local_rate import plot_rate_panel
    import plot_routing_capture as routing
    import plot_style as style

    output.mkdir(parents=True, exist_ok=False)
    paths = routing.load_trajectories(ROOT / "assets/fig_phase_portrait.tex")
    routing.check_inputs(paths)
    style.set_paper_style(base_size=9.5)
    fig, ax = plt.subplots(figsize=(1.9, 1.8))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.98)
    ax.set_aspect("equal")
    routing.phase_portrait_panel(ax, paths)
    style.save_all_formats(fig, output / "figure1")
    pair = selection.select_pair(selection.read_endpoints(ROOT / "results/selection/endpoints.csv"))
    with np.load(ROOT / "assets/figure2.npz", allow_pickle=False) as data:
        example = selection.Example(pair, data["images"], data["indices"], data["truth"],
            data["counts"], data["evidence"], data["max_violations"],
            int(data["continuation_horizon"]), True, data["hamming_distances"])
    style.set_paper_style(base_size=7)
    selection.render_example(example, output)
    render_main_figure(ROOT / "results/natural_access", output / "figure3")
    plot_joint_training(ROOT / "results/joint_training/runs.csv", ROOT / "results/joint_training/pairs.csv", output / "figure4")
    render_local_contraction_figure(ROOT / "results/minima/curves.csv", ROOT / "results/minima/result.json", output / "figure6")
    with plt.rc_context(plt.rcParamsDefault):
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
                             "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
                             "legend.fontsize": 9, "pdf.fonttype": 42, "svg.fonttype": "none",
                             "axes.spines.top": False, "axes.spines.right": False})
        fig, ax = plt.subplots(figsize=(4.4, 2.1))
        plot_rate_panel(ax, rows(ROOT / "results/local/rate_traces.csv"), legend_size=9)
        fig.tight_layout()
        for suffix in ("pdf", "svg", "png"):
            fig.savefig(output / f"figure5.{suffix}", bbox_inches="tight", dpi=600)
        plt.close(fig)
    print(f"Figures written to {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("check", *FROZEN_ENCODER, *ADDITIONAL_EXPERIMENTS, "figures", "all"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-root", type=Path, default=ROOT / "generated/cifar10",
                        help="CIFAR-10 image files, for joint_training and sudoku")
    parser.add_argument("--training-sets", type=int, nargs="+",
                        help="restrict joint_training, natural_access or sudoku to these training sets")
    parser.add_argument("--rates", choices=("expanded", "original"), help="joint-training rates from Appendix G.8")
    parser.add_argument("--quick", action="store_true", help="Reduced local-stability smoke run only")
    args = parser.parse_args()
    if args.rates and args.experiment != "joint_training":
        parser.error("--rates applies only to joint_training")
    if args.quick and args.experiment != "local":
        parser.error("--quick applies only to the local-stability smoke run")
    if args.training_sets and args.experiment not in ("joint_training", "natural_access", "sudoku"):
        parser.error("--training-sets applies only to joint_training, natural_access and sudoku")
    if args.experiment in ("joint_training", "natural_access", "sudoku") and not args.device.startswith("cuda"):
        parser.error(f"{args.experiment} needs a CUDA device, for example --device cuda:0")
    default_output = ROOT / "generated/figures" if args.experiment == "figures" else ROOT / "generated"
    output = (args.output or default_output).resolve()
    if args.experiment == "check":
        check()
        return
    if args.experiment == "figures":
        figures(output)
        return
    # `all` runs the five experiments of the fixed-encoder model, which need no GPU.
    experiments = FROZEN_ENCODER if args.experiment == "all" else (args.experiment,)
    for name in experiments:
        if name in ("selection", "restricted"):
            run_count_experiment(name, output)
        elif name == "recovery":
            run_recovery(output, args.device)
        elif name == "minima":
            run("run_cifar_identifiable_minima.py", ["--root", str(ROOT / "data/minima_inputs"),
                "--protocol", str(ROOT / "protocols/minima.json"), "--output", str(output / "minima"),
                "--device", args.device])
        elif name == "local":
            run_local(output, args.quick)
        elif name == "joint_training":
            run_joint_training(output, args.device, args.data_root.resolve(), args.training_sets, args.rates or "expanded")
        elif name == "capture_timing":
            run_capture_timing(output, args.device)
        elif name == "natural_access":
            run_natural_access(output, args.device, args.training_sets)
        elif name == "sudoku":
            run_sudoku(output, args.device, args.data_root.resolve(), args.training_sets)


if __name__ == "__main__":
    main()
