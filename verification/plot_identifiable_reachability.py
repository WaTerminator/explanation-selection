"""Figures 3 and 6. Inputs are the measured endpoints, GD continuation traces and local perturbation curves."""

from __future__ import annotations

import csv
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

import plot_style as style

WEIGHTS = (0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99)
DRAWS = (1, 2, 3)
HEADS_PER_DRAW = 32
RUNS_PER_CONDITION = len(DRAWS) * HEADS_PER_DRAW

TICK_SIZE = 6.0
LABEL_SIZE = 6.5
LEGEND_SIZE = 6.0
NOTE_SIZE = 5.8
PANEL_LABEL_SIZE = 7.0


def check_figure_bounds(fig: Figure) -> None:
    """Fail if any visible text extends beyond the canvas bounds or crowds another text."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.bbox.width, fig.bbox.height
    boxes: list[tuple[str, matplotlib.transforms.Bbox]] = []
    for artist in fig.findobj(matplotlib.text.Text):
        text = artist.get_text().strip()
        if not text or not artist.get_visible():
            continue
        box = artist.get_window_extent(renderer)
        if box.x0 < -0.5 or box.y0 < -0.5 or box.x1 > width + 0.5 or box.y1 > height + 0.5:
            raise ValueError(f"Text clipped by figure edge: {text!r} -> {box}")
        padded = box.expanded(1.04, 1.08)
        for prev_text, prev_box in boxes:
            if padded.overlaps(prev_box):
                raise ValueError(f"Text boxes overlap or crowd: {prev_text!r} vs {text!r}")
        boxes.append((text, padded))


def load_trajectories(results: Path) -> tuple[dict[str, list[int]], list[tuple[int, int, np.ndarray, np.ndarray]]]:
    """Match full-coverage random heads to their observed continuations at mixed-query weight 0.98."""
    with (results / "random_initialization.csv").open(newline="") as stream:
        endpoints = list(csv.DictReader(stream))
    counts: dict[str, list[int]] = {"gd": [], "sgd": []}
    expected_heads = {(draw, head) for draw in DRAWS for head in range(HEADS_PER_DRAW)}
    wrong_heads: set[tuple[int, int]] = set()
    for method in counts:
        for weight in WEIGHTS:
            cohort = [row for row in endpoints if int(row["identified_inputs"]) == 256
                      and row["optimizer"] == method and float(row["rho"]) == weight]
            keys = {(int(row["training_set"]), int(row["head"])) for row in cohort}
            if len(cohort) != len(expected_heads) or keys != expected_heads:
                raise ValueError(f"Incomplete {method} cohort at weight {weight}")
            wrong = {(int(row["training_set"]), int(row["head"])) for row in cohort
                     if row["all_training_concepts"] == "wrong"}
            counts[method].append(len(wrong))
            if method == "gd" and weight == 0.98:
                wrong_heads = wrong

    with (results / "neighborhood_entry.csv").open(newline="") as stream:
        entry = {(int(r["training_set"]), int(r["head"])): int(r["total_updates_at_entry"])
                 for r in csv.DictReader(stream) if r["enters_the_contracting_neighborhood"] == "True"}
    traces: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(list)
    with (results / "entry_traces.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = int(row["training_set"]), int(row["head"])
            traces[key].append((int(row["total_updates"]), float(row["distance_over_radius"])))
    if wrong_heads != set(traces) or wrong_heads != set(entry):
        raise ValueError("Continuation identities differ from the all-wrong GD endpoints")
    trajectories = []
    for key in sorted(traces):
        steps, ratios = np.asarray(sorted(traces[key])).T
        if (steps[0] != 500_000 or steps[-1] != entry[key] or not np.all(np.diff(steps) > 0)
                or not np.all(ratios[:-1] > 1) or not 0 < ratios[-1] <= 1):
            raise ValueError(f"Invalid first-entry trace for training set and head {key}")
        trajectories.append((*key, steps / 1e6, ratios))
    return counts, trajectories


def render_main_figure(results: Path, output: Path) -> None:
    counts, trajectories = load_trajectories(results)

    style.set_paper_style(LABEL_SIZE)
    fig_h = 1.54
    fig = plt.figure(figsize=(style.TEXTWIDTH_IN, fig_h))

    panel_w = 1.64
    panel_h = 0.96
    panel_bottom = 0.51
    left_x = (0.76, 3.22)

    ax_a: Axes = fig.add_axes((
        left_x[0] / style.TEXTWIDTH_IN,
        panel_bottom / fig_h,
        panel_w / style.TEXTWIDTH_IN,
        panel_h / fig_h,
    ))
    ax_b: Axes = fig.add_axes((
        left_x[1] / style.TEXTWIDTH_IN,
        panel_bottom / fig_h,
        panel_w / style.TEXTWIDTH_IN,
        panel_h / fig_h,
    ))

    # --- Panel (a): All-wrong rate across the 9 tested weights ---
    x_rho = np.asarray(WEIGHTS, dtype=np.float64)
    y_gd = 100.0 * np.asarray(counts["gd"], dtype=np.float64) / RUNS_PER_CONDITION
    y_sgd = 100.0 * np.asarray(counts["sgd"], dtype=np.float64) / RUNS_PER_CONDITION

    ax_a.axhline(0.0, color=style.GUIDE_GRAY, linewidth=0.45, linestyle="-", zorder=1)

    ax_a.plot(
        x_rho, y_gd,
        color=style.BLUE, linewidth=0.75, linestyle="-", zorder=2,
    )
    ax_a.plot(
        x_rho, y_gd,
        linestyle="none", marker="o", markersize=5.0,
        markerfacecolor="white", markeredgecolor=style.BLUE, markeredgewidth=0.72,
        zorder=3,
    )
    ax_a.plot(
        x_rho, y_sgd,
        linestyle="none", marker="o", markersize=2.1,
        markerfacecolor=style.ROSE, markeredgecolor="white", markeredgewidth=0.28,
        zorder=4,
    )

    # Clear single callout at rho = 0.98 identifying the 25/96 GD cohort followed in Panel (b).
    idx_098 = WEIGHTS.index(0.98)
    ax_a.annotate(
        r"$\rho=0.98$: 25/96 GD",
        xy=(x_rho[idx_098], y_gd[idx_098]),
        xytext=(-13, 0),
        textcoords="offset points",
        ha="right", va="center",
        fontsize=NOTE_SIZE, color=style.BLUE,
        arrowprops={
            "arrowstyle": "-",
            "color": style.BLUE,
            "lw": 0.50,
            "shrinkA": 2.0,
            "shrinkB": 3.4,
        },
        zorder=5,
    )

    ax_a.set_xlim(0.831, 0.998)
    ax_a.set_xticks([0.84, 0.88, 0.92, 0.96, 0.99], labels=["0.84", "0.88", "0.92", "0.96", "0.99"])
    ax_a.set_ylim(-3.2, 50.5)
    ax_a.set_yticks([0, 10, 20, 30, 40, 50])
    ax_a.set_ylabel("All-wrong runs (%)")

    legend_handles_a = [
        Line2D(
            [], [], color=style.BLUE, linewidth=0.75,
            marker="o", markersize=4.7, markerfacecolor="white",
            markeredgecolor=style.BLUE, markeredgewidth=0.72,
            label="Full-batch GD",
        ),
        Line2D(
            [], [], linestyle="none",
            marker="o", markersize=2.2, markerfacecolor=style.ROSE,
            markeredgecolor="white", markeredgewidth=0.28,
            label="Minibatch SGD",
        ),
    ]
    ax_a.legend(
        handles=legend_handles_a,
        loc="upper left",
        bbox_to_anchor=(0.04, 0.96),
        borderaxespad=0.0,
        handlelength=1.35,
        handletextpad=0.40,
        labelspacing=0.28,
        fontsize=LEGEND_SIZE,
    )

    # --- Panel (b): Following the 25 GD runs into proved contracting ball ---
    y_min, y_max = 0.44, 82.0
    ax_b.set_yscale("log")
    ax_b.axhspan(y_min, 1.0, facecolor=style.LIGHT_BLUE, alpha=0.17, zorder=0)
    ax_b.axhline(1.0, color=style.MUTED_INK, linewidth=0.60, linestyle=(0, (3.5, 2.5)), zorder=2)

    entry_x: list[float] = []
    entry_y: list[float] = []
    for _, _, steps_m, ratios in trajectories:
        ax_b.plot(steps_m, ratios, color=style.BLUE, linewidth=0.48, alpha=0.34, zorder=3)
        entry_x.append(float(steps_m[-1]))
        entry_y.append(float(ratios[-1]))

    ax_b.scatter(
        entry_x, entry_y,
        s=7.0, facecolor=style.LIGHT_BLUE, edgecolor=style.BLUE,
        linewidths=0.45, zorder=5,
    )

    ax_b.text(
        0.65, 0.64, "Convergence guaranteed",
        fontsize=NOTE_SIZE, color=style.INK, va="center", ha="left", zorder=5,
    )

    ax_b.set_xlim(0.15, 12.85)
    ax_b.set_xticks([0.5, 4, 8, 12], labels=["0.5", "4", "8", "12"])
    ax_b.set_ylim(y_min, y_max)
    ax_b.set_yticks([1, 10, 50], labels=["1", "10", "50"])
    ax_b.minorticks_off()
    ax_b.set_ylabel("Normalized distance")

    for ax in (ax_a, ax_b):
        style.clean_axes(ax)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.yaxis.label.set_fontsize(LABEL_SIZE)
        ax.yaxis.labelpad = 2.5

    xlabel_y = 0.28 / fig_h
    title_y = 0.12 / fig_h
    panel_specs = (
        (0, r"Mixed-query weight $\rho$", "a", "All-wrong rate from random heads"),
        (1, "Cumulative updates (millions)", "b", "Following the 25 GD runs"),
    )
    for idx, xlabel, letter, title in panel_specs:
        cx = (left_x[idx] + panel_w / 2) / style.TEXTWIDTH_IN
        fig.text(cx, xlabel_y, xlabel, fontsize=LABEL_SIZE, ha="center", va="center", color=style.INK)
        fig.text(cx, title_y, f"({letter}) {title}", fontsize=PANEL_LABEL_SIZE, ha="center", va="top", color=style.INK)

    check_figure_bounds(fig)
    output.parent.mkdir(parents=True, exist_ok=True)
    style.save_all_formats(fig, output)


def render_local_contraction_figure(
    local_curves: Path,
    local_summary: Path,
    output: Path,
) -> None:
    """Render the two-panel Appendix G.6 figure (logit distance + quadratic energy, without curvature)."""
    centers = json.loads(local_summary.read_text(encoding="utf-8"))["centers"]
    expected = {(draw, rho) for draw in DRAWS for rho in (0.85, 0.9)}
    assert len(centers) == 6 and {(int(c["draw"]), float(c["rho"])) for c in centers} == expected
    assert all(float(c["effective_hessian_min"]) > 0 for c in centers)

    with local_curves.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    style.set_paper_style(LABEL_SIZE)
    fig_h = 1.68
    fig = plt.figure(figsize=(style.TEXTWIDTH_IN, fig_h))

    panel_w = 1.64
    panel_h = 0.94
    panel_bottom = 0.50
    left_x = (0.76, 3.22)

    ax_dist: Axes = fig.add_axes((
        left_x[0] / style.TEXTWIDTH_IN,
        panel_bottom / fig_h,
        panel_w / style.TEXTWIDTH_IN,
        panel_h / fig_h,
    ))
    ax_energy: Axes = fig.add_axes((
        left_x[1] / style.TEXTWIDTH_IN,
        panel_bottom / fig_h,
        panel_w / style.TEXTWIDTH_IN,
        panel_h / fig_h,
    ))

    conditions = [
        ("weak_curvature", "Weak curvature", style.BLUE, "-"),
        ("strong_curvature", "Strong curvature", style.GOLD, "--"),
        ("random_parameter", "Random parameter", style.ROSE, "-."),
        ("random_logit", "Random logit", style.MUTED_INK, ":"),
    ]
    for branch, _, color, linestyle in conditions:
        grouped = [
            [
                row for row in rows
                if (int(row["draw"]), float(row["rho"]), row["branch"]) == (draw, rho, branch)
            ]
            for draw in DRAWS
            for rho in (0.85, 0.9)
        ]
        steps = np.array([int(row["step"]) for row in grouped[0]], dtype=np.float64) / 1000.0
        assert all(len(group) == 101 for group in grouped)
        for ax, metric in ((ax_dist, "logit_rms_distance"), (ax_energy, "quadratic_energy")):
            values = np.array([[float(row[metric]) for row in group] for group in grouped], dtype=np.float64)
            assert np.all(values > 0)
            ratios = values / values[:, :1]
            ax.fill_between(steps, ratios.min(axis=0), ratios.max(axis=0), color=color, alpha=0.15, linewidth=0)
            ax.plot(steps, np.median(ratios, axis=0), color=color, linestyle=linestyle, linewidth=1.05)

    for ax in (ax_dist, ax_energy):
        ax.set_xlim(0, 50)
        ax.set_xticks([0, 25, 50])
        style.clean_axes(ax)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.yaxis.label.set_fontsize(LABEL_SIZE)
        ax.yaxis.labelpad = 2.5

    ax_dist.set_ylim(-0.03, 1.04)
    ax_dist.set_yticks([0, 0.5, 1], labels=["0", "0.5", "1"])
    ax_dist.set_ylabel("Relative logit distance")

    ax_energy.set_yscale("log")
    ax_energy.set_ylim(1e-16, 2.2)
    ax_energy.set_yticks([1, 1e-5, 1e-10, 1e-15], labels=["1", r"$10^{-5}$", r"$10^{-10}$", r"$10^{-15}$"])
    ax_energy.minorticks_off()
    ax_energy.set_ylabel("Relative quadratic energy")

    handles = [
        Line2D([], [], color=color, linestyle=line, linewidth=1.25, label=label)
        for _, label, color, line in conditions
    ]
    legend_center_x = 0.5 * (left_x[0] + left_x[1] + panel_w) / style.TEXTWIDTH_IN
    fig.legend(
        handles=handles,
        ncol=4,
        loc="center",
        bbox_to_anchor=(legend_center_x, 1.56 / fig_h),
        borderaxespad=0.0,
        handlelength=1.6,
        handletextpad=0.38,
        columnspacing=0.95,
        fontsize=LEGEND_SIZE,
    )

    xlabel_y = 0.28 / fig_h
    title_y = 0.12 / fig_h
    for idx, (letter, title) in enumerate((
        ("a", "Return in logit distance"),
        ("b", "Decay of quadratic energy"),
    )):
        cx = (left_x[idx] + panel_w / 2) / style.TEXTWIDTH_IN
        fig.text(cx, xlabel_y, "Updates (thousands)", fontsize=LABEL_SIZE, ha="center", va="center", color=style.INK)
        fig.text(cx, title_y, f"({letter}) {title}", fontsize=PANEL_LABEL_SIZE, ha="center", va="top", color=style.INK)

    check_figure_bounds(fig)
    output.parent.mkdir(parents=True, exist_ok=True)
    style.save_all_formats(fig, output)
