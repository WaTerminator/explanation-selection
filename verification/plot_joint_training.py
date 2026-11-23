"""Plot the jointly trained SGD and Adam models of Section 6.4.

Three panels share one fixed geometry at the paper's text width:
(a) the number of training labels on which the SGD and Adam models trained
    from the same initial network disagree, for every initial network in
    which both models satisfy all training counts;
(b) mean accuracy on unseen images and unseen bag counts for the
    count-satisfying SGD and Adam models and for the reference trained on the
    correct concept labels;
(c) for every count-satisfying model, the test concept accuracy it loses
    against the reference from the same initial network, against its
    percentage of mislabeled training images.

Every plotted number is asserted against the values quoted in the manuscript.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

import plot_style as style

SGD = (style.ROSE, style.LIGHT_ROSE)
ADAM = (style.BLUE, style.LIGHT_BLUE)
REFERENCE = (style.GRAY, style.LIGHT_GRAY)

# One geometry for all panels, in inches. The rows below the axes are shared: tick labels, then the axis label
# (the legend of the middle panel takes that row, which has no axis label), then the panel label.
HEIGHT = 1.50
PANEL_WIDTH, PANEL_HEIGHT, PANEL_BOTTOM = 1.30, 0.91, 0.53
PANEL_LEFT = (0.42, 2.26, 4.10)
XLABEL_Y, PANEL_LABEL_Y = 0.30, 0.125
# One type scale for the three small panels.
TICK_SIZE, LABEL_SIZE, NOTE_SIZE, LEGEND_SIZE, PANEL_LABEL_SIZE = 6.0, 6.5, 5.5, 6.0, 7.0


def panel(fig, index: int):
    return fig.add_axes((PANEL_LEFT[index] / style.TEXTWIDTH_IN, PANEL_BOTTOM / HEIGHT,
                         PANEL_WIDTH / style.TEXTWIDTH_IN, PANEL_HEIGHT / HEIGHT))


def center(index: int) -> float:
    return (PANEL_LEFT[index] + PANEL_WIDTH / 2) / style.TEXTWIDTH_IN


def check_layout(fig, value_labels) -> None:
    """Fail when a text is clipped by the figure edge or when two value labels of the bar panel touch."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.bbox.width, fig.bbox.height
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_text().strip() or not artist.get_visible():
            continue
        box = artist.get_window_extent(renderer)
        if box.x0 < -0.5 or box.y0 < -0.5 or box.x1 > width + 0.5 or box.y1 > height + 0.5:
            raise ValueError(f"text clipped by the figure edge: {artist.get_text()!r}")
    boxes = [label.get_window_extent(renderer) for label in value_labels]
    for i, first in enumerate(boxes):
        for second in boxes[i + 1:]:
            if first.overlaps(second):
                raise ValueError("value labels of the accuracy panel overlap")


def load(runs_path: Path, pairs_path: Path) -> dict[str, object]:
    with runs_path.open(newline="", encoding="utf-8") as stream:
        runs = list(csv.DictReader(stream))
    with pairs_path.open(newline="", encoding="utf-8") as stream:
        pairs = list(csv.DictReader(stream))
    assert len(runs) == 64 and len(pairs) == 32
    legal = {optimizer: [r for r in runs if r["optimizer"] == optimizer and r["satisfies_all_counts"] == "True"] for optimizer in ("sgd", "adam")}
    assert (len(legal["sgd"]), len(legal["adam"])) == (28, 32)
    distances = sorted(int(p["training_labels_on_which_the_models_differ"]) for p in pairs if p["both_satisfy_all_counts"] == "True")
    assert len(distances) == 28 and (distances[0], float(np.median(distances)), distances[-1]) == (48, 90.0, 200)
    means = {optimizer: tuple(100 * float(np.mean([float(r[key]) for r in legal[optimizer]]))
                              for key in ("test_concept_accuracy", "test_count_accuracy")) for optimizer in ("sgd", "adam")}
    means["reference"] = tuple(100 * float(np.mean([float(r[key]) for r in runs if r["optimizer"] == "adam"]))
                               for key in ("reference_test_concept_accuracy", "reference_test_count_accuracy"))
    assert {k: tuple(f"{v:.2f}" for v in values) for k, values in means.items()} == {
        "sgd": ("84.72", "57.59"), "adam": ("87.46", "63.94"), "reference": ("94.93", "81.92")}
    points = {optimizer: np.array([[100 * int(r["mislabeled_training_images"]) / 1024,
                                    100 * (float(r["reference_test_concept_accuracy"]) - float(r["test_concept_accuracy"]))]
                                   for r in legal[optimizer]]) for optimizer in ("sgd", "adam")}
    pooled = np.concatenate(list(points.values()))
    assert len(pooled) == 60 and np.all(pooled > 0)
    correlation = float(np.corrcoef(pooled[:, 0], pooled[:, 1])[0, 1])
    slope, intercept = (float(v) for v in np.polyfit(pooled[:, 0], pooled[:, 1], 1))
    assert (f"{correlation:.2f}", f"{slope:.2f}") == ("0.97", "1.01")
    return {"distances": distances, "means": means, "points": points, "fit": (slope, intercept), "correlation": correlation}


def plot(runs_path: Path, pairs_path: Path, output: Path) -> None:
    data = load(runs_path, pairs_path)
    style.set_paper_style(LABEL_SIZE)
    fig = plt.figure(figsize=(style.TEXTWIDTH_IN, HEIGHT))
    disagreement, accuracy, cost = (panel(fig, index) for index in range(3))

    # (a) One bar per initial network. A third colour: a bar describes a pair of models, not one optimizer.
    distances = data["distances"]
    disagreement.bar(np.arange(1, 29), distances, width=0.62, color=style.LIGHT_GOLD, edgecolor=style.GOLD, linewidth=0.4)
    disagreement.axhline(90, color=style.MUTED_INK, linestyle=(0, (4, 2.5)), linewidth=0.6)
    disagreement.text(0.9, 97, "Median: 90", fontsize=NOTE_SIZE, color=style.MUTED_INK, ha="left", va="bottom")
    disagreement.set_xlim(0.1, 28.9)
    disagreement.set_xticks([1, 10, 20, 28])
    disagreement.set_ylim(0, 215)
    disagreement.set_yticks([0, 50, 100, 150, 200])
    disagreement.set_ylabel("Differing labels")

    # (b) Two groups of three narrow bars; the legend sits below the axes, so the bars use the full height.
    means = data["means"]
    positions = {"sgd": (0.0, 3.8), "adam": (1.0, 4.8), "reference": (2.0, 5.8)}
    value_labels, handles = [], []
    for name, (edge, face), label in (("sgd", SGD, "SGD"), ("adam", ADAM, "Adam"), ("reference", REFERENCE, "Reference")):
        accuracy.bar(positions[name], means[name], width=0.62, color=face, edgecolor=edge, linewidth=0.5)
        handles.append(Patch(facecolor=face, edgecolor=edge, linewidth=0.5, label=label))
        for x, value in zip(positions[name], means[name]):
            value_labels.append(accuracy.text(x, value + 1.6, f"{value:.1f}", fontsize=NOTE_SIZE, ha="center", va="bottom"))
    accuracy.set_xlim(-0.75, 6.55)
    accuracy.set_xticks([1.0, 4.8], labels=["Concepts", "Counts"])
    accuracy.set_ylim(0, 110)
    accuracy.set_yticks([0, 25, 50, 75, 100])
    accuracy.set_ylabel("Test accuracy (%)")
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(center(1), XLABEL_Y / HEIGHT), ncol=3, fontsize=LEGEND_SIZE,
               handlelength=0.9, handleheight=0.75, handletextpad=0.35, columnspacing=1.0, borderpad=0, borderaxespad=0)

    # (c) One marker per model, with the least-squares line over both optimizers.
    slope, intercept = data["fit"]
    span = np.array([0.0, 20.5])
    cost.plot(span, slope * span + intercept, color=style.MUTED_INK, linestyle=(0, (4, 2.5)), linewidth=0.6, zorder=1)
    for name, (edge, face), marker, size, label in (("sgd", SGD, "o", 2.7, "SGD"), ("adam", ADAM, "D", 2.3, "Adam")):
        xy = data["points"][name]
        cost.plot(xy[:, 0], xy[:, 1], linestyle="none", marker=marker, markersize=size, markerfacecolor=to_rgba(face, 0.85),
                  markeredgecolor=edge, markeredgewidth=0.5, label=label, zorder=2)
    cost.text(0.95, 0.07, rf"$r={data['correlation']:.2f}$", transform=cost.transAxes, fontsize=LABEL_SIZE, ha="right", va="bottom")
    cost.set_xlim(0, 20.5)
    cost.set_xticks([0, 5, 10, 15, 20])
    cost.set_ylim(0, 21.5)
    cost.set_yticks([0, 5, 10, 15, 20])
    cost.set_ylabel("Accuracy gap (points)")
    cost.legend(loc="upper left", fontsize=LEGEND_SIZE, handlelength=1.0, handletextpad=0.25, borderaxespad=0.35, labelspacing=0.3)

    for ax in (disagreement, accuracy, cost):
        style.clean_axes(ax)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.yaxis.label.set_fontsize(LABEL_SIZE)
        ax.yaxis.labelpad = 2.0
    accuracy.tick_params(axis="x", labelsize=LABEL_SIZE, length=0, pad=3.5)

    labels = (("Initial network (sorted)", "a", "Label disagreement"),
              (None, "b", "Accuracy on unseen data"),
              ("Mislabeled training images (%)", "c", "Concept error vs. test gap"))
    for index, (xlabel, letter, title) in enumerate(labels):
        if xlabel:
            fig.text(center(index), XLABEL_Y / HEIGHT, xlabel, fontsize=LABEL_SIZE, ha="center", va="center")
        fig.text(center(index), PANEL_LABEL_Y / HEIGHT, f"({letter}) {title}", fontsize=PANEL_LABEL_SIZE, ha="center", va="top")

    check_layout(fig, value_labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    style.save_all_formats(fig, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True, help="per-run results with the reference accuracies")
    parser.add_argument("--pairs", type=Path, required=True, help="per-pair label disagreement of the same runs")
    parser.add_argument("--output", type=Path, required=True, help="output path without suffix")
    args = parser.parse_args()
    plot(args.runs, args.pairs, args.output)
