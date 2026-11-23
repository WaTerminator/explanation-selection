"""Render Figure 2 from the supplied complete plotting inputs."""
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fontTools.pens.recordingPen import RecordingPen
from fontTools.svgLib.path import parse_path
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, PathPatch, Rectangle
from matplotlib.path import Path as VectorPath
from matplotlib.transforms import Affine2D
from numpy.typing import NDArray

import plot_style as style


ROOT = Path(__file__).resolve().parents[1]
BLUE, ROSE, INK, GRAY = style.BLUE, style.ROSE, style.INK, style.MUTED_INK
PAIR = ("feature", "feature_reverse")
COLORS = (BLUE, ROSE)
ICON_SOURCE = ROOT / "assets/truth_icons.svg"

# Panel (a) coordinates are points, including the space around its truth icons.
COHORT_HEIGHT_IN = 2.35
ASSIGNMENT_PLATE_H_IN = 2.10
PANEL_AXES_H_IN = 0.88
COLUMN_LEFT_IN, COLUMN_RIGHT_IN = 0.10, 0.06
COLUMN_GUTTER_IN, COLUMN_GAP_IN = 0.36, 0.20


def example_bands() -> dict[str, float]:
    """The main figure's band heights, in inches, from top to bottom."""
    return {"plate_h": ASSIGNMENT_PLATE_H_IN, "plate_label_h": 0.17,
            "panel_axes_h": PANEL_AXES_H_IN, "panel_label_h": 0.19,
            "label_gap_h": 0.40}


def example_height_in() -> float:
    """Height of the main example figure, from its own band budget."""
    band = example_bands()
    return (band["plate_h"] + band["plate_label_h"] + band["panel_axes_h"]
            + band["panel_label_h"] + band["label_gap_h"])


EXAMPLE_HEIGHT_IN = example_height_in()


@dataclass(frozen=True)
class Endpoint:
    draw: int
    head: int
    metric: str
    resolved: bool
    code: tuple[int, ...]
    capture: int | None
    fitted: int | None
    train_accuracy: float
    test_accuracy: float


@dataclass(frozen=True)
class Example:
    endpoints: tuple[Endpoint, Endpoint]
    images: NDArray[np.uint8]
    indices: NDArray[np.int64]
    truth: NDArray[np.int64]
    counts: NDArray[np.int64]
    evidence: NDArray[np.float64]
    max_violations: NDArray[np.int64]
    continuation_horizon: int
    preferred_selected: bool
    hamming_distances: NDArray[np.int64]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_endpoints(path: Path) -> list[Endpoint]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [Endpoint(
        draw=int(row["draw"]), head=int(row["head"]), metric=row["metric"],
        resolved=row["resolved"] == "True", code=tuple(json.loads(row["final_code"])),
        capture=int(row["T_capture"]) if row["T_capture"] else None,
        fitted=int(row["T_fit"]) if row["T_fit"] else None,
        train_accuracy=float(row["train_accuracy"]), test_accuracy=float(row["test_accuracy"]),
    ) for row in rows]


def select_pair(rows: list[Endpoint]) -> tuple[Endpoint, Endpoint]:
    """Use only the prescribed ordering, resolution, and code difference."""
    lookup = {(row.draw, row.head, row.metric): row for row in rows}
    keys = sorted({(row.draw, row.head) for row in rows})
    if (3, 0) in keys:
        keys.remove((3, 0))
        keys.insert(0, (3, 0))
    for draw, head in keys:
        a, b = (lookup[draw, head, metric] for metric in PAIR)
        if a.resolved and b.resolved and a.code != b.code:
            return a, b
    raise ValueError("No eligible resolved, different feature/reverse pair exists")


def read_hamming_distances(run_dir: Path) -> NDArray[np.int64]:
    with (run_dir / "evaluation/pairs.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 144:
        raise ValueError(f"Expected 144 within-group pairs, found {len(rows)}")
    return np.array([int(row["final_hamming_distance"]) for row in rows], dtype=np.int64)




def load_truth_icons() -> dict[str, VectorPath]:
    """Convert the approved Lucide paths to native Matplotlib vector objects."""
    source = ET.parse(ICON_SOURCE).getroot()
    ns = {"s": "http://www.w3.org/2000/svg"}
    paths: dict[str, VectorPath] = {}
    operations = {"moveTo": VectorPath.MOVETO, "lineTo": VectorPath.LINETO,
                  "curveTo": VectorPath.CURVE4}
    for name in ("cat", "bird"):
        group = source.find(f"s:g[@id='{name}']", ns)
        if group is None:
            raise ValueError(f"Missing {name} icon in {ICON_SOURCE}")
        pen = RecordingPen()
        for part in group.findall("s:path", ns):
            parse_path(part.attrib["d"], pen)
        vertices: list[tuple[float, float]] = []
        codes: list[int] = []
        for operation, points in pen.value:
            if operation == "closePath":
                vertices.append((0.0, 0.0))
                codes.append(VectorPath.CLOSEPOLY)
            elif operation == "endPath":
                continue
            elif operation in operations:
                vertices.extend(points)
                codes.extend([operations[operation]] * len(points))
            else:
                raise ValueError(f"Unsupported {operation} in {name} icon")
        paths[name] = VectorPath(vertices, codes)
    return paths


def annotate_icon_svg(path: Path) -> None:
    """Retain icon identities and license in the editable native SVG export."""
    tree = ET.parse(path)
    ns = {"s": "http://www.w3.org/2000/svg"}
    ET.register_namespace("", ns["s"])
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    for element in tree.getroot().iter():
        if element.get("id", "").startswith("label-T-"):
            element.set("data-icon", "lucide-" + element.attrib["id"].rsplit("-", 1)[1])
    description = ET.parse(ICON_SOURCE).getroot().find("s:desc", ns)
    if description is None:
        raise ValueError(f"Missing icon provenance in {ICON_SOURCE}")
    tree.getroot().insert(1, deepcopy(description))
    tree.write(path, encoding="utf-8", xml_declaration=True)


def image_plate(fig: Figure, example: Example, plate_bottom: float,
                plate_top: float) -> tuple[float, float]:
    """Draw the assignments and return the photo body's horizontal bounds in points."""
    width = fig.get_figwidth() * 72
    height = (plate_top - plate_bottom) * fig.get_figheight() * 72
    ax = fig.add_axes((0, plate_bottom, 1, plate_top - plate_bottom))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    ax.set_gid("assignment-plate")

    truth = np.asarray(example.truth).reshape(6, 4)
    codes = np.asarray([row.code for row in example.endpoints]).reshape(2, 6, 4)
    if not np.array_equal(truth.sum(axis=1), example.counts):
        raise ValueError("The displayed truth does not satisfy the six bag counts")
    if not np.all(codes.sum(axis=2) == example.counts):
        raise ValueError("The displayed learned codes do not satisfy the six bag counts")
    disagreements = codes[0] != codes[1]
    distance = int(disagreements.sum())
    icons = load_truth_icons()
    table_width, bag_gap = width - 57.3, 10.0
    # Center the bag body on the lower axes, excluding the row labels.
    table_center = (width + 72 * (COLUMN_LEFT_IN + COLUMN_GUTTER_IN - COLUMN_RIGHT_IN)) / 2
    table_left, table_right = table_center - table_width / 2, table_center + table_width / 2
    legend_y = height - 4.6
    for x, filled, label in ((table_left + 3, True, "Cat"),
                             (table_left + 33, False, "Bird")):
        ax.add_patch(Circle((x, legend_y), 2.0, facecolor=INK if filled else "white",
                            edgecolor=INK, linewidth=0.65))
        ax.text(x + 5.0, legend_y, label, va="center", fontsize=6.8)
    ax.text(table_right, legend_y, rf"Shaded columns: $A$ and $B$ differ ({distance}/24)",
            ha="right", va="center", fontsize=6.8, color=INK)

    bag_width = (table_right - table_left - 2 * bag_gap) / 3
    pitch, photo_size = bag_width / 4, 22.0
    row_tops = (height - 13.0, height - 85.0)
    label_rows = ((39.0, "T", "Truth", GRAY), (50.0, "A", "$A$: feature", BLUE),
                  (59.0, "B", "$B$: reverse", ROSE))
    for row_top in row_tops:
        for offset, key, label, color in label_rows:
            ax.text(table_left - 7, row_top - offset, label,
                    ha="right", va="center", fontsize=6.6, color=color)

    for bag in range(6):
        row, column = divmod(bag, 3)
        top = row_tops[row]
        left = table_left + column * (bag_width + bag_gap)
        ax.text(left + 1, top - 3.5, f"Bag {bag + 1}",
                fontsize=7.0, fontweight="bold", va="center")
        count = int(example.counts[bag])
        ax.text(left + bag_width - 1, top - 3.5,
                f"{count} cat{'s' if count != 1 else ''} / 4", fontsize=6.6,
                ha="right", va="center", color=GRAY)
        for local in range(4):
            center = left + (local + 0.5) * pitch
            index = 4 * bag + local
            photo = ax.imshow(example.images[index],
                              extent=(center - photo_size / 2, center + photo_size / 2,
                                      top - 32, top - 10),
                              interpolation="none", aspect="equal", zorder=2)
            photo.set_gid(f"photo-{index}-{example.indices[index]}")
            if disagreements[bag, local]:
                shade = Rectangle((center - pitch / 2 + 0.6, top - 63.0),
                                  pitch - 1.2, 18.0, facecolor=style.LIGHT_GRAY,
                                  edgecolor="none", alpha=0.35, zorder=1)
                shade.set_gid(f"disagreement-{index}")
                ax.add_patch(shade)
            for (offset, key, label, color), values, fill in zip(
                    label_rows, (truth, codes[0], codes[1]),
                    (style.GRAY, style.LIGHT_BLUE, style.LIGHT_ROSE)):
                is_cat = bool(values[bag, local])
                category = "cat" if is_cat else "bird"
                if key == "T":
                    transform = (Affine2D().translate(-12, -12).scale(1 / 3, -1 / 3)
                                 .translate(center, top - offset))
                    marker = PathPatch(icons[category], transform=transform + ax.transData,
                                       facecolor="none", edgecolor=color, linewidth=2 / 3,
                                       capstyle="round", joinstyle="round", zorder=3)
                else:
                    marker = Circle((center, top - offset), 2.05,
                                    facecolor=fill if is_cat else "white",
                                    edgecolor=color, linewidth=0.65, zorder=3)
                marker.set_gid(f"label-{key}-{index}-{category}")
                ax.add_patch(marker)

    photo_inset = (pitch - photo_size) / 2
    return table_left + photo_inset, table_right - photo_inset


def evidence_panel(fig: Figure, ax_box: tuple[float, float, float, float], example: Example) -> None:
    ax = fig.add_axes(ax_box)
    steps = np.arange(len(example.evidence))
    for j, (row, color, dash) in enumerate(zip(example.endpoints, COLORS, ("-", "--"))):
        name = "$A$: feature" if j == 0 else "$B$: reverse"
        ax.plot(steps, example.evidence[:, j], color=color, linestyle=dash, lw=0.9, label=name)
        capture = row.capture
        if capture is None:
            raise ValueError("Missing capture time")
        ax.scatter(capture, example.evidence[capture, j], marker="o" if j == 0 else "s",
                   s=14, facecolor="white", edgecolor=color, linewidth=0.7, zorder=5)
        ax.annotate(f"{capture}", (capture, example.evidence[capture, j]),
                    xytext=(-20, 13) if j == 0 else (8, -20), textcoords="offset points",
                    color=color, fontsize=6.5,
                    arrowprops={"arrowstyle": "-", "color": color, "lw": .5})
    ax.axhline(.5, color=GRAY, lw=.6, linestyle=(0, (3, 3)), zorder=0)
    ax.text(1.4, .545, "Sufficient threshold", fontsize=6.5, color=GRAY)
    ax.set_xscale("symlog", linthresh=100, linscale=.55)
    ax.set_xlim(0, steps[-1])
    ax.set_xticks([0, 100, 1000, 8000], ["0", "100", "1,000", "8,000"])
    ax.set_ylim(0, 1.015)
    ax.set_yticks([0, .5, 1], ["0", "0.5", "1"])
    ax.set_xlabel("Update step", fontsize=6.8, labelpad=1.5)
    ax.set_ylabel(r"$\mathcal{E}_{\rm all}$", fontsize=7, labelpad=1.5)
    ax.tick_params(labelsize=6.5, length=2.0)
    style.clean_axes(ax)
    ax.set_ylim(-0.05, 1.10)
    ax.legend(frameon=False, fontsize=6.2, loc="lower left", handlelength=1.3,
              handletextpad=0.4, borderaxespad=0.15, labelspacing=0.25)


def hamming_distribution_panel(fig: Figure, ax_box: tuple[float, float, float, float], example: Example) -> None:
    """Plot the 144 within-group Hamming distances and mark their median."""
    ax = fig.add_axes(ax_box)
    counts = np.bincount(example.hamming_distances, minlength=25)
    values = np.arange(len(counts))
    mask = counts > 0
    ax.bar(values[mask], counts[mask], color=style.LIGHT_BLUE, edgecolor=BLUE, linewidth=.45,
           width=.66, zorder=2, label="Pairs of runs")
    illustrated = sum(x != y for x, y in zip(example.endpoints[0].code, example.endpoints[1].code))
    ax.bar([illustrated], [counts[illustrated]], color=style.LIGHT_ROSE, edgecolor=ROSE,
           linewidth=.45, width=.66, zorder=3, label="Illustrated pair")
    median = int(np.median(example.hamming_distances))
    occupied = values[counts > 0]
    ax.set_ylim(0, counts.max() * 1.38)
    ax.set_xlim(occupied.min() - 0.85, occupied.max() + 0.85)
    ax.set_xticks(range(occupied.min(), occupied.max() + 1, 2))
    ax.set_yticks([0, 20, 40])
    ax.set_xlabel("Differing labels (out of 24)", fontsize=6.8, labelpad=1.5)
    ax.set_ylabel("Number of pairs", fontsize=7, labelpad=1.5)
    ax.tick_params(labelsize=6.5, length=2.0)
    style.clean_axes(ax)
    ax.annotate(f"{counts[illustrated]}", (illustrated, counts[illustrated]),
                xytext=(0, 2.5), textcoords="offset points", ha="center", va="bottom",
                fontsize=6.0, color=ROSE)
    # Bins 2, 4, 6 and 8 are at most 33 of a 73 ceiling, so the upper-left
    # corner is the one place a legend clears the data. The median marker gets
    # its own entry: without one, the triangle on the axis reads as unrelated
    # to the two series it sits between.
    handles, _ = ax.get_legend_handles_labels()
    median_handle = Line2D([], [], linestyle="none", marker="v", markersize=3.0,
                           color=INK, label=f"Median: {median}/24")
    ax.legend(handles=handles[:2] + [median_handle], frameon=False, fontsize=5.6,
              loc="upper left", bbox_to_anchor=(0.02, 0.99), borderaxespad=0.0,
              handlelength=1.1, handletextpad=0.28, labelspacing=0.18,
              borderpad=0.0)
    ax.plot([median], [0], marker="v", markersize=3.0, color=INK,
            clip_on=False, zorder=6)
    sample_size = ax.text(0.965, 0.96, f"n = {len(example.hamming_distances)}",
                          transform=ax.transAxes, fontsize=6.0, ha="right",
                          va="top", color=GRAY)
    sample_size.set_gid("hamming-sample-size")


def panel_label(fig: Figure, center_x: float, y: float, label: str, title: str) -> None:
    """Centre "(x) Title" beneath a panel's axes, the venue convention.

    Panel letters used to sit in each axes' top-left corner. Published
    multi-panel figures in this field centre them under the panel instead.
    """
    fig.text(center_x, y, f"({label}) {title}", fontsize=7.5, ha="center",
             va="top", color=INK)


def save_figure(fig: Figure, stem: Path) -> None:
    style.save_all_formats(fig, stem)


def render_example(example: Example, output: Path) -> None:
    """Place the assignment matrix above the two quantitative panels."""
    # Band heights, in inches, measured from the top of the figure downward.
    band = example_bands()
    plate_h = band["plate_h"]
    plate_label_h = band["plate_label_h"]
    panel_axes_h = band["panel_axes_h"]
    panel_label_h = band["panel_label_h"]
    label_gap_h = band["label_gap_h"]

    fig_h = (plate_h + plate_label_h + panel_axes_h
             + panel_label_h + label_gap_h)
    fig = plt.figure(figsize=(style.TEXTWIDTH_IN, fig_h))

    # Walk the bands down from the top edge.
    plate_top_in = fig_h
    plate_bottom_in = plate_top_in - plate_h
    plate_label_top_in = plate_bottom_in - 0.10
    panel_top_in = plate_label_top_in - plate_label_h
    panel_bottom_in = panel_top_in - panel_axes_h
    panel_label_top_in = panel_bottom_in - 0.30      # clears tick labels + xlabel

    body_left, body_right = image_plate(
        fig, example, plate_bottom=plate_bottom_in / fig_h,
        plate_top=plate_top_in / fig_h)
    left_in, right_in = body_left / 72, body_right / 72
    gap_in = COLUMN_GUTTER_IN + COLUMN_GAP_IN
    cell_w = (right_in - left_in - gap_in) / 2

    def column_cell(index: int) -> tuple[float, float]:
        """(left_in, width_in) of the *axes* cell in column `index`."""
        x = left_in + index * (cell_w + gap_in)
        return x, cell_w

    def column_center(index: int) -> float:
        """Centre of column `index`, for label placement."""
        x, width_in = column_cell(index)
        return (x + width_in / 2) / style.TEXTWIDTH_IN

    def column_box(index: int, bottom_in: float, height_in: float
                   ) -> tuple[float, float, float, float]:
        x_in, width_in = column_cell(index)
        return (x_in / style.TEXTWIDTH_IN, bottom_in / fig_h,
                width_in / style.TEXTWIDTH_IN, height_in / fig_h)

    grid_left_in = COLUMN_LEFT_IN
    grid_right_in = style.TEXTWIDTH_IN - COLUMN_RIGHT_IN
    fig.text(0.5 * (grid_left_in + grid_right_in) / style.TEXTWIDTH_IN,
             plate_label_top_in / fig_h,
             "(a) Same six bag counts, different complete codes",
             fontsize=7.5, ha="center", va="top", color=INK)

    # Band 2: two aligned quantitative panels on one shared label baseline.
    for index, (draw, label, title) in enumerate((
            (lambda box: hamming_distribution_panel(fig, box, example), "b", "Hamming distance"),
            (lambda box: evidence_panel(fig, box, example), "c", "Global legal evidence"),
    )):
        draw(column_box(index, panel_bottom_in, panel_axes_h))
        fig.text(column_center(index), panel_label_top_in / fig_h,
                 f"({label}) {title}", fontsize=7.5, ha="center", va="top", color=INK)

    save_figure(fig, output / "figure2")
    annotate_icon_svg(output / "figure2.svg")
