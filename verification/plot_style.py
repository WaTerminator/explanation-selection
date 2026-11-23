"""Shared typography, colors, and export settings for the paper figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

BLUE = "#576FA0"
LIGHT_BLUE = "#A7B9D7"
GOLD = "#E3B87F"
LIGHT_GOLD = "#FADCB4"
ROSE = "#B57979"
LIGHT_ROSE = "#DEA3A2"
GRAY = "#9F9F9F"
LIGHT_GRAY = "#CFCFCF"
INK = "#222222"
MUTED_INK = "#666666"
SPINE_GRAY = GRAY
GUIDE_GRAY = LIGHT_GRAY

SANS = ["Arial", "Liberation Sans", "Nimbus Sans", "Helvetica", "DejaVu Sans"]
MATH_FONTSET = "cm"

TEXTWIDTH_IN = 5.5


def set_paper_style(base_size: float = 7) -> None:
    """Apply the shared rcParams. Call once before creating any figure."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": SANS,
        "mathtext.fontset": MATH_FONTSET,
        "font.size": base_size,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size,
        "ytick.labelsize": base_size,
        "legend.fontsize": base_size,
        "legend.frameon": False,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "axes.linewidth": 0.5,
        "axes.edgecolor": SPINE_GRAY,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.color": SPINE_GRAY,
        "ytick.color": SPINE_GRAY,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "grid.linewidth": 0.4,
        "grid.color": "#E8E8E8",
        "grid.linestyle": "-",
        "lines.linewidth": 0.9,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "image.composite_image": False,
        "text.color": INK,
        "axes.labelcolor": INK,
    })


def clean_axes(ax: Axes) -> None:
    """Apply the common frame and tick style."""
    for spine in ax.spines.values():
        spine.set_color(SPINE_GRAY)
        spine.set_linewidth(0.5)
    ax.tick_params(length=2.0, width=0.5, pad=1.8, colors=SPINE_GRAY)
    ax.tick_params(which="both", labelcolor=INK)


def save_all_formats(fig: Figure, stem: Path) -> None:
    """Save a fixed-size figure as PDF, SVG and PNG."""
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=300)
    plt.close(fig)
