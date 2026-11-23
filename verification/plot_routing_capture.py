"""Render the phase portrait in Figure 1."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
import plot_style as style
Array = NDArray[np.float64]
LIMIT = 3.2


def load_trajectories(path: Path) -> list[Array]:
    matches = re.findall(
        r"\\draw\[black,very thick([^\]]*)\] plot coordinates \{([^}]+)\};",
        path.read_text(encoding="utf-8"),
    )
    if [style for style, _ in matches] != ["", ",dotted", ",dashdotted"]:
        raise ValueError(f"Expected the three original trajectory styles in {path}")
    paths: list[Array] = []
    for line_style, coordinates in matches:
        points = np.array(re.findall(r"\((-?[\d.]+),(-?[\d.]+)\)", coordinates), dtype=float)
        if not np.isfinite(points).all() or not np.array_equal(points[0], [-0.8, -0.8]):
            raise ValueError(f"Invalid saved trajectory {line_style!r} in {path}")
        clipped = np.flatnonzero(np.max(np.abs(points), axis=1) >= LIMIT)
        if not len(clipped) or clipped[0] < 2:
            raise ValueError(f"Missing expected clip boundary for trajectory {line_style!r}")
        paths.append(points[:clipped[0]])
    return paths


def legal_evidence(u: Array, v: Array) -> Array:
    a, b = np.exp(-u), np.exp(-v)
    return (1.0 + a * a + b * b) / ((1.0 + a) ** 2 * (1.0 + b) ** 2)


def check_inputs(paths: list[Array]) -> None:
    """Check saved-point truncation and evidence against three code products."""
    assert [len(path) for path in paths] == [542, 320, 320]
    assert all(np.max(np.abs(path)) < LIMIT for path in paths)
    np.testing.assert_array_equal(paths[0][:, 0], paths[0][:, 1])
    np.testing.assert_array_equal(paths[1], paths[2][:, ::-1])
    u = np.array([0.0, -0.8, -3.0, 3.0, 0.0, 0.0])
    v = np.array([0.0, -0.8, 3.0, -3.0, -3.0, 3.0])
    margins = np.column_stack((u, u, v, v))
    codes = np.array([[1, 1, 1, 1], [-1, -1, 1, 1], [1, 1, -1, -1]])
    products = (1.0 / (1.0 + np.exp(-margins[:, None, :] * codes))).prod(axis=2)
    np.testing.assert_allclose(legal_evidence(u, v), products.sum(axis=1), rtol=1e-13)
    assert legal_evidence(u, v)[0] == 3.0 / 16.0


def extension_guides(axis: Axes, x: float, ys: tuple[float, ...] = ()) -> None:
    """Draw dashed guides from a marked point to the coordinate axes."""
    guide = {"color": style.GUIDE_GRAY, "linewidth": 0.5,
             "linestyle": (0, (1.0, 1.6)), "zorder": 0.5}
    axis.axvline(x, **guide)
    for y in ys:
        axis.axhline(y, **guide)


def phase_portrait_panel(axis: Axes, paths: list[Array]) -> None:
    """Panel (a): saved m=2 routes and the sufficient capture region."""
    grid = np.linspace(-LIMIT, LIMIT, 801)
    u, v = np.meshgrid(grid, grid)
    evidence = legal_evidence(u, v)
    axis.contourf(u, v, evidence, levels=[0.5, 1.0], colors=[style.LIGHT_GRAY],
                  alpha=0.4, zorder=0)
    axis.contour(u, v, evidence, levels=[0.5], colors=[style.GRAY],
                 linewidths=0.55, linestyles="dotted", zorder=1)
    axis.axhline(0.0, color=style.LIGHT_GRAY, linewidth=0.5, zorder=1)
    axis.axvline(0.0, color=style.LIGHT_GRAY, linewidth=0.5, zorder=1)
    line_colors = (style.BLUE, style.ROSE, style.GOLD)
    for index, (path, color, dash) in enumerate(zip(paths, line_colors, ("-", "--", "-."))):
        axis.plot(path[:, 0], path[:, 1], color=color, linestyle=dash,
                  linewidth=1.0, label=rf"$H_{index}$", zorder=3)
        distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        arrow = np.searchsorted(distance, 0.60 * distance[-1])
        axis.annotate("", xy=path[arrow + 4], xytext=path[arrow - 4],
                      arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.7,
                                  "mutation_scale": 6}, zorder=4)
    axis.plot(-0.8, -0.8, "o", color=style.INK, markersize=2.6, zorder=5)
    # Dashed fine-dot guides extending the start out to both frames, so its
    # two coordinates can be read off the axes directly.
    extension_guides(axis, -0.8, (-0.8,))
    axis.annotate("start", (-0.8, -0.8), xytext=(-2.9, -1.75), va="center",
                  arrowprops={"arrowstyle": "-", "color": style.GRAY, "lw": 0.5},
                  fontsize=8)
    for label, code, (x, y), code_y in (
            ("$z_0$", "$({+},{+},{+},{+})$", (1.85, 2.35), 2.9),
            ("$z_1$", "$({-},{-},{+},{+})$", (-2.2, 2.35), 2.9),
            ("$z_2$", "$({+},{+},{-},{-})$", (2.25, -1.78), -2.23)):
        axis.text(x, y, label, fontsize=8.5, ha="center", va="center")
        axis.text(x, code_y, code, fontsize=5.5, ha="center", va="center")
    axis.set(xlim=(-LIMIT, LIMIT), ylim=(-LIMIT, LIMIT))
    axis.text(0.97, 0.53, r"$u$", transform=axis.transAxes, fontsize=9,
              ha="right", va="bottom")
    axis.text(0.53, 0.97, r"$v$", transform=axis.transAxes, fontsize=9,
              ha="left", va="top")
    axis.set_xticks([-3, 0, 3])
    axis.set_yticks([-3, 0, 3])
    axis.tick_params(labelsize=8)
    style.clean_axes(axis)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=3,
                frameon=False, fontsize=8, handlelength=1.2,
                columnspacing=0.6, handletextpad=0.35, borderaxespad=0.2)
