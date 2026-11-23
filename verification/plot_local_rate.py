"""Render the local convergence-rate traces."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GRAY = "#777777"


def plot_rate_panel(ax: plt.Axes, rate_rows: list[dict[str, str]], legend_size: float) -> None:
    rate_style = {
        "strict": (BLUE, "Strict, $d_{\\min}>\\tau$"),
        "critical_chi_0.8": (VERMILLION, r"Critical, $\chi_z=0.8$"),
    }
    for case, (color, label) in rate_style.items():
        selected = [row for row in rate_rows if row["case"] == case]
        time = np.array([float(row["step"]) for row in selected])
        radius = np.array([float(row["radius"]) for row in selected])
        tail = time >= time.max() / 2.0
        slope = np.polyfit(np.log(time[tail]), np.log(radius[tail]), 1)[0]
        ax.loglog(time, radius, color=color, linewidth=1.25,
                  label=f"{label}, slope {slope:.2f}")
    guide_time = np.array([2e3, 2e5])
    ax.loglog(guide_time, 40.0 / guide_time, color=GRAY, linestyle="--",
              linewidth=0.9, label=r"$t^{-1}$ guide")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Compactified radius $r_t$")
    ax.set_title("Asymptotic rate")
    ax.legend(frameon=False, fontsize=legend_size, loc="lower left")
    ax.grid(which="both", color="#E5E5E5", linewidth=0.4)
