"""Shared plotting style and paths for Activity 3."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
EXERCISE = HERE.parents[1]
ROOT = next(path for path in HERE.parents if (path / "mkdocs.yml").exists())
FIGDIR = EXERCISE / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

INK = "#17202a"
SOFT = "#5d6d7e"
GRID = "#d5d8dc"
COLORS = ["#2166ac", "#b2182b", "#4d9221"]
MARKERS = ["o", "s", "^"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.dpi": 140,
    "savefig.dpi": 160,
    # Every figure calls tight_layout(); a fixed canvas avoids one-pixel
    # bbox rounding differences between a Jupyter kernel and a clean CLI run.
    "savefig.bbox": None,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.alpha": 0.45,
    "grid.linewidth": 0.6,
})


def save(fig, filename):
    path = FIGDIR / filename
    fig.savefig(path)
    plt.close(fig)
    print(f"  figure -> {path.relative_to(ROOT)}")


def label_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
