"""Shared, self-contained plotting style for the perceptron report."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

HERE = Path(__file__).resolve()
EXERCISE = HERE.parents[1]
ROOT = next(p for p in HERE.parents if (p / "mkdocs.yml").exists())
FIGDIR = EXERCISE / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

INK = "#1a1a1a"
INK_SOFT = "#4a4a4a"
HAIRLINE = "#c8c8c8"
SURFACE = "#ffffff"
ACCENT = "#1a1a1a"

# Colourblind-safe palette plus redundant marker shapes, copied from Activity 1.
SERIES = ["#2166AC", "#8C510A", "#555555", "#CC79A7"]
MARKERS = ["o", "s", "^", "D"]
SERIF = "DejaVu Serif"
SANS = "DejaVu Sans"
MONO = "DejaVu Sans Mono"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "savefig.bbox": "tight",
    "font.family": SERIF,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "text.color": INK,
    "axes.labelcolor": INK_SOFT,
    "axes.titlecolor": INK,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "axes.edgecolor": HAIRLINE,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": HAIRLINE,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.45,
    "axes.axisbelow": True,
    "legend.frameon": True,
    "legend.facecolor": SURFACE,
    "legend.edgecolor": HAIRLINE,
    "legend.borderpad": 0.5,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
})


def title(ax, text, sub=None):
    """Draw a title and optional subtitle without collisions."""

    ax.text(0.0, 1.085 if sub else 1.02, text, transform=ax.transAxes,
            fontfamily=SERIF, fontsize=12, color=INK, va="bottom", ha="left")
    if sub:
        ax.text(0.0, 1.018, sub, transform=ax.transAxes, fontsize=8,
                color=INK_SOFT, va="bottom", ha="left")


def legend_row(ax, ncols=4, y=-0.16):
    """Place a horizontal legend below the plotting area."""

    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncols=ncols,
                     frameon=False, handletextpad=0.4, columnspacing=1.4)


def halo(txt, lw=2.2):
    """Add a white outline to text drawn over data."""

    txt.set_path_effects([pe.withStroke(linewidth=lw, foreground=SURFACE)])
    return txt


def mono_ticks(ax):
    """Render numerical tick labels in the report's monospaced face."""

    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_fontfamily(MONO)


def save(fig, name):
    """Save a figure beside the report and close it."""

    output = FIGDIR / name
    fig.savefig(output)
    plt.close(fig)
    print(f"  figure -> {output.relative_to(ROOT)}")
    return output
