"""
Shared plotting style for the report.

Two jobs:
  1. Register the report's fonts (Spectral for titles, Archivo for UI text,
     IBM Plex Mono for every measured number) and set global matplotlib rcParams.
  2. Expose the categorical palette used for class colours. The palette was
     validated for colour-vision deficiency: on the report's light surface
     (#F4F1E8) the four hues clear the all-pairs CVD and normal-vision
     separation floors, which matters here because every figure is a scatter
     plot (all pairs of colours end up side by side, not just adjacent ones).
     Marker shape is used as a redundant encoding, so identity never depends
     on colour alone.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # headless: figures are written to PNG, never shown
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import font_manager as fm

# This file lives at docs/exercises/data/code/, so the exercise folder is two
# levels up and the repository root is the nearest ancestor holding mkdocs.yml.
HERE     = Path(__file__).resolve()
EXERCISE = HERE.parents[1]                      # docs/exercises/data
ROOT     = next(p for p in HERE.parents if (p / "mkdocs.yml").exists())
FIGDIR   = EXERCISE / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- brand colours
INK        = "#0A1729"   # primary text
INK_SOFT   = "#11294A"   # secondary text
HAIRLINE   = "#97A3B4"   # grid lines, spines
SURFACE    = "#F4F1E8"   # figure/axes background
ACCENT     = "#8A6B33"   # thin accent strokes (sketched boundaries, reference lines)

# Categorical palette (fixed order, never cycled) + redundant marker shapes.
SERIES  = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
MARKERS = ["o", "s", "^", "D"]

# ------------------------------------------------------------------------ fonts
for ttf in sorted((ROOT / "assets" / "fonts").glob("*.ttf")):
    fm.fontManager.addfont(str(ttf))

_installed = {f.name for f in fm.fontManager.ttflist}
SERIF = "Spectral" if "Spectral" in _installed else "DejaVu Serif"
SANS  = "Archivo"  if "Archivo"  in _installed else "DejaVu Sans"
MONO  = "IBM Plex Mono" if "IBM Plex Mono" in _installed else "DejaVu Sans Mono"

plt.rcParams.update({
    "figure.facecolor":  SURFACE,
    "axes.facecolor":    SURFACE,
    "savefig.facecolor": SURFACE,
    "figure.dpi":        140,
    "savefig.dpi":       140,
    "savefig.bbox":      "tight",

    "font.family":       SANS,
    "font.size":         9,
    "axes.titlesize":    11,
    "axes.labelsize":    9.5,
    "legend.fontsize":   8.5,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,

    "text.color":        INK,
    "axes.labelcolor":   INK_SOFT,
    "axes.titlecolor":   INK,
    "xtick.color":       INK_SOFT,
    "ytick.color":       INK_SOFT,

    # recessive frame: only left/bottom hairlines, thin grid behind the data
    "axes.edgecolor":    HAIRLINE,
    "axes.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        HAIRLINE,
    "grid.linewidth":    0.5,
    "grid.alpha":        0.45,
    "axes.axisbelow":    True,

    "legend.frameon":    True,
    "legend.facecolor":  SURFACE,
    "legend.edgecolor":  HAIRLINE,
    "legend.borderpad":  0.5,

    "lines.linewidth":   2.0,
    "lines.markersize":  5,
})


def title(ax, text, sub=None):
    """Editorial title in Spectral, with an optional deck line underneath.

    Placed with ax.text (not set_title) so title and subtitle never collide.
    """
    ax.text(0.0, 1.085 if sub else 1.02, text, transform=ax.transAxes,
            fontfamily=SERIF, fontsize=12, color=INK, va="bottom", ha="left")
    if sub:
        ax.text(0.0, 1.018, sub, transform=ax.transAxes, fontsize=8,
                color=INK_SOFT, va="bottom", ha="left")


def legend_row(ax, ncols=4, y=-0.16):
    """Legend as a single row under the plot, so it never covers the data."""
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncols=ncols,
                    frameon=False, handletextpad=0.4, columnspacing=1.4)
    return leg


def halo(txt, lw=2.2):
    """Surface-coloured outline behind a label sitting on top of the data."""
    txt.set_path_effects([pe.withStroke(linewidth=lw, foreground=SURFACE)])
    return txt


def mono_ticks(ax):
    """Tick labels are numbers, so they wear the mono face."""
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_fontfamily(MONO)


def save(fig, name):
    """Write the figure next to the report and return its repo-relative path."""
    out = FIGDIR / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  figure -> {out.relative_to(ROOT)}")
    return out
