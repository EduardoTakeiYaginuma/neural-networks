"""
Shared plotting style for the report.

A plain academic look: white paper, near-black ink, a recessive hairline frame,
a serif face for prose and a monospaced one for every measured number. Only
fonts that ship with matplotlib are used, so the figures render identically on
a clean checkout with nothing to install and nothing vendored into the repo.

The categorical palette is validated rather than asserted. Every figure here is
a scatter or an overlaid histogram, so *all six* pairs of class colours end up
side by side, not just adjacent ones — the floor has to hold pairwise across
normal vision and the three dichromacies at once. Measured with CIEDE2000 on
the Vienot-Brettel-Mollon simulations, against a white background:

    worst pair, normal vision   dE 22.7      worst pair, protanopia   dE 20.7
    worst pair, deuteranopia    dE 23.6      worst pair, tritanopia   dE 20.6

so the palette clears a floor of dE 20 in every vision type. WCAG non-text
contrast against the page is 5.9 / 6.4 / 7.5 / 3.1, all above the 3.0 floor.
Marker shape is carried as a redundant channel on top of that, so class
identity never depends on colour alone.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # headless: figures are written to PNG, never shown
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# This file lives at docs/exercises/data/code/, so the exercise folder is two
# levels up and the repository root is the nearest ancestor holding mkdocs.yml.
HERE     = Path(__file__).resolve()
EXERCISE = HERE.parents[1]                      # docs/exercises/data
ROOT     = next(p for p in HERE.parents if (p / "mkdocs.yml").exists())
FIGDIR   = EXERCISE / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------- neutrals
INK        = "#1a1a1a"   # primary text
INK_SOFT   = "#4a4a4a"   # secondary text
HAIRLINE   = "#c8c8c8"   # grid lines, spines
SURFACE    = "#ffffff"   # figure/axes background
ACCENT     = "#1a1a1a"   # annotation strokes (sketched boundaries, reference lines);
                         # near-black and always dashed, so an annotation never
                         # reads as one more data series

# Categorical palette (fixed order, never cycled) + redundant marker shapes.
# See the module docstring for the measured CVD and contrast figures.
SERIES  = ["#2166AC",    # class 0 - blue
           "#8C510A",    # class 1 - ochre
           "#555555",    # class 2 - graphite
           "#CC79A7"]    # class 3 - mauve
MARKERS = ["o", "s", "^", "D"]

# ------------------------------------------------------------------------ fonts
# matplotlib ships all three, so there is nothing to download or vendor.
SERIF = "DejaVu Serif"
SANS  = "DejaVu Sans"
MONO  = "DejaVu Sans Mono"

plt.rcParams.update({
    "figure.facecolor":  SURFACE,
    "axes.facecolor":    SURFACE,
    "savefig.facecolor": SURFACE,
    "figure.dpi":        140,
    "savefig.dpi":       140,
    "savefig.bbox":      "tight",

    "font.family":       SERIF,
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
    """Serif title, with an optional deck line underneath.

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
