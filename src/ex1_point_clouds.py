"""
Exercise 1 - Point clouds: geometry and spread in 2D.

Everything is driven by a single Generator instance created in run_report.py
(np.random.default_rng(42)), so the whole report is reproducible end to end.
"""
import numpy as np

import style
from style import SERIES, MARKERS, ACCENT, HAIRLINE, INK, INK_SOFT, MONO
import matplotlib.pyplot as plt

# --8<-- [start:params]
# Class parameters, exactly as given in the statement.
MEANS = np.array([[2.0, 3.0],      # class 0
                  [5.0, 6.0],      # class 1
                  [8.0, 1.0],      # class 2
                  [15.0, 4.0]])    # class 3
STDS = np.array([[0.8, 2.5],       # class 0 - stretched along x2
                 [1.2, 1.9],       # class 1
                 [0.9, 0.9],       # class 2 - isotropic
                 [0.5, 2.0]])      # class 3 - stretched along x2
N_PER_CLASS = 100
SCALES = [0.5, 1.0, 2.0, 4.0]      # spread multipliers used in item B
LABELS = [f"Class {k}" for k in range(4)]
# --8<-- [end:params]


# --8<-- [start:generate]
def make_clouds(rng, scale=1.0):
    """Draw 100 axis-aligned Gaussian points per class.

    `scale` multiplies every standard deviation; the means never move, which is
    what makes the four datasets of item B comparable.
    """
    parts = [rng.normal(loc=MEANS[k], scale=STDS[k] * scale,
                        size=(N_PER_CLASS, 2)) for k in range(4)]
    X = np.vstack(parts)
    y = np.repeat(np.arange(4), N_PER_CLASS)
    return X, y
# --8<-- [end:generate]


# --8<-- [start:metrics]
def separation_ratio(scale=1.0):
    """r_ij = ||mu_i - mu_j|| / (sigma_bar_i + sigma_bar_j) for the 6 class pairs.

    sigma_bar_k is the mean of the two per-feature standard deviations of class
    k (a single spread number per cloud), exactly as defined in the statement.
    Since the means are fixed and every sigma is multiplied by `scale`, r_ij is
    exactly proportional to 1/scale.
    """
    sigma = (STDS * scale).mean(axis=1)          # one spread per class
    rows = []
    for i in range(4):
        for j in range(i + 1, 4):
            d = np.linalg.norm(MEANS[i] - MEANS[j])
            rows.append((i, j, d, sigma[i] + sigma[j], d / (sigma[i] + sigma[j])))
    return rows


def mixing_rate(X, y):
    """Fraction of points whose nearest class centre is not their own class.

    Purely geometric: each point is compared against the 4 theoretical means
    (the generating centres), no model and no fitting involved.
    """
    d = np.linalg.norm(X[:, None, :] - MEANS[None, :, :], axis=2)   # (N, 4)
    nearest = d.argmin(axis=1)
    return float((nearest != y).mean()), nearest
# --8<-- [end:metrics]


def _scatter(ax, X, y, alpha=0.85, s=16, legend=True):
    """One scatter per class: fixed colour slot + fixed marker shape."""
    for k in range(4):
        m = y == k
        ax.scatter(X[m, 0], X[m, 1], s=s, c=SERIES[k], marker=MARKERS[k],
                   alpha=alpha, linewidths=0.4, edgecolors=style.SURFACE,
                   label=LABELS[k] if legend else None)
    ax.set_xlabel("Feature $x_1$")
    ax.set_ylabel("Feature $x_2$")


def _centres(ax, annotate=True):
    """Mark the generating means; the label carries the coordinates."""
    # proxy handle so the legend key is not the oversized plotted marker
    ax.scatter([], [], s=55, marker="X", c=style.HAIRLINE, edgecolors=INK,
               linewidths=0.9, label="Class centres ($\\mu_k$)")
    for k, (mx, my) in enumerate(MEANS):
        ax.scatter([mx], [my], s=150, marker="X", c=SERIES[k],
                   edgecolors=INK, linewidths=1.1, zorder=5)
        if annotate:
            # labels flip to the left near the right-hand edge so nothing clips
            off = (-72, 8) if mx > 12 else (9, 7)
            style.halo(ax.annotate(f"$\\mu_{k}$=({mx:.0f}, {my:.0f})", (mx, my),
                                   textcoords="offset points", xytext=off,
                                   fontsize=7.5, family=MONO, color=INK))


def figure1(X, y):
    """Figure 1 - the four clouds at s = 1 with their centres marked."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _scatter(ax, X, y)
    _centres(ax)
    style.title(ax, "Figure 1 - Four Gaussian point clouds in 2D",
                "400 samples, 100 per class, s = 1.0 (original spread); X marks each generating mean")
    style.legend_row(ax, ncols=5)
    style.mono_ticks(ax)
    return style.save(fig, "fig1_clouds.png")


def figure1_boundaries(X, y):
    """Figure 1b - Figure 1 plus the boundaries a trained network might learn.

    The sketch is the nearest-centre (Voronoi) partition of the four means: the
    piecewise-linear frontier that a small network approximates when the clouds
    are roughly balanced and isotropic. Drawn as thin accent hairlines so the
    data stays the subject.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    pad = 3.0
    gx = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 700)
    gy = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 700)
    GX, GY = np.meshgrid(gx, gy)
    grid = np.column_stack([GX.ravel(), GY.ravel()])
    region = np.linalg.norm(grid[:, None, :] - MEANS[None, :, :], axis=2).argmin(axis=1)
    region = region.reshape(GX.shape)

    # faint region fills + hairline frontiers
    ax.contourf(GX, GY, region, levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
                colors=SERIES, alpha=0.07, zorder=0)
    ax.contour(GX, GY, region, levels=[0.5, 1.5, 2.5],
               colors=ACCENT, linewidths=1.0, linestyles="--", zorder=1)
    _scatter(ax, X, y)
    _centres(ax, annotate=False)
    style.title(ax, "Figure 1b - Sketched decision boundaries (s = 1.0)",
                "Dashed lines: nearest-centre (piecewise-linear) frontier a trained network would approximate")
    ax.plot([], [], ls="--", color=ACCENT, lw=1.0, label="Sketched boundary")
    style.legend_row(ax, ncols=6)
    style.mono_ticks(ax)
    return style.save(fig, "fig1b_boundaries.png")


def figure2(datasets):
    """Figure 2 - the same four classes at the four spread scales, shared axes."""
    allX = np.vstack([d[0] for d in datasets.values()])
    xlim = (allX[:, 0].min() - 1, allX[:, 0].max() + 1)
    ylim = (allX[:, 1].min() - 1, allX[:, 1].max() + 1)

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.8), sharex=True, sharey=True)
    for ax, (s, (X, y)) in zip(axes.ravel(), datasets.items()):
        _scatter(ax, X, y, s=11, alpha=0.75, legend=(ax is axes[0, 0]))
        _centres(ax, annotate=False)
        rate, _ = mixing_rate(X, y)
        ax.set_title(f"s = {s:.1f}   |   mixing rate = {rate:.1%}",
                     fontfamily=style.SERIF, fontsize=10, loc="left")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        style.mono_ticks(ax)
    for ax in axes[:, 1]:
        ax.set_ylabel("")
    for ax in axes[0, :]:
        ax.set_xlabel("")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].get_legend().remove() if axes[0, 0].get_legend() else None
    fig.suptitle("Figure 2 - Same four classes, four spread scales (shared axis limits)",
                 fontfamily=style.SERIF, fontsize=12.5, x=0.01, ha="left", y=0.995)
    fig.legend(handles, labels, loc="lower center", ncols=5, frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.035, 1, 0.96])
    return style.save(fig, "fig2_scales.png")


def figure3(rates):
    """Figure 3 - mixing rate as a function of the spread scale."""
    s = np.array(list(rates.keys()), dtype=float)
    r = np.array(list(rates.values()), dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(s, r * 100, color=SERIES[0], marker="o", markersize=7,
            markeredgecolor=style.SURFACE, markeredgewidth=1.2)
    for xi, yi in zip(s, r):
        ax.annotate(f"{yi:.1%}", (xi, yi * 100), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, family=MONO, color=INK)
    ax.set_xlabel("Spread scale factor $s$")
    ax.set_ylabel("Points closer to another centre (%)")
    ax.set_xticks(s)
    ax.set_ylim(0, max(r * 100) * 1.28)
    style.title(ax, "Figure 3 - Mixing rate vs spread scale",
                "Fraction of the 400 points whose nearest generating centre belongs to another class")
    style.mono_ticks(ax)
    return style.save(fig, "fig3_mixing.png")


def run(rng):
    """Run item A, B and the figures; return every number the report quotes."""
    print("Exercise 1 - point clouds")
    datasets = {s: make_clouds(rng, scale=s) for s in SCALES}   # 4 datasets, 4 classes each
    X1, y1 = datasets[1.0]

    figure1(X1, y1)
    figure1_boundaries(X1, y1)
    figure2(datasets)

    rates = {s: mixing_rate(X, y)[0] for s, (X, y) in datasets.items()}
    figure3(rates)

    pairs = separation_ratio(1.0)
    smallest = min(pairs, key=lambda r: r[-1])
    for i, j, d, sg, r in pairs:
        print(f"  r_{i}{j} = {r:.4f}   (d = {d:.4f}, sigma_i+sigma_j = {sg:.3f})")
    print(f"  smallest r at s=1: pair ({smallest[0]},{smallest[1]}) = {smallest[-1]:.4f}"
          f" -> at s=2 it becomes {smallest[-1] / 2.0:.4f}")
    for s, r in rates.items():
        print(f"  mixing rate at s={s}: {r:.4%}")

    return {
        "separation_ratios_s1": [
            {"pair": f"({i},{j})", "distance": d, "sigma_sum": sg, "r": r}
            for i, j, d, sg, r in pairs
        ],
        "smallest_r_s1": {"pair": f"({smallest[0]},{smallest[1]})", "r": smallest[-1]},
        "smallest_r_s2": smallest[-1] / 2.0,
        "mixing_rates": {str(s): r for s, r in rates.items()},
    }
