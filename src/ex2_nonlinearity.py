"""
Exercise 2 - Non-linearity in higher dimensions.

Dataset I  : two shifted 5D Gaussians with different covariance structures.
Dataset II : two concentric spherical shells in 5D (same centre, different radii).
"""
import numpy as np
from sklearn.decomposition import PCA

import style
from style import SERIES, MARKERS, ACCENT, INK, INK_SOFT, MONO
import matplotlib.pyplot as plt

# --8<-- [start:params]
N_PER_CLASS = 500

# ---- Dataset I: shifted Gaussians -------------------------------------------
MU_A = np.zeros(5)
MU_B = np.full(5, 1.5)

SIGMA_A = np.array([[1.0, 0.8, 0.1, 0.0, 0.0],
                    [0.8, 1.0, 0.3, 0.0, 0.0],
                    [0.1, 0.3, 1.0, 0.5, 0.0],
                    [0.0, 0.0, 0.5, 1.0, 0.2],
                    [0.0, 0.0, 0.0, 0.2, 1.0]])

SIGMA_B = np.array([[1.5, -0.7, 0.2, 0.0, 0.0],
                    [-0.7, 1.5, 0.4, 0.0, 0.0],
                    [0.2, 0.4, 1.5, 0.6, 0.0],
                    [0.0, 0.0, 0.6, 1.5, 0.3],
                    [0.0, 0.0, 0.0, 0.3, 1.5]])

# ---- Dataset II: concentric shells ------------------------------------------
R_CORE = (3.0, 0.4)     # class C: radius ~ N(3.0, 0.4)
R_SHELL = (8.0, 0.4)    # class D: radius ~ N(8.0, 0.4)
# --8<-- [end:params]


# --8<-- [start:dataset1]
def dataset_gaussians(rng):
    """Dataset I - 500 samples per class from two 5D multivariate normals."""
    A = rng.multivariate_normal(MU_A, SIGMA_A, size=N_PER_CLASS)
    B = rng.multivariate_normal(MU_B, SIGMA_B, size=N_PER_CLASS)
    X = np.vstack([A, B])
    y = np.repeat([0, 1], N_PER_CLASS)          # 0 = Class A, 1 = Class B
    return X, y
# --8<-- [end:dataset1]


# --8<-- [start:dataset2]
def dataset_shells(rng):
    """Dataset II - two concentric shells: same centre, different radii.

    Directions are uniform on the unit sphere of R^5: draw v ~ N(0, I_5) and
    normalise, u = v / ||v||. Then x = r * u with r drawn per class.
    """
    def shell(mu_r, sd_r):
        v = rng.normal(size=(N_PER_CLASS, 5))
        u = v / np.linalg.norm(v, axis=1, keepdims=True)   # uniform directions
        r = rng.normal(mu_r, sd_r, size=(N_PER_CLASS, 1))  # radius
        return r * u

    C = shell(*R_CORE)
    D = shell(*R_SHELL)
    X = np.vstack([C, D])
    y = np.repeat([0, 1], N_PER_CLASS)          # 0 = Class C, 1 = Class D
    return X, y
# --8<-- [end:dataset2]


# --8<-- [start:measures]
def centre_distance(X, y):
    """Euclidean distance between the two empirical class centres, in 5D."""
    c0 = X[y == 0].mean(axis=0)
    c1 = X[y == 1].mean(axis=0)
    return float(np.linalg.norm(c0 - c1)), c0, c1


def radii(X):
    """||x|| for every point (distance from the origin), in 5D."""
    return np.linalg.norm(X, axis=1)


def radius_rule_accuracy(X, y, threshold):
    """Accuracy of the non-linear rule  f(x) = ||x||^2 - t^2  (sign = class)."""
    pred = (radii(X) ** 2 - threshold ** 2 > 0).astype(int)
    return float((pred == y).mean())
# --8<-- [end:measures]


def _scatter2d(ax, Z, y, labels):
    for k in (0, 1):
        m = y == k
        ax.scatter(Z[m, 0], Z[m, 1], s=12, c=SERIES[k], marker=MARKERS[k],
                   alpha=0.7, linewidths=0.3, edgecolors=style.SURFACE,
                   label=labels[k])
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")


def figure4(proj):
    """Figure 4 - PCA projection of both datasets onto 2 components."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    for ax, (name, (Z, y, evr, labels)) in zip(axes, proj.items()):
        _scatter2d(ax, Z, y, labels)
        ax.set_title(f"{name}   |   PC1+PC2 = {evr.sum():.1%} of variance",
                     fontfamily=style.SERIF, fontsize=10.5, loc="left")
        ax.legend(loc="best")
        style.mono_ticks(ax)
    fig.suptitle("Figure 4 - 2D PCA projection of the two 5D datasets",
                 fontfamily=style.SERIF, fontsize=12.5, x=0.008, ha="left", y=1.0)
    fig.tight_layout()
    return style.save(fig, "fig4_pca.png")


def figure5(hists):
    """Figure 5 - histogram of the 5D radius ||x||, both classes overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    for ax, (name, (X, y, labels)) in zip(axes, hists.items()):
        r = radii(X)
        bins = np.linspace(r.min(), r.max(), 45)
        for k in (0, 1):
            ax.hist(r[y == k], bins=bins, color=SERIES[k], alpha=0.62,
                    label=labels[k], edgecolor=style.SURFACE, linewidth=0.5)
        ax.set_xlabel("Radius $\\|x\\|$ (5D)")
        ax.set_ylabel("Count")
        ax.set_title(name, fontfamily=style.SERIF, fontsize=10.5, loc="left")
        ax.legend(loc="upper right")
        style.mono_ticks(ax)
    axes[1].axvline(5.5, color=ACCENT, lw=1.0, ls="--")
    axes[1].annotate("$\\|x\\| = 5.5$", (5.5, axes[1].get_ylim()[1] * 0.92),
                     xytext=(6, 0), textcoords="offset points",
                     fontsize=8, family=MONO, color=ACCENT)
    fig.suptitle("Figure 5 - Distribution of the radius, computed in 5D",
                 fontfamily=style.SERIF, fontsize=12.5, x=0.008, ha="left", y=1.0)
    fig.tight_layout()
    return style.save(fig, "fig5_radii.png")


def run(rng):
    print("Exercise 2 - non-linearity in 5D")
    X1, y1 = dataset_gaussians(rng)
    X2, y2 = dataset_shells(rng)

    lab1 = ["Class A", "Class B"]
    lab2 = ["Class C (core)", "Class D (shell)"]

    d1, c1a, c1b = centre_distance(X1, y1)
    d2, c2a, c2b = centre_distance(X2, y2)

    p1 = PCA(n_components=2).fit(X1)
    p2 = PCA(n_components=2).fit(X2)
    Z1, Z2 = p1.transform(X1), p2.transform(X2)

    figure4({"Dataset I - shifted Gaussians": (Z1, y1, p1.explained_variance_ratio_, lab1),
             "Dataset II - concentric shells": (Z2, y2, p2.explained_variance_ratio_, lab2)})
    figure5({"Dataset I - shifted Gaussians": (X1, y1, lab1),
             "Dataset II - concentric shells": (X2, y2, lab2)})

    acc = radius_rule_accuracy(X2, y2, threshold=5.5)
    r2 = radii(X2)

    print(f"  Dataset I  : ||cA - cB|| = {d1:.4f}")
    print(f"  Dataset II : ||cC - cD|| = {d2:.4f}")
    print(f"  Dataset I  : EVR = {p1.explained_variance_ratio_} -> sum {p1.explained_variance_ratio_.sum():.4%}")
    print(f"  Dataset II : EVR = {p2.explained_variance_ratio_} -> sum {p2.explained_variance_ratio_.sum():.4%}")
    print(f"  Dataset II radius: C mean {r2[y2==0].mean():.3f} (min {r2[y2==0].min():.3f}, max {r2[y2==0].max():.3f}), "
          f"D mean {r2[y2==1].mean():.3f} (min {r2[y2==1].min():.3f}, max {r2[y2==1].max():.3f})")
    print(f"  Dataset II : accuracy of sign(||x||^2 - 5.5^2) = {acc:.4%}")

    return {
        "dataset1": {
            "centre_distance": d1,
            "evr": p1.explained_variance_ratio_.tolist(),
            "evr_sum": float(p1.explained_variance_ratio_.sum()),
        },
        "dataset2": {
            "centre_distance": d2,
            "evr": p2.explained_variance_ratio_.tolist(),
            "evr_sum": float(p2.explained_variance_ratio_.sum()),
            "radius_C": [float(r2[y2 == 0].min()), float(r2[y2 == 0].mean()), float(r2[y2 == 0].max())],
            "radius_D": [float(r2[y2 == 1].min()), float(r2[y2 == 1].mean()), float(r2[y2 == 1].max())],
            "radius_rule_accuracy": acc,
        },
    }
