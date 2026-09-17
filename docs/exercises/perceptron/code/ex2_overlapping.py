"""Exercise 2: train the same perceptron on overlapping Gaussian classes."""
import math

import numpy as np
import matplotlib.pyplot as plt

import style
from perceptron import Perceptron, initialize_weights
from ex1_separable import boundary_segment, scatter_classes


def generate_data(rng):
    """Generate 1,000 observations from each overlapping Gaussian."""

    covariance = np.array([[1.5, 0.0], [0.0, 1.5]])
    class_0 = rng.multivariate_normal([3.0, 3.0], covariance, size=1000)
    class_1 = rng.multivariate_normal([4.0, 4.0], covariance, size=1000)
    X = np.vstack([class_0, class_1])
    y = np.concatenate([np.zeros(1000, dtype=int), np.ones(1000, dtype=int)])
    return X, y


def bayes_linear_accuracy(mean_0, mean_1, variance):
    """Accuracy of the optimal linear rule for two equal-covariance Gaussians.

    With shared covariance ``variance * I`` and equal priors, the optimal
    boundary is perpendicular to the line joining the means.  Projected onto
    that direction the classes are one-dimensional normals separated by
    ``d = ||mean_1 - mean_0||`` with standard deviation ``sqrt(variance)``, so
    each class contributes an error of ``Phi(-d / (2 sigma))``.
    """

    separation = float(np.linalg.norm(np.asarray(mean_1) - np.asarray(mean_0)))
    z = separation / (2.0 * math.sqrt(variance))
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def best_linear_accuracy(X, y, n_angles=721):
    """Exhaustively search directions and thresholds for the best line.

    For each candidate normal direction the projections are sorted once and
    every threshold is scored at once from cumulative class counts, so the
    whole sweep stays cheap.  This is the empirical ceiling any perceptron
    boundary could reach on the realized sample.
    """

    n = len(y)
    best = 0.0
    for theta in np.linspace(0.0, np.pi, n_angles):
        # Written componentwise rather than as a matmul: some BLAS builds raise
        # spurious floating-point warnings on this shape inside a notebook.
        scores = X[:, 0] * np.cos(theta) + X[:, 1] * np.sin(theta)
        order = np.argsort(scores, kind="stable")
        labels = y[order]
        # Predicting 1 above the cut: correct = class-1 above + class-0 below.
        zeros_below = np.concatenate([[0], np.cumsum(labels == 0)])
        ones_above = np.concatenate([[np.sum(labels == 1)],
                                     np.sum(labels == 1) - np.cumsum(labels == 1)])
        correct = zeros_below + ones_above
        best = max(best, float(np.max(correct)) / n,
                   float(n - np.min(correct)) / n)   # the flipped orientation
    return best


def interleaved_order(X, y):
    """Reorder the sample so classes alternate, without drawing any randomness.

    The training loop visits samples in the order it is given.  Comparing the
    generation order (all of class 0, then all of class 1) against a strict
    alternation isolates how much of the final iterate's behaviour is an
    artefact of that ordering rather than of the overlap itself.
    """

    index_0 = np.flatnonzero(y == 0)
    index_1 = np.flatnonzero(y == 1)
    paired = np.empty(len(index_0) + len(index_1), dtype=int)
    paired[0::2] = index_0
    paired[1::2] = index_1
    return X[paired], y[paired]


def plot_scatter(X, y):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    scatter_classes(ax, X, y)
    style.title(ax, "Overlapping Gaussian classes",
                "Close means and three times the variance of Exercise 1")
    ax.set(xlabel=r"Feature $x_1$", ylabel=r"Feature $x_2$")
    style.legend_row(ax, ncols=2, y=-0.14)
    style.mono_ticks(ax)
    fig.subplots_adjust(bottom=0.18, top=0.86)
    style.save(fig, "fig4_overlap_scatter.png")


def plot_boundaries(X, y, result):
    """Draw both boundaries together, then one panel of errors for each.

    The left panel answers the statement literally — the data carrying both
    decision boundaries at once.  The two panels beside it separate the error
    sets, which would be unreadable stacked on a single axes.
    """

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharex=True, sharey=True)
    xlim = (float(X[:, 0].min() - 0.5), float(X[:, 0].max() + 0.5))
    ylim = (float(X[:, 1].min() - 0.5), float(X[:, 1].max() + 0.5))
    final_wrong = Perceptron.predict(X, result.weights, result.bias) != y
    pocket_wrong = Perceptron.predict(
        X, result.pocket_weights, result.pocket_bias) != y

    # Left panel: both boundaries over the data, no error layer.
    scatter_classes(axes[0], X, y, alpha=0.28)
    for weights, bias, colour, line_style, name in (
            (result.weights, result.bias, "#B2182B", "--", "Final iterate"),
            (result.pocket_weights, result.pocket_bias, style.INK, "-",
             "Pocket")):
        axes[0].plot(*boundary_segment(weights, bias, xlim, ylim), color=colour,
                     linestyle=line_style, linewidth=2.2, label=name)
    axes[0].set(xlim=xlim, ylim=ylim, xlabel=r"Feature $x_1$",
                ylabel=r"Feature $x_2$")
    axes[0].set_title("Both boundaries", fontfamily=style.SERIF, fontsize=10.5)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2,
                   frameon=False, fontsize=7.3, handletextpad=0.35,
                   columnspacing=0.9)
    style.mono_ticks(axes[0])

    panels = [
        (axes[1], result.weights, result.bias, final_wrong, "#B2182B", "x",
         f"Final iterate — {1 - final_wrong.mean():.2%}", "--"),
        (axes[2], result.pocket_weights, result.pocket_bias, pocket_wrong,
         style.INK, "o", f"Pocket — {1 - pocket_wrong.mean():.2%}", "-"),
    ]
    for ax, weights, bias, wrong, colour, marker, panel_title, line_style in panels:
        scatter_classes(ax, X, y, alpha=0.28)
        boundary = boundary_segment(weights, bias, xlim, ylim)
        ax.plot(*boundary, color=colour, linestyle=line_style, linewidth=2.2,
                label="Decision boundary")
        if marker == "x":
            ax.scatter(X[wrong, 0], X[wrong, 1], s=25, marker=marker,
                       c=colour, linewidths=0.65,
                       label=f"Misclassified ({wrong.sum()})", zorder=4)
        else:
            ax.scatter(X[wrong, 0], X[wrong, 1], s=31, marker=marker,
                       facecolors="none", edgecolors=colour, linewidths=0.65,
                       label=f"Misclassified ({wrong.sum()})", zorder=4)
        ax.set(xlim=xlim, ylim=ylim, xlabel=r"Feature $x_1$",
               ylabel=r"Feature $x_2$")
        ax.set_title(panel_title, fontfamily=style.SERIF, fontsize=10.5)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2,
                  frameon=False, fontsize=7.3, handletextpad=0.35,
                  columnspacing=0.9)
        style.mono_ticks(ax)

    fig.suptitle("Last iterate versus pocket boundary", x=0.06, ha="left",
                 fontfamily=style.SERIF, fontsize=13, color=style.INK)
    fig.text(0.06, 0.91,
             "Both boundaries together, then the errors each one produces",
             fontsize=8, color=style.INK_SOFT, ha="left")
    fig.subplots_adjust(bottom=0.24, top=0.82, wspace=0.12)
    style.save(fig, "fig5_overlap_boundaries.png")


def plot_accuracy(result):
    epochs = np.arange(1, result.epochs + 1)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(epochs, np.asarray(result.accuracy_history) * 100,
            color="#B2182B", linewidth=1.6, label="Current weights")
    ax.plot(epochs, np.asarray(result.pocket_accuracy_history) * 100,
            color=style.INK, linewidth=2.2, label="Pocket best-so-far")
    ax.set(xlabel="Epoch", ylabel="Full-dataset accuracy (%)",
           xlim=(1, result.epochs), ylim=(45, 78))
    style.title(ax, "The last iterate oscillates while the pocket holds",
                "Current and best-so-far accuracy after each epoch")
    style.legend_row(ax, ncols=2, y=-0.18)
    style.mono_ticks(ax)
    fig.subplots_adjust(bottom=0.22, top=0.83)
    style.save(fig, "fig6_overlap_accuracy.png")


def run(rng):
    X, y = generate_data(rng)
    initial_weights = initialize_weights(rng, X.shape[1])
    result = Perceptron(learning_rate=0.01, max_epochs=100).fit(
        X, y, initial_weights, initial_bias=0.0)

    plot_scatter(X, y)
    plot_boundaries(X, y, result)
    plot_accuracy(result)

    centroid = X.mean(axis=0)
    final_norm = float(np.linalg.norm(result.weights))
    pocket_norm = float(np.linalg.norm(result.pocket_weights))
    final_predictions = Perceptron.predict(X, result.weights, result.bias)
    pocket_predictions = Perceptron.predict(
        X, result.pocket_weights, result.pocket_bias)

    # Two reference points for the analysis, neither of which draws randomness:
    # the ceiling any straight line could reach, and the same training run with
    # the class-blocked visiting order replaced by a strict alternation.
    shuffled_X, shuffled_y = interleaved_order(X, y)
    interleaved = Perceptron(learning_rate=0.01, max_epochs=100).fit(
        shuffled_X, shuffled_y, initial_weights, initial_bias=0.0)

    diagnostics = {
        "bayes_linear_accuracy": bayes_linear_accuracy([3.0, 3.0], [4.0, 4.0], 1.5),
        "best_linear_accuracy": best_linear_accuracy(X, y),
        "interleaved_final_accuracy": interleaved.accuracy_history[-1],
        "interleaved_pocket_accuracy": interleaved.pocket_accuracy,
        "interleaved_total_updates": interleaved.total_updates,
        "mean_sample_norm": float(np.mean(np.linalg.norm(X, axis=1))),
        "centroid": centroid,
        "final_weight_norm": final_norm,
        "final_origin_offset": float(-result.bias / final_norm),
        "final_centroid_signed_distance": float(
            (np.dot(result.weights, centroid) + result.bias) / final_norm),
        "final_centroid_distance": float(
            abs(np.dot(result.weights, centroid) + result.bias) / final_norm),
        "final_predicted_class_0": int(np.sum(final_predictions == 0)),
        "final_predicted_class_1": int(np.sum(final_predictions == 1)),
        "pocket_weight_norm": pocket_norm,
        "pocket_centroid_distance": float(
            abs(np.dot(result.pocket_weights, centroid) + result.pocket_bias)
            / pocket_norm),
        "pocket_predicted_class_0": int(np.sum(pocket_predictions == 0)),
        "pocket_predicted_class_1": int(np.sum(pocket_predictions == 1)),
        "final_accuracy": Perceptron.accuracy(
            X, y, result.weights, result.bias),
    }

    return {
        "X": X,
        "y": y,
        "initial_weights": initial_weights,
        "training": result,
        "diagnostics": diagnostics,
    }
