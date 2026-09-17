"""Exercise 1: generate separable clouds, train and compare learning rates."""
import numpy as np
import matplotlib.pyplot as plt

import style
from perceptron import Perceptron, initialize_weights


def generate_data(rng):
    """Generate 1,000 observations from each prescribed Gaussian."""

    covariance = np.array([[0.5, 0.0], [0.0, 0.5]])
    class_0 = rng.multivariate_normal([1.5, 1.5], covariance, size=1000)
    class_1 = rng.multivariate_normal([5.0, 5.0], covariance, size=1000)
    X = np.vstack([class_0, class_1])
    y = np.concatenate([np.zeros(1000, dtype=int), np.ones(1000, dtype=int)])
    return X, y


def scatter_classes(ax, X, y, alpha=0.58):
    """Draw both classes with stable colours and redundant marker shapes."""

    for label in (0, 1):
        points = X[y == label]
        ax.scatter(points[:, 0], points[:, 1], s=14, alpha=alpha,
                   c=style.SERIES[label], marker=style.MARKERS[label],
                   edgecolors="none", label=f"Class {label}")


def boundary_segment(weights, bias, xlim, ylim, n=300):
    """Return coordinates of a linear boundary within the current view."""

    if abs(weights[1]) >= abs(weights[0]):
        xs = np.linspace(*xlim, n)
        ys = -(weights[0] * xs + bias) / weights[1]
    else:
        ys = np.linspace(*ylim, n)
        xs = -(weights[1] * ys + bias) / weights[0]
    visible = ((xs >= xlim[0]) & (xs <= xlim[1]) &
               (ys >= ylim[0]) & (ys <= ylim[1]))
    return xs[visible], ys[visible]


def plot_scatter(X, y):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    scatter_classes(ax, X, y)
    style.title(ax, "Separable Gaussian classes",
                "2,000 observations generated from the prescribed distributions")
    ax.set(xlabel=r"Feature $x_1$", ylabel=r"Feature $x_2$")
    style.legend_row(ax, ncols=2, y=-0.14)
    style.mono_ticks(ax)
    fig.subplots_adjust(bottom=0.18, top=0.86)
    style.save(fig, "fig1_separable_scatter.png")


def plot_boundary(X, y, result):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    scatter_classes(ax, X, y, alpha=0.42)
    xlim = (float(X[:, 0].min() - 0.5), float(X[:, 0].max() + 0.5))
    ylim = (float(X[:, 1].min() - 0.5), float(X[:, 1].max() + 0.5))
    xs, ys = boundary_segment(result.weights, result.bias, xlim, ylim)
    ax.plot(xs, ys, color=style.INK, linestyle="--", label="Decision boundary")

    predicted = Perceptron.predict(X, result.weights, result.bias)
    wrong = predicted != y
    ax.scatter(X[wrong, 0], X[wrong, 1], s=54, facecolors="none",
               edgecolors="#B2182B", linewidths=1.4, marker="o",
               label=f"Misclassified ({wrong.sum()})", zorder=5)
    ax.set(xlim=xlim, ylim=ylim, xlabel=r"Feature $x_1$", ylabel=r"Feature $x_2$")
    style.title(ax, "Converged perceptron boundary",
                r"Learning rate $\eta=0.01$; errors are outlined in red")
    style.legend_row(ax, ncols=4, y=-0.14)
    style.mono_ticks(ax)
    fig.subplots_adjust(bottom=0.19, top=0.86)
    style.save(fig, "fig2_separable_boundary.png")


def plot_accuracy(result):
    epochs = np.arange(1, result.epochs + 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(epochs, np.asarray(result.accuracy_history) * 100,
            color=style.SERIES[0], marker="o", label="Current weights")
    ax.set(xlabel="Epoch", ylabel="Full-dataset accuracy (%)",
           xlim=(1, max(2, result.epochs)), ylim=(49, 101))
    style.title(ax, "Accuracy converges on separable data",
                "Accuracy recorded after every complete pass")
    style.legend_row(ax, ncols=1, y=-0.18)
    style.mono_ticks(ax)
    fig.subplots_adjust(bottom=0.22, top=0.83)
    style.save(fig, "fig3_separable_accuracy.png")


def run(rng):
    """Run both learning rates from exactly the same data and initialization."""

    X, y = generate_data(rng)
    initial_weights = initialize_weights(rng, X.shape[1])

    result_001 = Perceptron(learning_rate=0.01, max_epochs=100).fit(
        X, y, initial_weights, initial_bias=0.0)
    result_1 = Perceptron(learning_rate=1.0, max_epochs=100).fit(
        X, y, initial_weights, initial_bias=0.0)

    plot_scatter(X, y)
    plot_boundary(X, y, result_001)
    plot_accuracy(result_001)

    direction_001 = result_001.weights / np.linalg.norm(result_001.weights)
    direction_1 = result_1.weights / np.linalg.norm(result_1.weights)
    cosine = float(np.clip(np.dot(direction_001, direction_1), -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cosine)))

    return {
        "X": X,
        "y": y,
        "initial_weights": initial_weights,
        "eta_001": result_001,
        "eta_1": result_1,
        "direction_eta_001": direction_001,
        "direction_eta_1": direction_1,
        "direction_cosine": cosine,
        "direction_angle_degrees": angle,
    }
