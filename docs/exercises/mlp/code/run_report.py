"""Regenerate all Activity 3 figures, metrics and result tables.

Run from the repository root:

    python3 docs/exercises/mlp/code/run_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlp import MLP
import style

RESULTS = style.ROOT / "results" / "mlp"
RESULTS.mkdir(parents=True, exist_ok=True)

BINARY_ARCHITECTURE = [2, 8, 1]
SHALLOW_ARCHITECTURE = [4, 12, 3]
DEEP_ARCHITECTURE = [4, 12, 12, 3]
BINARY_LR = 0.1
BINARY_EPOCHS = 2000
MULTICLASS_LR = 0.1
MULTICLASS_EPOCHS = 3000
COMPARISON_SEEDS = [7, 42, 2026]


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def standardize_train_test(X_train, X_test):
    """Scale using training statistics only, preventing test-data leakage."""
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    return (X_train - mean) / scale, (X_test - mean) / scale, mean, scale


def plot_binary_data(X, y):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for cls in (0, 1):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], s=18, alpha=0.65,
                   c=style.COLORS[cls], marker=style.MARKERS[cls],
                   label=f"Class {cls}", edgecolors="none")
    style.label_axes(ax, "Figure 1 — Non-linearly separable data", "Feature 1", "Feature 2")
    ax.legend(title="Class", frameon=False)
    fig.tight_layout()
    style.save(fig, "fig1_binary_data.png")


def plot_loss(history, number, filename, label):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(np.arange(1, len(history) + 1), history,
            color=style.COLORS[0], label=label)
    style.label_axes(ax, f"Figure {number} — Training loss", "Epoch", "Cross-entropy loss")
    ax.legend(title="Series", frameon=False)
    fig.tight_layout()
    style.save(fig, filename)


def plot_decision_boundary(model, X_test_raw, y_test, mean, scale):
    pad = 1.0
    x1 = np.linspace(X_test_raw[:, 0].min() - pad, X_test_raw[:, 0].max() + pad, 350)
    x2 = np.linspace(X_test_raw[:, 1].min() - pad, X_test_raw[:, 1].max() + pad, 350)
    xx, yy = np.meshgrid(x1, x2)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    prediction = model.predict((grid - mean) / scale).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    regions = ListedColormap(["#d9e8f5", "#f5d8d8"])
    ax.contourf(xx, yy, prediction, levels=[-0.5, 0.5, 1.5], cmap=regions, alpha=0.85)
    ax.contour(xx, yy, prediction, levels=[0.5], colors=style.INK, linewidths=1.2)
    for cls in (0, 1):
        mask = y_test == cls
        ax.scatter(X_test_raw[mask, 0], X_test_raw[mask, 1], s=30,
                   c=style.COLORS[cls], marker=style.MARKERS[cls],
                   label=f"Test class {cls}", edgecolors="white", linewidths=0.4)
    style.label_axes(ax, "Figure 3 — MLP decision regions on the test set", "Feature 1", "Feature 2")
    ax.legend(title="Observed class", frameon=False)
    fig.tight_layout()
    style.save(fig, "fig3_decision_boundary.png")


def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    image = ax.imshow(cm, cmap="Blues")
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            color = "white" if cm[row, col] > cm.max() * 0.55 else style.INK
            ax.text(col, row, str(cm[row, col]), ha="center", va="center",
                    color=color, fontweight="bold")
    ax.set_xticks(range(3), ["Class 0", "Class 1", "Class 2"])
    ax.set_yticks(range(3), ["Class 0", "Class 1", "Class 2"])
    style.label_axes(ax, "Figure 5 — Confusion matrix on the test set", "Predicted class", "True class")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Number of test samples")
    fig.tight_layout()
    style.save(fig, "fig5_confusion_matrix.png")


def plot_depth_comparison(shallow_loss, deep_loss):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    epochs = np.arange(1, len(shallow_loss) + 1)
    ax.plot(epochs, shallow_loss, color=style.COLORS[0], label="Exercise 3: 1 hidden layer")
    ax.plot(epochs, deep_loss, color=style.COLORS[1], label="Exercise 4: 2 hidden layers")
    style.label_axes(ax, "Figure 6 — Controlled depth comparison", "Epoch", "Categorical cross-entropy")
    ax.legend(title="Architecture", frameon=False)
    fig.tight_layout()
    style.save(fig, "fig6_depth_comparison.png")


def run_binary():
    centers = np.array([[0.0, 0.0], [3.0, 3.0], [-3.0, -3.0]])
    X, cluster = make_blobs(n_samples=[500, 250, 250], centers=centers,
                            cluster_std=1.2, random_state=42)
    y = (cluster > 0).astype(int)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_test, mean, scale = standardize_train_test(X_train_raw, X_test_raw)

    plot_binary_data(X, y)
    baseline = LogisticRegression(random_state=42).fit(X_train, y_train)
    baseline_accuracy = accuracy(y_test, baseline.predict(X_test))

    rng = np.random.default_rng(42)
    model = MLP(BINARY_ARCHITECTURE, "binary", rng)
    history = model.fit(X_train, y_train, BINARY_EPOCHS, BINARY_LR)
    test_accuracy = accuracy(y_test, model.predict(X_test))
    plot_loss(history, 2, "fig2_binary_loss.png", "Binary MLP — training")
    plot_decision_boundary(model, X_test_raw, y_test, mean, scale)
    return {
        "architecture": BINARY_ARCHITECTURE,
        "learning_rate": BINARY_LR,
        "epochs": BINARY_EPOCHS,
        "baseline_test_accuracy": baseline_accuracy,
        "test_accuracy": test_accuracy,
        "final_train_loss": float(history[-1]),
    }


def multiclass_data():
    X, y = make_classification(
        n_samples=1500, n_features=4, n_informative=4, n_redundant=0,
        n_repeated=0, n_classes=3, n_clusters_per_class=2, class_sep=1.0,
        random_state=42)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_test, mean, scale = standardize_train_test(X_train_raw, X_test_raw)
    return X_train, X_test, y_train, y_test, mean, scale


def train_multiclass(architecture, seed, X_train, y_train):
    model = MLP(architecture, "multiclass", np.random.default_rng(seed))
    history = model.fit(X_train, y_train, MULTICLASS_EPOCHS, MULTICLASS_LR)
    return model, history


def run_multiclass_and_depth():
    X_train, X_test, y_train, y_test, _, _ = multiclass_data()

    baseline = LogisticRegression(max_iter=2000, random_state=42).fit(X_train, y_train)
    baseline_accuracy = accuracy(y_test, baseline.predict(X_test))

    trials = []
    selected = None
    for seed in COMPARISON_SEEDS:
        shallow, shallow_history = train_multiclass(SHALLOW_ARCHITECTURE, seed, X_train, y_train)
        deep, deep_history = train_multiclass(DEEP_ARCHITECTURE, seed, X_train, y_train)
        trial = {
            "seed": seed,
            "shallow_accuracy": accuracy(y_test, shallow.predict(X_test)),
            "deep_accuracy": accuracy(y_test, deep.predict(X_test)),
            "shallow_final_train_loss": float(shallow_history[-1]),
            "deep_final_train_loss": float(deep_history[-1]),
        }
        trials.append(trial)
        if seed == 42:
            selected = (shallow, shallow_history, deep, deep_history)

    shallow, shallow_history, deep, deep_history = selected
    shallow_predictions = shallow.predict(X_test)
    matrix = confusion_matrix(y_test, shallow_predictions, labels=[0, 1, 2])
    off_diagonal_pairs = {
        "0-1": int(matrix[0, 1] + matrix[1, 0]),
        "0-2": int(matrix[0, 2] + matrix[2, 0]),
        "1-2": int(matrix[1, 2] + matrix[2, 1]),
    }
    most_confused_pair = max(off_diagonal_pairs, key=off_diagonal_pairs.get)

    plot_loss(shallow_history, 4, "fig4_multiclass_loss.png", "3-class MLP — training")
    plot_confusion(matrix)
    plot_depth_comparison(shallow_history, deep_history)

    shallow_mean = float(np.mean([trial["shallow_accuracy"] for trial in trials]))
    deep_mean = float(np.mean([trial["deep_accuracy"] for trial in trials]))
    shallow_mean_loss = float(np.mean(
        [trial["shallow_final_train_loss"] for trial in trials]))
    deep_mean_loss = float(np.mean(
        [trial["deep_final_train_loss"] for trial in trials]))
    return {
        "baseline_test_accuracy": baseline_accuracy,
        "learning_rate": MULTICLASS_LR,
        "epochs": MULTICLASS_EPOCHS,
        "exercise3": {
            "architecture": SHALLOW_ARCHITECTURE,
            "parameters": shallow.parameter_count,
            "seed_42_test_accuracy": accuracy(y_test, shallow_predictions),
            "seed_42_final_train_loss": float(shallow_history[-1]),
            "confusion_matrix": matrix.tolist(),
            "pairwise_confusions": off_diagonal_pairs,
            "most_confused_pair": most_confused_pair,
            "mean_test_accuracy": shallow_mean,
            "mean_final_train_loss": shallow_mean_loss,
        },
        "exercise4": {
            "architecture": DEEP_ARCHITECTURE,
            "parameters": deep.parameter_count,
            "seed_42_test_accuracy": accuracy(y_test, deep.predict(X_test)),
            "seed_42_final_train_loss": float(deep_history[-1]),
            "mean_test_accuracy": deep_mean,
            "mean_final_train_loss": deep_mean_loss,
            "mean_difference_deep_minus_shallow": deep_mean - shallow_mean,
        },
        "trials": trials,
    }


def exercise1_numbers():
    x = np.array([0.5, -0.2])
    y = 1.0
    W1 = np.array([[0.3, -0.1], [0.2, 0.4]])
    b1 = np.array([0.1, -0.2])
    W2 = np.array([0.5, -0.3])
    b2 = 0.2
    eta = 0.3

    z1 = W1 @ x + b1
    h1 = np.tanh(z1)
    u2 = float(W2 @ h1 + b2)
    yhat = float(np.tanh(u2))
    loss = (y - yhat) ** 2
    dL_dyhat = 2.0 * (yhat - y)
    delta2 = dL_dyhat * (1.0 - yhat**2)
    dW2 = delta2 * h1
    db2 = delta2
    dL_dh1 = delta2 * W2
    delta1 = dL_dh1 * (1.0 - h1**2)
    dW1 = np.outer(delta1, x)
    db1 = delta1
    W2_new = W2 - eta * dW2
    b2_new = b2 - eta * db2
    W1_new = W1 - eta * dW1
    b1_new = b1 - eta * db1

    new_h = np.tanh(W1_new @ x + b1_new)
    new_yhat = float(np.tanh(W2_new @ new_h + b2_new))
    new_loss = (y - new_yhat) ** 2
    return {
        "z1": z1.tolist(), "h1": h1.tolist(), "u2": u2,
        "yhat": yhat, "loss": loss, "dL_dyhat": dL_dyhat,
        "delta2": delta2, "dW2": dW2.tolist(), "db2": db2,
        "dL_dh1": dL_dh1.tolist(), "delta1": delta1.tolist(),
        "dW1": dW1.tolist(), "db1": db1.tolist(),
        "W1_updated": W1_new.tolist(), "b1_updated": b1_new.tolist(),
        "W2_updated": W2_new.tolist(), "b2_updated": b2_new,
        "second_yhat": new_yhat, "second_loss": new_loss,
    }


def markdown_table(rows, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def write_tables(results):
    ex3 = results["multiclass"]["exercise3"]
    ex4 = results["multiclass"]["exercise4"]
    rows = [
        ["Exercise 3", 1, ex3["parameters"], f"{ex3['mean_final_train_loss']:.6f}",
         f"{ex3['mean_test_accuracy']:.2%}"],
        ["Exercise 4", 2, ex4["parameters"], f"{ex4['mean_final_train_loss']:.6f}",
         f"{ex4['mean_test_accuracy']:.2%}"],
    ]
    (RESULTS / "tbl_depth_comparison.md").write_text(markdown_table(
        rows, ["Model", "Hidden layers", "Parameters",
               "Mean final train loss (3 seeds)", "Mean test accuracy (3 seeds)"]))

    trial_rows = [[t["seed"], f"{t['shallow_accuracy']:.2%}", f"{t['deep_accuracy']:.2%}",
                   f"{100 * (t['deep_accuracy'] - t['shallow_accuracy']):+.2f}"]
                  for t in results["multiclass"]["trials"]]
    (RESULTS / "tbl_seed_trials.md").write_text(markdown_table(
        trial_rows, ["Seed", "Exercise 3", "Exercise 4", "Deep − shallow (p.p.)"]))


def main():
    results = {
        "reproducibility": {
            "main_seed": 42,
            "comparison_seeds": COMPARISON_SEEDS,
            "preprocessing": "training-set mean and standard deviation only",
        },
        "exercise1": exercise1_numbers(),
        "exercise2": run_binary(),
        "multiclass": run_multiclass_and_depth(),
    }
    write_tables(results)
    (RESULTS / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"  results -> {(RESULTS / 'results.json').relative_to(style.ROOT)}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
