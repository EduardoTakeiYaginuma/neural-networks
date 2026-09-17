"""A binary perceptron implemented only with NumPy.

The same training loop is used for both exercises.  Besides the final iterate,
it keeps a pocket copy whenever an individual update improves full-dataset
accuracy.  For separable data the pocket is harmless; for overlapping data it
preserves a useful boundary that the last iterate may lose.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class TrainingResult:
    """All state needed to reproduce the training analysis and figures."""

    weights: np.ndarray
    bias: float
    pocket_weights: np.ndarray
    pocket_bias: float
    pocket_accuracy: float
    pocket_epoch: int
    pocket_update: int
    epochs: int
    accuracy_history: list[float]
    pocket_accuracy_history: list[float]
    updates_history: list[int]
    total_updates: int


def initialize_weights(rng: np.random.Generator, n_features: int) -> np.ndarray:
    """Draw the prescribed small, non-zero initial weight vector."""

    return rng.normal(0.0, 0.01, size=n_features)


class Perceptron:
    """Binary 0/1 perceptron with online, error-driven updates."""

    def __init__(self, learning_rate: float = 0.01, max_epochs: int = 100):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if max_epochs < 1:
            raise ValueError("max_epochs must be at least 1")
        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)

    @staticmethod
    def step(scores):
        """Return 1 at and above zero, and 0 below zero."""

        return (np.asarray(scores) >= 0.0).astype(int)

    @classmethod
    def predict(cls, X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
        """Predict labels for every row in ``X``."""

        return cls.step(X @ weights + bias)

    @classmethod
    def accuracy(cls, X, y, weights, bias) -> float:
        """Compute full-dataset classification accuracy."""

        return float(np.mean(cls.predict(X, weights, bias) == y))

    def fit(self, X, y, initial_weights, initial_bias: float = 0.0) -> TrainingResult:
        """Train until an update-free pass or the epoch cap is reached.

        Samples are deliberately visited in their supplied order.  Pocket
        accuracy is checked after every mistake-driven update, not merely at
        epoch boundaries, because useful iterates can be short-lived.
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        weights = np.asarray(initial_weights, dtype=float).copy()
        bias = float(initial_bias)

        if X.ndim != 2 or y.shape != (len(X),):
            raise ValueError("X must be 2D and y must contain one label per row")
        if weights.shape != (X.shape[1],):
            raise ValueError("initial_weights has the wrong number of features")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("perceptron labels must be 0 or 1")

        pocket_weights = weights.copy()
        pocket_bias = bias
        pocket_accuracy = self.accuracy(X, y, weights, bias)
        pocket_epoch = 0
        pocket_update = 0

        accuracy_history = []
        pocket_accuracy_history = []
        updates_history = []
        total_updates = 0

        for epoch in range(1, self.max_epochs + 1):
            updates = 0

            for xi, target in zip(X, y):
                prediction = int(self.step(np.dot(weights, xi) + bias).item())
                error = int(target - prediction)
                if error == 0:
                    continue

                weights += self.learning_rate * error * xi
                bias += self.learning_rate * error
                updates += 1
                total_updates += 1

                current_accuracy = self.accuracy(X, y, weights, bias)
                if current_accuracy > pocket_accuracy:
                    pocket_accuracy = current_accuracy
                    pocket_weights = weights.copy()
                    pocket_bias = bias
                    pocket_epoch = epoch
                    pocket_update = total_updates

            accuracy_history.append(self.accuracy(X, y, weights, bias))
            pocket_accuracy_history.append(pocket_accuracy)
            updates_history.append(updates)

            if updates == 0:
                break

        return TrainingResult(
            weights=weights.copy(),
            bias=bias,
            pocket_weights=pocket_weights,
            pocket_bias=pocket_bias,
            pocket_accuracy=pocket_accuracy,
            pocket_epoch=pocket_epoch,
            pocket_update=pocket_update,
            epochs=len(accuracy_history),
            accuracy_history=accuracy_history,
            pocket_accuracy_history=pocket_accuracy_history,
            updates_history=updates_history,
            total_updates=total_updates,
        )
