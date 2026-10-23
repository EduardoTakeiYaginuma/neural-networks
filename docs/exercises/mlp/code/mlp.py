"""A small, fully NumPy multi-layer perceptron.

The same implementation handles binary and multi-class classification.  There
is no autograd: ``backward`` contains the complete chain rule, and ``update``
performs gradient descent explicitly.
"""
from __future__ import annotations

import numpy as np


class MLP:
    """Dense tanh network with sigmoid or softmax output."""

    def __init__(self, layer_sizes, task, rng):
        if len(layer_sizes) < 3:
            raise ValueError("layer_sizes must include input, hidden and output")
        if task not in {"binary", "multiclass"}:
            raise ValueError("task must be 'binary' or 'multiclass'")
        if task == "binary" and layer_sizes[-1] != 1:
            raise ValueError("binary classification requires one output unit")

        self.layer_sizes = tuple(layer_sizes)
        self.task = task
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # Glorot initialization keeps tanh activations away from saturation.
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.weights.append(rng.uniform(-limit, limit, (fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))

    @staticmethod
    def _sigmoid(z):
        """Numerically stable sigmoid."""
        out = np.empty_like(z, dtype=float)
        positive = z >= 0
        out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        exp_z = np.exp(z[~positive])
        out[~positive] = exp_z / (1.0 + exp_z)
        return out

    @staticmethod
    def _softmax(z):
        """Row-wise stable softmax."""
        shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        """Return probabilities and cache every activation for backprop."""
        activations = [np.asarray(X, dtype=float)]
        preactivations = []
        a = activations[0]

        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            z = a @ W + b
            preactivations.append(z)
            a = np.tanh(z)
            activations.append(a)

        z_out = a @ self.weights[-1] + self.biases[-1]
        preactivations.append(z_out)
        if self.task == "binary":
            output = self._sigmoid(z_out)
        else:
            output = self._softmax(z_out)
        activations.append(output)
        self._cache = (activations, preactivations)
        return output

    def loss(self, probabilities, y):
        """Mean binary or categorical cross-entropy."""
        eps = 1e-12
        p = np.clip(probabilities, eps, 1.0 - eps)
        if self.task == "binary":
            target = np.asarray(y).reshape(-1, 1)
            return float(-np.mean(target * np.log(p) + (1 - target) * np.log(1 - p)))
        target = np.asarray(y, dtype=int)
        return float(-np.mean(np.log(p[np.arange(target.size), target])))

    def backward(self, y):
        """Compute every weight and bias gradient with the chain rule."""
        activations, _ = self._cache
        n = len(y)

        # For sigmoid+BCE and softmax+CCE alike, dL/dz_out = (p - y)/n.
        if self.task == "binary":
            target = np.asarray(y).reshape(-1, 1)
        else:
            target = np.zeros_like(activations[-1])
            target[np.arange(n), np.asarray(y, dtype=int)] = 1.0
        delta = (activations[-1] - target) / n

        grad_weights = [None] * len(self.weights)
        grad_biases = [None] * len(self.biases)
        for layer in range(len(self.weights) - 1, -1, -1):
            grad_weights[layer] = activations[layer].T @ delta
            grad_biases[layer] = np.sum(delta, axis=0, keepdims=True)
            if layer > 0:
                delta = (delta @ self.weights[layer].T) * (1.0 - activations[layer] ** 2)
        return grad_weights, grad_biases

    def update(self, gradients, learning_rate):
        """Apply one full-batch gradient-descent update."""
        grad_weights, grad_biases = gradients
        for layer in range(len(self.weights)):
            self.weights[layer] -= learning_rate * grad_weights[layer]
            self.biases[layer] -= learning_rate * grad_biases[layer]

    def fit(self, X, y, epochs, learning_rate):
        """Train for a fixed budget and return the post-update loss per epoch."""
        history = []
        for _ in range(epochs):
            probabilities = self.forward(X)
            self.update(self.backward(y), learning_rate)
            # Record the loss of the parameters that actually end this epoch.
            # Thus history[-1] and the final model always describe the same state.
            history.append(self.loss(self.forward(X), y))
        return np.asarray(history)

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        if self.task == "binary":
            return (probabilities[:, 0] >= 0.5).astype(int)
        return np.argmax(probabilities, axis=1)

    @property
    def parameter_count(self):
        return int(sum(W.size + b.size for W, b in zip(self.weights, self.biases)))
