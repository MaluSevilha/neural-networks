"""Perceptron exercise -- ANN-DL 2026.2.

The classifier itself is implemented from scratch. NumPy is used only for
arrays/basic linear algebra and for reproducible synthetic data generation.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)
OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)

class Perceptron:
    def __init__(self, learning_rate=0.01, max_epochs=100):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = np.zeros(2, dtype=float)
        self.bias = 0.0
        self.accuracy_history = []
        self.update_history = []

    def predict_one(self, x):
        return 1 if float(np.dot(self.weights, x) + self.bias) >= 0 else -1

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

    def accuracy(self, X, y):
        predictions = self.predict(X)
        correct = sum(int(p == target) for p, target in zip(predictions, y))
        return correct / len(y)

    def fit(self, X, y):
        for _ in range(self.max_epochs):
            updates = 0
            for x, target in zip(X, y):
                prediction = self.predict_one(x)
                if prediction != target:
                    self.weights = self.weights + self.learning_rate * target * x
                    self.bias = self.bias + self.learning_rate * target
                    updates += 1
            self.accuracy_history.append(self.accuracy(X, y))
            self.update_history.append(updates)
            if updates == 0:
                break
        return self


def make_dataset(mean0, mean1, variance):
    covariance = np.array([[variance, 0.0], [0.0, variance]])
    x0 = RNG.multivariate_normal(mean0, covariance, size=1000)
    x1 = RNG.multivariate_normal(mean1, covariance, size=1000)
    X = np.vstack([x0, x1])
    y = np.concatenate([-np.ones(1000, dtype=int), np.ones(1000, dtype=int)])
    order = RNG.permutation(len(y))
    return X[order], y[order]


def plot_dataset(X, y, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[y == -1, 0], X[y == -1, 1], s=12, alpha=.55, label="Class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=12, alpha=.55, label="Class 1")
    ax.set(title=title, xlabel="x1", ylabel="x2")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / filename, dpi=160); plt.close(fig)


def plot_boundary(X, y, model, title, filename):
    pred = model.predict(X)
    wrong = pred != y
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[y == -1, 0], X[y == -1, 1], s=12, alpha=.45, label="Class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=12, alpha=.45, label="Class 1")
    if wrong.any():
        ax.scatter(X[wrong, 0], X[wrong, 1], s=45, facecolors="none", edgecolors="black", linewidths=.8, label="Misclassified")
    xs = np.linspace(X[:,0].min(), X[:,0].max(), 200)
    if abs(model.weights[1]) > 1e-12:
        ys = -(model.weights[0] * xs + model.bias) / model.weights[1]
        ax.plot(xs, ys, "k--", linewidth=1.5, label="Decision boundary")
    ax.set(title=title, xlabel="x1", ylabel="x2")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / filename, dpi=160); plt.close(fig)


def plot_accuracy(model, title, filename):
    epochs = np.arange(1, len(model.accuracy_history) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, model.accuracy_history, marker="o", markersize=3)
    ax.set(title=title, xlabel="Epoch", ylabel="Training accuracy", ylim=(0, 1.02))
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUT / filename, dpi=160); plt.close(fig)


def run(name, mean0, mean1, variance):
    X, y = make_dataset(mean0, mean1, variance)
    model = Perceptron().fit(X, y)
    plot_dataset(X, y, f"{name}: generated data", f"{name.lower()}_data.png")
    plot_boundary(X, y, model, f"{name}: learned decision boundary", f"{name.lower()}_boundary.png")
    plot_accuracy(model, f"{name}: accuracy by epoch", f"{name.lower()}_accuracy.png")
    print(name)
    print("  epochs:", len(model.accuracy_history))
    print("  weights:", model.weights.tolist())
    print("  bias:", model.bias)
    print("  final accuracy:", model.accuracy_history[-1])
    print("  best accuracy:", max(model.accuracy_history))
    print("  final updates:", model.update_history[-1])
    return model

if __name__ == "__main__":
    run("Exercise 1", [1.5, 1.5], [5.0, 5.0], 0.5)
    run("Exercise 2", [3.0, 3.0], [4.0, 4.0], 1.5)
