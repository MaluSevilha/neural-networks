from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# A single RNG is reused throughout the whole report, as required by the activity.
rng = np.random.default_rng(42)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 180,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.size": 10,
})


# ============================================================
# The perceptron (written from scratch, reused in Ex. 1 and Ex. 2)
# ============================================================
class Perceptron:
    """
    Single-layer perceptron with a step activation and the
    error-driven update rule for {0, 1} labels.
    """

    def __init__(self, n_features, rng, lr=0.01):
        """
        Configure the necessary values and define a initial value for them.
        Non-zero initialization for the weights is required so that eta has an actual effect on the learned boundary.
        """

        self.w = rng.normal(0, 0.01, size=n_features) # weights start small
        self.b = 0.0                                  # bias start as 0

        # leraning rate: when a mistake is observed in training, how much should the perceptron adjust?
        self.lr = lr                                  

    def net_input(self, X):
        """
        features (X) * features relevance (self.w) + bias (self.b) = net input
        """
        return X @ self.w + self.b

    def predict(self, X):
        """
        Use the calculated net input to decide the classification of the observed input (X). Step 
        activation: there is a "step" at 0 that define the classification process.
        - net_input >= 0 → class 1
        - net_input < 0  → class 0
        """

        return (self.net_input(X) >= 0).astype(int)

    def accuracy(self, X, y):
        """
        Calculate the accuracy of the perceptron. Compare if the pŕedicted value is equal to the actual classification
        of the observed input.
        """

        return float(np.mean(self.predict(X) == y))

    def fit(self, X, y, max_epochs=100, track_pocket=False):
        """
        Percepton training.

        Train until a full epoch makes no update, or max_epochs is reached. Returns per-epoch 
        accuracy history (and pocket history if track_pocket=True).
        """

        history = []
        pocket_w, pocket_b, pocket_acc = self.w.copy(), self.b, self.accuracy(X, y)
        pocket_history = [pocket_acc]

        for epoch in range(max_epochs):
            updated = False

            # One pass over the dataset, in the given order.
            for xi, yi in zip(X, y):

                # Try to predict the value of y and calculate the error observed
                y_hat = int(self.net_input(xi) >= 0)
                error = yi - y_hat
                # false positive: error = +1
                # false negative: error = -1

                # If it got wrong
                if error != 0:
                    # Adjust weights and bias based on defined learning rate. The value of the error is responsable
                    # to give the "direction" of the adjustment
                    self.w += self.lr * error * xi
                    self.b += self.lr * error
                    updated = True

                    # Pocket the configuration fi it's the best seeing until now
                    if track_pocket:
                        acc_now = self.accuracy(X, y)
                        if acc_now > pocket_acc:
                            pocket_acc = acc_now
                            pocket_w = self.w.copy()
                            pocket_b = self.b

            acc = self.accuracy(X, y)
            history.append(acc)
            if track_pocket:
                pocket_history.append(pocket_acc)

            # If perceptron didn't learn from the epoch: stop learning process
            if not updated:
                break

        result = {"epochs": epoch + 1, "history": history}
        if track_pocket:
            result["pocket_w"] = pocket_w
            result["pocket_b"] = pocket_b
            result["pocket_history"] = pocket_history
            result["pocket_best_epoch"] = int(np.argmax(pocket_history))
        
        return result


# ============================================================
# Exercise 1 - Separable data
# ============================================================
def generate_separable_data():
    """
    Generate the data that will be separable by the Perceptron. 
    Values are defined on the statement:
        - Means are far apart relative to the spread
        - 1000 samples per class (only two classes)
    """

    mean0, cov0 = [1.5, 1.5], [[0.5, 0], [0, 0.5]]
    mean1, cov1 = [5.0, 5.0], [[0.5, 0], [0, 0.5]]
    X0 = rng.multivariate_normal(mean0, cov0, size=1000)
    X1 = rng.multivariate_normal(mean1, cov1, size=1000)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(1000), np.ones(1000)]).astype(int)
    return X, y


def plot_scatter(X, y, title, path, boundary=None, misclassified=None):
    """
    Function to generate a scatter for the plots in a consistent way throughout the exercises.
    """

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for cls, color, label in [(0, "#3B82C4", "Class 0"), (1, "#B3261E", "Class 1")]:
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], s=18, alpha=0.6, color=color, label=label)

    if misclassified is not None and misclassified.any():
        ax.scatter(
            X[misclassified, 0], X[misclassified, 1],
            s=55, facecolors="none", edgecolors="black", linewidths=1.4,
            label="Misclassified", zorder=5,
        )

    if boundary is not None:
        w, b, style, blabel = boundary
        x_vals = np.array([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5])
        if abs(w[1]) > 1e-9:
            y_vals = -(w[0] * x_vals + b) / w[1]
            ax.plot(x_vals, y_vals, style, linewidth=2, label=blabel, zorder=4)

    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", title="Legend")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def exercise_1():
    # Generate the data
    X, y = generate_separable_data()

    # Figure 1
    plot_scatter(X, y, "[Figure 1] Separable data ($\\eta = 0.01$)", FIG_DIR / "figure1.png")

    # B, C: train with learning rate of 0.01
    model = Perceptron(n_features=2, rng=rng, lr=0.01)
    result = model.fit(X, y, max_epochs=100)
    final_acc = model.accuracy(X, y)
    preds = model.predict(X)
    misclassified = preds != y

    # Figure 2: decision boundary of Perceptron trained with separable data
    plot_scatter(
        X, y, "[Figure 2] Decision boundary ($\\eta = 0.01$)", FIG_DIR / "figure2.png",
        boundary=(model.w, model.b, "k--", "Decision boundary"),
        misclassified=misclassified,
    )

    # Figure 3: accuracy per epoch
    fig, ax = plt.subplots(figsize=(7, 4.3), constrained_layout=True)
    ax.plot(range(1, len(result["history"]) + 1), result["history"], marker="o", color="#B3261E")
    ax.set_title("[Figure 3] Accuracy per epoch ($\\eta = 0.01$)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.2)
    fig.savefig(FIG_DIR / "figure3.png", bbox_inches="tight")
    plt.close(fig)

    # D.2: re-run with bigger learning rate (eta = 1.0)
    model_hi = Perceptron(n_features=2, rng=np.random.default_rng(42), lr=1.0)
    result_hi = model_hi.fit(X, y, max_epochs=100)
    acc_hi = model_hi.accuracy(X, y)
    dir_low = model.w / np.linalg.norm(model.w)
    dir_hi = model_hi.w / np.linalg.norm(model_hi.w)

    return {
        "w": model.w, "b": model.b,
        "epochs": result["epochs"], "accuracy": final_acc,
        "w_dir": dir_low,
        "epochs_hi": result_hi["epochs"], "accuracy_hi": acc_hi,
        "w_hi": model_hi.w, "b_hi": model_hi.b, "w_dir_hi": dir_hi,
    }


# ============================================================
# Exercise 2 - Overlapping data
# ============================================================

def generate_overlapping_data():
    """
        Generate the data that will NOT be separable by the Perceptron. 
        Values are defined on the statement:
            - Means are close together and spread is three times larger
            - 1000 samples per class (only two classes)
    """
     
    mean0, cov0 = [3.0, 3.0], [[1.5, 0], [0, 1.5]]
    mean1, cov1 = [4.0, 4.0], [[1.5, 0], [0, 1.5]]
    X0 = rng.multivariate_normal(mean0, cov0, size=1000)
    X1 = rng.multivariate_normal(mean1, cov1, size=1000)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(1000), np.ones(1000)]).astype(int)
    return X, y


def exercise_2():
    # Generate the data
    X, y = generate_overlapping_data()

    # Figure 4
    plot_scatter(X, y, "[Figure 4] Overlapping data", FIG_DIR / "figure4.png")

    # B - train, keep pocket
    model = Perceptron(n_features=2, rng=rng, lr=0.01)
    result = model.fit(X, y, max_epochs=100, track_pocket=True)

    final_w, final_b = model.w, model.b
    final_acc = model.accuracy(X, y)
    pocket_w, pocket_b = result["pocket_w"], result["pocket_b"]

    pocket_model = Perceptron(n_features=2, rng=rng, lr=0.01)
    pocket_model.w, pocket_model.b = pocket_w, pocket_b
    pocket_acc = pocket_model.accuracy(X, y)

    final_preds = model.predict(X)
    final_mis = final_preds != y
    pocket_preds = pocket_model.predict(X)
    pocket_mis = pocket_preds != y

    # Figure 5: both boundaries
    fig, ax = plt.subplots(figsize=(7.5, 6.3), constrained_layout=True)
    for cls, color, label in [(0, "#3B82C4", "Class 0"), (1, "#B3261E", "Class 1")]:
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], s=14, alpha=0.45, color=color, label=label)

    x_vals = np.array([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5])
    if abs(final_w[1]) > 1e-9:
        ax.plot(x_vals, -(final_w[0]*x_vals+final_b)/final_w[1], "k--", linewidth=2, label="Final boundary")
    if abs(pocket_w[1]) > 1e-9:
        ax.plot(x_vals, -(pocket_w[0]*x_vals+pocket_b)/pocket_w[1], "-", color="darkgreen", linewidth=2.4, label="Pocket boundary")

    both_mis = final_mis & pocket_mis
    ax.scatter(X[both_mis, 0], X[both_mis, 1], s=45, facecolors="none",
               edgecolors="black", linewidths=1.2, label="Misclassified (both)", zorder=5)

    ax.set_title("[Figure 5] Final vs. pocket decision boundaries")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(FIG_DIR / "figure5.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 6: accuracy curves
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    epochs = range(1, len(result["history"]) + 1)
    ax.plot(epochs, result["history"], label="Current-epoch accuracy", color="#3B82C4")
    ax.plot(range(0, len(result["pocket_history"])), result["pocket_history"],
            label="Pocket (best-so-far) accuracy", color="darkgreen", linewidth=2)
    ax.set_title("[Figure 6] Accuracy per epoch (overlapping data)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(FIG_DIR / "figure6.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "final_w": final_w, "final_b": final_b, "final_acc": final_acc,
        "pocket_w": pocket_w, "pocket_b": pocket_b, "pocket_acc": pocket_acc,
        "pocket_best_epoch": result["pocket_best_epoch"],
    }


if __name__ == "__main__":
    print("=== EXERCISE 1 ===")
    ex1 = exercise_1()
    print(f"Final w = {ex1['w']}, b = {ex1['b']:.4f}")
    print(f"Epochs to convergence = {ex1['epochs']}, accuracy = {ex1['accuracy']:.4f}")
    print(f"w / ||w|| (eta=0.01) = {ex1['w_dir']}")
    print(f"eta=1.0 -> epochs = {ex1['epochs_hi']}, accuracy = {ex1['accuracy_hi']:.4f}")
    print(f"w (eta=1.0) = {ex1['w_hi']}, b = {ex1['b_hi']:.4f}")
    print(f"w / ||w|| (eta=1.0) = {ex1['w_dir_hi']}")

    print("\n=== EXERCISE 2 ===")
    ex2 = exercise_2()
    print(f"Final w = {ex2['final_w']}, b = {ex2['final_b']:.4f}, acc = {ex2['final_acc']:.4f}")
    print(f"Pocket w = {ex2['pocket_w']}, b = {ex2['pocket_b']:.4f}, acc = {ex2['pocket_acc']:.4f}")
    print(f"Pocket best found at epoch = {ex2['pocket_best_epoch']}")
