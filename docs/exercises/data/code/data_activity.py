from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# One RNG is reused throughout the whole report, as required by the activity.
rng = np.random.default_rng(42)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "assets" / "figures"
DATA_DIR = ROOT / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 180,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.size": 10,
})

# ============================================================
# Exercise 1 - Point clouds in 2D
# ============================================================
MEANS = np.array([
    [2.0, 3.0],
    [5.0, 6.0],
    [8.0, 1.0],
    [15.0, 4.0],
])
STDS = np.array([
    [0.8, 2.5],
    [1.2, 1.9],
    [0.9, 0.9],
    [0.5, 2.0],
])
N_PER_CLASS = 100
SCALES = np.array([0.5, 1.0, 2.0, 4.0])


def generate_2d_clouds():
    """Generate the four datasets, using the same class means at every scale."""
    datasets = {}
    for s in SCALES:
        classes, labels = [], []
        for class_id, (mu, sigma) in enumerate(zip(MEANS, STDS)):
            points = rng.normal(loc=mu, scale=s * sigma, size=(N_PER_CLASS, 2))
            classes.append(points)
            labels.extend([class_id] * N_PER_CLASS)
        datasets[float(s)] = (np.vstack(classes), np.asarray(labels))
    return datasets


def separation_ratios():
    """Compute r_ij for all six class pairs at s = 1."""
    sigma_bar = STDS.mean(axis=1)
    rows = []
    for i in range(4):
        for j in range(i + 1, 4):
            distance = np.linalg.norm(MEANS[i] - MEANS[j])
            ratio = distance / (sigma_bar[i] + sigma_bar[j])
            rows.append((i, j, distance, ratio))
    return pd.DataFrame(
        rows,
        columns=["Class i", "Class j", "Center distance", "r_ij"],
    )


def mixing_rate(points, labels):
    """Fraction of samples whose nearest class center has the wrong label."""
    distances = np.linalg.norm(
        points[:, None, :] - MEANS[None, :, :], axis=2
    )
    nearest_center = distances.argmin(axis=1)
    return float(np.mean(nearest_center != labels))


def exercise_1():
    datasets = generate_2d_clouds()

    # Figure 1: original point cloud at s = 1.
    # The dashed lines are an illustrative sketch, not a fitted classifier.
    points, labels = datasets[1.0]
    fig, ax = plt.subplots(figsize=(10, 6.1), constrained_layout=True)

    for class_id in range(4):
        mask = labels == class_id
        ax.scatter(
            points[mask, 0], points[mask, 1],
            s=30, alpha=0.62, label=f"Class {class_id}"
        )
        ax.scatter(
            *MEANS[class_id], marker="X", s=155,
            edgecolor="black", linewidth=1.1, zorder=5
        )
        ax.annotate(
            rf"$\mu_{class_id}$", MEANS[class_id],
            xytext=(7, 7), textcoords="offset points",
            fontsize=10, weight="bold"
        )

    boundary_specs = [
        (np.linspace(1.2, 4.25, 100), lambda x: -x + 8.0),
        (np.linspace(4.8, 8.9, 100), lambda x: 0.6 * x - 0.4),
        (
            np.linspace(9.4, 12.9, 100),
            lambda x: -(7 / 3) * (x - 11.5) + 2.5,
        ),
    ]
    for idx, (x, boundary) in enumerate(boundary_specs):
        ax.plot(
            x, boundary(x), "--", linewidth=1.8,
            color="black",
            label="illustrative decision boundaries" if idx == 0 else None,
            zorder=4,
        )

    ax.set_title("Figure 1: Original point clouds ($s = 1$)")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(-0.8, 17.0)
    ax.set_ylim(-6.8, 12.1)
    ax.grid(alpha=0.18)
    ax.legend(
        title="Class", loc="upper left",
        bbox_to_anchor=(1.01, 1.0), frameon=True
    )
    fig.savefig(FIG_DIR / "figure1.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: common axis limits make the changes in spread comparable.
    all_points = np.vstack([dataset[0] for dataset in datasets.values()])
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    x_pad = 0.05 * (x_max - x_min)
    y_pad = 0.05 * (y_max - y_min)

    fig, axes = plt.subplots(
        2, 2, figsize=(10.8, 8.2),
        sharex=True, sharey=True, constrained_layout=True
    )
    for ax, s in zip(axes.flat, SCALES):
        pts, lab = datasets[float(s)]
        for class_id in range(4):
            mask = lab == class_id
            ax.scatter(
                pts[mask, 0], pts[mask, 1],
                s=17, alpha=0.55, label=f"Class {class_id}"
            )
        ax.set_title(f"$s = {s:g}$")
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.grid(alpha=0.18)

    for ax in axes[:, 0]:
        ax.set_ylabel("$x_2$")
    for ax in axes[-1, :]:
        ax.set_xlabel("$x_1$")
    axes[0, 0].legend(title="Class", fontsize=8)
    fig.suptitle("Figure 2: Effect of increasing the spread", fontsize=15)
    fig.savefig(FIG_DIR / "figure2.png", bbox_inches="tight")
    plt.close(fig)

    ratios = separation_ratios()
    mixing = pd.DataFrame({
        "s": SCALES,
        "mixing_rate": [
            mixing_rate(datasets[float(s)][0], datasets[float(s)][1])
            for s in SCALES
        ],
    })

    # Figure 3: mixing rate as a function of the spread factor.
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    ax.plot(mixing["s"], mixing["mixing_rate"], marker="o", linewidth=2)
    for _, row in mixing.iterrows():
        ax.annotate(
            f"{row['mixing_rate']:.1%}",
            (row["s"], row["mixing_rate"]),
            xytext=(0, 9), textcoords="offset points", ha="center"
        )
    ax.set_title("Figure 3: Mixing rate as a function of spread")
    ax.set_xlabel("Spread scale factor $s$")
    ax.set_ylabel("Mixing rate")
    ax.set_xticks(SCALES)
    ax.set_ylim(-0.03, 0.53)
    ax.grid(alpha=0.2)
    fig.savefig(FIG_DIR / "figure3.png", bbox_inches="tight")
    plt.close(fig)

    return ratios, mixing


# ============================================================
# Exercise 2 - Non-linearity in higher dimensions
# ============================================================
MU_A = np.zeros(5)
SIGMA_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])
MU_B = np.full(5, 1.5)
SIGMA_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])
N_5D = 500


def generate_5d_datasets():
    A = rng.multivariate_normal(MU_A, SIGMA_A, size=N_5D)
    B = rng.multivariate_normal(MU_B, SIGMA_B, size=N_5D)

    v_c = rng.normal(size=(N_5D, 5))
    u_c = v_c / np.linalg.norm(v_c, axis=1, keepdims=True)
    rho_c = rng.normal(2.0, 0.4, size=N_5D)
    C = rho_c[:, None] * u_c

    v_d = rng.normal(size=(N_5D, 5))
    u_d = v_d / np.linalg.norm(v_d, axis=1, keepdims=True)
    rho_d = rng.normal(5.0, 0.4, size=N_5D)
    D = rho_d[:, None] * u_d
    return A, B, C, D


def exercise_2():
    A, B, C, D = generate_5d_datasets()

    AB = np.vstack([A, B])
    CD = np.vstack([C, D])
    labels_ab = np.array(["A"] * N_5D + ["B"] * N_5D)
    labels_cd = np.array(["C"] * N_5D + ["D"] * N_5D)

    pca_ab = PCA(n_components=2)
    pca_cd = PCA(n_components=2)
    AB_2d = pca_ab.fit_transform(AB)
    CD_2d = pca_cd.fit_transform(CD)

    # Figure 4: PCA projections.
    fig, axes = plt.subplots(
        1, 2, figsize=(11.2, 5.0), constrained_layout=True
    )
    for cls in ["A", "B"]:
        mask = labels_ab == cls
        axes[0].scatter(
            AB_2d[mask, 0], AB_2d[mask, 1],
            s=15, alpha=0.5, label=f"Class {cls}"
        )
    for cls in ["C", "D"]:
        mask = labels_cd == cls
        axes[1].scatter(
            CD_2d[mask, 0], CD_2d[mask, 1],
            s=15, alpha=0.5, label=f"Class {cls}"
        )

    axes[0].set_title("Dataset I: shifted Gaussians")
    axes[1].set_title("Dataset II: concentric shells")
    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(title="Class")
        ax.grid(alpha=0.18)
    fig.suptitle("Figure 4: PCA projections into two dimensions", fontsize=15)
    fig.savefig(FIG_DIR / "figure4.png", bbox_inches="tight")
    plt.close(fig)

    radii_A = np.linalg.norm(A, axis=1)
    radii_B = np.linalg.norm(B, axis=1)
    radii_C = np.linalg.norm(C, axis=1)
    radii_D = np.linalg.norm(D, axis=1)

    # Figure 5: radius distributions.
    fig, axes = plt.subplots(
        1, 2, figsize=(11.2, 5.0), constrained_layout=True
    )
    axes[0].hist(radii_A, bins=28, alpha=0.55, label="Class A")
    axes[0].hist(radii_B, bins=28, alpha=0.55, label="Class B")
    axes[1].hist(radii_C, bins=28, alpha=0.55, label="Class C")
    axes[1].hist(radii_D, bins=28, alpha=0.55, label="Class D")
    axes[0].set_title("Dataset I")
    axes[1].set_title("Dataset II")
    for ax in axes:
        ax.set_xlabel("Radius ||x||")
        ax.set_ylabel("Count")
        ax.legend(title="Class")
        ax.grid(alpha=0.18)
    fig.suptitle("Figure 5: Radius distributions in 5D", fontsize=15)
    fig.savefig(FIG_DIR / "figure5.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "distance_ab": float(np.linalg.norm(MU_A - MU_B)),
        "distance_cd": float(np.linalg.norm(C.mean(axis=0) - D.mean(axis=0))),
        "pca_ab": float(pca_ab.explained_variance_ratio_.sum()),
        "pca_cd": float(pca_cd.explained_variance_ratio_.sum()),
        "radius_summary": {
            "A_mean": float(radii_A.mean()),
            "B_mean": float(radii_B.mean()),
            "C_mean": float(radii_C.mean()),
            "D_mean": float(radii_D.mean()),
            "C_median": float(np.median(radii_C)),
            "D_median": float(np.median(radii_D)),
        },
    }


# ============================================================
# Exercise 3 - Real-world data
# ============================================================
SPEND_COLS = [
    "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"
]
CATEGORICAL = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
NUMERICAL = ["Age"] + SPEND_COLS + ["TotalSpend"]
DROP_COLS = ["Cabin", "Name", "PassengerId"]


def stratified_split(y, test_size=0.2):
    """80/20 stratified split using the one global RNG."""
    y = np.asarray(y)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        idx = rng.permutation(idx)
        n_test = int(round(test_size * len(idx)))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return rng.permutation(train_idx), rng.permutation(test_idx)


def run_exercise_3(train_path=None):
    if train_path is None:
        train_path = DATA_DIR / "train.csv"
    train_path = Path(train_path)
    if not train_path.exists():
        raise FileNotFoundError(
            f"Place Kaggle train.csv at {train_path} before running Exercise 3."
        )

    df = pd.read_csv(train_path)
    y = df["Transported"].astype(bool)
    X = df.drop(columns=["Transported"]).copy()

    missing = pd.DataFrame({
        "Missing": X.isna().sum(),
        "Percentage": 100 * X.isna().mean(),
    })

    spending_stats = pd.DataFrame({
        "Mean": X[SPEND_COLS].mean(),
        "Median": X[SPEND_COLS].median(),
        "Maximum": X[SPEND_COLS].max(),
    })

    train_idx, test_idx = stratified_split(y.to_numpy(), test_size=0.2)
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    # Numerical imputation: training medians only.
    numeric_base = ["Age"] + SPEND_COLS
    medians = X_train[numeric_base].median()
    X_train[numeric_base] = X_train[numeric_base].fillna(medians)
    X_test[numeric_base] = X_test[numeric_base].fillna(medians)

    # Categorical imputation: training modes only.
    modes = {
        col: X_train[col].mode(dropna=True).iloc[0]
        for col in CATEGORICAL
    }
    for col in CATEGORICAL:
        # Convert to pandas' string dtype first; this avoids pandas' deprecated
        # silent downcasting warning when filling object columns.
        train_col = X_train[col].astype("string")
        test_col = X_test[col].astype("string")
        mode_value = str(modes[col])
        X_train[col] = train_col.fillna(mode_value)
        X_test[col] = test_col.fillna(mode_value)

    # Feature engineering is performed after imputation, so TotalSpend has no NaNs.
    X_train["TotalSpend"] = X_train[SPEND_COLS].sum(axis=1)
    X_test["TotalSpend"] = X_test[SPEND_COLS].sum(axis=1)

    # Preserve the pre-log FoodCourt distribution for Figure 6.
    foodcourt_before = X_train["FoodCourt"].copy()

    # Heavy tails: log(1+x) on the five spending variables.
    for col in SPEND_COLS:
        X_train[col] = np.log1p(X_train[col])
        X_test[col] = np.log1p(X_test[col])

    # Drop identifiers / fields requested by the statement.
    X_train = X_train.drop(columns=DROP_COLS)
    X_test = X_test.drop(columns=DROP_COLS)

    # Unknown test categories are ignored rather than causing a transform error.
    encoder = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False, dtype=float
    )
    train_cat = encoder.fit_transform(X_train[CATEGORICAL])
    test_cat = encoder.transform(X_test[CATEGORICAL])

    # Standardization is fitted only on training data.
    scaler = StandardScaler()
    train_num = scaler.fit_transform(X_train[NUMERICAL])
    test_num = scaler.transform(X_test[NUMERICAL])

    X_train_final = np.hstack([train_num, train_cat])
    X_test_final = np.hstack([test_num, test_cat])

    # Figure 6: heavy-tailed feature before and after log1p.
    foodcourt_after = X_train["FoodCourt"]
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 4.3), constrained_layout=True
    )
    axes[0].hist(foodcourt_before, bins=40, alpha=0.75)
    axes[0].set_title("Before: FoodCourt")
    axes[0].set_xlabel("FoodCourt")
    axes[0].set_ylabel("Count")
    axes[1].hist(foodcourt_after, bins=40, alpha=0.75)
    axes[1].set_title("After: log(1 + FoodCourt)")
    axes[1].set_xlabel("log(1 + FoodCourt)")
    axes[1].set_ylabel("Count")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Figure 6: Effect of the log transformation", fontsize=15)
    fig.savefig(FIG_DIR / "figure6.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "positive_share": float(y.mean()),
        "train_positive_share": float(y_train.mean()),
        "test_positive_share": float(y_test.mean()),
        "missing": missing,
        "spending_stats": spending_stats,
        "foodcourt_train_mean": float(X.iloc[train_idx]["FoodCourt"].mean()),
        "foodcourt_train_median": float(X.iloc[train_idx]["FoodCourt"].median()),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "shape_train": X_train_final.shape,
        "shape_test": X_test_final.shape,
        "train_range": (float(X_train_final.min()), float(X_train_final.max())),
        "test_range": (float(X_test_final.min()), float(X_test_final.max())),
        "no_nan_train": bool(np.isnan(X_train_final).sum() == 0),
        "no_nan_test": bool(np.isnan(X_test_final).sum() == 0),
    }


if __name__ == "__main__":
    ratios, mixing = exercise_1()
    stats = exercise_2()

    print("EXERCISE 1: separation ratios")
    print(ratios.to_string(index=False))
    print("\nEXERCISE 1: mixing rates")
    print(mixing.to_string(index=False))

    print("\nEXERCISE 2")
    print(stats)

    train_file = DATA_DIR / "train.csv"
    if train_file.exists():
        real = run_exercise_3(train_file)
        print("\nEXERCISE 3")
        print("positive_share", real["positive_share"])
        print("FoodCourt train mean/median", real["foodcourt_train_mean"], real["foodcourt_train_median"])
        print("shape_train", real["shape_train"])
        print("shape_test", real["shape_test"])
        print("train_range", real["train_range"])
        print("test_range", real["test_range"])
        print("no_nan_train", real["no_nan_train"])
        print("no_nan_test", real["no_nan_test"])
        print("\nMissing-value table")
        print(real["missing"].to_string())
        print("\nSpending statistics")
        print(real["spending_stats"].to_string())
    else:
        print(f"\nEXERCISE 3 skipped: place train.csv at {train_file}")
