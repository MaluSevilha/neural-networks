---
exercise: data
ai_use: "Claude used to help structure the report and write plotting; the data-generation logic, the geometric measures and the written analysis are mine, reviewed and understood in full"
---

# 1 · Data

This report studies how the geometry, spread, dimensionality, and preprocessing of data affect the structure and difficulty of a classification problem. It follows one central idea: before choosing a neural-network architecture, it is necessary to understand the distribution of the data that the model will receive.

!!! info "Reproducibility"
    The entire activity uses one random-number generator, initialized exactly once with `rng = np.random.default_rng(42)`. The same `rng` instance is reused throughout the report, guaranteeing reproducibility. The complete implementation is available in [`data_activity.py`](code/data_activity.py).

???+ "See `data_activity.py`"
    ```python
    --8<-- "docs/exercises/data/code/data_activity.py"
    ```

## Exercise 1

### Point Clouds: Geometry and Spread in 2D

Understanding how data is distributed is the first step before designing a network architecture. In this exercise the idea is to generate and measure two-dimensional point clouds, observing how the distribution affects the complexity of the decision boundaries a neural network would need to learn.

### A — Generate the clouds

Four Gaussian Synthetic Point Clouds were generated in two dimensions, with 100 samples per class. The means and standard deviations follow the given configuration:

| Class | Mean | Standard deviation |
|---|---|---|
| 0 | (2, 3) | (0.8, 2.5) |
| 1 | (5, 6) | (1.2, 1.9) |
| 2 | (8, 1) | (0.9, 0.9) |
| 3 | (15, 4) | (0.5, 2.0) |

The points are generated independently for each coordinate according to the corresponding Gaussian distribution. With that definition, the following plot can be presented with the generated points, segregated by their classes:

![Figure 1: Original point clouds with class centers and sketched boundaries](figures/figure1.png)

On Figure 1, the `X` markers indicate the class centers.

The implementation used for the four classes is:

```python
rng = np.random.default_rng(42)

for k, (mu, sigma) in enumerate(zip(MEANS, STDS)):
    points = rng.normal(
        loc=mu,
        scale=sigma,
        size=(100, 2),
    )
```

### B — More or less spread out

To isolate the effect of dispersion, the same four class means were kept fixed while every standard deviation was multiplied by a scale factor $s$. This way, the menas never change, but the standard deviation does, changing how spread each cloud will be around it's center. The values for the scale factors are:

<p style="display: flex; align-items: center; justify-content: center; gap: 0.3rem;">
  <var>s</var> &isin; {0.5, 1.0, 2.0, 4.0}
</p>

This creates four independent datasets drived by a respective value of $s$, each containing the same four classes with 100 points per class.

![Figure 2: Effect of increasing the spread](figures/figure2.png)

All four subplots use the same axis limits, so the visual comparison reflects only the change in dispersion honestly.

For $s=1$, the separation ratio was calculated as

$$
r_{ij}=\frac{\lVert\mu_i-\mu_j\rVert}{\bar\sigma_i+\bar\sigma_j},
\qquad
\bar\sigma_k=\frac{\sigma_{k,x}+\sigma_{k,y}}{2}.
$$

After applying the formulas, the six pairwise ratios are:

| Pair | Center distance | $r_{ij}$ |
| --- | ---: | ---: |
| $(0,1)$ | 4.2426 | **1.3258** |
| $(0,2)$ | 6.3246 | 2.4802 |
| $(0,3)$ | 13.0384 | 4.4960 |
| $(1,2)$ | 5.8310 | 2.3800 |
| $(1,3)$ | 10.1980 | 3.6422 |
| $(2,3)$ | 7.6158 | 3.5422 |

The smallest ratio is therefore obtained by **classes 0 and 1**, with $r_{01}=1.3258$. Because the means stay fixed while all standard deviations are multiplied by $s$, every separation ratio scales as $1/s$. Consequently,

$$
r_{01}(s=2)=\frac{1.3258}{2}\approx\mathbf{0.6629}.
$$

The **mixing rate** is defined geometrically as the fraction of points whose nearest class center is not the center of their own class. Using the specified class means as the reference centers gives, we have the following calculated values for the metric:

| Spread factor $s$ | Mixing rate |
| ---: | ---: |
| 0.5 | **0.25%** |
| 1.0 | **7.25%** |
| 2.0 | **19.25%** |
| 4.0 | **48.25%** |

![Figure 3: Mixing rate as a function of spread](figures/figure3.png)

The geometric mixing rate increases monotonically as the standard deviations grow, demonstrating a clear trend. Consequently, increasing the spread does not shift the class centers, but rather expands the volume of the regions occupied by each class. As a result, points from different classes increasingly overlap within each other's neighborhoods (thus causing an increase in the mixing rate).

### C — Analysis {: #exercise-1-analysis}

At $s=1$, the four classes are still organized around distinct centers, with classes 0 and 1 already showing the most relevant overlap. A set of straight boundaries can approximate the partition of the four cluster centers, but perfect separation is already lost in the generated sample: the mixing rate is 7.25%. In other words, **there is no set of linear boundaries that can classify every sampled point correctly at $s = 1$**.

**The effect becomes substantially stronger at $s=2$**. The smallest separation ratio falls from 1.3258 to approximately 0.6629, while the mixing rate increases to 19.25%. At this point, the class regions around the first three centers overlap enough that **straight boundaries are no longer a convincing description** of the class structure. At $s=4$, almost half of the points are geometrically assigned to a different nearest class center, indicating severe mixing.

On the opposite side, a contrary effect can be seen as the $s$ factor is decreased. At $s=0.5$, the class distributions become considerably more compact around their respective centers, reducing the mixing rate to only 0.25%. This indicates that almost all sampled points remain closest to their own class center, making all the classes clearly separated and allowing **straight boundaries to provide a much more accurate description of the class structure**.

The key lesson from this exercise is that **spread is not merely a visualization property**. It directly changes the overlap between classes and therefore changes the complexity and difficulty of the decision problem that a neural network has to learn.

---

## Exercise 2

### A — Dataset I: shifted Gaussians

Dataset I contains 500 samples from Class A and 500 from Class B in five dimensions. The classes have different mean vectors and different covariance matrices, so both their centers and their shapes contribute to the classification problem.

Class A uses

$$
\mu_A=[0,0,0,0,0],
\qquad
\Sigma_A = \begin{pmatrix}
    1.0 & 0.8 & 0.1 & 0.0 & 0.0 \\ 
    0.8 & 1.0 & 0.3 & 0.0 & 0.0 \\ 
    0.1 & 0.3 & 1.0 & 0.5 & 0.0 \\
    0.0 & 0.0 & 0.5 & 1.0 & 0.2 \\
    0.0 & 0.0 & 0.0 & 0.2 & 1.0
\end{pmatrix}
$$


Meanwhile, class B uses

$$
\mu_B=[1.5,1.5,1.5,1.5,1.5],
\qquad
\Sigma_B = \begin{pmatrix}
    1.5  & -0.7 & 0.2 & 0.0 & 0.0 \\ 
    -0.7 &  1.5 & 0.4 & 0.0 & 0.0 \\ 
    0.2  &  0.4 & 1.5 & 0.6 & 0.0 \\
    0.0  &  0.0 & 0.6 & 1.5 & 0.3 \\
    0.0  &  0.0 & 0.0 & 0.3 & 1.5
\end{pmatrix}
$$

Given those definitions, the Euclidean distance between the two class centers is

$$
\lVert\mu_A-\mu_B\rVert = \mathbf{3.3541}.
$$

This nonzero displacement gives the classes a clear directional difference in the original five-dimensional space.

### B — Dataset II: concentric shells

Dataset II also contains 500 samples per class, but the geometry is different. For every sample, a five-dimensional direction is generated from a standard Gaussian vector and normalized to unit length. The radius is then drawn independently for the two classes:

- Class C: $\rho\sim\mathcal N(2.0,0.4)$
- Class D: $\rho\sim\mathcal N(5.0,0.4)$

The final point is

$$
 x=\rho \cdot u,
$$

where $u$ is a unit vector. The result is **an inner cloud surrounded by an outer shell**.

Because the directions are approximately symmetric around the origin, the class centers are both close to zero. In this realization, their distance is only

$$
\lVert\mu_C-\mu_D\rVert=\mathbf{0.2215}.
$$

This is the central contrast of the exercise: the centers are almost coincident even though the classes are well separated by radius.

### C — Visualize and compare

Since the datasets live in five dimensions, PCA was used to obtain a two-dimensional linear projection.

![Figure 4 — PCA projections](figures/figure4.png)

Dataset I retains a visible directional separation, whereas the two shell classes in Dataset II appear heavily mixed after projection.

The first two principal components explain:

| Dataset | PC1 + PC2 explained variance |
| --- | ---: |
| Dataset I | **67.35%** |
| Dataset II | **42.27%** |

The 2D projection therefore preserves more of the overall variance of Dataset I. More importantly, in Dataset I this preserved variance contains information related to the displacement between the class centers. In Dataset II, the classification-relevant information is radial and is not preserved as a simple separation in the first two principal components.

The relevant five-dimensional center distances are:

| Dataset | Center distance |
| --- | ---: |
| I - A vs B | **3.3541** |
| II - C vs D | **0.2215** |

The radius distributions make the hidden structure of Dataset II explicit:

![Figure 5 — Radius distributions](figures/figure5.png)

The shell classes are strongly separated by $\lVert x\rVert$, even though their centers are nearly identical.

### D — Analysis

Dataset II demonstrates why the distance between class centers is not sufficient to characterize a classification problem. Its centers are almost coincident, but the two classes occupy different radial regions. A hyperplane cannot isolate an inner ball from a surrounding shell: any straight boundary that keeps the origin on one side necessarily cuts through the surrounding directions, because the shell extends around the core in every direction.

This means that **no linear boundary can perfectly separate the two classes**, regardless of the number of samples available. More data would estimate the same nonlinear geometry more accurately, but it would not change the fact that the required boundary is curved.

The poor-looking PCA projection does not prove that Dataset II is inseparable in the original space. PCA is a linear dimensionality-reduction method that keeps directions of greatest variance; it does not guarantee preservation of the feature that best separates classes. Here, the relevant feature is radial distance.

A simple function that separates the two shell classes is

$$
 f(x)=\lVert x\rVert^2=\sum_{i=1}^{5}x_i^2.
$$

A threshold $t$ between the two radius ranges can then be used to classify the samples:

$$
\hat y =
\begin{cases}
C, & f(x) < t,\\
D, & f(x) \ge t.
\end{cases}
$$

Thus, the original space contains a simple nonlinear rule even though a two-dimensional PCA projection can make the classes look almost indistinguishable.

---

## Exercise 3

### A — Get to know the data

The real-world dataset is the **Spaceship Titanic** training set from Kaggle. The target `Transported` indicates whether a passenger was transported to another dimension. Here, the proposal is that a Spaceship Titanic collision occured in which some of the pessangers were transported to an alternate dimension. That's the **target** the neural network would need to predict.

The original training set contains 8,693 labeled passengers, with 4,378 transported and 4,315 not transported, corresponding to a positive-class share of **50.36%**. The dataset is, therefore, close to balanced.

The features can be divided into numerical and categorical variables, as shown in the table below.

| Type | Features |
| --- | --- |
| Numerical | `Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` |
| Categorical | `HomePlanet`, `CryoSleep`, `Cabin`, `Destination`, `VIP`, `Name`, `PassengerId` |
| Target | `Transported` |

`Cabin`, `Name`, and `PassengerId` are removed during preprocessing because they are identifiers or high-cardinality fields rather than direct numerical predictors.

Missing values are present in almost every column, at a fairly uniform rate around 2%:

| Column | Missing (count) | Missing (%) |
| --- | ---: | ---: |
| PassengerId | 0 | 0.00% |
| HomePlanet | 201 | 2.31% |
| CryoSleep | 217 | 2.50% |
| Cabin | 199 | 2.29% |
| Destination | 182 | 2.09% |
| Age | 179 | 2.06% |
| VIP | 203 | 2.34% |
| RoomService | 181 | 2.08% |
| FoodCourt | 183 | 2.11% |
| ShoppingMall | 208 | 2.39% |
| Spa | 183 | 2.11% |
| VRDeck | 188 | 2.16% |
| Name | 200 | 2.30% |

The spending variables are strongly right-skewed. The full results can be seen on the table below.

| Column | Mean | Median | Maximum |
| --- | ---: | ---: | ---: |
| RoomService | 224.69 | 0.0 | 14327.0 |
| FoodCourt | 458.08 | 0.0 | 29813.0 |
| ShoppingMall | 173.73 | 0.0 | 23492.0 |
| Spa | 311.14 | 0.0 | 22408.0 |
| VRDeck | 304.85 | 0.0 | 24133.0 |


On the full labeled dataset, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa` and `VRDeck` all have a **median of 0.00** with **maximums** as high as 29,813 (`FoodCourt`). The fact that the median sits at zero while the mean is in the hundreds means that the **majority** of passengers spend nothing at all, and a relatively small number of high-spending passengers pull the mean far above the median: a textbook heavy-tailed, right-skewed distribution.

### B — Split before you transform

The dataset is split into 80% training and 20% test data **before** fitting any imputer, encoder, or scaler. The split is stratified by `Transported` and uses the same global random-number generator used in the report.

This ordering prevents **data leakage**. If a median, standard deviation, or observed category were calculated using the complete dataset, information from the test set would influence the transformation applied to the training data. The resulting preprocessing would therefore indirectly expose information about data that should remain unseen until evaluation.

The split is implemented manually so that the same `rng` instance is reused throughout the entire activity:

```python
def stratified_split(X, y, test_size=0.2):
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        idx = rng.permutation(idx)
        n_test = int(round(test_size * len(idx)))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return rng.permutation(train_idx), rng.permutation(test_idx)
```

### C — Preprocess

The numerical variables are imputed with the **training-set median**, which is more robust than the mean for strongly skewed spending variables. The categorical variables are filled with their most frequent training category.

The four categorical predictors are **one-hot encoded**. The encoder uses `handle_unknown="ignore"`, so a category that appears in the test set but was not observed during training does not cause an error; it simply contributes zeros to the corresponding one-hot representation.

The five spending variables are transformed with

$$
 x' = \log(1+x),
$$

which compresses the extreme right tail while preserving zeros. This produces a much more manageable numerical range for a network with `tanh` activations.

After the logarithmic transformation, the engineered `TotalSpend` feature is computed as the sum of the five transformed spending variables. The fields `Cabin`, `Name`, and `PassengerId` are then removed.

Finally, the numerical variables are standardized using the training data only. Standardization was chosen because it produces centered, unit-scale numerical inputs while avoiding the dependence of MinMax scaling on the particular sample extremes. The categorical one-hot features remain in the $[0,1]$ range.

The full preprocessing implementation is available in [`data_activity.py`](code/data_activity.py). The important training-only fitting steps are:

```python
medians = X_train[numeric_base].median()
X_train[numeric_base] = X_train[numeric_base].fillna(medians)
X_test[numeric_base] = X_test[numeric_base].fillna(medians)

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
)
train_cat = encoder.fit_transform(X_train[CATEGORICAL])
test_cat = encoder.transform(X_test[CATEGORICAL])

scaler = StandardScaler()
train_num = scaler.fit_transform(X_train[NUMERICAL])
test_num = scaler.transform(X_test[NUMERICAL])
```

### D — Verify and visualize

Figure 6 shows the effect of the logarithmic transformation on `FoodCourt`, computed on the actual training split:

![Figure 6 — Effect of the log transformation on `FoodCourt`](figures/figure6.png)

The raw training distribution (left) is dominated by a single spike at 0 with a long right tail stretching past 25,000. After `log(1 + FoodCourt)` (right), that same spike at 0 remains exactly at 0 (since $\log(1+0)=0$), while the non-zero spenders are spread out over a much more compact, roughly bell-shaped range (the compression a `tanh`-based network benefits from).

The pipeline was run end-to-end on the real Kaggle `train.csv` (8,693 rows, stratified 80/20 split). It satisfies three final conditions: no remaining `NaN` values in either split, a finite final feature matrix, and scaled numerical inputs centered around a range appropriate for a `tanh`-based network. The training split's positive-class share is **50.36%**, confirming the dataset stayed balanced after the split.

The preprocessing decision expected to have the largest impact on training is the combination of **heavy-tail compression and scaling**. Without these steps, a few very large spending values would dominate the numerical scale of the inputs, forcing the network to operate in a poorly conditioned region and potentially pushing `tanh` units toward saturation. The imputation and categorical encoding are also necessary for a complete numerical matrix, but the log transformation and scaling directly affect the numerical geometry seen by every layer.

---

## Results Summary

The following table collects the numerical results discussed above, all produced by the single reproducible run of [`data_activity.py`](code/data_activity.py) against the real Kaggle `train.csv`.

| # | Item | Result observed |
| ---: | --- | ---: |
| 1 | Mixing rate at $s=0.5$ | **0.25%** |
| 2 | Mixing rate at $s=1.0$ | **7.25%** |
| 3 | Mixing rate at $s=2.0$ | **19.25%** |
| 4 | Mixing rate at $s=4.0$ | **48.25%** |
| 5 | Smallest $r_{ij}$ at $s=1.0$, and pair | **1.3258 · classes (0, 1)** |
| 6 | Distance between centers — Dataset I | **3.3541** |
| 7 | Distance between centers — Dataset II | **0.2215** |
| 8 | Explained variance PC1 + PC2 — Dataset I | **67.35%** |
| 9 | Explained variance PC1 + PC2 — Dataset II | **42.27%** |
| 10 | Share of the positive class in `Transported` | **50.36%** (training split) |
| 11 | Mean and median of `FoodCourt` on the training set, before transforming | **mean = 462.14 · median = 0.00** |
| 12 | Final shape of the training feature matrix | **(6954, 17)** |
| 13 | Minimum and maximum of the training and test sets after scaling | **train: [−2.0120, 12.2615] · test: [−2.0120, 10.5175]** |
