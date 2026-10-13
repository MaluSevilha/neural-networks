---
exercise: mlp
ai_use: "Claude helped structure an earlier draft and check the hand-worked backpropagation; Codex helped audit the current rubric, validate the calculations and results, and revise the code and report. The final implementation and analysis remain my responsibility to review and understand before submission."
---

# 3 · MLP

This report follows one central question: **what does a hidden layer buy us?** It begins with one backpropagation step, then studies a binary problem, extends the same implementation to multiple classes, and finally tests whether adding a second hidden layer helps.

!!! info "Reproducibility"
    The single-seed experiments create one `rng = np.random.default_rng(42)` and reuse it throughout the report. The multi-seed comparison uses seeds 42–46, with a fresh but identically seeded generator for each architecture so that the comparison is fair. The complete implementation for Exercises 2–4 is in [`mlp_activity.py`](code/mlp_activity.py).

???+ "See `mlp_activity.py`"
    ```python
    --8<-- "docs/exercises/mlp/code/mlp_activity.py"
    ```

## Exercise 1

### One Step of Backpropagation

The network has 2 inputs, one hidden layer with 2 neurons, and one output neuron. Both layers use $\tanh$, and the loss for the single sample is

$$
L=(y-\hat y)^2.
$$

The given values are

$$
\mathbf{x}=[0.5,-0.2], \quad y=1.0,
$$

$$
\mathbf{W}^{(1)}=
\begin{bmatrix}
0.3&-0.1\\
0.2&0.4
\end{bmatrix}, \quad
\mathbf{b}^{(1)}=[0.1,-0.2],
$$

$$
\mathbf{W}^{(2)}=[0.5,-0.3], \quad b^{(2)}=0.2, \quad \eta=0.3.
$$

### A — Forward pass

The hidden pre-activations are

$$
\begin{aligned}
\mathbf{z}^{(1)}
&=\mathbf{W}^{(1)}\mathbf{x}+\mathbf{b}^{(1)} \\
&=
\begin{bmatrix}
0.3(0.5)+(-0.1)(-0.2)+0.1\\
0.2(0.5)+0.4(-0.2)-0.2
\end{bmatrix} \\
&=[0.2700,-0.1800].
\end{aligned}
$$

Applying $\tanh$ elementwise gives

$$
\mathbf{h}^{(1)}=\tanh(\mathbf{z}^{(1)})=[0.2636,-0.1781].
$$

The output pre-activation and prediction are

$$
\begin{aligned}
u^{(2)}
&=\mathbf{W}^{(2)}\mathbf{h}^{(1)}+b^{(2)} \\
&=0.5(0.2636)-0.3(-0.1781)+0.2 \\
&=0.3852,
\end{aligned}
$$

$$
\hat y=\tanh(u^{(2)})=0.3672.
$$

Therefore,

$$
L=(1.0-0.3672)^2=0.4004.
$$

### B — Backward pass

Starting at the loss,

$$
\frac{\partial L}{\partial \hat y}
=-2(y-\hat y)
=-2(1.0-0.3672)
=-1.2655.
$$

Since $\frac{d}{du}\tanh(u)=1-\tanh^2(u)$,

$$
\begin{aligned}
\frac{\partial L}{\partial u^{(2)}}
&=\frac{\partial L}{\partial \hat y}\left(1-\hat y^2\right) \\
&=-1.2655(1-0.3672^2) \\
&=-1.0948.
\end{aligned}
$$

For the output layer,

$$
\begin{aligned}
\frac{\partial L}{\partial \mathbf{W}^{(2)}}
&=\frac{\partial L}{\partial u^{(2)}}\mathbf{h}^{(1)} \\
&=-1.0948[0.2636,-0.1781] \\
&=[-0.2886,0.1950],
\end{aligned}
$$

$$
\frac{\partial L}{\partial b^{(2)}}
=\frac{\partial L}{\partial u^{(2)}}
=-1.0948.
$$

Propagating the gradient into the hidden layer,

$$
\begin{aligned}
\frac{\partial L}{\partial \mathbf{h}^{(1)}}
&=\frac{\partial L}{\partial u^{(2)}}\mathbf{W}^{(2)} \\
&=-1.0948[0.5,-0.3] \\
&=[-0.5474,0.3284],
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial L}{\partial \mathbf{z}^{(1)}}
&=\frac{\partial L}{\partial \mathbf{h}^{(1)}}
\odot\left(1-(\mathbf{h}^{(1)})^2\right) \\
&=[-0.5474(1-0.2636^2),\ 0.3284(1-(-0.1781)^2)] \\
&=[-0.5094,0.3180].
\end{aligned}
$$

The hidden-layer gradients are therefore

$$
\begin{aligned}
\frac{\partial L}{\partial \mathbf{W}^{(1)}}
&=\frac{\partial L}{\partial \mathbf{z}^{(1)}}\mathbf{x}^T \\
&=
\begin{bmatrix}
-0.5094(0.5)&-0.5094(-0.2)\\
0.3180(0.5)&0.3180(-0.2)
\end{bmatrix} \\
&=
\begin{bmatrix}
-0.2547&0.1019\\
0.1590&-0.0636
\end{bmatrix},
\end{aligned}
$$

$$
\frac{\partial L}{\partial \mathbf{b}^{(1)}}
=\frac{\partial L}{\partial \mathbf{z}^{(1)}}
=[-0.5094,0.3180].
$$

### C — Parameter update

Gradient descent applies $\theta\leftarrow\theta-\eta\nabla_\theta L$ with $\eta=0.3$. For the output layer,

$$
\begin{aligned}
\mathbf{W}^{(2)}
&\leftarrow[0.5,-0.3]-0.3[-0.2886,0.1950] \\
&=[0.5866,-0.3585],
\end{aligned}
$$

$$
b^{(2)}\leftarrow0.2-0.3(-1.0948)=0.5284.
$$

For the hidden layer,

$$
\begin{aligned}
\mathbf{W}^{(1)}
&\leftarrow
\begin{bmatrix}0.3&-0.1\\0.2&0.4\end{bmatrix}
-0.3
\begin{bmatrix}-0.2547&0.1019\\0.1590&-0.0636\end{bmatrix} \\
&=
\begin{bmatrix}0.3764&-0.1306\\0.1523&0.4191\end{bmatrix},
\end{aligned}
$$

$$
\mathbf{b}^{(1)}
\leftarrow[0.1,-0.2]-0.3[-0.5094,0.3180]
=[0.2528,-0.2954].
$$

A second forward pass with the updated parameters gives $L_{\text{new}}=0.0834$, down from $0.4004$. The loss decreases because the update moves every parameter in the negative-gradient direction, which is the local direction of steepest decrease.

---

## Exercise 2

### Binary Classification a Straight Line Cannot Do

### A — Generate the data

The dataset contains 1,000 two-dimensional samples generated with the required centers $[0,0]$, $[3,3]$, and $[-3,-3]$, `cluster_std=1.2`, and `random_state=42`. The center cluster is Class 0, while the two outer clusters together form Class 1. The data is split into 80% training and 20% test samples with stratification.

![Figure 1: Binary data with Class 1 split across two clusters](figures/figure1.png)

### B — Establish the baseline

A logistic-regression baseline trained on the training set reaches **47.50% test accuracy**. A linear classifier produces only one half-plane per class, so it cannot place both outer Class-1 clusters on the same side without also capturing much of the central Class-0 cluster.

### C — Implement the MLP

The `MLP` class implements the forward pass, binary cross-entropy, all backward-pass gradients, and parameter updates directly with NumPy. Its layer sizes are constructor arguments, allowing the same class and the same `forward` and `train` methods to be reused in the remaining exercises.

The binary architecture is `[2, 8, 1]`: 2 inputs, 8 hidden ReLU units, and one sigmoid output. With a learning rate of 0.1 and 2,000 full-batch epochs, it reaches **92.50% test accuracy**.

![Figure 2: Binary cross-entropy during training](figures/figure2.png)

![Figure 3: MLP decision boundary over the test samples](figures/figure3.png)

### D — Analysis

Figure 3 shows why the MLP succeeds where the baseline fails. Its Class-1 region bends around the central cloud, covering both the upper-right and lower-left clusters while retaining a diagonal central region for Class 0. Those two outer regions cannot be represented by a single straight boundary. Each hidden ReLU unit contributes a linear segment, and their composition forms the piecewise-linear boundary needed for this geometry. This raises test accuracy from 47.50% to 92.50%.

---

## Exercise 3

### Multi-Class, Same Network

### A — Generate the data

The required `make_classification` configuration generates 1,500 samples with 4 informative features, 3 classes, and 2 clusters per class, for a total of 6 clusters. The data uses `class_sep=1.0` and `random_state=42`, followed by the same stratified 80/20 split.

### B — Adapt the network

The output layer now has 3 units. For logits $\mathbf{u}$, softmax produces

$$
p_i=\frac{e^{u_i}}{\sum_j e^{u_j}},
$$

and categorical cross-entropy for one-hot targets is

$$
L=-\sum_i y_i\log p_i.
$$

The softmax Jacobian is

$$
\frac{\partial p_i}{\partial u_k}=p_i(\delta_{ik}-p_k),
$$

where $\delta_{ik}$ is 1 when $i=k$ and 0 otherwise. Applying the chain rule,

$$
\begin{aligned}
\frac{\partial L}{\partial u_k}
&=\sum_i\frac{\partial L}{\partial p_i}
\frac{\partial p_i}{\partial u_k} \\
&=-\sum_i\frac{y_i}{p_i}p_i(\delta_{ik}-p_k) \\
&=-\sum_i y_i\delta_{ik}+p_k\sum_i y_i \\
&=-y_k+p_k,
\end{aligned}
$$

because a one-hot target satisfies $\sum_i y_i=1$. Therefore,

$$
\boxed{\frac{\partial L}{\partial \mathbf{u}}=\mathbf{p}-\mathbf{y}}.
$$

This has the same prediction-minus-target form as sigmoid plus binary cross-entropy. Consequently, the core forward, backward, and update code from Exercise 2 is reused unchanged; only the output activation, loss, output size, and hyperparameters are selected through arguments.

### C — Train and evaluate

The architecture is `[4, 16, 3]`, trained for 2,000 epochs with a learning rate of 0.1. It reaches **85.33% test accuracy**. The logistic-regression baseline on the same split reaches **66.33%**.

![Figure 4: Categorical cross-entropy during training](figures/figure4.png)

![Figure 5: Confusion matrix on the test set](figures/figure5.png)

### D — Analysis

The most confused unordered class pair is **Classes 0 and 1**. The largest individual off-diagonal cell is **true Class 1 predicted as Class 0**, with 10 samples; the reverse direction contains 6 errors. Class 1 is also predicted as Class 2 in 9 cases.

The dataset contains six clusters distributed across only four informative dimensions with moderate separation. A linear softmax classifier can only partition the space with linear surfaces, which produces the 66.33% baseline. The hidden layer combines multiple linear pieces and can allocate different regions to the two clusters belonging to each class, improving accuracy to 85.33%. The confusion matrix demonstrates that some overlap remains without requiring assumptions about the unobserved positions of individual cluster centers.

---

## Exercise 4

### Does a Second Hidden Layer Help?

The comparison uses the same Exercise-3 dataset, split, learning rate, epoch count, and first hidden-layer width. The only architectural change is an additional hidden layer.

### A — Train the deeper network

The deeper architecture is `[4, 16, 8, 3]`, containing two hidden layers. With the same 2,000 epochs and learning rate of 0.1, its single-run test accuracy is **84.00%**.

### B — Compare

![Figure 6: Training-loss comparison between shallow and deep networks](figures/figure6.png)

| Model | Hidden layers | Parameters | Final train loss | Test accuracy |
| --- | ---: | ---: | ---: | ---: |
| Exercise 3: `[4, 16, 3]` | 1 | 131 | 0.3686 | 85.33% |
| Exercise 4: `[4, 16, 8, 3]` | 2 | 243 | 0.2575 | 84.00% |

Across five seeds (42–46), using the same data, split, learning rate, and training budget for both architectures, the shallow network averages **84.40%**, while the deeper network averages **85.20%**. The difference is **+0.80 percentage points** for the deeper model, small relative to the variation across runs. On the main single-seed run, the deeper model instead performs 1.33 percentage points worse.

### C — Analysis

The deeper model obtains a lower training loss but does not produce a reliable test-accuracy gain. With six clusters and four informative features, one hidden layer with 16 ReLU units already has enough piecewise-linear capacity for the useful structure. The second layer increases the parameter count from 131 to 243 and can fit the training set more closely, but it also makes optimization more complex and adds capacity that does not consistently generalize.

The single-run and multi-seed results therefore support the same conclusion: depth does not meaningfully help on this dataset. Additional depth is more likely to pay off when the data contains hierarchical or compositional structure—for example, images in which edges combine into shapes and shapes combine into objects—because successive layers can reuse increasingly abstract features.

---

## Results summary

| # | Quantity | Value |
| ---: | --- | --- |
| 1 | Ex. 1 — $\mathbf{z}^{(1)}$ and $\mathbf{h}^{(1)}$ | $\mathbf{z}^{(1)}=[0.2700,-0.1800]$, $\mathbf{h}^{(1)}=[0.2636,-0.1781]$ |
| 2 | Ex. 1 — $u^{(2)}$, $\hat y$, and $L$ | $u^{(2)}=0.3852$, $\hat y=0.3672$, $L=0.4004$ |
| 3 | Ex. 1 — all updated parameters | $\mathbf{W}^{(2)}=[0.5866,-0.3585]$, $b^{(2)}=0.5284$, $\mathbf{W}^{(1)}=\begin{bmatrix}0.3764&-0.1306\\0.1523&0.4191\end{bmatrix}$, $\mathbf{b}^{(1)}=[0.2528,-0.2954]$ |
| 4 | Ex. 2 — linear baseline test accuracy | 47.50% |
| 5 | Ex. 2 — architecture and test accuracy | `[2, 8, 1]`, 92.50% |
| 6 | Ex. 3 — architecture and test accuracy | `[4, 16, 3]`, 85.33% |
| 7 | Ex. 3 — most-confused class pair | Classes 0–1; largest direction: true 1 → predicted 0 (10 samples) |
| 8 | Ex. 4 — architecture and test accuracy | `[4, 16, 8, 3]`, 84.00% |
| 9 | Ex. 4 — accuracy difference against Ex. 3 | +0.80 pp over five seeds (85.20% versus 84.40%) |
