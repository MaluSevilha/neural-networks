# 01 · Data
Artificial Neural Networks and Deep Learning · Insper · 2026.2

This report studies how the geometry, distribution, dimensionality, and preprocessing of **data** affect the structure of a classification problem.

## Report structure

- **Exercise 1 - Point Clouds:** geometry and spread in 2D
- **Exercise 2 - Non-Linearity in Higher Dimensions:** PCA and hyperspherical structure
- **Exercise 3 - Real-World Data:** Spaceship Titanic preprocessing
- **Results Summary:** key numerical results required by the assignment

---

## Exercise 1 — Point Clouds

> **Implementation placeholder**
>
> Add one subsection for each item in the assignment statement. Keep code, figures, numerical results, and interpretation together.

### 1.a · Generate the point clouds

*Add analysis and code here.*

### 1.b · Study the effect of spread

*Add analysis and code here.*

### 1.c · Separation and mixing

*Add analysis and code here.*

### 1.d · Decision boundaries

*Add analysis and code here.*

---

## Exercise 2 — Non-Linearity in Higher Dimensions

> **Implementation placeholder**

### 2.a · Generate the datasets

*Add analysis and code here.*

### 2.b · PCA projection

*Add analysis and code here.*

### 2.c · Class-center distances

*Add analysis and code here.*

### 2.d · Radius distributions and non-linearity

*Add analysis and code here.*

---

## Exercise 3 — Real-World Data

> **Implementation placeholder**

### 3.a · Target and feature types

*Add analysis and code here.*

### 3.b · Missing-data analysis

*Add analysis and code here.*

### 3.c · Spending statistics

*Add analysis and code here.*

### 3.d · Stratified split and preprocessing

*Add analysis and code here.*

### 3.e · Feature engineering and transformations

*Add analysis and code here.*

### 3.f · Final dataset verification

*Add analysis and code here.*

---

## Results Summary

| Required result | Value |
|---|---:|
| Mixing rate at `s = 0.5` | — |
| Mixing rate at `s = 1.0` | — |
| Mixing rate at `s = 2.0` | — |
| Mixing rate at `s = 4.0` | — |
| Smallest `r_ij` at `s = 1` | — |
| Distance between centers — Dataset I | — |
| Distance between centers — Dataset II | — |
| PCA explained variance — Dataset I | — |
| PCA explained variance — Dataset II | — |
| Positive-class share in `Transported` | — |
| `FoodCourt` mean / median in training data | — |
| Final training feature-matrix shape | — |
| Min / max after scaling | — |

> The summary should contain the same values already discussed in the report. It is an index of the final numerical results, not a substitute for the analysis.

---

## Reproducibility

Use one RNG for the entire report:

```python
rng = np.random.default_rng(42)
```

The same `rng` instance should be reused throughout the activity.
