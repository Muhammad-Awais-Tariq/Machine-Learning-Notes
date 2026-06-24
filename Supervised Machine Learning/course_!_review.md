# Andrew Ng — Machine Learning Specialization: Course 1 Complete Review
---

## Table of Contents
1. [What is Machine Learning?](#1-what-is-machine-learning)
2. [Linear Regression](#2-linear-regression)
3. [Cost Function](#3-cost-function)
4. [Gradient Descent](#4-gradient-descent)
5. [Multiple Linear Regression & Vectorization](#5-multiple-linear-regression--vectorization)
6. [Feature Scaling](#6-feature-scaling)
7. [Polynomial Regression](#7-polynomial-regression)
8. [Overfitting & Underfitting](#8-overfitting--underfitting)
9. [Regularization](#9-regularization)
10. [Logistic Regression](#10-logistic-regression)
11. [Full sklearn Workflow](#11-full-sklearn-workflow)

---

## 1. What is Machine Learning?

Machine learning is teaching a computer to learn patterns from data without explicitly programming every rule.

| Type | What it does | Example |
|---|---|---|
| **Supervised Learning** | Learns from labeled data (input → output pairs) | Predicting house price, spam detection |
| **Unsupervised Learning** | Finds patterns in unlabeled data | Customer clustering |
| **Reinforcement Learning** | Agent learns by reward/punishment | Game playing AI |

**Course 1 focuses entirely on Supervised Learning.**

Two types of supervised learning tasks:
- **Regression** → output is a continuous number (e.g. house price = $350,000)
- **Classification** → output is a class/category (e.g. tumor = malignant / benign)

---

## 2. Linear Regression

### Concept
Linear regression fits a straight line through your data to predict a continuous output.

### Hypothesis Function (Univariate)
```
f(x) = wx + b
```
- `x` = input feature
- `w` = weight (slope) — how much y changes per unit of x
- `b` = bias (intercept) — value of y when x = 0
- `f(x)` = predicted value (ŷ)

### Goal
Find the values of `w` and `b` that make predictions as close to actual values as possible.

### sklearn
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)         # learns w and b from data
y_pred = model.predict(X_val)       # applies f(x) = wx + b

print(model.coef_)                  # w (weights)
print(model.intercept_)             # b (bias)
```

> In sklearn, LinearRegression internally uses the Normal Equation (closed-form math), not gradient descent — but the result is the same.

---

## 3. Cost Function

### Concept
The cost function measures how wrong your model's predictions are. It is the average squared difference between predicted and actual values.

### Formula — Mean Squared Error (MSE)
```
J(w, b) = (1 / 2m) * Σ (f(xᵢ) - yᵢ)²
```
- `m` = number of training examples
- `f(xᵢ)` = predicted value for example i
- `yᵢ` = actual value for example i
- We divide by `2m` (the 2 makes the derivative cleaner — Andrew Ng's notation)

### Why squared error?
1. Squaring makes all errors positive (no cancellation between positive and negative errors)
2. Squaring penalizes large errors more than small ones — a prediction 10 off is penalized 100x more than a prediction 1 off

### Goal
**Minimize J(w, b)** — find the w and b that make J as small as possible.

### sklearn — Evaluating Cost After Training
```python
from sklearn.metrics import mean_squared_error
import numpy as np

mse  = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)                        # RMSE: same unit as target
print(f'RMSE: {rmse:.4f}')

# sklearn uses MSE = (1/m)*Σ(pred - actual)² (no 2m denominator)
# mathematically equivalent for minimization purposes
```

> **RMSE (Root MSE)** is more interpretable — it's in the same unit as your target variable.

---

## 4. Gradient Descent

### Concept
Gradient descent is the algorithm that finds the minimum of the cost function by repeatedly updating `w` and `b` in the direction that reduces cost.

Think of it as: you're standing on a hilly surface (the cost function) and you take small steps downhill until you reach the lowest point (minimum cost).

### Update Rules
```
w = w - α * (∂J/∂w)
b = b - α * (∂J/∂b)
```

For linear regression, the derivatives work out to:
```
w = w - α * (1/m) * Σ (f(xᵢ) - yᵢ) * xᵢ
b = b - α * (1/m) * Σ (f(xᵢ) - yᵢ)
```

- `α` (alpha) = **learning rate** — how big each step is
- `∂J/∂w` = derivative — which direction is downhill

### ⚠️ Simultaneous Update Rule
`w` and `b` **must be updated at the same time** using the old values of both.

```python
# WRONG — b uses already-updated w
w = w - alpha * dw
b = b - alpha * db   # ← db was computed with old w but now w is already changed

# CORRECT — compute both derivatives first, then update
temp_w = w - alpha * dw
temp_b = b - alpha * db
w = temp_w
b = temp_b
```

### Learning Rate α
| α value | Effect |
|---|---|
| Too large | Gradient descent overshoots the minimum, may never converge (oscillates or diverges) |
| Too small | Converges correctly but takes too many iterations — very slow |
| Just right | Smoothly reaches minimum in reasonable iterations |

### Convergence
Gradient descent has **converged** when the cost function stops decreasing meaningfully between iterations. You plot J vs iterations and look for when the curve flattens out.

### sklearn Note
sklearn's `LinearRegression` uses the Normal Equation (exact math solution), not gradient descent. Gradient descent is used internally by `SGDRegressor`:

```python
from sklearn.linear_model import SGDRegressor

model = SGDRegressor(learning_rate='constant', eta0=0.01, max_iter=1000)
model.fit(X_train, y_train)
```

In practice, always prefer `LinearRegression` for small/medium datasets — it's more stable.

---

## 5. Multiple Linear Regression & Vectorization

### Concept
Real datasets have multiple features (columns). Multiple linear regression extends the line to multiple dimensions.

### Hypothesis Function
```
f(x) = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

**Vectorized form:**
```
f(x) = w · x + b
```
where `·` is the dot product: multiply matching elements and sum them up.

### Features vs Parameters
| Term | What it is | Example |
|---|---|---|
| **Feature** (x) | A column in your dataset | Area, No. of bedrooms, Age |
| **Weight/Parameter** (w) | How much that feature affects the prediction | w₁ = 200 means each sq ft adds $200 |
| **Bias** (b) | Base value when all features are zero | Intercept of the line |

### Vectorization
Instead of computing each term in a loop:
```python
# slow loop
prediction = 0
for j in range(n):
    prediction += w[j] * x[j]
prediction += b
```

Use NumPy's dot product which runs in **parallel on hardware (SIMD)**:
```python
# fast — vectorized
prediction = np.dot(w, x) + b
```

Vectorization is not just shorter code — it's fundamentally faster because it uses CPU/GPU parallel processing instead of sequential loops.

### sklearn
```python
from sklearn.linear_model import LinearRegression

# X_train is shape (m, n) — m samples, n features
model = LinearRegression()
model.fit(X_train, y_train)
# sklearn internally does the vectorized dot product
```

---

## 6. Feature Scaling

### Why it matters
If features have very different ranges:
- Feature A: House area — ranges 500 to 5000
- Feature B: Number of bedrooms — ranges 1 to 5

The contour plot of J(w,b) becomes a very elongated ellipse. Gradient descent has to zig-zag slowly across the narrow axis and takes forever to converge.

After scaling, both features have similar ranges → contour plot becomes circular → gradient descent goes straight to minimum quickly.

### Method 1: Z-score Normalization (StandardScaler)
Centers data at 0 with standard deviation 1.
```
x_scaled = (x - μ) / σ
```
- `μ` = mean of the feature
- `σ` = standard deviation of the feature
- Result: most values fall between -3 and +3

**Use when:** data has outliers, or you don't know the range in advance.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)              # learns μ and σ from TRAINING data only
X_train_scaled = scaler.transform(X_train)
X_val_scaled   = scaler.transform(X_val)   # uses same μ and σ from train
X_test_scaled  = scaler.transform(X_test)  # uses same μ and σ from train
```

### Method 2: Min-Max Scaling (MinMaxScaler)
Squashes all values to range [0, 1].
```
x_scaled = (x - x_min) / (x_max - x_min)
```

**Use when:** you know data has no extreme outliers and you need values between 0 and 1.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
```

### ⚠️ Critical Rule: Fit only on Training Data
```python
# WRONG — data leakage
scaler.fit(X)                    # sees test data during fit → cheating

# CORRECT
scaler.fit(X_train)              # fit on train only
scaler.transform(X_val)          # apply same scale to val
scaler.transform(X_test)         # apply same scale to test
```

Fitting on the full dataset leaks information about the test set into training — your model appears better than it really is.

### Pipeline (Best Practice)
Use `Pipeline` so scaling is always applied correctly and automatically:
```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('model', LinearRegression()),
])

pipe.fit(X_train, y_train)       # fits scaler on train, then fits model
pipe.predict(X_val)              # automatically scales X_val before predicting
```

---

## 7. Polynomial Regression

### Concept
When the relationship between x and y is curved (not a straight line), we add polynomial features — squares, cubes, cross-products — to capture the curve while still using linear regression internally.

```
f(x) = w₁x + w₂x² + w₃x³ + b
```

The model is still "linear" in the weights (w) — it's just that the features are now x, x², x³.

### When to use
Plot each feature vs target in a scatter plot:
- **Straight line pattern** → use Linear/Ridge Regression
- **Curved/U-shape pattern** → use Polynomial + Ridge

### ⚠️ Always pair Polynomial with Regularization
Adding polynomial features creates many new features → high risk of overfitting. Always use Ridge with polynomial features.

### sklearn
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

pipe_poly = Pipeline([
    ('scale', StandardScaler()),
    ('poly',  PolynomialFeatures(degree=2, include_bias=False)),
    ('model', Ridge(alpha=1.0)),
])

pipe_poly.fit(X_train, y_train)
y_pred = pipe_poly.predict(X_val)
```

`PolynomialFeatures(degree=2)` on a feature `x` creates: `[x, x²]`  
On two features `[x₁, x₂]` creates: `[x₁, x₂, x₁², x₁x₂, x₂²]`

---

## 8. Overfitting & Underfitting

### The Three Cases

| Situation | Training Error | Validation Error | Name | Cause |
|---|---|---|---|---|
| **Underfitting** | High | High | High Bias | Model too simple — can't capture pattern |
| **Good Fit** | Low | Low (close to train) | Balanced | Model complexity matches data |
| **Overfitting** | Very Low | High | High Variance | Model memorized training data, doesn't generalize |

### Intuition
- **Underfitting (High Bias):** A straight line trying to fit a curved dataset. The model has a wrong assumption (bias) about the data's shape.
- **Overfitting (High Variance):** A very wiggly high-degree polynomial that passes exactly through every training point — but falls apart on new data.

### Train / Validation / Test Split
```
All data → 70% Train | 15% Validation | 15% Test
```
- **Train:** Model learns from this
- **Validation:** Used to tune and compare models (pick best hyperparameters)
- **Test:** Used ONCE at the very end to report final performance — never used to make decisions

```python
from sklearn.model_selection import train_test_split

X_train, X_tmp,  y_train, y_tmp  = train_test_split(X, y, test_size=0.30, random_state=42)
X_val,   X_test, y_val,   y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42)
```

### How to Diagnose
```python
train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
val_rmse   = np.sqrt(mean_squared_error(y_val,   model.predict(X_val)))

if train_rmse > threshold and val_rmse > threshold:
    print("HIGH BIAS — underfitting")
elif val_rmse > train_rmse * 1.5:
    print("HIGH VARIANCE — overfitting")
else:
    print("Good fit")
```

### How to Fix
| Problem | Fix |
|---|---|
| Underfitting | Add more features, increase polynomial degree, decrease regularization |
| Overfitting | Get more training data, reduce features, add/increase regularization |

---

## 9. Regularization

### Concept
Regularization adds a penalty to the cost function to discourage the model from learning very large weights. Large weights → very complex, wiggly model → overfitting.

### Modified Cost Function
```
J(w, b) = (1/2m) * Σ(f(xᵢ) - yᵢ)² + (λ/2m) * Σwⱼ²
```
- The first term = original MSE loss
- The second term = regularization penalty (sum of squared weights)
- `λ` (lambda) = regularization parameter — controls how hard you penalize large weights

### Effect of λ
| λ value | Effect |
|---|---|
| λ = 0 | No regularization — model can overfit freely |
| λ small (0.01) | Mild penalty — slight regularization |
| λ = 1 | Moderate regularization — good starting point |
| λ very large | All weights → 0, model becomes just `f = b` — underfits |

### Ridge Regression (L2)
The sklearn equivalent of regularization for linear regression. Andrew Ng's `λ` = sklearn's `alpha`.

```python
from sklearn.linear_model import Ridge

# alpha is lambda — bigger alpha = more regularization
pipe_ridge = Pipeline([
    ('scale', StandardScaler()),
    ('model', Ridge(alpha=1.0)),   # try: 0.01, 0.1, 1.0, 10, 100
])

pipe_ridge.fit(X_train, y_train)
```

### Tuning Alpha
```python
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('model', Ridge(alpha=alpha)),
    ])
    pipe.fit(X_train, y_train)
    val_rmse = np.sqrt(mean_squared_error(y_val, pipe.predict(X_val)))
    print(f'alpha={alpha:<8}  val RMSE={val_rmse:.4f}')
# pick the alpha with lowest val RMSE
```

### Note on Bias Term
Regularization is applied to the weights `w` only — **not to the bias `b`**. The bias is just a shift and doesn't contribute to overfitting.

---

## 10. Logistic Regression

### Concept
Logistic regression is used for **binary classification** — predicting whether something belongs to class 0 or class 1.

Despite the name "regression," it's a classification algorithm.

### Why Not Linear Regression for Classification?
Linear regression output is unbounded — it can give values like -3 or 150. These don't make sense as probabilities or class labels. Also, one outlier can completely shift the decision boundary.

### Sigmoid Function
Logistic regression wraps linear regression output through the sigmoid function to squash it into [0, 1]:
```
g(z) = 1 / (1 + e^(-z))
```
```
z = w·x + b
f(x) = g(z) = 1 / (1 + e^(-(w·x + b)))
```

- Output is the **probability of belonging to class 1**: P(y=1 | x)
- If output > 0.5 → predict class 1
- If output ≤ 0.5 → predict class 0

### Decision Boundary
The decision boundary is where `f(x) = 0.5`, which means `z = w·x + b = 0`. It is the line (or curve) that separates class 0 from class 1 in feature space.

### Loss Function — Log Loss (Binary Cross-Entropy)
We cannot use MSE with logistic regression because the cost function becomes non-convex (many local minima) → gradient descent can't reliably find the global minimum.

Log loss gives a convex cost function:
```
If y = 1:  Loss = -log(f(x))
If y = 0:  Loss = -log(1 - f(x))
```

Combined:
```
J(w,b) = -(1/m) * Σ [yᵢ log(f(xᵢ)) + (1 - yᵢ) log(1 - f(xᵢ))]
```

### Regularization in Logistic Regression
sklearn uses `C = 1/λ` — the **inverse** of regularization strength:
- `C` large → less regularization (like small λ)
- `C` small → more regularization (like large λ)

```python
from sklearn.linear_model import LogisticRegression

pipe_clf = Pipeline([
    ('scale', StandardScaler()),
    ('model', LogisticRegression(C=1.0, max_iter=1000)),
])

pipe_clf.fit(X_train_c, y_train_c)

y_pred   = pipe_clf.predict(X_val_c)              # class label: 0 or 1
y_proba  = pipe_clf.predict_proba(X_val_c)[:, 1]  # probability of class 1
```

### Tuning C
```python
for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('model', LogisticRegression(C=C, max_iter=1000)),
    ])
    pipe.fit(X_train_c, y_train_c)
    val_acc = accuracy_score(y_val_c, pipe.predict(X_val_c))
    print(f'C={C:<8}  val accuracy={val_acc:.4f}')
```

### Evaluation Metric
```python
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_val_c, y_pred)
print(f'Accuracy: {acc:.4f}')
# accuracy = correct predictions / total predictions
```

---

## 11. Full sklearn Workflow

This is the complete end-to-end workflow for any Course 1 problem. Replace column names with your dataset.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, PolynomialFeatures
from sklearn.linear_model    import LinearRegression, Ridge, LogisticRegression
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import mean_squared_error, r2_score, accuracy_score

# ── 1. Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv('your_file.csv')
df.head()

# ── 2. Plot Features vs Target (decide linear vs polynomial) ──────────────────
feature_cols = ['col1', 'col2', 'col3']
target_col   = 'target'

for col in feature_cols:
    plt.scatter(df[col], df[target_col], alpha=0.5)
    plt.xlabel(col); plt.ylabel(target_col)
    plt.title(f'{col} vs {target_col}'); plt.show()
# straight line → use Linear/Ridge | curved → use Polynomial + Ridge

# ── 3. Prepare X, y and Split ─────────────────────────────────────────────────
X = df[feature_cols].values
y = df[target_col].values

X_train, X_tmp,  y_train, y_tmp  = train_test_split(X, y, test_size=0.30, random_state=42)
X_val,   X_test, y_val,   y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42)
print(f'Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}')

# ══════════════════════════════════════════════════════════════════════════════
# LINEAR REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

# ── 4a. Scale + Linear Regression ────────────────────────────────────────────
pipe_linear = Pipeline([
    ('scale', StandardScaler()),
    ('model', LinearRegression()),
])
pipe_linear.fit(X_train, y_train)

y_pred_train = pipe_linear.predict(X_train)
y_pred_val   = pipe_linear.predict(X_val)

print(f'Train RMSE : {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}')
print(f'Val   RMSE : {np.sqrt(mean_squared_error(y_val,   y_pred_val)):.4f}')
print(f'Val   R²   : {r2_score(y_val, y_pred_val):.4f}')

# ── 4b. Plot Actual vs Predicted ──────────────────────────────────────────────
plt.scatter(y_val, y_pred_val, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel('Actual'); plt.ylabel('Predicted')
plt.title('Linear Regression — Actual vs Predicted'); plt.show()
# dots close to red line = good predictions

# ── 4c. Diagnose Bias / Variance ──────────────────────────────────────────────
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
val_rmse   = np.sqrt(mean_squared_error(y_val,   y_pred_val))
if train_rmse > 0.3 and val_rmse > 0.3:
    print('→ HIGH BIAS (underfitting) — try polynomial features')
elif val_rmse > train_rmse * 1.5:
    print('→ HIGH VARIANCE (overfitting) — try Ridge regularization')
else:
    print('→ Good fit')

# ── 4d. Ridge (if overfitting) ────────────────────────────────────────────────
pipe_ridge = Pipeline([
    ('scale', StandardScaler()),
    ('model', Ridge(alpha=1.0)),   # alpha = lambda; bigger = more regularization
])
pipe_ridge.fit(X_train, y_train)
print(f'Ridge Train RMSE : {np.sqrt(mean_squared_error(y_train, pipe_ridge.predict(X_train))):.4f}')
print(f'Ridge Val   RMSE : {np.sqrt(mean_squared_error(y_val,   pipe_ridge.predict(X_val))):.4f}')

# ══════════════════════════════════════════════════════════════════════════════
# POLYNOMIAL REGRESSION  (use when plots showed curved relationship)
# ══════════════════════════════════════════════════════════════════════════════

# ── 5a. Scale + Poly + Ridge ──────────────────────────────────────────────────
pipe_poly = Pipeline([
    ('scale', StandardScaler()),
    ('poly',  PolynomialFeatures(degree=2, include_bias=False)),
    ('model', Ridge(alpha=1.0)),
])
pipe_poly.fit(X_train, y_train)

y_pred_poly_val = pipe_poly.predict(X_val)
print(f'Poly Val RMSE : {np.sqrt(mean_squared_error(y_val, y_pred_poly_val)):.4f}')

# ── 5b. Tune degree + alpha ───────────────────────────────────────────────────
for degree in [2, 3]:
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        pipe = Pipeline([
            ('scale', StandardScaler()),
            ('poly',  PolynomialFeatures(degree=degree, include_bias=False)),
            ('model', Ridge(alpha=alpha)),
        ])
        pipe.fit(X_train, y_train)
        val_rmse = np.sqrt(mean_squared_error(y_val, pipe.predict(X_val)))
        print(f'degree={degree}  alpha={alpha:<6}  val RMSE={val_rmse:.4f}')
# pick combination with lowest val RMSE

# ══════════════════════════════════════════════════════════════════════════════
# LOGISTIC REGRESSION  (binary classification — target must be 0/1)
# ══════════════════════════════════════════════════════════════════════════════

target_clf = 'target_binary'
X_c = df[feature_cols].values
y_c = df[target_clf].values

X_train_c, X_tmp_c,  y_train_c, y_tmp_c  = train_test_split(X_c, y_c, test_size=0.30, random_state=42)
X_val_c,   X_test_c, y_val_c,   y_test_c = train_test_split(X_tmp_c, y_tmp_c, test_size=0.50, random_state=42)

# ── 6a. Scale + Logistic Regression ──────────────────────────────────────────
# C = 1/lambda → smaller C = more regularization (inverse of Ridge alpha)
pipe_clf = Pipeline([
    ('scale', StandardScaler()),
    ('model', LogisticRegression(C=1.0, max_iter=1000)),
])
pipe_clf.fit(X_train_c, y_train_c)

y_pred_clf  = pipe_clf.predict(X_val_c)
y_proba_clf = pipe_clf.predict_proba(X_val_c)[:, 1]  # P(y=1)
print(f'Val Accuracy : {accuracy_score(y_val_c, y_pred_clf):.4f}')

# ── 6b. Plot Predicted Probabilities ─────────────────────────────────────────
plt.hist(y_proba_clf[y_val_c == 0], bins=20, alpha=0.6, label='Actual 0')
plt.hist(y_proba_clf[y_val_c == 1], bins=20, alpha=0.6, label='Actual 1')
plt.axvline(0.5, color='red', linestyle='--', label='threshold 0.5')
plt.xlabel('Predicted Probability'); plt.legend()
plt.title('Logistic Regression — Predicted Probabilities'); plt.show()
# two separate humps = model separating classes well
# overlapping humps  = model is confused

# ── 6c. Diagnose & Tune C ─────────────────────────────────────────────────────
train_acc = accuracy_score(y_train_c, pipe_clf.predict(X_train_c))
val_acc   = accuracy_score(y_val_c,   y_pred_clf)
if train_acc < 0.75 and val_acc < 0.75:
    print('→ HIGH BIAS — add features or increase C')
elif train_acc - val_acc > 0.1:
    print('→ HIGH VARIANCE — decrease C')
else:
    print('→ Good fit')

for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('model', LogisticRegression(C=C, max_iter=1000)),
    ])
    pipe.fit(X_train_c, y_train_c)
    print(f'C={C:<8}  val accuracy={accuracy_score(y_val_c, pipe.predict(X_val_c)):.4f}')

# ══════════════════════════════════════════════════════════════════════════════
# FINAL EVALUATION — run ONCE after picking best model
# ══════════════════════════════════════════════════════════════════════════════
test_rmse = np.sqrt(mean_squared_error(y_test, pipe_ridge.predict(X_test)))
print(f'Test RMSE : {test_rmse:.4f}')

test_acc = accuracy_score(y_test_c, pipe_clf.predict(X_test_c))
print(f'Test Accuracy : {test_acc:.4f}')
# test ≈ val → model generalizes well
# test much worse than val → you over-tuned on val (overfitting to val set)
```

---

## Quick Reference Cheatsheet

| Concept | Andrew Ng Notation | sklearn |
|---|---|---|
| Hypothesis | `f(x) = wx + b` | `model.predict()` |
| Cost | `J(w,b)` = MSE | `mean_squared_error()` |
| Gradient Descent | Updates w, b | Done internally by `.fit()` |
| Learning Rate | `α` (alpha) | `eta0` in SGDRegressor |
| Regularization strength | `λ` (lambda) | `alpha` in Ridge, `1/C` in LogisticRegression |
| Z-score scaling | `(x-μ)/σ` | `StandardScaler` |
| Min-max scaling | `(x-min)/(max-min)` | `MinMaxScaler` |
| Polynomial features | `x, x², x³` | `PolynomialFeatures(degree=n)` |
| Sigmoid output | `P(y=1\|x)` | `predict_proba()[:, 1]` |
| Classification metric | Accuracy | `accuracy_score()` |
| Regression metric | RMSE | `np.sqrt(mean_squared_error())` |

---
