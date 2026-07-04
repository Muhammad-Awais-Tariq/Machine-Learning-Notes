# Andrew Ng — Machine Learning Specialization: Course 2 Complete Review
## (Advanced Learning Algorithms)
---

## Table of Contents
1. [Neural Networks — Intuition](#1-neural-networks--intuition)
2. [Neural Network Architecture & Notation](#2-neural-network-architecture--notation)
3. [Activation Functions](#3-activation-functions)
4. [Forward Propagation](#4-forward-propagation)
5. [TensorFlow Implementation](#5-tensorflow-implementation)
6. [Multiclass Classification & Softmax](#6-multiclass-classification--softmax)
7. [Advanced Optimization (Adam)](#7-advanced-optimization-adam)
8. [Bias / Variance Diagnostics](#8-bias--variance-diagnostics)
9. [Machine Learning Development Process](#9-machine-learning-development-process)
10. [Decision Trees](#10-decision-trees)
11. [Tree Ensembles](#11-tree-ensembles)
12. [Decision Trees vs Neural Networks](#12-decision-trees-vs-neural-networks)
13. [Full Implementation Workflow](#13-full-implementation-workflow)

---

## 1. Neural Networks — Intuition

### Concept
A neural network is a stack of layers, where each layer is a group of neurons (units). Each neuron is basically its own small logistic/linear regression unit — the network "learns" its own features instead of you having to hand-engineer them (unlike Course 1, where you manually created polynomial features).

### Why Neural Networks?
- They scale better with large, complex datasets than traditional ML algorithms.
- They automatically learn increasingly abstract features layer by layer (e.g. in image recognition: edges → shapes → object parts → objects).

### Layer Types
| Layer | Description |
|---|---|
| **Input layer** | The raw features, `x` |
| **Hidden layer(s)** | Layers between input and output — their outputs (activations) aren't directly observed, hence "hidden" |
| **Output layer** | Produces the final prediction |

---

## 2. Neural Network Architecture & Notation

### A Single Neuron
Every neuron computes a weighted sum of inputs plus a bias, then applies an activation function:
```
z = w · x + b
a = g(z)
```
- `w`, `b` = parameters of that neuron (learned during training)
- `g` = activation function
- `a` = the neuron's output ("activation")

### Layer Notation
For layer `l`, neuron `j`:
```
a[l]_j = g( w[l]_j · a[l-1] + b[l]_j )
```
- `a[l-1]` = the activation vector from the previous layer (or `x` if `l=1`)
- Superscript `[l]` = layer number, subscript `j` = neuron index within that layer

### "A layer has N units" means...
If a layer has 5 units, it contains 5 independent neurons — each with its own `w` and `b` — all taking the *same* input vector from the previous layer and producing 5 separate outputs, which together form the output vector `a[l]` passed to the next layer.

---

## 3. Activation Functions

| Function | Formula | Range | Typical Use |
|---|---|---|---|
| **Linear** | `g(z) = z` | (-∞, ∞) | Output layer for regression when y can be negative |
| **Sigmoid** | `g(z) = 1/(1+e^-z)` | (0, 1) | Output layer for binary classification |
| **ReLU** | `g(z) = max(0, z)` | [0, ∞) | Default for hidden layers |
| **Softmax** | see §6 | (0, 1), sums to 1 | Output layer for multiclass classification |

### Why We Need Non-Linear Activations
If every layer used a **linear** activation function, the entire network — no matter how many layers — would mathematically collapse into a single linear function. You'd just be doing linear regression with extra steps. Non-linear activations (ReLU, sigmoid) are what let the network model complex, curved decision boundaries and relationships.

### Why ReLU Is Preferred Over Sigmoid for Hidden Layers
- Sigmoid saturates (flattens) at both ends → gradients become near-zero → slow learning ("vanishing gradient").
- ReLU only saturates on one side (`z < 0`), so gradients flow better and training converges faster.

### Choosing Output Layer Activation
| Problem Type | Output Activation |
|---|---|
| Regression, y can be + or - | Linear |
| Regression, y ≥ 0 only | ReLU |
| Binary classification | Sigmoid |
| Multiclass classification | Softmax |

---

## 4. Forward Propagation

### Concept
Forward propagation is the process of computing predictions by passing data through the network layer by layer — each layer's output (activation) becomes the next layer's input.

```
a[0] = x                                  (input)
a[1] = g(W[1] · a[0] + b[1])              (layer 1 output)
a[2] = g(W[2] · a[1] + b[2])              (layer 2 output)
...
a[L] = g(W[L] · a[L-1] + b[L])            (final output / prediction)
```

### Important Distinction
Forward propagation only **computes** predictions using the *current* weights — it does not change any weights. Training (via backpropagation + gradient descent) is a separate process that *updates* the weights based on the error from forward propagation.

---

## 5. TensorFlow Implementation

### Basic Workflow
```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# 1. Define architecture
model = Sequential([
    Dense(units=25, activation='relu'),
    Dense(units=15, activation='relu'),
    Dense(units=1,  activation='sigmoid')   # binary classification output
])

# 2. Compile — define loss function and optimizer
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# 3. Fit — actually train the model (this is where backprop + gradient descent happen)
model.fit(X_train, y_train, epochs=100)

# 4. Predict
predictions = model.predict(X_val)
```

- `Sequential(...)` — just **defines** the architecture (layers, units, activations). No learning happens here.
- `.compile(...)` — configures **how** training will happen (loss function, optimizer).
- `.fit(...)` — **actually trains** the model: runs forward prop, computes loss, runs backprop, updates weights, repeats for each epoch.

### Loss Functions
| Task | Loss Function |
|---|---|
| Binary classification | `BinaryCrossentropy()` |
| Multiclass classification | `SparseCategoricalCrossentropy()` |
| Regression | `MeanSquaredError()` |

### The `from_logits=True` Trick
If you compute softmax (or sigmoid) and cross-entropy loss as two *separate* steps, tiny numbers close to 0 or 1 can suffer floating-point round-off error, which gets worse in larger networks and can destabilize training.

**Fix:** Use a **linear** activation in the final layer (output raw scores, called "logits," instead of probabilities), and pass `from_logits=True` to the loss function. TensorFlow then combines the softmax/sigmoid and cross-entropy computation into a single, numerically stable step internally, rather than computing them independently.

```python
model = Sequential([
    Dense(units=25, activation='relu'),
    Dense(units=10, activation='linear')     # note: linear, NOT softmax
])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
)

# predictions now need softmax applied manually afterward:
logits = model.predict(X_val)
probabilities = tf.nn.softmax(logits).numpy()
```

---

## 6. Multiclass Classification & Softmax

### Concept
Softmax regression is the generalization of logistic regression to more than 2 classes. Instead of producing a single probability (class 1 vs. not), softmax produces a full probability distribution across **all N classes simultaneously**, where all probabilities sum to exactly 1.

### Formula
For class `j` out of N classes:
```
a_j = e^(z_j) / Σ(k=1 to N) e^(z_k)
```
Each class's raw score `z_j` competes directly against every other class's score — logistic regression is just the special 2-class case of this formula.

### Why Not Just Use N Independent Sigmoids?
Using an independent sigmoid on each output unit treats every class as its own separate yes/no question — the outputs don't relate to each other and won't necessarily sum to 1. This works fine for **multi-label** problems (an image can be both "cat" AND "outdoors"), but is wrong for **mutually exclusive** multiclass problems (a digit is either 0, 1, 2... not several at once).

Softmax couples all classes together: pushing up the score for one class *necessarily* pulls down the probabilities of the others — exactly the behavior you want when only one class can be correct.

### Multiclass Loss
```
SparseCategoricalCrossentropy — use when labels are integers (0, 1, 2, 3...)
CategoricalCrossentropy       — use when labels are one-hot encoded ([0,0,1,0])
```

---

## 7. Advanced Optimization (Adam)

### Concept
Adam ("Adaptive Moment Estimation") is an improved version of gradient descent that automatically adjusts the learning rate for each parameter individually during training, rather than using one fixed global `α`.

- If a parameter's updates keep moving in the same direction → Adam increases its learning rate (speed up).
- If a parameter's updates keep oscillating back and forth → Adam decreases its learning rate (slow down, stabilize).

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)
```

In practice, Adam converges faster and more reliably than plain gradient descent for neural networks, and is the default choice in almost all modern implementations.

---

## 8. Bias / Variance Diagnostics

### Definitions
| | Training Error | Cross-Validation Error | Diagnosis |
|---|---|---|---|
| **High Bias** | High | High | Underfitting — model too simple |
| **High Variance** | Low | High (much higher than train) | Overfitting — model memorized training data |
| **Good fit** | Low | Low (close to train) | Balanced |

### Baseline vs. Desired Performance
To correctly diagnose bias, you need a **baseline level of performance** (e.g. human-level performance, or a known achievable benchmark) to compare against — not just an arbitrary target.

- Training error much higher than baseline → **high bias**
- Training error close to baseline, but validation error much higher than training error → **high variance**

### Fixes
| Problem | Fixes |
|---|---|
| High Bias | Add more features, increase polynomial degree, decrease regularization (λ), use a bigger/more complex network, train longer |
| High Variance | Get more training data, reduce number of features, increase regularization (λ), use a simpler network |

### Regularization in Neural Networks
```python
from tensorflow.keras.regularizers import l2

model = Sequential([
    Dense(units=25, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(units=15, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(units=1,  activation='linear')
])
```

---

## 9. Machine Learning Development Process

### The Iterative Loop
```
1. Choose architecture (model, data, features)
        ↓
2. Train model
        ↓
3. Run diagnostics (bias/variance, error analysis)
        ↓
4. Go back to Step 1 and adjust based on findings
```

### Error Analysis
Manually examine a sample of the misclassified cross-validation examples and categorize them by *why* they were wrong (e.g. "confused similar-looking classes," "blurry image," "mislabeled data"). This tells you where to focus effort — e.g. collecting more data of a specific type, rather than more data in general.

### Data Augmentation
Artificially expanding your training set by creating modified versions of existing examples (e.g. rotating/distorting images, adding background noise to audio) — useful when getting brand new labeled data is expensive, but modifying existing examples is cheap.

### Transfer Learning
Reusing a neural network already trained on a large, different dataset (e.g. ImageNet), and fine-tuning only the later layers (or all layers) on your smaller, specific dataset. Works because early layers tend to learn generic features (edges, shapes) that transfer across tasks.

---

## 10. Decision Trees

### Concept
A decision tree makes predictions by asking a sequence of yes/no questions about feature values, splitting the dataset at each node until it reaches a prediction at a leaf.

### Purity & Entropy
"Purity" at a node describes how much of one single class dominates the examples at that node. It's measured using **entropy**:
```
H(p1) = -p1 * log2(p1) - (1-p1) * log2(1-p1)
```
- `p1` = fraction of positive-class examples at that node
- Entropy = 0 → node is perfectly pure (all one class)
- Entropy = 1 → node is maximally impure (50/50 split)

### Information Gain
Information gain measures how much a split reduces entropy — the feature that produces the **highest information gain** is chosen for splitting:
```
Information Gain = H(p1_root) - [ w_left * H(p1_left) + w_right * H(p1_right) ]
```
- `w_left`, `w_right` = fraction of examples that went to each branch
- We pick the feature/split that maximizes this reduction in entropy

### Stopping Criteria (prevents overfitting)
1. Node becomes 100% pure
2. Maximum tree depth reached
3. Information gain from further splits falls below a threshold
4. Number of examples at a node falls below a threshold

### Continuous-Valued Features
For a feature with continuous values, the tree tests multiple candidate threshold values (e.g. weight ≤ 9kg?) and picks the threshold that gives the highest information gain — examples above the threshold go to one branch, examples at or below go to the other.

### sklearn
```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=42)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_val)
```

---

## 11. Tree Ensembles

### Why Ensembles?
A single decision tree is very sensitive to small changes in the training data — changing just a few examples can produce a completely different tree ("high variance"). Ensembles combine many trees to average out this instability.

### Random Forest
Builds many decision trees, each on a slightly different version of the data, then has them vote on the final prediction (majority vote for classification, average for regression). Two sources of randomness make each tree different:

1. **Sampling with replacement (bagging):** Each tree is trained on a random sample of the training set (same size as original, but with duplicates and omissions possible), so each tree sees a slightly different dataset.
2. **Random feature selection:** At each split, only a random subset of features is considered (instead of all features), so trees end up splitting on different features and don't all look alike.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
```

### Boosted Trees (e.g., XGBoost)
Unlike Random Forest (where trees are built independently and in parallel), boosting builds trees **sequentially** — each new tree is trained to focus specifically on the examples the previous trees got wrong (misclassified or high-error examples are given more weight). This means each iteration deliberately corrects the mistakes of the ones before it, rather than just being another independent random view of the data.

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
xgb.fit(X_train, y_train)
```

### Random Forest vs. Boosting — Core Difference
| | Random Forest | Boosted Trees (XGBoost) |
|---|---|---|
| Tree building | Parallel, independent | Sequential, each depends on the last |
| Focus | Random subsets of data/features | Misclassified examples from previous trees |
| Main benefit | Reduces variance | Reduces bias, often higher accuracy |

---

## 12. Decision Trees vs Neural Networks

| Use Case | Prefer |
|---|---|
| Structured/tabular data (spreadsheets, databases) | Decision Trees / Tree Ensembles |
| Unstructured data (images, audio, text) | Neural Networks |
| Need fast training, small dataset | Decision Trees |
| Need transfer learning | Neural Networks |
| Need model to be part of a larger system trained end-to-end (e.g. with other neural nets) | Neural Networks |
| Need interpretability / to inspect individual decisions | Decision Trees |

Tree ensembles (Random Forest, XGBoost) are generally the strong default for structured data and are often faster to train than neural networks for the same task.

---

## 13. Full Implementation Workflow

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ── 1. Split data ─────────────────────────────────────────────────────────────
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42)

# ── 2. Scale features (neural networks need this; trees don't) ────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# ══════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK — multiclass example
# ══════════════════════════════════════════════════════════════════════════════

model = Sequential([
    Dense(units=64, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(units=32, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(units=10, activation='linear')   # linear output for from_logits
])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

history = model.fit(X_train_s, y_train, validation_data=(X_val_s, y_val), epochs=100, verbose=0)

# ── Diagnose bias/variance from training curves ────────────────────────────────
train_acc = history.history['accuracy'][-1]
val_acc   = history.history['val_accuracy'][-1]
print(f'Train acc: {train_acc:.4f} | Val acc: {val_acc:.4f}')
if train_acc < 0.85 and val_acc < 0.85:
    print('→ HIGH BIAS — try bigger network, train longer, fewer regularization')
elif train_acc - val_acc > 0.1:
    print('→ HIGH VARIANCE — add regularization, more data, or simpler network')
else:
    print('→ Good fit')

# ── Get predictions (apply softmax since output is logits) ────────────────────
logits = model.predict(X_val_s)
probabilities = tf.nn.softmax(logits).numpy()
predicted_classes = np.argmax(probabilities, axis=1)

# ══════════════════════════════════════════════════════════════════════════════
# TREE ENSEMBLE — same task, tabular baseline (trees don't need scaling)
# ══════════════════════════════════════════════════════════════════════════════

tree = DecisionTreeClassifier(max_depth=6, criterion='entropy', random_state=42)
tree.fit(X_train, y_train)
print(f'Single Tree Val Accuracy: {accuracy_score(y_val, tree.predict(X_val)):.4f}')

rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
rf.fit(X_train, y_train)
print(f'Random Forest Val Accuracy: {accuracy_score(y_val, rf.predict(X_val)):.4f}')

# ══════════════════════════════════════════════════════════════════════════════
# FINAL EVALUATION — once, at the end, on the chosen best model
# ══════════════════════════════════════════════════════════════════════════════
final_preds = np.argmax(tf.nn.softmax(model.predict(X_test_s)).numpy(), axis=1)
print(f'Final Test Accuracy: {accuracy_score(y_test, final_preds):.4f}')
```

---

## Quick Reference Cheatsheet

| Concept | Formula / Notation | TensorFlow / sklearn |
|---|---|---|
| Neuron computation | `z = w·x + b`, `a = g(z)` | `Dense(units, activation=...)` |
| Linear activation | `g(z) = z` | `activation='linear'` |
| Sigmoid | `g(z) = 1/(1+e^-z)` | `activation='sigmoid'` |
| ReLU | `g(z) = max(0,z)` | `activation='relu'` |
| Softmax | `a_j = e^z_j / Σe^z_k` | `activation='softmax'` or via `from_logits=True` |
| Define architecture | — | `Sequential([...])` |
| Configure training | — | `.compile(loss=..., optimizer=...)` |
| Train | — | `.fit(X, y, epochs=...)` |
| Binary loss | Log loss | `BinaryCrossentropy()` |
| Multiclass loss | Cross-entropy | `SparseCategoricalCrossentropy(from_logits=True)` |
| Optimizer | Adaptive gradient descent | `Adam(learning_rate=...)` |
| Entropy | `-p1*log2(p1) - (1-p1)*log2(1-p1)` | `criterion='entropy'` |
| Information gain | `H(root) - weighted avg H(children)` | used internally by `DecisionTreeClassifier` |
| Random Forest | Bagging + random feature subsets | `RandomForestClassifier()` |
| Boosting | Sequential, focuses on past mistakes | `XGBClassifier()` |
| Diagnose bias | train error ≈ CV error, both high | compare to baseline |
| Diagnose variance | train error low, CV error high | compare train vs CV gap |

---