# Andrew Ng — Machine Learning Specialization: Course 3 Complete Review
## (Unsupervised Learning, Recommenders, Reinforcement Learning)
---

## Table of Contents
1. [Course 3 Overview](#1-course-3-overview)
2. [K-Means Clustering](#2-k-means-clustering)
3. [Anomaly Detection](#3-anomaly-detection)
4. [Recommender Systems — Collaborative Filtering](#4-recommender-systems--collaborative-filtering)
5. [Recommender Systems — Content-Based Filtering](#5-recommender-systems--content-based-filtering)
6. [Principal Component Analysis (PCA)](#6-principal-component-analysis-pca)
7. [Reinforcement Learning — Core Concepts](#7-reinforcement-learning--core-concepts)
8. [Reinforcement Learning — Deep Q-Network (DQN)](#8-reinforcement-learning--deep-q-network-dqn)
9. [Full Workflow Examples](#9-full-workflow-examples)
10. [Quick Reference Cheatsheet](#10-quick-reference-cheatsheet)

---

## 1. Course 3 Overview

Course 3 moves away from "given labeled examples, predict an output" (Courses 1 & 2) into three new problem types:

| Topic | Type | Core Question |
|---|---|---|
| **Clustering** | Unsupervised | "Which points naturally group together?" |
| **Anomaly Detection** | Unsupervised | "Is this new point normal or weird?" |
| **Recommender Systems** | Supervised-ish (structured) | "What rating would this user give this item?" |
| **Reinforcement Learning** | Sequential decision-making | "What action should I take now to maximize future reward?" |

None of these have a single "correct" `y` label the way regression/classification did — they either have no labels at all (clustering, anomaly detection), sparse/partial labels (recommenders), or delayed labels that only arrive as rewards (RL).

---

## 2. K-Means Clustering

### Concept
K-means groups unlabeled data into `K` clusters by repeatedly assigning points to the nearest cluster center, then moving each center to the average of its assigned points.

### Algorithm (two-step loop)
1. **Assign step:** For every point, assign it to the closest centroid (by Euclidean distance).
2. **Update step:** Move each centroid to the mean of all points assigned to it.
3. Repeat until centroids stop moving (converged) or max iterations reached.

```
repeat:
    for i in range(m):
        c[i] = index of centroid closest to x[i]
    for k in range(K):
        mu[k] = mean of all x[i] where c[i] == k
```

### Cost Function (Distortion)
```
J = (1/m) * Σ ||x[i] - mu[c[i]]||²
```
K-means minimizes the average squared distance between each point and its assigned centroid. This value should **decrease every iteration** — if it ever goes up, there's a bug.

### Random Initialization
K-means can converge to a **local minimum** depending on where centroids start. Standard fix: run K-means many times (e.g. 50–1000 random initializations) and keep the run with the lowest cost J.

```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3, n_init=50, random_state=42)
model.fit(X)

labels    = model.labels_        # cluster assigned to each point
centroids = model.cluster_centers_
```

### Choosing K — The Elbow Method
Plot J (inertia) against different values of K. Look for the "elbow" where the cost stops dropping sharply.

```python
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

plt.plot(range(1, 11), inertias, marker='o')
plt.xlabel('K'); plt.ylabel('Inertia (J)')
plt.title('Elbow Method'); plt.show()
```

> ⚠️ In practice the elbow is often ambiguous. Usually K is chosen based on what's actually useful downstream (e.g. "we want 5 customer segments for 5 marketing campaigns"), not purely from the plot.

### Feature Scaling Matters
Just like linear regression, K-means uses distance — unscaled features (e.g. income in the 10,000s vs age in the 10s–90s) will dominate the distance calculation. Always `StandardScaler` before K-means.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('kmeans', KMeans(n_clusters=4, n_init=50, random_state=42)),
])
pipe.fit(X)
```

---

## 3. Anomaly Detection

### Concept
Instead of clustering, anomaly detection asks: "Given mostly normal examples, is this *new* example weird enough to flag?" Common uses: fraud detection, manufacturing defect detection, server health monitoring.

### Approach — Gaussian (Normal) Distribution
For each feature, fit a Gaussian distribution using the mean `μ` and variance `σ²` from **normal** training examples only.

```
p(x) = (1 / sqrt(2π·σ²)) * exp(-(x - μ)² / (2σ²))
```

For multiple features, assuming independence, multiply the probabilities:
```
p(x) = p(x₁; μ₁, σ₁²) * p(x₂; μ₂, σ₂²) * ... * p(xₙ; μₙ, σₙ²)
```

### Flagging Anomalies
```
if p(x) < ε:  flag as anomaly
else:         normal
```
`ε` (epsilon) is a threshold — chosen using a small labeled cross-validation set of known anomalies.

### Choosing ε — Use Precision/Recall, not Accuracy
Anomalies are rare (highly imbalanced), so accuracy is misleading — a model that never flags anything can still be 99.9% "accurate." Use **F1-score** on a cross-validation set with a handful of known anomalies to pick the best `ε`.

```python
from sklearn.metrics import f1_score
import numpy as np

def select_epsilon(p_val, y_val):
    best_eps, best_f1 = 0, 0
    for eps in np.linspace(p_val.min(), p_val.max(), 1000):
        preds = (p_val < eps).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_f1:
            best_f1, best_eps = f1, eps
    return best_eps, best_f1
```

### sklearn Equivalent
sklearn doesn't have a plain "Gaussian anomaly detector" out of the box the way Andrew Ng builds it by hand, but `EllipticEnvelope` and `IsolationForest` solve the same problem in practice:

```python
from sklearn.covariance import EllipticEnvelope

# assumes data is Gaussian — closest match to the course's method
detector = EllipticEnvelope(contamination=0.01, random_state=42)
detector.fit(X_train_normal)          # fit ONLY on normal examples
preds = detector.predict(X_val)       # 1 = normal, -1 = anomaly
```

```python
from sklearn.ensemble import IsolationForest

# tree-based, doesn't assume Gaussian shape — more robust in practice
detector = IsolationForest(contamination=0.01, random_state=42)
detector.fit(X_train)
preds = detector.predict(X_val)       # 1 = normal, -1 = anomaly
```

### Anomaly Detection vs Supervised Learning
| Use Anomaly Detection when... | Use Supervised Learning when... |
|---|---|
| Very few positive (anomaly) examples | Large number of positive AND negative examples |
| Many *different* types of anomalies, future anomalies may look nothing like past ones | Future positive examples likely look like the ones you've already seen |
| Examples: fraud, manufacturing defects, novel security threats | Examples: spam detection, weather prediction |

---

## 4. Recommender Systems — Collaborative Filtering

### Concept
Collaborative filtering predicts how a user would rate an item, based on ratings given by *other* users with similar taste — no knowledge of the item's actual content is required.

### Notation
| Symbol | Meaning |
|---|---|
| `n_u` | number of users |
| `n_m` | number of movies/items |
| `r(i,j)` | 1 if user j rated movie i, else 0 |
| `y(i,j)` | rating user j gave movie i (only defined if r(i,j)=1) |
| `w(j), b(j)` | parameters for user j |
| `x(i)` | feature vector for movie i (learned, not hand-designed) |

### Prediction
```
prediction for movie i, user j = w(j) · x(i) + b(j)
```

### Cost Function — Learn w, b, AND x Together
The key trick in collaborative filtering: since we don't have hand-labeled movie features, we **learn them** at the same time as we learn user parameters.

```
J(w,b,x) = (1/2) * Σ over all rated (i,j) [ (w(j)·x(i) + b(j) - y(i,j))² ]
           + (λ/2) * Σ w(j)²          ← regularize user params
           + (λ/2) * Σ x(i)²          ← regularize movie features
```

Both `w,b` (per user) and `x` (per movie) are updated simultaneously via gradient descent.

### Mean Normalization
Before training, subtract each movie's average rating from all ratings for that movie. This matters a lot for **new users who haven't rated anything** — without it, the model predicts 0 for them, which isn't a meaningful default. After normalization, a brand-new user's prediction defaults to the movie's average rating instead.

```python
import numpy as np

def normalize_ratings(Y, R):
    # Y: (n_movies, n_users) ratings matrix, R: same shape, 1 if rated
    Ymean = np.sum(Y * R, axis=1, keepdims=True) / (np.sum(R, axis=1, keepdims=True) + 1e-12)
    Ynorm = (Y - Ymean) * R
    return Ynorm, Ymean
```

### Implementation (NumPy — this is hand-rolled, sklearn has no direct equivalent)
```python
import numpy as np

def cofi_cost_function(X, W, b, Y, R, lambda_):
    """
    X: (n_movies, n_features)  — movie feature vectors (learned)
    W: (n_users,  n_features)  — user weight vectors
    b: (1, n_users)            — user biases
    Y: (n_movies, n_users)     — ratings
    R: (n_movies, n_users)     — 1 if rated, 0 otherwise
    """
    predictions = (X @ W.T) + b
    err = (predictions - Y) * R
    J = 0.5 * np.sum(err ** 2)
    J += (lambda_ / 2) * (np.sum(W ** 2) + np.sum(X ** 2))
    return J
```

```python
import tensorflow as tf

# TensorFlow's GradientTape is the standard tool the course uses for
# collaborative filtering, since it needs custom gradients w.r.t. X too
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

for epoch in range(200):
    with tf.GradientTape() as tape:
        predictions = tf.linalg.matmul(X, W, transpose_b=True) + b
        err = (predictions - Ynorm) * R
        cost = 0.5 * tf.reduce_sum(err ** 2)
        cost += (lambda_ / 2) * (tf.reduce_sum(W ** 2) + tf.reduce_sum(X ** 2))
    grads = tape.gradient(cost, [X, W, b])
    optimizer.apply_gradients(zip(grads, [X, W, b]))
```

### Finding Similar Items
Once `x(i)` (learned movie features) exist, similar movies are simply the ones with the smallest distance in feature space:
```
distance = ||x(i) - x(k)||²
```
This is the same idea as nearest-neighbors, but applied to *learned* features instead of raw ones.

### Limitation — Cold Start Problem
Collaborative filtering struggles with:
- **New users** — no ratings yet → nothing to base recommendations on
- **New items** — no one has rated it yet → no feature signal
- Doesn't naturally use side information (genre, price, user demographics)

Content-based filtering (next section) fixes this.

---

## 5. Recommender Systems — Content-Based Filtering

### Concept
Instead of learning movie features from scratch, content-based filtering uses **known features** of the user (age, location, past genres liked) and **known features** of the item (genre, year, actors) and learns a function that combines them.

### Two Neural Networks Trained Together
```
v_u = user_network(user_features)     # user embedding vector
v_m = movie_network(movie_features)   # movie embedding vector

prediction = v_u · v_m                # dot product = predicted rating/match
```

Both networks output vectors of the **same size** so their dot product is defined. They're trained jointly, minimizing the same squared-error cost as collaborative filtering, but now the "features" are real (age, genre, etc.) instead of learned-from-scratch.

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

num_outputs = 32  # embedding size, both networks must match

user_NN = tf.keras.Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_outputs),
])

item_NN = tf.keras.Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_outputs),
])

input_user = layers.Input(shape=(num_user_features,))
vu = user_NN(input_user)
vu = tf.linalg.l2_normalize(vu, axis=1)

input_item = layers.Input(shape=(num_item_features,))
vm = item_NN(input_item)
vm = tf.linalg.l2_normalize(vm, axis=1)

output = layers.Dot(axes=1)([vu, vm])
model = Model([input_user, input_item], output)

model.compile(optimizer='adam', loss='mse')
model.fit([user_train, item_train], y_train, epochs=30)
```

### Retrieval + Ranking (How Real Systems Scale)
Real-world recommenders (Netflix, YouTube) can't run the neural network against every single item for every user in real time. They use a two-step process:

1. **Retrieval:** Quickly generate a broad list of plausible candidates (e.g. "movies similar to ones you've watched," "top movies in your favorite genres") — cheap, approximate.
2. **Ranking:** Run the trained model only on that smaller candidate list to produce a precise, ranked final list — expensive but now feasible since the list is small.

| Collaborative Filtering | Content-Based Filtering |
|---|---|
| No item/user features needed | Requires known features |
| Suffers from cold-start | Handles cold-start reasonably |
| Simple linear model | Neural network, more flexible |
| Learns features implicitly | Uses features explicitly |

---

## 6. Principal Component Analysis (PCA)

### Concept
PCA reduces the number of features (dimensionality reduction) while keeping as much of the meaningful variation in the data as possible. Mainly used for **visualization** (compress to 2D/3D to plot high-dimensional data) rather than as a preprocessing step for supervised learning in this course.

### How It Works (Intuition)
PCA finds new axes ("principal components") that are:
1. Linear combinations of the original features
2. Ordered by how much variance in the data they capture
3. Uncorrelated with each other (orthogonal)

The first principal component points in the direction of maximum spread in the data; the second is the next-best direction orthogonal to the first, and so on.

### sklearn
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Always scale before PCA — variance-based method is scale-sensitive
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

print(pca.explained_variance_ratio_)   # % of variance each component captures

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.title('PCA Visualization'); plt.show()
```

> PCA is **not** feature selection — it creates brand new features that are combinations of the originals, so individual principal components usually aren't directly interpretable.

---

## 7. Reinforcement Learning — Core Concepts

### The Setup
An **agent** interacts with an **environment** over discrete time steps. At each step:
1. Agent observes the current **state** `s`
2. Agent picks an **action** `a`
3. Environment returns a **reward** `R(s)` and a new state `s'`

The agent's job is to learn a **policy** `π(s) → a` that picks the action maximizing long-term reward — not just the immediate reward.

### Return — Why "Long-Term" Matters
```
Return = R₁ + γR₂ + γ²R₃ + γ³R₄ + ...
```
`γ` (gamma) = **discount factor** (e.g. 0.9), a number slightly less than 1. It makes rewards received sooner more valuable than the same reward received later — the same logic as "money now is worth more than money later."

### Markov Decision Process (MDP)
The formal name for this setup. "Markov" means the future only depends on the **current** state — not on the full history of how you got there.

### State-Action Value Function — Q(s,a)
```
Q(s,a) = return if you:
         1. start in state s
         2. take action a (once)
         3. behave optimally after that
```
The best possible action in any state `s` is simply `argmax_a Q(s,a)` — whichever action has the highest Q-value.

### Bellman Equation
The core recursive relationship that RL algorithms are built on:
```
Q(s,a) = R(s) + γ * max_a' Q(s', a')
```
In words: *the value of taking action `a` in state `s` = the immediate reward, plus the discounted value of behaving optimally from whatever state you land in next.*

### Exploration vs Exploitation — ε-greedy Policy
While learning, the agent doesn't always take the action it currently thinks is best (exploitation) — it also needs to try random actions sometimes to discover better strategies (exploration).

```python
import random

def epsilon_greedy_action(Q_values, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)   # explore — random action
    else:
        return int(np.argmax(Q_values))            # exploit — best known action
```
`ε` typically starts high (e.g. 1.0, mostly random) and decays over training toward a small value (e.g. 0.01, mostly exploiting what's been learned).

### Continuous State Spaces
Real problems (e.g. a lunar lander) don't have a small countable set of states — position, velocity, and angle are all continuous real numbers. This is exactly why **Deep** Q-Learning (next section) exists: a neural network can approximate `Q(s,a)` for any continuous state, where a lookup table cannot.

---

## 8. Reinforcement Learning — Deep Q-Network (DQN)

### Concept
Instead of a lookup table for `Q(s,a)`, train a neural network to **approximate** it. Input: state (and action, or one output neuron per action). Output: predicted Q-value.

### Architecture (Typical)
```
Input: state vector (e.g. 8 numbers for Lunar Lander)
  ↓
Dense(64, relu)
  ↓
Dense(64, relu)
  ↓
Output: Dense(n_actions)   ← one Q-value per possible action, no activation
```

### Training Target — Bellman Equation as a Label
DQN turns RL into a supervised learning problem at each step: use the Bellman equation to compute a "target" Q-value, then train the network to predict it.

```
y_target = R(s) + γ * max_a' Q(s', a')     # using a snapshot/"target" network
```

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_q_network(state_dim, n_actions):
    model = tf.keras.Sequential([
        layers.Input(shape=(state_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_actions),   # linear output — raw Q-values
    ])
    return model

q_network      = build_q_network(state_dim=8, n_actions=4)
target_q_network = build_q_network(state_dim=8, n_actions=4)
target_q_network.set_weights(q_network.get_weights())  # start identical

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
```

### Experience Replay
Instead of training immediately on each single (state, action, reward, next_state) transition as it happens (which correlates badly with neighboring steps), store transitions in a buffer and train on **random mini-batches** sampled from that buffer.

```python
from collections import deque
import random

replay_buffer = deque(maxlen=100_000)

def store_experience(s, a, r, s_next, done):
    replay_buffer.append((s, a, r, s_next, done))

def sample_batch(batch_size=64):
    return random.sample(replay_buffer, batch_size)
```

### Target Network — Why Two Networks?
If the same network that's predicting Q-values is also used to *compute the targets it's trained on*, the target keeps shifting every update — training becomes unstable ("chasing a moving target"). Fix: keep a separate, slowly-updated **target network** just for computing `y_target`, and periodically sync its weights from the main network (a "soft update").

```python
def soft_update(q_network, target_q_network, tau=0.001):
    for target_w, main_w in zip(target_q_network.weights, q_network.weights):
        target_w.assign(tau * main_w + (1 - tau) * target_w)
```

### Full Training Step
```python
def train_step(batch, gamma=0.99):
    states, actions, rewards, next_states, dones = zip(*batch)
    states      = tf.convert_to_tensor(states, dtype=tf.float32)
    next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
    rewards     = tf.convert_to_tensor(rewards, dtype=tf.float32)
    dones       = tf.convert_to_tensor(dones, dtype=tf.float32)

    max_next_q = tf.reduce_max(target_q_network(next_states), axis=1)
    y_targets  = rewards + (1 - dones) * gamma * max_next_q

    with tf.GradientTape() as tape:
        q_values = q_network(states)
        action_masks = tf.one_hot(actions, depth=q_values.shape[1])
        q_selected = tf.reduce_sum(q_values * action_masks, axis=1)
        loss = tf.reduce_mean((q_selected - y_targets) ** 2)

    grads = tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(grads, q_network.trainable_variables))
    soft_update(q_network, target_q_network)
    return loss
```

---

## 9. Full Workflow Examples

### 9a. Clustering Workflow
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv('your_file.csv')
X = df[['feature1', 'feature2']].values

# elbow method
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(StandardScaler().fit_transform(X))
    inertias.append(km.inertia_)
plt.plot(range(1, 11), inertias, marker='o'); plt.show()

# final model
pipe = Pipeline([
    ('scale', StandardScaler()),
    ('kmeans', KMeans(n_clusters=4, n_init=50, random_state=42)),
])
pipe.fit(X)
df['cluster'] = pipe.predict(X)
```

### 9b. Anomaly Detection Workflow
```python
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score

detector = IsolationForest(contamination=0.02, random_state=42)
detector.fit(X_train_normal)

preds = detector.predict(X_val)              # 1 normal, -1 anomaly
preds_binary = (preds == -1).astype(int)     # 1 = anomaly, matches label convention
f1 = f1_score(y_val, preds_binary)
print(f'F1: {f1:.4f}')
```

### 9c. Collaborative Filtering Workflow
```python
import tensorflow as tf
import numpy as np

n_movies, n_users, n_features = Y.shape[0], Y.shape[1], 10
lambda_ = 1.0

X = tf.Variable(tf.random.normal((n_movies, n_features), stddev=0.1))
W = tf.Variable(tf.random.normal((n_users, n_features),  stddev=0.1))
b = tf.Variable(tf.zeros((1, n_users)))

Ynorm, Ymean = normalize_ratings(Y, R)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

for epoch in range(200):
    with tf.GradientTape() as tape:
        pred = tf.linalg.matmul(X, W, transpose_b=True) + b
        err = (pred - Ynorm) * R
        cost = 0.5 * tf.reduce_sum(err ** 2)
        cost += (lambda_ / 2) * (tf.reduce_sum(W ** 2) + tf.reduce_sum(X ** 2))
    grads = tape.gradient(cost, [X, W, b])
    optimizer.apply_gradients(zip(grads, [X, W, b]))
    if epoch % 20 == 0:
        print(f'epoch {epoch}, cost {cost:.2f}')

# predict for a specific user
predictions = (tf.linalg.matmul(X, W, transpose_b=True) + b).numpy() + Ymean
```

---

## 10. Quick Reference Cheatsheet

| Concept | Andrew Ng Notation | Practical Tool |
|---|---|---|
| Cluster assignment | `c(i)` | `KMeans().predict()` |
| Cluster centers | `μₖ` | `KMeans().cluster_centers_` |
| Distortion cost | `J` | `KMeans().inertia_` |
| Gaussian probability | `p(x)` | `EllipticEnvelope` / hand-rolled |
| Anomaly threshold | `ε` | tuned via F1-score |
| User params | `w(j), b(j)` | learned embedding + bias |
| Movie/item features | `x(i)` | learned embedding |
| Mean normalization | subtract row mean | `normalize_ratings()` |
| User/item embeddings (content-based) | `v_u, v_m` | output of two neural nets |
| Discount factor | `γ` (gamma) | typically 0.9–0.99 |
| State-action value | `Q(s,a)` | neural net output (DQN) |
| Bellman equation | `Q(s,a)=R(s)+γ·max Q(s',a')` | training target for DQN |
| Exploration rate | `ε` (epsilon) | decays over training |
| Experience storage | — | `deque` replay buffer |
| Stabilizing target | — | separate target network, soft update |
| Dimensionality reduction | — | `PCA(n_components=k)` |

### When to Use What (Course 3 Decision Guide)
| You have... | Use... |
|---|---|
| Unlabeled data, want to find natural groups | K-Means |
| Mostly-normal data, want to flag rare weird points | Anomaly Detection |
| User-item rating matrix, no item/user metadata | Collaborative Filtering |
| Known user features + known item features | Content-Based Filtering |
| High-dimensional data you want to visualize in 2D/3D | PCA |
| An agent that takes sequential actions in an environment for reward | Reinforcement Learning (DQN for continuous states) |

---