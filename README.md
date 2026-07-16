# Machine Learning Notes

Personal notes and code repository for Andrew Ng's Machine Learning Specialization (DeepLearning.AI / Coursera). Every course is documented with a written review (markdown), a full sklearn/TensorFlow code notebook, and the original in-course lab folders, all kept in one place as a running record of the learning process.

This is a working notes repo, not a polished course — see the [note on spelling and naming](#a-note-on-spelling-and-naming) below before judging the folder names too harshly.

---

## Repository Structure

```
Machine-Learning-Notes/
│
├── Advance_Learning_Algorithm/            # Course 2 — Advanced Learning Algorithms
│   ├── Decesion_trees/
│   ├── ml_practical_advice/
│   ├── nerual_networks/
│   ├── neural_network_training/
│   ├── course_2_review.md
│   └── tensorflow_trees_course2.ipynb
│
├── Supervised Machine Learning/           # Course 1 — Regression and Classification
│   ├── Classification/
│   ├── Introduction_To_Ml/
│   ├── Regression_with_multi_variable/
│   ├── course_1_review.md
│   └── sklearn_course1.ipynb
│
├── Unsupervised Learning/                 # Course 3 — Unsupervised Learning, Recommenders, RL
│   ├── Recommender_system/
│   ├── reinforcement_learning/
│   ├── unsupervised_learning/
│   │   ├── Images/
│   │   └── Labs/
│   ├── course_3_review.md
│   └── ml_specialization_course3.ipynb
│
├── .gitignore
├── .python-version
├── main.py
├── pyproject.toml
├── README.md
└── uv.lock
```

---

## What's Inside Each Course

### Course 1 — Supervised Machine Learning: Regression and Classification
`Supervised Machine Learning/`

- `Introduction_To_Ml/`, `Regression_with_multi_variable/`, `Classification/` — original in-course lab folders
- `course_1_review.md` — full written review: linear regression, cost function, gradient descent, vectorization, feature scaling, polynomial regression, overfitting/regularization, logistic regression
- `sklearn_course1.ipynb` — matching end-to-end sklearn workflow (linear regression, Ridge, polynomial features, logistic regression, bias/variance diagnostics)

### Course 2 — Advanced Learning Algorithms
`Advance_Learning_Algorithm/`

- `nerual_networks/`, `neural_network_training/`, `ml_practical_advice/`, `Decesion_trees/` — original in-course lab folders
- `course_2_review.md` — full written review: neural network architecture, activation functions, forward propagation, TensorFlow implementation, softmax/multiclass, Adam optimizer, bias/variance diagnostics, decision trees, tree ensembles
- `tensorflow_trees_course2.ipynb` — neural networks (binary + multiclass), decision trees, random forest, and XGBoost all built and compared side by side on the same data

### Course 3 — Unsupervised Learning, Recommenders, Reinforcement Learning
`Unsupervised Learning/`

- `unsupervised_learning/` — original in-course lab folder, includes `Images/` (lecture screenshots/diagrams used as reference) and `Labs/` (original notebooks)
- `Recommender_system/`, `reinforcement_learning/` — original in-course lab folders
- `course_3_review.md` — full written review: K-means clustering, anomaly detection, collaborative filtering, content-based filtering, PCA, reinforcement learning and the Bellman equation, Deep Q-Networks
- `ml_specialization_course3.ipynb` — hand-rolled + sklearn/TensorFlow implementations of everything above, including a custom GridWorld environment for the DQN section

---

## Course Review Files

| File | Covers |
|---|---|
| `course_1_review.md` | Linear regression, cost function, gradient descent, feature scaling, polynomial regression, regularization, logistic regression |
| `course_2_review.md` | Neural networks, activation functions, TensorFlow, softmax, Adam, bias/variance, decision trees, tree ensembles |
| `course_3_review.md` | K-means, anomaly detection, collaborative filtering, content-based filtering, PCA, reinforcement learning, DQN |

Each review follows the same format: concept explanation, the math/notation from the lectures, and the matching sklearn or TensorFlow code.

## Notebooks

| Notebook | Contents |
|---|---|
| `sklearn_course1.ipynb` | Linear/Ridge/Polynomial regression, logistic regression, full diagnostic workflow |
| `tensorflow_trees_course2.ipynb` | Neural nets (binary + multiclass), decision tree, random forest, XGBoost, model comparison |
| `ml_specialization_course3.ipynb` | K-means, anomaly detection, collaborative filtering, content-based filtering, PCA, DQN |

All three notebooks are self-contained and runnable end to end on synthetic/placeholder data — swap in a real CSV or dataset to apply them to an actual project.

---

## Technologies Used

- **Python 3.12**
- **[scikit-learn](https://scikit-learn.org/)** — pipelines, preprocessing, classic ML models
- **[TensorFlow / Keras](https://www.tensorflow.org/)** — neural networks, custom training loops (GradientTape)
- **[XGBoost](https://xgboost.readthedocs.io/)** — boosted trees
- **NumPy / pandas / matplotlib** — data handling and visualization
- **Jupyter Notebooks** — all code notebooks
- **[uv](https://docs.astral.sh/uv/)** — dependency management

---

## How to Run Locally

### Prerequisites
- Python 3.12
- `uv` package manager

### Setup
```bash
git clone <repository-url>
cd Machine-Learning-Notes

uv sync
```

### Open a notebook
```bash
uv run jupyter notebook
```
Then open any of `sklearn_course1.ipynb`, `tensorflow_trees_course2.ipynb`, or `ml_specialization_course3.ipynb`.

---

## A Note on Spelling and Naming

These are personal notes typed while going through the lectures, not a cleaned-up publication, so a few folder names and bits of the written reviews contain spelling mistakes — for example `Decesion_trees` instead of `Decision_trees`, and `nerual_networks` instead of `neural_networks`. These are left as-is on purpose rather than silently fixed after the fact, so anyone browsing the repo should expect the occasional typo rather than assume it's a rendering issue.

---

## Author

Muhammad Awais Tariq

## References

- [Machine Learning Specialization — DeepLearning.AI / Coursera](https://www.coursera.org/specializations/machine-learning-introduction)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)