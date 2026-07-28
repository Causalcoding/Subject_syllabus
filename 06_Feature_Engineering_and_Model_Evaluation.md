# Feature Engineering and Model Evaluation — Interview Prep Syllabus

Feature engineering and model evaluation are the two skills that separate candidates who can "call `.fit()`" from candidates who can ship a model that survives production. For a **Data Scientist**, this is the daily bread-and-butter: turning raw, messy business data into signal, and proving to stakeholders (with the right metric, not just accuracy) that a model actually works. For a **Machine Learning Engineer**, feature pipelines must be leak-free, reproducible, and identical between training and serving — interviewers probe hard on leakage, encoding at scale, and how evaluation gates a deployment. For an **AI Engineer** working with LLMs, RAG, and agentic systems, the same underlying principles reappear as feature stores for retrieval ranking, embedding-based encodings for high-cardinality categoricals, calibration of model confidence, and statistically rigorous A/B or offline evaluation of prompts/pipelines. Interviewers across all three roles use this topic to test whether you understand *why* a technique works, not just how to call it in `sklearn`.

---

## Table of Contents

1. [Feature Engineering](#feature-engineering)
   - [Categorical Encoding](#categorical-encoding)
   - [Numerical Feature Transforms](#numerical-feature-transforms)
   - [Datetime Features and Cyclical Encoding](#datetime-features-and-cyclical-encoding)
   - [Text and Image Features as Engineered Inputs](#text-and-image-features-as-engineered-inputs)
   - [Feature Selection](#feature-selection)
   - [Dimensionality Reduction](#dimensionality-reduction)
   - [Handling Imbalanced Datasets](#handling-imbalanced-datasets)
   - [Feature Leakage](#feature-leakage)
   - [Interview Questions — Feature Engineering](#interview-questions--feature-engineering)
2. [Model Evaluation](#model-evaluation)
   - [Data Splitting Strategies](#data-splitting-strategies)
   - [Classification Metrics](#classification-metrics)
   - [Regression Metrics](#regression-metrics)
   - [Ranking / Recommendation Metrics](#ranking--recommendation-metrics)
   - [Calibration of Probabilistic Predictions](#calibration-of-probabilistic-predictions)
   - [Bias-Variance Tradeoff](#bias-variance-tradeoff)
   - [Overfitting, Underfitting, and Regularization](#overfitting-underfitting-and-regularization)
   - [Hyperparameter Tuning](#hyperparameter-tuning)
   - [Nested Cross-Validation](#nested-cross-validation)
   - [Statistical Significance of Model Comparisons](#statistical-significance-of-model-comparisons)
   - [Interview Questions — Model Evaluation](#interview-questions--model-evaluation)
3. [Rapid-Fire Interview Q&A](#rapid-fire-interview-qa)

---

## Feature Engineering

### Categorical Encoding

Categorical variables must be converted to numeric representations before most ML algorithms can consume them. The choice of encoding affects model performance, training speed, memory, and — critically — leakage risk.

#### One-Hot Encoding

Creates one binary column per category. For a feature with $k$ categories, produces $k$ (or $k-1$ to avoid the "dummy variable trap" for linear models) columns.

```python
import pandas as pd
df = pd.DataFrame({"color": ["red", "blue", "green", "blue"]})
pd.get_dummies(df, columns=["color"], drop_first=False)
#    color_blue  color_green  color_red
# 0       0            0          1
# 1       1            0          0
# 2       0            1          0
# 3       1            0          0
```

- **When to use:** low-to-medium cardinality (roughly < 20–50 categories), nominal (unordered) categories, linear models / tree models where you don't want a false ordinal relationship.
- **Pitfalls:**
  - Curse of dimensionality with high cardinality (e.g., zip codes, user IDs) — sparse, memory-heavy matrices.
  - Train/test category mismatch: unseen categories at inference time produce all-zero rows unless explicitly handled (`handle_unknown="ignore"` in `sklearn.OneHotEncoder`).
  - Multicollinearity for linear regression if you keep all $k$ columns (use `drop_first=True` / drop one level).

#### Label / Ordinal Encoding

Maps each category to an integer, e.g., `{low: 0, medium: 1, high: 2}`.

```python
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder(categories=[["low", "medium", "high"]])
enc.fit_transform(df[["risk_level"]])
```

- **When to use:** truly ordinal features (education level, satisfaction rating) where order carries meaning, and for tree-based models (trees can split on arbitrary thresholds, so an arbitrary integer mapping for *nominal* data is often tolerable, though not ideal).
- **Pitfalls:** Using label encoding on **nominal** data fed into a linear/distance-based model (e.g., KNN, linear regression) implies a false numeric ordering/distance (`red=0, green=1, blue=2` implies blue is "further" from red than green is) — this is a classic interview trap.

#### Target / Mean Encoding

Replace each category with a statistic of the target variable computed over training rows with that category, most commonly the mean of the target.

$$\text{Encode}(c) = \frac{1}{n_c}\sum_{i: x_i = c} y_i$$

Smoothed (Bayesian) version blends the category mean with the global mean, weighted by category frequency $n_c$ and a smoothing factor $m$:

$$\text{Encode}(c) = \frac{n_c \cdot \bar{y}_c + m \cdot \bar{y}_{\text{global}}}{n_c + m}$$

```python
import pandas as pd
import numpy as np

def smoothed_target_encode(train, col, target, m=10):
    global_mean = train[target].mean()
    agg = train.groupby(col)[target].agg(["mean", "count"])
    smooth = (agg["count"] * agg["mean"] + m * global_mean) / (agg["count"] + m)
    return smooth  # mapping dict: category -> encoded value
```

- **When to use:** high-cardinality categoricals in tree-based / gradient boosting models (very effective for LightGBM/XGBoost/CatBoost pipelines), especially when one-hot would blow up dimensionality.
- **Leakage risk (critical interview topic):** if you compute the encoding using the *entire* training set including the row being encoded, you leak the target into the feature — the model effectively "sees" the label. Symptoms: suspiciously high CV/train performance that collapses in production.
  - **Fix 1 — Out-of-fold encoding:** compute encodings using K-fold: for each fold, encode using statistics from the *other* folds only.
  - **Fix 2 — Leave-one-out encoding:** exclude the current row's own target when computing its group mean.
  - **Fix 3 — Expanding/time-based encoding:** for time-series, only use past data up to time $t$ to encode row at time $t$.
  - Always fit the encoding map on the **training fold only**, then apply (transform) to validation/test — never fit on the full dataset before splitting.

```python
from sklearn.model_selection import KFold
import numpy as np

def oof_target_encode(train_df, col, target, n_splits=5, m=10):
    oof = np.zeros(len(train_df))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    global_mean = train_df[target].mean()
    for tr_idx, val_idx in kf.split(train_df):
        tr, val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
        agg = tr.groupby(col)[target].agg(["mean", "count"])
        mapping = (agg["count"] * agg["mean"] + m * global_mean) / (agg["count"] + m)
        oof[val_idx] = val[col].map(mapping).fillna(global_mean).values
    return oof
```

#### Frequency Encoding

Replace each category with its occurrence count/frequency in the training data.

```python
freq = df["city"].value_counts(normalize=True)
df["city_freq"] = df["city"].map(freq)
```

- **When to use:** quick baseline feature, useful signal when category prevalence itself correlates with the target (e.g., popular products sell differently than niche ones). Cheap, low leakage risk (does not use the target), but less informative than target encoding.
- **Pitfall:** unseen categories at test time get no frequency (map to 0 or a fallback); two unrelated categories with the same frequency get collapsed to the same encoded value (a form of information loss).

#### Embeddings for High-Cardinality Categoricals

Learn a dense, low-dimensional vector per category (e.g., 8–50 dims) jointly with the model — the standard approach in deep learning (entity embeddings, popularized by the "Rossmann" Kaggle solution, and standard in recommender systems for user/item IDs).

```python
import torch.nn as nn
embedding = nn.Embedding(num_embeddings=100_000, embedding_dim=16)  # e.g. user_id
```

- **When to use:** very high cardinality (user IDs, product SKUs, words) in a neural network context, where one-hot/target encoding either explodes dimensionality or discards relational structure. Embeddings can capture similarity (similar users cluster nearby in embedding space).
- **Pitfalls:** requires enough data per category to learn a meaningful embedding (cold-start problem for rare/new categories — mitigate with a shared "unknown" bucket or hashing); not directly usable by non-NN models (though can be pre-trained and exported as static features, e.g., `word2vec`-style category embeddings for use in GBMs).

#### Hashing Trick

Map categories to a fixed number of buckets via a hash function, `bucket = hash(category) % n_buckets`, avoiding the need to store a category-to-index dictionary at all.

```python
from sklearn.feature_extraction import FeatureHasher
h = FeatureHasher(n_features=1024, input_type="string")
X = h.transform([["red"], ["blue"], ["green"]])
```

- **When to use:** streaming/online learning where the category vocabulary is unknown in advance or unbounded (e.g., URLs, ad IDs), memory-constrained environments, or when you want a stateless, deployable-anywhere transform (no fitted vocabulary to ship).
- **Pitfalls:** hash collisions merge unrelated categories into the same bucket, adding noise (mitigated by using more buckets, or the signed hashing trick which adds ±1 to reduce collision bias); encoded features are not interpretable (you can't map a bucket back to a category name).

**Encoding method comparison:**

| Method | Cardinality fit | Leakage risk | Model compatibility | Interpretability |
|---|---|---|---|---|
| One-hot | Low–medium | None | All models | High |
| Label/Ordinal | Any (best for true ordinal) | None | Trees best; risky for linear/distance models on nominal data | High |
| Target/Mean | High | **High if done wrong** | Best for trees/GBMs | Medium |
| Frequency | High | Low | All models | Medium |
| Embeddings | Very high | None (learned, not target-leaked, but still supervised) | Neural nets primarily | Low |
| Hashing | Unbounded/streaming | None | All models | Very low |

---

### Numerical Feature Transforms

#### Scaling

| Method | Formula | Range/Property | Sensitive to outliers? |
|---|---|---|---|
| Min-Max | $x' = \dfrac{x - x_{min}}{x_{max} - x_{min}}$ | $[0, 1]$ | Yes — very sensitive |
| Standardization (Z-score) | $x' = \dfrac{x - \mu}{\sigma}$ | mean 0, std 1 | Somewhat |
| Robust Scaler | $x' = \dfrac{x - \text{median}}{\text{IQR}}$ | Centered on median | No — robust by design |

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
X_scaled = StandardScaler().fit_transform(X_train)  # fit on TRAIN ONLY, transform test with same scaler
```

- **When to use:**
  - Min-Max: neural networks (bounded activations like sigmoid), image pixel intensities, algorithms needing bounded input.
  - Standardization: linear/logistic regression, SVM, PCA, KNN, anything using L2 distance or gradient descent with mixed-scale features (prevents features with large numeric ranges from dominating the loss/distance).
  - Robust Scaler: data with heavy outliers (financial/sensor data) where min-max or z-score would be skewed by extremes.
- **Trees (Random Forest, GBM) generally do NOT need scaling** — splits are based on order/thresholds, invariant to monotonic transforms. Interviewers love asking this to check you understand *why* scaling matters (gradient-based / distance-based methods) vs. doesn't (axis-aligned split-based methods).
- **Pitfall — the #1 scaling leakage bug:** fitting the scaler on the full dataset (train+test) before splitting leaks test-set statistics (mean, std, min, max) into training. Always `fit` on training data only, then `transform` validation/test data with those same fitted parameters.

#### Log, Box-Cox, and Power Transforms

Used to reduce skewness, stabilize variance, and make distributions closer to Gaussian (helps linear models' assumptions and can improve gradient-based optimization).

- **Log transform:** $x' = \log(x + c)$ (add constant $c$ if zeros/negatives present, e.g., $\log(1+x)$ via `np.log1p`). Good for right-skewed positive data (income, prices, counts).
- **Box-Cox:** requires strictly positive $x$.

$$x'(\lambda) = \begin{cases} \dfrac{x^\lambda - 1}{\lambda} & \lambda \neq 0 \\ \log(x) & \lambda = 0 \end{cases}$$

$\lambda$ is estimated via maximum likelihood to best normalize the distribution.

- **Yeo-Johnson:** generalization of Box-Cox that supports zero and negative values.

```python
from sklearn.preprocessing import PowerTransformer
import numpy as np

pt = PowerTransformer(method="yeo-johnson")  # or "box-cox" (requires x > 0)
X_transformed = pt.fit_transform(X_train)
print(pt.lambdas_)  # optimal lambda per feature
```

- **When to use:** skewed numeric features feeding into linear regression, linear SVM, or any model whose loss assumes roughly Gaussian residuals; also useful before PCA since PCA is sensitive to variance dominance from skewed/large-scale features.
- **Pitfalls:** target-variable log transforms require back-transforming predictions ($\hat{y} = e^{\hat{y}'} - 1$) and introduce bias if you naively exponentiate (Jensen's inequality — the mean of a log-transformed prediction under-predicts the true mean; correct with a bias correction like $e^{\hat{y}' + \sigma^2/2}$ for log-normal residuals). Fit `lambda`/parameters on training data only.

#### Binning / Discretization

Convert continuous variables into discrete bins/buckets.

```python
import pandas as pd
df["age_bin"] = pd.cut(df["age"], bins=[0, 18, 35, 60, 100], labels=["minor", "young_adult", "adult", "senior"])
# Equal-frequency (quantile) binning:
df["income_qbin"] = pd.qcut(df["income"], q=4, labels=False)
```

- **When to use:** capturing non-linear effects in linear models (e.g., a "U-shaped" relationship between age and risk that a single linear coefficient can't capture); creating interpretable buckets for business rules/credit scoring (WOE — weight of evidence — binning is standard in credit risk models); reducing the impact of noisy/outlier numeric values.
- **Pitfalls:** loses information (two very different ages in the same bin are treated identically); bin boundaries chosen post-hoc on the full data (including test) leak information — choose bins based on domain knowledge or training-data-only quantiles; arbitrary bin edges can introduce discontinuities a smooth model wouldn't have.

#### Polynomial and Interaction Features

Explicitly create products and powers of existing features so that a linear model can capture non-linear/interaction effects.

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X[["x1", "x2"]])
# columns: x1, x2, x1^2, x1*x2, x2^2
```

- **When to use:** linear/logistic regression where you suspect interaction effects (e.g., `price * promotion_flag`) or curvature (e.g., `age^2` for a U-shaped effect); when a domain expert flags a known multiplicative relationship.
- **Pitfalls:** feature count grows combinatorially ($O(d^2)$ for degree 2, $O(d^k)$ for degree $k$) — causes overfitting and multicollinearity; almost always needs regularization (Ridge/Lasso) paired with it; tree-based models can often learn interactions natively and don't need explicit polynomial features.
- **Detecting the multicollinearity this introduces:** polynomial/interaction terms are by construction correlated with their parent features (e.g., `x1` and `x1^2` are often highly correlated over a limited range), so it's worth screening the engineered feature set with the **Variance Inflation Factor (VIF)** before feeding it to a linear model. VIF and the linear-algebra reasoning behind it (why it makes $(X^TX)^{-1}$ ill-conditioned, and why Ridge fixes this) are covered in depth in the **Machine Learning Fundamentals** syllabus file's linear regression section — as a practical rule of thumb, a newly engineered feature with VIF > 10 relative to the rest of the feature set is a signal to drop it, combine it with its correlated partner, or lean on Ridge/Elastic Net rather than fitting plain OLS.

---

### Datetime Features and Cyclical Encoding

Raw timestamps are rarely useful directly — you must decompose them into components and encode periodicity correctly.

```python
import pandas as pd
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["day_of_week"] = df["timestamp"].dt.dayofweek       # 0=Monday
df["hour"] = df["timestamp"].dt.hour
df["is_weekend"] = df["day_of_week"] >= 5
df["days_since_epoch"] = (df["timestamp"] - pd.Timestamp("1970-01-01")).dt.days
```

**The cyclical encoding problem:** hour 23 and hour 0 are numerically far apart (`|23-0|=23`) but are actually adjacent in time (1 hour apart). A plain integer or one-hot encoding either loses this adjacency (integer) or wastes dimensions and still doesn't capture the *distance* structure (one-hot). The fix is to project the cyclical feature onto a circle using sine and cosine:

$$x_{\sin} = \sin\left(\frac{2\pi \cdot t}{T}\right), \qquad x_{\cos} = \cos\left(\frac{2\pi \cdot t}{T}\right)$$

where $T$ is the period (24 for hour-of-day, 7 for day-of-week, 12 for month, 365.25 for day-of-year).

```python
import numpy as np
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
```

- **Why both sin AND cos are needed:** sine alone is not injective over a period (`sin(0) == sin(π)` in value pattern maps two different times to related y-values); the pair `(sin, cos)` uniquely identifies a point on the unit circle, preserving true cyclical distance — e.g., Euclidean distance between `(sin,cos)` pairs for hour 23 and hour 0 is small, correctly reflecting their proximity.
- **Other datetime feature ideas:** time since a reference event (recency), lag features and rolling aggregates for time series (`rolling(7).mean()`), holiday/event flags, business-day indicators, "is_month_end", fiscal quarter.
- **Pitfalls:**
  - Extracting date parts *after* an incorrect timezone conversion silently shifts hour/day features.
  - For tree models, raw cyclical integers (e.g., `hour` as 0–23) can actually work reasonably well because trees split at learned thresholds and CAN partially recover cyclic structure through multiple splits — but they still can't wrap around (a split can never say "hour ≤ 2 OR hour ≥ 22"), so sin/cos still helps.
  - Computing rolling/lag features without respecting row order or without grouping by entity (e.g., per-customer) leaks information across unrelated groups or future rows.

---

### Text and Image Features as Engineered Inputs

A brief orientation — deep dives belong in dedicated NLP and Computer Vision syllabus files.

- **Text → features (classical):** bag-of-words / TF-IDF vectors, n-gram counts, handcrafted features (text length, punctuation count, sentiment score, readability score), topic-model outputs (LDA topic proportions) as dense features feeding into a downstream classical model (e.g., GBM on TF-IDF + metadata).

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_text = tfidf.fit_transform(df["review_text"])
```

- **Text → features (modern):** dense sentence/document embeddings from pretrained transformer encoders (e.g., `sentence-transformers`) used as fixed-size numeric feature vectors feeding a downstream model or similarity search — this is the standard "feature engineering" step in RAG pipelines and semantic search (AI Engineer relevance).
- **Image → features (classical):** color histograms, edge/texture descriptors (HOG, SIFT) — largely superseded by learned representations.
- **Image → features (modern):** embeddings from a pretrained CNN/ViT backbone (e.g., penultimate layer of ResNet/CLIP) used as fixed feature vectors for a downstream classifier, clustering, or similarity search — "transfer learning as feature engineering."
- **Cross-cutting pitfall:** fitting a TF-IDF vocabulary or vectorizer on the *entire* corpus (train+test) before splitting leaks vocabulary/IDF statistics; always fit text vectorizers/embeddings pipelines on training data only, exactly like any other encoder.
- See the dedicated **NLP** and **Computer Vision** syllabus files for full depth on tokenization, embeddings, and vision architectures.

---

### Feature Selection

Feature selection reduces overfitting, improves interpretability, reduces training/inference cost, and mitigates the curse of dimensionality.

#### Filter Methods

Evaluate each feature independently of any model, using a statistical measure of relationship with the target. Fast, model-agnostic, but ignore feature interactions.

| Method | Use case | Notes |
|---|---|---|
| Pearson correlation | Numeric feature vs numeric target | Captures only linear relationships |
| Chi-square test | Categorical feature vs categorical target | Requires non-negative frequencies |
| Mutual information | Any feature type vs any target | Captures non-linear dependence |
| ANOVA F-test | Numeric feature vs categorical target | Assumes normality/equal variance |
| Variance threshold | Any feature (unsupervised) | Removes near-constant features |

```python
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif, VarianceThreshold

chi2_scores, p_values = chi2(X_categorical, y)
mi_scores = mutual_info_classif(X, y, discrete_features="auto")
f_scores, p_values = f_classif(X_numeric, y)

vt = VarianceThreshold(threshold=0.01)
X_reduced = vt.fit_transform(X)
```

- **Mutual information** $I(X;Y) = \sum_{x,y} p(x,y)\log\frac{p(x,y)}{p(x)p(y)}$ measures general statistical dependence (linear or not), making it more powerful than correlation for capturing non-monotonic relationships, at the cost of needing more data to estimate reliably and requiring discretization for continuous variables.
- **Variance threshold:** drop features with variance below a cutoff (e.g., a column that's 99.9% one value) — cheap first-pass cleanup, unsupervised (doesn't look at the target), so it's safe to run before any train/test split decisions.

#### Wrapper Methods

Use a model's actual performance to guide feature subset search. More expensive, but capture feature interactions and are optimized for the specific downstream model.

**Recursive Feature Elimination (RFE):** train the model, rank features by importance/coefficient magnitude, remove the weakest, repeat until the desired number remains.

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

estimator = LogisticRegression(max_iter=1000)
selector = RFE(estimator, n_features_to_select=10, step=1)
selector.fit(X_train, y_train)
print(selector.support_, selector.ranking_)
```

- **RFECV** wraps RFE in cross-validation to automatically choose the optimal number of features rather than a fixed count.
- **Pitfalls:** computationally expensive (retrains the model at every elimination step — $O(d)$ or more model fits); results are model-specific (features good for a linear model's RFE ranking may not be optimal for a tree model); must be done inside a CV loop / on training folds only to avoid selection leakage (choosing features using information from the validation/test fold biases evaluation optimistically).

#### Embedded Methods

Feature selection happens as a byproduct of model training itself.

- **L1 / Lasso regularization:** the $L1$ penalty $\lambda \sum_j |\beta_j|$ drives many coefficients exactly to zero, performing automatic feature selection.

```python
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5).fit(X_train, y_train)
selected_features = X_train.columns[lasso.coef_ != 0]
```

- **Tree-based feature importance:** Gini importance / mean decrease in impurity, or permutation importance (more reliable — measures the actual drop in performance when a feature's values are shuffled).

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
# Impurity-based (biased toward high-cardinality/continuous features):
importances = rf.feature_importances_
# Permutation-based (more trustworthy, model-agnostic):
perm = permutation_importance(rf, X_val, y_val, n_repeats=10, random_state=42)
```

- **Pitfall with impurity-based importance:** biased toward high-cardinality and continuous features (they get more opportunities to be chosen as a split), and computed on training data by default, which can overstate importance for features the model overfit to. Permutation importance computed on a held-out set is the more defensible choice in interviews.

**Filter vs Wrapper vs Embedded comparison:**

| Aspect | Filter | Wrapper | Embedded |
|---|---|---|---|
| Speed | Fastest | Slowest | Moderate |
| Considers feature interactions | No | Yes | Partially |
| Model-specific | No | Yes | Yes |
| Overfitting risk | Low | Higher (if not CV'd) | Moderate |
| Example | Correlation, chi-square | RFE | Lasso, tree importance |

---

### Dimensionality Reduction

#### PCA (Principal Component Analysis)

PCA finds an orthogonal linear transformation of the data into a new coordinate system where axes (principal components) are ordered by the amount of variance they explain.

**Math:** Given centered data matrix $X$ ($n \times d$), compute the covariance matrix $\Sigma = \frac{1}{n-1}X^TX$. Find eigenvectors $v_1, v_2, \ldots$ and eigenvalues $\lambda_1 \geq \lambda_2 \geq \ldots$ of $\Sigma$. The top-$k$ eigenvectors (by eigenvalue) form the projection matrix $W$. Project: $Z = XW$. Equivalently derived via Singular Value Decomposition $X = U S V^T$, where the columns of $V$ are the principal components and $S$ contains singular values ($\lambda_i = s_i^2/(n-1)$).

Each component's **explained variance ratio** is $\lambda_i / \sum_j \lambda_j$.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X_train)   # PCA is scale-sensitive — always standardize first
pca = PCA(n_components=0.95)   # keep enough components to explain 95% of variance
Z = pca.fit_transform(X_scaled)
print(pca.explained_variance_ratio_)
```

- **Intuition:** PCA finds the directions of maximum spread in the data and re-expresses points using fewer coordinates while minimizing reconstruction (squared) error — it is the optimal *linear* lossy compression under an L2 objective.
- **When to use:** reducing dimensionality before a distance-based or linear model, de-correlating features (e.g., before linear regression to fight multicollinearity), compressing data for storage/visualization, denoising (dropping low-variance components often drops noise).
- **Pitfalls:**
  - **Interpretability loss:** principal components are linear combinations of original features and often lack clear business meaning — a huge issue when stakeholders demand "why did the model predict this."
  - PCA is sensitive to feature scale — always standardize before PCA, or a large-scale feature will dominate the variance and the components.
  - PCA is unsupervised — it maximizes variance, not predictive power for the target; the top components are not guaranteed to be the most predictive ones (a low-variance component can still be the most useful signal).
  - Assumes linear structure — cannot capture non-linear manifolds (use kernel PCA, t-SNE, UMAP, or autoencoders instead).
  - Must fit PCA on training data only, then transform validation/test with the same fitted components (leakage risk identical to scaling).

#### t-SNE (t-distributed Stochastic Neighbor Embedding)

A non-linear technique that models pairwise similarities in high-dimensional space (via a Gaussian kernel) and tries to reproduce them in low-dimensional space (2D/3D) using a heavy-tailed Student-t distribution, minimizing KL divergence between the two similarity distributions.

```python
from sklearn.manifold import TSNE
Z = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_scaled)
```

- **When to use:** exploratory visualization of clusters/structure in high-dimensional data (e.g., visualizing embeddings, verifying that a classifier's learned representations separate classes).
- **Pitfalls (major interview topic — "pitfalls of interpreting reduced dimensions"):**
  - **Cluster sizes and inter-cluster distances in a t-SNE plot are NOT meaningful.** t-SNE only tries to preserve *local* neighborhood structure; global distances and apparent cluster density are artifacts of the optimization, not real relationships in the original space.
  - Results are highly sensitive to the `perplexity` hyperparameter and to random initialization — re-running with a different seed can produce visually different layouts.
  - t-SNE has no simple "transform" for new data — it's not naturally suited for out-of-sample projection (must re-run or use parametric variants), unlike PCA which has a reusable projection matrix.
  - Computationally expensive for large $n$ (naive is $O(n^2)$; use Barnes-Hut approximation, `sklearn`'s default, for better scaling).
  - Should not be used as a preprocessing step before a downstream supervised model (it distorts distances) — it is a **visualization tool**, not a general-purpose dimensionality-reduction feature engineering step.

#### UMAP (Uniform Manifold Approximation and Projection)

A more recent non-linear technique based on manifold learning and topological data analysis (fuzzy simplicial sets), generally faster than t-SNE and better at preserving both local *and* some global structure.

```python
import umap
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
Z = reducer.fit_transform(X_scaled)
```

- **When to use:** similar use cases to t-SNE (visualization, clustering pre-processing) but preferred when speed matters (scales better to large datasets) or when downstream clustering on the reduced space is needed (UMAP embeddings are more often usable as actual features for clustering, unlike t-SNE).
- **Pitfalls:** still has hyperparameters (`n_neighbors`, `min_dist`) that materially change the output shape; global distances are more trustworthy than t-SNE's but still should be interpreted cautiously; like t-SNE, it is non-parametric by default though it does support `.transform()` for new points (an advantage over classic t-SNE).

**PCA vs t-SNE vs UMAP:**

| Aspect | PCA | t-SNE | UMAP |
|---|---|---|---|
| Linear/Non-linear | Linear | Non-linear | Non-linear |
| Preserves | Global variance | Local structure only | Local + some global structure |
| Speed | Fast, scales well | Slow ($O(n^2)$ naive) | Faster than t-SNE |
| Out-of-sample transform | Yes (native) | No (not naturally) | Yes (supported) |
| Deterministic | Yes | No (seed-dependent) | Mostly (seed-dependent but more stable) |
| Use as ML feature input | Yes, common | Generally no (visualization only) | Sometimes, with caution |
| Interpretability of axes | Loadings interpretable | Not interpretable | Not interpretable |

---

### Handling Imbalanced Datasets

When one class vastly outnumbers another (e.g., fraud: 0.1% positive), naive training and accuracy-based evaluation are misleading — a model predicting "always negative" gets 99.9% accuracy while being useless.

#### Class Weighting

Penalize misclassification of the minority class more heavily in the loss function, without changing the data itself.

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight="balanced")   # weight_c = n_samples / (n_classes * n_samples_c)
# or explicit:
model = LogisticRegression(class_weight={0: 1, 1: 20})
```

For gradient boosting: `scale_pos_weight` in XGBoost, `class_weight` in LightGBM.

- **When to use:** almost always a good first thing to try — cheap, no data duplication/loss, works directly with the model's native loss function.

#### Oversampling (SMOTE, ADASYN)

**Random oversampling:** duplicate minority-class samples until balanced. Simple but can cause overfitting to duplicated points.

**SMOTE (Synthetic Minority Oversampling Technique):** generates *synthetic* minority samples by interpolating between a minority sample and one of its $k$ nearest minority-class neighbors:

$$x_{\text{new}} = x_i + \lambda \cdot (x_{\text{neighbor}} - x_i), \quad \lambda \sim \text{Uniform}(0, 1)$$

```python
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)   # fit ONLY on training data
```

**ADASYN (Adaptive Synthetic Sampling):** like SMOTE but generates more synthetic samples in regions where the minority class is harder to learn (near the decision boundary, i.e., where minority samples have more majority-class neighbors) — adaptively focuses effort where the classifier struggles most.

- **When to use:** minority class has too few examples for the model to learn meaningful patterns via weighting alone; works best when minority-class feature space is reasonably continuous/dense (SMOTE interpolation assumes nearby points are semantically valid — problematic for categorical-heavy or sparse data).
- **Pitfalls:**
  - **Never apply SMOTE/oversampling before the train/test split** — synthetic points derived from information "leaking" across the split will inflate test performance. Always split first, then resample only the training set.
  - Similarly, if using cross-validation, resampling must happen *inside* each fold (e.g., via an `imblearn.pipeline.Pipeline`), not before the CV loop.
  - SMOTE can generate noisy synthetic samples in overlapping class regions, exacerbating boundary confusion.
  - Oversampling does not add new information — it re-weights the loss surface implicitly and can lead to overconfident models if overdone.

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

pipe = ImbPipeline([("smote", SMOTE(random_state=42)), ("clf", LogisticRegression())])
scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1")  # resampling happens per-fold automatically
```

#### Undersampling

Randomly (or intelligently, e.g., via `NearMiss`, Tomek links, edited nearest neighbors) remove majority-class samples to balance the classes.

```python
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_train, y_train)
```

- **When to use:** majority class is so large that training is computationally expensive, and you can afford to discard majority examples without losing important signal (large datasets where the majority class has redundant/repetitive examples).
- **Pitfalls:** discards potentially useful data, can hurt performance if the majority class has meaningful internal sub-structure; usually combined with ensembling (e.g., `BalancedRandomForestClassifier`, `EasyEnsemble`) to avoid throwing away too much data in a single random draw.

#### Threshold Moving

Instead of changing the data, keep the default 0.5 decision threshold aside and choose a different probability cutoff for classification based on the business cost of false positives vs. false negatives, using the Precision-Recall or ROC curve to pick an operating point.

```python
from sklearn.metrics import precision_recall_curve
probs = model.predict_proba(X_val)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
# choose threshold that achieves e.g. recall >= 0.9 at best precision
best_idx = next(i for i, r in enumerate(recalls) if r < 0.9)
best_threshold = thresholds[best_idx]
```

- **When to use:** almost always worth doing for imbalanced problems — it's free (no retraining), directly optimizes for the deployment-relevant cost tradeoff, and is often more effective than resampling for a well-calibrated model.

#### Cost-Sensitive Learning

Explicitly encode the asymmetric cost of different error types into the loss function or a decision rule, e.g., a cost matrix where a false negative on fraud costs \$500 and a false positive costs \$5. The optimal decision threshold under cost-sensitive learning is:

$$\text{threshold}^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$$

- **When to use:** whenever business costs of the two error types are known and asymmetric (fraud, medical diagnosis, churn) — this reframes the entire problem around expected cost minimization rather than a generic classification metric.

**Imbalance strategy decision table:**

| Situation | Recommended approach |
|---|---|
| Moderate imbalance (e.g., 1:10), enough minority samples | Class weighting + threshold tuning |
| Severe imbalance (e.g., 1:1000), few minority samples | SMOTE/ADASYN + class weighting, careful CV |
| Very large dataset, majority class redundant | Undersampling (with ensembling) |
| Known business costs for errors | Cost-sensitive learning / threshold moving via cost matrix |
| Any imbalance | Never trust accuracy — use PR-AUC, F1, recall, or MCC |

---

### Feature Leakage

**Definition:** feature leakage (a.k.a. data leakage) occurs when information that would not be available at genuine prediction time is used during training, causing inflated validation/test performance that collapses in production.

#### Common Causes

1. **Target leakage:** a feature is a proxy for, or directly derived from, the target (e.g., using `days_hospitalized` to predict `is_severe_case` when severity determines hospitalization length — the causality is backwards).
2. **Train/test contamination:** fitting a preprocessing step (scaler, imputer, encoder, PCA, feature selector, SMOTE) on the full dataset *before* splitting into train/test, so test-set statistics leak into the training pipeline.
3. **Temporal leakage:** using future information to predict the past — e.g., using a customer's total lifetime purchases (which includes purchases *after* the prediction date) to predict churn *at* that date. Extremely common in time-series and forecasting problems.
4. **Group leakage:** rows from the same entity (patient, user, session) appear in both train and test sets, letting the model memorize entity-specific patterns instead of generalizing — must use group-aware splitting (`GroupKFold`) when multiple rows share an entity.
5. **Duplicate rows/records across the split** (including near-duplicates from data augmentation) — if the same or highly similar row lands in both train and test, performance is inflated.
6. **Preprocessing using target-derived statistics** (target/mean encoding) computed globally instead of out-of-fold, as detailed above.

#### How to Detect

- **Suspiciously high performance:** near-perfect accuracy/AUC on validation, especially early in a project, is a red flag — investigate before celebrating.
- **Feature importance audit:** if a single feature dominates importance and it's not obviously causally sound, dig into how it's computed and when it becomes available.
- **Timestamp audit:** for every feature, ask "would this value have been known at the exact moment we need to make the prediction in production?" If unsure, check the data pipeline/ETL timestamps.
- **Train/production performance gap:** a large drop in performance after deployment (vs. offline validation) is a lagging indicator that leakage existed — better to catch it before deployment via careful pipeline review.
- **Ablation:** remove a suspicious feature and see if performance drops implausibly much given the feature's plausible real-world predictive power.

#### How to Prevent

- Always split data **before** any fitting step (scalers, encoders, imputers, feature selectors, resamplers) — fit only on training folds, then transform validation/test.
- Use `sklearn.Pipeline` (or `imblearn.pipeline.Pipeline`) to bundle preprocessing + model so that `cross_val_score`/`GridSearchCV` automatically refits preprocessing per fold, structurally preventing leakage.
- For time series, use **walk-forward validation** and ensure every feature (including rolling aggregates, target encodings) is computed using only information available strictly before the prediction timestamp.
- For grouped data, use `GroupKFold` / `GroupShuffleSplit` keyed on the entity ID.
- Maintain a **feature availability timestamp** in the data dictionary/feature store — a discipline standard in production ML (feature stores like Feast enforce point-in-time correctness explicitly for this reason — highly relevant for ML Engineer interviews).

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# CORRECT: preprocessing is refit inside each CV fold automatically
pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
scores = cross_val_score(pipe, X, y, cv=5)

# WRONG (leakage): fitting scaler on all of X before CV split
# X_scaled = StandardScaler().fit_transform(X)
# scores = cross_val_score(LogisticRegression(), X_scaled, y, cv=5)
```

---

### Interview Questions — Feature Engineering

**Q1. Why can't you use label encoding for a nominal categorical feature in a linear regression model?**
Label encoding assigns arbitrary integers to categories, implying a numeric ordering and equal spacing between categories that doesn't exist for nominal data (e.g., `red=0, blue=1, green=2` implies `blue` is exactly "between" `red` and `green` and that the model can multiply the category by a single coefficient meaningfully). Linear models interpret the encoded value as having a linear effect on the target, which produces meaningless, misleading coefficients. One-hot encoding (or target encoding for high cardinality) avoids this false ordinal assumption. Tree-based models are more tolerant since they split on thresholds rather than assuming linear/ordinal relationships, though it's still not ideal.

**Q2. You're building a churn model and want to target-encode a `zip_code` feature with 30,000 unique values. Walk through how you'd do this without leakage.**
I'd never fit the target encoding on the full dataset. Instead: (1) split data into train/validation/test first; (2) within the training set, use K-fold out-of-fold encoding — for each fold, compute the smoothed target mean per zip code using only the *other* folds, and assign that value to the held-out fold's rows; (3) after computing the final encoding map from the entire training set (once, after OOF is done for model training), apply that fixed map to validation and test sets; (4) use smoothing (blend with the global mean, weighted by category frequency) so that rare zip codes with only 1–2 samples don't get an extreme, noisy encoded value; (5) provide a fallback (global mean) for unseen zip codes at inference time.

**Q3. When would you prefer frequency encoding over target encoding?**
When target encoding's leakage risk is too costly to manage carefully (e.g., a fast prototyping phase or a pipeline without robust CV infrastructure), when the *prevalence* of a category itself is informative independent of the label (e.g., popular products behave differently regardless of the outcome), or when you want a cheap, leakage-free encoding as a baseline before investing in more careful target encoding. Frequency encoding doesn't use the target at all, so it carries essentially zero leakage risk, at the cost of less predictive power.

**Q4. Explain the hashing trick and a scenario where you'd choose it over a learned vocabulary encoding.**
The hashing trick maps each category to one of a fixed number of buckets via a hash function (`bucket = hash(x) % n_buckets`), avoiding the need to build and store an explicit category→index dictionary. I'd choose it for streaming/online learning systems where new categories appear continuously and are unbounded (e.g., ad click IDs, URLs) — a fitted vocabulary would need constant updating and could run out of memory, whereas hashing is stateless and works identically at serving time regardless of vocabulary drift. The tradeoff is potential hash collisions injecting noise, mitigated by using enough buckets and/or the signed hashing variant.

**Q5. Why does PCA require standardized input, but a Random Forest generally doesn't?**
PCA finds directions of maximum variance; if features are on different scales (e.g., income in dollars vs. age in years), the higher-scale feature will dominate the variance calculation purely due to units, not because it's actually more informative — this distorts the principal components. Random Forest splits are based on ordering/thresholds within a single feature at a time; multiplying a feature by a constant doesn't change the relative order of values, so the optimal split points (and thus the tree structure) are unaffected by scale.

**Q6. What's wrong with running SMOTE on the entire dataset before doing train/test split or cross-validation?**
SMOTE generates synthetic minority samples by interpolating between real minority samples and their nearest neighbors. If applied before splitting, synthetic points in the test set can be interpolated from real points that ended up in the training set (or vice versa), meaning information about the "test" data effectively leaked into training via shared nearby neighbors — this inflates validation/test performance unrealistically. The fix is to split first, then apply SMOTE only inside the training fold (ideally via an `imblearn` Pipeline so it happens fresh inside each CV fold).

**Q7. Which metric/approach would you use for a highly imbalanced fraud detection dataset, and how would you engineer features/training to address the imbalance?**
Accuracy is useless here (predicting "not fraud" always might give >99% accuracy). I'd use PR-AUC (precision-recall AUC) or F1/F-beta (favoring recall since missing fraud is usually costlier than a false alarm) as the primary offline metric, and monitor recall at a fixed, business-acceptable false-positive rate. For training, I'd start with class weighting (`scale_pos_weight` in XGBoost) as a cheap first lever, potentially add SMOTE/ADASYN if the minority class is too sparse to learn from directly, and finally use threshold moving on the validation set's PR curve to select an operating point that matches the actual cost ratio of a missed fraud vs. a false alarm (cost-sensitive threshold = $C_{FP}/(C_{FP}+C_{FN})$).

**Q8. Explain why cyclical (sin/cos) encoding is preferred over raw integer encoding for "hour of day" in a linear model, and why you need both sine AND cosine.**
Raw integers (0–23) impose a false linear ordering where hour 23 and hour 0 appear maximally far apart (`|23-0|=23`) despite being only 1 hour apart in reality — this breaks the natural periodicity for a linear model. Projecting onto a circle via `sin(2π·hour/24)` and `cos(2π·hour/24)` preserves true cyclical adjacency (23 and 0 map to nearby points on the circle). You need both sine and cosine together because sine alone is not injective across the full period (multiple distinct hours can share the same sine value, e.g., hour 6 and hour 18 have different cosines but could share characteristics); the (sin, cos) pair uniquely and smoothly parameterizes every point on the circle.

**Q9. What's the danger of interpreting distances or cluster sizes in a t-SNE plot?**
t-SNE optimizes to preserve *local* neighborhood similarity (which points are close to which), not global distances or densities. The apparent size of a cluster, the distance between separate clusters, and even whether clusters that look far apart are "more different" than clusters that look close, are artifacts of the optimization and hyperparameters (especially perplexity) — they do not reflect real relationships in the original high-dimensional space. Treating t-SNE plot geometry as quantitatively meaningful (e.g., "cluster A is twice as far from B as from C, so A is twice as similar to C") is a common and serious misinterpretation.

**Q10. Give an example of temporal feature leakage and how you'd catch it in a code/data review.**
Example: predicting customer churn using a feature `total_purchases_lifetime` that is computed as of the *data extraction date*, not as of the *prediction date* for each customer — so it includes purchases made after the point at which we'd actually need to predict churn in production, effectively looking into the future. To catch it: audit every feature's computation window against the label's timestamp and ask "would this value have existed at prediction time in production?"; check the ETL/ feature-store logic for any `GROUP BY customer` aggregation that doesn't filter by `event_date <= prediction_date`; watch for implausibly high offline performance as a red flag warranting this audit.

**Q11. You have 500 raw features and suspect many are redundant. Describe a practical pipeline combining filter, embedded, and wrapper methods.**
Step 1 (filter, cheap): drop near-zero-variance features with `VarianceThreshold`, and drop features with correlation above ~0.95 with another feature (keep one of a pair) to cut obvious redundancy fast. Step 2 (embedded): fit a LassoCV or a gradient boosting model and rank remaining features by coefficient magnitude / permutation importance; drop the bottom tier that contributes negligible signal. Step 3 (wrapper, expensive but final polish): run RFECV with the target model type on the reduced feature set (now maybe 50–100 features, so RFE is computationally tractable) to fine-tune the final subset, using cross-validated scoring to choose the feature count that maximizes validation performance rather than training performance.

**Q12. Why might tree-based feature importance mislead you about which features are "important"?**
Impurity-based (Gini/MDI) importance is biased toward high-cardinality and continuous features simply because they offer the model more possible split points to be chosen (more opportunities to reduce impurity), not necessarily because they're more causally predictive. It's also typically computed on training data, so a feature the model overfit to can appear falsely important. Permutation importance, computed on a held-out validation set by shuffling one feature at a time and measuring the actual performance drop, is a more trustworthy, model-agnostic alternative, though it can still be inflated by correlated features (shuffling one correlated feature barely hurts performance because a correlated partner compensates).

**Q13. When would you choose UMAP over t-SNE, and when would you choose PCA over both?**
UMAP over t-SNE: when you need faster runtime on larger datasets, want embeddings that are more usable as actual downstream features (not just visualization) because UMAP tends to better preserve some global structure and supports transforming new points, or want more stable results across reruns. PCA over both: when you need a fast, linear, deterministic, and interpretable reduction (loadings map back to original features), need an exact reusable transform for new/streaming data, or are reducing dimensionality as a genuine preprocessing step before a downstream supervised model (t-SNE/UMAP are generally not appropriate as generic ML feature preprocessing — they're primarily for visualization/exploration or clustering-specific pipelines).

**Q14. What's the difference between oversampling the minority class and using `class_weight="balanced"`, and are they redundant if used together?**
`class_weight="balanced"` changes the loss function to penalize minority-class errors more, without altering the dataset — cheap and avoids introducing duplicate/synthetic data. Oversampling (SMOTE, random duplication) physically changes the training set's class distribution. They achieve a conceptually similar goal (both effectively upweight the minority class's influence on training) via different mechanisms, so combining both aggressively can *overcorrect* and hurt precision by making the model too eager to predict the minority class — in practice I'd tune one primary lever first (usually class weighting, since it's simpler and doesn't risk synthetic-sample artifacts) and only add oversampling if class weighting alone isn't sufficient, validating with a proper metric like PR-AUC on a held-out set.

**Q15. A colleague computed the mean target value per category across the WHOLE dataset (train+test combined) and used it as a feature before splitting. What's wrong, and what's the fix?**
This is textbook target leakage: the encoding for each training row implicitly incorporates information from the test set's target values (through the shared category-level statistic), and — even worse — a training row's own label directly informs its own encoded feature value if no out-of-fold scheme is used, which is close to injecting the label itself as a feature. The fix: split into train/validation/test first; compute the target encoding using only training data, ideally with K-fold out-of-fold assignment within the training set to avoid a row informing its own encoding; apply the resulting (frozen) mapping to validation/test without ever recomputing it using their targets.

**Q16. You add `price`, `price^2`, and `price * promo_flag` to a linear regression via `PolynomialFeatures`. How would you check whether you've just introduced problematic multicollinearity, and what would you do about it?**
I'd compute the Variance Inflation Factor (VIF) for each column in the new feature set (regress each feature on all the others and take $1/(1-R_j^2)$) rather than eyeballing a correlation matrix, since VIF captures multicollinearity with the *combination* of other features, not just pairwise correlation. `price` and `price^2` are almost always going to show elevated VIF because one is a monotonic function of the other over most real data ranges. If VIF is in the moderate range (5–10) I'd usually tolerate it if the terms are theoretically justified (e.g., I have a real reason to believe a quadratic price effect exists) and switch the estimator to Ridge/Elastic Net, which handles correlated predictors gracefully by shrinking and sharing weight instead of producing wild, unstable coefficients. If VIF is very high (>10) and the term isn't adding real predictive value (check via held-out score before/after dropping it), I'd just drop it rather than fight the instability.

---

## Model Evaluation

### Data Splitting Strategies

#### Train/Validation/Test Split

- **Train set:** used to fit model parameters.
- **Validation set:** used to tune hyperparameters and make model-selection decisions.
- **Test set:** touched only once, at the very end, to report an unbiased estimate of generalization performance — never used for any decision-making.

```python
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
# common ratio: 60/20/20 or 70/15/15
```

- **Pitfall:** "peeking" at the test set repeatedly during model iteration (even just checking test performance to decide between two models) turns the test set into a de facto validation set, invalidating the "unbiased" claim — this is a subtle but common form of leakage via repeated evaluation.

#### Validation Set Roles: Early Stopping vs. Hyperparameter Search vs. Test Set

A single "validation set" is often silently asked to do three *different* jobs, and conflating them is one of the most common trip-ups in an interview (and in real pipelines):

1. **Early-stopping validation set:** used purely to decide *when to stop training* an iterative model (boosting rounds, NN epochs) for a *given, already-chosen* hyperparameter configuration. It answers "has this specific model started overfitting yet?"
2. **Hyperparameter-search validation set (or CV folds):** used to *compare different configurations/model families* and pick a winner. It answers "which of these candidate setups generalizes best?"
3. **Test set:** touched exactly once, after every decision above (architecture, hyperparameters, stopping point) is already frozen, purely to *report* an unbiased number. It never answers a "which one is better" question — it only measures.

The trip-up: if you use the *same* held-out split both to decide when to stop training AND to pick the best hyperparameters, that set's score has now informed two rounds of decision-making — it is no longer an unbiased estimate of generalization error, for exactly the same reason a plain (non-nested) CV score is biased once you tune on it (see **Nested Cross-Validation** below). It is still fine to reuse it for *both* early stopping and hyperparameter selection in practice (this is extremely common and computationally cheap) — the mistake is reporting *that same score* as if it were the final, held-out test performance.

```python
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

best_score, best_params = -1.0, None
for params in candidate_param_grid:                            # hyperparameter SEARCH loop
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],                             # early-STOPPING decision uses X_val
        callbacks=[lgb.early_stopping(stopping_rounds=20)],
    )
    val_auc = model.best_score_["valid_0"]["auc"]              # hyperparameter SELECTION also uses X_val
    if val_auc > best_score:
        best_score, best_params = val_auc, params

final_model = lgb.LGBMClassifier(**best_params).fit(
    X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=20)]
)
test_auc = roc_auc_score(y_test, final_model.predict_proba(X_test)[:, 1])  # X_test touched ONCE — the real number
```

- **Why `X_val`'s score above is not a trustworthy final number:** it directly drove which `params` were chosen (`best_params`) *and* how long each candidate trained (early stopping), so it has been optimized against twice. Only `test_auc`, computed once at the very end on data no decision ever looked at, can be quoted as "expected production performance."
- **Minimum viable setup when you need both early stopping and hyperparameter search done properly:** at least three splits/roles — train (fit weights), validation (drive stopping and/or hyperparameter choice, possibly via CV), and test (report once) — or replace the middle step with **nested cross-validation** if you also want a robust, low-variance *estimate* of generalization error rather than a single validation-set number.

#### K-Fold Cross-Validation

Split the training data into $k$ equally sized folds; for each of $k$ iterations, train on $k-1$ folds and validate on the remaining fold; average the $k$ validation scores.

```python
from sklearn.model_selection import KFold, cross_val_score
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="roc_auc")
print(scores.mean(), scores.std())
```

- **Why:** gives a more robust, lower-variance estimate of generalization performance than a single train/val split, and uses all data for both training and validation across folds (efficient use of limited data).
- **Choosing $k$:** typical values are 5 or 10. Larger $k$ → less bias (each fold's training set is closer in size to the full dataset) but higher variance and more compute; $k=n$ is leave-one-out (see below).

#### Stratified K-Fold

Same as K-Fold, but ensures each fold preserves the overall class distribution — critical for imbalanced classification, since plain random folds could produce a fold with zero minority-class examples.

```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1")
```

- **When to use:** virtually always for classification, especially with any class imbalance; not applicable directly to regression (though binning the continuous target and stratifying on bins is a common workaround).

#### Time-Series-Aware Splitting (Walk-Forward / Expanding Window)

Standard K-Fold shuffles data randomly, which for time series causes future data to leak into training folds that predict past validation folds — invalid for temporal data. Instead, use **walk-forward validation**: train on data up to time $t$, validate on data from $t$ to $t+\Delta$, then expand (or roll) the window forward.

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)   # each split's train set only contains data BEFORE its validation set
for train_idx, val_idx in tscv.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    # train is always strictly earlier in time than val
```

- **Expanding window:** training set grows each fold (always starts from time 0). **Rolling/sliding window:** training set is a fixed-size window that slides forward — useful when older data becomes less relevant (concept drift) or for computational efficiency.
- **Pitfall:** using ordinary shuffled K-Fold or `train_test_split(shuffle=True)` on time series data is one of the most common leakage bugs in practice — it lets the model "see the future" during training and produces wildly over-optimistic validation metrics that fail in live deployment.

#### Leave-One-Out Cross-Validation (LOOCV)

Special case of K-Fold with $k = n$: each fold validates on exactly one sample, training on all others.

```python
from sklearn.model_selection import LeaveOneOut, cross_val_score
loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)
```

- **When to use:** very small datasets where every data point matters and you want the least-biased estimate of performance (training set size is $n-1$, nearly the full dataset every time).
- **Pitfalls:** extremely expensive ($n$ model fits) — infeasible for large datasets or slow-to-train models; the $n$ validation scores are highly correlated (each fold differs by only one sample), so the variance of the overall estimate can actually be *higher* than a well-chosen K-fold in some regimes, despite low bias — a genuinely subtle, popular interview point ("LOOCV has low bias but can have high variance for the estimate of test error").

**Splitting strategy decision table:**

| Data characteristic | Recommended split strategy |
|---|---|
| i.i.d. tabular data, ample size | K-Fold (k=5 or 10) |
| i.i.d. classification, imbalanced | Stratified K-Fold |
| Time series / sequential data | TimeSeriesSplit / walk-forward |
| Multiple rows per entity (patients, users) | GroupKFold |
| Very small dataset | Leave-one-out or repeated K-Fold |
| Need one final unbiased performance number | Untouched held-out test set (used once) |

---

### Classification Metrics

#### Confusion Matrix

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

#### Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

- **When appropriate:** balanced classes, and all error types are roughly equally costly.
- **Pitfall:** misleading under class imbalance — a model predicting the majority class always can achieve very high accuracy while being useless (the "accuracy paradox").

#### Precision, Recall, F1, F-beta

$$\text{Precision} = \frac{TP}{TP+FP}, \qquad \text{Recall (Sensitivity)} = \frac{TP}{TP+FN}$$

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}, \qquad F_\beta = (1+\beta^2)\cdot\frac{\text{Precision}\cdot\text{Recall}}{\beta^2\cdot\text{Precision} + \text{Recall}}$$

- **Precision** answers "of everything I flagged positive, how much was actually positive?" — matters when false positives are costly (e.g., spam filter blocking legitimate email).
- **Recall** answers "of everything actually positive, how much did I catch?" — matters when false negatives are costly (e.g., missing a cancer diagnosis, missing fraud).
- **F-beta** generalizes F1 by weighting recall $\beta$ times as important as precision; $\beta > 1$ favors recall (e.g., $F_2$ for fraud/disease screening), $\beta < 1$ favors precision (e.g., $F_{0.5}$ for a recommendation system where irrelevant suggestions annoy users).

```python
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, classification_report
print(classification_report(y_true, y_pred))
f2 = fbeta_score(y_true, y_pred, beta=2)
```

#### ROC Curve and AUC

The **ROC curve** plots True Positive Rate ($TPR = \text{Recall}$) against False Positive Rate ($FPR = FP/(FP+TN)$) across all classification thresholds. **AUC-ROC** is the area under this curve, interpretable as the probability that a randomly chosen positive example is ranked (scored) higher than a randomly chosen negative example.

```python
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
auc = roc_auc_score(y_true, y_scores)
```

- **Pitfall under imbalance:** ROC-AUC can look deceptively good on highly imbalanced data because FPR's denominator ($FP+TN$) is dominated by the huge number of true negatives, so even a large *absolute* number of false positives barely moves FPR. A model can have a high ROC-AUC while producing many false positives relative to the (small) number of true positives — this is precisely why **PR-AUC is preferred for imbalanced problems**.

#### Precision-Recall Curve and AUC

Plots Precision against Recall across thresholds; **PR-AUC** (a.k.a. Average Precision) summarizes this curve.

```python
from sklearn.metrics import precision_recall_curve, average_precision_score
precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
pr_auc = average_precision_score(y_true, y_scores)
```

- **Why preferred for imbalance:** precision's denominator ($TP+FP$) is NOT inflated by the large number of true negatives, so PR-AUC directly reflects how well the model manages the tradeoff on the (rare) positive class, making it far more sensitive to a degrading minority-class model than ROC-AUC.
- The baseline (random classifier) PR-AUC equals the positive class prevalence, unlike ROC-AUC's constant 0.5 baseline — always compare your PR-AUC against this baseline, not against 0.5.

#### Log Loss (Binary Cross-Entropy)

$$\text{LogLoss} = -\frac{1}{n}\sum_{i=1}^n \left[y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\right]$$

```python
from sklearn.metrics import log_loss
loss = log_loss(y_true, y_pred_proba)
```

- **When to use:** you care about the *quality of predicted probabilities*, not just the final class decision (e.g., risk scoring, ranking by probability, any downstream system consuming the probability itself) — it heavily penalizes confident-and-wrong predictions.
- **Pitfall:** a single very confident wrong prediction ($\hat{p}\to 0$ or $1$ on the wrong side) can dominate/blow up the average loss; requires clipping probabilities away from exactly 0/1 in implementation (`sklearn` does this internally).

#### Matthews Correlation Coefficient (MCC)

$$MCC = \frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

Ranges from $-1$ (total disagreement) to $+1$ (perfect prediction), with $0$ being no better than random. Uses all four confusion matrix cells symmetrically, making it robust to class imbalance.

```python
from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef(y_true, y_pred)
```

- **When to use:** widely regarded (backed by empirical studies, e.g., in bioinformatics/genomics benchmarking) as one of the most reliable single-number summaries for imbalanced binary classification, since — unlike F1 — it accounts for true negatives too, and is symmetric with respect to swapping the positive/negative label definition.

**Classification metric selection guide:**

| Scenario | Recommended primary metric(s) |
|---|---|
| Balanced classes, equal error cost | Accuracy |
| Imbalanced classes, general purpose | PR-AUC, F1, MCC |
| False negatives much costlier (fraud, disease) | Recall, $F_2$, PR-AUC (recall-weighted) |
| False positives much costlier (spam, content moderation) | Precision, $F_{0.5}$ |
| Need probability quality (risk scoring) | Log loss, calibration curve |
| Ranking-quality comparison independent of threshold | ROC-AUC (balanced) or PR-AUC (imbalanced) |
| Want single robust summary immune to imbalance | Matthews Correlation Coefficient |

---

### Regression Metrics

#### MSE and RMSE

$$MSE = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2, \qquad RMSE = \sqrt{MSE}$$

```python
import numpy as np
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)   # scikit-learn >= 1.4 removed the `squared=False` kwarg; use root_mean_squared_error(y_true, y_pred) if available, or np.sqrt(mse) always works
```

- RMSE is in the same units as the target, making it more interpretable than MSE.
- **Pitfall:** squaring the error heavily penalizes large errors/outliers — a single bad prediction can dominate the metric, which is sometimes desirable (you really care about big misses) and sometimes not (a few noisy labels shouldn't dominate your evaluation).

#### MAE (Mean Absolute Error)

$$MAE = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$

- More robust to outliers than RMSE (linear penalty instead of quadratic); directly interpretable as "average absolute miss."
- **RMSE vs MAE:** RMSE $\geq$ MAE always; the gap between them signals the variance of error magnitudes — a large gap means a few large errors are present (heavy-tailed error distribution).

#### MAPE (Mean Absolute Percentage Error)

$$MAPE = \frac{100\%}{n}\sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

- **When to use:** stakeholders want a scale-free, percentage-based interpretation (e.g., "our forecast is off by 8% on average") — common in demand forecasting/business reporting.
- **Pitfalls:** undefined/explodes when $y_i \approx 0$; asymmetric — penalizes over-predictions and under-predictions differently (an under-prediction is capped at 100% error while an over-prediction is unbounded), which can bias model selection toward under-forecasting. **sMAPE** (symmetric MAPE) or **WAPE** (weighted APE) are common fixes.

#### R-squared and Adjusted R-squared

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2} = 1 - \frac{SS_{res}}{SS_{tot}}$$

Interpreted as the proportion of variance in the target explained by the model, relative to a naive baseline that always predicts the mean.

$$R^2_{adj} = 1 - (1-R^2)\cdot\frac{n-1}{n-p-1}$$

where $p$ is the number of predictors. Adjusted $R^2$ penalizes adding predictors that don't meaningfully improve fit, unlike plain $R^2$ which never decreases as you add more features (even useless ones).

```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
n, p = len(y_true), X.shape[1]
r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
```

- **Pitfalls:** $R^2$ can be negative on a test/validation set if the model is worse than predicting the mean (a very common source of confusion — many think $R^2 \in [0,1]$ always, but that's only guaranteed on the training set of the exact model that minimized $SS_{res}$); high $R^2$ doesn't imply the model is *causally* correct or free of bias — it only measures explained variance, and can be inflated by overfitting on the training set (hence adjusted $R^2$ or, better, computing $R^2$ on held-out data).

**Regression metric selection guide:**

| Scenario | Recommended metric |
|---|---|
| Outliers matter a lot, want to penalize big misses | RMSE / MSE |
| Outliers are noise, want robustness | MAE (or Huber loss for training) |
| Need scale-free % interpretation for stakeholders | MAPE / sMAPE / WAPE |
| Comparing model fit vs. baseline, communicating "variance explained" | R² |
| Comparing models with different numbers of features | Adjusted R² |

---

### Ranking / Recommendation Metrics

For systems that output a ranked list (search, recommendations) rather than a single label, metrics must account for *position* in the list.

#### Precision@k and Recall@k

$$\text{Precision@k} = \frac{\text{\# relevant items in top-}k}{k}, \qquad \text{Recall@k} = \frac{\text{\# relevant items in top-}k}{\text{total \# relevant items}}$$

```python
def precision_at_k(recommended, relevant, k):
    top_k = recommended[:k]
    return len(set(top_k) & set(relevant)) / k
```

- **When to use:** you care about the quality of a fixed-size result page (e.g., top-10 search results, top-5 product recommendations shown on a homepage carousel).

#### Mean Average Precision (MAP)

Average Precision (AP) for one query/user is the average of precision values computed at each rank where a relevant item is found; MAP averages AP across all queries/users.

$$AP = \frac{1}{|\text{relevant}|}\sum_{k: \text{item}_k \text{ relevant}} \text{Precision@}k$$

- **When to use:** binary relevance ranking tasks (relevant/not relevant) where you want a single number that rewards putting relevant items earlier, averaged over many queries — classic for search engine and document retrieval evaluation.

#### NDCG (Normalized Discounted Cumulative Gain)

Handles **graded relevance** (not just binary) and discounts the value of relevant items appearing later in the ranking.

$$DCG@k = \sum_{i=1}^k \frac{2^{rel_i}-1}{\log_2(i+1)}, \qquad NDCG@k = \frac{DCG@k}{IDCG@k}$$

where $IDCG@k$ is the DCG of the ideal (perfectly sorted) ranking, used to normalize the score into $[0,1]$.

```python
import numpy as np

def dcg_at_k(relevances, k):
    relevances = np.asarray(relevances)[:k]
    gains = (2 ** relevances - 1) / np.log2(np.arange(2, len(relevances) + 2))
    return gains.sum()

def ndcg_at_k(relevances, k):
    ideal = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg_at_k(relevances, k) / idcg if idcg > 0 else 0.0
```

- **When to use:** whenever relevance is graded rather than binary (e.g., a 0–5 relevance score, or click/add-to-cart/purchase as increasing relevance tiers) — this is the standard metric for search ranking and recommender systems in industry (e.g., learning-to-rank models like LightGBM's LambdaMART are directly optimized against NDCG).

#### Mean Reciprocal Rank (MRR)

$$MRR = \frac{1}{|Q|}\sum_{q=1}^{|Q|} \frac{1}{\text{rank}_q}$$

where $\text{rank}_q$ is the position of the *first* relevant item for query $q$.

- **When to use:** tasks where only the position of the *first* correct/relevant answer matters (e.g., "did the right answer to a factual question appear as the #1 or #2 result," chatbot/QA retrieval, autocomplete suggestion ranking) rather than the overall quality of the full ranked list.

**Ranking metric comparison:**

| Metric | Relevance type | Sensitive to position? | Typical use case |
|---|---|---|---|
| Precision@k / Recall@k | Binary | Only within top-k, not to exact position | Fixed-size result pages |
| MAP | Binary | Yes | Document/search retrieval |
| NDCG | Graded | Yes (log discount) | Search ranking, recommenders |
| MRR | Binary | Only first hit | QA/first-correct-answer tasks |

---

### Calibration of Probabilistic Predictions

A model is **well-calibrated** if, among all instances where it predicts probability $p$, approximately $p$ fraction are actually positive. Calibration is distinct from discrimination (AUC) — a model can rank examples perfectly (high AUC) while its probability *values* are systematically over- or under-confident.

#### Reliability Diagrams

Bin predictions by predicted probability (e.g., 10 bins: [0,0.1), [0.1,0.2), ...), and plot the mean predicted probability in each bin against the observed fraction of positives in that bin. A perfectly calibrated model lies on the diagonal $y=x$.

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

frac_pos, mean_pred = calibration_curve(y_true, y_pred_proba, n_bins=10, strategy="quantile")
plt.plot(mean_pred, frac_pos, marker="o", label="Model")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
```

#### Expected Calibration Error (ECE)

$$ECE = \sum_{m=1}^M \frac{n_m}{n} \left| \text{acc}(m) - \text{conf}(m) \right|$$

summed over $M$ bins, where $\text{acc}(m)$ is observed positive fraction in bin $m$ and $\text{conf}(m)$ is the average predicted probability in bin $m$ — a single scalar summary of miscalibration.

#### Brier Score

$$\text{Brier} = \frac{1}{n}\sum_{i=1}^n (\hat{p}_i - y_i)^2$$

The mean squared error between predicted probability and the actual binary outcome — literally "MSE for probabilities." Ranges from 0 (perfect) to 1 (worst possible).

```python
from sklearn.metrics import brier_score_loss
brier = brier_score_loss(y_true, y_pred_proba)
```

- **Brier score vs. log loss:** both jointly reward good calibration *and* good discrimination (unlike ECE, which measures calibration in isolation), but Brier's squared penalty is bounded and gentler on very confident wrong predictions than log loss's logarithmic penalty, which blows up toward infinity as $\hat p \to 0$ or $1$ on the wrong side. Brier is often preferred when a few extreme predictions shouldn't be allowed to dominate the metric; log loss is preferred when you specifically want to punish overconfidence hard.
- **Brier score decomposition:** it can be decomposed into calibration + refinement (resolution) terms, which is why it's a popular single number in forecasting/meteorology (its original home field) and in Kaggle-style probabilistic competitions — a model can improve its Brier score either by getting better calibrated *or* by getting better at discriminating, and the decomposition tells you which lever to pull.
- **Pitfall:** like accuracy, a low (good) Brier score on a heavily imbalanced dataset can be achieved by a lazy model that just predicts the base rate for everyone — always sanity-check against the Brier score of the trivial "always predict the base rate" baseline before celebrating.

#### Fixing Miscalibration

- **Platt Scaling:** fit a logistic regression on top of the model's raw scores (good for SVMs and models producing scores that are monotonically related to true probability but not well-scaled).
- **Isotonic Regression:** fit a non-parametric, monotonic step function mapping raw scores to calibrated probabilities — more flexible than Platt scaling, needs more data to avoid overfitting.

```python
from sklearn.calibration import CalibratedClassifierCV
calibrated_model = CalibratedClassifierCV(estimator=uncalibrated_model, method="isotonic", cv=5)   # `base_estimator` was renamed to `estimator` in sklearn 1.2 and removed in 1.4
calibrated_model.fit(X_train, y_train)
```

- **When calibration matters:** any downstream decision that uses the probability *value* itself, not just a threshold-based label — e.g., expected-value calculations (bidding systems, risk-based pricing), combining multiple models' probabilities, medical risk scores communicated to clinicians, or anywhere probabilities feed into a cost-benefit calculation.
- **Common causes of poor calibration:** tree ensembles (Random Forest, GBM) tend to produce probabilities pushed toward 0/1 or clustered (overconfident) due to how leaf averaging/boosting works; class-imbalance corrections (oversampling, class weighting) systematically distort the predicted probabilities away from true class frequencies; models trained with a non-probabilistic loss (e.g., raw SVM margin) don't naturally output calibrated probabilities at all.
- **Pitfall:** always calibrate using a **held-out** set (not the training set the base model was fit on) to avoid overfitting the calibration mapping itself; a model can have excellent AUC and terrible calibration simultaneously — always check both, they answer different questions ("can it rank well?" vs. "can I trust the actual number?").

---

### Bias-Variance Tradeoff

#### Formal Decomposition

For a regression model with squared-error loss, the expected test error at a point $x$ decomposes as:

$$\mathbb{E}\left[(y - \hat{f}(x))^2\right] = \underbrace{\left(\mathbb{E}[\hat{f}(x)] - f(x)\right)^2}_{\text{Bias}^2} + \underbrace{\text{Var}[\hat{f}(x)]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible noise}}$$

- **Bias:** error from the model's simplifying assumptions being wrong — how far the *average* prediction (over many training sets) is from the true function. High bias → underfitting.
- **Variance:** how much predictions fluctuate across different training sets — sensitivity to the specific training sample. High variance → overfitting.
- **Irreducible error ($\sigma^2$):** noise inherent in the data-generating process that no model can eliminate (sets a floor on achievable error).

| Model complexity | Bias | Variance | Typical symptom |
|---|---|---|---|
| Too simple (e.g., linear model on non-linear data) | High | Low | Underfitting — high train AND test error |
| Too complex (e.g., deep unpruned tree, huge NN, no regularization) | Low | High | Overfitting — low train error, high test error |
| Well-tuned | Balanced | Balanced | Train and validation error both low and close together |

#### Practical Diagnosis via Learning Curves

Plot training and validation error (or score) as a function of training set size.

```python
from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring="neg_mean_squared_error"
)
train_mean, val_mean = -train_scores.mean(axis=1), -val_scores.mean(axis=1)
```

**Reading learning curves:**

| Pattern | Diagnosis | Fix |
|---|---|---|
| Both train and validation error high, and close together, plateauing early | High bias (underfitting) | Increase model complexity, add features, reduce regularization |
| Train error low, validation error much higher, large persistent gap | High variance (overfitting) | Add regularization, get more data, reduce complexity, add dropout/early stopping |
| Train and validation error both decreasing and gap narrowing as data grows, not yet plateaued | Needs more data / not yet converged | Collect more training data if feasible |
| Both errors low and close | Good bias-variance balance | Model is well-tuned for this data |

- **Practical rule of thumb:** if adding more training data doesn't close the train/validation gap (pure variance problem with a wide, non-narrowing gap), regularization or architecture changes are needed instead of just "get more data."

---

### Overfitting, Underfitting, and Regularization

#### Detection

- **Overfitting signs:** large gap between training and validation metrics; validation performance degrades after continuing to train (e.g., validation loss increases while training loss keeps decreasing — visible in loss curves); model performs implausibly well on training data (near-zero training error).
- **Underfitting signs:** both training and validation error are high and similar; model fails to capture even obvious patterns visible via exploratory analysis.

#### Remedies for Overfitting

- Simplify the model (fewer features/parameters, shallower trees, smaller network).
- Add more training data (directly reduces variance).
- Regularization (L1/L2/Elastic Net, dropout, early stopping — see below).
- Ensembling (bagging reduces variance, e.g., Random Forest).
- Cross-validation-driven hyperparameter selection instead of eyeballing training performance.
- Data augmentation (especially in vision/audio/text).

#### Remedies for Underfitting

- Increase model complexity (more features, deeper trees, more layers/units, less regularization).
- Feature engineering — add interaction terms, polynomial features, better-engineered signals.
- Train longer / check optimization hasn't stalled (e.g., learning rate too low, hasn't converged).
- Reduce regularization strength if it's currently too aggressive.

#### L1, L2, and Elastic Net Regularization (math and intuition)

For linear/logistic regression, add a penalty term to the loss $L(\beta)$:

$$L_{\text{Ridge}}(\beta) = L(\beta) + \lambda \sum_j \beta_j^2 \qquad \text{(L2)}$$
$$L_{\text{Lasso}}(\beta) = L(\beta) + \lambda \sum_j |\beta_j| \qquad \text{(L1)}$$
$$L_{\text{ElasticNet}}(\beta) = L(\beta) + \lambda \left(\alpha \sum_j |\beta_j| + (1-\alpha)\sum_j \beta_j^2\right)$$

- **L2 (Ridge):** shrinks all coefficients smoothly toward zero but rarely to *exactly* zero (the squared penalty's gradient vanishes as $\beta_j \to 0$, so it never quite reaches zero) — good when many features are mildly useful and you want to avoid discarding any, and handles multicollinearity well (spreads weight across correlated features).
- **L1 (Lasso):** the penalty's constant-magnitude gradient means it can drive coefficients exactly to zero, performing automatic feature selection — good when you believe only a subset of features truly matters (sparse solutions), but arbitrarily picks one among a group of highly correlated features (unstable selection under multicollinearity).
- **Elastic Net:** blends L1 and L2 to get sparse solutions like Lasso while handling correlated features more gracefully like Ridge — controlled by mixing parameter $\alpha$.

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
ridge = Ridge(alpha=1.0).fit(X_train, y_train)
lasso = Lasso(alpha=0.1).fit(X_train, y_train)
enet = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_train, y_train)   # l1_ratio: alpha in the formula above
```

**Geometric intuition:** the L1 penalty's constraint region is a diamond (has corners on the axes), and the loss function's contours are more likely to first touch the constraint region exactly at a corner (where one or more coefficients are exactly zero); the L2 penalty's constraint region is a smooth circle/sphere with no corners, so the optimum rarely lands exactly on an axis.

#### Dropout

In neural networks, randomly zero out a fraction $p$ of neuron activations during each training forward pass (typically $p=0.2$–$0.5$), forcing the network to not rely on any single neuron/feature combination — a form of implicit ensembling/regularization. At inference time, all neurons are active, typically with activations scaled (or, in "inverted dropout," scaled during training) to keep expected magnitude consistent.

```python
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(64, 1)
)
```

- **Intuition:** prevents co-adaptation of neurons (where units become overly reliant on specific other units' outputs), pushing the network toward more redundant, robust representations.

#### Early Stopping

Monitor validation loss during training and stop (or save the best checkpoint) once validation loss stops improving for a patience window, even though training loss might keep decreasing — directly targets the overfitting point on the learning curve.

```python
best_val_loss, patience, patience_counter = float("inf"), 5, 0
for epoch in range(max_epochs):
    train_one_epoch(model, train_loader)
    val_loss = evaluate(model, val_loader)
    if val_loss < best_val_loss:
        best_val_loss, patience_counter = val_loss, 0
        save_checkpoint(model)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break   # stop training, restore best checkpoint
```

- **Why it's a form of regularization:** it implicitly limits the effective model capacity/complexity reached during optimization, similar in spirit to constraining the parameter search space, without changing the loss function itself.

**Regularization technique comparison:**

| Technique | Mechanism | Best for | Produces sparsity? |
|---|---|---|---|
| L2 / Ridge | Shrinks weights smoothly | Multicollinearity, many mildly-useful features | No |
| L1 / Lasso | Shrinks weights, can zero them out | Feature selection, sparse true signal | Yes |
| Elastic Net | Blend of L1+L2 | Correlated features + desire for sparsity | Partial |
| Dropout | Randomly disables neurons during training | Deep neural networks | N/A |
| Early stopping | Halts training at optimal validation point | Any iterative training (NN, boosting) | N/A |

---

### Hyperparameter Tuning

#### Grid Search

Exhaustively evaluate every combination of hyperparameter values from a predefined grid.

```python
from sklearn.model_selection import GridSearchCV
param_grid = {"n_estimators": [100, 200, 500], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1]}
grid = GridSearchCV(estimator, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_params_, grid.best_score_)
```

- **When to use:** small, low-dimensional hyperparameter spaces where exhaustiveness is affordable and you want a guaranteed, reproducible best-in-grid answer.
- **Pitfall:** cost grows exponentially with the number of hyperparameters (combinatorial explosion) — infeasible beyond a handful of parameters with a handful of values each.

#### Random Search

Sample hyperparameter combinations randomly from specified distributions for a fixed budget of iterations.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_dist = {"n_estimators": randint(50, 1000), "max_depth": randint(2, 12), "learning_rate": uniform(0.005, 0.3)}
rand_search = RandomizedSearchCV(estimator, param_dist, n_iter=50, cv=5, scoring="roc_auc", random_state=42, n_jobs=-1)
rand_search.fit(X_train, y_train)
```

- **Why it often beats grid search in practice:** for a fixed compute budget, random search explores more distinct values along each individual hyperparameter dimension than grid search does (a key finding from Bergstra & Bengio, 2012) — especially valuable when only a few hyperparameters actually matter much (common in practice), since grid search wastes evaluations on redundant combinations of the unimportant ones.

#### Bayesian Optimization (e.g., Gaussian Processes)

Model the objective function (validation score as a function of hyperparameters) as a surrogate probabilistic model — commonly a **Gaussian Process (GP)** — and use it to intelligently choose the next hyperparameter combination to try, balancing **exploration** (trying uncertain regions) and **exploitation** (trying regions predicted to be good), typically via an acquisition function like Expected Improvement (EI) or Upper Confidence Bound (UCB).

```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer

search_space = {"n_estimators": Integer(50, 1000), "max_depth": Integer(2, 12), "learning_rate": Real(0.005, 0.3, prior="log-uniform")}
bayes_search = BayesSearchCV(estimator, search_space, n_iter=30, cv=5, scoring="roc_auc", random_state=42)
bayes_search.fit(X_train, y_train)
```

- **Why it's more sample-efficient:** unlike grid/random search which choose points independently of past results, Bayesian optimization uses every prior evaluation to update its belief about the objective's shape, focusing subsequent trials in promising regions — this matters a lot when each model fit is expensive (large models, big data, long training times).
- **Pitfalls:** the surrogate model (GP) itself has hyperparameters and scales poorly with the number of tuned hyperparameters (curse of dimensionality) and with the number of observations ($O(n^3)$ for exact GP inference); sequential by nature, making it harder to parallelize than grid/random search (though batch/parallel variants exist); overkill for cheap-to-train models where you could just run thousands of random search trials instead.

#### Successive Halving / Hyperband

Allocate a small budget (e.g., few training epochs, small data subset) to many candidate configurations, evaluate, discard the worst-performing fraction, and reallocate a larger budget to the survivors — repeating until one (or a few) configurations remain with the full budget.

```python
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV

halving = HalvingRandomSearchCV(estimator, param_dist, factor=3, resource="n_estimators", max_resources=1000, cv=5, random_state=42)
halving.fit(X_train, y_train)
```

- **Hyperband** extends successive halving by running multiple "brackets" with different tradeoffs between the number of initial configurations and the aggressiveness of elimination, hedging against the risk of being too aggressive (eliminating a config that would have been good with more budget) or too conservative (wasting budget on many configs for too long).
- **When to use:** deep learning / iterative models where "budget" naturally maps to training epochs or data fraction, and where a poorly performing configuration reveals itself early (e.g., a bad learning rate diverges quickly) — dramatically more compute-efficient than grid/random search for these cases, because bad configurations are killed early rather than run to completion.
- **Pitfall:** assumes early performance is a reasonably good proxy for final performance — can incorrectly eliminate configurations that start slow but would have overtaken others with more budget (e.g., some learning rate schedules or regularization strengths that need longer to show benefit).

**Hyperparameter tuning method comparison:**

| Method | Sample efficiency | Parallelizable | Best for |
|---|---|---|---|
| Grid Search | Low | Yes (fully) | Few hyperparameters, exhaustiveness desired |
| Random Search | Medium | Yes (fully) | Moderate dimensions, good default choice |
| Bayesian Optimization (GP) | High | Limited (sequential) | Expensive model fits, few hyperparameters |
| Successive Halving / Hyperband | High (via early stopping of bad configs) | Yes (within a rung) | Iterative models (NNs, boosting) with many candidates |

---

### Nested Cross-Validation

#### Why Plain CV Leaks When You Also Tune Hyperparameters On It

A very common mistake: run `GridSearchCV` with 5-fold CV, then report `grid.best_score_` as "the model's expected performance." This number is **optimistically biased**. The 5 folds were used to *select* the hyperparameters that scored best on exactly those folds — so those folds have been used for two jobs (tuning AND "evaluation") instead of one, the same underlying problem that motivates keeping a test set separate from a validation set at all. The more hyperparameter combinations you try, the more chances one of them has to score well on validation-fold noise rather than true signal (a form of multiple-comparisons / winner's-curse bias), so the bias grows with search-space size — a wide random/Bayesian search over many configurations leaks more than a 3-value grid.

**Nested cross-validation** fixes this by using two loops with different jobs:

- **Inner loop:** for each outer-training fold, run a full CV-based hyperparameter search (e.g., `GridSearchCV`) *using only that fold's data* to pick the best configuration.
- **Outer loop:** evaluate the winning configuration (refit on the entire outer-training fold) on the outer-test fold, which the inner loop never touched in any way. Average the outer-fold scores for an unbiased estimate of generalization performance.

```python
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

param_grid = {"max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1]}
inner_cv = KFold(n_splits=3, shuffle=True, random_state=1)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# The GridSearchCV object IS the "estimator" passed to the outer cross_val_score —
# it gets refit (inner-tuned) from scratch inside every single outer training fold.
inner_search = GridSearchCV(
    GradientBoostingClassifier(random_state=0), param_grid, cv=inner_cv, scoring="roc_auc"
)
outer_scores = cross_val_score(inner_search, X, y, cv=outer_cv, scoring="roc_auc")
print(outer_scores.mean(), outer_scores.std())   # unbiased estimate of generalization performance
```

The same logic spelled out as explicit loops, so the "inner vs. outer" separation is unambiguous:

```python
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV

outer_scores = []
for train_idx, test_idx in outer_cv.split(X, y):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    # INNER LOOP: hyperparameter search using ONLY the outer-training fold (X_tr, y_tr)
    grid = GridSearchCV(GradientBoostingClassifier(random_state=0), param_grid, cv=inner_cv, scoring="roc_auc")
    grid.fit(X_tr, y_tr)

    # OUTER LOOP: score the tuned model on the outer-test fold it never influenced in any way
    outer_scores.append(grid.score(X_te, y_te))

print(np.mean(outer_scores), np.std(outer_scores))
```

- **What nested CV is (and isn't) for:** it produces an honest *estimate* of how well "this modeling *process*" (tuning + fitting on data of this size) generalizes — it does **not** hand you a single final model to deploy, and the "best hyperparameters" can differ across outer folds. For the actual production model, you still run one (non-nested) hyperparameter search over the *entire* training set after nested CV has told you the process is trustworthy and given you an unbiased performance number to report.
- **Cost:** $k_{\text{outer}} \times k_{\text{inner}} \times |\text{grid}|$ model fits — expensive. For expensive models, prefer a smaller inner grid, `RandomizedSearchCV`/Bayesian optimization as the inner search, or fewer outer folds (e.g., 5x3 instead of 10x10).
- **Time series:** the same nesting principle applies with `TimeSeriesSplit` in both loops instead of `KFold` — the inner search must still only see data strictly before its outer-training cutoff, and the outer-test fold must still be strictly after its outer-training fold, or the "unbiased estimate" nested CV promises is invalidated by the same future-leakage problem plain shuffled CV has.
- **When you can skip it:** if you have a large enough held-out test set that a single train/val-tune/test split already gives a low-variance enough estimate, nested CV's extra compute may not be worth it — it matters most when data is limited and every fold's score estimate is otherwise noisy.

---

### Statistical Significance of Model Comparisons

Comparing two models by a single validation score (e.g., "Model A got 0.85 AUC, Model B got 0.86 AUC — Model B wins") ignores estimation noise. Statistical tests quantify whether an observed difference is likely real or within the noise of the evaluation procedure itself.

#### Paired t-test on Cross-Validation Fold Scores

Run both models on the *same* K folds, obtaining $k$ paired score differences $d_i = \text{score}_A^{(i)} - \text{score}_B^{(i)}$. Test whether the mean difference is significantly different from zero:

$$t = \frac{\bar{d}}{s_d / \sqrt{k}}$$

where $\bar{d}$ is the mean paired difference and $s_d$ is the sample standard deviation of the differences, compared against a t-distribution with $k-1$ degrees of freedom.

```python
from scipy import stats
import numpy as np

scores_a = np.array([0.81, 0.83, 0.80, 0.84, 0.82])  # per-fold CV scores, model A
scores_b = np.array([0.79, 0.80, 0.78, 0.81, 0.79])  # per-fold CV scores, model B (same folds!)
t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
print(t_stat, p_value)   # p < 0.05 -> reject H0, difference likely real
```

- **Pitfall — the "corrected" paired t-test:** standard CV fold scores are not independent (folds overlap in training data since each fold's training set shares $k-2$ folds with every other fold's training set), which violates the independence assumption of a naive t-test and inflates false-positive rates (Type I error). The **Nadeau & Bengio corrected paired t-test** adjusts the variance estimate to account for this correlation and should be preferred over the naive version in rigorous settings; alternatively, use repeated K-fold with distinct random splits or a dedicated held-out test set for the final significance claim.
- **When to use:** comparing two models' cross-validated performance on the same dataset, when you have per-fold scores and the metric is continuous (accuracy, AUC, RMSE, etc.).

#### McNemar's Test

Compares two classifiers' predictions on the *same test set* by examining only the samples where they disagree, using a 2x2 contingency table:

| | Model B correct | Model B incorrect |
|---|---|---|
| **Model A correct** | $n_{00}$ | $n_{01}$ |
| **Model A incorrect** | $n_{10}$ | $n_{11}$ |

Test statistic (with continuity correction):

$$\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}}$$

compared against a chi-square distribution with 1 degree of freedom. Only the discordant cells ($n_{01}, n_{10}$ — where exactly one model got it right) matter.

```python
from statsmodels.stats.contingency_tables import mcnemar
table = [[n00, n01], [n10, n11]]
result = mcnemar(table, exact=False, correction=True)
print(result.statistic, result.pvalue)
```

- **When to use:** comparing two classifiers evaluated once on the same fixed test set (not cross-validated fold scores) — very common for comparing a new model against a production baseline on a shared holdout set.
- **Pitfall:** requires a reasonably large number of discordant predictions ($n_{01}+n_{10}$) for the chi-square approximation to be valid; use the exact binomial version (`exact=True`) for small discordant counts.

#### Practical Guidance

- A p-value below 0.05 suggests the difference is unlikely due to chance, but always pair statistical significance with **practical/business significance** — a statistically significant 0.1% AUC improvement may not be worth the added model complexity or latency cost.
- Report confidence intervals on metrics (e.g., via bootstrap resampling of the test set) in addition to, or instead of, a single point estimate, to communicate uncertainty honestly to stakeholders.

```python
import numpy as np

def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=1000, alpha=0.05):
    scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    lower, upper = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return np.mean(scores), lower, upper
```

---

### Interview Questions — Model Evaluation

**Q1. Why is accuracy a poor metric for a dataset with 99% negative and 1% positive class, and what would you use instead?**
A trivial classifier that always predicts "negative" achieves 99% accuracy while catching zero positives, so accuracy tells us nothing about whether the model is actually useful. I'd use PR-AUC or F1/F-beta (recall-weighted if false negatives are costlier), Matthews Correlation Coefficient for a single robust summary, and inspect the confusion matrix directly at a chosen operating threshold rather than relying on a single aggregate number.

**Q2. Walk through why ROC-AUC can look artificially good on an imbalanced dataset while PR-AUC reveals a real problem.**
False Positive Rate's denominator is $FP + TN$; when negatives vastly outnumber positives, $TN$ is huge, so even a large absolute number of false positives barely nudges FPR upward, making the ROC curve look good. Precision's denominator is $TP + FP$ — it directly reflects how "polluted" the positive predictions are with false positives, with no dilution from the huge true-negative pool — so PR-AUC will visibly drop if the model produces many false positives relative to true positives, even when ROC-AUC still looks fine. This is why PR-AUC (and comparing it against the baseline equal to positive-class prevalence) is the standard for imbalanced problems.

**Q3. You cross-validate a time series model using standard shuffled K-Fold and get great results, but the model performs terribly in production. What went wrong?**
Standard K-Fold shuffles rows randomly, so training folds can contain data points from *after* the validation fold in time — the model implicitly learns from future information (e.g., via features engineered with future context, or simply pattern-matching against nearly-identical adjacent time points) that will never be available at real prediction time. The fix is `TimeSeriesSplit` / walk-forward validation, ensuring every validation fold's data is strictly later in time than its training data, matching how the model will actually be used in deployment.

**Q4. Explain the bias-variance decomposition and how you'd diagnose which one is the dominant problem using a learning curve.**
Expected test error decomposes into $\text{Bias}^2 + \text{Variance} + \text{Irreducible noise}$. Bias is error from wrong model assumptions (systematic, doesn't improve with more data); variance is sensitivity to the specific training sample (does improve with more data, up to a point). On a learning curve (train/validation error vs. training set size): if both curves are high and close together and plateaued, that's high bias — you need a more complex model or better features, not more data. If training error is low but there's a large, persistent gap to validation error, that's high variance — regularization, simplification, or more data will help.

**Q5. What's the difference between L1 and L2 regularization mathematically and behaviorally, and when would you choose Elastic Net instead of either alone?**
L2 adds a penalty proportional to $\sum \beta_j^2$, whose gradient shrinks toward (but not exactly to) zero, producing smooth shrinkage across all coefficients — good for many mildly useful, possibly correlated features. L1 adds $\sum|\beta_j|$, whose constant-magnitude gradient can drive coefficients exactly to zero, giving automatic feature selection and sparse models — but with correlated features, L1 arbitrarily picks one and zeroes the rest, which can be unstable. Elastic Net blends both, giving sparsity like Lasso while sharing weight among correlated features more like Ridge — I'd choose it when I suspect both sparsity is desirable AND there's meaningful multicollinearity among the true signal features.

**Q6. Explain calibration and why a model can have a high AUC but still be poorly calibrated. Give a scenario where this matters.**
AUC measures *discrimination* — whether the model ranks positive examples above negative ones — and is invariant to any monotonic transformation of the predicted scores. Calibration measures whether the predicted probability values themselves match observed frequencies (e.g., among all predictions of 0.7, are ~70% actually positive). A model can perfectly rank-order examples (high AUC) while its probability outputs are systematically over- or under-confident (poor calibration), because AUC never looks at the absolute scale of the scores. This matters in scenarios where the probability value feeds a downstream calculation, e.g., a fraud model whose score is multiplied by transaction value to compute expected loss for a blocking decision — miscalibrated probabilities would make that expected-loss calculation wrong even if the model correctly ranks fraud higher than non-fraud.

**Q7. Describe walk-forward validation for a demand forecasting model and why leave-one-out cross-validation would be inappropriate here.**
Walk-forward validation trains on all data up to time $t$, validates on the next window $[t, t+\Delta]$, then slides or expands the window forward and repeats — always keeping training data strictly earlier than validation data, respecting the temporal order the model will face in production. LOOCV would be inappropriate because it requires randomly removing one point and training on "the rest," which for time series would mean training on future data to predict a point in the past for most iterations — a direct violation of temporal causality that would leak future information and produce meaningless, overly optimistic error estimates.

**Q8. When would you prefer MAE over RMSE, and what does a large gap between the two tell you?**
I'd prefer MAE when I want a metric robust to outliers (linear penalty, doesn't let a handful of extreme errors dominate the score) and when all errors should be weighted equally regardless of magnitude — e.g., median household income prediction where a few billionaire outliers shouldn't dictate model selection. RMSE, because it squares errors, is more sensitive to large errors and is preferable when large misses are disproportionately costly (and mathematically convenient since it's differentiable everywhere and directly tied to the squared-error loss most regression models optimize). Since $RMSE \geq MAE$ always, a large gap between them signals that error magnitudes are heavy-tailed — a subset of predictions have much larger errors than the typical case — worth investigating those specific data points.

**Q9. A stakeholder reports R² = 0.92 on the training set but you compute R² = -0.15 on the test set. Is that possible, and what does it mean?**
Yes — $R^2$ is guaranteed to be non-negative only on the exact data used to fit the least-squares model; on a different (test) dataset, if the model's predictions are worse than simply predicting the target's mean, $SS_{res} > SS_{tot}$ and $R^2$ goes negative. A large gap like this (0.92 train vs. -0.15 test) is a strong signal of severe overfitting — the model has essentially memorized noise/idiosyncrasies of the training data with no generalizable signal, and needs regularization, simpler architecture, more data, or a review for leakage.

**Q10. Explain Bayesian optimization for hyperparameter tuning and when it's worth the added complexity over random search.**
Bayesian optimization fits a probabilistic surrogate model (commonly a Gaussian Process) over the space of hyperparameters to the observed (hyperparameters → validation score) pairs so far, then uses an acquisition function (e.g., Expected Improvement) to pick the next point that best balances trying promising regions (exploitation) against reducing uncertainty in unexplored regions (exploration). It's worth the complexity when each model evaluation is expensive (large models, big data, long training runs) so that squeezing more value out of every trial via informed search matters; it's overkill for cheap models where you could just brute-force hundreds of random search trials for the same or less wall-clock cost, and it becomes less effective in very high-dimensional hyperparameter spaces due to the surrogate model's own scaling limits.

**Q11. How would you determine if a 1% AUC improvement from a new model is statistically significant, and why can't you just trust a naive paired t-test on standard K-fold CV scores?**
I'd compute per-fold scores for both models on the *same* CV folds and run a paired t-test on the differences, or better, the Nadeau & Bengio corrected paired t-test, or bootstrap a confidence interval on the score difference. The reason a naive paired t-test on standard K-fold scores can be misleading is that the folds are not independent — each fold's training set overlaps heavily with every other fold's training set (they share $k-2$ folds' worth of data), violating the independence assumption behind the standard t-test's variance estimate and inflating the false-positive rate (you're more likely to call a difference "significant" than you should be). I'd also weigh statistical significance against practical significance — a p<0.05 but tiny, operationally irrelevant improvement may not justify shipping a more complex model.

**Q12. What's the difference between McNemar's test and a paired t-test on CV fold scores, and when would you use each?**
McNemar's test compares two classifiers' predictions on a single shared test set by looking only at the disagreement cases (samples where exactly one model got it right), testing whether the disagreements are symmetric — appropriate when you have one fixed holdout set and binary correct/incorrect outcomes. The paired t-test (or its corrected variant) compares two models' *aggregate scores* (e.g., accuracy, AUC, RMSE) across multiple CV folds, testing whether the average score difference is significant — appropriate when you have cross-validated performance across several folds/repeats rather than a single fixed test set. Use McNemar's when comparing against a fixed production baseline on one held-out set; use the paired t-test approach when you have your own CV setup for both models on the same splits.

**Q13. Explain early stopping as a regularization technique and how you'd implement it correctly (including pitfalls).**
Early stopping monitors validation loss during iterative training (e.g., gradient boosting rounds, neural network epochs) and halts training (restoring the best checkpoint) once validation loss stops improving for a set patience window, even if training loss keeps falling — this implicitly caps model complexity/capacity reached during optimization, preventing the model from continuing to fit training-set noise. Implementation pitfalls: the validation set used for the early-stopping decision must be separate from the final test set (otherwise you're implicitly tuning on your "unbiased" test set); patience should be tuned — too small risks stopping before a temporary plateau resolves (some loss curves have noisy plateaus followed by further improvement), too large wastes compute and risks eventual overfitting anyway; always restore the *best* checkpoint, not simply the model state at the stopping iteration.

**Q14. Describe NDCG and explain why it's preferred over precision@k for evaluating a search ranking system with graded relevance labels (e.g., 0–4 relevance scale).**
Precision@k treats every item in the top-k as equally either "relevant" or "not," and ignores order within the top-k entirely (a relevant item at rank 1 counts the same as one at rank k). NDCG uses graded relevance (via the $2^{rel}-1$ gain term, so a relevance-4 item contributes far more than a relevance-1 item) and applies a logarithmic *position discount* so that the same relevant item contributes less the further down the ranking it appears, then normalizes against the ideal (perfectly sorted) ranking to produce a comparable [0,1] score across queries with different numbers of relevant items. This makes NDCG far better suited to systems where relevance is a spectrum and where the ordering — not just membership in the top-k — is what the business actually cares about (e.g., search engine result ordering, recommendation carousels).

**Q15. You need to choose a metric for a medical screening test where missing a disease case (false negative) is far more dangerous than a false alarm (false positive). Explain your metric choice and the tradeoff involved.**
I'd prioritize recall (sensitivity) — the fraction of actual disease cases correctly caught — potentially reporting an $F_2$ score (weights recall twice as heavily as precision) as a single summary, and I'd tune the classification threshold (via the PR curve) to hit a target recall (e.g., 95%+) even at the cost of more false positives, since a missed diagnosis is far costlier than an unnecessary follow-up test. The tradeoff: pushing recall very high inevitably drives precision down (more false alarms, more unnecessary follow-up procedures/anxiety/cost), so I'd work with clinical stakeholders to explicitly define the acceptable false-positive rate/cost ratio (this is essentially cost-sensitive learning, choosing threshold $= C_{FP}/(C_{FP}+C_{FN})$) rather than optimizing a generic metric in a vacuum.

**Q16. Your teammate runs `GridSearchCV(cv=5)` and reports `grid.best_score_` as "our model's expected AUC in production." Why is that number optimistically biased, and what would you do instead?**
The 5 folds were used to *select* which hyperparameter combination scores best on exactly those folds, so `best_score_` is the maximum over many candidate scores on the same validation data — not an independent estimate of any single model's generalization error. This is a form of winner's-curse / multiple-comparisons bias: the more configurations tried, the more likely one of them scored well partly due to validation-fold noise rather than genuinely better generalization. I'd use **nested cross-validation** instead: an outer CV loop provides held-out folds that never influence hyperparameter selection, and an inner CV loop (run only on each outer-training fold) does the actual tuning — the outer-fold scores, averaged, give an honest estimate of what to expect in production. `grid.best_score_` is still useful for actually *choosing* the final hyperparameters to deploy, just not for *reporting* expected performance.

**Q17. Walk through the structure of nested cross-validation — what does the inner loop do, what does the outer loop do, and why must they use disjoint data?**
The outer loop splits the full dataset into $k_{outer}$ folds; for each outer fold, everything except that fold is treated as an "outer-training" set and the fold itself as an "outer-test" set, touched only for final scoring. Within each outer-training set, the inner loop runs its own CV (e.g., $k_{inner}$-fold `GridSearchCV`) purely to pick the best hyperparameters for that outer-training set — the inner loop never sees the outer-test fold in any capacity, not even indirectly through a fitted preprocessing step. The winning inner configuration is then refit on the whole outer-training set and scored once on the untouched outer-test fold. They must be disjoint because the entire point is to measure how a "tune-then-fit" *process* generalizes to genuinely unseen data; if the inner loop ever saw outer-test data, the outer score would suffer exactly the same optimistic bias that motivated nesting in the first place.

**Q18. Your training pipeline uses `X_val` both to decide when to early-stop an LGBM model and to pick between three candidate `learning_rate` values. Is it OK to then report `X_val`'s AUC as your model's test performance? Why or why not?**
No. `X_val` drove two decisions — the stopping round for each candidate and the choice of which candidate to keep — so its score has already been optimized against twice, the same underlying issue nested CV addresses at the CV-fold level. It's a genuinely different, unbiased number that's needed: a `X_test` set that no early-stopping decision and no hyperparameter comparison ever touched, evaluated exactly once after `best_params` and the corresponding stopped model are fully frozen. Reusing `X_val`'s already-optimized score as if it were that number overstates expected production performance, sometimes substantially if many candidates were compared.

**Q19. What's the difference between "the validation set used for early stopping" and "the validation set used for hyperparameter search," and can they be the same physical data?**
Early stopping uses validation performance to decide *when* to stop training a specific, already-fixed configuration (a within-model decision). Hyperparameter search uses validation performance to decide *which* configuration to keep at all (a between-model decision). They answer different questions, but in practice they're very often the same physical held-out split — e.g., in an XGBoost/LightGBM tuning loop, each candidate's `eval_set` both stops its own training early and produces the score used to compare it against other candidates. That's fine as a modeling choice; the trip-up is only in treating that split's resulting score as an unbiased *report* of generalization — for that, a separate test set (or nested CV) is still required.

---

## Rapid-Fire Interview Q&A

| # | Question | Answer |
|---|---|---|
| 1 | What's the main risk of target encoding? | Leakage — the target influences the very feature derived from it; mitigate with out-of-fold/smoothed encoding. |
| 2 | Does scaling matter for tree-based models? | No — tree splits are invariant to monotonic transforms of a single feature. |
| 3 | Why use `log1p` instead of `log`? | Handles zero values safely (`log(0)` is undefined; `log1p(0)=0`). |
| 4 | Why do you need both sin and cos for cyclical encoding? | Sine alone isn't injective over a full period; the (sin, cos) pair uniquely locates a point on the circle. |
| 5 | What does PCA maximize? | Variance captured by each successive orthogonal component. |
| 6 | Is t-SNE deterministic? | No — sensitive to random initialization and perplexity; re-runs can differ. |
| 7 | Can you use t-SNE output as input to a classifier? | Not recommended — it distorts global distances and has no reliable out-of-sample transform. |
| 8 | What's SMOTE short for? | Synthetic Minority Oversampling Technique. |
| 9 | Where must resampling happen relative to CV folds? | Inside each fold — never before the train/test split. |
| 10 | Name one advantage of MCC over F1. | MCC uses all four confusion-matrix cells (including TN), making it more robust to class imbalance and label-swap symmetric. |
| 11 | Why is ROC-AUC potentially misleading under severe imbalance? | FPR's denominator is dominated by the large number of true negatives, hiding a high absolute false-positive count. |
| 12 | What does a PR-AUC baseline (random classifier) equal? | The positive-class prevalence, not 0.5. |
| 13 | What does adjusted R² correct for that plain R² doesn't? | Penalizes adding predictors that don't genuinely improve fit; plain R² never decreases with more features. |
| 14 | Can R² be negative? | Yes, on held-out data, if the model performs worse than predicting the mean. |
| 15 | MAPE's biggest weakness? | Undefined/explodes near y=0 and is asymmetric between over- and under-predictions. |
| 16 | What does NDCG add over MAP? | Support for graded (non-binary) relevance plus a logarithmic position discount. |
| 17 | What does MRR focus on? | Only the rank position of the first relevant result. |
| 18 | What is a reliability diagram used for? | Visualizing calibration — predicted probability vs. observed frequency per bin. |
| 19 | Name two ways to fix probability miscalibration. | Platt scaling (logistic fit on scores) and isotonic regression (monotonic step-function fit). |
| 20 | State the bias-variance decomposition. | Expected error = Bias² + Variance + Irreducible noise. |
| 21 | High train error and high, similar validation error means what? | High bias / underfitting. |
| 22 | Low train error, much higher validation error means what? | High variance / overfitting. |
| 23 | Which regularizer can zero out coefficients: L1 or L2? | L1 (Lasso). |
| 24 | Why doesn't L2 typically produce exact zeros? | Its penalty gradient shrinks to zero as the coefficient approaches zero, so it stops driving it further down. |
| 25 | What does dropout do at training time vs inference time? | Randomly zeroes activations during training; all neurons active (with scaling) at inference. |
| 26 | Why does grid search scale poorly? | Combinations grow exponentially with the number of hyperparameters. |
| 27 | Why can random search beat grid search on a fixed budget? | It samples more distinct values per hyperparameter dimension rather than wasting trials on redundant combinations. |
| 28 | What surrogate model is classically used in Bayesian optimization? | A Gaussian Process. |
| 29 | What does successive halving/Hyperband exploit? | Early performance signals to kill bad configurations before wasting the full compute budget on them. |
| 30 | Why is standard K-Fold invalid for time series? | It shuffles rows, letting future data leak into training folds that predict "earlier" validation folds. |
| 31 | What does GroupKFold prevent? | Rows from the same entity appearing in both train and validation, which would let the model memorize entity-specific patterns. |
| 32 | Why can LOOCV have high variance despite low bias? | Its n validation folds are highly correlated (differ by one sample), so the overall error estimate can be noisy across different datasets. |
| 33 | What test compares two classifiers' errors on one fixed shared test set? | McNemar's test. |
| 34 | What test compares two models' scores across CV folds? | A paired t-test (ideally the Nadeau-Bengio corrected version). |
| 35 | Why is the naive paired t-test on K-fold scores risky? | CV fold training sets overlap, violating the independence assumption and inflating false positives. |
| 36 | What single number does average precision (AP) summarize? | The area under the precision-recall curve for one ranked list/query. |
| 37 | Name one filter, one wrapper, and one embedded feature selection method. | Filter: chi-square/mutual information; Wrapper: RFE; Embedded: Lasso/tree importance. |
| 38 | Why is impurity-based tree feature importance sometimes misleading? | Biased toward high-cardinality/continuous features and computed on training data, which can overstate importance from overfitting. |
| 39 | What's the fix for that bias? | Use permutation importance computed on a held-out validation set instead. |
| 40 | Give the formula for the optimal cost-sensitive decision threshold. | threshold = C_FP / (C_FP + C_FN). |
| 41 | Why is `GridSearchCV.best_score_` biased as a report of generalization performance? | It's the max over many candidates scored on the same folds used to pick the winner — a winner's-curse/multiple-comparisons bias. |
| 42 | What does the inner loop do in nested cross-validation? | Hyperparameter search, using only the current outer-training fold's data. |
| 43 | What does the outer loop do in nested cross-validation? | Scores the inner loop's winning, refit model on a fold it never influenced — the unbiased generalization estimate. |
| 44 | Can the same validation split drive both early stopping and hyperparameter selection? | Yes, that's common practice — but its resulting score can't then be reported as an unbiased test number. |
| 45 | What single formula gives the Variance Inflation Factor for feature j? | VIF_j = 1 / (1 - R_j²), from regressing feature j on all other predictors. |
| 46 | What does the Brier score measure? | Mean squared error between predicted probability and the actual binary outcome. |
| 47 | Brier score vs. log loss — which punishes confident wrong predictions more severely? | Log loss (unbounded, logarithmic); Brier is bounded and gentler. |

