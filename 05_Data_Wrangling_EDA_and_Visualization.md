# Data Wrangling, Exploratory Data Analysis (EDA), and Data Visualization

Data wrangling, EDA, and visualization are the load-bearing skills behind every model, dashboard, and decision that follows them — and they are also the single most heavily interviewed practical skill area across data roles, because they cannot be faked with buzzwords.

- **Data Scientists** spend the majority of real project time here — not modeling. A model trained on unvalidated, leaky, or silently-corrupted data produces confidently wrong answers, and the interviewer's real question when they hand you a messy CSV is "can I trust your judgment with raw data before anyone reviews it?" Expect live take-home-style exercises, "walk me through your EDA process," and deep questions on missing data, outliers, and distributional assumptions.
- **Machine Learning Engineers** are judged on whether the *pipelines* that wrangle data are correct, reproducible, and won't leak information from train to test, break in production when a schema drifts, or silently corrupt a feature store. Interviews probe data validation, type safety, scaling in pipelines, and how you'd catch a broken upstream feed before it degrades a model in production.
- **AI Engineers** increasingly work with messy, semi-structured, and streaming data feeding retrieval systems, agents, and evaluation harnesses — profiling embeddings, catching drifted inputs, and visualizing model behavior over time. The fundamentals (distributions, correlation, outliers, honest charts) are identical; only the artifacts change.

Across all three roles, the bar is the same: know *why* a technique works, when it breaks, what its assumptions are, and how to explain a chart or a cleaning decision to a non-technical stakeholder in one sentence.

## Table of Contents

1. [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
   - [Missing Data: Types, Detection, and Imputation](#missing-data-types-detection-and-imputation)
   - [Outlier Detection and Treatment](#outlier-detection-and-treatment)
   - [Duplicate Detection and Handling](#duplicate-detection-and-handling)
   - [Data Type Conversions](#data-type-conversions)
   - [String Cleaning and Standardization](#string-cleaning-and-standardization)
   - [Handling Inconsistent Categorical Labels](#handling-inconsistent-categorical-labels)
   - [Data Validation and Schema Enforcement](#data-validation-and-schema-enforcement)
   - [Interview Questions](#interview-questions)
2. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [A Structured Framework for Exploring an Unfamiliar Dataset](#a-structured-framework-for-exploring-an-unfamiliar-dataset)
   - [Univariate Analysis](#univariate-analysis)
   - [Bivariate and Multivariate Analysis](#bivariate-and-multivariate-analysis)
   - [Selection Bias, Survivorship Bias, and Spurious Correlations](#selection-bias-survivorship-bias-and-spurious-correlations)
   - [Detecting Relationships, Interactions, and Multicollinearity (VIF)](#detecting-relationships-interactions-and-multicollinearity-vif)
   - [Time Series EDA Basics](#time-series-eda-basics)
   - [Handling Large Datasets: Sampling and Out-of-Core EDA](#handling-large-datasets-sampling-and-out-of-core-eda)
   - [Data Profiling and Automated EDA Tools](#data-profiling-and-automated-eda-tools)
   - [Interview Questions](#interview-questions-1)
3. [Data Visualization Principles and Techniques](#data-visualization-principles-and-techniques)
   - [Choosing the Right Chart Type](#choosing-the-right-chart-type)
   - [Geospatial Visualization: Choropleths and Map Pitfalls](#geospatial-visualization-choropleths-and-map-pitfalls)
   - [Visualization Best Practices](#visualization-best-practices)
   - [Tools: Matplotlib, Seaborn, Plotly, and BI Tools](#tools-matplotlib-seaborn-plotly-and-bi-tools)
   - [Dashboarding Concepts](#dashboarding-concepts)
   - [Interview Questions](#interview-questions-2)
4. [Rapid-Fire Interview Q&A](#rapid-fire-interview-qa)

---

## Data Cleaning and Preprocessing

### Missing Data: Types, Detection, and Imputation

**Why missing data matters:** the *mechanism* by which data goes missing determines whether a technique will fix your dataset or quietly bias it. Interviewers care far more about whether you can classify the mechanism and justify a strategy than whether you know the pandas syntax.

**The three mechanisms (Rubin's classification):**

| Type | Definition | Example | Consequence if ignored |
|---|---|---|---|
| **MCAR** (Missing Completely At Random) | Missingness is unrelated to any observed or unobserved variable | A lab sample is dropped due to a random equipment glitch | Listwise deletion is unbiased (but loses power) |
| **MAR** (Missing At Random) | Missingness depends on *observed* variables, not on the missing value itself | Older survey respondents are less likely to report income, but "age" is observed | Deletion biases estimates; imputation conditioned on observed variables (e.g., regression/MICE) can be unbiased |
| **MNAR** (Missing Not At Random) | Missingness depends on the *unobserved* value itself | High earners refuse to report income specifically because it is high | No standard imputation fully fixes this; requires domain modeling, sensitivity analysis, or a missingness indicator that captures signal |

**How to reason about the mechanism in practice** (there is no statistical test that proves MCAR vs. MAR vs. MNAR with certainty, but you can gather evidence):
1. Compare distributions of other variables between rows with missing vs. non-missing target — if they differ significantly, it's not MCAR.
2. Little's MCAR test gives a formal p-value for the MCAR hypothesis (low p → reject MCAR).
3. Ask *why* the data would be missing from a domain standpoint — this is usually more convincing to an interviewer than a test.

**Detection:**

```python
import pandas as pd
import numpy as np

# Basic detection
df.isna().sum()                      # count per column
df.isna().mean().sort_values(ascending=False)   # % missing per column
df.isna().sum(axis=1).value_counts()  # rows by number of missing fields

# Visual: missingness matrix / heatmap (great for spotting MAR patterns)
import missingno as msno
msno.matrix(df)
msno.heatmap(df)     # correlation of "is-missing" indicators between columns
msno.dendrogram(df)  # clusters columns that go missing together (often = same upstream cause)
```

A strong signal of **MAR** (and a good interview answer) is when `msno.heatmap` shows two columns' missingness indicators are highly correlated with each other or with a third *fully observed* column — e.g., `income` is missing exactly when `employment_status == 'unemployed'`.

**Little's MCAR test:**

```python
# pip install pyampute (or use statsmodels / R's naniar equivalent conceptually)
from pyampute.exploration.mcar_statistical_tests import MCARTest
mt = MCARTest(method="little")
p_value = mt(df[numeric_cols])
# p < 0.05 -> reject MCAR (missingness is likely MAR or MNAR)
```

**Imputation strategies, from simplest to most sophisticated:**

| Strategy | How it works | Best for | Weakness |
|---|---|---|---|
| **Listwise / pairwise deletion** | Drop rows (or just the pair of columns used in a calc) with missing values | MCAR, small % missing, plenty of data | Loses power; biased under MAR/MNAR |
| **Mean/median imputation** | Fill with column mean (numeric, symmetric) or median (skewed/outlier-prone) | Quick baselines, MCAR, low missing % | Shrinks variance, distorts correlations, ignores relationships between features |
| **Mode imputation** | Fill categorical with most frequent category | Categorical MCAR | Can inflate the dominant class artificially |
| **Constant / "Missing" category** | Fill with a sentinel value or explicit "Unknown" category | When missingness itself is informative (tree models handle this natively) | Not valid for models assuming continuity |
| **Forward/backward fill** | Carry last/next observed value forward/backward | Time series, sensor data | Wrong if the true value actually changed during the gap |
| **KNN Imputation** | Impute using the average (or weighted average) of the *k* nearest neighbors in feature space | MAR, moderate size numeric datasets, when features are correlated | Sensitive to feature scaling and distance metric; slow on large data; needs complete neighbors |
| **MICE (Multiple Imputation by Chained Equations)** | Iteratively models each missing column as a function of all other columns (round-robin regression), repeated across multiple imputed datasets, results pooled | MAR, when you need to preserve inter-variable relationships and quantify imputation uncertainty | Computationally heavier; requires careful convergence checking |
| **Model-based (e.g., regression, random forest, deep learning like `MissForest`/autoencoders)** | Train a supervised model per column to predict missing values from the rest | Complex non-linear relationships, MAR | Risk of overfitting/leaking target information if not done inside CV folds |
| **Indicator + impute** | Add a binary `col_was_missing` flag alongside any imputation | MAR/MNAR where missingness itself is predictive | Adds dimensionality; must be added consistently in train/test |

```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Mean / median / most_frequent
num_imputer = SimpleImputer(strategy="median")
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# KNN imputation — scale first, KNN is distance-based
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled = scaler.fit_transform(df[num_cols])
knn_imputer = KNNImputer(n_neighbors=5, weights="distance")
imputed = knn_imputer.fit_transform(scaled)

# MICE-style iterative imputation
mice_imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=50, random_state=0),
    max_iter=10,
    random_state=0,
)
df[num_cols] = mice_imputer.fit_transform(df[num_cols])

# Missingness indicator
for col in ["income", "credit_score"]:
    df[f"{col}_was_missing"] = df[col].isna().astype(int)
```

**When to drop vs. impute — decision guide:**

| Situation | Recommendation |
|---|---|
| Column is > 60–70% missing and not domain-critical | Drop the column |
| Column is missing < 5%, MCAR, large dataset | Simple deletion or mean/median is safe |
| Missing % is moderate (5–40%), relationships exist between features | KNN or MICE |
| Missingness is itself predictive (MNAR-like) | Keep a missingness indicator; consider tree-based models that handle `NaN` natively (XGBoost, LightGBM) |
| Time series with short gaps | Forward-fill / interpolation |
| You need valid statistical inference (not just point estimates) | Multiple imputation (MICE) to propagate uncertainty, not single imputation |

**Critical pitfall — leakage:** always `fit` imputers (mean, KNN, IterativeImputer) on the **training fold only**, inside a `Pipeline`/`ColumnTransformer`, and `transform` on validation/test. Fitting an imputer on the full dataset before splitting leaks test-set statistics into training.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                       ("scale", StandardScaler())]), num_cols),
])
# preprocess.fit(X_train) then preprocess.transform(X_test) — never fit on X_test
```

**Other pitfalls:**
- Imputing before splitting train/test (leakage, see above).
- Using mean imputation on skewed data — the mean is pulled by the tail; use median.
- Imputing categorical missingness with mode when "missing" is itself meaningful (e.g., a survey skip pattern) — losing that signal.
- Not checking whether missingness correlates with the *target* variable — that correlation, if present, means deletion changes the target distribution.
- Forgetting to re-check for missing values introduced by joins/merges (a left join with no match creates new `NaN`s downstream, after your original missing-data pass).

### Outlier Detection and Treatment

**Concept:** an outlier is an observation that deviates markedly from the rest of the data — but "markedly" needs to be operationalized, and the correct response (remove, cap, transform, or keep and investigate) depends entirely on *why* it's there: measurement error, data-entry error, or a genuine rare event.

**Detection methods:**

| Method | How it works | Assumptions | Good for |
|---|---|---|---|
| **Z-score** | `z = (x - mean) / std`; flag `\|z\| > 3` (or 2.5) | Approximately normal distribution | Quick univariate screening on roughly Gaussian features |
| **Modified Z-score (MAD-based)** | Uses median and median absolute deviation instead of mean/std | Robust to the outliers themselves | Skewed data, small samples |
| **IQR method** | Flag values outside `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]` | Distribution-free | The default box-plot rule; robust, easy to explain |
| **Isolation Forest** | Randomly partitions data; outliers require fewer splits to isolate | None (tree-based, non-parametric) | High-dimensional, multivariate outliers, large datasets |
| **DBSCAN-based** | Points not belonging to any density cluster (labeled `-1`) are outliers | Density-based, works with arbitrary cluster shapes | Multivariate outliers when clusters are non-convex; noisy spatial/geo data |
| **Local Outlier Factor (LOF)** | Compares local density of a point to its neighbors' density | None | Outliers that are only "unusual" relative to a local neighborhood, not globally |
| **Visual methods** | Box plot, scatter plot, histogram, pair plot | None | First-pass exploration, communicating to stakeholders |
| **Mahalanobis distance** | Multivariate distance accounting for covariance structure | Roughly elliptical/Gaussian multivariate data | Multivariate outliers when features are correlated |

```python
import numpy as np
import pandas as pd

# 1. Z-score
z_scores = (df["value"] - df["value"].mean()) / df["value"].std()
outliers_z = df[z_scores.abs() > 3]

# 2. Modified Z-score (robust)
median = df["value"].median()
mad = (df["value"] - median).abs().median()
mod_z = 0.6745 * (df["value"] - median) / mad
outliers_mad = df[mod_z.abs() > 3.5]

# 3. IQR method
Q1, Q3 = df["value"].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
outliers_iqr = df[(df["value"] < lower) | (df["value"] > upper)]

# 4. Isolation Forest (multivariate)
from sklearn.ensemble import IsolationForest
iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=0)
df["outlier_flag"] = iso.fit_predict(df[num_cols])   # -1 = outlier, 1 = inlier
df["anomaly_score"] = iso.decision_function(df[num_cols])  # lower = more abnormal

# 5. DBSCAN-based
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(df[num_cols])
labels = DBSCAN(eps=0.5, min_samples=10).fit_predict(X_scaled)
df["is_outlier_dbscan"] = labels == -1

# 6. Visual
import matplotlib.pyplot as plt
import seaborn as sns
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(y=df["value"], ax=axes[0])
sns.scatterplot(x=df.index, y=df["value"], ax=axes[1])
plt.tight_layout()
```

**Deciding what to do once found:**

| Decision | When to use it | Notes |
|---|---|---|
| **Investigate first, always** | Every time before removing anything | Is it a data-entry typo (age = 999), a unit mismatch (kg vs. lb), or a real extreme (a billionaire's income)? |
| **Remove** | Confirmed data-entry error, sensor malfunction, or truly irrelevant to the question being asked | Document and justify — never silently drop rows |
| **Cap / Winsorize** | Genuine extreme values you want to keep but whose influence you want to bound (e.g., for linear regression) | `df['value'].clip(lower, upper)`; preserves row count |
| **Transform** | Right-skewed data with natural extreme values (income, price) | Log, Box-Cox, or Yeo-Johnson transforms reduce the influence of extreme values without deleting them |
| **Keep, use robust methods** | Outliers are real and meaningful (fraud, rare disease) | Use robust statistics (median, IQR) or robust models (tree-based methods, RANSAC, Huber loss) instead of removing signal |
| **Keep, flag as a feature** | Outlier status is itself predictive (e.g., "unusually large transaction" is the fraud signal) | Add a binary/anomaly-score feature rather than deleting the row |

```python
# Winsorizing / capping
df["value_capped"] = df["value"].clip(lower=lower, upper=upper)

# Log transform (handles right skew, requires positive values)
df["value_log"] = np.log1p(df["value"])   # log1p handles zeros safely

# Box-Cox (positive only) / Yeo-Johnson (handles zero/negative)
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method="yeo-johnson")
df["value_transformed"] = pt.fit_transform(df[["value"]])
```

**Pitfalls:**
- Applying z-score to skewed data — the mean/std are themselves distorted by the outliers you're trying to find (chicken-and-egg problem); prefer IQR or MAD-based z-scores for skewed data.
- Removing outliers *before* splitting train/test — if you fit thresholds using the full dataset, you leak test information.
- Treating outliers identically across a heterogeneous population — a "high" transaction value for a retail customer vs. a wholesale account are different distributions; segment before flagging.
- Blind removal without investigating context — the outlier might be the most business-critical row in the dataset (e.g., the exact fraud case you're building the model to catch).
- Using Isolation Forest / DBSCAN without scaling features first — distance/density-based methods are scale-sensitive.
- Not re-running outlier detection after fixing an upstream bug — a bug fix can reveal new statistical outliers that were previously masked.

### Duplicate Detection and Handling

**Concept:** duplicates range from trivial (byte-identical rows from a double `INSERT`) to subtle (the same customer entered twice with a typo'd name, or the same event logged once by two different pipelines with different timestamps).

```python
# Exact duplicates
df.duplicated().sum()
df[df.duplicated(keep=False)]          # show all copies, not just later ones
df_deduped = df.drop_duplicates()

# Duplicates on a subset of columns (e.g., business key)
df.duplicated(subset=["customer_id", "order_date"]).sum()
df_deduped = df.drop_duplicates(subset=["customer_id", "order_date"], keep="last")

# Fuzzy / near-duplicate detection (e.g., "John Smith" vs "Jon Smith")
from rapidfuzz import fuzz
def is_near_duplicate(a, b, threshold=90):
    return fuzz.ratio(a, b) >= threshold

# Record linkage at scale (blocking + similarity scoring)
import recordlinkage
indexer = recordlinkage.Index()
indexer.block("zip_code")          # only compare records sharing a block key
candidate_links = indexer.index(df)
compare = recordlinkage.Compare()
compare.string("name", "name", method="jarowinkler", threshold=0.85)
compare.exact("dob", "dob")
features = compare.compute(candidate_links, df)
matches = features[features.sum(axis=1) >= 2]
```

**Decision guide:**

| Situation | Action |
|---|---|
| 100% identical rows across all columns | Safe to drop — almost always an ingestion artifact |
| Same business key (e.g., `order_id`), different timestamps/values | Investigate: is it an update? Keep `last` by timestamp, or reconcile |
| Same entity, slightly different text (typos, casing, whitespace) | Normalize/standardize first (see next section), then re-check duplicates |
| Same entity across systems with different IDs | Use fuzzy matching / record linkage, not a simple `.duplicated()` |
| Legitimate repeated events (e.g., a customer buying the same product twice on different days) | **Not** a duplicate — verify grain of the data before deduplicating |

**Pitfalls:**
- Deduplicating before understanding the table's grain — a "duplicate" `customer_id` in an `orders` table is expected; the same `order_id` twice is not.
- Using `keep="first"` vs `keep="last"` without checking which record has more complete/correct information.
- Losing an audit trail — instead of `drop_duplicates()` silently, log how many rows were removed and why, especially in production pipelines.

### Data Type Conversions

**Concept:** raw data (especially from CSV, APIs, or scraped sources) frequently mistypes numbers as strings, dates as objects, and booleans as `"Yes"/"No"` strings — and pandas' default type inference is often wrong or memory-inefficient.

```python
# Inspect current types and memory footprint
df.info(memory_usage="deep")
df.dtypes

# Numeric conversion, coercing bad values to NaN instead of raising
df["price"] = pd.to_numeric(df["price"], errors="coerce")

# Datetime conversion
df["order_date"] = pd.to_datetime(df["order_date"], format="%Y-%m-%d", errors="coerce")
df["order_date"] = pd.to_datetime(df["order_date"], utc=True)  # normalize timezone

# Categorical (huge memory savings + faster groupby for low-cardinality strings)
df["region"] = df["region"].astype("category")

# Boolean from string flags
df["is_active"] = df["is_active"].map({"Y": True, "N": False, "yes": True, "no": False})

# Downcasting numeric types to save memory
df["quantity"] = pd.to_numeric(df["quantity"], downcast="integer")
df["amount"] = pd.to_numeric(df["amount"], downcast="float")

# Nullable integer type (int columns with NaN get silently upcast to float otherwise)
df["age"] = df["age"].astype("Int64")   # capital I = pandas nullable integer
```

**Pitfalls:**
- Integer columns with missing values silently become `float64` in classic pandas (`NaN` isn't representable in `int64`) — use pandas nullable dtypes (`Int64`, `boolean`) to preserve intent.
- `errors="raise"` (the default in some contexts) crashing a whole pipeline on one bad row — decide deliberately between `coerce` (silently null bad values, but log how many) and `raise` (fail loudly).
- Parsing dates without specifying `format=` on large data — pandas' fallback date inference is slow and can misparse ambiguous formats (`01/02/2023` = Jan 2 or Feb 1?).
- Converting high-cardinality strings (e.g., free-text) to `category` — no memory benefit, and can hurt performance.
- Forgetting timezone handling — comparing naive and timezone-aware timestamps raises errors or silently misaligns events.

### String Cleaning and Standardization

**Concept:** text fields are the messiest columns in almost any dataset — encoding issues, inconsistent casing, leading/trailing whitespace, mixed delimiters, and typos.

```python
# Whitespace and casing
df["city"] = df["city"].str.strip().str.lower()

# Remove punctuation / special characters
df["city"] = df["city"].str.replace(r"[^\w\s]", "", regex=True)

# Standardize whitespace (multiple spaces -> one)
df["name"] = df["name"].str.replace(r"\s+", " ", regex=True).str.strip()

# Unicode normalization (accents, full-width chars, etc.)
import unicodedata
df["name"] = df["name"].apply(lambda s: unicodedata.normalize("NFKC", s) if pd.notna(s) else s)

# Extract structured info from free text with regex
df["area_code"] = df["phone"].str.extract(r"\((\d{3})\)")

# Encoding fixes when reading files
df = pd.read_csv("file.csv", encoding="utf-8-sig")   # handles BOM
# df = pd.read_csv("file.csv", encoding="latin-1")   # fallback for legacy exports

# Vectorized string ops are much faster than .apply() with a lambda
df["email_domain"] = df["email"].str.split("@").str[1]
```

**Pitfalls:**
- Using `.apply(lambda x: x.strip())` instead of vectorized `.str.strip()` — much slower on large data.
- Lowercasing before extracting information that's case-sensitive (e.g., ticker symbols, IDs).
- Regex that's "close enough" but silently mismatches edge cases (e.g., international phone formats) — test regex against a sample of unique values first: `df['phone'].unique()`.
- Not handling `NaN` in string operations — `.str` methods propagate `NaN` correctly, but custom `.apply()` lambdas often crash on `None`.

### Handling Inconsistent Categorical Labels

**Concept:** the same category is often recorded multiple ways: `"USA"`, `"U.S.A."`, `"United States"`, `"us"` all mean the same country. Left unresolved, this fragments a categorical variable into dozens of near-duplicate levels, destroying groupby aggregates and one-hot encodings.

```python
# Discover the problem first
df["country"].value_counts()          # eyeball for obvious duplicates/typos
df["country"].nunique()

# Manual mapping (most reliable for small, known category sets)
country_map = {
    "usa": "United States", "u.s.a.": "United States", "us": "United States",
    "uk": "United Kingdom", "england": "United Kingdom",
}
df["country_clean"] = df["country"].str.lower().str.strip().map(country_map).fillna(df["country"])

# Fuzzy grouping for larger unknown label sets
from rapidfuzz import process, fuzz
unique_vals = df["country"].dropna().unique()
canonical = ["United States", "United Kingdom", "Canada", "Germany"]
def to_canonical(val, choices=canonical, threshold=80):
    match, score, _ = process.extractOne(val, choices, scorer=fuzz.WRatio)
    return match if score >= threshold else val
df["country_clean"] = df["country"].apply(to_canonical)

# Consolidating rare categories into "Other" (reduces overfitting/noise in models)
freq = df["category"].value_counts(normalize=True)
rare = freq[freq < 0.01].index
df["category_grouped"] = df["category"].where(~df["category"].isin(rare), "Other")
```

**Pitfalls:**
- Silent information loss when auto-fuzzy-matching merges two genuinely different categories that happen to be textually similar (e.g., "Georgia" the US state vs. "Georgia" the country).
- Not versioning/logging the mapping dictionary — six months later no one remembers why `"NY"` became `"New York"` but `"NJ"` stayed `"NJ"`.
- Forgetting that new unseen categories will appear in production — build the pipeline to route unknown categories to a defined `"Other"`/`"Unknown"` bucket rather than crashing or silently one-hot-encoding a new column at inference time.

### Data Validation and Schema Enforcement

**Concept:** validation catches problems *before* they propagate — expected types, value ranges, referential integrity, null constraints — ideally as an automated, versioned contract rather than ad hoc `assert` statements sprinkled through notebooks.

```python
# Lightweight: pandas + assertions (good for scripts/notebooks)
assert df["age"].between(0, 120).all(), "Age out of plausible range"
assert df["order_id"].is_unique, "Duplicate order IDs found"
assert not df["email"].isna().any(), "Missing emails"

# Pandera: declarative schema validation, integrates with pandas
import pandera as pa
from pandera import Column, DataFrameSchema, Check

schema = DataFrameSchema({
    "age": Column(int, Check.in_range(0, 120), nullable=False),
    "email": Column(str, Check.str_matches(r"^[^@]+@[^@]+\.[^@]+$")),
    "order_amount": Column(float, Check.greater_than(0)),
    "region": Column(str, Check.isin(["North", "South", "East", "West"])),
}, strict=True, coerce=True)

validated_df = schema.validate(df, lazy=True)   # lazy=True collects ALL failures, not just the first

# Great Expectations: production-grade, generates data docs, integrates with pipelines/CI
import great_expectations as gx
context = gx.get_context()
validator = context.sources.pandas_default.read_dataframe(df)
validator.expect_column_values_to_not_be_null("customer_id")
validator.expect_column_values_to_be_between("age", min_value=0, max_value=120)
validator.expect_column_values_to_be_in_set("status", ["active", "inactive", "pending"])
results = validator.validate()

# JSON Schema for semi-structured/API data
import jsonschema
person_schema = {
    "type": "object",
    "properties": {"age": {"type": "integer", "minimum": 0}, "name": {"type": "string"}},
    "required": ["age", "name"],
}
jsonschema.validate(instance={"age": 30, "name": "Ada"}, schema=person_schema)
```

**What to validate, at minimum:**

| Category | Examples |
|---|---|
| **Type** | column is numeric/string/datetime as expected |
| **Range/domain** | age ∈ [0, 120], probability ∈ [0, 1] |
| **Uniqueness** | primary keys have no duplicates |
| **Null constraints** | required fields are never null |
| **Referential integrity** | foreign keys exist in the parent table |
| **Set membership** | categorical values fall in an allowed list |
| **Cross-field logic** | `end_date >= start_date`, `discount_price <= list_price` |
| **Distributional drift** | column mean/std/null-rate hasn't shifted abnormally vs. a historical baseline (critical in production ML monitoring) |

**Pitfalls:**
- Validating only at training time, never at inference/production time — schema drift (a new category, a renamed column, a changed unit) is one of the most common causes of silent production model failures.
- Failing the whole pipeline on the first validation error instead of collecting all failures (`lazy=True` in Pandera) — makes debugging painfully slow and iterative.
- Treating validation as a one-time notebook check instead of a versioned, automated, CI-integrated contract.
- Not distinguishing between a hard failure (reject the batch) and a soft warning (log and proceed) — not every validation issue should halt a pipeline.

### Interview Questions

**Q1: What's the difference between MCAR, MAR, and MNAR, and why does it matter for how you handle missing data?**
A: MCAR means missingness is unrelated to any variable (observed or not) — safe to delete or simply impute. MAR means missingness depends on other *observed* variables (e.g., missing income correlates with age) — you can get unbiased results by imputing conditional on those observed variables (regression, KNN, MICE). MNAR means missingness depends on the unobserved value itself (e.g., high earners refuse to disclose income) — no standard imputation is unbiased here; it requires modeling the missingness mechanism explicitly, sensitivity analysis, or accepting the bias and documenting it. Ignoring the mechanism and always doing listwise deletion or mean imputation can quietly bias every downstream statistic and model.

**Q2: How would you decide whether to drop a column vs. impute its missing values?**
A: I'd weigh (a) the percentage missing — above roughly 60–70% with no strong signal, dropping is often better than imputing noise; (b) the missingness mechanism — MCAR with low % is safe to impute simply, MAR/MNAR needs more careful methods or a missingness indicator; (c) the column's importance to the business question or target correlation — a highly predictive column is worth the extra imputation effort even at high missing %; (d) whether the model I'm using handles missing values natively (e.g., XGBoost/LightGBM split on missingness) making imputation unnecessary.

**Q3: Walk me through how you'd detect outliers in a dataset with multiple correlated numeric features.**
A: Univariate methods (z-score, IQR) applied column-by-column can miss multivariate outliers — a point can be normal on every single feature but abnormal in combination (e.g., a person 190cm tall weighing 40kg). I'd scale features, then use Mahalanobis distance (accounts for covariance) or an Isolation Forest / Local Outlier Factor for a non-parametric, high-dimensional-friendly approach. I'd visualize with a pair plot colored by the anomaly flag to sanity check the multivariate outliers make sense, then investigate the top-scoring points individually before deciding to cap, transform, or remove.

**Q4: When would you use z-score vs. IQR for outlier detection?**
A: Z-score assumes roughly normal data and is itself distorted by extreme outliers (since it uses mean/std) — best for large, roughly Gaussian, outlier-light data. IQR is distribution-free and robust (based on quartiles, not affected by extreme tails), making it the safer default for skewed data or when you don't know the distribution shape; it's also the standard behind box plots, so it's easy to communicate visually.

**Q5: You find a $10 million transaction in a dataset of otherwise $10–$500 transactions. What do you do?**
A: I would not delete it reflexively. First, investigate: check if it's a data-entry error (extra zero), a unit mismatch (cents vs. dollars), a legitimate but rare bulk/wholesale order, or test/dummy data. I'd check for supporting evidence — is there a matching invoice, does the customer segment support a large order, are there other transactions of similar magnitude from the same account. If confirmed as an error, correct or remove it and document why. If it's genuine, I'd keep it, potentially cap its influence for models sensitive to scale (e.g., linear regression) via winsorizing/log-transform, or model it explicitly as a distinct segment (wholesale vs. retail) rather than discard real signal.

**Q6: How does mean imputation distort a dataset, and what would you use instead?**
A: Mean imputation shrinks the variance of the imputed column (every filled value collapses toward the same central point), attenuates correlations between that column and others (since imputed values carry no relationship to the other features, they add uncorrelated noise), and can distort the distribution shape, especially under skew. For distributions with meaningful relationships, KNN imputation or MICE (iterative regression-based imputation across all columns) preserves inter-variable structure far better, at the cost of more computation and more careful cross-validation to avoid leakage.

**Q7: What's the difference between the IQR outlier rule and Isolation Forest, and when would you choose each?**
A: The IQR rule is univariate, deterministic, transparent, and cheap — great for a quick first pass on individual features and for explaining results to non-technical stakeholders. Isolation Forest is multivariate, model-based, and captures interaction effects across many features simultaneously, at the cost of being less interpretable and needing a `contamination` hyperparameter tuned or estimated. I'd use IQR for exploratory single-column screening and Isolation Forest (or LOF/DBSCAN) when outliers can only be identified by combinations of features, such as fraud detection across many transaction attributes.

**Q8: How do you detect near-duplicate records that aren't byte-identical (e.g., "Jon Smith" vs "John Smith" at the same address)?**
A: Exact `.duplicated()` won't catch this. I'd use fuzzy string matching (Levenshtein/Jaro-Winkler similarity via `rapidfuzz` or `fuzzywuzzy`) on the name field combined with exact or near-exact matches on other fields (address, DOB) to build a similarity score, then threshold it. For larger datasets I'd use blocking (e.g., group by ZIP code first) via a record-linkage library like `recordlinkage` or `dedupe` to avoid an O(n²) comparison, then review borderline matches manually rather than fully automating merges above a fixed threshold, since false merges of distinct people are costly.

**Q9: What is schema drift, and how would you guard against it in a production ML pipeline?**
A: Schema drift is when the structure or statistical properties of incoming data changes from what the pipeline/model was built for — a new categorical value appears, a column is renamed or dropped upstream, a numeric column's units change, or a null rate spikes. I'd guard against it with an explicit, versioned schema (e.g., Pandera or Great Expectations) validated at ingestion time in production, not just during training; alerting on validation failures; routing unknown categorical levels to an explicit "Unknown" bucket instead of crashing or silently mis-encoding; and monitoring feature distributions over time (e.g., population stability index) to catch drift even when it doesn't technically break the schema.

**Q10: Why is it dangerous to fit an imputer, scaler, or outlier threshold on the entire dataset before train/test split?**
A: It's data leakage — the training process gains information about the test set's distribution (its mean, its quantiles, its outlier boundary) that it wouldn't have in a genuine production setting, making validation metrics overly optimistic. The fix is to `fit` any of these on the training fold only (inside a `Pipeline`, and inside each cross-validation fold if doing CV) and merely `transform` the validation/test data with those already-fitted parameters.

**Q11: A categorical column has 500 unique string values that should really be about 20 categories due to typos and formatting inconsistencies. How would you approach cleaning it at scale?**
A: First, profile with `.value_counts()` to see the distribution — usually a small number of "correct" canonical labels dominate and a long tail of near-duplicates/typos exists. I'd normalize case/whitespace/punctuation first (often collapses many duplicates immediately), then use fuzzy matching (`rapidfuzz.process.extractOne`) against a canonical list to map remaining variants, reviewing matches below a confidence threshold manually rather than auto-accepting everything. I'd persist the final mapping as a versioned lookup table/config so it's reproducible and auditable, and route any future unseen value to an explicit "Other"/"Needs Review" bucket instead of silently creating a new category.

**Q12: How would you validate incoming data before it reaches a trained model in production?**
A: I would enforce a schema contract (types, ranges, allowed categories, null constraints, referential integrity) using a tool like Pandera or Great Expectations at the ingestion boundary, failing loudly (or routing to a dead-letter queue) on hard violations like missing required fields or impossible values, while logging softer anomalies like a new categorical value. I'd also monitor input feature distributions against a training-time baseline (data drift detection) since a batch can pass schema validation yet still represent a meaningful distributional shift the model wasn't trained for.

**Q13: What's the risk of using `errors="coerce"` when converting types with pandas, and how do you mitigate it?**
A: `errors="coerce"` silently turns unparseable values into `NaN` instead of raising — convenient for robustness, but dangerous because it can silently null out large portions of a column if there's a systemic formatting problem (e.g., every date is in an unexpected format) and you won't notice unless you explicitly check the resulting null count. I mitigate this by comparing `isna().sum()` before and after the coercion, and treating any large jump as a signal to investigate the source format rather than proceeding.

**Q14: Give an example where treating a duplicate row as "dirty data" would actually be a mistake.**
A: In an `orders` or `transactions` table, the same `customer_id` appearing multiple times is expected and correct — that's the normal grain of the data (one customer, many orders). Blindly deduplicating on `customer_id` would destroy legitimate repeat-purchase history. The correct duplicate check must respect the table's actual grain — e.g., dedupe on the full natural key (`order_id`) or on `(customer_id, order_date, product_id)`, not on a single non-unique business entity column.

**Q15: How would you design an automated pipeline stage that both imputes missing values and remains robust to unseen missingness patterns at inference time?**
A: I'd wrap imputation in a `sklearn.Pipeline`/`ColumnTransformer` so the same fitted transformer (mean/median/KNN/IterativeImputer parameters learned on training data) is applied identically at inference, avoiding train/serve skew. I'd add missingness indicator columns for any feature where missingness has historically correlated with the target, persist the fitted pipeline (e.g., via `joblib`) alongside the model artifact, and add a runtime check that alerts if the *rate* of missingness at inference deviates sharply from the training-time rate (a sign of upstream pipeline breakage rather than "normal" MCAR missingness).

---

## Exploratory Data Analysis (EDA)

### A Structured Framework for Exploring an Unfamiliar Dataset

**Why this matters:** "Here's a dataset you've never seen — walk me through your first 30 minutes" is one of the most common open-ended DS interview prompts, precisely because it can't be crammed for with syntax alone — it tests whether you follow a repeatable process or just poke around randomly. Interviewers are grading the *order of operations* and the questions you ask, not whether you land on a single "correct" insight.

**A repeatable five-phase process:**

1. **Orient before touching statistics.** What does one row represent (the grain)? What is the target/label, and over what time horizon? How was the data collected, and does that population match the population you actually care about (see selection bias below)? Answering these from a data dictionary or a five-minute conversation with a stakeholder saves hours of misdirected analysis.
2. **Structural sanity pass.** Shape, dtypes, memory footprint, missingness map, duplicate/grain check, obvious constant or all-null columns.
3. **Univariate pass, target first.** Distribution and class balance of the target, then the handful of features you'd guess matter most from domain knowledge — not every column, yet.
4. **Bivariate pass toward the target.** Correlation heatmap / cross-tabs / group-by comparisons against the target, specifically looking for leakage (a feature that's suspiciously perfectly predictive is often computed *after* the target is known) and for the strongest, most explainable relationships.
5. **Write down data-quality caveats and hypotheses**, not just findings — what you don't trust yet is as valuable to report as what you do.

```python
# A concrete "first 10 minutes" script
print(df.shape)
display(df.head())
display(df.dtypes)
display(df.describe(include="all").T)
print(df.isna().mean().sort_values(ascending=False).head(10))
print(df.duplicated().sum())
if "target" in df.columns:
    display(df["target"].value_counts(normalize=True))
```

**Pitfalls:**
- Jumping straight to a correlation heatmap or a model before confirming the table's grain — a "duplicate" or "outlier" flag is meaningless until you know what one row is supposed to represent.
- Treating every column as equally worth deep exploration in a time-boxed setting — triage using domain knowledge and a quick univariate pass first.
- Never asking how the data was collected or filtered before it reached you — see selection/survivorship bias below; the answer to "is this dataset representative of what I actually care about" is rarely obvious from the columns alone.
- Reporting only positive findings — an interviewer is specifically listening for "here's what I don't trust yet" as a sign of calibrated judgment.

### Univariate Analysis

**Concept:** univariate analysis examines one variable in isolation — its central tendency, spread, shape, and outliers — and is always the first step before looking at relationships between variables.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Summary statistics
df["price"].describe()               # count, mean, std, min, 25/50/75%, max
df.describe(include="all")           # full dataframe, mixed types

# Skewness and kurtosis
skewness = df["price"].skew()        # 0 = symmetric; >0 right-skewed; <0 left-skewed
kurt = df["price"].kurt()            # 0 = normal-like (excess kurtosis); >0 heavy tails (leptokurtic); <0 light tails (platykurtic)

# Normality test
stat, p_value = stats.shapiro(df["price"].sample(min(len(df), 5000)))  # Shapiro sensitive to n; sample if large
print("Normal-ish" if p_value > 0.05 else "Not normal")

# Visualizations
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df["price"], kde=True, ax=axes[0], bins=30)
sns.boxplot(y=df["price"], ax=axes[1])
stats.probplot(df["price"].dropna(), dist="norm", plot=axes[2])  # Q-Q plot
plt.tight_layout()

# Categorical univariate
df["region"].value_counts()
df["region"].value_counts(normalize=True).plot(kind="bar")
```

**Interpreting skewness and kurtosis:**

| Skewness | Meaning | Typical fix if needed for modeling |
|---|---|---|
| ≈ 0 | Symmetric | None needed |
| > 1 (or < -1) | Strongly skewed | Log/sqrt/Box-Cox transform |
| 0.5 to 1 | Moderately skewed | Consider transform depending on model |

| Kurtosis (excess) | Meaning |
|---|---|
| ≈ 0 | Normal-like tails |
| > 0 | Heavy tails / more outliers than normal (leptokurtic) |
| < 0 | Light tails, more uniform-like (platykurtic) |

**Decision guidance:**
- Use a **histogram + KDE** to see shape, modality (uni/bi/multi-modal), and skew at a glance.
- Use a **box plot** to quickly communicate median, IQR, and outliers, especially across many groups side by side.
- Use a **Q-Q plot** when you specifically need to assess normality (e.g., before applying a method that assumes it, like OLS inference or z-score outlier detection).
- Use **ECDF (empirical CDF)** when you need exact percentile reads without binning artifacts — `sns.ecdfplot(df['price'])`.

**Pitfalls:**
- Histogram bin count changes the *apparent* shape dramatically — always try a few bin widths (or use `bins="auto"`/Freedman-Diaconis rule) before concluding a distribution is unimodal or multimodal.
- Reporting only the mean when data is skewed — always pair mean with median so a reader can gauge skew; the mean alone on income data is highly misleading.
- Assuming normality without checking — many "obviously normal-looking" histograms fail Shapiro-Wilk at large `n` because the test becomes extremely sensitive to n; in practice, visual inspection (Q-Q plot) and skew/kurtosis values matter more than a strict test p-value at scale.

### Bivariate and Multivariate Analysis

**Concept:** bivariate/multivariate analysis studies relationships *between* variables — critical for feature selection, spotting confounders, and understanding what actually drives a target before any modeling begins.

**Correlation coefficients — decision table:**

| Method | Measures | Assumptions | Use when |
|---|---|---|---|
| **Pearson** | Linear relationship strength | Continuous, roughly linear, sensitive to outliers | Two continuous, roughly normal/linear variables |
| **Spearman** | Monotonic relationship strength (rank-based) | No linearity assumption, robust to outliers | Non-linear but monotonic relationships, ordinal data, outlier-prone data |
| **Kendall's Tau** | Concordance of rank pairs | Similar to Spearman, more robust for small samples/ties | Small sample sizes, many tied ranks |

```python
# Correlation matrix
corr_pearson = df[num_cols].corr(method="pearson")
corr_spearman = df[num_cols].corr(method="spearman")
corr_kendall = df[num_cols].corr(method="kendall")

# Heatmap visualization
plt.figure(figsize=(8, 6))
mask = np.triu(np.ones_like(corr_pearson, dtype=bool))   # show only lower triangle, avoid redundancy
sns.heatmap(corr_pearson, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)

# Scatter plot for a single pair, with regression line
sns.regplot(data=df, x="feature_a", y="feature_b", scatter_kws={"alpha": 0.4})

# Pair plot for many features at once
sns.pairplot(df[num_cols + ["target"]], hue="target", diag_kind="kde", corner=True)

# Cross-tabulation for two categoricals
ct = pd.crosstab(df["region"], df["churned"], normalize="index")  # row-wise %
ct_counts = pd.crosstab(df["region"], df["churned"])

# Chi-square test of independence for two categoricals
from scipy.stats import chi2_contingency
chi2, p, dof, expected = chi2_contingency(ct_counts)

# Categorical vs numeric: grouped summary + visual
df.groupby("region")["revenue"].describe()
sns.boxplot(data=df, x="region", y="revenue")
sns.violinplot(data=df, x="region", y="revenue")

# ANOVA: is the mean of a numeric variable different across categorical groups?
from scipy.stats import f_oneway
groups = [g["revenue"].values for _, g in df.groupby("region")]
f_stat, p_value = f_oneway(*groups)
```

**Decision guidance for bivariate exploration:**

| Variable types | Technique |
|---|---|
| Numeric vs. Numeric | Scatter plot, correlation coefficient (Pearson/Spearman), 2D hexbin/KDE for large n |
| Categorical vs. Categorical | Cross-tab / contingency table, chi-square test, stacked/grouped bar chart, Cramér's V for effect size |
| Categorical vs. Numeric | Box/violin plot by group, grouped `describe()`, ANOVA / t-test for significance |
| Many numeric variables at once | Correlation heatmap, pair plot, PCA biplot |

**Pitfalls:**
- Correlation ≠ causation — always state this explicitly in an interview when discussing a strong correlation.
- Using Pearson on non-linear (e.g., quadratic or exponential) relationships — Pearson can report near-zero correlation for a strong non-linear relationship (classic Anscombe's quartet lesson); always look at the scatter plot, not just the coefficient.
- Correlation heatmaps hide non-linear and higher-order interactions entirely — they are not a substitute for pair plots or modeling.
- Simpson's Paradox: an overall correlation can reverse sign when the data is split by a confounding subgroup — always check if a relationship holds within meaningful subgroups, not just in aggregate.
- Overplotting in scatter plots with large n — use `alpha` transparency, hexbin, or 2D KDE/density plots instead of a solid blob of points.

### Selection Bias, Survivorship Bias, and Spurious Correlations

**Concept:** the correlation-≠-causation warning already covered is necessary but not sufficient — a rigorous EDA also has to ask whether the *sample itself* is a biased window onto the population you actually care about, and whether a correlation found by scanning many columns is real or a statistical artifact of how many comparisons you ran.

**Survivorship bias** — drawing conclusions only from entities that "survived" some filtering process, while the ones that didn't survive are silently missing from the dataset and are often the more informative cases:
- Classic example: WWII researchers initially proposed reinforcing the areas of returning aircraft with the most bullet holes; Abraham Wald pointed out the planes hit in *other* areas never made it back to be studied — the undamaged-looking areas on survivors were the areas where damage was fatal.
- DS/analytics examples: "average tenure of current employees" ignores everyone who already left (survivors only); backtesting a trading strategy only on stocks/funds still listed today silently excludes the ones that went bankrupt or were delisted; churn analysis run only on currently active customers can't see the early-warning signals that preceded the churn of people who already left, unless historical (not just current) records are included.

**Selection bias** — the sample was assembled by the data-generating process (not by your later slicing) in a way that's systematically non-representative of the target population:
- Self-selected survey respondents (people with strong opinions respond more often).
- Berkson's paradox: studying only hospitalized patients can produce a spurious negative association between two diseases, because being admitted for at least one of them changes who ends up in the sampled population at all.
- Training data pulled only from customers who already converted, when the model needs to score both converters and non-converters.

**How to probe for both during EDA (not a statistical test — a set of questions):**
1. How exactly did a row end up in this dataset — what had to be true for it to be observed at all?
2. Does the dataset's population match the population the resulting decision will be applied to?
3. Are there entities you'd expect to see that are conspicuously absent (e.g., no bankrupt companies in a decade of "all companies" financial data)?
4. Plot cohort retention/attrition over time — a dataset that only ever grows (never loses rows for entities that should have dropped out) is a red flag for survivor-only data.

**Spurious correlation from multiple comparisons:** scanning a correlation heatmap or pairwise-testing dozens of columns against a target will surface "significant" relationships by chance alone — at α = 0.05, testing 100 independent pairs yields ~5 false positives on average even if nothing is truly related.

```python
# Demonstrating spurious correlation from pure chance
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
noise = pd.DataFrame(rng.normal(size=(200, 100)))   # 100 independent random columns
corr_with_target = noise.corrwith(noise[0]).drop(0)
n_spurious = (corr_with_target.abs() > 0.2).sum()    # several will look "correlated" by chance alone
```

Mitigations: correct for multiple comparisons (Bonferroni, Benjamini-Hochberg/FDR) before treating a scanned correlation as a real finding, require the relationship to hold on a held-out sample, and always demand a plausible causal or domain mechanism before acting on a correlation discovered by broad automated scanning rather than a specific hypothesis.

**Pitfalls:**
- Treating "we found a strong correlation while scanning 50 column pairs" as equivalent in strength to "we hypothesized this specific relationship and then tested it" — the former needs a multiple-comparisons correction, the latter doesn't.
- Not asking who or what is systematically missing from a dataset, only analyzing who/what is present.
- Assuming a training population automatically matches the production/scoring population — this is a selection-bias question, not just a modeling one, and belongs in EDA before any model is built.

### Detecting Relationships, Interactions, and Multicollinearity (VIF)

**Concept:** beyond pairwise correlation, real datasets have interaction effects (the effect of X on Y depends on the level of Z) and multicollinearity (features that are themselves highly correlated with each other), both of which distort model interpretation and, in the multicollinearity case, can destabilize coefficient estimates.

```python
# Detecting interaction effects visually
sns.lmplot(data=df, x="feature_a", y="target", hue="segment", height=5)  # different slopes per segment = interaction

# Numerically: interaction term significance in a regression
import statsmodels.formula.api as smf
model = smf.ols("target ~ feature_a + segment + feature_a:segment", data=df).fit()
print(model.summary())   # significant feature_a:segment coefficient = real interaction

# Multicollinearity: Variance Inflation Factor (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

X = df[num_cols].dropna()
X = (X - X.mean()) / X.std()   # standardizing helps interpretability, doesn't change VIF ranking
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_data.sort_values("VIF", ascending=False)
```

**Interpreting VIF:**

| VIF value | Interpretation |
|---|---|
| 1 | No correlation with other predictors |
| 1–5 | Moderate correlation, generally acceptable |
| 5–10 | High correlation, investigate/consider action |
| > 10 | Severe multicollinearity, action usually required |

**What to do about multicollinearity:**
- Drop one of the correlated variables (keep the more interpretable or more complete one).
- Combine correlated variables into a single index/composite feature.
- Use dimensionality reduction (PCA) to create orthogonal components.
- Use regularized regression (Ridge/Lasso/Elastic Net), which handles correlated predictors more gracefully than OLS.
- If the goal is *prediction* only (not coefficient interpretation), multicollinearity is far less damaging — tree-based models and pure prediction accuracy are largely unaffected; it mainly matters when you need to interpret individual coefficients.

**Pitfalls:**
- Computing VIF on features with missing values without handling them first (silently drops rows, changing the sample).
- Treating multicollinearity as always fatal — for pure prediction with tree ensembles, it usually isn't a problem at all; the concern is specifically for coefficient interpretability in linear models.
- Forgetting that VIF only detects *linear* multicollinearity between predictors — non-linear dependencies between features won't show up numerically but can still cause instability in some models.
- Missing genuine interaction effects because you only looked at a correlation heatmap of main effects — interactions require explicit modeling (interaction terms, tree-based feature-importance interaction detection like SHAP interaction values, or stratified visualizations).

### Time Series EDA Basics

**Concept:** time series data has structure that generic EDA misses — order matters, past values inform future values, and most classical models require (or benefit from) stationarity.

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = df.set_index("date").sort_index()

# 1. Visual: trend + seasonality at a glance
df["sales"].plot(figsize=(12, 4), title="Raw series")

# 2. Decomposition: trend, seasonal, residual components
decomposition = seasonal_decompose(df["sales"], model="additive", period=12)  # or model="multiplicative"
decomposition.plot()

# 3. Rolling statistics (visual stationarity check)
df["rolling_mean"] = df["sales"].rolling(window=12).mean()
df["rolling_std"] = df["sales"].rolling(window=12).std()
fig, ax = plt.subplots(figsize=(12, 4))
df["sales"].plot(ax=ax, label="Original")
df["rolling_mean"].plot(ax=ax, label="Rolling Mean")
df["rolling_std"].plot(ax=ax, label="Rolling Std")
plt.legend()

# 4. Augmented Dickey-Fuller test for stationarity
adf_result = adfuller(df["sales"].dropna())
print(f"ADF Statistic: {adf_result[0]:.3f}, p-value: {adf_result[1]:.3f}")
# H0: series is non-stationary. p < 0.05 -> reject H0 -> series IS stationary

# 5. KPSS test (complementary — opposite null hypothesis, good to run both)
kpss_result = kpss(df["sales"].dropna(), regression="c")
print(f"KPSS Statistic: {kpss_result[0]:.3f}, p-value: {kpss_result[1]:.3f}")
# H0: series IS stationary. p < 0.05 -> reject H0 -> series is NOT stationary

# 6. Differencing to achieve stationarity
df["sales_diff"] = df["sales"].diff().dropna()
df["sales_diff2"] = df["sales"].diff().diff().dropna()   # second-order if needed
df["sales_log_diff"] = np.log(df["sales"]).diff().dropna()  # stabilizes variance + trend

# 7. ACF / PACF plots — identify AR/MA order, confirm remaining autocorrelation after differencing
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(df["sales_diff"].dropna(), lags=36, ax=axes[0])
plot_pacf(df["sales_diff"].dropna(), lags=36, ax=axes[1])
```

**ADF vs. KPSS — use both, because their null hypotheses are opposite:**

| Test | Null hypothesis (H0) | If p < 0.05 |
|---|---|---|
| **ADF** (Augmented Dickey-Fuller) | Series is non-stationary (has a unit root) | Reject H0 → series is stationary |
| **KPSS** | Series is stationary | Reject H0 → series is NOT stationary |

Running both avoids being misled by a single test's assumptions — if ADF says stationary but KPSS also rejects stationarity, the series may be "trend stationary" needing detrending rather than differencing.

**Reading ACF/PACF for model order (classic Box-Jenkins heuristics):**

| Pattern | Suggests |
|---|---|
| ACF tails off gradually, PACF cuts off sharply after lag *p* | AR(p) process |
| ACF cuts off sharply after lag *q*, PACF tails off gradually | MA(q) process |
| Both tail off gradually | ARMA(p, q) process — mixed |
| Strong spike at seasonal lag (e.g., lag 12 for monthly data) | Seasonal component present; consider SARIMA or seasonal differencing |

**Pitfalls:**
- Forgetting to sort by date and set a proper `DatetimeIndex` before rolling/resampling operations — silent misalignment.
- Testing stationarity on a series that still has strong seasonality — seasonal patterns can make ADF/KPSS results misleading; deseasonalize first, or use seasonal differencing (`.diff(12)`).
- Confusing "no visible trend line" with "stationary" — variance can still change over time (heteroskedasticity) even with a flat mean; check rolling std too, not just rolling mean.
- Over-differencing — differencing more than necessary introduces artificial negative autocorrelation and throws away information; stop as soon as ADF/KPSS agree on stationarity.
- Ignoring irregular time gaps/missing timestamps in the index before computing lag-based features — resample to a consistent frequency (`.asfreq()`, `.resample()`) first.

### Handling Large Datasets: Sampling and Out-of-Core EDA

**Concept:** once a dataset no longer fits comfortably in memory (or a full profiling pass would take hours), the EDA workflow has to change — from "load everything into a pandas DataFrame" to a mix of smart sampling, chunked/streaming reads, and out-of-core or distributed tooling. The core judgment interviewers probe: do you know *when* pandas-as-usual breaks down, and what you'd reach for instead?

**Sampling strategies:**

| Strategy | How it works | Use when |
|---|---|---|
| **Simple random sampling** | Uniform random subset of rows | Data is IID and no rare subgroup matters disproportionately |
| **Stratified sampling** | Sample within each group (e.g., class label, region) to preserve group proportions | Class imbalance exists and the rare class must remain represented (e.g., fraud, churn) |
| **Systematic sampling** | Take every *k*-th row | Data has no periodic structure aligned with *k*; fast, low-overhead approximation of random sampling |
| **Cluster sampling** | Sample whole natural groups (e.g., all rows for a sampled subset of customers), not individual rows | Rows within a group are correlated (e.g., repeated measures per customer) — sampling individual rows would leak information across train/test or double-count structure |
| **Reservoir sampling** | Maintain a fixed-size uniform random sample while streaming through data that can't be held in memory or whose total size is unknown upfront | Streaming/log data, single-pass constraints |

```python
# Simple random sample for a fast first-pass EDA
sample_df = df.sample(frac=0.05, random_state=42)

# Stratified sample that preserves rare-class proportions
sample_df = (
    df.groupby("target", group_keys=False)
      .apply(lambda g: g.sample(frac=0.05, random_state=42))
)

# Chunked reading for files too big to load at once
chunk_iter = pd.read_csv("huge_file.csv", chunksize=500_000)
missing_counts = None
for chunk in chunk_iter:
    counts = chunk.isna().sum()
    missing_counts = counts if missing_counts is None else missing_counts.add(counts, fill_value=0)

# Out-of-core / lazy DataFrames that scale beyond RAM
import dask.dataframe as dd
ddf = dd.read_csv("huge_file*.csv")          # lazy, partitioned, pandas-like API
summary = ddf.describe().compute()           # only materializes the result

import polars as pl
lazy = pl.scan_csv("huge_file.csv")           # lazy query plan, executes only on .collect()
result = lazy.group_by("region").agg(pl.col("revenue").mean()).collect()
```

**When a full automated profiling report doesn't scale:**
- Run `ydata-profiling`'s `minimal=True` mode (skips the most expensive pairwise interactions/correlations) or profile a representative stratified sample instead of the full table.
- For pairwise correlation/pair plots specifically, cost grows roughly quadratically with column count and linearly with row count — sample rows before computing, not after.
- For quick cardinality/quantile estimates at true big-data scale (billions of rows, distributed), exact computation is often replaced with approximate algorithms (e.g., HyperLogLog for distinct counts, t-digest/GK-sketch for quantiles) inside the data warehouse (`APPROX_COUNT_DISTINCT`, `APPROX_QUANTILES` in BigQuery/Snowflake) rather than pulling everything into Python at all.

**Pitfalls:**
- Random row sampling when rows aren't independent — sampling individual transaction rows instead of whole customers can put part of a customer's history in a "train" sample and part in a "test" sample, leaking information.
- Sampling uniformly when the thing you care about is rare — a 1% random sample of a dataset with 0.1% fraud can leave almost no positive examples to look at.
- Computing summary statistics chunk-by-chunk naively (e.g., averaging per-chunk means) — this is only correct if every chunk has equal size; otherwise weight by chunk row count, or use a running/Welford-style algorithm for an exact one-pass mean/variance.
- Assuming a sample small enough to profile quickly is still large enough for correlation or rare-category estimates to be stable — correlation coefficients and rare-level frequencies are noisier on small samples; sanity-check against the full data's basic aggregates (row count, overall mean) before trusting sample-based conclusions.
- Defaulting to Spark/Dask for a dataset that actually fits in memory on a bigger machine — distributed tooling adds real operational complexity; "buy more RAM" or switching to `polars`/downcasting dtypes is often the simpler right answer before reaching for a cluster.

### Data Profiling and Automated EDA Tools

**Concept:** automated profiling tools generate a full first-pass EDA report (types, missingness, distributions, correlations, duplicates, warnings) in one line of code — invaluable for speed, but not a substitute for domain-driven, hypothesis-guided manual EDA.

```python
# ydata-profiling (formerly pandas-profiling)
from ydata_profiling import ProfileReport
profile = ProfileReport(df, title="Dataset EDA Report", explorative=True)
profile.to_file("eda_report.html")
# profile.to_notebook_iframe()  # inline in Jupyter

# Sweetviz — great for comparing train vs. test distributions
import sweetviz as sv
report = sv.compare([train_df, "Train"], [test_df, "Test"])
report.show_html("comparison_report.html")

# D-Tale — interactive, spreadsheet-like exploration in the browser
import dtale
dtale.show(df)

# Quick built-in pandas alternative for a fast manual pass
def quick_profile(df):
    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
    })
    return summary.sort_values("pct_missing", ascending=False)

quick_profile(df)
```

**Decision guidance — automated vs. manual EDA:**

| Use automated tools (ydata-profiling, Sweetviz, D-Tale) for | Use manual, hypothesis-driven EDA for |
|---|---|
| Fast first-pass overview of a new/unfamiliar dataset | Answering a specific business question |
| Catching obvious data quality issues (high cardinality, high correlation, zeros/constant columns, skew warnings) | Investigating a suspected relationship or anomaly in depth |
| Comparing train/test or before/after distributions quickly | Communicating a specific insight with a tailored, clean chart |
| Documentation/handoff artifacts for stakeholders | Feature engineering decisions requiring domain judgment |

**Pitfalls:**
- Treating an auto-generated report as "the EDA" and skipping manual investigation — automated tools flag *statistical* patterns, not business meaning; they won't tell you two columns are duplicated for a legitimate reason or that an "outlier" is your most important customer segment.
- Running full profiling reports on very large datasets without sampling first — correlation/pairwise computations can be slow or memory-heavy at scale; profile a representative sample.
- Blindly trusting "high correlation" or "high cardinality" warnings without context — a high-cardinality ID column is expected to be flagged and is not a data quality problem.

### Interview Questions

**Q1: What is the difference between Pearson, Spearman, and Kendall correlation, and when would you use each?**
A: Pearson measures the strength of a *linear* relationship and assumes continuous, roughly normally distributed, outlier-light data. Spearman measures the strength of a *monotonic* relationship by correlating ranks rather than raw values, so it captures non-linear-but-monotonic relationships and is robust to outliers; it also works for ordinal data. Kendall's Tau is similar to Spearman (rank-based) but based on concordant/discordant pairs, and is generally preferred for small samples or data with many tied ranks, where it's statistically more robust than Spearman. In practice: start with Pearson for a quick linear read, switch to Spearman if the scatter plot shows a non-linear monotonic pattern or heavy outliers are present.

**Q2: Two variables have a Pearson correlation of 0.02. Does that mean they're unrelated?**
A: Not necessarily — Pearson only captures *linear* relationships. Two variables can have a strong non-linear relationship (e.g., U-shaped, quadratic) and still show near-zero Pearson correlation. This is the classic lesson from Anscombe's Quartet: always visualize the scatter plot alongside any correlation coefficient before concluding "no relationship." I would also check Spearman correlation and consider mutual information, which captures more general (not just monotonic) dependence.

**Q3: What is Simpson's Paradox, and how would you detect it during EDA?**
A: Simpson's Paradox is when a trend that appears in aggregated data reverses or disappears when the data is split by a confounding subgroup — e.g., a drug appears less effective overall but is actually more effective within every individual age group, because age is unevenly distributed between treatment and control. I'd detect it by always checking whether an aggregate relationship holds within meaningful subgroups (stratified analysis, `groupby` + visualize by segment, or an interaction term in a regression) rather than trusting a single pooled correlation or mean comparison.

**Q4: How do you check whether a time series is stationary, and why does it matter?**
A: Visually, I'd plot the raw series plus rolling mean/std over a window to see if the mean and variance look stable over time. Formally, I'd run the Augmented Dickey-Fuller test (null hypothesis: non-stationary) and the KPSS test (null hypothesis: stationary) together, since they have opposite nulls and agreement between them gives more confidence. It matters because most classical time series models (ARIMA and its relatives) assume stationarity — a non-stationary series (with trend or changing variance) will produce unreliable, spurious-looking model fits and forecasts unless it's differenced, detrended, or log-transformed first.

**Q5: What do ACF and PACF plots tell you, and how do you use them to choose an ARIMA order?**
A: The ACF (autocorrelation function) shows the correlation between the series and its own lagged values at each lag, including indirect effects passed through intermediate lags. The PACF (partial autocorrelation) shows the correlation at each lag after removing the effect of all shorter lags. Classic Box-Jenkins heuristics: if the ACF tails off gradually while PACF cuts off sharply after lag *p*, it suggests an AR(p) process; if the reverse (ACF cuts off, PACF tails off), it suggests an MA(q) process; if both tail off, an ARMA(p,q) mix. A strong spike at a seasonal lag (e.g., lag 12 in monthly data) signals seasonal structure. In practice, I'd use these as a starting hypothesis and confirm/tune the final order with information criteria (AIC/BIC) via a grid search (e.g., `auto_arima`).

**Q6: What is VIF, and what would you do if a feature has a VIF of 25?**
A: VIF (Variance Inflation Factor) quantifies how much a predictor's variance is inflated due to linear correlation with the other predictors — a VIF of 25 means severe multicollinearity, and that feature's regression coefficient is likely unstable and hard to interpret (small data changes could flip its sign). My response depends on the goal: if I only need predictive accuracy and I'm using a tree-based or regularized model, I might not act at all since those handle correlated features more gracefully. If I need interpretable linear coefficients, I'd drop one of the highly correlated features (keeping the more complete/interpretable one), combine them into a composite feature, or move to a regularized model (Ridge/Lasso) or PCA to decorrelate.

**Q7: How would you detect an interaction effect between two features during EDA, before building any model?**
A: I'd visualize the relationship between a feature and the target separately for different levels/segments of a second feature — e.g., `sns.lmplot` with different regression lines per group (hue), and check whether the slopes differ meaningfully across groups; parallel slopes suggest no interaction, diverging slopes suggest one. Numerically, I could fit a simple model with an explicit interaction term (`feature_a * feature_b`) and check if that term is statistically significant, or use tree-based feature importance / SHAP interaction values, which naturally capture non-linear interactions without manually specifying them.

**Q8: You're given a completely unfamiliar dataset and 30 minutes to explore it before an interview panel asks you what you found. What's your process?**
A: First, structural overview: shape, dtypes, `head()`, `describe()`, missingness map. Second, univariate pass on the target variable and key features: distributions, skew, obvious outliers. Third, an automated profiling report (ydata-profiling) running in the background while I do targeted checks manually, since it's fast and flags things like high cardinality, duplicated columns, or heavy skew I might otherwise miss. Fourth, bivariate: correlation heatmap for numeric features, cross-tabs for categoricals, and a few scatter/box plots against the target to look for the most promising relationships. Throughout, I'd note data quality concerns (missingness patterns, suspicious duplicates, inconsistent categories) since those are usually as important to report as the "interesting" statistical findings.

**Q9: What's the difference between a correlation heatmap and a pair plot, and when would you use one over the other?**
A: A correlation heatmap summarizes pairwise *linear* correlation strength for every pair of numeric variables in a single compact grid — fast to scan for the strongest linear relationships across many variables, but it hides shape (non-linearity, clusters, outliers) and only works for numeric data. A pair plot shows the actual scatter (or KDE) for every pair, plus univariate distributions on the diagonal, at the cost of taking much more screen space and computation time as the number of variables grows — it reveals non-linear patterns, multimodality, and outliers a heatmap can't show. I'd use a heatmap first to narrow down to the most promising ~5–8 variables, then a pair plot (optionally colored by target/segment) to actually inspect those relationships visually.

**Q10: How would you profile and compare a training set and a test/production set to check for distribution drift before deploying a model?**
A: I'd compare summary statistics (mean, std, quantiles, null rate) per feature across the two sets, and visualize overlaid histograms/KDEs or use a tool like Sweetviz's comparison report for a fast pass. For a more rigorous statistical check, I'd run a two-sample test per feature — Kolmogorov-Smirnov test for continuous features, chi-square for categorical — and/or compute the Population Stability Index (PSI), a standard drift metric in industry, flagging any feature above a chosen PSI threshold (commonly 0.1–0.25) for investigation before trusting the model's performance on the new data.

**Q11: What are skewness and kurtosis, and how would they influence your choice of a statistical test or transformation?**
A: Skewness measures asymmetry (positive = right tail longer, negative = left tail longer); kurtosis measures tail heaviness relative to a normal distribution (positive excess kurtosis = heavier tails / more extreme values than normal). High skew violates the normality assumption behind many parametric tests and can make the mean a misleading summary statistic — I'd apply a log, square-root, or Box-Cox/Yeo-Johnson transform to reduce skew before using methods like Pearson correlation, z-score outlier detection, or linear regression that assume roughly symmetric, normal-like residuals. If skew/kurtosis remain high after transformation, I'd switch to non-parametric alternatives (Spearman correlation, rank-based tests, tree-based models) that don't carry the normality assumption.

**Q12: A stakeholder asks "why does the automated EDA report say two columns are 99% correlated but the business says they measure different things?" How do you respond?**
A: I'd first verify the correlation is real and not a computation artifact (check for a shared upstream source, a join bug that duplicated a column under two names, or a derived column that's an arithmetic function of another). If it's genuinely coincidental in this dataset's specific sample (e.g., both happen to trend with time in a short window), I'd flag that the strong correlation is sample-specific and may not generalize, and recommend not treating one as a proxy for the other outside this dataset. If it turns out one column actually *is* derived from the other (e.g., `total_price` and `unit_price × quantity`), that's a genuine multicollinearity concern for any linear model and one should likely be dropped or the redundancy should be documented explicitly.

**Q13: How would you explore relationships between three or more variables simultaneously, beyond pairwise correlation?**
A: Options include: a pair plot colored by a third categorical variable (`hue=`) to see how a bivariate relationship shifts across groups; a 3D scatter or bubble chart encoding a third numeric variable as point size/color; faceted small multiples (`FacetGrid`/`catplot`) splitting by a third or fourth categorical dimension; a correlation heatmap combined with hierarchical clustering (clustermap) to reveal groups of jointly-correlated variables; or dimensionality reduction (PCA/UMAP/t-SNE) to visualize high-dimensional structure in 2D, colored by an outcome of interest. The right choice depends on whether the third variable is categorical (facet/hue) or continuous (color gradient/size), and how many total variables need to be shown at once.

**Q14: Why might Little's MCAR test or standard EDA diagnostics give a false sense of confidence about a dataset's quality?**
A: These are statistical, population-level diagnostics — they can't detect systematic errors that look statistically "normal," such as a unit conversion applied consistently to an entire column, a timezone offset baked into every timestamp, a business rule change mid-period that shifts the definition of a category, or missingness that's MNAR in a way no observed-data test can distinguish from MAR. Statistical diagnostics are necessary but not sufficient — domain knowledge, checking against a known ground truth or a second independent data source, and asking the people who generated the data remain essential complements to automated EDA.

**Q15: How would you decide which of several highly correlated features to keep for a linear model, versus for a tree-based model?**
A: For a linear model where multicollinearity destabilizes coefficients, I'd typically keep the feature that's more complete (less missing), more directly interpretable to stakeholders, more likely to be available at inference time and less prone to future data quality issues, or the one with a clearer causal story — and consider combining the correlated set into a single composite/index feature or using PCA/Ridge regression instead of manually picking a winner. For a tree-based model, multicollinearity among features barely affects predictive performance (the tree splits on whichever correlated feature reduces impurity most at each node), so I'd be more inclined to keep both raw features and let feature importance / permutation importance / SHAP analysis tell me post-hoc which one the model actually relies on, only pruning for reasons like inference-time latency, cost, or maintainability.

**Q16: How would you approach EDA on a dataset too large to fit in memory?**
A: I'd switch from "load everything into pandas" to a workflow built around sampling and chunked/out-of-core tools: take a stratified random sample (preserving rare-class proportions) for the fast, iterative exploratory pass; use `pd.read_csv(chunksize=...)` or a lazy engine (Dask, Polars, DuckDB) for exact aggregate statistics that must cover the full dataset; and skip the most expensive parts of automated profiling (full pairwise correlation/pair plots) on anything beyond a sample, since their cost grows quickly with both rows and columns. I'd also push simple aggregations (counts, means, distinct counts) down into the source system (SQL warehouse) rather than pulling raw rows into Python whenever possible.

**Q17: A colleague takes a 1% random sample of a highly imbalanced dataset (0.3% positive class) to speed up EDA and reports "the positive class looks basically absent." What's wrong with their approach?**
A: A uniform 1% sample of a 0.3%-positive-rate dataset leaves an expected ~0.003% of the total rows as positives — often only a handful of examples, nowhere near enough to draw any real conclusion about that class's behavior. The fix is stratified sampling that preserves (or even oversamples) the rare class specifically for exploratory purposes, so the sample used to explore the minority class has enough examples to be informative, while being explicit that the sample's class *proportions* no longer represent the true population — any prevalence estimate must still come from the full data, not the stratified sample.

**Q18: What is survivorship bias, and how would you check for it while exploring a dataset of "all companies that received Series A funding"?**
A: Survivorship bias means the dataset excludes entities that didn't "survive" a filtering step, and those missing entities are often the most informative. For "all companies that received Series A" I'd ask: does this include companies that later failed, got acquired, or shut down, or only ones still operating/trackable today? If the data source (e.g., a scraped list of "active" companies) systematically drops failed companies, any pattern found (e.g., "companies with feature X grew fast") is biased toward survivors and can't be used to infer what predicts success, since the failures that lacked feature X (or had it and still failed) are invisible. I'd check by cross-referencing against an independent, closure-inclusive source (e.g., a registry that tracks dissolutions) and comparing counts.

**Q19: You scan a correlation heatmap across 60 numeric columns and find 4 pairs with |r| > 0.3 that have no obvious mechanism connecting them. How do you interpret this?**
A: With 60 columns there are (60×59)/2 = 1,770 pairwise comparisons; at a naive per-test false-positive rate, several "significant"-looking correlations are expected purely by chance even if no real relationship exists. I'd apply a multiple-comparisons correction (Bonferroni or Benjamini-Hochberg/FDR) before treating any of the four as real, check whether each holds on an independent held-out sample, and only pursue ones with a plausible mechanism or domain story — an unexplained correlation found by broad scanning is a hypothesis to test, not a finding to act on.

**Q20: What questions would you ask about how a dataset was collected before trusting any EDA conclusion drawn from it?**
A: I'd ask: what population was this data drawn from, and does it match the population any resulting decision will be applied to (selection bias)? Does the dataset include entities that dropped out, failed, or were excluded at any stage, or only ones that "survived" to be recorded (survivorship bias)? Was inclusion in the dataset itself influenced by the outcome being studied (e.g., only hospitalized patients, only responders to a survey — Berkson's paradox)? Is there a systematic reason certain values would never be observed (censoring, MNAR missingness)? These questions can't be answered by a statistical test on the data itself — they require understanding how and why the data was generated in the first place.

**Q21: What's the biggest mistake candidates make when asked to walk through EDA on an unfamiliar dataset?**
A: The most common mistake is skipping straight to a correlation heatmap or a model without first confirming the grain of the data (what one row represents) and the population it was drawn from — leading to nonsensical "duplicate" or "outlier" judgments and missing selection/survivorship bias entirely. A strong answer instead follows a repeatable order: orient (grain, target, collection process) → structural sanity checks → univariate on the target and key features → bivariate toward the target with an eye for leakage → an explicit write-up of data-quality caveats, not just interesting findings.

---

## Data Visualization Principles and Techniques

### Choosing the Right Chart Type

**Concept:** chart choice is not a matter of taste — it should follow directly from (a) the number and types of variables involved, and (b) the specific comparison or pattern you want the viewer to perceive.

**Decision guide table:**

| Chart type | Best for | Variable types | Avoid when |
|---|---|---|---|
| **Bar chart** | Comparing magnitudes across discrete categories | 1 categorical + 1 numeric | Too many categories (>15–20, use sorted/top-N or a different chart); showing a trend over continuous time |
| **Line chart** | Trend over a continuous/ordered axis (usually time) | 1 continuous ordered axis + 1+ numeric | Categorical/unordered x-axis (implies a false trend); too many lines (>5–6, becomes spaghetti) |
| **Scatter plot** | Relationship/correlation between two continuous variables | 2 numeric (+ optional color/size for 1–2 more) | Large n with heavy overplotting (use hexbin/2D density instead); categorical x-axis |
| **Histogram** | Distribution shape of one continuous variable | 1 numeric | Comparing many groups at once (box/violin instead) |
| **Box plot** | Comparing distributions (median, IQR, outliers) across groups | 1 categorical + 1 numeric | Need to show multimodality (box plots hide it — use violin instead); very small n per group |
| **Violin plot** | Comparing full distribution shape (including multimodality) across groups | 1 categorical + 1 numeric | Small n (KDE estimate becomes unreliable/misleading); audience unfamiliar with density plots |
| **Heatmap** | Magnitude across two categorical/ordinal dimensions, or correlation matrices | 2 categorical/ordinal + 1 numeric (color) | Precise value reading is critical (numbers get lost in color); too many unordered categories |
| **Pair plot** | Overview of pairwise relationships + distributions across many numeric variables | 3+ numeric | More than ~8–10 variables (becomes unreadably dense); presenting to non-technical audiences |
| **Stacked bar chart** | Part-to-whole comparison across categories | 1 categorical (x) + 1 categorical (stack) + 1 numeric | More than ~4–5 stack segments (hard to compare middle segments); precise comparison of any segment except the first/total |
| **Area chart** | Cumulative trend / part-to-whole over time | 1 time axis + 1+ numeric | Comparing exact values between series (use line chart); too many overlapping series |
| **Pie / donut chart** | Simple part-to-whole with very few (2–5) categories | 1 categorical + 1 numeric (must sum to a meaningful whole) | More than ~5 categories; comparing across multiple pies; precise comparison (bar charts are almost always better) |
| **Bubble chart** | Relationship between 2 numeric variables + a 3rd numeric via size (+ optionally a 4th via color) | 3–4 numeric/categorical | More than ~50 points (overlap/occlusion); precise size comparison (humans are bad at judging area) |
| **Small multiples / facets** | Comparing the same chart across many subgroups | 1+ categorical (facet) + any base chart | Too many facets to fit on one screen without shrinking axes illegibly |

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Bar: category comparison
sns.barplot(data=df, x="region", y="revenue", estimator="mean", errorbar="sd")

# Line: trend over time
df.set_index("date")["sales"].plot(figsize=(10, 4))

# Scatter: relationship between two continuous variables
sns.scatterplot(data=df, x="ad_spend", y="revenue", hue="channel", alpha=0.6)

# Histogram: distribution
sns.histplot(df["age"], bins=30, kde=True)

# Box plot: distribution comparison across groups
sns.boxplot(data=df, x="region", y="revenue")

# Violin plot: distribution shape comparison (shows multimodality box plots hide)
sns.violinplot(data=df, x="region", y="revenue", inner="quartile")

# Heatmap: correlation matrix or 2D magnitude
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", center=0)

# Pair plot: many numeric relationships at once
sns.pairplot(df[["age", "income", "spend", "target"]], hue="target", corner=True)

# Small multiples / facets
g = sns.FacetGrid(df, col="region", col_wrap=3, height=3)
g.map(sns.histplot, "revenue")
```

**A simple mental checklist for chart choice:**
1. What is the *comparison* the viewer needs to make? (magnitude, trend, distribution, relationship, part-to-whole, ranking)
2. How many variables, and what types (categorical vs. continuous, ordered vs. unordered)?
3. How many categories/series will actually be shown — will it still be legible at that count?
4. Does the chart need to support precise value-reading, or just show a pattern/shape?

**Pitfalls:**
- Using a pie chart with many slices, or slices that don't sum to a meaningful whole.
- Using a line chart to connect an unordered categorical x-axis — implies a trend that doesn't exist.
- Using dual y-axes to overlay two differently-scaled series on one chart — this is one of the most common ways to mislead (or confuse) an audience; prefer two aligned small-multiple panels, indexing both series to a common baseline (e.g., % change from period 0), or a scatter plot instead.
- Defaulting to a bar chart for distributions when a box/violin/histogram would show far more (bars of means hide all variance and shape information).
- 3D charts for anything other than a genuinely spatial 3D relationship — they distort perceived magnitude and are almost always worse than a well-chosen 2D encoding (color, size, facet).

### Geospatial Visualization: Choropleths and Map Pitfalls

**Concept:** maps are a specialized chart type most EDA guidance skips — they're the right choice specifically when *where* something happens is itself the pattern, but they carry unique failure modes (area bias, projection distortion, unnormalized rates) that don't exist in standard charts.

**Choosing a geospatial chart type:**

| Chart | Shows | Use when | Avoid when |
|---|---|---|---|
| **Point / scatter map** | Exact location of individual events | Event-level precision matters (e.g., crime incidents, store locations) | Too many overlapping points (use a density/heatmap layer instead) |
| **Choropleth map** | A value aggregated per region, encoded as fill color | Comparing a *rate* across well-defined administrative regions (states, countries, zip codes) | The value is a raw count rather than a rate (see below); regions vary hugely in size and size itself isn't meaningful |
| **Heatmap / density (KDE) map** | Smoothed density of point events | Many overlapping point events where density itself is the pattern | Precise boundaries/values need to be read off |
| **Cartogram** | Regions resized by a variable (e.g., population) instead of true geographic area | Geographic area itself would visually mislead about importance (e.g., election maps by county) | Audience needs to recognize familiar geographic shapes at a glance |

```python
import plotly.express as px

# Choropleth of a RATE, not a raw count — normalize first
state_summary["cases_per_100k"] = state_summary["cases"] / state_summary["population"] * 100_000

fig = px.choropleth(
    state_summary,
    locations="state_code",
    locationmode="USA-states",
    color="cases_per_100k",         # rate, not raw count
    scope="usa",
    color_continuous_scale="Viridis",
)
fig.show()

# geopandas for custom shapefiles / equal-area projections
import geopandas as gpd
gdf = gpd.read_file("regions.shp")
gdf = gdf.to_crs("EPSG:6933")   # an equal-area projection, not the default Web Mercator (EPSG:3857)
gdf.plot(column="rate", cmap="viridis", legend=True)
```

**The single biggest choropleth pitfall — unnormalized raw counts:** a choropleth of raw case counts, raw sales, or raw population by region mostly just re-draws a population-density map — large, populous regions look "worse" or "bigger" purely because they have more people, not necessarily because the underlying rate is higher. Always ask "should this be divided by population, area, or another denominator before coloring the map?" before building it.

**Other map-specific pitfalls:**
- **Projection distortion:** the default Web Mercator projection used by most web mapping libraries dramatically inflates the visual area of regions far from the equator (Greenland appears comparable in size to Africa despite being roughly 14x smaller) — use an equal-area projection when the map's color/size encoding depends on area being visually honest.
- **Classification breaks change the story:** the same continuous variable binned into choropleth color buckets via quantile breaks vs. equal-interval breaks vs. Jenks natural breaks can make a region look dramatically more or less extreme — state which binning method was used.
- **Too many small regions:** a choropleth of thousands of small regions (e.g., US counties) at a national zoom level renders illegibly small slivers of color — consider aggregating to a coarser region or using a cartogram/bar chart instead.
- **Sequential/diverging color rules still apply** — a rate with a meaningful zero midpoint (e.g., % change vs. last year) needs a diverging palette centered at zero, not a single-hue sequential ramp that implies only magnitude, not direction.

### Visualization Best Practices

**Concept:** a chart's job is to be read correctly and quickly; every design choice either supports or undermines that.

**Axes and scale:**
- Bar charts comparing magnitude **must** start the numeric axis at zero — truncating the axis exaggerates differences and is one of the most common ways charts mislead.
- Line charts tracking change over time can sometimes reasonably use a non-zero baseline if the goal is showing *shape of change* rather than magnitude comparison, but this should be labeled clearly.
- Use a **log scale** when data spans multiple orders of magnitude (e.g., company revenue from startups to Fortune 500) or when the underlying relationship is multiplicative — label the axis explicitly as logarithmic so readers aren't misled about linear differences.
- Keep aspect ratio reasonable — stretching or squashing an axis can visually exaggerate or minimize a trend (Tufte's "lie factor": the visual size of an effect in the chart should match its actual size in the data).

```python
# BAD: truncated y-axis on a bar chart exaggerates the difference
fig, ax = plt.subplots()
ax.bar(["A", "B"], [98, 102])
ax.set_ylim(95, 105)   # misleading — makes a 4% difference look enormous

# GOOD: zero-based axis for magnitude comparison
fig, ax = plt.subplots()
ax.bar(["A", "B"], [98, 102])
ax.set_ylim(0, 110)

# GOOD: log scale when data spans orders of magnitude
fig, ax = plt.subplots()
ax.scatter(df["company_size"], df["revenue"])
ax.set_yscale("log")
ax.set_ylabel("Revenue (log scale)")
```

**Color choice:**

| Data type | Palette type | Example |
|---|---|---|
| Categorical (identity, no order) | Qualitative palette, distinct hues | `sns.color_palette("Set2")`, `tab10` |
| Sequential/magnitude (ordered, one direction) | Single-hue sequential ramp, light→dark | `sns.color_palette("Blues", as_cmap=True)`, `viridis` |
| Diverging (polarity around a meaningful midpoint, e.g., 0 or "average") | Two hues diverging from a neutral center | `coolwarm`, `RdBu`, `sns.diverging_palette(...)` |
| Highlighting one series/category among many | Muted grays for context + one saturated accent color for the highlighted item | Manual color mapping |

- Never use a **rainbow colormap** (e.g., `jet`) for continuous/sequential data — it has no consistent perceptual ordering and creates false visual boundaries; use `viridis`, `cividis`, or a single-hue sequential ramp instead.
- Always check **colorblind accessibility** — roughly 8% of men have some form of color vision deficiency (most commonly red-green). Avoid relying on red/green alone to encode meaning; use colorblind-safe palettes (`viridis`, `cividis`, ColorBrewer sets) and redundant encoding (position, shape, or direct labels) so color is never the *only* channel carrying meaning.
- Keep color consistent across a related set of charts (e.g., "Region A" is always blue in every chart in a report) — inconsistent color mapping across charts in the same deck is a common source of confusion.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Colorblind-safe categorical palette
sns.set_palette("colorblind")

# Sequential (magnitude) — perceptually uniform, not a rainbow
sns.heatmap(corr_matrix, cmap="viridis")

# Diverging (polarity around zero)
sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
```

**Chart junk and minimalism (Tufte's data-ink ratio):**
- Maximize the "data-ink ratio" — every mark should carry information; remove gridlines, borders, backgrounds, and decorative elements that don't help interpretation.
- Avoid 3D effects, unnecessary drop shadows, gradient fills, and heavy borders on 2D data — they distort perceived value and add no information.
- Remove redundant legends when direct labeling is clearer (e.g., labeling each line directly at its end point instead of a separate legend box, for a small number of series).
- Use gridlines sparingly and make them light/muted (recessive) — they should support reading values, not compete visually with the data itself.

```python
# Minimal, high data-ink-ratio styling
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df["date"], df["sales"], color="#2C6E91", linewidth=2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)
ax.set_title("Monthly Sales Trend", fontsize=13, weight="bold")
```

**Accessibility checklist:**
- Sufficient contrast between text/marks and background (WCAG AA: ≥4.5:1 for normal text).
- Colorblind-safe palettes, with a secondary encoding (pattern, shape, direct label) for critical distinctions.
- Font sizes legible at the intended viewing context (a chart embedded in a slide needs larger text than one in a dense report).
- Alt text / data tables as an alternative to any chart used in a public-facing or accessibility-audited product.

**Pitfalls:**
- Truncated/non-zero axes on bar charts (see above) — the single most common way charts mislead.
- Using red/green exclusively to encode good/bad — invisible to red-green colorblind viewers; use blue/orange or add icons/labels.
- Overloading a chart with too many series/colors ("spaghetti chart") — beyond ~5–6 distinct colors, viewers can no longer reliably track individual series; use small multiples or highlight-one-context-rest-gray instead.
- Inconsistent scales across small multiples meant to be compared — if the comparison matters, use the same axis range across all panels; if showing each panel's own shape matters more, label that clearly.
- Choosing chart types to make results look more dramatic/impressive rather than more accurate — an interviewer evaluating "business acumen" or "communication" competencies specifically probes for this kind of intellectual honesty.

### Tools: Matplotlib, Seaborn, Plotly, and BI Tools

**Concept:** each tool trades off control, speed of iteration, interactivity, and audience — knowing when to reach for which is itself an interview signal of practical maturity.

| Tool | Layer / Style | Strengths | Weaknesses | Best for |
|---|---|---|---|---|
| **Matplotlib** | Low-level, imperative | Total control over every visual element; the foundation other Python libraries build on; publication-quality static output | Verbose for common statistical charts; no built-in interactivity | Custom, precise, publication/paper-quality static figures |
| **Seaborn** | Built on matplotlib, declarative/statistical | Beautiful defaults; built-in statistical aggregation (regression lines, CIs, distributions) with one line; integrates with pandas DataFrames directly | Less low-level control than raw matplotlib (though you can drop down into matplotlib axes) | Fast, good-looking statistical EDA charts during analysis |
| **Plotly** | High-level, interactive, web-based (built on D3.js/WebGL under the hood) | Zoom/pan/hover tooltips out of the box; easy export to interactive HTML/dashboards (Dash); works well in notebooks and web apps | Larger file sizes for HTML export; some chart types less mature than matplotlib's | Interactive exploration, stakeholder-facing dashboards, web apps |
| **Tableau** | BI tool, drag-and-drop | Extremely fast for interactive dashboards without code; strong for business users; live database connections; polished out-of-box interactivity (filters, drill-down) | Less reproducible/version-controllable than code; licensing cost; harder to embed custom statistical logic | Business dashboards, self-service analytics for non-technical stakeholders |
| **Power BI** | BI tool, drag-and-drop, Microsoft ecosystem | Tight integration with Excel/Azure/Microsoft 365; DAX for custom calculations; strong enterprise governance | Less flexible for bespoke/statistical visuals than code; steeper DAX learning curve for advanced logic | Enterprise reporting, especially in Microsoft-centric organizations |

```python
# Matplotlib — full control, static
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df["date"], df["sales"], marker="o")
ax.set(title="Sales Over Time", xlabel="Date", ylabel="Sales ($)")
plt.savefig("sales_trend.png", dpi=300, bbox_inches="tight")

# Seaborn — fast statistical charts, pandas-native
import seaborn as sns
sns.lmplot(data=df, x="ad_spend", y="revenue", hue="channel", height=5, aspect=1.3)

# Plotly — interactive, web-ready
import plotly.express as px
fig = px.scatter(df, x="ad_spend", y="revenue", color="channel",
                  hover_data=["campaign_name"], trendline="ols")
fig.update_layout(title="Ad Spend vs Revenue by Channel")
fig.show()
# fig.write_html("interactive_chart.html")

# Plotly for a quick interactive dashboard-style figure
fig2 = px.line(df, x="date", y="sales", color="region")
fig2.update_xaxes(rangeslider_visible=True)   # built-in zoomable time range
```

**Decision guidance — which tool for which stage of work:**

| Stage | Tool of choice |
|---|---|
| Quick exploratory look during analysis | Seaborn (fast, statistical defaults) |
| Deep-dive interactive exploration, needing to zoom/hover/filter | Plotly (or Tableau/Power BI if non-technical collaborators need to explore too) |
| Final, precise, publication/report-quality static figure | Matplotlib (fine control over every element) |
| Recurring business dashboard for non-technical stakeholders | Tableau or Power BI (self-service, no code required to explore) |
| Embedding a chart in a web app or shareable interactive artifact | Plotly / Plotly Dash, or a JS library (D3, Recharts) for full custom control |

**Pitfalls:**
- Defaulting to matplotlib for every EDA chart out of habit — seaborn is faster for the vast majority of standard statistical charts and produces better default aesthetics.
- Shipping a static chart when the audience actually needs to filter/drill down themselves — that's a sign a BI tool or Plotly dashboard is the right artifact, not a PNG in a slide deck.
- Using a BI tool for something that needs custom statistical modeling behind the visual (e.g., a bespoke Bayesian confidence band) — code-based tools remain more flexible for bespoke analytics logic even if a BI tool is used for the final presentation layer.
- Forgetting `dpi` and `bbox_inches="tight"` when saving matplotlib figures for reports/slides — default exports are often low-resolution or have layout/cropping issues.

### Dashboarding Concepts

**Concept:** a dashboard is not "several charts on one page" — it's a decision-support tool built around specific KPIs, audiences, and actions, and interviewers probe whether you think about *purpose* before building.

**KPIs (Key Performance Indicators):**
- A good KPI is: tied to a business objective, actionable (someone can change their behavior based on it), measurable consistently over time, and has a clear owner.
- Distinguish **lagging indicators** (outcomes, e.g., quarterly revenue — confirm what already happened) from **leading indicators** (inputs/predictors, e.g., number of demos booked this week — predict what's coming and are actionable sooner).
- Every KPI on a dashboard should answer "so what should I do differently if this number changes?" — if no one can answer that, it's a vanity metric, not a KPI.

**Drill-down and hierarchy:**
- Design dashboards with a top-level summary view (headline KPIs, sparkline trends) and the ability to drill into detail (by region → by store → by SKU) without leaving the same context — this avoids overwhelming the top-level view while still supporting deep investigation.
- Use consistent filters (date range, segment, region) that apply across all charts in the dashboard simultaneously, not per-chart independent filters that can get out of sync.

**Storytelling with data:**
- Order visuals to build an argument: context → observation → insight → recommendation, not just "here are twelve charts."
- Lead with the takeaway in a title or annotation (e.g., "Churn rose 12% after the March price change" beats a generic title like "Monthly Churn Rate") — this is often called an "active title" or "so-what title."
- Use annotations directly on the chart to call out the specific point that matters (an event marker, a threshold line) rather than relying on the audience to notice it themselves.
- Match the level of detail to the audience: executives need the headline number and trend; analysts need the ability to drill into segments and raw data.

```python
# Example: an annotated chart that tells a story, not just shows data
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["date"], df["churn_rate"], color="#C0392B", linewidth=2)
ax.axvline(pd.Timestamp("2026-03-01"), color="gray", linestyle="--", alpha=0.7)
ax.annotate("Price change\n(March 2026)", xy=(pd.Timestamp("2026-03-01"), df["churn_rate"].max()),
            xytext=(10, 10), textcoords="offset points", fontsize=9, color="gray")
ax.set_title("Churn Rate Rose 12% After the March Price Change", fontsize=13, weight="bold")
ax.set_ylabel("Monthly Churn Rate (%)")
ax.spines[["top", "right"]].set_visible(False)
```

**Common dashboard anti-patterns to avoid:**
- Too many KPIs on one screen with no visual hierarchy — everything looks equally important, so nothing is.
- Vanity metrics (e.g., raw page views with no context) that don't connect to a decision.
- Dashboards that only show current state with no trend/comparison (a single number without "vs. last period" or "vs. target" context is hard to interpret).
- No clear owner or refresh cadence — a stale, unmaintained dashboard erodes trust in data generally.

**Pitfalls:**
- Building the dashboard the analyst finds interesting rather than the one the stakeholder needs to make a decision — always start from "what decision does this support?"
- Real-time/high-frequency refresh for metrics that don't actually change meaningfully at that frequency — adds engineering cost and can create false urgency/noise-chasing.
- Not aligning on KPI *definitions* across teams (e.g., two teams calculating "active user" differently) before building a shared dashboard — leads to distrust when numbers don't match between reports.

### Interview Questions

**Q1: You need to compare the distributions of a numeric variable across 6 categories. Would you use a box plot, violin plot, or something else, and why?**
A: I'd lean toward a violin plot if the distributions might be multimodal or have interesting shape beyond quartiles, since a box plot only shows median/IQR/whiskers and would hide bimodality. If the audience is non-technical and unfamiliar with density plots, or if I need a compact, easily-compared summary across many categories, a box plot is often clearer and faster to read. In either case I'd add the individual data points as a jittered strip/swarm plot overlay if `n` per category is small enough to show them without overplotting, since raw points combined with a summary give the most complete and honest picture.

**Q2: What's wrong with truncating the y-axis on a bar chart, and when (if ever) is a non-zero baseline acceptable?**
A: Bar charts encode magnitude via bar length/area, and human perception reads that length relative to a zero baseline — truncating the axis exaggerates the *visual* difference between bars relative to their true proportional difference, which is a classic way charts mislead (intentionally or not). A non-zero baseline is more defensible for line charts specifically tracking the *shape* of change over time rather than absolute magnitude comparison (e.g., zooming into a narrow but meaningful fluctuation), and even then it should be clearly labeled so the reader isn't misled about the absolute scale.

**Q3: How would you visualize the relationship between three variables — two continuous and one categorical — on a single chart?**
A: A scatter plot with the two continuous variables on the x/y axes and the categorical variable encoded via color (`hue`) is usually the cleanest option, since color is a strong pre-attentive channel for a small number of categories (up to about 5-8 with a good qualitative palette). If the categorical variable has too many levels for color to stay legible, I'd use small multiples/facets instead — one scatter plot panel per category, with consistent axis scales across panels so they're visually comparable.

**Q4: A stakeholder asks you to add a second y-axis to a line chart to show revenue and customer count on the same plot. What do you tell them?**
A: I'd explain that dual-axis charts are one of the most common ways to visually mislead an audience, because the two independently-scaled axes can be manipulated (intentionally or not) to make two unrelated trends appear to move together or apart, and viewers instinctively compare the line *positions* without registering that the scales differ. I'd suggest alternatives: two vertically stacked panels sharing the same x-axis (easy to compare timing of movements without conflating scale), indexing both series to a common baseline (e.g., % change from period 1) so they share one meaningful axis, or a scatter plot if the actual question is about correlation between the two rather than their trends over time.

**Q5: What makes a color palette "colorblind-safe," and how would you check one?**
A: A colorblind-safe palette chooses hues that remain distinguishable under the common forms of color vision deficiency (primarily red-green deficiencies like deuteranopia/protanopia), typically by varying lightness/saturation alongside hue rather than relying on hue alone, and by avoiding red-green as the sole distinguishing pair. I'd check a palette using tools like a colorblind simulator (e.g., Coblis, or matplotlib/seaborn's built-in colorblind-safe palettes such as `"colorblind"` or `viridis`/`cividis`), or programmatically validate perceptual distance between colors in a colorblind-safe color space rather than eyeballing it, and I'd always pair color with a redundant encoding (direct labels, shape, or position) for anything critical.

**Q6: When would you choose Plotly over Seaborn, or vice versa, for a given task?**
A: I'd use Seaborn for fast, static, statistically-aware exploratory charts during analysis — it has excellent defaults for things like regression lines with confidence bands, distribution plots, and categorical comparisons, and integrates directly with pandas DataFrames with minimal code. I'd switch to Plotly when the audience needs to interact with the chart themselves — zooming into a time range, hovering for exact values, filtering by clicking a legend entry — or when the deliverable is a web-embeddable interactive artifact or dashboard rather than a static image in a report or paper.

**Q7: How would you design a dashboard for an executive audience differently than one for a team of analysts?**
A: For executives, I'd surface a small number of headline KPIs with clear trend context (vs. last period, vs. target), an "active" title stating the key takeaway, and minimal ability (or need) to drill down — the goal is a 10-second read that supports a decision. For analysts, I'd build in drill-down capability (region → store → SKU), consistent cross-filtering across all charts, access to underlying data tables, and more granular time controls — the goal is supporting investigation, not just headline consumption. I'd avoid giving executives an analyst-style dense multi-filter dashboard, since it burns their attention on navigation rather than insight.

**Q8: What is "chart junk," and give three concrete examples you'd remove from a chart.**
A: Chart junk (Tufte's term) is any visual element that doesn't carry data and reduces the "data-ink ratio" — it adds visual noise without adding information, and often actively distracts from the pattern the chart is meant to convey. Three concrete examples: (1) unnecessary 3D effects/gradients/drop shadows on bars or pie slices, which distort perceived magnitude with no added information; (2) heavy gridlines and borders competing visually with the data marks — these should be minimal/muted, present only enough to support reading values; (3) redundant legends when direct labeling on the chart itself (e.g., labeling each line at its endpoint) would be clearer and remove the need for a separate legend box for a small number of series.

**Q9: How would you pick between a heatmap and a pair plot to explore correlations among 10 numeric variables?**
A: I'd start with a correlation heatmap to get a fast, compact overview of pairwise linear correlation strength across all 10 variables at once, sorted or clustered (e.g., a seaborn `clustermap`) to visually group correlated blocks. From there, I'd select the handful of pairs with the strongest (or most surprising) correlations and generate targeted scatter plots or a smaller pair plot for just those, since a full 10x10 pair plot is visually dense and slow to render, and a heatmap alone can't reveal whether a "correlation" is actually driven by outliers or non-linearity.

**Q10: What is a "leading" vs. "lagging" KPI, and why does the distinction matter for dashboard design?**
A: A lagging indicator measures an outcome that has already happened (e.g., quarterly revenue, churn last month) — useful for confirming results but not actionable in the moment since the underlying behavior is already in the past. A leading indicator measures an input or predictor that correlates with a future outcome and is actionable now (e.g., number of sales demos booked this week, website signup rate) — it gives stakeholders a chance to intervene before the lagging outcome materializes. A well-designed dashboard usually pairs both: leading indicators for early warning and course-correction, lagging indicators for accountability and confirming whether interventions actually worked.

**Q11: A pie chart shows 9 slices, several nearly the same size. What would you change and why?**
A: I'd replace it with a sorted horizontal bar chart. Pie charts rely on humans accurately judging angle/area, which we're poor at, especially for slices of similar size — a bar chart lets viewers compare lengths directly and precisely, and sorting by value makes rank order immediately visible. If part-to-whole is genuinely the point (not just comparison), I'd consider a 100% stacked bar instead, or collapse the smallest slices into an "Other" category to reduce the pie to 4-5 meaningfully distinct segments if a pie is specifically required.

**Q12: How do you decide whether a table of numbers or a chart is the better way to present a specific piece of information?**
A: If the audience needs to read exact, precise values (e.g., a specific customer's exact invoice amounts), a table is usually better — charts sacrifice precision for pattern-recognition speed. If the point is to convey a *pattern*, *trend*, *comparison*, or *outlier* quickly — something the human visual system is good at parsing pre-attentively (magnitude, trend direction, relative ranking) — a chart wins. In practice, the best dashboards often combine both: a chart for the pattern plus a linked/expandable table for the precise underlying numbers.

**Q13: What's the difference conceptually between a tool like Tableau/Power BI and writing charts in Python (matplotlib/seaborn/plotly)? When would you recommend each to a team?**
A: BI tools (Tableau, Power BI) are built for fast, code-free, interactive exploration and reporting by business users, with strong live-database connectivity, built-in filtering/drill-down UI, and enterprise governance/sharing features — the tradeoff is less flexibility for bespoke statistical logic and weaker reproducibility/version control compared to code. Python visualization libraries offer full programmatic control, integrate directly with the modeling/analysis codebase, are fully reproducible and version-controlled, and can implement arbitrary custom statistical logic — the tradeoff is more development time and typically no built-in interactivity unless using Plotly/Dash specifically. I'd recommend BI tools for recurring business reporting consumed by many non-technical stakeholders, and Python-based visualization for one-off deep analysis, research reproducibility, or when the visualization needs to be tightly coupled to custom modeling code.

**Q14: How would you visualize a dataset with a highly skewed numeric variable (e.g., income) without misleading the audience?**
A: I'd avoid a plain linear-scale histogram, since the long right tail compresses most of the data into a few narrow bars and gives a misleading sense of "typical" values. Options: plot on a log scale (clearly labeled) to spread out the compressed low-value region; show both mean and median together (and explicitly note the gap between them as evidence of skew) rather than just the mean; use a box plot or violin plot which explicitly show the median/IQR and outliers rather than a potentially misleading average; or bin using unequal-width bins (e.g., quantile-based bins) so each bar represents a similar amount of data rather than a similar range of values.

**Q15: What's an "active" or "so-what" title, and why does it matter for data storytelling?**
A: An active title states the key takeaway directly (e.g., "Churn rose 12% after the March price change") instead of merely describing what the chart contains (e.g., "Monthly Churn Rate, Jan–Jun 2026"). It matters because most viewers scan a title before deciding whether to engage with the chart at all — a descriptive title puts the interpretive burden entirely on the viewer, while an active title front-loads the insight, guides the viewer's attention to what matters in the chart, and makes the presentation feel like an argument with a point rather than a data dump. The tradeoff is that an active title reflects the presenter's specific interpretation, so it should still be accurate and defensible, not spun.

**Q16: A choropleth map colors US states by total number of reported cases of a disease. What's wrong with this, and how would you fix it?**
A: Coloring by raw count mostly reproduces a population map — California and Texas will look "worst" simply because they have the most people, not necessarily the highest rate of the disease. I'd normalize by population (cases per 100,000 residents) or another meaningful denominator before mapping, and choose a sequential (or diverging, if there's a meaningful baseline like "vs. last year") color scale consistent with the rest of the deck's palette conventions.

**Q17: Why might Greenland look almost as large as Africa on a web map, and why does that matter for a choropleth?**
A: Most web mapping libraries default to the Web Mercator projection, which preserves angles/shape locally but distorts area dramatically at higher latitudes — Greenland is actually about 14x smaller than Africa despite appearing similar in size on a Mercator map. For a choropleth, where color is meant to represent a rate but *area* is also a strong visual cue, this projection distortion can make far-north or far-south regions visually overrepresented; using an equal-area projection avoids letting projection artifacts compete with the actual data encoding.

**Q18: When would you use a point/bubble map instead of a choropleth for the same underlying location data?**
A: A choropleth aggregates values to predefined regions and is right when the comparison is inherently regional (e.g., state-level tax rates) and regions are a natural, meaningful unit. A point or bubble map is better when the events themselves have precise locations that matter at finer resolution than any administrative boundary (e.g., individual store locations, crime incidents, sensor readings) — aggregating those to a choropleth would hide meaningful within-region clustering. If there are too many overlapping points, a density/heatmap layer on top of the point map bridges the two by showing concentration without forcing an artificial regional boundary.

---

## Rapid-Fire Interview Q&A

| # | Question | Answer |
|---|---|---|
| 1 | What's the difference between MCAR and MAR? | MCAR: missingness unrelated to any variable. MAR: missingness depends on observed variables, not the missing value itself. |
| 2 | Median or mean for skewed data imputation? | Median — it's robust to the skew/outliers that pull the mean away from the "typical" value. |
| 3 | Name two robust alternatives to z-score for outlier detection. | IQR method and modified z-score (MAD-based). |
| 4 | What does IQR stand for and how is it computed? | Interquartile Range; Q3 − Q1. |
| 5 | Why scale features before KNN imputation or KNN-based outlier detection? | KNN uses distance; unscaled features with larger numeric ranges dominate the distance calculation. |
| 6 | What's data leakage in the context of preprocessing? | Fitting a transformer (imputer, scaler, outlier bound) on data that includes information from the validation/test set. |
| 7 | Pearson vs. Spearman in one line? | Pearson measures linear correlation; Spearman measures monotonic (rank-based) correlation. |
| 8 | What does a VIF above 10 usually indicate? | Severe multicollinearity between that predictor and the others. |
| 9 | What's the null hypothesis of the ADF test? | The series is non-stationary (has a unit root). |
| 10 | What's the null hypothesis of the KPSS test? | The series is stationary — opposite of ADF, which is why both are often run together. |
| 11 | ACF vs. PACF, one-line difference? | ACF includes indirect correlation through intermediate lags; PACF removes that, showing only the direct correlation at each lag. |
| 12 | Name one pitfall of mean imputation. | It shrinks variance and attenuates correlations with other variables. |
| 13 | What chart type is best for comparing distributions across many groups while showing multimodality? | Violin plot. |
| 14 | Why avoid rainbow colormaps like `jet`? | No consistent perceptual ordering; creates false visual boundaries in continuous data. |
| 15 | Why must a magnitude-comparison bar chart start its y-axis at zero? | Truncating the axis exaggerates the visual difference between bars relative to the true proportional difference. |
| 16 | What is chart junk? | Visual elements that don't carry information and lower the data-ink ratio (e.g., unnecessary 3D effects, heavy gridlines). |
| 17 | What's the main risk of a dual-axis chart? | It can make two unrelated or differently-scaled trends appear artificially correlated or uncorrelated. |
| 18 | Give one reason tree-based models are less sensitive to multicollinearity than linear regression. | Trees split on whichever correlated feature reduces impurity most at each node; coefficient instability isn't a concept that applies. |
| 19 | What's Simpson's Paradox? | A trend in aggregated data reverses when the data is split by a confounding subgroup. |
| 20 | Name a fast way to check if two datasets (train vs. production) have drifted. | Compare Population Stability Index (PSI) or run a two-sample test (KS test for continuous, chi-square for categorical) per feature. |
| 21 | What does `errors="coerce"` do in `pd.to_numeric`? | Converts unparseable values to `NaN` instead of raising an error. |
| 22 | Why use `category` dtype in pandas? | Reduces memory usage and speeds up groupby/filter operations for low-cardinality string columns. |
| 23 | What's the difference between a lagging and leading KPI? | Lagging measures a past outcome (confirms results); leading measures a predictive input (enables proactive action). |
| 24 | When is a log transform appropriate? | Right-skewed, strictly positive data, or when relationships are multiplicative rather than additive. |
| 25 | What's the safest deduplication approach when unsure of a table's grain? | Deduplicate on the full natural/business key, not a single non-unique column, after confirming what "duplicate" means for that table. |
| 26 | Name one reason to prefer Isolation Forest over IQR for outlier detection. | It captures multivariate outliers (unusual combinations of features), which single-column IQR checks cannot detect. |
| 27 | What's a "so-what" or active chart title? | A title that states the key takeaway/insight directly, instead of merely describing the chart's contents. |
| 28 | Why run both ADF and KPSS tests on a time series? | They have opposite null hypotheses; agreement between them gives more confidence in the stationarity conclusion. |
| 29 | What's the main weakness of pandas-profiling / ydata-profiling style automated EDA? | It surfaces statistical patterns but has no domain context — it can't tell you if a correlation or outlier is meaningful or expected. |
| 30 | One-line difference between Isolation Forest and DBSCAN for outlier detection? | Isolation Forest isolates points via random partitioning (works well in high dimensions); DBSCAN flags points that don't belong to any density cluster (works well for non-convex cluster shapes). |
| 31 | Why is a violin plot sometimes misleading with small sample sizes? | The kernel density estimate underlying its shape becomes unreliable and can suggest structure that isn't really there with too few points. |
| 32 | What's the recommended way to handle an unseen categorical value at inference time in production? | Route it to an explicit "Unknown"/"Other" bucket rather than crashing or silently mis-encoding it. |
| 33 | What's the main risk of a purely random sample of a highly imbalanced dataset? | It can leave the rare class with too few examples to draw any conclusion; use stratified sampling to preserve/oversample it. |
| 34 | Name one out-of-core/lazy alternative to pandas for datasets that don't fit in memory. | Dask, Polars (lazy scan), DuckDB, or Vaex. |
| 35 | What is survivorship bias, in one line? | Drawing conclusions only from entities that "survived" a filtering process while the ones that didn't are missing from the data. |
| 36 | What is Berkson's paradox? | A form of selection bias where conditioning on being included in a sample (e.g., hospital admission) creates a spurious association between two variables. |
| 37 | Why can scanning 100 pairwise correlations produce several "significant" results even with random data? | Multiple comparisons — at alpha 0.05, ~5% of independent tests are expected to be false positives by chance alone. |
| 38 | What's the single biggest choropleth map pitfall? | Coloring by a raw count instead of a normalized rate, which mostly just re-draws a population/size map. |
| 39 | Why does the default Web Mercator projection distort a choropleth? | It inflates the visual area of regions far from the equator, making them look disproportionately large/important. |
| 40 | What's the first thing to confirm before running any EDA on an unfamiliar dataset? | The grain of the data — what a single row represents — since duplicate/outlier judgments are meaningless without it. |

---
