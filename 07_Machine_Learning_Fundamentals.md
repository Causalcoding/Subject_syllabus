# Machine Learning Fundamentals (Classical ML) — Interview Prep Syllabus

Classical ML is the bedrock that every Data Scientist, Machine Learning Engineer, and AI Engineer interview loop tests, regardless of how much deep learning or LLM work the role actually involves. **Data Scientists** are expected to derive loss functions, check model assumptions, choose the right algorithm for tabular data, and interpret results for stakeholders. **Machine Learning Engineers** are expected to know training complexity, hyperparameter tuning, production trade-offs (latency, memory, retraining cost), and how to debug models that behave unexpectedly in production. **AI Engineers** — even those building primarily on top of LLMs — are expected to understand classical ML because it underlies feature engineering, evaluation baselines, retrieval ranking, guardrail classifiers, anomaly detection for monitoring, and because interviewers use these fundamentals as a proxy for general ML reasoning ability. This document covers classical ML from first principles through advanced/expert-level nuance, with math, code, comparisons, and full interview Q&A.

## Table of Contents

1. [Supervised Learning — Regression](#supervised-learning--regression)
2. [Supervised Learning — Classification](#supervised-learning--classification)
3. [Ensemble Methods](#ensemble-methods)
4. [Unsupervised Learning](#unsupervised-learning)
5. [Model Interpretability](#model-interpretability)
6. [Rapid-Fire Interview Q&A](#rapid-fire-interview-qa)

---

## Supervised Learning — Regression

### Linear Regression

**Concept.** Linear regression models a continuous target $y$ as a linear combination of features: $\hat{y} = \mathbf{w}^T\mathbf{x} + b$. It is the foundation for nearly every other model in this document (logistic regression, linear SVM, GLMs, and even the "linear head" of many deep nets).

**Assumptions (classical OLS / Gauss-Markov):**

| Assumption | Meaning | Consequence if violated |
|---|---|---|
| Linearity | $E[y\mid X] = X\beta$ | Underfitting, biased coefficients |
| Independence of errors | $\text{Cov}(\epsilon_i, \epsilon_j)=0$ | Autocorrelation → wrong SEs, time series leakage |
| Homoscedasticity | $\text{Var}(\epsilon_i) = \sigma^2$ constant | Heteroscedastic residuals → inefficient (not BLUE) estimates, wrong CIs |
| No/low multicollinearity | Features not (nearly) linear combinations of each other | Unstable, high-variance coefficient estimates |
| Normality of errors | $\epsilon \sim N(0,\sigma^2)$ | Needed for exact t/F tests and CIs, not for point estimates |
| No measurement error in X | Features measured without noise | Attenuation bias |

**OLS derivation.** We minimize the residual sum of squares (RSS):

$$J(\beta) = \sum_{i=1}^n (y_i - x_i^T\beta)^2 = (y - X\beta)^T(y-X\beta)$$

Take the gradient with respect to $\beta$ and set to zero:

$$\nabla_\beta J = -2X^T(y - X\beta) = 0 \implies X^TX\beta = X^Ty \implies \hat\beta = (X^TX)^{-1}X^Ty$$

This is the **normal equation**. Under Gauss-Markov assumptions, OLS is BLUE (Best Linear Unbiased Estimator) — lowest variance among all unbiased linear estimators.

**Normal equation vs Gradient Descent:**

| | Normal Equation | Gradient Descent |
|---|---|---|
| Complexity | $O(n \cdot d^2 + d^3)$ (matrix inversion) | $O(n \cdot d)$ per iteration |
| Scales to large $d$ (features)? | Poor (cubic in $d$) | Good |
| Scales to large $n$ (rows)? | Fine if $d$ small | Good, especially SGD/mini-batch |
| Requires feature scaling? | No | Yes (for convergence speed) |
| Requires $X^TX$ invertible? | Yes (else use pseudo-inverse) | No |
| Iterative/tunable? | No, closed form | Yes — learning rate, epochs |

**Multicollinearity.** When features are highly correlated, $X^TX$ becomes near-singular, so $(X^TX)^{-1}$ has huge entries → unstable, high-variance coefficient estimates (sign flips, huge SEs) even though predictions may still be fine. Diagnose with **Variance Inflation Factor (VIF)**: $VIF_j = \frac{1}{1-R_j^2}$ where $R_j^2$ is from regressing feature $j$ on all other features. VIF > 5–10 is a common warning threshold. Fixes: drop/combine correlated features, PCA, regularization (Ridge is the classic remedy since it directly shrinks and stabilizes coefficients).

**Residual analysis.** Standard diagnostic plots:
- **Residuals vs fitted**: should show no pattern (random cloud around 0). Curvature → non-linearity. Funnel shape → heteroscedasticity.
- **Q-Q plot**: check normality of residuals.
- **Scale-location plot**: check homoscedasticity.
- **Residuals vs leverage (Cook's distance)**: identify influential outliers.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

model = LinearRegression().fit(X_train, y_train)
residuals = y_train - model.predict(X_train)

# VIF check
X_const = sm.add_constant(X_train)
vif = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
```

**Strengths/weaknesses.** Fast, interpretable, well-understood inference (p-values, CIs), no hyperparameters to tune (base form). Weak on non-linear relationships, sensitive to outliers (squared loss), assumes additive effects unless you engineer interactions.

### Regularized Regression: Ridge, Lasso, Elastic Net

**Motivation.** Plain OLS can overfit with many/correlated features (high variance). Regularization adds a penalty on coefficient magnitude, trading a little bias for a lot of variance reduction.

**Ridge (L2):**
$$J(\beta) = \sum_i (y_i - x_i^T\beta)^2 + \lambda \sum_j \beta_j^2$$
Closed form: $\hat\beta_{ridge} = (X^TX + \lambda I)^{-1}X^Ty$. Adding $\lambda I$ makes the matrix always invertible — this is literally why Ridge was invented (to fix multicollinearity-induced singularity).

**Lasso (L1):**
$$J(\beta) = \sum_i (y_i - x_i^T\beta)^2 + \lambda \sum_j |\beta_j|$$
No closed form (non-differentiable at 0); solved via coordinate descent or LARS. Performs automatic feature selection by driving some coefficients exactly to zero.

**Elastic Net:** convex combination of both penalties:
$$J(\beta) = \sum_i (y_i-x_i^T\beta)^2 + \lambda\left(\alpha\sum_j|\beta_j| + (1-\alpha)\sum_j\beta_j^2\right)$$
Handles correlated features better than pure Lasso (which tends to arbitrarily pick one of a correlated group), while still allowing sparsity.

**Why Lasso induces sparsity — geometric intuition.** Think of the regularized problem as constrained optimization: minimize RSS subject to $\sum|\beta_j| \le t$ (L1) or $\sum\beta_j^2 \le t$ (L2). The L1 constraint region is a **diamond/cross-polytope** with sharp corners on the axes; the L2 constraint region is a **circle/sphere** with no corners. The RSS contours are ellipses centered at the OLS solution. Because the L1 diamond has corners exactly on the axes (where one or more coefficients are zero), the elliptical contours are much more likely to first touch the constraint region at a corner — hence some $\beta_j = 0$ exactly. The smooth L2 ball has no such corners, so the intersection point almost never lands exactly on an axis — coefficients shrink toward zero but rarely hit it exactly.

An algebraic view: for a single coordinate, the Lasso subgradient optimality condition is $x_j^T(y - X\beta) = \lambda \cdot \text{sign}(\beta_j)$ for $\beta_j \neq 0$, and $|x_j^T(y-X\beta)| \le \lambda$ for $\beta_j = 0$ — there's a whole range of correlations that map to exactly zero. Ridge's gradient condition $x_j^T(y-X\beta) = \lambda\beta_j$ has no such flat zero region — the derivative of $\beta_j^2$ vanishes at 0, so there's no "penalty force" strong enough to hold it exactly there.

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Always scale features before regularized regression!
ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.1))
enet  = make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.5))
```

**Hyperparameters:**
- `alpha` (λ): higher → more shrinkage → more bias, less variance. Tune via `RidgeCV`/`LassoCV`/`ElasticNetCV` (built-in efficient CV paths).
- `l1_ratio` (Elastic Net, $\alpha$ in the formula above): 0 = pure Ridge, 1 = pure Lasso.

**When to use which:**

| Scenario | Best choice |
|---|---|
| Many features, believe most are irrelevant | Lasso (sparse, feature selection) |
| Many correlated features, want to keep all with shrinkage | Ridge |
| Many features, some correlated groups, want sparsity too | Elastic Net |
| $p > n$ (more features than samples) | Lasso/Elastic Net (Ridge can't select but works; Lasso can select at most $n$ features) |
| Need stable, low-variance coefficients for inference | Ridge |

### Polynomial Regression & Bias-Variance in Model Complexity

**Concept.** Extend linear regression by adding polynomial/interaction terms of the features ($x, x^2, x^3, \ldots$), fit with plain linear regression on the expanded feature set. Still "linear" in the parameters, non-linear in the original inputs.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

poly_model = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
```

**Bias-variance decomposition.** For squared-error loss, expected test error decomposes as:
$$E[(y-\hat f(x))^2] = \underbrace{(\text{Bias}[\hat f(x)])^2}_{\text{model too simple}} + \underbrace{\text{Var}[\hat f(x)]}_{\text{model too sensitive to training data}} + \underbrace{\sigma^2}_{\text{irreducible noise}}$$

- **Degree too low** (e.g., degree 1 fit to quadratic data): high bias, low variance → underfitting.
- **Degree too high**: low bias, high variance → overfits noise, wiggly curve, terrible generalization.
- The **sweet spot** minimizes total error — found via validation curves / cross-validation, not training error (which monotonically decreases with degree).

```python
from sklearn.model_selection import validation_curve
train_scores, val_scores = validation_curve(
    make_pipeline(PolynomialFeatures(), LinearRegression()),
    X, y, param_name="polynomialfeatures__degree", param_range=range(1, 10), cv=5,
    scoring="neg_mean_squared_error")
```

**Complexity/dimensionality note:** degree-$d$ polynomial features on $p$ original features produce $O(p^d)$ terms (with interactions) — curse of dimensionality kicks in fast; regularization (Ridge/Lasso) is almost mandatory beyond low degrees.

### Loss Functions for Regression: MSE, MAE, Huber, Quantile

The choice of loss function is what the model actually optimizes during training — distinct from the *evaluation metric* you report afterward (see [Model Evaluation](06_Feature_Engineering_and_Model_Evaluation.md) for the full metrics survey). For regression, the training loss determines what "best fit" means and how sensitive the model is to outliers.

| Loss | Formula (per point, residual $r=y-\hat y$) | Gradient w.r.t. $\hat y$ | Behavior |
|---|---|---|---|
| MSE / L2 | $r^2$ | $-2r$ (grows linearly with error) | Optimal under Gaussian-noise MLE assumption; heavily penalizes large residuals — very sensitive to outliers since error is squared |
| MAE / L1 | $\lvert r\rvert$ | $-\text{sign}(r)$ (constant magnitude) | Optimal under Laplace-noise MLE assumption; robust to outliers, but non-differentiable at $r=0$ and gradient doesn't shrink near the optimum (can cause oscillation late in training) |
| Huber | $\begin{cases}\frac12 r^2 & \lvert r\rvert\le\delta\\ \delta(\lvert r\rvert-\frac12\delta) & \lvert r\rvert>\delta\end{cases}$ | $-r$ inside $\delta$, $-\delta\,\text{sign}(r)$ outside | Quadratic (MSE-like) for small residuals, linear (MAE-like) for large ones — combines MSE's smooth gradient near the optimum with MAE's outlier robustness. $\delta$ is a tunable threshold |
| Quantile (pinball) loss | $\begin{cases}\tau\, r & r\ge 0\\ (\tau-1)\, r & r<0\end{cases}$ for quantile $\tau\in(0,1)$ | Asymmetric constant slope | Minimizing it yields the $\tau$-th conditional quantile of $y\mid x$ (not the mean) — used to produce **prediction intervals** (fit $\tau=0.1$ and $\tau=0.9$ for an 80% interval) rather than a single point estimate; $\tau=0.5$ recovers MAE |

```python
from sklearn.linear_model import HuberRegressor, QuantileRegressor
huber = HuberRegressor(epsilon=1.35)          # epsilon is the delta threshold (in units of scaled residual std)
q90 = QuantileRegressor(quantile=0.9, alpha=0.0)  # upper bound of a prediction interval
```

- **Why it matters beyond regression:** the same MSE-vs-MAE-vs-Huber tradeoff reappears in gradient boosting (`loss="huber"` in `GradientBoostingRegressor`), in deep learning regression heads, and in any "robust regression" discussion.
- **Pitfall:** picking MSE by default when the target has heavy-tailed noise or occasional data-entry outliers silently lets a handful of extreme points dominate the fit — Huber or MAE is usually the safer default when outlier contamination is plausible and unverified.

### Interview Questions

1. **Derive the OLS normal equation from the RSS objective.**
   Answer: $J(\beta)=(y-X\beta)^T(y-X\beta)$. Expand and differentiate: $\nabla_\beta J = -2X^Ty + 2X^TX\beta$. Set to 0: $X^TX\beta = X^Ty \Rightarrow \hat\beta=(X^TX)^{-1}X^Ty$, valid when $X^TX$ is invertible (full column rank).

2. **What happens to OLS coefficients under perfect multicollinearity?**
   Answer: $X^TX$ becomes singular (not invertible), so there's no unique solution — infinitely many $\beta$ give the same fitted values. In practice with near-collinearity, $(X^TX)^{-1}$ has extremely large entries, producing unstable, high-variance coefficient estimates with inflated standard errors.

3. **Why does Lasso produce sparse solutions but Ridge does not?**
   Answer: Geometrically, the L1 constraint region is a diamond with corners on the coordinate axes; RSS elliptical contours are likely to intersect at a corner, zeroing out some coefficients. The L2 constraint is a smooth sphere with no corners, so intersections rarely fall exactly on an axis. Algebraically, the L1 penalty's subgradient at 0 spans $[-\lambda,\lambda]$, creating a "dead zone" that holds coefficients at exactly zero; L2's gradient at 0 is 0, providing no such force.

4. **When would you prefer Elastic Net over Lasso?**
   Answer: When features are highly correlated — pure Lasso tends to arbitrarily select one feature from a correlated group and zero out the rest (unstable selection across resamples), while Elastic Net's L2 component encourages correlated features to have similar coefficients (grouping effect) while still allowing sparsity from the L1 component.

5. **Explain the bias-variance tradeoff using polynomial regression as an example.**
   Answer: Low-degree polynomials underfit (high bias — can't capture the true curvature — but low variance across resamples). High-degree polynomials overfit (low bias, fit training points almost exactly, but high variance — small changes in training data produce wildly different fitted curves). Total expected error = bias² + variance + irreducible noise; the optimal degree minimizes this sum, typically identified via cross-validation, not training error.

6. **What is heteroscedasticity, how do you detect it, and why does it matter?**
   Answer: Non-constant variance of residuals across the range of fitted values/predictors. Detect via residuals-vs-fitted plots (funnel shape) or statistical tests (Breusch-Pagan, White test). It doesn't bias OLS coefficients but violates the Gauss-Markov assumption, so OLS is no longer BLUE — standard errors and hypothesis tests become invalid. Fixes: weighted least squares, robust (heteroscedasticity-consistent) standard errors, log-transform the target.

7. **Compare normal equation and gradient descent for solving linear regression.**
   Answer: Normal equation gives an exact closed-form solution in one step but costs $O(d^3)$ for matrix inversion — infeasible for very high-dimensional data ($d$ in the thousands+) or when $X^TX$ is singular. Gradient descent is iterative, $O(nd)$ per step, scales to huge datasets/dimensions (especially with SGD/mini-batches), but requires tuning learning rate and iterations, and needs feature scaling for good conditioning/convergence speed.

8. **Why must you scale features before applying Ridge or Lasso, but not plain OLS?**
   Answer: OLS predictions/fit are scale-invariant (rescaling a feature just rescales its coefficient inversely). But regularization penalizes the *magnitude* of coefficients directly and uses the same $\lambda$ for all coefficients — an unscaled feature with large numeric range will get an artificially small coefficient and be under-penalized (or vice versa for small-range features), biasing which features get shrunk/zeroed.

9. **What does VIF measure and what's a rule of thumb threshold?**
   Answer: VIF for feature $j$ is $1/(1-R_j^2)$ where $R_j^2$ comes from regressing $X_j$ on all other predictors. It quantifies how much the variance of $\hat\beta_j$ is inflated due to collinearity with other features. VIF > 5 is a caution sign, VIF > 10 typically flags problematic multicollinearity.

10. **Derive why Ridge regression's closed-form solution always exists even when $X^TX$ is singular.**
    Answer: Ridge solves $\hat\beta=(X^TX+\lambda I)^{-1}X^Ty$ for $\lambda>0$. $X^TX$ is positive semi-definite (eigenvalues ≥ 0); adding $\lambda I$ shifts every eigenvalue up by $\lambda>0$, making all eigenvalues strictly positive, hence $X^TX+\lambda I$ is positive definite and invertible regardless of collinearity in $X$.

11. **Scenario: you fit linear regression and $R^2$ is high, but residual plots show a clear U-shape. What's wrong and what do you do?**
    Answer: The U-shape indicates a non-linear relationship the model isn't capturing (violates linearity assumption) — high $R^2$ can coexist with a systematic misspecification if the non-linear component is small relative to overall variance explained. Fix: add polynomial/interaction terms, transform variables (e.g., log), or use a non-linear model (tree-based, splines, GAM).

12. **What is the difference between $R^2$ and adjusted $R^2$, and why does it matter for model comparison?**
    Answer: $R^2 = 1 - SS_{res}/SS_{tot}$ always increases (or stays the same) as you add features, even useless ones, because it never penalizes complexity. Adjusted $R^2 = 1-(1-R^2)\frac{n-1}{n-p-1}$ penalizes for the number of predictors $p$, so it can decrease if an added feature doesn't improve fit enough to justify the added complexity — better for comparing models with different numbers of features.

13. **How would you detect and handle outliers before fitting a linear regression?**
    Answer: Detect via studentized residuals, Cook's distance (influence), leverage (hat values), or simple z-score/IQR checks on raw variables. Handle by investigating if they're data errors (correct/remove), using robust regression (Huber, RANSAC) instead of OLS (which is sensitive to squared-loss outlier amplification), or winsorizing/transforming.

14. **Why is Lasso solved via coordinate descent rather than a closed form?**
    Answer: The L1 penalty $|\beta_j|$ is non-differentiable at $\beta_j=0$, so there's no closed-form gradient=0 solution. Coordinate descent optimizes one coefficient at a time (holding others fixed), which has a simple closed-form soft-thresholding update at each step and provably converges for this convex problem.

15. **In production, your linear regression model's coefficients change sign between retraining runs on similar data. What's the likely cause and fix?**
    Answer: Likely multicollinearity — near-singular $X^TX$ makes coefficient estimates highly sensitive to small data perturbations, and different retraining samples can flip signs even though predictions stay similar. Fix: check VIF, apply Ridge/Elastic Net for stability, or drop/combine redundant features. Emphasize that if the goal is prediction accuracy (not coefficient interpretation), this instability may not matter much — but it's dangerous if coefficients are used for business decisions.

16. **Why is MSE the "default" loss for linear regression, and under what distributional assumption is it the maximum-likelihood choice?**
    Answer: Minimizing MSE is equivalent to maximum likelihood estimation under the assumption that residuals are i.i.d. Gaussian with constant variance: the Gaussian log-likelihood's only term depending on $\hat y$ is $-\frac{1}{2\sigma^2}\sum (y-\hat y)^2$, so maximizing likelihood = minimizing $\sum(y-\hat y)^2$ = minimizing MSE. It's "default" because it's differentiable everywhere (clean closed-form/gradient-based optimization) and the Gaussian-noise assumption is a reasonable default for many real-valued measurement errors.

17. **Your regression target has occasional extreme outliers (data-entry errors you can't fully filter out) — why might Huber loss beat both plain MSE and plain MAE here?**
    Answer: Plain MSE squares the residual, so a handful of extreme outliers can dominate the total loss and pull the fitted line toward them. Plain MAE is robust to outliers but has a constant-magnitude gradient everywhere (including near zero), which can cause the optimizer to oscillate around the minimum instead of settling smoothly. Huber loss gets the best of both: quadratic (MSE-like, smoothly differentiable) behavior for small residuals so it converges cleanly near the optimum, but linear (MAE-like) behavior beyond the threshold $\delta$ so a few extreme outliers contribute bounded, not squared, loss.

18. **How does quantile (pinball) loss differ conceptually from MSE/MAE, and what does minimizing it actually estimate?**
    Answer: MSE minimization estimates the conditional *mean* $E[y\mid x]$; MAE minimization estimates the conditional *median*. Quantile loss is asymmetric — it penalizes over-predictions and under-predictions differently depending on $\tau$ — and minimizing it for a given $\tau$ recovers the conditional $\tau$-th *quantile* of $y\mid x$. This is why it's used to build prediction intervals (e.g., fit $\tau=0.05$ and $\tau=0.95$ models to get a 90% interval) rather than a single point forecast, which matters whenever stakeholders need calibrated uncertainty bounds, not just a point estimate.

---

## Supervised Learning — Classification

### Logistic Regression

**Concept.** Models the probability of a binary outcome using the sigmoid (logistic) function applied to a linear combination of features:
$$P(y=1\mid x) = \sigma(w^Tx+b) = \frac{1}{1+e^{-(w^Tx+b)}}$$

**Log-odds (logit) interpretation.** Rearranging: $\ln\frac{P(y=1)}{1-P(y=1)} = w^Tx+b$. So logistic regression is a *linear model of log-odds* — each coefficient $w_j$ represents the change in log-odds per unit increase in $x_j$; $e^{w_j}$ is the multiplicative change in odds.

**MLE derivation of the loss (log loss / binary cross-entropy).** Given labels $y_i \in \{0,1\}$ and $p_i = \sigma(w^Tx_i)$, the likelihood assuming Bernoulli-distributed labels is:
$$L(w) = \prod_i p_i^{y_i}(1-p_i)^{1-y_i}$$
Log-likelihood:
$$\ell(w) = \sum_i \big[y_i\ln p_i + (1-y_i)\ln(1-p_i)\big]$$
Maximizing $\ell(w)$ is equivalent to minimizing the negative log-likelihood (cross-entropy loss):
$$J(w) = -\sum_i \big[y_i\ln p_i + (1-y_i)\ln(1-p_i)\big]$$
This is **convex** in $w$ (unlike squared error would be, given the sigmoid nonlinearity), so gradient-based methods reliably converge to the global optimum. The gradient has an elegant closed form:
$$\nabla_w J = \sum_i (p_i - y_i)x_i = X^T(p-y)$$
Note the strong resemblance to OLS's gradient — this is a general property of GLMs with canonical link functions.

**No closed-form solution** — solved iteratively via gradient descent, Newton's method, or IRLS (Iteratively Reweighted Least Squares).

**Decision boundary.** Since $P(y=1|x)=0.5 \iff w^Tx+b=0$, the decision boundary is **linear** (a hyperplane) in feature space — logistic regression is fundamentally a linear classifier, even though the probability output is nonlinear (sigmoid-shaped).

**Multiclass extensions:**

| Approach | Mechanism |
|---|---|
| One-vs-Rest (OvR) | Train $K$ independent binary classifiers, each distinguishing class $k$ vs all others; predict class with highest score. Simple, parallelizable, but probabilities don't naturally sum to 1. |
| Softmax (multinomial) regression | Single joint model: $P(y=k\mid x) = \frac{e^{w_k^Tx}}{\sum_{j=1}^K e^{w_j^Tx}}$. Loss = categorical cross-entropy. Naturally produces a valid probability distribution over all classes; jointly optimized. |

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", multi_class="multinomial")
model.fit(X_train, y_train)
```

**Hyperparameters:** `C` (inverse of regularization strength λ — smaller C = stronger regularization), `penalty` (l1/l2/elasticnet — l1 gives sparse/interpretable coefficients), `class_weight` (handle imbalance), `solver` (`liblinear` for small/l1, `lbfgs`/`newton-cg` for l2/multinomial, `saga` for large-scale/elasticnet).

**Strengths/weaknesses.** Fast, probabilistic outputs, highly interpretable (odds ratios), well-calibrated by default. Cannot capture non-linear boundaries without manual feature engineering (interactions, polynomial terms), sensitive to outliers in a moderate way, assumes independent observations.

### The Perceptron

**Concept.** The perceptron (Rosenblatt, 1958) is the historical ancestor of both logistic regression and linear SVM — the earliest algorithm to learn a linear decision boundary directly from mistakes, using a simple **online, mistake-driven update rule** rather than an explicit probabilistic or margin-based objective.

**Prediction rule.** $\hat y = \text{sign}(w^Tx+b) \in \{-1,+1\}$ — a hard threshold, not a probability (unlike logistic regression's sigmoid).

**Learning rule.** For each training example, only update weights if the current model misclassifies it:
$$\text{if } y_i(w^Tx_i+b) \le 0: \quad w \leftarrow w + \eta\, y_i x_i,\qquad b \leftarrow b + \eta\, y_i$$
Correctly classified points cause no update at all — the perceptron only "learns from mistakes."

**Perceptron criterion (implicit loss).** This update rule is exactly gradient descent on the perceptron loss $L(w) = \max(0, -y(w^Tx+b))$, summed over misclassified points only. Compare to hinge loss $\max(0, 1-y(w^Tx+b))$ (SVM) — the perceptron loss has no margin term, so it stops updating as soon as a point crosses to the correct side of the boundary with zero margin, whereas SVM keeps pushing until a margin of 1 is achieved. This is the key mathematical difference between "any separating hyperplane" (perceptron) and "the maximum-margin separating hyperplane" (SVM).

**Convergence (Novikoff's theorem).** If the training data is linearly separable with margin $\gamma$ (distance from the closest point to the true separating hyperplane) and all points satisfy $\|x_i\|\le R$, the perceptron algorithm makes at most $(R/\gamma)^2$ mistakes before converging to a separating hyperplane — a finite bound with no dependence on the number of training points or dimensionality. If the data is **not** linearly separable, the vanilla perceptron never converges (it will cycle indefinitely) — this was the historical motivation for margin-based methods (SVM) and, when the XOR problem was famously identified as not linearly separable, contributed to the "AI winter" until multi-layer perceptrons with backpropagation (see [Deep Learning](08_Deep_Learning.md)) resolved it by stacking learned non-linear feature transforms.

```python
from sklearn.linear_model import Perceptron
clf = Perceptron(eta0=1.0, max_iter=1000, tol=1e-3)
clf.fit(X_train, y_train)
```

**Comparison — Perceptron vs Logistic Regression vs Linear SVM:**

| | Perceptron | Logistic Regression | Linear SVM |
|---|---|---|---|
| Output | Hard label (sign) | Calibrated probability | Hard label (+ optional Platt scaling) |
| Loss | Perceptron loss (0 for any correct side) | Log loss (never exactly 0) | Hinge loss (0 once margin ≥ 1) |
| Boundary found | *Any* separating hyperplane | Max-likelihood boundary | *Maximum-margin* separating hyperplane |
| Convergence on separable data | Finite mistake bound, but boundary not unique | Converges to unique MLE optimum (convex) | Converges to unique max-margin optimum (convex) |
| Behavior on non-separable data | Never converges (oscillates) | Converges (log loss always finite) | Converges (soft margin via slack $\xi_i$) |

**Strengths/weaknesses.** Extremely simple, cheap, online/streaming-friendly (one point at a time, no batch requirement). No probability output, no margin robustness, doesn't converge on non-separable data without modifications (e.g., pocket algorithm, averaging), superseded in practice by logistic regression/SVM — its main modern relevance is pedagogical (foundation of neural network units) and historical.

### Naive Bayes

**Concept.** A generative probabilistic classifier based on Bayes' theorem with a strong ("naive") conditional independence assumption between features given the class:
$$P(y\mid x_1,\ldots,x_n) \propto P(y)\prod_{i=1}^n P(x_i\mid y)$$
Predict $\hat y = \arg\max_y P(y)\prod_i P(x_i|y)$.

**Why "naive" works well in practice despite the unrealistic independence assumption:**
- Classification only needs the correct **argmax**, not correct probability estimates — even if the individual $P(x_i|y)$ estimates are biased due to violated independence, the biases often affect all classes similarly enough that ranking (argmax) stays correct.
- With limited training data, a simple model with few parameters (just per-feature-per-class stats) has much lower variance than trying to model the full joint distribution — a favorable bias-variance tradeoff.
- Correlated redundant features get "double counted," which in practice often just amplifies genuinely informative signal rather than causing catastrophic error.

**Variants:**

| Variant | Feature type | Likelihood model |
|---|---|---|
| Gaussian NB | Continuous | $P(x_i|y)\sim N(\mu_{y,i},\sigma_{y,i}^2)$ |
| Multinomial NB | Discrete counts (e.g., word counts) | Multinomial distribution over counts |
| Bernoulli NB | Binary (e.g., word present/absent) | Bernoulli distribution per feature |
| Complement NB | Imbalanced text data | Uses complement class stats, more stable for skewed class distributions |

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
gnb = GaussianNB()
mnb = MultinomialNB(alpha=1.0)   # alpha = Laplace/Lidstone smoothing
```

**Laplace smoothing.** Prevents zero probabilities for unseen feature values in training (which would zero out the entire product): $P(x_i|y) = \frac{count(x_i,y)+\alpha}{count(y)+\alpha \cdot |V|}$.

**Strengths/weaknesses.** Extremely fast to train (closed-form counting, no iterative optimization), works well with small data and high-dimensional sparse data (text/NLP classic baseline), naturally multiclass. Weak when features are strongly correlated in a way that matters, poor probability calibration (though argmax/ranking is often fine), Gaussian NB assumes normality per feature per class.

### K-Nearest Neighbors (KNN)

**Concept.** A non-parametric, instance-based ("lazy") method: to classify a new point, find the $k$ closest training points (by some distance metric) and take a majority vote (classification) or average (regression).

**Distance metrics:**

| Metric | Formula | Use case |
|---|---|---|
| Euclidean (L2) | $\sqrt{\sum(x_i-y_i)^2}$ | Continuous, isotropic features |
| Manhattan (L1) | $\sum|x_i-y_i|$ | Robust to outliers, high-dim sparse |
| Minkowski | $(\sum|x_i-y_i|^p)^{1/p}$ | Generalizes L1/L2 (p=1,2) |
| Cosine similarity/distance | $1 - \frac{x\cdot y}{\|x\|\|y\|}$ | Text/embeddings, direction matters more than magnitude |
| Hamming | count of differing positions | Categorical/binary features |
| Mahalanobis | $\sqrt{(x-y)^T\Sigma^{-1}(x-y)}$ | Accounts for feature covariance/correlation |

**Curse of dimensionality.** As dimensionality $d$ grows, the volume of a hypersphere concentrates near its surface, and pairwise distances between random points converge — the ratio of distance to nearest vs farthest neighbor approaches 1, so "nearest" loses meaning. Practically: KNN degrades badly beyond ~10-20 features without dimensionality reduction (PCA) or feature selection.

**Choosing k.**
- Small $k$ (e.g., 1) → low bias, high variance (very sensitive to noise, jagged decision boundary, overfits).
- Large $k$ → high bias, low variance (smooth boundary, may underfit, washes out local structure — and at $k=n$ it degenerates to just predicting the majority class always).
- Use cross-validation over a range of odd $k$ values (odd avoids ties in binary classification) to pick the sweet spot.

**Weighted KNN.** Instead of equal-vote among the $k$ neighbors, weight votes by inverse distance ($w_i = 1/d_i$), so closer neighbors have more influence — often more robust than uniform voting, especially with moderate/large $k$.

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="minkowski", p=2)
```

**Complexity.** Training: $O(1)$ (just store data — "lazy learner"). Prediction (brute force): $O(n \cdot d)$ per query. Structures like KD-trees/Ball-trees improve to $O(\log n \cdot d)$ in low dimensions but degrade to brute force in high dimensions.

**Strengths/weaknesses.** Simple, no training phase, naturally multiclass, adapts to complex/non-linear boundaries with enough data. Slow at inference for large datasets, memory-heavy (stores all training data), very sensitive to feature scaling and irrelevant features, suffers badly from curse of dimensionality.

### Decision Trees

**Concept.** Recursively partition the feature space into axis-aligned regions by choosing, at each node, the feature and threshold that best separates the classes/reduces variance, forming a tree of if-then rules.

**Splitting criteria:**

| Criterion | Formula | Task | Notes |
|---|---|---|---|
| Gini impurity | $Gini = 1-\sum_k p_k^2$ | Classification | Measures probability two random samples from the node have different classes; range $[0, 1-1/K]$ |
| Entropy | $H=-\sum_k p_k\log_2 p_k$ | Classification | Information-theoretic; slightly more computationally expensive (log) |
| Information Gain | $IG = H(parent) - \sum_{children}\frac{n_c}{n}H(child)$ | Classification | Reduction in entropy from a split |
| Variance reduction (MSE) | $\sum(y_i-\bar y)^2$ reduction | Regression | Split minimizes weighted sum of child variances |

Gini and entropy usually produce very similar trees in practice; Gini is slightly faster (no log) and is sklearn's default for `CART`.

**Greedy, recursive algorithm (CART):** at each node, exhaustively try every feature and every candidate threshold, pick the split maximizing impurity reduction, recurse on children until a stopping criterion (max depth, min samples, pure node) is met.

**Overfitting behavior.** Fully grown trees (grown until leaves are pure) essentially memorize the training data — each leaf can end up describing a single training point, giving 0 training error but poor generalization (high variance). This is *the* canonical example of overfitting via excessive model complexity in interviews.

**Pruning:**
- **Pre-pruning (early stopping):** limit `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes` during growth.
- **Post-pruning (cost-complexity/minimal cost-complexity pruning, CCP):** grow full tree, then prune back nodes whose removal doesn't increase the cost-complexity metric $R_\alpha(T) = R(T) + \alpha|T|$ (error + penalty on number of leaves) beyond a threshold $\alpha$; sklearn exposes this via `ccp_alpha`.

```python
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_leaf=10, ccp_alpha=0.001)
tree.fit(X_train, y_train)
```

**Strengths/weaknesses.** Highly interpretable (visualizable rules), handles mixed data types and non-linear relationships/interactions natively, requires no feature scaling, robust to outliers in features. High variance/instability (small data changes → very different tree), prone to overfitting without pruning, biased toward features with more levels/splits, axis-aligned splits struggle with diagonal decision boundaries (needs many splits to approximate).

**Complexity.** Training: $O(n \cdot d \cdot \log n)$ typical (sorting features at each level). Prediction: $O(\log n)$ (tree depth) per sample.

### Support Vector Machines (SVM)

**Concept.** Finds the hyperplane that maximizes the margin (distance) between the two classes' closest points, giving the classifier maximum robustness to new data.

**Margin maximization — primal formulation.** For linearly separable data with labels $y_i \in \{-1,+1\}$: find hyperplane $w^Tx+b=0$ such that $y_i(w^Tx_i+b)\ge 1$ for all $i$, maximizing the margin $2/\|w\|$. Equivalent to minimizing:
$$\min_w \frac{1}{2}\|w\|^2 \quad \text{s.t. } y_i(w^Tx_i+b)\ge 1 \,\forall i$$

**Soft margin / hinge loss (for non-separable data).** Introduce slack variables $\xi_i \ge 0$ allowing some misclassification:
$$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C\sum_i \xi_i \quad \text{s.t. } y_i(w^Tx_i+b)\ge 1-\xi_i,\ \xi_i\ge 0$$
Equivalently, unconstrained hinge-loss form: $\min_w \frac{1}{2}\|w\|^2 + C\sum_i \max(0, 1-y_i(w^Tx_i+b))$. The hinge loss $\max(0,1-y\hat y)$ is zero once a point is correctly classified with margin ≥ 1, and grows linearly for violations — unlike log loss, it doesn't keep pushing already-correct, confidently-classified points, which is why SVM solutions are **sparse** in support vectors.

**C parameter (regularization).**
- Large $C$: heavily penalizes margin violations → narrow margin, fits training data closely → low bias, high variance (can overfit).
- Small $C$: tolerates more violations → wide margin, smoother boundary → higher bias, lower variance (more regularized).

**Dual formulation & kernel trick.** Using Lagrangian duality, the optimization becomes:
$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_iy_j\, x_i^Tx_j \quad \text{s.t. } 0\le\alpha_i\le C,\ \sum_i\alpha_i y_i=0$$
The key insight: the dual objective depends on training data **only through dot products** $x_i^Tx_j$. This means we can replace $x_i^Tx_j$ with a **kernel function** $K(x_i,x_j) = \phi(x_i)^T\phi(x_j)$ that computes the dot product in some (possibly infinite-dimensional) transformed feature space $\phi$, without ever explicitly computing $\phi(x)$ — the "kernel trick." Points with $\alpha_i > 0$ are the **support vectors** — only these determine the decision boundary, giving SVM sparse, memory-efficient predictions.

**Common kernels:**

| Kernel | Formula | Effect |
|---|---|---|
| Linear | $x_i^Tx_j$ | Original SVM, linear boundary |
| Polynomial | $(\gamma x_i^Tx_j + r)^d$ | Curved boundaries, captures feature interactions up to degree $d$ |
| RBF (Gaussian) | $\exp(-\gamma\|x_i-x_j\|^2)$ | Infinite-dimensional feature space, very flexible, most popular default |
| Sigmoid | $\tanh(\gamma x_i^Tx_j + r)$ | Behaves like a neural net layer, less commonly used |

**RBF `gamma` parameter:** controls the "reach" of a single training point's influence — high gamma → tight, wiggly boundary around individual points (risk of overfitting); low gamma → smoother, more global boundary (risk of underfitting). Analogous role to $1/\sigma^2$ (inverse variance) of the Gaussian kernel.

```python
from sklearn.svm import SVC
svm_rbf = SVC(kernel="rbf", C=1.0, gamma="scale")
svm_lin = SVC(kernel="linear", C=1.0)
```

**Complexity.** Training: roughly $O(n^2)$ to $O(n^3)$ depending on solver (SMO) — doesn't scale well beyond tens of thousands of samples without approximations (`LinearSVC`, `SGDClassifier` with hinge loss, kernel approximations like Nyström/random features). Prediction: $O(s \cdot d)$ where $s$ = number of support vectors.

**Strengths/weaknesses.** Effective in high dimensions (even $d > n$), robust to overfitting with proper $C$/kernel choice, memory-efficient (sparse support vectors), works well with clear margin of separation. Slow to train on large datasets, sensitive to kernel/hyperparameter choice, no direct probability output (needs Platt scaling), less interpretable than trees/linear models, requires feature scaling.

### Linear & Quadratic Discriminant Analysis (LDA / QDA)

**Concept.** LDA and QDA are **generative** classifiers (like Naive Bayes): instead of directly modeling $P(y|x)$, they model each class's feature distribution $P(x|y=k)$ as multivariate Gaussian, then apply Bayes' rule to classify. This is distinct from — and not to be confused with — **Latent Dirichlet Allocation** (also abbreviated LDA), an unrelated unsupervised topic-modeling algorithm covered in [NLP](09_NLP.md).

**Assumption.** $x\mid y=k \sim \mathcal{N}(\mu_k, \Sigma_k)$. **LDA** assumes all classes share the *same* covariance matrix $\Sigma_k=\Sigma$; **QDA** allows each class its own $\Sigma_k$.

**Discriminant functions (from Bayes' rule, dropping the shared normalizing constant).**

LDA (shared $\Sigma$):
$$\delta_k(x) = x^T\Sigma^{-1}\mu_k - \tfrac12\mu_k^T\Sigma^{-1}\mu_k + \ln\pi_k$$
This is **linear** in $x$ — the quadratic term $x^T\Sigma^{-1}x$ cancels when comparing any two classes' discriminant functions because $\Sigma$ is shared, leaving a linear decision boundary (hence "linear" discriminant analysis).

QDA (class-specific $\Sigma_k$):
$$\delta_k(x) = -\tfrac12\ln|\Sigma_k| - \tfrac12(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k) + \ln\pi_k$$
The $x^T\Sigma_k^{-1}x$ term no longer cancels across classes, giving a **quadratic** decision boundary — more flexible, but many more parameters to estimate ($K$ separate $d\times d$ covariance matrices instead of 1 shared one).

Predict $\hat y = \arg\max_k \delta_k(x)$.

**LDA as dimensionality reduction (Fisher's LDA).** Beyond classification, LDA also gives a *supervised* dimensionality-reduction technique: find the projection direction(s) $w$ that maximize between-class separation relative to within-class scatter:
$$J(w) = \frac{w^TS_Bw}{w^TS_Ww}, \qquad S_B=\sum_k n_k(\mu_k-\bar\mu)(\mu_k-\bar\mu)^T,\quad S_W=\sum_k\sum_{i\in k}(x_i-\mu_k)(x_i-\mu_k)^T$$
The optimal $w$ is the top eigenvector(s) of $S_W^{-1}S_B$. Unlike **PCA** (unsupervised, maximizes total variance regardless of class labels), LDA explicitly uses class labels to find the projection that best *separates* classes — for a $K$-class problem, LDA can produce at most $K-1$ discriminant directions (since $S_B$ has rank $\le K-1$), which is why it's commonly used as a supervised pre-processing/visualization step (e.g., project to 2D for a 3-class problem) as well as a standalone classifier.

**LDA vs QDA vs Logistic Regression:**

| | LDA | QDA | Logistic Regression |
|---|---|---|---|
| Model type | Generative ($P(x\mid y)$) | Generative ($P(x\mid y)$) | Discriminative ($P(y\mid x)$ directly) |
| Boundary shape | Linear | Quadratic | Linear |
| Covariance assumption | Shared across classes | Class-specific | None (no distributional assumption on $x$) |
| Parameters to estimate | $O(Kd + d^2)$ | $O(Kd + Kd^2)$ | $O(Kd)$ |
| Data efficiency | Good with limited data (fewer params) | Needs more data per class (full $\Sigma_k$ each) | Good, robust to non-Gaussian $x$ |
| Best when | Gaussian-ish classes, similar spread, small-to-moderate $n$ | Gaussian-ish classes, genuinely different spreads, enough data per class | Don't want to assume a feature distribution at all; only need the boundary |
| Sensitive to outliers? | Yes (Gaussian MLE) | Yes, more so (small $\Sigma_k$ per class) | Moderately |

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)   # use as classifier or as a supervised dim-reduction transform
X_lda = lda.fit_transform(X_train, y_train)
qda = QuadraticDiscriminantAnalysis().fit(X_train, y_train)
```

**Pitfalls.** Requires $\Sigma$ (or each $\Sigma_k$) to be invertible — fails or becomes unstable when $d>n$ or features are highly collinear (fix: shrinkage estimator, `shrinkage="auto"` in sklearn, which regularizes $\Sigma$ toward a diagonal/scaled-identity matrix, analogous to Ridge). Both LDA and QDA degrade when the Gaussian assumption is badly violated (e.g., multimodal or heavy-tailed class distributions) — logistic regression or tree-based models are more robust in that case. QDA needs meaningfully more training data than LDA to reliably estimate each class's full covariance matrix, especially in high dimensions.

### Multiclass Classification Strategies: One-vs-Rest, One-vs-One, Native Multiclass

Many classifiers (logistic regression via softmax, decision trees, Naive Bayes, KNN) generalize to $K>2$ classes natively. Others — most notably SVM — are fundamentally *binary* classifiers and need an explicit decomposition strategy to handle multiclass problems. This generalizes the OvR/softmax comparison already introduced under Logistic Regression to any base binary classifier.

**One-vs-Rest (OvR / OvA).** Train $K$ independent binary classifiers, classifier $k$ distinguishing "class $k$" vs "everything else." Predict the class whose classifier gives the highest confidence score. Simple and parallelizable, but: (a) each binary sub-problem is inherently class-imbalanced (1 class vs the other $K-1$ combined), and (b) scores from independently-trained classifiers aren't on a comparable scale, so raw argmax comparison can be unreliable unless scores are well-calibrated.

**One-vs-One (OvO).** Train $\binom{K}{2}$ binary classifiers, one per pair of classes, each trained only on the subset of data belonging to those two classes. Predict via majority vote across all pairwise classifiers (the class winning the most pairwise "duels"). Each individual classifier trains faster (smaller, more balanced sub-datasets), and OvO is often more accurate for SVM specifically since each pairwise problem is simpler/more separable — this is why sklearn's `SVC` uses OvO internally by default for multiclass, while `LinearSVC` uses OvR. The downside is $O(K^2)$ classifiers, which becomes expensive for large $K$.

**Native multiclass.** Some algorithms need no decomposition at all: decision trees/Random Forest/GBM split on impurity criteria (Gini/entropy) that generalize directly to $K$ classes, Naive Bayes takes $\arg\max_k$ over any number of classes, KNN majority-votes over however many classes appear among neighbors, and softmax regression jointly models all $K$ classes' probabilities in one coherent model.

**Comparison:**

| | OvR | OvO | Native multiclass |
|---|---|---|---|
| # classifiers | $K$ | $\binom{K}{2}$ | 1 |
| Training data per classifier | Full dataset (relabeled) | Only the 2 relevant classes | Full dataset |
| Scales to large $K$? | Moderate (linear in $K$) | Poor (quadratic in $K$) | Best |
| Class imbalance per sub-problem | Yes (1 vs rest) | No (balanced pairs) | N/A |
| Typical use | Any binary classifier, default for `LinearSVC`, `SGDClassifier` | Kernel SVM (`SVC` default), when pairwise separability is easier | Trees, Naive Bayes, KNN, softmax regression |

```python
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC

ovr = OneVsRestClassifier(SVC(kernel="rbf")).fit(X_train, y_train)
ovo = OneVsOneClassifier(SVC(kernel="rbf")).fit(X_train, y_train)
```

**Pitfall.** OvR with poorly-calibrated base classifiers can produce ties or nonsensical rankings (e.g., all $K$ "vs rest" classifiers predicting "not me" with low confidence) — wrapping base classifiers with `probability=True`/Platt scaling before comparing scores helps. OvO's majority vote can also produce ties for even $K$; sklearn breaks ties using aggregated decision-function magnitude.

### Interview Questions

1. **Derive the log loss for logistic regression from the maximum likelihood principle.**
   Answer: Assume $y_i \sim \text{Bernoulli}(p_i)$ with $p_i=\sigma(w^Tx_i)$. Likelihood $L(w)=\prod p_i^{y_i}(1-p_i)^{1-y_i}$. Log-likelihood $\ell(w) = \sum [y_i\ln p_i + (1-y_i)\ln(1-p_i)]$. Negative log-likelihood (minimize) = binary cross-entropy loss $J(w) = -\sum[y_i\ln p_i + (1-y_i)\ln(1-p_i)]$, which is convex in $w$ due to sigmoid's properties, guaranteeing a unique global minimum found via gradient descent/Newton's method.

2. **Why is logistic regression a "linear" classifier even though it uses a nonlinear sigmoid function?**
   Answer: The decision boundary is defined by $P(y=1|x)=0.5$, which occurs exactly when the linear argument $w^Tx+b=0$ — a hyperplane. The sigmoid transforms the linear score into a probability, but the boundary itself (and the log-odds) remain linear in $x$.

3. **Compare One-vs-Rest and Softmax (multinomial) approaches for multiclass logistic regression.**
   Answer: OvR trains $K$ independent binary classifiers and picks the highest score — simple, parallelizable, but outputs aren't a coherent joint probability distribution (scores don't sum to 1) and classes are not directly compared against each other during training. Softmax trains one joint model with a shared parameterization and cross-entropy loss over all $K$ classes simultaneously, producing a valid probability distribution and generally better-calibrated multiclass probabilities, at the cost of a slightly more complex optimization.

4. **Explain why Naive Bayes can perform well in practice despite its unrealistic independence assumption.**
   Answer: Classification only requires correct ranking (argmax) of posterior probabilities, not accurate probability values — systematic bias from violated independence often affects competing classes similarly and cancels out in the argmax. Also, the strong assumption drastically reduces the number of parameters to estimate, giving low variance especially valuable with limited training data — a favorable bias-variance tradeoff versus trying to model the full joint feature distribution.

5. **What is the curse of dimensionality and how does it specifically hurt KNN?**
   Answer: As dimensions increase, data becomes sparse and distances between points concentrate — the difference between nearest and farthest neighbor distance shrinks toward zero relative to the average distance, so "nearest" neighbors stop being meaningfully closer than random points. KNN relies entirely on distance-based locality, so in high dimensions it essentially degenerates to random guessing unless dimensionality is reduced or irrelevant features are removed.

6. **How do you choose k in KNN, and what happens at the extremes k=1 and k=n?**
   Answer: Use cross-validation across a range of k values, picking the one minimizing validation error (often plotting error vs k to look for an elbow/minimum). At k=1: zero bias but maximum variance — decision boundary is jagged, extremely sensitive to noise/outliers (nearest neighbor overfits). At k=n: the prediction becomes the global majority class/mean for every point — maximum bias, zero variance, completely ignoring local structure.

7. **Derive/explain Gini impurity and show why a pure node has Gini = 0.**
   Answer: $Gini = 1-\sum_k p_k^2$ where $p_k$ is the class-$k$ proportion in the node. If the node is pure (all points one class, say $p_1=1$, others 0), $Gini = 1-1^2 = 0$. Gini is maximized when classes are maximally mixed (e.g., binary case, $p=0.5$ each, $Gini=1-0.25-0.25=0.5$).

8. **Why do fully-grown, unpruned decision trees overfit, and what are the two main pruning strategies?**
   Answer: A fully grown tree keeps splitting until nodes are pure, effectively creating a leaf for small groups (even single points) of training data — it memorizes noise rather than learning generalizable patterns, giving near-zero training error but high test error (high variance). Pre-pruning stops growth early via constraints (max_depth, min_samples_leaf, min_samples_split). Post-pruning (cost-complexity pruning) grows the full tree then removes subtrees that don't sufficiently improve the penalized error $R(T)+\alpha|T|$.

9. **Derive the SVM margin width formula and explain why minimizing $\|w\|^2$ maximizes the margin.**
   Answer: For hyperplane $w^Tx+b=0$ with support vectors satisfying $y_i(w^Tx_i+b)=1$, the perpendicular distance from a support vector to the hyperplane is $1/\|w\|$, so total margin width (both sides) is $2/\|w\|$. Maximizing $2/\|w\|$ is equivalent to minimizing $\|w\|$, and for convexity/differentiability convenience we minimize $\frac{1}{2}\|w\|^2$ instead — same minimizer, easier optimization (quadratic program).

10. **What is the kernel trick, and why is it computationally valuable?**
    Answer: The SVM dual formulation depends on the data only through pairwise dot products $x_i^Tx_j$. The kernel trick replaces this dot product with a kernel function $K(x_i,x_j)=\phi(x_i)^T\phi(x_j)$ that implicitly computes the dot product in a higher (even infinite) dimensional feature space $\phi$, without ever explicitly constructing $\phi(x)$. This lets SVM find non-linear boundaries in original space while only paying the computational cost of evaluating $K$, not the cost of operating in the high-dimensional space directly.

11. **Explain the effect of the C hyperparameter in soft-margin SVM, including extremes.**
    Answer: C controls the trade-off between margin width and misclassification penalty in $\min \frac{1}{2}\|w\|^2 + C\sum\xi_i$. Large C → heavily penalize violations → narrow margin, closely fits training data (risk overfitting, low bias/high variance). Small C → tolerate more violations → wide margin, smoother boundary (risk underfitting, high bias/low variance). As $C\to\infty$, approaches hard-margin SVM (no tolerance for misclassification); as $C\to0$, the margin term dominates and the model essentially ignores training errors.

12. **Compare Decision Trees and SVM for a tabular classification problem with mixed categorical/numeric features and a business need for interpretability.**
    Answer: Decision Trees handle mixed data types natively (no scaling/encoding needed for splits, though sklearn's implementation requires numeric encoding), produce human-readable if-then rules, and naturally model feature interactions and non-linear boundaries — ideal when interpretability matters. SVM requires numeric, scaled features, offers no built-in interpretability (kernel SVMs are black boxes), but often achieves better generalization/margin robustness on cleaner numeric data with proper kernel/tuning. Given the stated business need for interpretability, Decision Trees (or a more robust ensemble like Random Forest, paired with SHAP) would generally be preferred.

13. **Why does hinge loss lead to sparse support vectors while logistic loss does not produce an analogous sparsity?**
    Answer: Hinge loss $\max(0, 1-y\hat y)$ is exactly zero for any point correctly classified with margin ≥ 1 — such points contribute nothing to the loss gradient, so their dual coefficients $\alpha_i=0$ (not support vectors). Logistic (log) loss is never exactly zero for any finite score — it asymptotically approaches zero but always contributes a small gradient, so all points continue to (slightly) influence the logistic regression solution, giving no analogous sparsity.

14. **Scenario: your Gaussian Naive Bayes classifier performs poorly on a dataset where two features are near-perfectly correlated (e.g., height in cm and height in inches). Why, and how would you fix it?**
    Answer: The conditional independence assumption is badly violated — the model effectively "double counts" the same information, over-weighting that signal relative to other features, which can skew posterior probabilities and hurt calibration/accuracy, especially if other truly independent features are diluted. Fix: drop one of the redundant features, or use PCA/feature engineering to decorrelate before applying Naive Bayes, or switch to a model that doesn't assume independence (logistic regression, tree-based).

15. **Scenario: You need a model for a spam-filtering system with millions of emails and a strict 5 ms latency budget per prediction. Rank Naive Bayes, KNN, and kernel SVM for this use case and justify.**
    Answer: Naive Bayes is the best fit — training is a single pass of counting (very fast, can retrain frequently on streaming data), and prediction is $O(d)$ (just a sum over feature log-probabilities), easily meeting tight latency budgets, and it has a long, proven track record for text classification. KNN is poor here — prediction requires scanning/searching a huge training set per query, and with millions of emails/high-dimensional bag-of-words features, this is both slow and hurt by the curse of dimensionality. Kernel SVM is also poor for latency — prediction cost scales with the number of support vectors, and both training and serving are heavier than Naive Bayes; a *linear* SVM would be more competitive but still generally not faster than Naive Bayes at this scale.

16. **Explain the perceptron convergence theorem, and what happens if you run the perceptron algorithm on data that isn't linearly separable?**
    Answer: If the data is linearly separable with margin $\gamma$ (the distance from the closest point to some true separating hyperplane) and all feature vectors satisfy $\|x_i\|\le R$, Novikoff's theorem guarantees the perceptron makes at most $(R/\gamma)^2$ mistakes (weight updates) before it converges to a hyperplane that separates the data perfectly — a finite bound independent of $n$ or $d$. If the data is not linearly separable, no such hyperplane exists, so the algorithm never stops making mistakes — weights oscillate indefinitely rather than converging, which is why non-separable/noisy data requires either a margin-tolerant method (soft-margin SVM), a probabilistic method (logistic regression), or perceptron variants like the pocket algorithm (keep the best-performing weight vector seen so far).

17. **What is the key mathematical difference between the perceptron's loss and SVM's hinge loss, and what behavioral difference does it cause?**
    Answer: Perceptron loss is $\max(0, -y(w^Tx+b))$ — exactly zero the instant a point is on the correct side of the boundary, regardless of how close it is. Hinge loss is $\max(0, 1-y(w^Tx+b))$ — it remains positive (and keeps contributing gradient) until a point is correctly classified with margin *at least 1*, not just margin 0. This means the perceptron will happily settle on any separating hyperplane, even one that passes uncomfortably close to some points, while SVM's hinge loss keeps pushing the boundary until it maximizes the margin, producing a solution that generalizes better to unseen points near the boundary.

18. **Derive why LDA's decision boundary is linear while QDA's is quadratic.**
    Answer: Both derive discriminant functions from Bayes' rule with Gaussian class-conditional densities: $\delta_k(x) = -\frac12\ln|\Sigma_k| - \frac12(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k) + \ln\pi_k$. Expanding the quadratic form gives a term $-\frac12x^T\Sigma_k^{-1}x$ that depends on $x$ quadratically. In QDA, each class has its own $\Sigma_k$, so this quadratic-in-$x$ term differs between classes and survives when you compare $\delta_j(x)-\delta_k(x)$, producing a quadratic decision boundary. In LDA, all classes share the same $\Sigma$, so the $-\frac12x^T\Sigma^{-1}x$ term is identical across every class's discriminant function and exactly cancels when comparing any two classes — only the linear-in-$x$ terms ($x^T\Sigma^{-1}\mu_k$) and constants remain, giving a linear boundary.

19. **How does LDA used as a dimensionality-reduction technique differ from PCA, and what's the maximum number of components each can produce?**
    Answer: PCA is unsupervised — it finds directions that maximize total variance in the data, completely ignoring class labels, and can produce up to $\min(n,d)$ components. LDA (Fisher's LDA) is supervised — it uses class labels to find directions maximizing the ratio of between-class scatter to within-class scatter ($w^TS_Bw / w^TS_Ww$), explicitly optimizing for class separability rather than raw variance. Because the between-class scatter matrix $S_B$ has rank at most $K-1$ for $K$ classes, LDA can produce at most $K-1$ meaningful discriminant directions, regardless of $d$ — e.g., a 3-class problem can be losslessly (for separability purposes) projected to at most 2 dimensions.

20. **When would you prefer QDA over LDA, and what's the main risk of doing so?**
    Answer: Prefer QDA when you have good reason to believe classes genuinely have different covariance structures (e.g., one class is tightly clustered, another is diffusely spread) and you have enough data per class to estimate each full covariance matrix reliably. The main risk is that QDA has far more parameters ($K$ separate $d\times d$ covariance matrices vs. LDA's single shared one) — with limited data per class or high dimensionality, these covariance estimates become noisy/unstable, and QDA's added flexibility turns into overfitting (high variance) rather than a genuine accuracy gain; in that regime LDA's shared-covariance assumption acts as a regularizer.

21. **Compare One-vs-Rest and One-vs-One for multiclass SVM, and explain why sklearn's `SVC` defaults to OvO while `LinearSVC` defaults to OvR.**
    Answer: OvR trains $K$ binary classifiers on the full (relabeled) dataset each; OvO trains $\binom{K}{2}$ classifiers, each on only the two relevant classes' data. For kernel SVM specifically, each pairwise problem in OvO tends to be simpler/more cleanly separable than a "one class vs. all others combined" problem, and each classifier trains on a smaller subset — since kernel SVM training scales roughly $O(n^2)$-$O(n^3)$, many small pairwise problems can actually be cheaper and more accurate than $K$ large one-vs-rest problems, which is why `SVC` defaults to OvO. `LinearSVC` uses a different, much more scalable linear solver where OvR's $O(K)$ classifiers trained on the full data is already cheap, so there's less incentive to pay OvO's $O(K^2)$ classifier-count cost.

22. **Scenario: you're building a 50-class product-categorization classifier using kernel SVM as the base model, and training time is becoming a bottleneck. How would multiclass strategy choice affect this, and what would you try?**
    Answer: With $K=50$, OvO requires $\binom{50}{2}=1225$ pairwise classifiers — even though each is cheap individually (small, balanced sub-datasets), the sheer count can dominate wall-clock time, especially with per-classifier overhead. OvR requires only 50 classifiers but each trains on the full (highly imbalanced, 1-vs-49) dataset, which is slower per classifier for kernel SVM. Practical fixes: switch to a linear SVM (`LinearSVC`, OvR by default, scales far better with $n$) if the classes are roughly linearly separable in the given feature space, use a native multiclass model instead (softmax logistic regression, gradient boosting) to avoid the decomposition overhead entirely, or use kernel approximations (Nystroem/random Fourier features) to make a linear solver approximate an RBF kernel at much lower cost.

23. **A colleague argues Naive Bayes and LDA are "basically the same idea." What's actually shared between them, and what's the key structural difference?**
    Answer: Both are generative classifiers — they model $P(x\mid y)$ and apply Bayes' rule to get $P(y\mid x)$, rather than modeling $P(y\mid x)$ directly (unlike logistic regression). The key structural difference is the independence/covariance assumption: Naive Bayes assumes each feature is conditionally independent given the class (effectively a diagonal covariance with no cross-feature terms, and any per-feature distribution — Gaussian, multinomial, Bernoulli), while LDA assumes a full multivariate Gaussian per class with a *shared* (but not diagonal) covariance matrix across classes, explicitly modeling feature correlations. Gaussian Naive Bayes is, in fact, a restricted special case of LDA/QDA where the shared or class-specific covariance matrix is forced to be diagonal.

---

## Ensemble Methods

### Bagging: Random Forest

**Concept.** **Bagging (Bootstrap Aggregating)** trains many base models independently on bootstrap resamples (sampling with replacement) of the training data, then averages (regression) or majority-votes (classification) their predictions. **Random Forest** = bagging applied to decision trees, with an added twist: at each split, only a random subset of features (`max_features`) is considered, decorrelating the trees further.

**Why bagging reduces variance.** For $B$ i.i.d. models each with variance $\sigma^2$, the variance of their average is $\sigma^2/B$ — averaging cancels out uncorrelated errors. Real trees aren't fully independent (they're trained on overlapping bootstrap samples of the same data), so with pairwise correlation $\rho$, variance of the average is:
$$\text{Var}(\bar f) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$
As $B\to\infty$, the second term vanishes but the $\rho\sigma^2$ term remains — this is exactly why Random Forest's random feature subsampling matters: it **reduces $\rho$** (decorrelates trees) beyond what bagging alone achieves, further lowering variance. Individual trees are deliberately grown deep (low bias, high variance) because the ensemble's averaging step handles the variance reduction — this is the opposite tuning philosophy from boosting.

**Out-of-Bag (OOB) error.** Each bootstrap sample leaves out ~37% of data on average ($\lim (1-1/n)^n = e^{-1} \approx 0.368$). Each tree can be evaluated on its own held-out (OOB) points, and aggregating these gives a free, nearly unbiased estimate of test error — no separate validation set required.

**Feature importance — Gini (impurity-based) vs Permutation:**

| Method | How it works | Pros | Cons |
|---|---|---|---|
| Gini/impurity-based (MDI) | Sum of impurity decrease each feature contributes across all trees/splits, weighted by number of samples reaching that node | Free byproduct of training, fast | Biased toward high-cardinality/continuous features; computed on training data (can be misleading, doesn't reflect true predictive value on unseen data) |
| Permutation importance | Shuffle one feature's values in validation data, measure drop in model performance; repeat per feature | Model-agnostic, reflects actual predictive contribution on held-out data, not biased by cardinality | Computationally expensive (retrain-free but requires many prediction passes); can be misleading with highly correlated features (importance gets split/diluted) |

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

rf = RandomForestClassifier(n_estimators=500, max_features="sqrt", oob_score=True, n_jobs=-1)
rf.fit(X_train, y_train)
print("OOB score:", rf.oob_score_)
print("Gini importances:", rf.feature_importances_)

perm = permutation_importance(rf, X_val, y_val, n_repeats=10, random_state=0)
print("Permutation importances:", perm.importances_mean)
```

**Hyperparameters:** `n_estimators` (more trees → lower variance, diminishing returns, more compute — rarely hurts accuracy, just cost), `max_depth`/`min_samples_leaf` (control individual tree complexity — usually left deep in RF, unlike boosting), `max_features` (size of random feature subset per split — smaller = more decorrelation/variance reduction but potentially higher bias; sqrt(d) classification, d/3 regression are common defaults), `bootstrap` (True for standard RF).

**Strengths/weaknesses.** Robust to overfitting (relative to single trees), handles non-linear relationships and interactions, minimal preprocessing needed, parallelizable (trees are independent), built-in OOB validation and feature importance. Less interpretable than a single tree, can still overfit with very noisy data/too few decorrelating constraints, biased feature importance without permutation correction, generally underperforms well-tuned boosting on structured/tabular competitions, larger memory footprint (stores many full trees).

### Boosting: AdaBoost, Gradient Boosting Machines

**Core idea of boosting.** Unlike bagging (parallel, independent, variance-focused), boosting builds models **sequentially**, where each new model focuses on correcting the errors of the ensemble so far — primarily a **bias-reduction** technique (though it also affects variance).

**AdaBoost step-by-step:**
1. Initialize equal sample weights $w_i = 1/n$.
2. For $t=1\ldots T$: train a weak learner (often a depth-1 "stump") on weighted data; compute its weighted error rate $\epsilon_t$; compute learner weight $\alpha_t = \frac{1}{2}\ln\frac{1-\epsilon_t}{\epsilon_t}$ (higher for more accurate learners).
3. Update sample weights: increase weight on misclassified points, decrease on correctly classified ones: $w_i \leftarrow w_i \exp(\alpha_t \cdot \mathbb{1}[y_i \neq \hat y_i])$, then renormalize.
4. Final prediction: weighted vote $\hat y = \text{sign}\left(\sum_t \alpha_t h_t(x)\right)$.

AdaBoost can be shown to minimize an exponential loss function via forward stagewise additive modeling.

**Gradient Boosting Machines (GBM) — general framework.** Generalizes boosting to arbitrary differentiable loss functions via gradient descent *in function space*:
1. Initialize $F_0(x) = $ constant minimizing the loss (e.g., mean for MSE).
2. For $m=1\ldots M$: compute the **pseudo-residuals** $r_i = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$ (for squared-error loss, these are just the ordinary residuals $y_i - F_{m-1}(x_i)$).
3. Fit a weak learner (typically a shallow regression tree) $h_m(x)$ to predict these pseudo-residuals.
4. Update: $F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)$, where $\nu$ is the **learning rate/shrinkage**.
5. Repeat for $M$ rounds; final model $F_M(x) = F_0 + \nu\sum_{m=1}^M h_m(x)$.

Each new tree is literally fit to the *gradient* of the loss with respect to current predictions — hence "gradient boosting." Small $\nu$ (e.g., 0.01–0.1) combined with more trees generally generalizes better than large $\nu$ with fewer trees (this is the boosting analog of L2 regularization — "shrinkage").

```python
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
gbm = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8)
ada = AdaBoostClassifier(n_estimators=200, learning_rate=1.0)
```

**Hyperparameters (GBM family):** `n_estimators` (more rounds → lower bias but risk of overfitting, especially with high learning rate), `learning_rate` (shrinkage — smaller needs more trees but generalizes better; classic tradeoff with `n_estimators`), `max_depth` (per-tree complexity — usually shallow, 3-8, since boosting adds complexity across rounds), `subsample` (stochastic gradient boosting — row subsampling per round reduces variance/overfitting, adds randomness like bagging), `min_samples_leaf`/`min_child_weight` (regularize individual trees).

**XGBoost / LightGBM / CatBoost — differences:**

| Feature | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| Tree growth | Level-wise (depth-wise) by default (also supports leaf-wise) | Leaf-wise (best-first) — grows the leaf with max loss reduction, deeper/asymmetric trees | Symmetric (oblivious) trees — same split condition across an entire level |
| Split finding | Exact greedy or histogram-based (`tree_method=hist`) | Histogram-based by default — bins continuous features, much faster | Histogram-based, plus greedy/ordered boosting-specific optimizations |
| Categorical handling | Requires manual encoding (one-hot/label) unless using newer categorical support | Native categorical support via optimal partitioning (Fisher-based grouping) | Best-in-class native handling — ordered target statistics/ordered boosting avoids target leakage from naive mean-encoding |
| Regularization | L1 (`alpha`) and L2 (`lambda`) on leaf weights, plus min_child_weight, gamma (min split loss) | Similar L1/L2, plus leaf-wise growth needs `max_depth`/`num_leaves` constraints to avoid overfitting | Strong built-in regularization via ordered boosting (reduces target leakage/overfitting bias inherent in greedy boosting) |
| Speed on large data | Fast, but generally slower than LightGBM on very large/high-cardinality data | Fastest typically — smaller memory, faster training via histogram + leaf-wise growth + GOSS/EFB | Slower training than LightGBM typically, but often needs less hyperparameter tuning |
| Overfitting risk | Moderate | Higher (leaf-wise can overfit on small data — control via `num_leaves`, `min_data_in_leaf`) | Lower by design (ordered boosting combats prediction shift/target leakage) |
| Best for | General-purpose, mature ecosystem, strong community | Large datasets, many categorical/high-cardinality features, speed-critical | Datasets heavy in categorical features, wanting good defaults with less tuning |

Key algorithmic notes:
- **Histogram-based splits**: instead of evaluating every possible split point (sorting continuous features, $O(n)$ candidates), bucket values into a fixed number of bins (e.g., 255) and only evaluate bin boundaries — turns split-finding from $O(n \cdot d)$ to $O(\text{bins} \cdot d)$, a massive speedup with minimal accuracy loss.
- **GOSS (Gradient-based One-Side Sampling, LightGBM)**: keeps all instances with large gradients (poorly predicted, more informative) and randomly samples instances with small gradients, reducing data size per iteration without losing much accuracy.
- **EFB (Exclusive Feature Bundling, LightGBM)**: bundles mutually exclusive sparse features (never nonzero simultaneously, common after one-hot encoding) into a single feature, reducing effective dimensionality.
- **Ordered boosting (CatBoost)**: standard gradient boosting reuses the same data to both compute residuals and fit the next tree, causing subtle target leakage ("prediction shift"). CatBoost computes residuals using models trained on a random permutation *excluding* the current point, closer to a leave-one-out estimate, reducing overfitting bias.

### Stacking and Blending

**Stacking.** Train several diverse base models (level-0), then train a **meta-model** (level-1) on the base models' out-of-fold predictions (to avoid leakage) as features, learning how to optimally combine them. Typically uses K-fold cross-validation to generate out-of-fold predictions for the meta-model's training set, then retrains base models on full data for final predictions.

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

stack = StackingClassifier(
    estimators=[("rf", RandomForestClassifier()), ("svc", SVC(probability=True))],
    final_estimator=LogisticRegression(),
    cv=5)
```

**Blending.** A simpler variant: split training data into two parts — train base models on part A, generate predictions on held-out part B, then train the meta-model on part B's predictions only (no cross-validation folds). Faster and simpler than stacking, but uses less data for the meta-model (higher variance in the meta-model's training) and depends heavily on the size/representativeness of the holdout.

**When to use.** Both work best when base models are diverse (different algorithm families, e.g., tree-based + linear + KNN) so their errors are uncorrelated — stacking/blending a bunch of similar models yields little benefit. Common in competitions (Kaggle) for squeezing out final performance gains; less common in production due to added complexity, latency (must run all base models), and maintenance burden.

### Comparison: Bagging vs Boosting, Random Forest vs Gradient Boosting

| | Bagging (Random Forest) | Boosting (GBM/XGBoost/LightGBM) |
|---|---|---|
| Model building | Parallel, independent base learners | Sequential, each corrects prior errors |
| Primary effect | Reduces variance | Reduces bias (also can reduce variance via shrinkage/subsampling) |
| Base learner complexity | Deep/complex trees (low bias, high variance individually) | Shallow/weak trees (high bias, low variance individually) |
| Overfitting risk | Lower, more robust out-of-the-box | Higher, needs careful tuning (learning rate, depth, early stopping) |
| Training speed | Fast, parallelizable across trees | Slower, inherently sequential (though histogram methods help) |
| Typical accuracy (tabular) | Strong, reliable baseline | Usually higher with proper tuning — dominates most tabular ML competitions |
| Sensitivity to hyperparameters | Low — fairly robust defaults | High — learning rate/depth/n_estimators interact strongly |
| Handles noisy data/outliers | More robust (averaging dampens outlier influence) | More sensitive (sequential correction can chase noisy residuals/overfit outliers) |
| Interpretability of ensemble | Feature importance readily available | Feature importance available; SHAP works very well with trees (TreeSHAP) |
| When to choose | Quick strong baseline, limited tuning time/expertise, want robustness and easy parallelization, less risk of overfitting | Maximum predictive performance on structured/tabular data, willing to invest in tuning/early stopping/monitoring for overfitting |

### Interview Questions

1. **Explain mathematically why bagging reduces variance and why Random Forest's random feature selection helps beyond plain bagging.**
   Answer: For $B$ models with variance $\sigma^2$ and pairwise correlation $\rho$, the ensemble average has variance $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$. Simply increasing $B$ only kills the second term — the $\rho\sigma^2$ floor remains. Random feature subsampling at each split decorrelates the trees (lowers $\rho$) because different trees are forced to consider different features, so they make different (less correlated) errors, lowering the irreducible $\rho\sigma^2$ term beyond what bagging alone achieves.

2. **What is Out-of-Bag error and why is it useful?**
   Answer: Each bootstrap sample excludes ~36.8% of the data ($e^{-1}$) on average; those excluded points are "out-of-bag" for that tree. Evaluating each tree only on its own OOB points and aggregating gives a nearly unbiased estimate of test error without needing a separate validation split — essentially free cross-validation baked into the bagging procedure.

3. **Compare Gini/impurity-based feature importance with permutation importance. Which would you trust more and why?**
   Answer: Impurity-based (MDI) importance sums impurity reduction attributable to a feature across all splits/trees on training data — it's fast (free byproduct) but biased toward high-cardinality/continuous features (they get more candidate split points, inflating apparent importance) and doesn't reflect out-of-sample predictive value. Permutation importance measures the actual drop in validation performance when a feature's values are shuffled — model-agnostic and reflects real predictive contribution, though it's more expensive and can underestimate importance for groups of correlated features (shuffling one still leaves correlated proxies available). Generally, permutation importance (on held-out data) is more trustworthy for feature selection/reporting decisions.

4. **Walk through the AdaBoost algorithm step by step.**
   Answer: (1) Initialize equal weights on all training samples. (2) Train a weak learner on weighted data. (3) Compute the learner's weighted error and its vote-weight $\alpha_t=\frac12\ln\frac{1-\epsilon_t}{\epsilon_t}$ — more accurate learners get higher weight. (4) Increase the weights of misclassified samples and decrease weights of correctly classified ones, so the next weak learner focuses more on hard cases. (5) Repeat for $T$ rounds. (6) Final prediction is the sign of the weighted sum of all weak learners' votes.

5. **Derive/explain the core update rule of Gradient Boosting for squared-error loss, and explain what "pseudo-residual" means for a general loss.**
   Answer: For squared error $L=\frac12(y-F(x))^2$, the negative gradient w.r.t. $F(x)$ is exactly $y-F(x)$, the ordinary residual — so gradient boosting with squared loss literally fits each new tree to the residuals of the current ensemble, then adds $\nu \cdot h_m(x)$ to the running prediction. For an arbitrary differentiable loss (e.g., log loss for classification), the "pseudo-residual" is the negative gradient of the loss w.r.t. the current prediction, evaluated at each training point — it generalizes "residual" to any loss function, and gradient boosting is literally gradient descent performed in function space, one weak learner per step.

6. **Why is the learning rate (shrinkage) in gradient boosting important, and how does it interact with n_estimators?**
   Answer: Learning rate $\nu$ scales each new tree's contribution: $F_m = F_{m-1} + \nu h_m$. Small $\nu$ makes each step conservative, requiring more rounds ($n\_estimators$) to fit the data fully, but tends to generalize better (analogous to L2 shrinkage — many small corrective steps average out noise better than a few large ones). There's a direct trade-off: lowering $\nu$ typically requires proportionally increasing `n_estimators` to reach similar training fit, but the pair (low $\nu$, high $n$) usually generalizes better than (high $\nu$, low $n$), at the cost of more training time. Early stopping on a validation set is used to find the right number of rounds for a given learning rate.

7. **Compare bagging and boosting on bias-variance grounds, and explain why their base learners are tuned oppositely (deep trees for RF, shallow trees for GBM).**
   Answer: Bagging reduces variance by averaging many independent, low-bias/high-variance base learners — so RF deliberately uses deep, complex trees (each individually overfits) because averaging cancels the variance. Boosting reduces bias by sequentially correcting errors — it uses shallow, high-bias/low-variance weak learners (e.g., depth 3-6 trees, or even stumps) because the sequential additive process itself builds up complexity/reduces bias over rounds; using deep trees in boosting would risk fitting noise directly at each step with no averaging to counteract it, causing rapid overfitting.

8. **How does LightGBM's leaf-wise tree growth differ from XGBoost's (default) level-wise growth, and what's the trade-off?**
   Answer: Level-wise (depth-wise) growth expands all nodes at the current depth before going deeper, producing balanced, symmetric trees. Leaf-wise growth always splits the single leaf that yields the greatest loss reduction, regardless of depth/balance, producing deeper, asymmetric trees that can achieve lower loss with fewer splits/faster convergence. The trade-off: leaf-wise trees can overfit more easily, especially on smaller datasets, so LightGBM requires careful tuning of `max_depth`/`num_leaves`/`min_data_in_leaf` to control this.

9. **Explain CatBoost's "ordered boosting" and why it addresses a subtle overfitting problem in standard gradient boosting.**
   Answer: Standard GBM computes residuals for training point $i$ using a model that was itself trained partly on point $i$'s target — a mild target leakage causing a systematic bias called "prediction shift," where the ensemble is slightly overconfident on training data in a way that doesn't generalize. Ordered boosting computes the residual for each point using only a model trained on points that precede it in a random permutation (similar to leave-one-out), avoiding this leakage and producing a less biased, better-generalizing model — a big reason CatBoost often needs less tuning to avoid overfitting.

10. **What is histogram-based split finding and why does it dramatically speed up gradient boosting training?**
    Answer: Exact greedy split finding sorts each continuous feature and evaluates every unique value as a candidate threshold — $O(n)$ candidates per feature per node. Histogram-based methods discretize each feature into a fixed number of bins (e.g., 255) up front, then only evaluate bin boundaries as candidate splits — reducing candidates to $O(\text{bins})$ regardless of $n$, and enabling efficient cumulative-sum computation of gradient statistics per bin. This turns split-finding cost from roughly $O(n \cdot d)$ to $O(\text{bins}\cdot d)$, with negligible loss in split quality for most datasets.

11. **How does CatBoost handle categorical variables natively, and why is this better than naive one-hot or mean target encoding?**
    Answer: CatBoost converts categorical values to numeric statistics based on the target (similar to mean/target encoding), but computes these statistics using only prior examples in a random permutation order (ordered target statistics) rather than the full dataset — avoiding target leakage that occurs when a category's encoding is computed using its own label. Naive one-hot encoding explodes dimensionality for high-cardinality categoricals; naive mean-target-encoding leaks target information into training features. CatBoost's ordered approach gets the predictive benefit of target-aware encoding while controlling leakage/overfitting.

12. **Scenario: your Random Forest has near-perfect training accuracy but mediocre test accuracy. Is this "overfitting" in the same sense as an overfit single decision tree? What would you check/tune?**
    Answer: It's a related but different phenomenon — individual trees in RF *are* meant to overfit (deep, low-bias, high-variance), and that's usually fine because averaging over many decorrelated trees controls the ensemble's variance. If the overall forest still overfits, likely causes include: too few trees (`n_estimators` too low to average out variance), `max_features` too high (trees too correlated, not decorrelated enough), individual trees still too deep/complex relative to a small/noisy dataset (raise `min_samples_leaf`), or genuine target leakage/small-n high-d situation. Check OOB score vs test score gap, tune `max_features` down, increase `min_samples_leaf`, add more trees, and verify no data leakage.

13. **Why might you choose stacking over simply picking the single best-performing base model?**
    Answer: Stacking can capture complementary strengths of diverse models — e.g., a linear model may excel in certain regions of feature space while a tree-based model excels in others; a meta-learner can learn to weight/combine their predictions conditionally, often outperforming any single base model, especially when base model errors are weakly correlated. The cost is added complexity, latency (must score multiple models), and higher risk of overfitting the meta-model if not using proper out-of-fold predictions for its training data.

14. **Random Forest vs Gradient Boosting: you have a tabular dataset with 50k rows, mild noise, and a tight 2-week project timeline with limited hyperparameter-tuning bandwidth. Which would you pick as a first model, and why?**
    Answer: Random Forest — it's far more robust to default/near-default hyperparameters, less prone to overfitting on noisy data without extensive tuning, trains quickly in parallel, and provides a strong, reliable baseline with minimal tuning effort. Gradient boosting (especially XGBoost/LightGBM) can likely beat it with proper tuning of learning rate, depth, regularization, and early stopping, but under a tight timeline with limited tuning bandwidth, boosting risks either underperforming (undertrained/untuned) or overfitting (default settings pushed too far), so RF is the safer first choice; boosting can be attempted afterward if time permits.

15. **Explain what GOSS and EFB do in LightGBM and why they matter for very large datasets.**
    Answer: GOSS (Gradient-based One-Side Sampling) keeps all training instances with large gradients (currently poorly predicted, most informative for the next tree) and randomly subsamples instances with small gradients (already well-predicted, less informative), reducing the number of data points used per boosting round without much accuracy loss — speeding up training on large datasets. EFB (Exclusive Feature Bundling) identifies sparse features that are (almost) never simultaneously nonzero (common after one-hot encoding categorical variables) and bundles them into a single compound feature, reducing effective dimensionality and histogram-building cost. Together they let LightGBM train much faster than naive gradient boosting on large, high-dimensional, sparse datasets.

---

## Unsupervised Learning

### K-Means Clustering

**Concept.** Partition $n$ points into $k$ clusters by minimizing within-cluster sum of squares (WCSS/inertia):
$$J = \sum_{j=1}^k \sum_{x_i \in C_j} \|x_i - \mu_j\|^2$$

**Algorithm (Lloyd's algorithm):**
1. Initialize $k$ centroids (randomly, or via k-means++).
2. **Assignment step:** assign each point to the nearest centroid.
3. **Update step:** recompute each centroid as the mean of points assigned to it.
4. Repeat steps 2-3 until assignments stop changing (convergence — guaranteed since WCSS monotonically decreases and is bounded below, though only to a *local* optimum).

**K-means++ initialization.** Random initialization can lead to poor local optima (bad luck placing initial centroids). K-means++ picks the first centroid randomly, then picks each subsequent centroid with probability proportional to squared distance from the nearest already-chosen centroid — spreading initial centroids apart, dramatically improving convergence quality and speed versus naive random init. This is sklearn's default (`init="k-means++"`).

**Choosing k:**
- **Elbow method:** plot WCSS (inertia) vs $k$; look for the "elbow" where marginal improvement drops sharply — subjective but useful visual heuristic.
- **Silhouette score:** for each point, $s_i = \frac{b_i - a_i}{\max(a_i,b_i)}$ where $a_i$ = mean intra-cluster distance, $b_i$ = mean distance to nearest other cluster. Ranges $[-1,1]$; higher is better (well-separated, cohesive clusters). Average across all points gives a single score per $k$ — pick $k$ maximizing it.

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

inertias, sil_scores = [], []
for k in range(2, 10):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=0).fit(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, km.labels_))
```

**Limitations.** Assumes spherical, similarly-sized, isotropic clusters (Euclidean distance) — fails on elongated, non-convex, or highly imbalanced-size/density clusters. Sensitive to initialization (mitigated by k-means++ / multiple `n_init` runs). Requires specifying $k$ in advance. Sensitive to feature scaling (always standardize first) and outliers (means are pulled by extreme points — k-medoids is more robust). Only handles numeric data natively.

**Complexity.** $O(n \cdot k \cdot d \cdot i)$ where $i$ = number of iterations to converge — scales well, a major reason for its popularity.

### Hierarchical Clustering

**Concept.** Build a tree (dendrogram) of nested clusters without pre-specifying $k$.

**Agglomerative (bottom-up):** start with each point as its own cluster; repeatedly merge the two closest clusters until one cluster remains. Most common approach.
**Divisive (top-down):** start with all points in one cluster; recursively split. Computationally more expensive, less commonly used.

**Linkage methods (define "distance between clusters"):**

| Linkage | Definition | Tendency |
|---|---|---|
| Single | min distance between any pair of points across clusters | Can produce long, "chained" clusters; sensitive to noise |
| Complete | max distance between any pair | Compact, roughly equal-sized clusters; sensitive to outliers |
| Average | mean of all pairwise distances | Balance between single/complete |
| Ward | minimizes increase in total within-cluster variance after merging | Tends to produce compact, similarly-sized clusters; most popular default, analogous to k-means' objective |

**Dendrograms.** A tree diagram showing the merge order and the distance (height) at which each merge occurred. Cutting the dendrogram horizontally at a chosen height yields a specific number of clusters — a major advantage: you can explore multiple granularities of clustering from one fitted structure without refitting.

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering

Z = linkage(X, method="ward")
dendrogram(Z)
clusters = fcluster(Z, t=3, criterion="maxclust")

agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
labels = agg.fit_predict(X)
```

**Strengths/weaknesses.** No need to pre-specify $k$ (explore dendrogram), deterministic (no random initialization issues), produces interpretable hierarchy of relationships. Computationally expensive — $O(n^2 \log n)$ to $O(n^3)$ depending on linkage/implementation, doesn't scale to very large datasets, greedy merges are irreversible (can't undo a bad early merge), sensitive to noise/outliers depending on linkage choice.

### DBSCAN

**Concept.** Density-Based Spatial Clustering of Applications with Noise — groups points that are densely packed together, marking sparse points as noise/outliers, without needing to specify $k$.

**Key definitions:**
- **eps (ε):** neighborhood radius.
- **min_samples:** minimum number of points (including itself) required within ε to be a "core point."
- **Core point:** has ≥ `min_samples` points within ε.
- **Border point:** within ε of a core point but doesn't itself meet `min_samples`.
- **Noise point:** neither core nor border — doesn't belong to any cluster.

**Algorithm:** for each unvisited point, if it's a core point, start a new cluster and recursively add all points density-reachable from it (directly or through chains of core points); if it's not a core point and isn't reachable from any core point, label it as noise.

**Handling noise and arbitrary shapes.** Because clusters are defined by connected regions of sufficient density rather than distance-to-centroid, DBSCAN naturally finds non-convex, arbitrarily shaped clusters (crescents, rings, etc.) that k-means fundamentally cannot, and explicitly labels sparse/isolated points as noise rather than forcing them into the nearest cluster.

**Parameter sensitivity.** Highly sensitive to `eps` and `min_samples`: too small `eps` → most points become noise (over-fragmented clusters); too large `eps` → clusters merge inappropriately (under-fragmented, everything becomes one blob). A common heuristic for `eps`: plot a k-distance graph (distance to the $k$-th nearest neighbor, $k=$`min_samples`, sorted ascending) and look for the "knee." DBSCAN also struggles when clusters have very different densities (a single global `eps` can't fit all of them) — HDBSCAN (hierarchical DBSCAN) addresses this by working across a range of densities.

```python
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
labels = db.labels_   # -1 indicates noise
```

**Strengths/weaknesses.** No need to specify $k$, robust to outliers (explicit noise label), finds arbitrary-shaped clusters, works well when density is roughly uniform. Struggles with varying-density clusters, sensitive to `eps`/`min_samples` choice, poor performance in high dimensions (distance concentration, same curse-of-dimensionality issue as KNN), $O(n\log n)$ with spatial indexing but $O(n^2)$ worst case without.

### Gaussian Mixture Models & the EM Algorithm

**Concept.** Models data as generated from a mixture of $K$ Gaussian distributions:
$$p(x) = \sum_{k=1}^K \pi_k \, \mathcal{N}(x\mid \mu_k, \Sigma_k)$$
where $\pi_k$ are mixing weights ($\sum\pi_k=1$). Unlike k-means (hard assignment), GMM gives **soft, probabilistic** cluster assignments — each point has a probability of belonging to each cluster.

**EM (Expectation-Maximization) algorithm:**
- **E-step:** given current parameters $(\pi_k,\mu_k,\Sigma_k)$, compute the posterior responsibility of each cluster for each point:
$$\gamma_{ik} = P(z_i=k\mid x_i) = \frac{\pi_k\mathcal{N}(x_i\mid\mu_k,\Sigma_k)}{\sum_j \pi_j\mathcal{N}(x_i\mid\mu_j,\Sigma_j)}$$
- **M-step:** given responsibilities, update parameters to maximize expected log-likelihood (weighted MLE):
$$\mu_k = \frac{\sum_i \gamma_{ik}x_i}{\sum_i\gamma_{ik}}, \quad \Sigma_k = \frac{\sum_i\gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^T}{\sum_i\gamma_{ik}}, \quad \pi_k = \frac{1}{n}\sum_i\gamma_{ik}$$
- Repeat E and M steps until log-likelihood converges. Guaranteed to monotonically increase log-likelihood at each step (though only to a local optimum — sensitive to initialization, often initialized via k-means).

**Relationship to k-means.** K-means is a special case of GMM/EM where covariances are assumed spherical and equal across clusters ($\Sigma_k=\sigma^2 I$, $\sigma\to0$), and responsibilities are "hardened" to 0/1 (hard assignment) instead of soft probabilities. GMM generalizes k-means to elliptical clusters of varying size/orientation and provides uncertainty estimates.

**Choosing k for GMM:** use BIC (Bayesian Information Criterion) or AIC, which penalize model complexity — lower is better; unlike log-likelihood alone (which always improves with more components), BIC/AIC help avoid overfitting the number of clusters.

```python
from sklearn.mixture import GaussianMixture
bics = []
for k in range(1, 10):
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=0).fit(X)
    bics.append(gmm.bic(X))
best_k = range(1, 10)[bics.index(min(bics))]
gmm = GaussianMixture(n_components=best_k).fit(X)
probs = gmm.predict_proba(X)  # soft assignments
```

**Strengths/weaknesses.** Soft assignments with uncertainty quantification, models elliptical/correlated clusters (via full covariance), principled probabilistic framework (density estimation, generative sampling). Sensitive to initialization (use k-means init or multiple restarts), can converge to degenerate solutions (a component collapsing onto a single point with $\Sigma\to0$, infinite likelihood — needs regularization), assumes Gaussian-shaped clusters, $O(n\cdot k\cdot d^2)$ or more per iteration for full covariance (expensive in high dimensions).

### Anomaly / Novelty Detection

**Isolation Forest.** Builds random trees that recursively partition data via random feature/threshold splits. Key insight: anomalies are "few and different," so they get **isolated in fewer splits** (shorter average path length from root to leaf) than normal points, which require many splits to separate from the dense majority. Anomaly score is a function of average path length across many random trees — shorter path → more anomalous.

```python
from sklearn.ensemble import IsolationForest
iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=0)
preds = iso.fit_predict(X)  # -1 = anomaly, 1 = normal
```

**One-Class SVM.** Learns a decision boundary (typically via RBF kernel) that encloses the bulk ("normal") of the data in feature space, maximizing the margin between the origin and the data (in the SVDD/One-Class SVM dual formulation) — points falling outside this learned boundary are flagged as novel/anomalous. The `nu` parameter controls the expected fraction of outliers/training errors allowed.

```python
from sklearn.svm import OneClassSVM
ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
preds = ocsvm.fit_predict(X_scaled)
```

**Autoencoder-based detection.** Train a neural network (encoder-decoder) to reconstruct normal data through a compressed bottleneck. Because it's trained only (or predominantly) on normal data, it learns to reconstruct normal patterns well but fails to reconstruct out-of-distribution/anomalous inputs accurately. Anomaly score = reconstruction error (e.g., MSE between input and output); threshold on a validation set of known-normal data (e.g., 95th/99th percentile of reconstruction error).

```python
# Conceptual sketch (Keras-style)
# autoencoder.fit(X_normal, X_normal, epochs=50)
# recon_error = np.mean((X_test - autoencoder.predict(X_test))**2, axis=1)
# anomalies = recon_error > threshold
```

**Comparison:**

| Method | Assumption | Scales to high-dim? | Handles non-linear structure? | Notes |
|---|---|---|---|---|
| Isolation Forest | Anomalies are easier to isolate via random splits | Good | Yes (tree-based, non-parametric) | Fast, no distance computation, handles mixed-scale features reasonably, widely used default |
| One-Class SVM | Normal data occupies a compact region separable via kernel | Moderate (kernel methods scale poorly with n) | Yes (via RBF kernel) | Sensitive to `nu`/`gamma`, slower on large n |
| Autoencoder | Normal data has learnable, compressible structure that anomalies violate | Excellent (deep learning scales), especially for structured/image/sequence data | Yes (arbitrary non-linear via NN) | Needs enough data and compute to train well; overkill for simple tabular anomaly detection |

### Association Rule Mining (Apriori, FP-Growth)

A different flavor of unsupervised learning: instead of finding clusters or reducing dimensions, association rule mining finds **co-occurrence patterns** in transactional/basket data (classic example: "customers who buy bread and butter also buy milk").

**Key metrics** for a rule $X \Rightarrow Y$ (itemset $X$ implies itemset $Y$):

| Metric | Formula | Meaning |
|---|---|---|
| Support | $P(X \cup Y) = \frac{\text{count}(X \cup Y)}{N}$ | How frequently the itemset appears at all — filters out rare, statistically unreliable combinations |
| Confidence | $P(Y\mid X) = \frac{\text{support}(X\cup Y)}{\text{support}(X)}$ | Of the transactions containing $X$, what fraction also contain $Y$ — the rule's "accuracy" |
| Lift | $\frac{P(Y\mid X)}{P(Y)} = \frac{\text{confidence}(X\Rightarrow Y)}{\text{support}(Y)}$ | How much more likely $Y$ is given $X$, versus $Y$'s baseline rate. Lift $>1$ means positive association; lift $=1$ means $X$ and $Y$ are independent (confidence alone can be misleadingly high just because $Y$ is common) |

**Apriori algorithm.** Exploits the *Apriori property* (downward closure): if an itemset is frequent (support ≥ min_support), all of its subsets must also be frequent — equivalently, if any subset is infrequent, the superset cannot be frequent. This lets the algorithm prune the search space level-by-level:
1. Find all frequent 1-itemsets (support ≥ threshold).
2. Generate candidate 2-itemsets only from frequent 1-itemsets, keep the frequent ones.
3. Repeat for $k=3,4,\ldots$, generating candidate $k$-itemsets only from frequent $(k-1)$-itemsets, until no frequent itemsets remain.
4. From the final frequent itemsets, generate rules and filter by minimum confidence/lift.

```python
from mlxtend.frequent_patterns import apriori, association_rules

frequent_itemsets = apriori(basket_df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
rules.sort_values("lift", ascending=False).head()
```

**FP-Growth.** Apriori's repeated full-database scans per level are expensive at scale. FP-Growth instead compresses the database into a prefix tree (FP-tree) built from just two passes, then mines frequent itemsets by recursively traversing conditional sub-trees — no explicit candidate generation step, generally much faster than Apriori on large/dense datasets while finding identical results.

- **When to use:** market-basket analysis, recommendation ("frequently bought together"), cross-sell strategy, web clickstream sequencing, even non-retail uses like finding co-occurring symptoms in medical records or co-occurring genes.
- **Pitfalls:** combinatorial explosion of candidate itemsets with low min_support on high-cardinality item catalogs (mitigate by raising the support threshold or using FP-Growth); high confidence does **not** imply causation or even real correlation — always check lift; rare-but-high-value items (e.g., expensive products bought infrequently) get filtered out by a naive support threshold, requiring tricks like multiple minimum supports.

### Semi-Supervised Learning: Self-Training & Label Propagation

**Motivation.** Fully supervised learning needs labels for every training point; fully unsupervised learning uses none. In practice, labels are often expensive/slow to obtain (medical annotations, manual review) while unlabeled data is abundant — **semi-supervised learning** sits between the two, using a small labeled set plus a much larger unlabeled set to train a better model than the labeled set alone would allow.

**Core assumptions that make this work at all** (without at least one of these holding, unlabeled data provides no information about $y$):
- **Smoothness assumption:** points close together in feature space likely share the same label.
- **Cluster assumption:** points in the same cluster/high-density region likely share the same label; decision boundaries should lie in low-density regions.
- **Manifold assumption:** high-dimensional data actually lies on a lower-dimensional manifold, and labels vary smoothly along that manifold.

**Self-training (pseudo-labeling).**
1. Train a base classifier on the small labeled set.
2. Predict on the unlabeled set; take the predictions the model is most **confident** about (above some probability threshold) and add them to the training set as if they were true labels ("pseudo-labels").
3. Retrain on labeled + pseudo-labeled data; repeat, progressively expanding the labeled set.

```python
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC

base = SVC(probability=True, kernel="rbf")
self_train = SelfTrainingClassifier(base, threshold=0.75)
self_train.fit(X_combined, y_combined)  # y_combined uses -1 for unlabeled points
```

**Pitfall — confirmation bias / error amplification.** If the base classifier is systematically wrong on some subregion of feature space (not just randomly noisy), self-training confidently mislabels that region and then *reinforces its own mistake* by retraining on those wrong pseudo-labels — errors compound over iterations rather than average out. Mitigations: conservative confidence thresholds, capping how many pseudo-labels are added per round, and monitoring validation accuracy (on a held-out truly-labeled set) each iteration to catch drift early.

**Label Propagation / Label Spreading (graph-based).** Build a similarity graph over *all* points (labeled and unlabeled), typically with edge weights from an RBF kernel on feature distance, then iteratively propagate label information along graph edges — nearby points converge toward the same label distribution. Label Propagation clamps labeled points' labels permanently; Label Spreading (a regularized variant) allows labeled points' labels to be slightly adjusted too, making it more robust to noisy labels.

```python
from sklearn.semi_supervised import LabelSpreading
ls = LabelSpreading(kernel="rbf", gamma=20, alpha=0.2)
ls.fit(X_combined, y_combined)   # -1 for unlabeled
```

**Comparison:**

| | Self-Training | Label Propagation/Spreading | Fully-supervised baseline |
|---|---|---|---|
| Mechanism | Iteratively retrain on high-confidence pseudo-labels | Propagate labels through a similarity graph | Train only on labeled points |
| Assumption relied on | Base classifier's confidence is trustworthy | Smoothness/cluster/manifold assumption on the graph | None beyond usual i.i.d. |
| Works with any base classifier? | Yes (wraps any classifier with `predict_proba`) | No — specific graph-based algorithm | N/A |
| Main failure mode | Confirmation bias compounding errors | Poor graph construction (wrong `gamma`/kernel) merges or fragments manifold structure | Simply has less data to learn from |
| Scales to large unlabeled sets? | Yes (bounded by base classifier's cost) | Poor — graph construction/propagation is $O(n^2)$ or worse | N/A |

**When it matters in interviews.** A common follow-up: "you have 500 labeled examples and 50,000 unlabeled ones — what do you do before reaching for deep learning/active learning?" Self-training or label propagation are the classical-ML answers; contrast with **active learning** (query a human to label the *most informative* unlabeled points, rather than trusting the model's own pseudo-labels) as a complementary strategy when a labeling budget exists.

### Interview Questions

1. **Derive/explain the K-means objective function and describe why Lloyd's algorithm is guaranteed to converge but only to a local optimum.**
   Answer: K-means minimizes WCSS $J=\sum_j\sum_{x_i\in C_j}\|x_i-\mu_j\|^2$. The assignment step (assign each point to nearest centroid) can only decrease or maintain $J$ for fixed centroids; the update step (recompute centroids as cluster means) can only decrease or maintain $J$ for fixed assignments (mean minimizes sum of squared distances). Since $J$ is monotonically non-increasing and bounded below by 0, the algorithm converges. However, it only guarantees convergence to *some* stationary point of this non-convex objective — not the global optimum — hence sensitivity to initialization and the practice of multiple random restarts (`n_init`).

2. **What problem does k-means++ solve and how?**
   Answer: Naive random centroid initialization can place initial centroids poorly (e.g., multiple centroids in the same true cluster), leading to slow convergence or convergence to poor local optima. K-means++ selects the first centroid uniformly at random, then each subsequent centroid with probability proportional to its squared distance from the nearest already-selected centroid — biasing selection toward spreading centroids across the data's actual spread, which empirically yields faster convergence and better final clustering quality.

3. **Compare the elbow method and silhouette score for choosing k. Which is more objective?**
   Answer: The elbow method plots inertia (WCSS) vs $k$ and looks for a visual "bend" where adding clusters stops helping much — it's inherently subjective (the elbow isn't always clear) and inertia always decreases with $k$ (down to 0 at $k=n$), so there's no natural stopping rule besides visual judgment. Silhouette score computes, per point, how well-separated and cohesive its cluster assignment is, averaged across all points into a single number per $k$ — you can objectively pick the $k$ that maximizes this score, making it more rigorous, though it's more computationally expensive ($O(n^2)$ pairwise distances) and can favor a small number of well-separated clusters over a more nuanced structure.

4. **Why does k-means struggle with non-convex or unequal-density clusters, and what algorithm would you use instead?**
   Answer: K-means assumes clusters are roughly spherical, similarly sized, and separable by Euclidean distance to a centroid (Voronoi partitioning) — it cannot represent elongated, ring-shaped, or nested clusters because a single centroid+radius can't capture such geometry, and it forces every point into some cluster (can't identify noise). DBSCAN handles arbitrary shapes via density-connectivity and explicitly marks sparse points as noise; hierarchical clustering with non-Ward linkage can also capture non-convex structure via chaining (single linkage) at some risk of noise sensitivity.

5. **Explain Ward linkage in hierarchical clustering and why it's often preferred.**
   Answer: Ward linkage merges the pair of clusters whose merge causes the smallest increase in total within-cluster variance (sum of squared deviations from cluster means) — conceptually similar to the k-means objective. This tends to produce compact, roughly equally-sized clusters and is less prone to the "chaining" problem of single linkage, making it the most commonly used default for agglomerative clustering on continuous numeric data.

6. **Walk through the DBSCAN algorithm and define core, border, and noise points.**
   Answer: For each point, count neighbors within radius `eps`. If a point has ≥ `min_samples` neighbors (including itself), it's a **core point** and starts/extends a cluster, recursively absorbing all points density-reachable through chains of core points. A **border point** lies within `eps` of a core point but doesn't itself have enough neighbors to be core — it joins that cluster but doesn't propagate further. A **noise point** is neither core nor reachable from any core point — it's left unclustered (label -1).

7. **Why does DBSCAN fail when clusters have very different densities, and what's a fix?**
   Answer: DBSCAN uses a single global `eps`/`min_samples` threshold to define "dense enough" — a value tuned for a sparse cluster will merge/over-include points in a dense cluster's surrounding area, while a value tuned for a dense cluster will fragment or entirely miss a sparser cluster (labeling it all as noise). HDBSCAN (Hierarchical DBSCAN) fixes this by building a hierarchy over a range of density thresholds and extracting clusters that are stable across that range, effectively adapting to varying local density without needing one global `eps`.

8. **Derive the E-step and M-step of the EM algorithm for a Gaussian Mixture Model at a high level, and explain the convergence guarantee.**
   Answer: E-step: given current parameters, compute the posterior probability (responsibility) that each Gaussian component generated each point, via Bayes' rule: $\gamma_{ik}\propto \pi_k\mathcal{N}(x_i|\mu_k,\Sigma_k)$. M-step: given these responsibilities (soft cluster memberships), re-estimate each component's mean, covariance, and mixing weight as the responsibility-weighted MLE (weighted mean, weighted covariance, average responsibility). Because each step is derived to maximize (a lower bound on) the log-likelihood given the other step's current values, alternating E and M steps is guaranteed to never decrease the overall log-likelihood — it converges monotonically, though only to a local optimum since the likelihood surface is non-convex (multimodal due to label-switching symmetry and multiple local maxima).

9. **How is k-means a special case of GMM?**
   Answer: If you constrain all GMM components to have equal, spherical, isotropic covariance ($\Sigma_k=\sigma^2I$ for all $k$) and take the limit $\sigma^2\to0$, the soft responsibilities $\gamma_{ik}$ become hard 0/1 assignments (each point deterministically assigned to its nearest-mean component), and the EM updates reduce exactly to k-means' assignment and centroid-update steps. So k-means is EM/GMM with hard assignment and a restrictive covariance structure; GMM generalizes it to soft assignment and elliptical, differently-shaped/sized clusters.

10. **Explain how Isolation Forest detects anomalies without any distance or density calculation.**
    Answer: It builds an ensemble of random trees, each splitting on a randomly chosen feature and threshold at every node (no optimization criterion, purely random). Because anomalies are few and different from the bulk of the data, they tend to get separated ("isolated") into their own leaf after only a few random splits, while normal points — densely packed and similar to many neighbors — require many splits to isolate. The anomaly score is derived from the average path length (depth) to isolate each point across all trees in the forest — shorter average path length indicates higher anomaly likelihood, all without ever explicitly computing pairwise distances or density estimates.

11. **Compare Isolation Forest, One-Class SVM, and Autoencoders for anomaly detection on a large, high-dimensional, mixed-type tabular dataset. Which would you choose and why?**
    Answer: One-Class SVM scales poorly with large $n$ (kernel methods are roughly $O(n^2)$-$O(n^3)$) and struggles with mixed categorical/numeric types without careful preprocessing — not ideal here. Autoencoders can model complex non-linear structure but need substantial clean (mostly-normal) training data, careful architecture/hyperparameter tuning, and more engineering overhead. Isolation Forest is tree-based (handles mixed feature scales/types reasonably, no distance computation), scales roughly $O(n\log n)$, requires minimal tuning, and is the standard first choice for large, mixed-type tabular anomaly detection — often the best cost-to-benefit ratio unless there's a specific reason (e.g., very complex non-linear structure, image/sequence data) to prefer autoencoders.

12. **Why must you standardize features before K-means, DBSCAN, or KNN-based methods, but this matters less for tree-based methods?**
    Answer: Distance-based methods (K-means centroids, DBSCAN's `eps` radius, KNN distances) compute (weighted) sums/norms across features — a feature with a much larger numeric range will dominate the distance calculation regardless of its actual relevance, distorting cluster/neighbor structure. Tree-based methods split on one feature at a time based on threshold comparisons within that feature's own range — the split-finding process is invariant to monotonic rescaling of any individual feature, so scaling doesn't change which splits get chosen.

13. **Scenario: a GMM you fit shows one component's covariance collapsing to near-zero (essentially fitting a single data point) with runaway likelihood. What's happening and how do you fix it?**
    Answer: This is a known degenerate solution of unconstrained GMM/EM — a Gaussian component can shrink its covariance around a single point (or a tiny cluster of nearly-identical points), driving the density (and thus likelihood) toward infinity, which is a singularity in the likelihood surface, not a meaningful solution. Fixes: add covariance regularization (`reg_covar` in sklearn, a small value added to the diagonal), constrain covariance type (e.g., `tied` or `diag` instead of `full`), initialize with k-means (avoiding degenerate starting points), or use a Bayesian GMM (Dirichlet process prior) that naturally penalizes such degeneracies.

14. **Scenario: you need to cluster customer transaction data where the number of natural segments is unknown, cluster shapes are irregular, and roughly 5% of the data are clearly outlier accounts you want flagged rather than force-fit into a cluster. Which unsupervised method(s) would you use and why?**
    Answer: DBSCAN (or HDBSCAN for varying density) is well suited here — it doesn't require specifying $k$ in advance, naturally handles irregularly shaped clusters via density-connectivity, and explicitly labels sparse/outlier points as noise rather than forcing them into the nearest cluster centroid (unlike K-means). If density varies substantially across customer segments, HDBSCAN would be preferred over vanilla DBSCAN. A complementary approach: run Isolation Forest first purely for outlier flagging/removal, then cluster the remaining "normal" population with DBSCAN or K-means if the remaining clusters are roughly convex.

15. **What is the fundamental difference between how K-means and GMM assign points to clusters, and when does that difference matter practically?**
    Answer: K-means performs **hard** assignment — each point belongs to exactly one cluster (its nearest centroid), with no notion of confidence/uncertainty. GMM performs **soft** assignment — each point gets a probability distribution over all clusters (responsibilities), reflecting genuine ambiguity for points near cluster boundaries. This matters practically when: points genuinely straddle multiple segments (e.g., a customer who behaves like two personas) and you want to represent that ambiguity rather than force a single label; you need calibrated cluster-membership probabilities for downstream decision-making; or cluster shapes are elliptical/correlated rather than spherical, where GMM's full covariance can represent this while K-means cannot.

16. **Explain support, confidence, and lift, and why confidence alone can be misleading.**
    Answer: Support = frequency of the itemset in the whole dataset; confidence $P(Y\mid X)$ = accuracy of the rule given $X$ occurred; lift = confidence divided by $Y$'s baseline support. Confidence can be high purely because $Y$ is a very common item overall (e.g., "bread") regardless of $X$ — the rule "buy anything ⇒ buy bread" would have high confidence but lift ≈ 1, meaning $X$ tells you nothing extra about $Y$. Lift is what actually confirms a meaningful association rather than just $Y$'s popularity.

17. **Why does Apriori scan the database multiple times and how does the Apriori (downward-closure) property reduce that cost?**
    Answer: Naively, checking every possible itemset for frequency requires evaluating $2^{|items|}-1$ subsets — intractable for real catalogs. The downward-closure property guarantees that a $k$-itemset can only be frequent if all of its $(k-1)$-subsets are frequent, so Apriori only needs to generate candidate $k$-itemsets from *already-confirmed-frequent* $(k-1)$-itemsets, pruning the vast majority of the search space level by level, at the cost of one database scan per level to count candidate support.

18. **When would you choose FP-Growth over Apriori, and why does it avoid Apriori's main bottleneck?**
    Answer: FP-Growth is preferred on large or dense transactional datasets (many items, long transactions) where Apriori's repeated full-database scans and massive candidate-itemset generation become the bottleneck. FP-Growth builds a compressed FP-tree in two passes and mines frequent itemsets by recursing on conditional trees — no explicit candidate generation step at all — so it scales much better while provably returning the identical set of frequent itemsets as Apriori.

19. **What core assumption must hold for semi-supervised learning to actually benefit from unlabeled data, and why doesn't unlabeled data help without it?**
    Answer: At least one of the smoothness, cluster, or manifold assumptions must hold — i.e., the unlabeled points' positions in feature space must carry real information correlated with the label (nearby points share labels, labels vary smoothly along a low-dimensional structure, or decision boundaries lie in low-density regions). If labels were essentially independent of feature-space position/geometry (assumption violated), then knowing where unlabeled points sit tells you nothing about what their labels would be, so adding them to training can't sharpen the decision boundary — it can only add noise or, worse, actively mislead a self-training loop into confidently wrong pseudo-labels.

20. **Explain the confirmation bias risk in self-training and how you'd detect it in practice.**
    Answer: If the base classifier is systematically (not just randomly) wrong on some region of feature space — e.g., underrepresented subgroup, a feature interaction it can't capture — self-training will pseudo-label that region confidently but incorrectly, then retrain on those wrong labels, which reinforces the same mistake and can even amplify it over iterations, since the model has no independent signal to correct itself. Detect it by holding out a small truly-labeled validation set (never pseudo-labeled) and tracking validation accuracy across self-training rounds — a self-training run that's overfitting to its own mistakes will show validation performance plateauing or *degrading* even as training-set/pseudo-label agreement looks great.

21. **Compare Label Propagation and self-training on the same task: 1,000 labeled and 20,000 unlabeled points, feature space with well-separated manifold structure. Which would you try first and why?**
    Answer: Label Propagation/Spreading is a strong first choice here specifically because the scenario states well-separated manifold structure — the graph-based method directly exploits the manifold/smoothness assumption by propagating labels along the similarity graph, and with clearly separated structure the graph edges will correctly reflect true class boundaries, giving clean propagation with low risk of merging unrelated regions. Self-training is a reasonable fallback but is more sensitive to the base classifier's own biases (confirmation bias) and doesn't explicitly leverage the geometric/manifold structure the way a graph-based method does. In practice, both are cheap enough to try and validate against a held-out labeled subset before committing.

22. **How does self-training relate to (and differ from) active learning as a strategy for making the most of a limited labeling budget?**
    Answer: Both address the same underlying scarcity — limited labels, abundant unlabeled data — but from opposite directions. Self-training trusts the model's *own* most-confident predictions and adds them as pseudo-labels without any human involvement, expanding the training set for free but risking compounding the model's existing blind spots. Active learning instead identifies the unlabeled points the model is *most uncertain* about (or otherwise most informative, e.g., via query-by-committee or expected model change) and routes exactly those to a human annotator, spending a labeling budget where it will improve the model the most, rather than reinforcing what the model already "knows." They're often combined: self-train confidently on easy/confident points while active-learning the hard/ambiguous ones.

---

## Model Interpretability

### Global vs Local Interpretability

**Global interpretability** explains the model's overall behavior across the entire dataset/feature space — e.g., "which features matter most on average," "what's the overall shape of the relationship between feature X and the prediction." Examples: linear regression coefficients, tree structure/feature importances, partial dependence plots.

**Local interpretability** explains a single prediction for a single instance — "why did the model predict *this* output for *this* specific customer." Examples: SHAP values for one row, LIME explanation for one instance, individual conditional expectation (ICE) curves for one point.

**Why the distinction matters in practice.** A model can have low global importance for a feature overall (e.g., age barely matters on average) yet that feature can be the *decisive* factor for a specific individual's prediction (e.g., an unusually young or old applicant) — local methods catch this, global methods can miss it. Regulatory/compliance contexts (e.g., credit denial reasons, GDPR "right to explanation") typically require local explanations for individual decisions, while model validation/monitoring typically relies on global explanations.

### Feature Importance Methods

Already detailed under Ensembles (Gini/impurity vs permutation) — summarized here as a general interpretability toolkit:

| Method | Scope | Model-agnostic? | Key idea |
|---|---|---|---|
| Coefficients (linear/logistic) | Global | No (linear models only) | Magnitude/sign of $\beta_j$ (on standardized features) |
| Impurity-based (MDI) | Global | No (tree-based only) | Sum of impurity decrease attributed to feature across splits |
| Permutation importance | Global (can be computed per-instance too) | Yes | Performance drop when feature values are shuffled |
| Drop-column importance | Global | Yes | Retrain without the feature, measure performance drop (expensive but unambiguous) |
| SHAP | Global (aggregated) and Local | Yes (model-agnostic KernelSHAP; fast TreeSHAP for trees) | Game-theoretic fair attribution of prediction to features |
| LIME | Local | Yes | Local linear surrogate model approximation |

```python
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_val, y_val, n_repeats=30, scoring="accuracy")
importances = result.importances_mean
```

### SHAP Values

**Theoretical foundation — Shapley values from cooperative game theory.** Shapley values (Lloyd Shapley, 1953) answer: "in a cooperative game where a coalition of players jointly produce some payoff, how should the payoff be fairly divided among players based on their individual contributions?" In ML, "players" = features, "payoff" = the difference between the model's prediction and the baseline (expected) prediction.

For a feature $i$, its Shapley value is the average marginal contribution across **all possible orderings/subsets** of the other features:
$$\phi_i = \sum_{S \subseteq F\setminus\{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!}\Big[v(S\cup\{i\}) - v(S)\Big]$$
where $F$ is the full feature set, $S$ ranges over all subsets not containing $i$, and $v(S)$ is the model's expected prediction given only features in $S$ are "known" (others marginalized out). This averages feature $i$'s marginal contribution over every possible order in which features could be "added," ensuring a fair, order-independent attribution.

**Key properties (why SHAP is theoretically well-grounded, unique among attribution methods satisfying these axioms):**
- **Local accuracy (efficiency):** the sum of all feature SHAP values plus the baseline exactly equals the actual model prediction: $f(x) = \phi_0 + \sum_i \phi_i$.
- **Missingness:** a feature not present/used contributes 0.
- **Consistency (monotonicity):** if a model changes so that a feature's marginal contribution increases (weakly) for every possible feature-subset, that feature's SHAP value cannot decrease — enables safe comparison of feature importance *between* models.

**Interpretation.** A positive SHAP value for feature $i$ on instance $x$ means that feature pushed the prediction *above* the baseline (expected value over the dataset) for this instance; negative pushes it below. Summing all SHAP values + baseline reconstructs the exact prediction — this exact decomposition is SHAP's main advantage over LIME (no local accuracy guarantee).

**Practical computation.** Exact Shapley values require evaluating $2^{|F|}$ coalitions — intractable for many features. **TreeSHAP** exploits tree structure to compute exact SHAP values in polynomial time ($O(TLD^2)$ for $T$ trees, $L$ leaves, $D$ depth) for tree ensembles (Random Forest, XGBoost, LightGBM) — this is why SHAP is so popular specifically in the boosting/RF ecosystem. **KernelSHAP** is a model-agnostic approximation using weighted linear regression on sampled coalitions, applicable to any model but slower/approximate.

```python
import shap
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)          # global: feature importance + direction
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])  # local: single prediction
shap.dependence_plot("feature_x", shap_values, X_test)  # feature effect + interaction coloring
```

### LIME (Local Interpretable Model-agnostic Explanations)

**Concept.** Explains a single prediction by approximating the model's *local* decision behavior with an interpretable surrogate model (typically sparse linear regression) fit around that specific instance.

**Algorithm:**
1. Pick the instance $x$ to explain.
2. Generate perturbed samples around $x$ (e.g., randomly zero-out/perturb features, or sample from feature distributions).
3. Get the black-box model's predictions on all perturbed samples.
4. Weight each perturbed sample by its proximity to $x$ (closer perturbations weighted more, via a kernel like exponential distance).
5. Fit an interpretable model (sparse linear regression, e.g., via Lasso for feature selection) on the perturbed samples, weighted, to approximate the black-box's local behavior.
6. The surrogate model's coefficients become the local explanation for $x$.

$$\xi(x) = \arg\min_{g\in G} \mathcal{L}(f, g, \pi_x) + \Omega(g)$$
where $f$ is the black-box model, $g$ is the interpretable surrogate, $\pi_x$ is the proximity weighting kernel around $x$, $\mathcal{L}$ is the weighted fit loss, and $\Omega(g)$ penalizes surrogate complexity (encouraging sparsity/interpretability).

**Key difference from SHAP.** LIME approximates local behavior with a separately-fit simple model (no theoretical fairness guarantee, no guarantee that explanations sum to the actual prediction) and results can be somewhat unstable (different perturbation samples can give different explanations for the same instance). SHAP provides an exact, theoretically-grounded, order-independent attribution that sums exactly to the actual prediction (at least in its exact/TreeSHAP forms). LIME is generally faster and simpler for arbitrary black-box models (images, text) where SHAP's exact computation is intractable and TreeSHAP doesn't apply.

```python
import lime.lime_tabular
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values, feature_names=X_train.columns, class_names=["neg","pos"], mode="classification")
exp = explainer.explain_instance(X_test.iloc[0].values, model.predict_proba, num_features=10)
exp.show_in_notebook()
```

### Partial Dependence Plots (PDP) & ICE Plots

**Partial Dependence Plot (PDP).** Shows the *average* marginal effect of one (or two) features on the predicted outcome, marginalizing out all other features:
$$\hat f_S(x_S) = \frac{1}{n}\sum_{i=1}^n \hat f(x_S, x_{C}^{(i)})$$
where $x_S$ is the feature(s) of interest and $x_C^{(i)}$ are the other features' actual values for instance $i$ (held fixed while $x_S$ is swept across its range). This answers "on average, how does changing this feature affect the prediction, holding the overall population's other features as they naturally occur?"

**ICE (Individual Conditional Expectation) plots.** Same idea but plotted **per individual instance** rather than averaged — shows one line per data point tracing how *that instance's* prediction would change as the feature of interest varies, with all its other features held at their actual values. PDP is literally the pointwise average of all ICE curves.

**Why ICE matters beyond PDP: detecting heterogeneity/interactions.** PDP can be misleading when there are strong interaction effects — e.g., if feature X's effect on the prediction is positive for half the population and negative for the other half, PDP might show a flat (near-zero) average line, hiding the real heterogeneous relationship entirely. ICE plots reveal this by showing the individual curves fan out or diverge — if ICE curves are all roughly parallel to the PDP line, there's little interaction; if they diverge substantially, there's meaningful heterogeneity/interaction with other features that PDP alone would mask.

```python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(model, X_train, features=["feature_x"], kind="average")   # PDP
PartialDependenceDisplay.from_estimator(model, X_train, features=["feature_x"], kind="individual") # ICE
PartialDependenceDisplay.from_estimator(model, X_train, features=["feature_x"], kind="both")        # both overlaid
```

**Limitations.** Assumes features in $x_C$ are independent of $x_S$ (marginalizing by averaging over observed joint data can create unrealistic/impossible feature combinations if features are correlated — e.g., averaging over "house size" while sweeping "number of rooms" can create physically implausible tiny houses with many rooms). Accumulated Local Effects (ALE) plots were developed specifically to address this correlated-feature bias by using conditional (local) differences instead of marginal averaging.

### Interview Questions

1. **Distinguish global and local interpretability with a concrete example where they'd disagree.**
   Answer: Global interpretability summarizes overall feature influence (e.g., "income is the most important feature on average across all customers"). Local interpretability explains one specific prediction. They can "disagree" when a globally unimportant feature (e.g., "zip code" with low average importance) turns out to be the dominant driver for one specific individual's prediction (e.g., an applicant from an unusual zip code triggering a specific decision-tree branch) — global importance averages this rare-but-strong effect away, while a local SHAP/LIME explanation for that individual would reveal it clearly.

2. **Why is impurity-based feature importance considered biased, and biased toward what?**
   Answer: It's computed from training-data impurity reductions across tree splits, summed per feature — this favors continuous or high-cardinality categorical features because they offer more possible split thresholds, giving the tree-building (greedy) algorithm more opportunities to find a spuriously good-looking split on that feature purely by chance, even if it has no true predictive signal in a causal sense. It also reflects training-data-fit importance, not necessarily importance for generalization to unseen data — permutation importance (computed on held-out data) corrects both issues.

3. **Explain the Shapley value formula conceptually — what problem is it solving and why does it require averaging over all feature subsets/orderings?**
   Answer: It solves the "fair credit allocation" problem: given a coalition (all features) jointly producing a prediction (versus a baseline), how do you split credit for the prediction among individual features when their effects interact and aren't simply additive? Because a feature's marginal contribution can depend on which other features are already "in the coalition" (interaction effects), Shapley values average the feature's marginal contribution across every possible order of adding features, ensuring the attribution reflects the feature's contribution across all possible contexts, not just one arbitrary ordering — this symmetric averaging is what gives Shapley values their fairness guarantees (efficiency, symmetry, additivity).

4. **What does "local accuracy" mean for SHAP values and why is it a meaningful advantage over methods like raw impurity importance or LIME?**
   Answer: Local accuracy (efficiency) means the sum of all SHAP feature attributions plus the baseline expected value exactly reconstructs the model's actual prediction for that instance: $f(x)=\phi_0+\sum_i\phi_i$. This gives a hard mathematical guarantee that the explanation is a complete, exact decomposition of the prediction — nothing is left "unexplained." Impurity importance has no such per-instance decomposition (it's a training-aggregate statistic). LIME's local linear surrogate is only an *approximation* of the black box locally and has no guarantee that its coefficients sum to the actual prediction.

5. **Explain how TreeSHAP achieves polynomial-time exact SHAP computation when the general Shapley value formula requires exponential ($2^{|F|}$) time.**
   Answer: TreeSHAP exploits the tree's recursive partition structure — instead of enumerating all $2^{|F|}$ feature coalitions explicitly, it tracks, for each path from root to leaf, which features were used for splits and recursively computes each feature's expected contribution by conditioning on the tree's actual branching structure rather than brute-force marginalization. This reduces computation to polynomial time in the number of leaves, trees, and tree depth ($O(TLD^2)$), making exact Shapley-value computation tractable specifically for tree ensembles, which is why SHAP is the default interpretability tool for XGBoost/LightGBM/Random Forest.

6. **Compare SHAP and LIME along theoretical guarantees, speed, and typical use cases.**
   Answer: SHAP has strong theoretical guarantees (efficiency/local accuracy, consistency, symmetry, derived from cooperative game theory) and, via TreeSHAP, can be computed exactly and fast for tree ensembles — the standard choice when using tree-based models. LIME has no such guarantees (approximate local surrogate, can be unstable across repeated runs due to random perturbation sampling), but is more flexible/generalizable to arbitrary black-box models (images, text, any model type) where exact/fast SHAP computation isn't available, at the cost of KernelSHAP's slower general-purpose sampling. In practice: use TreeSHAP for tree ensembles, KernelSHAP or LIME for other model types (deep nets, SVMs, custom pipelines), and SHAP generally over LIME whenever computation is feasible due to its stronger guarantees.

7. **What is a Partial Dependence Plot and what key assumption can make it misleading?**
   Answer: PDP shows the average predicted outcome as a function of one (or two) features, marginalizing out (averaging over the observed distribution of) all other features. The key problematic assumption is that it implicitly treats the feature of interest as independent of the other features when constructing the average — if features are correlated, sweeping one feature's value while holding others at observed values can generate unrealistic/impossible combinations (e.g., a very large house with very few rooms), producing a misleading average effect that doesn't correspond to any plausible real instance.

8. **Explain the relationship between ICE plots and PDP, and describe a scenario where PDP would hide important information that ICE reveals.**
   Answer: PDP is the average of all individual ICE curves — one ICE curve per instance shows how that instance's prediction changes as the feature sweeps its range, holding its other features fixed; averaging all such curves pointwise gives the PDP line. If there's a strong interaction — e.g., feature X increases the prediction for one subgroup (defined by another feature) but decreases it for another subgroup — the individual ICE curves will have opposite slopes and could average out to a roughly flat PDP line, falsely suggesting feature X has little effect overall, when in fact it has a strong, heterogeneous effect that only ICE plots (showing the fan of diverging individual curves) would reveal.

9. **Scenario: A model is rejecting loan applications and you must give each rejected applicant a specific, individual reason (regulatory requirement). Which interpretability method(s) would you use, and why not just report global feature importance?**
   Answer: Use local SHAP values (or LIME) per rejected applicant — regulatory explanation requirements (e.g., adverse action notices, GDPR) need a per-individual, per-decision explanation, not an aggregate statement about the model overall. Global feature importance (e.g., "income is the most important feature on average") doesn't tell a specific applicant why *their* application was rejected — that requires decomposing *their* specific prediction into per-feature contributions, i.e., local interpretability. SHAP is generally preferred for its exact, provably-complete attribution (sums to the actual prediction), giving a defensible, mathematically grounded explanation for compliance purposes.

10. **Why can two different, but equally accurate, models produce very different SHAP-based feature importance rankings, and what does this imply about interpreting "true" feature importance?**
    Answer: SHAP (and any importance method) explains what a *specific fitted model* is doing, not the underlying causal or "true" data-generating relationship — when features are correlated or the modeling algorithm has different capacities to exploit certain feature interactions/redundancies, two equally accurate models can distribute "credit" among correlated/redundant features very differently (e.g., one model relying heavily on feature A, another achieving similar accuracy relying more on correlated feature B). This implies that feature importance is model-relative, not necessarily a ground truth about real-world causal drivers — interpretability results should be communicated as "how this model uses features" rather than definitive statements about real-world causality, and important business/causal claims should be validated with domain expertise, causal inference methods, or experiments (A/B tests), not attribution methods alone.

11. **How would you use permutation importance to detect that your model is relying on a leaked/proxy feature?**
    Answer: If permutation importance reveals one feature with dramatically higher importance than all others — especially if that feature is not plausibly causally related to the target at prediction time (e.g., a feature that could only be known after the outcome occurs, or an ID field weirdly correlated with the target due to data collection artifacts) — that's a strong signal of leakage. Cross-check by testing the model's performance with that feature removed: if performance craters unexpectedly (far more than a domain-plausible feature would explain), investigate the feature's provenance/timing to confirm leakage before deploying.

12. **What is the difference between a feature having a large SHAP value magnitude for a specific instance versus that feature having high average |SHAP value| (global importance)?**
    Answer: A large SHAP value magnitude for a specific instance means that feature strongly pushed *that individual's* prediction away from the baseline — it's a local, per-instance statement. High average |SHAP value| across all instances means the feature *tends to* have strong influence across the dataset broadly — a global statement. A feature can have a huge local SHAP value for one unusual instance (e.g., an outlier) while having low average importance overall (because for most instances it barely matters), and vice versa a feature could have moderate importance for almost every instance, adding up to high average importance without ever being the single dominant factor for any one instance.

13. **Explain LIME's "fidelity-interpretability tradeoff" — why can't LIME just fit a perfectly accurate local model?**
    Answer: LIME intentionally constrains the surrogate model to be simple/interpretable (e.g., sparse linear regression with limited features via a complexity penalty $\Omega(g)$) — a highly complex local surrogate could fit the black box's local behavior more faithfully (higher fidelity) but would no longer be human-interpretable, defeating the purpose. There's an inherent tradeoff: increasing surrogate complexity improves how faithfully it mimics the black-box locally (lower $\mathcal{L}$) but reduces how easily a human can understand and trust the explanation (higher effective complexity) — LIME's objective explicitly balances both terms rather than optimizing fidelity alone.

14. **Why might permutation importance give misleading results when two features are highly correlated?**
    Answer: When shuffling feature A (highly correlated with feature B), the model can still largely recover the lost information from A's original signal via its correlated proxy B (which remains unshuffled and unchanged) — so performance may barely drop even though A is, in some causal/informational sense, genuinely important; the importance gets "split" or diluted between A and B rather than attributed fully to either. This can make both correlated features look artificially unimportant individually, even though jointly (or via either alone) they carry substantial predictive signal — a known limitation motivating grouped permutation importance (shuffle correlated features together) or SHAP interaction values for a fuller picture.

15. **Scenario: your model's PDP for "years of experience" shows a flat/near-zero curve, but a domain expert insists experience should matter for salary prediction. What are three possible explanations and how would you investigate?**
    Answer: (1) **Masked interaction/heterogeneity**: experience may matter strongly but with opposite effects in different subgroups (e.g., different job roles), which cancel out in the averaged PDP — check individual ICE curves for fanning/divergence. (2) **The model isn't using the feature meaningfully** — e.g., a linear model with a near-zero coefficient due to collinearity with another proxy feature (like "age" or "job level") that's absorbing the signal instead — check correlations and try SHAP dependence plots or drop-column importance for the correlated group. (3) **The feature genuinely has little marginal predictive power in this dataset** conditional on other included features (e.g., "years of experience" might correlate with "job level," which is a more direct/proximate driver already in the model, so experience's *marginal* contribution given job level is already present is legitimately small) — this is a real possibility to validate with domain reasoning, not necessarily a bug. Investigate via ICE plots, correlation analysis, SHAP dependence plots colored by a candidate interacting feature, and if needed, a simpler univariate check (raw correlation of experience with target) to confirm whether the "expected" relationship exists at all in this data before assuming a modeling problem.

---

## Rapid-Fire Interview Q&A

1. **Q: What's the difference between parametric and non-parametric models?**
   A: Parametric models (linear/logistic regression) assume a fixed functional form with a fixed number of parameters regardless of data size. Non-parametric models (KNN, decision trees, kernel SVM) have complexity that can grow with the data and make fewer structural assumptions.

2. **Q: Why is squared error loss sensitive to outliers?**
   A: Squaring the residual amplifies large errors quadratically, so a single large outlier contributes disproportionately to the total loss and pulls the fitted line/model toward it.

3. **Q: What does "convex loss function" guarantee?**
   A: Any local minimum is the global minimum, so gradient-based optimization is guaranteed to converge to the optimal solution (given appropriate learning rate/conditions).

4. **Q: What is regularization, in one sentence?**
   A: A penalty added to the loss function that discourages model complexity (large coefficients) to reduce overfitting/variance at the cost of some bias.

5. **Q: L1 vs L2 penalty — which induces sparsity?**
   A: L1 (Lasso) induces sparsity by driving some coefficients exactly to zero; L2 (Ridge) shrinks coefficients toward zero but rarely to exactly zero.

6. **Q: What's the difference between bagging and boosting in one line?**
   A: Bagging trains models in parallel on bootstrap samples to reduce variance; boosting trains models sequentially, each correcting prior errors, to reduce bias.

7. **Q: Why does Random Forest use both bootstrap sampling and random feature selection?**
   A: Both mechanisms decorrelate the individual trees, which further reduces the variance of the averaged ensemble beyond bootstrap sampling alone.

8. **Q: What's the "kernel trick" in one sentence?**
   A: Computing dot products in a high-dimensional (or infinite-dimensional) transformed feature space implicitly via a kernel function, without ever explicitly computing the transformation.

9. **Q: What does the C parameter control in SVM?**
   A: The trade-off between maximizing the margin and minimizing misclassification of training points — large C = less tolerance for errors (narrower margin), small C = more tolerance (wider margin).

10. **Q: Gini impurity vs entropy — practical difference?**
    A: Both measure node impurity for classification splits and usually produce similar trees; Gini is computationally cheaper (no logarithm), while entropy is information-theoretically motivated.

11. **Q: What is the curse of dimensionality?**
    A: As the number of dimensions grows, data becomes sparse and distance-based notions like "nearest neighbor" lose meaning because distances between points converge.

12. **Q: Why does KNN require feature scaling?**
    A: Because it computes distances across all features, unscaled features with larger numeric ranges dominate the distance calculation regardless of true relevance.

13. **Q: What is OOB error?**
    A: The error estimated on the ~37% of samples left out of each tree's bootstrap sample in bagging/Random Forest — acts as a free validation estimate.

14. **Q: What does k-means++ improve over random initialization?**
    A: It spreads initial centroids apart (probabilistically favoring points far from existing centroids), leading to faster convergence and better final clustering quality.

15. **Q: How do you choose the number of clusters in K-means?**
    A: Elbow method on inertia/WCSS, or maximize the silhouette score across candidate k values.

16. **Q: What's the key advantage of DBSCAN over K-means?**
    A: DBSCAN finds arbitrarily shaped clusters and explicitly labels noise/outliers, without needing to pre-specify the number of clusters.

17. **Q: GMM vs K-means — key difference?**
    A: GMM gives soft, probabilistic cluster assignments and models elliptical/correlated clusters via full covariance; K-means gives hard assignments and assumes spherical clusters.

18. **Q: What does the EM algorithm guarantee?**
    A: Monotonically non-decreasing log-likelihood at each iteration, converging to a local (not necessarily global) optimum.

19. **Q: What is Naive Bayes' core assumption?**
    A: Conditional independence of all features given the class label.

20. **Q: Why does gradient boosting use shallow trees while Random Forest uses deep trees?**
    A: Boosting reduces bias sequentially and can overfit if base learners are too strong; Random Forest reduces variance via averaging, so it benefits from deep, low-bias/high-variance individual trees.

21. **Q: What is shrinkage/learning rate in boosting?**
    A: A multiplier (<1) applied to each new tree's contribution, slowing learning to improve generalization, generally paired with more boosting rounds.

22. **Q: What's the difference between Gini/impurity-based and permutation feature importance?**
    A: Impurity-based importance is computed from training-data splits and is biased toward high-cardinality features; permutation importance measures actual validation performance drop when a feature is shuffled, and is model-agnostic.

23. **Q: In one sentence, what is a Shapley value?**
    A: The fairly-averaged marginal contribution of a feature to a prediction, computed across all possible orderings/subsets of the other features.

24. **Q: SHAP vs LIME — key structural difference?**
    A: SHAP has game-theoretic guarantees and (via TreeSHAP) exact fast computation for trees; LIME fits an approximate local linear surrogate with no such guarantees.

25. **Q: What does a Partial Dependence Plot show?**
    A: The average marginal effect of a feature on the prediction, holding/marginalizing all other features at their observed values.

26. **Q: When would ICE plots reveal something a PDP hides?**
    A: When there's a strong interaction effect — individual instance curves diverge/cancel out in a way that produces a misleadingly flat average PDP.

27. **Q: What is multicollinearity and its main symptom?**
    A: High correlation among predictor features, causing unstable, high-variance coefficient estimates in OLS (though predictions may remain fine).

28. **Q: What does VIF > 10 typically indicate?**
    A: Problematic multicollinearity for that feature relative to the others.

29. **Q: One-vs-Rest vs Softmax for multiclass — key trade-off?**
    A: OvR trains independent binary classifiers (simple, parallelizable, probabilities don't sum to 1); softmax jointly models all classes with a proper probability distribution.

30. **Q: What is the hinge loss and why does it create sparse support vectors?**
    A: $\max(0, 1-y\hat y)$ — it's exactly zero for confidently correct points, so only points near/violating the margin (support vectors) contribute to the solution.

31. **Q: What is the main weakness of decision trees that ensembles solve?**
    A: High variance/instability — small data changes can produce very different trees; ensembling (bagging/boosting) averages this out.

32. **Q: What does `max_features` control in Random Forest, and what's the effect of decreasing it?**
    A: The number of features randomly considered at each split; decreasing it decorrelates trees further (lower ensemble variance) but can increase individual tree bias.

33. **Q: Isolation Forest's core intuition in one sentence?**
    A: Anomalies are isolated in fewer random splits than normal points because they are rare and different from the dense majority.

34. **Q: What's the difference between novelty detection and outlier detection?**
    A: Outlier detection identifies anomalies within the training data itself; novelty detection trains only on "normal" data and flags new/unseen points as anomalous relative to that learned normal distribution.

35. **Q: Why is log loss (cross-entropy) preferred over squared error for classification?**
    A: Squared error on probability outputs is non-convex when composed with sigmoid, penalizes confident correct predictions unnecessarily, and doesn't align with the MLE-derived, well-calibrated probabilistic interpretation that log loss provides.

36. **Q: What's the main downside of stacking/blending in production?**
    A: Added latency (must run multiple base models), complexity, and maintenance burden, for often marginal accuracy gains versus a single well-tuned model.

37. **Q: Why is Elastic Net often preferred over pure Lasso with correlated features?**
    A: Pure Lasso arbitrarily picks one feature from a correlated group and zeros out the rest (unstable across resamples); Elastic Net's L2 component encourages correlated features to share coefficient weight (grouping effect).

38. **Q: What is the "no free lunch" implication for choosing between these algorithms?**
    A: No single algorithm dominates across all datasets/tasks — the right choice depends on data size, dimensionality, noise, interpretability needs, and computational constraints, which is why understanding the trade-offs of each method (covered throughout this document) matters more than memorizing a "best" algorithm.

39. **Q: What's the practical difference between MAE and Huber loss when your data has outliers?**
    A: MAE is robust but has a constant-magnitude gradient everywhere, which can cause optimization to oscillate near the minimum; Huber is quadratic (smooth gradient) near zero and linear (bounded, outlier-robust) beyond a threshold $\delta$, giving smoother convergence while still capping outlier influence.

40. **Q: What does minimizing quantile loss at $\tau=0.5$ reduce to?**
    A: MAE — it recovers the conditional median.

41. **Q: In association rule mining, what does "lift = 1" tell you?**
    A: $X$ and $Y$ are statistically independent — buying $X$ tells you nothing about the likelihood of buying $Y$, even if the rule's confidence looks high just because $Y$ is popular overall.

42. **Q: Why does Apriori need to scan the database once per itemset-size level?**
    A: Because it generates and counts the support of all candidate $k$-itemsets (built only from confirmed frequent $(k-1)$-itemsets) in that pass before moving to level $k+1$ — support counts must come from the actual transaction data, not be inferred.

43. **Q: One-sentence reason to prefer FP-Growth over Apriori at scale?**
    A: FP-Growth mines frequent itemsets from a compressed FP-tree with no explicit candidate generation, avoiding Apriori's expensive repeated full-database scans.

44. **Q: For model evaluation, cross-validation strategy, dimensionality reduction, imbalanced-data handling, feature encoding, and hyperparameter search — which file covers these in depth?**
    A: [06_Feature_Engineering_and_Model_Evaluation.md](06_Feature_Engineering_and_Model_Evaluation.md) — this ML Fundamentals file focuses on the algorithms themselves (how each model works, trains, and its assumptions); metrics, validation strategy, PCA/t-SNE/UMAP, SMOTE/class-weighting, encoding, and Grid/Random/Bayesian hyperparameter search live in the dedicated Feature Engineering & Model Evaluation syllabus to avoid duplicating that content across files.

45. **Q: What does the perceptron update rule do, in one sentence?**
    A: It only adjusts weights when a training point is misclassified, nudging the decision boundary in the direction that would have classified that point correctly.

46. **Q: Why did the perceptron's inability to solve XOR matter historically?**
    A: XOR isn't linearly separable, so a single perceptron provably cannot learn it — this limitation (highlighted by Minsky & Papert) contributed to reduced interest in neural network research until multi-layer perceptrons with backpropagation showed stacked non-linear layers could solve it.

47. **Q: LDA vs QDA — which has more parameters and why?**
    A: QDA, because it estimates a separate full covariance matrix per class instead of one shared covariance matrix across all classes, at the cost of needing more data to estimate reliably.

48. **Q: Is "LDA" for classification the same algorithm as "LDA" in topic modeling?**
    A: No — Linear Discriminant Analysis (supervised classifier/dimensionality reduction, assumes Gaussian class-conditional densities) and Latent Dirichlet Allocation (unsupervised generative topic model over documents/words) are unrelated algorithms that happen to share an acronym.

49. **Q: One-vs-One requires how many classifiers for K classes?**
    A: $\binom{K}{2} = K(K-1)/2$ — one per pair of classes.

50. **Q: Name three classifiers that handle multiclass natively without needing OvR/OvO decomposition.**
    A: Decision trees (and tree ensembles), Naive Bayes, and KNN — all generalize directly to any number of classes via their core mechanism (impurity criteria, argmax posterior, majority vote).

51. **Q: What's the main risk of self-training / pseudo-labeling?**
    A: Confirmation bias — the model can confidently mislabel unlabeled points it's systematically wrong about, then reinforce that same mistake by retraining on its own incorrect pseudo-labels.

52. **Q: What's the difference between self-training and active learning in one sentence?**
    A: Self-training trusts the model's own most-confident predictions as free labels; active learning sends the model's *least*-confident (most informative) points to a human for real labels.
