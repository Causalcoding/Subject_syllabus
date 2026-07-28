# Mathematics for Machine Learning and AI — Interview Prep Syllabus

Every model you will ever train, deploy, or debug is a mathematical object wearing a software costume. **Data Scientists** lean hardest on statistics-flavored linear algebra and information theory (covariance structure, dimensionality reduction, hypothesis-adjacent entropy measures) because their job is inference and explanation. **Machine Learning Engineers** live in optimization and matrix calculus daily — they tune gradient descent variants, diagnose exploding/vanishing gradients, and reason about convergence and numerical stability in production training pipelines. **AI Engineers** (building on top of LLMs, RAG systems, and generative models) need vectors/norms/similarity metrics for embeddings and retrieval, information theory for perplexity and loss functions, and just enough optimization theory to reason about fine-tuning, quantization, and attention mechanics (which are themselves linear-algebra objects). Interviewers use math questions as a fast proxy for "can this person reason about why a model is failing, not just call `.fit()`." This document tags the primary audience for each area as **(DS)**, **(MLE)**, **(AIE)**, or **(All)**.

## Table of Contents

1. [Linear Algebra](#linear-algebra)
   - [Scalars, Vectors, Matrices, Tensors](#scalars-vectors-matrices-tensors)
   - [Vector Spaces, Basis, Span, Linear Independence](#vector-spaces-basis-span-linear-independence)
   - [Matrix Operations: Multiplication, Transpose, Inverse, Trace, Rank](#matrix-operations-multiplication-transpose-inverse-trace-rank)
   - [Norms (L0, L1, L2, L∞, Frobenius)](#norms-l0-l1-l2-l-frobenius)
   - [Eigenvalues, Eigenvectors, Eigendecomposition, Diagonalization](#eigenvalues-eigenvectors-eigendecomposition-diagonalization)
   - [Singular Value Decomposition (SVD) and PCA](#singular-value-decomposition-svd-and-pca)
   - [Positive Definite / Semi-Definite Matrices, Quadratic Forms](#positive-definite--semi-definite-matrices-quadratic-forms)
   - [Condition Number and Numerical Stability of Linear Systems](#condition-number-and-numerical-stability-of-linear-systems)
   - [Cholesky, LU, and QR Decompositions](#cholesky-lu-and-qr-decompositions)
   - [Matrix Calculus Basics (Gradients of Vector/Matrix Expressions)](#matrix-calculus-basics-gradients-of-vectormatrix-expressions)
   - [Interview Questions — Linear Algebra](#interview-questions--linear-algebra)
2. [Calculus and Optimization](#calculus-and-optimization)
   - [Derivatives, Partial Derivatives, Gradients, Directional Derivatives](#derivatives-partial-derivatives-gradients-directional-derivatives)
   - [Chain Rule and Backpropagation](#chain-rule-and-backpropagation)
   - [Gradient Checking (Finite-Difference Verification)](#gradient-checking-finite-difference-verification)
   - [Jacobians and Hessians, Second-Order Optimization](#jacobians-and-hessians-second-order-optimization)
   - [Quasi-Newton Methods: BFGS and L-BFGS](#quasi-newton-methods-bfgs-and-l-bfgs)
   - [Taylor Series Approximation](#taylor-series-approximation)
   - [Convexity: Convex Functions, Convex Sets, Global vs Local Minima](#convexity-convex-functions-convex-sets-global-vs-local-minima)
   - [Gradient Descent Variants](#gradient-descent-variants)
   - [Learning Rate Schedules, Warmup, Convergence Criteria](#learning-rate-schedules-warmup-convergence-criteria)
   - [Numerical Stability: Log-Sum-Exp and Overflow-Safe Softmax](#numerical-stability-log-sum-exp-and-overflow-safe-softmax)
   - [Constrained Optimization: Lagrange Multipliers, KKT Conditions](#constrained-optimization-lagrange-multipliers-kkt-conditions)
   - [Interview Questions — Calculus and Optimization](#interview-questions--calculus-and-optimization)
3. [Information Theory](#information-theory)
   - [Entropy, Cross-Entropy, Joint Entropy, Conditional Entropy](#entropy-cross-entropy-joint-entropy-conditional-entropy)
   - [KL Divergence and Jensen-Shannon Divergence](#kl-divergence-and-jensen-shannon-divergence)
   - [Mutual Information and Feature Selection](#mutual-information-and-feature-selection)
   - [Perplexity and Language Modeling](#perplexity-and-language-modeling)
   - [Interview Questions — Information Theory](#interview-questions--information-theory)
4. [Rapid-Fire Interview Q&A](#rapid-fire-interview-qa)

---

## Linear Algebra

*Primary audience: **(All)** — but especially **(MLE)** for backprop/matrix-calculus, **(DS)** for PCA/dimensionality reduction, **(AIE)** for embeddings/similarity.*

### Scalars, Vectors, Matrices, Tensors

**Definitions.**
- A **scalar** is a single number, $a \in \mathbb{R}$.
- A **vector** is an ordered array of numbers, $\mathbf{x} \in \mathbb{R}^n$, representing a point or direction in $n$-dimensional space: `x = [x_1, x_2, ..., x_n]^T`.
- A **matrix** is a 2-D array, $A \in \mathbb{R}^{m \times n}$ ($m$ rows, $n$ columns).
- A **tensor** is the generalization to arbitrary dimensionality (rank/order): a scalar is a rank-0 tensor, a vector rank-1, a matrix rank-2, and a batch of RGB images is a rank-4 tensor `(batch, height, width, channels)`.

**Intuition.** Think of a tensor's *rank* (number of axes) as "how many independent indices do I need to address one element?" `A[i, j, k]` needs three indices → rank-3 tensor. This is unrelated to *matrix rank* (linear-algebraic rank, covered below) — a huge source of interview confusion.

**Why it matters in ML.** Every framework (NumPy, PyTorch, TensorFlow) represents data, weights, activations, and gradients as tensors. Shape mismatches are the #1 debugging task for MLEs. Broadcasting rules (how NumPy/PyTorch align mismatched shapes) are a direct consequence of how tensor dimensions are defined.

**Pitfalls.**
- Confusing tensor *rank/order* (number of axes) with matrix *rank* (dimension of column space).
- Row-vector vs column-vector convention inconsistency — always clarify whether $\mathbf{x}$ is $n \times 1$ or $1 \times n$ before writing $A\mathbf{x}$ vs $\mathbf{x}^T A$.
- Off-by-one/broadcasting bugs when unsqueezing/expanding tensors (e.g., `(batch,)` vs `(batch, 1)`).

### Vector Spaces, Basis, Span, Linear Independence

**Vector space**: a set $V$ closed under vector addition and scalar multiplication, satisfying associativity, commutativity, existence of a zero vector, additive inverses, and distributivity. $\mathbb{R}^n$ with standard operations is the canonical example.

**Span**: the span of a set of vectors $\{\mathbf{v}_1, \dots, \mathbf{v}_k\}$ is the set of all linear combinations:
```
span{v_1,...,v_k} = { c_1 v_1 + c_2 v_2 + ... + c_k v_k : c_i ∈ ℝ }
```
This is itself a subspace of $V$.

**Linear independence**: vectors $\{\mathbf{v}_1, \dots, \mathbf{v}_k\}$ are linearly independent if the only solution to
```
c_1 v_1 + c_2 v_2 + ... + c_k v_k = 0
```
is $c_1 = c_2 = \dots = c_k = 0$. If some non-trivial combination equals zero, the set is linearly *dependent* — meaning at least one vector is redundant (expressible via the others).

**Basis**: a linearly independent spanning set for $V$. Every vector in $V$ has a *unique* representation as a linear combination of basis vectors. The number of vectors in any basis of $V$ is the **dimension** of $V$, $\dim(V)$.

**Worked example.** In $\mathbb{R}^3$, are $\mathbf{v}_1=(1,0,0)$, $\mathbf{v}_2=(1,1,0)$, $\mathbf{v}_3=(1,1,1)$ independent?
Set $c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + c_3 \mathbf{v}_3 = 0$:
```
c1 + c2 + c3 = 0
     c2 + c3 = 0
          c3 = 0
```
Back-substitution forces $c_3=0 \Rightarrow c_2=0 \Rightarrow c_1=0$. Only the trivial solution — the vectors are independent, and since there are 3 of them in $\mathbb{R}^3$, they form a basis.

**Why it matters in ML.** Feature redundancy = linear dependence among feature columns → this is exactly what causes multicollinearity in linear regression (the design matrix $X^TX$ becomes singular or ill-conditioned, coefficients blow up / are non-unique). PCA finds a *new basis* aligned with directions of maximum variance. Embedding spaces in deep learning are (approximately) vector spaces where semantic relationships correspond to linear structure (`king - man + woman ≈ queen`).

**Pitfalls.** Linear independence is about combinations summing to the **zero vector**, not about vectors being "different" or "non-parallel" in higher dimensions — three vectors in $\mathbb{R}^2$ can never be independent (dimension caps how many independent vectors can exist).

### Matrix Operations: Multiplication, Transpose, Inverse, Trace, Rank

**Multiplication.** For $A \in \mathbb{R}^{m\times n}$, $B \in \mathbb{R}^{n \times p}$, the product $C = AB \in \mathbb{R}^{m \times p}$ has entries
```
C_ij = Σ_k A_ik * B_kj
```
Cost: $O(mnp)$ naively. Matrix multiplication is **associative** ($(AB)C = A(BC)$) and **distributive**, but **not commutative** ($AB \neq BA$ in general, and shapes may not even allow both orders).

**Transpose.** $(A^T)_{ij} = A_{ji}$. Properties: $(AB)^T = B^T A^T$ (order reverses!), $(A^T)^T = A$, $(A+B)^T = A^T+B^T$.

**Inverse.** $A^{-1}$ exists iff $A$ is square and full rank (equivalently $\det(A) \neq 0$), satisfying $AA^{-1}=A^{-1}A=I$. $(AB)^{-1} = B^{-1}A^{-1}$. For a $2\times2$ matrix:
```
A = [[a, b], [c, d]]   →   A^{-1} = 1/(ad-bc) * [[d, -b], [-c, a]]
```
Computing an explicit inverse is $O(n^3)$ and numerically unstable for ill-conditioned matrices; in practice you solve $A\mathbf{x}=\mathbf{b}$ via LU/Cholesky/QR decomposition rather than computing $A^{-1}$ explicitly.

**Trace.** $\text{tr}(A) = \sum_i A_{ii}$ (sum of diagonal entries, square matrices only). Key identities: $\text{tr}(A) = \sum_i \lambda_i$ (sum of eigenvalues), $\text{tr}(AB) = \text{tr}(BA)$ (**cyclic property**, holds even though $AB \neq BA$), $\text{tr}(A^T) = \text{tr}(A)$.

**Rank.** $\text{rank}(A)$ = dimension of the column space = dimension of the row space (they're always equal) = number of linearly independent rows/columns = number of non-zero singular values. **Full rank** for $A \in \mathbb{R}^{m\times n}$ means $\text{rank}(A) = \min(m,n)$. Rank-deficient matrices are non-invertible (if square) and signal redundant features or degenerate data.

**Worked example (rank).**
```
A = [[1, 2], [2, 4]]
```
Row 2 = 2 × Row 1 → linearly dependent rows → $\text{rank}(A) = 1$, and $\det(A) = 1\cdot4 - 2\cdot2 = 0$ confirms singularity.

**Why it matters in ML.** Matrix multiplication *is* the forward pass of every dense/conv/attention layer. `rank` diagnoses multicollinearity and informs low-rank approximation techniques (LoRA fine-tuning of LLMs assumes weight *updates* are low-rank!). `trace` shows up in the derivative of quadratic forms and in computing loss like the Frobenius norm.

**Pitfalls.**
- Assuming $AB=BA$ — false in general (this is *why* order matters in chained linear layers and why attention's $QK^T$ ordering matters).
- Forgetting that inverses don't distribute over addition: $(A+B)^{-1} \neq A^{-1}+B^{-1}$.
- Numerically inverting near-singular matrices instead of using `solve`, ridge regularization ($A + \lambda I$), or pseudo-inverse for non-square/rank-deficient systems.

### Norms (L0, L1, L2, L∞, Frobenius)

A **norm** $\|\cdot\|$ measures "size"/"length" and must satisfy: non-negativity ($\|x\|\ge0$, $=0$ iff $x=0$), absolute homogeneity ($\|\alpha x\| = |\alpha|\|x\|$), and the triangle inequality ($\|x+y\| \le \|x\|+\|y\|$).

| Norm | Formula | Geometric meaning | ML use |
|---|---|---|---|
| $L_0$ | `‖x‖_0 = count(x_i ≠ 0)` (not a true norm — fails homogeneity) | number of non-zero entries | sparsity target; feature-selection ideal (NP-hard to optimize directly) |
| $L_1$ | `‖x‖_1 = Σ|x_i|` | "Manhattan"/taxicab distance | Lasso regression, sparse solutions, robust loss (MAE) |
| $L_2$ | `‖x‖_2 = sqrt(Σ x_i²)` | Euclidean length | Ridge regression, weight decay, Euclidean distance, gradient clipping |
| $L_\infty$ | `‖x‖_∞ = max_i |x_i|` | largest coordinate magnitude | adversarial robustness (bounding perturbation), gradient clipping variants |
| Frobenius | `‖A‖_F = sqrt(Σ_ij A_ij²) = sqrt(tr(A^T A))` | $L_2$ norm applied to a flattened matrix | matrix regularization, low-rank approximation error, weight-decay on whole layers |

**Intuition — why $L_1$ induces sparsity but $L_2$ doesn't.** The $L_1$ ball is a diamond (cross-polytope) with sharp corners *on the axes*; the $L_2$ ball is a smooth sphere. When you minimize a loss subject to a norm-ball constraint, the optimum tends to land where the loss's contour first touches the constraint boundary. Because the $L_1$ diamond has corners exactly on the coordinate axes, the intersection is more likely to occur *at* a corner — i.e., with some coordinates exactly zero. The smooth $L_2$ ball has no such preferred "corner" points, so it shrinks coefficients toward zero without ever hitting exactly zero.

**Worked example.** $\mathbf{x} = (3, -4, 0, 1)$:
```
‖x‖_0 = 3        (three non-zero entries)
‖x‖_1 = 3+4+0+1 = 8
‖x‖_2 = sqrt(9+16+0+1) = sqrt(26) ≈ 5.10
‖x‖_∞ = 4
```

**Why it matters in ML.** Regularization terms added to loss functions ($\lambda\|\mathbf{w}\|_1$ for Lasso, $\lambda\|\mathbf{w}\|_2^2$ for Ridge/weight-decay) directly use these norms. Gradient clipping in RNN/Transformer training rescales gradients by their $L_2$ (or $L_\infty$) norm to prevent exploding gradients. Distance metrics in KNN/clustering/vector-search embeddings ($L_2$/cosine) determine retrieval quality in RAG systems.

**Pitfalls.**
- $L_0$ is not technically a norm (violates homogeneity: $\|\alpha x\|_0 = \|x\|_0$ for any $\alpha\neq0$, not $|\alpha|\|x\|_0$).
- Confusing $\|\mathbf{w}\|_2$ (used in the penalty) with $\|\mathbf{w}\|_2^2$ (used in Ridge's *actual* objective, since it's differentiable everywhere unlike $\|\mathbf{w}\|_2$ at the origin, and the squared form makes gradients linear in $\mathbf{w}$).
- Frobenius norm is *not* the induced operator (spectral) norm — $\|A\|_F \ge \|A\|_2^{\text{spectral}}$ (largest singular value), and interviewers love testing this distinction.

### Eigenvalues, Eigenvectors, Eigendecomposition, Diagonalization

**Definition.** For square $A \in \mathbb{R}^{n\times n}$, a non-zero vector $\mathbf{v}$ is an **eigenvector** with **eigenvalue** $\lambda$ if:
```
A v = λ v
```
$A$ acting on $\mathbf{v}$ only *scales* it (by $\lambda$), it doesn't rotate/change its direction. Eigenvalues solve the **characteristic equation** $\det(A - \lambda I) = 0$.

**Worked example.**
```
A = [[2, 0], [0, 3]]
```
Since $A$ is diagonal, eigenvalues are the diagonal entries: $\lambda_1=2$ (eigenvector $(1,0)$), $\lambda_2=3$ (eigenvector $(0,1)$).

Slightly harder:
```
A = [[2, 1], [1, 2]]
det(A - λI) = (2-λ)² - 1 = 0  →  λ² - 4λ + 3 = 0  →  λ = 1, 3
```
For $\lambda=3$: solve $(A-3I)\mathbf{v}=0$ → $\begin{bmatrix}-1&1\\1&-1\end{bmatrix}\mathbf{v}=0$ → $\mathbf{v}=(1,1)$ (normalized: $\frac{1}{\sqrt2}(1,1)$). For $\lambda=1$: $\mathbf{v}=(1,-1)$.

**Eigendecomposition.** If $A$ has $n$ linearly independent eigenvectors, $A = Q\Lambda Q^{-1}$, where $Q$'s columns are the eigenvectors and $\Lambda=\text{diag}(\lambda_1,\dots,\lambda_n)$. This is **diagonalization**. Not every matrix is diagonalizable (e.g., defective matrices with repeated eigenvalues but insufficient independent eigenvectors, like $\begin{bmatrix}1&1\\0&1\end{bmatrix}$). If $A$ is symmetric, the **Spectral Theorem** guarantees $A=Q\Lambda Q^T$ with $Q$ *orthogonal* ($Q^TQ=I$) and all $\lambda_i$ real — this is the case that matters most in ML (covariance matrices, Hessians of quadratic losses).

**Intuition.** Eigenvectors are the "natural axes" of a linear transformation — directions that don't get rotated, only stretched/compressed. For a covariance matrix, eigenvectors point along directions of the data's spread, and eigenvalues quantify how much variance lies along each direction — this is literally what PCA extracts.

**Why it matters in ML.**
- **PCA**: eigenvectors of the covariance matrix = principal component directions; eigenvalues = variance explained by each component.
- **Hessian analysis**: eigenvalues of the loss Hessian tell you about the local curvature/conditioning of the optimization landscape — large eigenvalue spread (ill-conditioning) causes slow, zig-zagging gradient descent.
- **Spectral clustering**, **PageRank** (dominant eigenvector of a transition matrix), **stability analysis** of dynamical systems / RNNs (eigenvalues of the recurrent weight matrix determine whether hidden states explode/vanish over time — magnitude $>1$ explodes, $<1$ vanishes).
- **Power iteration** finds the dominant eigenvector by repeatedly applying $A$ and normalizing — this underlies PageRank and fast approximate SVD.

**Pitfalls.**
- Eigenvalues can be complex for general (non-symmetric) real matrices — only symmetric matrices guarantee real eigenvalues.
- A repeated eigenvalue doesn't guarantee enough independent eigenvectors (algebraic vs geometric multiplicity can differ) → not diagonalizable.
- $\det(A) = \prod \lambda_i$ and $\text{tr}(A) = \sum \lambda_i$ — a very common "compute this without doing full eigendecomposition" trick question.

### Singular Value Decomposition (SVD) and PCA

**SVD**: *any* real matrix $A \in \mathbb{R}^{m\times n}$ (need not be square!) factors as
```
A = U Σ V^T
```
where $U \in \mathbb{R}^{m\times m}$ is orthogonal (columns = left singular vectors), $V \in \mathbb{R}^{n\times n}$ is orthogonal (columns = right singular vectors), and $\Sigma \in \mathbb{R}^{m\times n}$ is diagonal with non-negative entries $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$ (the **singular values**).

**Relation to eigendecomposition.** Singular values of $A$ are the square roots of the eigenvalues of $A^TA$ (or $AA^T$): $\sigma_i = \sqrt{\lambda_i(A^TA)}$. Right singular vectors $V$ = eigenvectors of $A^TA$; left singular vectors $U$ = eigenvectors of $AA^T$. SVD is more general than eigendecomposition because it works for rectangular, non-square, non-symmetric matrices, and always exists (unlike eigendecomposition, which can fail to exist in the real domain for non-symmetric matrices).

**SVD ↔ PCA.** Let $X\in\mathbb{R}^{n\times d}$ be a mean-centered data matrix ($n$ samples, $d$ features). PCA seeks eigenvectors of the covariance matrix $C = \frac{1}{n-1}X^TX$. Taking the SVD $X = U\Sigma V^T$:
```
X^T X = V Σ^T U^T U Σ V^T = V Σ² V^T
```
so $V$'s columns *are* the eigenvectors of $X^TX$ (hence the principal component directions), and the eigenvalues of the covariance matrix relate to singular values by $\lambda_i = \sigma_i^2/(n-1)$. This means you can compute PCA directly via SVD of the *data matrix* without ever forming $X^TX$ explicitly — which is both faster and numerically more stable (forming $X^TX$ squares the condition number).

**Worked example (rank-1 approximation via SVD — Eckart–Young theorem).** The best rank-$k$ approximation of $A$ under Frobenius norm is $A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$, keeping only the top $k$ singular values/vectors. This is exactly what **truncated SVD dimensionality reduction** and **image/matrix compression** do — it's provably optimal among all rank-$k$ matrices.

**Why it matters in ML.** PCA/dimensionality reduction, **recommender systems** (matrix factorization / collaborative filtering decomposes the user-item matrix), **latent semantic analysis** (LSA) for text, noise reduction (truncating small singular values removes noise directions), and computing the **pseudo-inverse** $A^+ = V\Sigma^+U^T$ for solving least-squares problems when $A$ is not invertible.

**Pitfalls.**
- SVD always exists; eigendecomposition doesn't (for non-symmetric/non-square matrices) — don't say "SVD is eigendecomposition of non-square matrices," say it's a *generalization* built from eigendecompositions of $A^TA$/$AA^T$.
- PCA requires **mean-centering** (sometimes scaling to unit variance too) before SVD — skipping this is a classic bug that causes the first "principal component" to just capture the mean.
- Explained variance ratio = $\lambda_i / \sum_j \lambda_j = \sigma_i^2/\sum_j\sigma_j^2$; forgetting to square the singular values is a common computational error.

### Positive Definite / Semi-Definite Matrices, Quadratic Forms

**Quadratic form.** For symmetric $A\in\mathbb{R}^{n\times n}$ and $\mathbf{x}\in\mathbb{R}^n$: $Q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} = \sum_{i,j} A_{ij}x_ix_j$.

**Definitions:**
- **Positive definite (PD)**: $\mathbf{x}^TA\mathbf{x} > 0$ for all $\mathbf{x}\neq0$. Equivalent conditions: all eigenvalues $>0$; all leading principal minors (upper-left $k\times k$ determinants) $>0$ (Sylvester's criterion); Cholesky decomposition exists ($A=LL^T$).
- **Positive semi-definite (PSD)**: $\mathbf{x}^TA\mathbf{x} \ge 0$ for all $\mathbf{x}$ (eigenvalues $\ge0$).
- **Negative definite/semi-definite**: reverse inequalities.
- **Indefinite**: has both positive and negative eigenvalues (saddle point signature).

**Worked example.**
```
A = [[2, 0], [0, 3]]  →  x^T A x = 2x_1² + 3x_2² ≥ 0, = 0 only at x=0  →  PD
A = [[1, 2], [2, 1]]  →  eigenvalues: 3, -1  →  indefinite
```

**Geometric intuition.** PD quadratic forms produce elliptical (bowl-shaped) contours — a single, unique global minimum. Indefinite forms produce saddle-shaped contours. This is exactly the shape of a **loss function's local quadratic approximation** near a critical point (via the Hessian, see Taylor series section) — PD Hessian ⇒ local minimum, negative definite ⇒ local maximum, indefinite ⇒ saddle point.

**Why it matters in ML.**
- **Covariance matrices** are always PSD by construction (and PD if features are linearly independent / no perfect collinearity), which is why variance can never be negative and why you can always take a "square root" (Cholesky) of a valid covariance matrix for sampling from a multivariate Gaussian.
- **Convexity check**: a twice-differentiable function is convex iff its Hessian is PSD everywhere — this is the primary tool for proving loss functions (MSE, logistic loss) are convex.
- **Kernel matrices (Gram matrices)** in SVMs/Gaussian Processes must be PSD (**Mercer's condition**) for the kernel to correspond to a valid inner product in some feature space.
- **Newton's method** requires inverting the Hessian — if it's not PD, the update can move *toward* a saddle/maximum rather than the minimum, motivating damped/regularized variants (Levenberg-Marquardt adds $\lambda I$ to force PD-ness).

**Pitfalls.**
- PSD ≠ PD: a PSD matrix can be singular (zero eigenvalue) and hence non-invertible — ridge regression's $\lambda I$ term is added specifically to convert a PSD-but-singular $X^TX$ into a strictly PD (and invertible) matrix.
- Testing PD-ness by checking $A_{ii}>0$ for diagonal entries alone is insufficient — need the full eigenvalue or principal-minor test.
- The quadratic form definition requires $A$ effectively symmetric; for asymmetric $A$, only the symmetric part $\frac{1}{2}(A+A^T)$ contributes to $\mathbf{x}^TA\mathbf{x}$.

### Condition Number and Numerical Stability of Linear Systems

**Definition.** The condition number of $A$ measures how much relative error in the input of a linear system can be amplified in the output. Formally $\kappa(A) = \|A\|\cdot\|A^{-1}\|$ for any chosen matrix norm; using the spectral (induced $L_2$) norm, this reduces to
```
κ(A) = σ_max / σ_min
```
the ratio of $A$'s largest to smallest singular value. For symmetric positive definite $A$ (e.g., a Hessian or covariance matrix), singular values equal eigenvalues, so $\kappa(A)=\lambda_{max}/\lambda_{min}$. An orthogonal matrix has $\kappa=1$ (perfectly conditioned — it doesn't distort lengths at all); $\kappa\to\infty$ as $A$ approaches singularity.

**Why it matters — error amplification.** For $A\mathbf{x}=\mathbf{b}$, a small relative perturbation in $\mathbf{b}$ (e.g., floating-point rounding, sensor noise) propagates into the solution bounded by
```
‖δx‖/‖x‖ ≤ κ(A) · ‖δb‖/‖b‖
```
A large $\kappa(A)$ means tiny input noise can produce a hugely different solution — the system is **ill-conditioned**, and no algorithm (however numerically careful) can fully undo this; it's a property of the *problem*, not the solver.

**Worked example.** $A=\text{diag}(1000, 1)$: $\kappa(A)=1000/1=1000$. Geometrically, $A$'s unit-circle image is an extremely elongated ellipse — exactly the "narrow valley" picture from the Hessian/gradient-descent discussion above (recall the $H=\text{diag}(2,6)$ example, $\kappa=3$, that already zig-zagged). At $\kappa=1000$, gradient descent on the corresponding quadratic needs a step size small enough to avoid diverging along the stiff ($\lambda_{max}$) direction, so it crawls extremely slowly along the flat ($\lambda_{min}$) direction — roughly $O(\kappa)$ iterations to converge to fixed accuracy, vs. $O(\sqrt\kappa)$ for momentum/Nesterov, vs. condition-number-independent for exact Newton.

**Why it matters in ML.** Feature scaling/standardization directly improves the conditioning of $X^TX$ (and hence the loss Hessian for linear/logistic regression), which is a large part of *why* "always standardize your features" is such robust default advice — it's not just about comparable units, it's about optimization speed. Forming $X^TX$ explicitly **squares** the condition number ($\kappa(X^TX)=\kappa(X)^2$), which is why numerically-aware solvers use QR or SVD directly on $X$ (see below) rather than the normal equations when $X$ is even moderately ill-conditioned. Libraries like `numpy.linalg.cond` let you check conditioning before trusting a solved system.

**Pitfalls.**
- Condition number is norm-dependent; interviewers almost always mean the spectral/2-norm version ($\sigma_{max}/\sigma_{min}$) unless stated otherwise.
- A low condition number does *not* by itself guarantee fast convergence for non-convex or non-quadratic objectives — the clean $O(\kappa)$/$O(\sqrt\kappa)$ iteration bounds are proven for convex quadratics; they're only a useful local intuition elsewhere.
- Confusing "ill-conditioned" (a numerical/optimization property, fixable by rescaling/regularizing) with "rank-deficient" (a structural property — $\kappa=\infty$ exactly, not just large) — ridge's $\lambda I$ fixes both simultaneously but they're conceptually distinct failure modes.

### Cholesky, LU, and QR Decompositions

**Why decompositions instead of explicit inverses.** As noted above, computing $A^{-1}$ explicitly is $O(n^3)$ and numerically fragile. In practice, solving $A\mathbf{x}=\mathbf{b}$ or evaluating quantities that "look like" they need an inverse is done via a matrix **factorization** chosen to match $A$'s structure, then cheap forward/back-substitution.

**LU decomposition** (general square $A$): $A=LU$ ($L$ unit lower-triangular, $U$ upper-triangular), computed via Gaussian elimination. For numerical stability with a general (non-symmetric) matrix, **partial pivoting** is required: $PA=LU$ for some row-permutation matrix $P$ — without pivoting, a small pivot element can cause catastrophic error amplification during elimination.

**Cholesky decomposition** (symmetric positive definite $A$ only): a unique factorization $A=LL^T$ with $L$ lower-triangular and positive diagonal entries, computed via
```
L_jj = sqrt( A_jj - Σ_{k<j} L_jk² )
L_ij = ( A_ij - Σ_{k<j} L_ik L_jk ) / L_jj      for i > j
```
Costs about $n^3/3$ flops — roughly half of general LU — and needs **no pivoting**, because positive-definiteness itself guarantees numerical stability of the elimination. If, while computing it, some $A_{jj}-\sum L_{jk}^2$ comes out negative (square root of a negative number), that's a fast, standard numerical *test* that $A$ is not actually PD.

**Worked example.** $A=\begin{bmatrix}4&2\\2&5\end{bmatrix}$ (check PD: leading minors $4>0$ and $\det=20-4=16>0$). $L_{11}=\sqrt4=2$; $L_{21}=2/2=1$; $L_{22}=\sqrt{5-1^2}=2$. So $L=\begin{bmatrix}2&0\\1&2\end{bmatrix}$, and indeed $LL^T=\begin{bmatrix}4&2\\2&5\end{bmatrix}=A$. To solve $A\mathbf{x}=\mathbf{b}$: forward-substitute $L\mathbf{y}=\mathbf{b}$, then back-substitute $L^T\mathbf{x}=\mathbf{y}$ — never forming $A^{-1}$.

**QR decomposition** (any $A$, including rectangular): $A=QR$ ($Q$ orthogonal/orthonormal columns, $R$ upper-triangular), computed via Gram-Schmidt or (more stably) Householder reflections. Used to solve least-squares $\min_w\|Xw-y\|_2^2$ *directly on $X$* — since $Q^TQ=I$, the normal equations $X^TXw=X^Ty$ reduce to the triangular system $Rw=Q^Ty$, avoiding ever forming $X^TX$ and therefore avoiding the condition-number-squaring problem described above.

| Decomposition | Requires | Cost | Typical use |
|---|---|---|---|
| LU (+ pivoting) | square, invertible | $O(n^3)$ | general linear solve (`np.linalg.solve`) |
| Cholesky | symmetric PD | $\sim n^3/3$ | covariance/kernel/Hessian solves, Gaussian sampling, GP regression |
| QR | any shape | $O(mn^2)$ for $m\times n$ | least squares, orthogonalization, numerically stable regression |

**Why it matters in ML.** Gaussian Process regression solves and evaluates marginal likelihood via Cholesky of the kernel matrix (log-determinant falls out as $2\sum_i \log L_{ii}$, itself numerically far safer than computing $\det$ directly); sampling a multivariate Gaussian (Q11 above) needs exactly this $L$; natural-gradient and second-order optimizers that need $H^{-1}\mathbf{v}$ solve a linear system via Cholesky/CG rather than inverting $H$; `scikit-learn`'s and `statsmodels`' linear regression solvers default to QR/SVD-based least squares specifically to sidestep normal-equation ill-conditioning.

**Pitfalls.**
- Attempting Cholesky on a matrix that *should* be PD but isn't quite (due to floating-point error, e.g., a sample covariance matrix with $n<d$) throws a decomposition error — the standard fix is adding a small "jitter" $\epsilon I$ before decomposing, mirroring ridge regularization's role in inversion.
- LU *without* pivoting can be numerically unstable or fail outright even for a perfectly invertible matrix, if elimination happens to divide by a very small pivot.
- Solving via normal equations + Cholesky is faster than QR when it's numerically safe to do so (well-conditioned $X$), but QR (or SVD) is the safer default when $X$'s conditioning is unknown or suspect.

### Matrix Calculus Basics (Gradients of Vector/Matrix Expressions)

**Why this exists.** Backpropagation is repeated application of the chain rule to vector/matrix-valued functions. You need a small toolbox of standard derivative identities (using the convention that the gradient of scalar $f$ w.r.t. vector $\mathbf{x}$, $\nabla_x f$, has the same shape as $\mathbf{x}$ — the "denominator layout"/numerator-vs-denominator convention varies by textbook, so always confirm shapes by consistency-checking dimensions).

**Core identities** (for $\mathbf{x}\in\mathbb{R}^n$, $A$ constant matrix, $\mathbf{a}$ constant vector):

| Expression | Gradient |
|---|---|
| $f=\mathbf{a}^T\mathbf{x}$ | $\nabla_x f = \mathbf{a}$ |
| $f=\mathbf{x}^T\mathbf{a}$ | $\nabla_x f = \mathbf{a}$ |
| $f=\mathbf{x}^T\mathbf{x}$ | $\nabla_x f = 2\mathbf{x}$ |
| $f=\mathbf{x}^TA\mathbf{x}$ | $\nabla_x f = (A+A^T)\mathbf{x}$ (reduces to $2A\mathbf{x}$ if $A$ symmetric) |
| $f=A\mathbf{x}$ (vector-valued) | Jacobian $\frac{\partial f}{\partial \mathbf{x}} = A$ |
| $f = \|\mathbf{x}-\mathbf{a}\|_2^2$ | $\nabla_x f = 2(\mathbf{x}-\mathbf{a})$ |

**Worked example — deriving the OLS normal equations.** Linear regression loss: $L(\mathbf{w}) = \|X\mathbf{w}-\mathbf{y}\|_2^2 = (X\mathbf{w}-\mathbf{y})^T(X\mathbf{w}-\mathbf{y})$. Expand:
```
L(w) = w^T X^T X w - 2 w^T X^T y + y^T y
```
Using $\nabla_w(\mathbf{w}^TA\mathbf{w}) = 2A\mathbf{w}$ (since $X^TX$ is symmetric) and $\nabla_w(\mathbf{w}^T\mathbf{b}) = \mathbf{b}$:
```
∇_w L = 2 X^T X w - 2 X^T y
```
Setting to zero: $X^TX\mathbf{w} = X^T\mathbf{y}$ → $\mathbf{w}^* = (X^TX)^{-1}X^T\mathbf{y}$ — the closed-form OLS solution, derived purely from matrix calculus.

**Worked example — softmax + cross-entropy gradient (the backbone of every classifier's output layer).** For logits $\mathbf{z}$, softmax $p_i = e^{z_i}/\sum_j e^{z_j}$, and one-hot true label $\mathbf{y}$, cross-entropy loss $L=-\sum_i y_i \log p_i$. The famously clean result:
```
∂L/∂z = p - y
```
This elegant simplification (the "predicted probabilities minus one-hot true label") is *why* softmax + cross-entropy is used together almost everywhere — the gradient has no exponential/log terms left after differentiating, which is both a mathematically pleasant coincidence and numerically favorable.

**Why it matters in ML.** Every `loss.backward()` call is an automatic-differentiation engine mechanically chaining these identities across a computation graph. Understanding matrix calculus lets you (a) derive gradients for custom loss functions/layers, (b) debug NaN/exploding gradients by inspecting which term's Jacobian is unstable, (c) understand papers describing new architectures via their forward/backward math.

**Pitfalls.**
- Shape mismatches: always sanity-check that $\nabla_x f$ has the same shape as $\mathbf{x}$.
- Forgetting the transpose flip when chaining Jacobians: for $\mathbf{y}=A\mathbf{x}$, $\mathbf{z}=g(\mathbf{y})$, $\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y}\cdot A$ (matrix on the right, not left, given row-gradient convention) — get this backwards and dimensions won't even multiply.
- Element-wise (Hadamard) operations like activation functions have *diagonal* Jacobians — treating them like dense Jacobians wastes memory/compute; frameworks exploit this for efficiency.

### Interview Questions — Linear Algebra

**Q1 (basic).** What's the difference between a vector's dimension and a matrix's rank?
**A.** Dimension of a vector is just how many components/entries it has (its length as an array). Rank of a matrix is the number of linearly independent rows/columns — a property of the *linear transformation*, not a count of entries. A $5\times5$ matrix could have rank as low as 1.

**Q2 (basic).** Why is matrix multiplication not commutative? Give an example.
**A.** $AB$ composes two linear transformations in a specific order — apply $B$ first, then $A$ — and shapes/geometric effects generally differ if you swap the order. Example: $A=\begin{bmatrix}0&1\\0&0\end{bmatrix}$, $B=\begin{bmatrix}0&0\\1&0\end{bmatrix}$: $AB=\begin{bmatrix}1&0\\0&0\end{bmatrix}$ but $BA=\begin{bmatrix}0&0\\0&1\end{bmatrix}$ — different results.

**Q3 (conceptual).** When does a matrix inverse fail to exist, and what's the practical alternative?
**A.** The inverse fails to exist when the matrix is non-square, or square but singular ($\det=0$, rank-deficient, i.e., columns are linearly dependent). The practical alternative is the **Moore-Penrose pseudo-inverse** $A^+$ (computed via SVD, $A^+=V\Sigma^+U^T$), which gives the minimum-norm least-squares solution to $A\mathbf{x}=\mathbf{b}$ even when $A$ isn't invertible.

**Q4 (conceptual, L1 vs L2).** Why does L1 regularization produce sparse solutions while L2 doesn't?
**A.** Geometrically, the $L_1$ constraint region is a diamond with vertices on the coordinate axes; the loss's elliptical contours are likely to first intersect the constraint boundary exactly at a vertex, zeroing out some coefficients. The $L_2$ ball is smooth/round with no preferred axis-aligned points, so the intersection point generally has all coordinates non-zero (just shrunk). Analytically, the $L_1$ penalty's subgradient at zero is a constant-magnitude "pull" toward zero regardless of coefficient size, capable of overpowering a small gradient signal and driving weights exactly to 0; the $L_2$ penalty's gradient ($2\lambda w$) shrinks proportionally to $w$, so it gets weaker and weaker as $w\to0$ and never quite reaches it.

**Q5 (derivation).** Derive the eigenvalues of $A = \begin{bmatrix}4 & 1\\2 & 3\end{bmatrix}$.
**A.** $\det(A-\lambda I) = (4-\lambda)(3-\lambda)-2 = \lambda^2-7\lambda+10=0 \Rightarrow \lambda=5,2$.

**Q6 (applied).** How does PCA use eigendecomposition, step by step?
**A.** (1) Mean-center the data matrix $X$ (subtract each feature's mean). (2) Compute the covariance matrix $C=\frac{1}{n-1}X^TX$. (3) Eigendecompose $C=Q\Lambda Q^T$ ($C$ is symmetric PSD so this always works with real, non-negative eigenvalues). (4) Sort eigenvectors by descending eigenvalue — these are the principal component directions, and eigenvalues equal the variance explained along each. (5) Project data onto the top $k$ eigenvectors for dimensionality reduction: $X_{reduced}=XQ_k$.

**Q7 (conceptual, tricky).** Is SVD the same thing as eigendecomposition? Justify.
**A.** No. Eigendecomposition ($A=Q\Lambda Q^{-1}$) requires $A$ to be square and only guarantees real eigenvalues/orthogonal eigenvectors when $A$ is symmetric; it doesn't exist for all matrices. SVD ($A=U\Sigma V^T$) exists for *any* matrix, square or rectangular, and uses two different orthogonal bases ($U$, $V$) rather than one. For a symmetric PSD matrix, SVD and eigendecomposition coincide ($U=V=Q$, $\Sigma=\Lambda$).

**Q8 (applied, scenario).** Your design matrix $X$ has near-perfectly correlated features (multicollinearity). What linear-algebra phenomenon is occurring, and how would you fix it?
**A.** $X^TX$ becomes ill-conditioned (near-singular; smallest eigenvalue/singular value close to 0), so $(X^TX)^{-1}$ has huge entries — coefficient estimates become unstable and hypersensitive to small data perturbations. Fixes: add ridge regularization ($X^TX+\lambda I$, guaranteed invertible for $\lambda>0$), drop/combine correlated features, or use PCA to project onto orthogonal components first.

**Q9 (derivation, matrix calculus).** Derive $\nabla_w \|X\mathbf{w}-\mathbf{y}\|_2^2$ and set up the normal equations.
**A.** Expand $L=\mathbf{w}^TX^TX\mathbf{w}-2\mathbf{w}^TX^T\mathbf{y}+\mathbf{y}^T\mathbf{y}$. Since $X^TX$ is symmetric, $\nabla_w L=2X^TX\mathbf{w}-2X^T\mathbf{y}$. Setting to 0: $X^TX\mathbf{w}=X^T\mathbf{y} \Rightarrow \mathbf{w}^*=(X^TX)^{-1}X^T\mathbf{y}$.

**Q10 (advanced).** What's the relationship between the condition number of a matrix and gradient descent convergence?
**A.** Condition number $\kappa(A)=\sigma_{max}/\sigma_{min}$ (ratio of largest to smallest singular value). For quadratic losses, $A$ is (proportional to) the Hessian; a large $\kappa$ means the loss surface is a very elongated ellipse — gradient descent zig-zags slowly along the narrow direction, requiring many more iterations (convergence rate roughly scales with $\kappa$ for gradient descent vs. $\sqrt{\kappa}$ for accelerated/momentum methods, and is condition-number-independent for Newton's method). This motivates feature normalization (which improves conditioning) and adaptive optimizers.

**Q11 (advanced, tricky).** Why is Cholesky decomposition preferred over general matrix inversion for sampling from a multivariate Gaussian?
**A.** A valid covariance matrix $\Sigma$ is symmetric PD, so $\Sigma=LL^T$ (Cholesky) exists uniquely and is computed in $O(n^3/3)$ — about half the cost of general inversion/decomposition — and is numerically more stable since it exploits symmetry and positive-definiteness. To sample $\mathbf{x}\sim\mathcal{N}(\mu,\Sigma)$: draw $\mathbf{z}\sim\mathcal{N}(0,I)$, then $\mathbf{x}=\mu+L\mathbf{z}$, since $\text{Cov}(L\mathbf{z})=L\,\text{Cov}(\mathbf{z})\,L^T=LL^T=\Sigma$.

**Q12 (advanced, applied).** In a Transformer's self-attention, why does scaling $QK^T$ by $1/\sqrt{d_k}$ matter mathematically?
**A.** If $Q,K$ entries are i.i.d. with variance 1, each dot-product entry of $QK^T$ (sum of $d_k$ products) has variance $\approx d_k$, so its magnitude grows with $\sqrt{d_k}$. Without scaling, large-magnitude logits push softmax into a near one-hot, saturated regime where gradients vanish almost everywhere except the max. Dividing by $\sqrt{d_k}$ normalizes the variance back to $\approx1$, keeping softmax's input in a well-conditioned range with healthy gradients — a direct application of variance-of-a-sum reasoning plus norm/scale intuition.

**Q13 (tricky, gotcha).** Can a covariance matrix have negative eigenvalues? Why or why not?
**A.** No — a valid (theoretical) covariance matrix is always PSD by construction, since for any vector $\mathbf{a}$, $\mathbf{a}^T\Sigma\mathbf{a}=\text{Var}(\mathbf{a}^T\mathbf{x})\ge0$. If you compute a "covariance matrix" empirically and it comes out with a small negative eigenvalue, that's a numerical artifact (floating-point error, or $n<d$ making the sample covariance rank-deficient with eigenvalues that are supposed to be exactly zero but round to slightly negative) — not a true property of the covariance structure.

**Q14 (advanced).** Explain how truncated SVD relates to the "best rank-$k$ approximation" claim, and why that matters for recommender systems.
**A.** Eckart-Young theorem: among all rank-$k$ matrices, $A_k=\sum_{i=1}^k \sigma_i u_i v_i^T$ (keeping the top-$k$ singular triplets) minimizes both the Frobenius norm and spectral norm of the approximation error $\|A-A_k\|$. In recommender systems, the user-item ratings matrix is huge and sparse; approximating it with a low-rank $A_k$ captures the dominant latent factors (e.g., genre preference) driving ratings while filtering noise, and lets you predict missing entries (unrated items) via the reconstructed $A_k$ — the mathematical basis of classic matrix-factorization collaborative filtering.

---

## Calculus and Optimization

*Primary audience: **(MLE)** for training-loop internals and optimizer tuning, **(All)** for understanding why models converge or fail to.*

### Derivatives, Partial Derivatives, Gradients, Directional Derivatives

**Derivative** (single variable): $f'(x) = \lim_{h\to0} \frac{f(x+h)-f(x)}{h}$ — the instantaneous rate of change / slope of the tangent line.

**Partial derivative** (multivariable $f(x_1,\dots,x_n)$): $\frac{\partial f}{\partial x_i}$ holds all other variables fixed and differentiates w.r.t. $x_i$ alone.

**Gradient**: the vector of all partial derivatives, $\nabla f = \left(\frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n}\right)$. Points in the direction of **steepest ascent**; its negative points in the direction of steepest descent (why gradient descent moves along $-\nabla f$).

**Directional derivative**: rate of change of $f$ along an arbitrary unit direction $\mathbf{u}$: $D_{\mathbf{u}}f = \nabla f \cdot \mathbf{u} = \|\nabla f\|\cos\theta$, where $\theta$ is the angle between $\nabla f$ and $\mathbf{u}$. This is maximized when $\mathbf{u}$ points exactly along $\nabla f$ ($\theta=0$), formally proving gradient = steepest-ascent direction.

**Worked example.** $f(x,y)=x^2y+3y^2$. $\nabla f = (2xy,\ x^2+6y)$. At $(1,2)$: $\nabla f=(4,13)$. Directional derivative along $\mathbf{u}=(1/\sqrt2,1/\sqrt2)$: $D_uf = 4/\sqrt2+13/\sqrt2=17/\sqrt2\approx12.02$.

**Why it matters in ML.** The gradient is *the* fundamental object optimizers use to update parameters: $\theta \leftarrow \theta-\eta\nabla_\theta L$. Every "vanishing/exploding gradient" problem is about the magnitude of these partial derivatives shrinking or blowing up as they're propagated through many layers.

**Pitfalls.** A zero gradient doesn't necessarily mean a minimum — could be a saddle point or maximum (need second-order info to distinguish, see Hessian section). Non-differentiable points (e.g., ReLU at 0, $|x|$ at 0) require subgradients — a generalization allowing a *set* of valid "slopes."

### Chain Rule and Backpropagation

**Single-variable chain rule**: if $y=f(g(x))$, then $\frac{dy}{dx}=f'(g(x))\cdot g'(x)$.

**Multivariable chain rule** (the one backprop actually uses): if $z=f(y_1,\dots,y_k)$ and each $y_i=g_i(x_1,\dots,x_n)$, then
```
∂z/∂x_j = Σ_i (∂z/∂y_i) * (∂y_i/∂x_j)
```
— sum over *all paths* by which $x_j$ influences $z$.

**Backpropagation as chain-rule bookkeeping.** For a network $L = \ell(a^{[K]})$ where $a^{[k]} = \sigma(z^{[k]})$, $z^{[k]}=W^{[k]}a^{[k-1]}+b^{[k]}$, define $\delta^{[k]} = \frac{\partial L}{\partial z^{[k]}}$. Backprop recursion:
```
δ^[K] = ∇_a L ⊙ σ'(z^[K])                     (output layer)
δ^[k] = (W^[k+1])^T δ^[k+1] ⊙ σ'(z^[k])        (hidden layers, propagated backward)
∂L/∂W^[k] = δ^[k] (a^[k-1])^T
∂L/∂b^[k] = δ^[k]
```
This is literally the multivariate chain rule applied layer by layer, computed backward (output→input) for efficiency: computing gradients this way costs roughly the same as one forward pass ($O(\text{\#params})$), whereas naively perturbing each parameter one at a time (finite differences) would cost $O(\text{\#params}^2)$.

**Worked mini-example.** $x=2$, $y = \sigma(wx+b)$ with $w=0.5,b=0$, $\sigma$=sigmoid, target $t=1$, $L=\frac{1}{2}(y-t)^2$.
```
z = wx+b = 1.0
y = σ(1.0) ≈ 0.7311
L = 0.5*(0.7311-1)² ≈ 0.0362
∂L/∂y = y - t = -0.2689
∂y/∂z = y(1-y) ≈ 0.7311*0.2689 ≈ 0.1966
∂L/∂z = ∂L/∂y * ∂y/∂z ≈ -0.0529
∂L/∂w = ∂L/∂z * x ≈ -0.1057
```
This is exactly what `loss.backward()` computes symbolically/automatically for arbitrarily deep networks.

**Why it matters in ML.** Backpropagation *is* the multivariate chain rule; every deep learning framework's autograd engine builds a computation graph and applies this recursively. Understanding it explains **vanishing gradients** (repeatedly multiplying by sigmoid/tanh derivatives that are $\le0.25$/$\le1$ shrinks the signal exponentially with depth) and **exploding gradients** (repeatedly multiplying by large weight-matrix norms grows it exponentially) — directly motivating ReLU activations, residual/skip connections (which add an identity path so gradients don't have to survive multiplication through every layer), and gradient clipping.

**Pitfalls.** Forgetting the Hadamard (element-wise) product $\odot$ vs matrix multiplication when activation derivatives are involved. Believing backprop "recomputes the forward pass" — it reuses cached forward activations, which is why memory (not just compute) scales with depth (motivating gradient checkpointing).

### Jacobians and Hessians, Second-Order Optimization

**Jacobian**: for a vector-valued function $\mathbf{f}:\mathbb{R}^n\to\mathbb{R}^m$, the Jacobian $J\in\mathbb{R}^{m\times n}$ collects all first partial derivatives:
```
J_ij = ∂f_i/∂x_j
```
It's the best *linear* approximation to $\mathbf{f}$ near a point (generalizing the derivative/gradient to vector outputs).

**Hessian**: for scalar $f:\mathbb{R}^n\to\mathbb{R}$, the Hessian $H\in\mathbb{R}^{n\times n}$ collects second partial derivatives:
```
H_ij = ∂²f/∂x_i∂x_j
```
By **Clairaut/Schwarz's theorem**, if $f$ is $C^2$ (continuous second partials), $H$ is symmetric ($H_{ij}=H_{ji}$).

**Second-order optimization (Newton's method).** Update rule: $\theta \leftarrow \theta - H^{-1}\nabla f$. Intuition: gradient descent takes a fixed-size step; Newton's method uses curvature to jump straight to the minimum of the *local quadratic approximation* in one step — converges quadratically near the optimum (vs linear for gradient descent) but costs $O(n^3)$ per step to invert $H$ (infeasible for models with millions/billions of params) and can diverge or move toward saddle points if $H$ isn't PD.

**Worked example.** $f(x,y)=x^2+3y^2$. $\nabla f=(2x,6y)$, $H=\begin{bmatrix}2&0\\0&6\end{bmatrix}$ (constant, since $f$ is purely quadratic). Newton's step from any point jumps directly to $(0,0)$ in one iteration since the quadratic model is *exact* here; gradient descent would need many small steps, especially zig-zagging because $H$'s eigenvalues (2 and 6) differ (condition number 3).

**Why it matters in ML.** Full Hessians are intractable for deep nets (billions of parameters → Hessian has billions² entries), so practical methods use **Hessian-free approximations**: quasi-Newton methods (L-BFGS, common for smaller convex problems / fine-tuning), or diagonal/low-rank approximations that adaptive optimizers (Adam, RMSProp) implicitly use via squared gradients as a cheap curvature proxy. The Hessian's eigenvalue spectrum around a trained model's minimum is used in research to study "sharp vs flat minima" and generalization. The Jacobian is central to **backprop through vector-valued layers** (e.g., attention outputs, sequence models) and to **adversarial robustness** analysis (Jacobian of output w.r.t. input governs sensitivity to input perturbations).

**Pitfalls.** Confusing Jacobian (first-order, can be non-square, for vector functions) with Hessian (second-order, always square, for scalar functions). Assuming Adam is a true second-order method — it's not; it uses only first-moment and second-*moment-of-gradient* (not second derivative) statistics, an important interview distinction.

### Taylor Series Approximation

**General Taylor expansion** of $f$ around point $a$:
```
f(x) = f(a) + f'(a)(x-a) + f''(a)/2! * (x-a)² + f'''(a)/3! * (x-a)³ + ...
```
**Multivariable second-order Taylor expansion** (the one optimization theory uses constantly):
```
f(x) ≈ f(a) + ∇f(a)^T (x-a) + 1/2 (x-a)^T H(a) (x-a)
```
This is the **local quadratic approximation** — a paraboloid matching $f$'s value, slope, and curvature at $a$.

**Why it matters in ML.**
- **Gradient descent derivation**: dropping the second-order term, $f(x)\approx f(a)+\nabla f(a)^T(x-a)$; moving in direction $-\nabla f(a)$ guarantees local decrease for small enough steps — this *is* the theoretical justification for gradient descent.
- **Newton's method derivation**: keep the second-order term and minimize the quadratic approximation exactly by setting its gradient to zero: $\nabla f(a) + H(a)(x-a)=0 \Rightarrow x = a - H(a)^{-1}\nabla f(a)$ — exactly the Newton update.
- **Classifying critical points**: at a point where $\nabla f=0$, the sign-definiteness of $H$ (from the second-order term, since first-order vanishes) determines whether it's a min (PD), max (ND), or saddle (indefinite) — this is the **second derivative test** generalized to $n$ dimensions.
- Used to justify **loss landscape smoothness assumptions** (Lipschitz-continuous gradients) behind convergence proofs for SGD variants.

**Worked example.** $f(x)=e^x$ near $a=0$: $f(x)\approx1+x+x^2/2$. At $x=0.1$: approx $=1.105$, true $e^{0.1}\approx1.10517$ — very close, illustrating how good local quadratic approximations are for small steps (which is why gradient-based methods work well with appropriately small learning rates).

**Pitfalls.** Taylor approximations are *local* — valid only near the expansion point; large steps (large learning rates) can leave the region where the approximation holds, causing overshoot/divergence, which is exactly why learning rate tuning matters so much.

### Convexity: Convex Functions, Convex Sets, Global vs Local Minima

**Convex set**: a set $S$ is convex if for any $\mathbf{x},\mathbf{y}\in S$ and $t\in[0,1]$, $t\mathbf{x}+(1-t)\mathbf{y}\in S$ (the line segment between any two points stays inside the set).

**Convex function**: $f$ is convex if its domain is a convex set and
```
f(t x + (1-t) y) ≤ t f(x) + (1-t) f(y)   for all x,y, t∈[0,1]
```
Geometrically: the line segment (chord) connecting any two points on the graph lies *above* the graph. Equivalent first-order condition (differentiable $f$): $f(y) \ge f(x)+\nabla f(x)^T(y-x)$ for all $x,y$ (the function lies above every tangent line/plane). Equivalent second-order condition (twice-differentiable $f$): Hessian $H(x)$ is PSD for all $x$ in the domain.

**Strict convexity**: strict inequality above; guarantees at most one global minimum (no flat regions).

**Key theorem — why convexity matters so much**: for a convex function, **every local minimum is a global minimum**, and there's no distinction between "getting stuck" and "succeeding" — any local optimization method that reaches a stationary point ($\nabla f=0$) has found the global optimum.

**Worked example.** $f(x)=x^2$: $f''(x)=2>0$ everywhere → strictly convex → unique global min at $x=0$. $f(x)=x^4-3x^2$: $f''(x)=12x^2-6$, negative for $|x|<\sqrt{0.5}$ → **not convex** (it's actually got two symmetric global minima and a local max at 0) — a classic example of a **non-convex** function with multiple local optima, much like typical deep-learning loss landscapes.

**Convex vs non-convex ML losses**: Linear/logistic regression losses (MSE, log-loss) are convex in the parameters — guaranteed global convergence with gradient descent given appropriate learning rate. SVM's hinge-loss objective is convex. **Deep neural network losses are generally non-convex** in their parameters (due to compositions of nonlinear activations and layer-permutation symmetries creating many equivalent minima/saddle points) — this is why deep learning theory focuses heavily on saddle-point escape, loss landscape geometry, and empirical tricks (initialization schemes, batch norm, skip connections, appropriate learning rates) rather than convexity guarantees.

**Why it matters in ML.** Determines whether you can trust that gradient descent finds *the* best solution (convex) vs merely *a* reasonably good one (non-convex, common for deep nets). Motivates convex relaxations (e.g., replacing 0/1 loss with convex surrogates like hinge or logistic loss) and understanding SVM duality (only tractable because the primal is convex, satisfying strong duality via Slater's condition).

**Pitfalls.**
- A function can have $f''\ge0$ but not be *strictly* convex (e.g., a linear function is convex but not strictly — infinitely many global minima along a flat region if combined with constraints).
- "Local minimum in a non-convex problem" doesn't mean "bad" — empirically, most local minima found by SGD in over-parameterized deep nets generalize nearly as well as any other, a nontrivial and actively researched fact.
- Sum of convex functions is convex; but convex ∘ convex composition is NOT generally convex unless the outer function is also non-decreasing — a frequent trick question.

### Gradient Descent Variants

**Vanilla (batch) gradient descent**: use the *full* dataset gradient each step:
```
θ ← θ - η ∇_θ L(θ; entire dataset)
```
Accurate gradient direction but expensive per step and impractical for large datasets; can get stuck precisely at saddle points due to the lack of noise.

**Stochastic gradient descent (SGD)**: use a single (random) sample's gradient per step:
```
θ ← θ - η ∇_θ L(θ; x_i, y_i)
```
Noisy but cheap steps; the noise itself helps escape shallow local minima/saddle points, at the cost of a noisier convergence path.

**Mini-batch SGD**: the practical standard — average gradient over a small batch ($B$ samples):
```
θ ← θ - η * (1/B) Σ_{i∈batch} ∇_θ L(θ; x_i, y_i)
```
Balances gradient-estimate variance (↓ as batch size ↑) against compute efficiency (parallelizable) and generalization (very large batches can generalize worse without other adjustments).

**Momentum**: accumulate an exponentially decaying moving average of past gradients to smooth the trajectory and accelerate through consistent-direction ("valley") regions:
```
v_t = β v_{t-1} + (1-β) ∇_θ L(θ_t)      (β typically 0.9)
θ_t+1 = θ_t - η v_t
```
Intuition: a heavy ball rolling downhill — it builds up speed in a consistent direction and damps oscillation across a narrow valley (helps with the ill-conditioning problem discussed in the Hessian section).

**Nesterov Accelerated Gradient (NAG)**: evaluates the gradient at the "look-ahead" point (where momentum is about to carry you), giving a corrective effect:
```
v_t = β v_{t-1} + ∇_θ L(θ_t - η β v_{t-1})
θ_t+1 = θ_t - η v_t
```
Provably faster convergence rate than plain momentum for convex problems.

**AdaGrad**: adapts the learning rate *per parameter*, dividing by the accumulated sum of squared past gradients — large, frequently-updated gradients get smaller effective steps, sparse/rare features get relatively larger steps:
```
G_t = G_{t-1} + (∇_θ L)²        (element-wise, accumulated forever)
θ ← θ - η/√(G_t + ε) * ∇_θ L
```
Weakness: $G_t$ only grows, so the effective learning rate monotonically shrinks toward zero — training can stall prematurely, especially in non-convex/deep settings.

**RMSProp**: fixes AdaGrad's decay problem by using an *exponentially decaying* moving average of squared gradients instead of an ever-growing sum:
```
E[g²]_t = γ E[g²]_{t-1} + (1-γ) (∇_θ L)²
θ ← θ - η/√(E[g²]_t + ε) * ∇_θ L
```
Effectively "forgets" old gradient history, so the learning rate can grow again if gradients shrink — much better for non-stationary/deep-learning objectives.

**Adam** (Adaptive Moment Estimation) — combines momentum (1st moment) + RMSProp (2nd moment), plus bias correction for early-step initialization artifacts:
```
m_t = β1 m_{t-1} + (1-β1) g_t              (1st moment / mean, typically β1=0.9)
v_t = β2 v_{t-1} + (1-β2) g_t²             (2nd moment / uncentered variance, β2=0.999)
m̂_t = m_t / (1 - β1^t)                    (bias correction)
v̂_t = v_t / (1 - β2^t)
θ ← θ - η * m̂_t / (√v̂_t + ε)
```
The bias correction matters because $m_0=v_0=0$, so early estimates are biased toward zero without it — dividing by $(1-\beta^t)$ compensates, mattering most in the first few dozen steps.

**AdamW**: fixes a subtle but important bug/design flaw in Adam's interaction with L2 regularization. In vanilla Adam, adding an $L_2$ penalty ($+\lambda\|\theta\|^2$ to the loss) gets folded into the gradient $g_t$ *before* the adaptive scaling, so the effective weight decay ends up scaled by $1/\sqrt{\hat v_t}$ — parameters with large historical gradients get *less* decay, which is not the intended, uniform weight-decay behavior. AdamW **decouples** weight decay from the gradient-based update, applying it directly and separately:
```
θ ← θ - η * ( m̂_t / (√v̂_t + ε) + λ θ )
```
This decoupled version is now the de facto standard for training Transformers/LLMs.

**Comparison table:**

| Optimizer | Adapts LR per-param? | Uses momentum? | Key weakness |
|---|---|---|---|
| SGD | No | No | Slow, sensitive to conditioning, needs LR tuning |
| Momentum | No | Yes | Can overshoot with high β |
| Nesterov | No | Yes (look-ahead) | Same tuning sensitivity as momentum |
| AdaGrad | Yes | No | LR decays to ~0 over long training |
| RMSProp | Yes | No | No momentum smoothing |
| Adam | Yes | Yes | L2/weight-decay coupling issue (fixed by AdamW); can generalize slightly worse than well-tuned SGD in some vision tasks |
| AdamW | Yes | Yes | Still needs LR/schedule tuning; extra hyperparameters (β1, β2, ε, λ) |

**Why it matters in ML.** Optimizer choice and hyperparameters (especially learning rate) are usually the single highest-leverage training decision an MLE makes; understanding *why* Adam/AdamW work (adaptive per-parameter scaling handling features/gradients of wildly different scales, e.g., in Transformers with embedding vs. attention-weight gradients) is a top interview differentiator from someone who just imports `torch.optim.AdamW`.

**Pitfalls.** Believing Adam always beats SGD — for some CNN/vision tasks, well-tuned SGD+momentum generalizes better. Forgetting that Adam's adaptivity means different effective learning rates per parameter — so a "learning rate" of 1e-3 in Adam is not comparable to 1e-3 in SGD. Using standard (non-decoupled) Adam with weight decay when AdamW is intended — a very common real-world bug (many PyTorch users unknowingly use `Adam` + manual `weight_decay` param, not realizing it's coupled).

### Learning Rate Schedules, Warmup, Convergence Criteria

**Why schedule the learning rate.** A fixed learning rate trades off: too large → divergence/oscillation around (but never reaching) the minimum; too small → painfully slow convergence. Schedules aim to be large early (fast progress) and small late (fine convergence).

**Common schedules:**
- **Step decay**: multiply $\eta$ by a factor (e.g., 0.1) every $N$ epochs.
- **Exponential decay**: $\eta_t = \eta_0 e^{-kt}$.
- **Cosine annealing**: $\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max}-\eta_{min})\left(1+\cos\left(\frac{t}{T}\pi\right)\right)$ — smooth decay to near-zero by the end of training, widely used and empirically strong.
- **Cosine with warm restarts (SGDR)**: periodically resets $\eta$ back up following the cosine curve, helping escape sharp minima repeatedly.
- **ReduceLROnPlateau**: adaptive — cut $\eta$ when validation metric stops improving for $k$ epochs.

**Warmup.** Start training with a very small learning rate and linearly (or otherwise) ramp it up over the first few hundred/thousand steps before switching to the main schedule (decay). Critical for **Transformer** training and large-batch training: early in training, weights are randomly initialized, gradient estimates and Adam's second-moment estimates ($v_t$) are unreliable/noisy, so a large step immediately can cause instability (loss spikes, divergence) before the adaptive statistics have "warmed up" to meaningful values.
```
η_t = η_max * (t / t_warmup)      for t ≤ t_warmup
```
followed by decay thereafter (e.g., the original Transformer paper's `η ∝ min(t^{-0.5}, t·t_warmup^{-1.5})`).

**Convergence criteria.** In practice: (a) validation loss/metric stops improving for $k$ epochs (**early stopping**), (b) gradient norm falls below a threshold ($\|\nabla L\|<\epsilon$), (c) parameter update magnitude falls below a threshold, (d) fixed compute/epoch budget (common for LLM pretraining, where "convergence" in the classical sense is often not the practical stopping criterion — compute-optimal scaling laws are).

**Why it matters in ML.** Poor LR scheduling is one of the most common causes of a "model that just doesn't train well" in interviews and practice — e.g., LLM pretraining loss spikes/diverges without warmup; models plateau early without decay. Understanding cosine annealing / warmup is table-stakes for any AIE/MLE fine-tuning an LLM.

**Pitfalls.** Forgetting to warm up when using large batch sizes (large batch → less gradient noise → larger *effective* steps → higher divergence risk, needing either warmup or LR scaling rules like "linear scaling rule": scale $\eta$ linearly with batch size). Using early stopping on a metric so noisy that you stop on a lucky/unlucky fluctuation rather than a true plateau — mitigated by patience windows / smoothing.

### Constrained Optimization: Lagrange Multipliers, KKT Conditions

**Lagrange multipliers (equality constraints).** To minimize $f(\mathbf{x})$ subject to $g(\mathbf{x})=0$, form the **Lagrangian**:
```
L(x, λ) = f(x) + λ g(x)
```
At the constrained optimum, $\nabla f(\mathbf{x}) = -\lambda\nabla g(\mathbf{x})$ — the gradients of the objective and constraint are parallel (if they weren't, you could move along the constraint surface to further decrease $f$). Solve the system $\nabla_x L=0,\ \nabla_\lambda L=0$ simultaneously.

**Worked example.** Minimize $f(x,y)=x^2+y^2$ subject to $x+y=1$. $L=x^2+y^2+\lambda(x+y-1)$. $\partial L/\partial x=2x+\lambda=0$, $\partial L/\partial y=2y+\lambda=0 \Rightarrow x=y$. With $x+y=1 \Rightarrow x=y=0.5$, giving minimum value $f=0.5$.

**KKT conditions (inequality + equality constraints).** For minimizing $f(\mathbf{x})$ subject to $g_i(\mathbf{x})\le0$ ($i=1..m$) and $h_j(\mathbf{x})=0$ ($j=1..p$), form the Lagrangian $L(\mathbf{x},\boldsymbol\mu,\boldsymbol\lambda)=f(\mathbf{x})+\sum_i\mu_i g_i(\mathbf{x})+\sum_j\lambda_j h_j(\mathbf{x})$. The **Karush-Kuhn-Tucker (KKT) conditions**, necessary for optimality (and sufficient for convex problems):

1. **Stationarity**: $\nabla_x L = 0$
2. **Primal feasibility**: $g_i(\mathbf{x})\le0$, $h_j(\mathbf{x})=0$
3. **Dual feasibility**: $\mu_i \ge 0$
4. **Complementary slackness**: $\mu_i g_i(\mathbf{x}) = 0$ for all $i$ (either the constraint is *active*, $g_i=0$, or its multiplier is zero, $\mu_i=0$ — you can't have "slack" in the constraint *and* a non-zero penalty simultaneously)

**Link to SVM dual.** The (soft-margin) SVM primal problem is:
```
minimize   (1/2)‖w‖² + C Σ ξ_i
subject to y_i(w^T x_i + b) ≥ 1 - ξ_i,   ξ_i ≥ 0
```
Applying KKT/Lagrangian duality (this problem is convex, so strong duality holds — the dual solves the exact same optimum as the primal) yields the dual:
```
maximize   Σ α_i - (1/2) Σ_i Σ_j α_i α_j y_i y_j (x_i^T x_j)
subject to 0 ≤ α_i ≤ C,   Σ α_i y_i = 0
```
**Why this matters**: (1) the dual depends on training data *only* through dot products $x_i^Tx_j$ — this is exactly what enables the **kernel trick** (replace the dot product with a kernel function $K(x_i,x_j)$ computing an implicit dot product in a higher-dimensional feature space, without ever materializing that space); (2) complementary slackness says $\alpha_i>0$ only for points *on or violating* the margin — these are the **support vectors**; all other points have $\alpha_i=0$ and don't affect the decision boundary at all, explaining SVM's sparsity and the name.

**Why it matters in ML.** Beyond SVMs: constrained optimization underlies portfolio optimization, resource-constrained model deployment (e.g., minimizing loss subject to a latency/parameter budget), PCA (can be derived as maximizing variance subject to $\|\mathbf{w}\|=1$, itself solved via Lagrange multipliers — leading directly back to the eigenvector equation!), and the derivation of reinforcement learning trust-region methods (TRPO's KL-constrained policy update).

**Pitfalls.** Sign convention errors for $\mu_i\ge0$ depend on whether constraints are written as $g\le0$ or $g\ge0$ — always double check. Complementary slackness is often the "aha" fact interviewers are fishing for when asking "why are most $\alpha_i=0$ in SVM." Forgetting that KKT conditions are only *sufficient* for global optimality when the problem is convex; for non-convex problems, KKT points can be saddle points too.

### Interview Questions — Calculus and Optimization

**Q1 (basic).** What is the gradient, and in what direction does it point?
**A.** The gradient $\nabla f$ is the vector of all partial derivatives of a scalar function; it points in the direction of steepest *ascent* of $f$ at that point, with magnitude equal to the rate of increase in that direction. Gradient descent moves in $-\nabla f$ because that's the steepest *descent* direction.

**Q2 (basic).** State the multivariable chain rule and explain its role in backpropagation.
**A.** If $z$ depends on $y_1,\dots,y_k$ and each $y_i$ depends on $x_j$, then $\partial z/\partial x_j = \sum_i (\partial z/\partial y_i)(\partial y_i/\partial x_j)$ — summing contributions over every path of influence. Backpropagation applies this recursively, layer by layer, from the output loss back to each parameter, computing all gradients in one backward pass instead of separately perturbing each parameter.

**Q3 (conceptual).** What's the difference between a Jacobian and a Hessian?
**A.** The Jacobian is the matrix of first-order partial derivatives of a *vector-valued* function ($\mathbb{R}^n\to\mathbb{R}^m$), giving the best linear approximation. The Hessian is the matrix of *second*-order partial derivatives of a *scalar-valued* function, always square and (for smooth functions) symmetric, capturing local curvature.

**Q4 (derivation).** Derive gradient descent's update rule from the first-order Taylor expansion.
**A.** $f(\theta+\Delta\theta)\approx f(\theta)+\nabla f(\theta)^T\Delta\theta$. To decrease $f$ as much as possible for a step of fixed small size, choose $\Delta\theta$ anti-parallel to $\nabla f(\theta)$: $\Delta\theta=-\eta\nabla f(\theta)$, giving $\theta\leftarrow\theta-\eta\nabla f(\theta)$, guaranteeing $f(\theta+\Delta\theta)<f(\theta)$ for sufficiently small $\eta$ (as long as $\nabla f\neq0$).

**Q5 (conceptual).** Why can Newton's method converge faster than gradient descent but is rarely used for deep learning?
**A.** Newton's method uses curvature (the Hessian) to jump to the minimum of the local quadratic model in one step, giving quadratic convergence near the optimum vs. gradient descent's linear convergence. But for deep nets with millions/billions of parameters, forming and inverting an $n\times n$ Hessian is computationally infeasible ($O(n^3)$ for inversion, $O(n^2)$ just to store it), and the Hessian may not be PD away from a true minimum, risking divergence toward saddle points.

**Q6 (conceptual).** Why is the loss surface of a deep neural network generally non-convex, and does that mean training is hopeless?
**A.** Non-convex due to compositions of nonlinear activations, and there also exist many equivalent minima from permutation/scaling symmetries of hidden units. It's not hopeless: empirically, SGD-family optimizers in over-parameterized networks tend to find "good enough" local minima/wide flat basins that generalize well; theoretically, most critical points in high dimensions for these networks tend to be saddle points rather than bad local minima, and momentum/noise from SGD helps escape saddles.

**Q7 (derivation).** Derive the softmax + cross-entropy gradient $\partial L/\partial z = p - y$.
**A.** $p_i=e^{z_i}/\sum_k e^{z_k}$, $L=-\sum_i y_i\log p_i$. For $i=j$: $\partial p_i/\partial z_i = p_i(1-p_i)$; for $i\neq j$: $\partial p_i/\partial z_j=-p_ip_j$. Then $\partial L/\partial z_j = -\sum_i y_i \frac{1}{p_i}\frac{\partial p_i}{\partial z_j}$. Splitting the $i=j$ and $i\neq j$ terms and simplifying (using $\sum_i y_i=1$ for one-hot labels) collapses to $\partial L/\partial z_j = p_j - y_j$.

**Q8 (applied).** Explain the difference between SGD, momentum, and Adam in terms of what statistics each tracks, and when you'd prefer one.
**A.** Plain SGD uses only the current gradient. Momentum additionally tracks an exponential moving average of past gradients (1st moment) to smooth updates and accelerate through consistent directions. Adam tracks both the 1st moment (like momentum) and the 2nd moment (moving average of squared gradients) to adapt the effective learning rate per parameter, plus bias-correction for early steps. Prefer plain SGD+momentum when you have time to tune a schedule and want potentially better generalization (common in vision/CNN training); prefer Adam/AdamW for fast, robust convergence with less tuning, especially for Transformers/LLMs and sparse-gradient settings (e.g., embeddings).

**Q9 (derivation, tricky).** Show why Adam needs bias correction in early training steps.
**A.** $m_t=(1-\beta_1)\sum_{i=1}^t\beta_1^{t-i}g_i$; if all $g_i\approx g$ (constant), $E[m_t]\approx g(1-\beta_1^t)$, which is *not* $g$ unless $t\to\infty$ — early estimates are biased toward 0 because $m_0=0$ and $\beta_1$ is close to 1. Dividing by $(1-\beta_1^t)$ (and similarly $(1-\beta_2^t)$ for $v_t$) corrects this bias so $E[\hat m_t]\approx g$ even for small $t$.

**Q10 (conceptual).** What problem does AdamW solve that vanilla Adam + L2 penalty does not?
**A.** In vanilla Adam, adding an $L_2$ penalty to the loss folds the weight-decay term into the gradient before it's divided by $\sqrt{\hat v_t}$, so parameters with historically large gradients get proportionally *less* decay — an unintended, inconsistent regularization strength across parameters. AdamW decouples weight decay, subtracting $\lambda\theta$ directly from the parameter update outside the adaptive-scaling term, giving uniform, well-behaved regularization — now standard for training Transformers.

**Q11 (applied, scenario).** Your Transformer's training loss spikes to NaN within the first 100 steps. Diagnose using optimization theory.
**A.** Likely causes tied to gradient/curvature instability early in training: (1) no learning-rate warmup — Adam's second-moment estimate $v_t$ is unreliable/near-zero early, so $\eta/\sqrt{\hat v_t}$ can be enormous, causing huge steps; (2) learning rate too high generally, pushing outside the region where the local quadratic (Taylor) approximation underlying gradient-based updates is valid, causing divergence; (3) missing gradient clipping, letting a rare large gradient explode through the adaptive normalization. Fix: add linear warmup, clip gradient norm (e.g., to 1.0), and/or lower peak LR.

**Q12 (advanced, derivation).** Derive the SVM dual from the primal using Lagrangian duality, and explain complementary slackness's role in identifying support vectors.
**A.** Primal: minimize $\frac12\|w\|^2$ s.t. $y_i(w^Tx_i+b)\ge1$. Lagrangian: $L=\frac12\|w\|^2-\sum_i\alpha_i[y_i(w^Tx_i+b)-1]$, $\alpha_i\ge0$. Stationarity: $\partial L/\partial w=w-\sum_i\alpha_iy_ix_i=0\Rightarrow w=\sum_i\alpha_iy_ix_i$; $\partial L/\partial b=-\sum_i\alpha_iy_i=0$. Substituting back gives the dual $\max_\alpha \sum_i\alpha_i-\frac12\sum_{i,j}\alpha_i\alpha_jy_iy_jx_i^Tx_j$ s.t. $\alpha_i\ge0,\sum_i\alpha_iy_i=0$. Complementary slackness: $\alpha_i[y_i(w^Tx_i+b)-1]=0$ — so $\alpha_i>0$ only when the point lies exactly on the margin (constraint active); points strictly outside the margin have $\alpha_i=0$ and don't contribute to $w$ — these active points are the support vectors.

**Q13 (advanced, tricky).** Why does batch size affect the "effective" learning rate, and what's the linear scaling rule?
**A.** Larger batches give a lower-variance (more accurate) gradient estimate, effectively making each step "trust" the gradient direction more, which behaves similarly to taking a larger, more confident step. Empirically (and semi-theoretically, from the SGD noise/diffusion analogy) this motivates the **linear scaling rule**: when you multiply batch size by $k$, multiply the learning rate by $k$ as well (with a warmup period), to preserve similar training dynamics/number-of-epochs-to-convergence — though this scaling breaks down at very large batch sizes, requiring warmup and sometimes other stabilizers (e.g., LARS/LAMB optimizers).

**Q14 (advanced).** What does it mean for the Hessian to be "ill-conditioned," and how do different optimizers mitigate it?
**A.** Ill-conditioning means the Hessian's eigenvalues span a wide range (large condition number $\kappa=\lambda_{max}/\lambda_{min}$) — the loss surface looks like a long, narrow valley. Plain gradient descent zig-zags: it takes steps sized for the steep direction (small, to avoid overshoot) even along the flat direction (where it could safely move much further), so convergence is slow. Momentum accumulates velocity along consistently-signed (flat) directions while the oscillating steep direction partially cancels out. Adaptive methods (Adam/RMSProp/AdaGrad) explicitly divide each coordinate's step by its own gradient-magnitude history, approximating a diagonal preconditioner that rescales each direction toward similar effective step sizes — a cheap, diagonal approximation to what Newton's method does exactly with the full inverse Hessian.

**Q15 (tricky, gotcha).** True or false: a critical point where the gradient is zero is always a local minimum. Explain with an example.
**A.** False. $\nabla f=0$ just means the point is *stationary* — it could be a local min, local max, or saddle point. Example: $f(x,y)=x^2-y^2$ has $\nabla f(0,0)=(0,0)$ but the Hessian $\begin{bmatrix}2&0\\0&-2\end{bmatrix}$ is indefinite → $(0,0)$ is a saddle point (minimum along $x$, maximum along $y$). In high-dimensional non-convex deep learning loss surfaces, saddle points are in fact far more numerous than genuine local minima, which is a major reason plain gradient descent (without momentum/noise) can stall.

---

## Information Theory

*Primary audience: **(All)** for cross-entropy loss, **(DS)** for feature selection/mutual information, **(AIE)** for perplexity/LLM evaluation.*

### Entropy, Cross-Entropy, Joint Entropy, Conditional Entropy

**Shannon entropy** measures the average uncertainty/information content of a random variable $X$ with distribution $p$:
```
H(X) = -Σ_x p(x) log p(x)          (discrete)
     = -∫ p(x) log p(x) dx          (continuous, "differential entropy")
```
Units: **bits** if $\log_2$, **nats** if $\log_e$ (natural log) — ML frameworks almost always use natural log (nats) internally. Intuition: entropy is the expected number of bits needed to optimally encode outcomes of $X$ (Shannon's source coding theorem) — high entropy = unpredictable/uniform distribution (max uncertainty), low entropy = predictable/peaked distribution (min uncertainty). $H(X)=0$ iff $X$ is deterministic (all probability mass on one outcome). Maximum entropy for $n$ outcomes is $\log n$, achieved by the uniform distribution.

**Worked example.** Fair coin: $H = -[0.5\log_2 0.5 + 0.5\log_2 0.5] = 1$ bit. Biased coin $p(\text{heads})=0.9$: $H=-[0.9\log_2 0.9+0.1\log_2 0.1]\approx0.469$ bits — less uncertainty, since outcomes are more predictable.

**Cross-entropy** between true distribution $p$ and predicted distribution $q$:
```
H(p, q) = -Σ_x p(x) log q(x)
```
Measures the average number of bits needed to encode samples from $p$ using a code optimized for $q$ — always $\ge H(p)$, with equality iff $p=q$. This is *exactly* the standard classification loss function: with one-hot true label $p$ (all mass on the correct class) and predicted softmax probabilities $q$, cross-entropy reduces to $-\log q(\text{correct class})$ — penalizing low predicted probability on the true class, with the penalty growing unboundedly as $q(\text{correct})\to0$.

**Joint entropy**: uncertainty of two variables together: $H(X,Y) = -\sum_{x,y}p(x,y)\log p(x,y)$.

**Conditional entropy**: remaining uncertainty in $Y$ once $X$ is known: $H(Y|X) = -\sum_{x,y}p(x,y)\log p(y|x) = H(X,Y)-H(X)$. Chain rule of entropy: $H(X,Y)=H(X)+H(Y|X)$.

**Why it matters in ML.** Cross-entropy loss is the workhorse loss for virtually all classification and language modeling — understanding it as "expected code length under a mismatched distribution" explains *why* it heavily penalizes confident-and-wrong predictions (as $q\to0$ for the true class, $-\log q\to\infty$) far more than squared-error would, which is desirable for well-calibrated probabilistic classifiers. Decision trees use entropy/information gain (reduction in conditional entropy after a split) as the splitting criterion.

**Pitfalls.** Entropy is always $\ge0$ for discrete variables but *differential* entropy (continuous case) can be **negative** (e.g., a very peaked/narrow Gaussian) — a common trick question. Cross-entropy is *not* symmetric ($H(p,q)\neq H(q,p)$ in general), unlike KL divergence's... wait, KL divergence is also not symmetric (see below) — neither is a true "distance" metric.

### KL Divergence and Jensen-Shannon Divergence

**Kullback-Leibler (KL) divergence** measures how much distribution $q$ diverges from a reference distribution $p$:
```
D_KL(p ‖ q) = Σ_x p(x) log( p(x) / q(x) )  =  H(p,q) - H(p)
```
i.e., cross-entropy minus entropy — the *extra* bits needed to encode $p$-distributed data using a $q$-optimized code, beyond the theoretical minimum $H(p)$. Always $\ge0$ (**Gibbs' inequality**, provable via Jensen's inequality on the concave $\log$ function), with equality iff $p=q$ almost everywhere.

**Why minimizing cross-entropy = minimizing KL divergence in ML.** Since $H(p,q)=H(p)+D_{KL}(p\|q)$ and $H(p)$ (entropy of the fixed true/empirical label distribution) doesn't depend on model parameters, minimizing cross-entropy loss w.r.t. model parameters is *exactly equivalent* to minimizing $D_{KL}(p\|q)$ — pushing the model's predicted distribution $q$ to match the true distribution $p$ as closely as possible. This is also the same objective as **maximum likelihood estimation** (minimizing NLL = minimizing cross-entropy between empirical data distribution and model distribution).

**Asymmetry — the key gotcha.** $D_{KL}(p\|q)\neq D_{KL}(q\|p)$ in general, so KL divergence is **not a true metric/distance** (fails symmetry, and also fails the triangle inequality). This asymmetry has real modeling consequences:
- $D_{KL}(p\|q)$ ("forward KL", used in standard MLE/cross-entropy training): heavily penalizes $q(x)\approx0$ wherever $p(x)>0$ — forces $q$ to cover all of $p$'s support (**mode-covering / zero-avoiding**), which can make $q$ overly spread out ("blurry").
- $D_{KL}(q\|p)$ ("reverse KL", used in variational inference/VAEs): heavily penalizes $q(x)>0$ wherever $p(x)\approx0$ — forces $q$ to avoid placing mass where $p$ has none, tending to lock onto a single mode of a multi-modal $p$ (**mode-seeking / zero-forcing**), rather than spreading across all modes.

**Jensen-Shannon (JS) divergence** — a symmetrized, smoothed, bounded version:
```
M = (p+q)/2
D_JS(p, q) = (1/2) D_KL(p‖M) + (1/2) D_KL(q‖M)
```
Properties: symmetric ($D_{JS}(p,q)=D_{JS}(q,p)$), always finite/bounded ($0\le D_{JS}\le\log2$ in nats, or $\le1$ bit), and its square root is a true metric (satisfies triangle inequality) — the **Jensen-Shannon distance**. Famously, the original GAN objective (Goodfellow et al.) is theoretically shown to be equivalent (at the optimal discriminator) to minimizing $2\cdot D_{JS}(p_{data}, p_{generator}) - \log4$.

**Worked example.** $p=(0.5,0.5)$, $q=(0.9,0.1)$ (both over 2 outcomes, natural log):
```
D_KL(p‖q) = 0.5 log(0.5/0.9) + 0.5 log(0.5/0.1) = 0.5(-0.588) + 0.5(1.609) ≈ 0.511 nats
D_KL(q‖p) = 0.9 log(0.9/0.5) + 0.1 log(0.1/0.5) = 0.9(0.588) + 0.1(-1.609) ≈ 0.368 nats
```
Confirms asymmetry: $D_{KL}(p\|q)\neq D_{KL}(q\|p)$.

**Why it matters in ML.** KL divergence appears in: cross-entropy loss (as shown above), the **VAE ELBO** loss (KL term regularizing the learned latent posterior toward a prior, typically $\mathcal{N}(0,I)$), **knowledge distillation** (student matches teacher's softened output distribution via KL), **RLHF/PPO fine-tuning** of LLMs (KL penalty keeping the fine-tuned policy close to the reference/base model to prevent reward hacking / catastrophic drift), and t-SNE (matches high-D and low-D neighbor distributions via KL).

**Pitfalls.** $D_{KL}(p\|q)$ is undefined (infinite) wherever $q(x)=0$ but $p(x)>0$ — a real numerical issue requiring smoothing/clipping predicted probabilities away from exactly 0 or 1 (label smoothing partly exists for this reason too). Calling KL divergence a "distance" without qualification, when it's not symmetric and doesn't satisfy the triangle inequality.

### Mutual Information and Feature Selection

**Mutual information (MI)** quantifies how much knowing one variable reduces uncertainty about another:
```
I(X; Y) = Σ_x Σ_y p(x,y) log( p(x,y) / (p(x)p(y)) )
        = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X)+H(Y)-H(X,Y)
        = D_KL( p(x,y) ‖ p(x)p(y) )
```
Intuition: MI is the KL divergence between the *joint* distribution and the product of *marginals* — i.e., it directly measures how far $X,Y$ are from being independent (if independent, $p(x,y)=p(x)p(y)$ exactly, and $I(X;Y)=0$). $I(X;Y)\ge0$ always, symmetric ($I(X;Y)=I(Y;X)$, unlike KL divergence itself), and $I(X;Y)=0$ iff $X\perp Y$ (statistically independent) — a strictly stronger statement than zero *correlation*, since MI captures **any** kind of dependence (linear or non-linear), whereas Pearson correlation only captures linear dependence.

**Worked example — MI vs correlation.** Let $X\sim\text{Uniform}(-1,1)$ and $Y=X^2$. Pearson correlation $\rho(X,Y)=0$ (by symmetry, since $\text{Cov}(X,X^2)=E[X^3]-E[X]E[X^2]=0-0=0$) — correlation says "no relationship." But $Y$ is a *deterministic function* of $X$, so knowing $X$ tells you $Y$ exactly: $H(Y|X)=0 \Rightarrow I(X;Y)=H(Y)-H(Y|X)=H(Y)>0$ — mutual information correctly detects the (non-linear) dependence that correlation misses entirely.

**Feature selection via MI.** For a feature $X_i$ and target $Y$, compute $I(X_i;Y)$ for each candidate feature; keep features with high MI to the target — this captures non-linear predictive relationships that a linear correlation-based filter would miss. Related: **minimum Redundancy Maximum Relevance (mRMR)** feature selection explicitly maximizes $I(X_i;Y)$ (relevance) while minimizing $I(X_i;X_j)$ among selected features (redundancy). In deep learning, the **information bottleneck** theory of generalization studies $I(X;Z)$ and $I(Z;Y)$ for internal representations $Z$, trying to compress input information while preserving task-relevant information.

**Why it matters in ML.** MI-based feature selection generalizes beyond linear-correlation-based filters (like a plain correlation matrix heatmap) to catch non-linear and even non-monotonic relationships. MI also underlies **InfoGAN** (maximizing MI between latent codes and generated outputs to learn disentangled representations) and various representation-learning/self-supervised objectives (e.g., contrastive learning objectives like InfoNCE are motivated as MI lower bounds).

**Pitfalls.** Estimating MI from finite, especially continuous, data is statistically hard — naive histogram-binning estimators are biased and sensitive to bin count/width; k-NN-based estimators (e.g., the Kraskov estimator) are more robust but still noisy in high dimensions. MI being high doesn't tell you the *direction* or *form* of a relationship, only that one exists — unlike correlation's sign, which at least indicates direction for linear relationships.

### Perplexity and Language Modeling

**Definition.** Perplexity (PPL) of a language model on a sequence $w_1,\dots,w_N$ is the exponentiated average negative log-likelihood (cross-entropy) per token:
```
PPL = exp( -(1/N) Σ_i log P(w_i | w_1,...,w_{i-1}) )     (natural-log cross-entropy version)
    = 2^{ -(1/N) Σ_i log_2 P(w_i | w_1,...,w_{i-1}) }     (bits version)
```
Equivalently, $\text{PPL} = e^{H(p,q)}$ where $H(p,q)$ is the model's average cross-entropy per token against the empirical data distribution — perplexity is literally "exponentiated cross-entropy," expressed on a more interpretable scale.

**Intuition.** Perplexity is interpretable as the **effective (weighted-average) branching factor / vocabulary size the model is choosing among at each step**. A perplexity of $k$ means the model is, on average, as uncertain about the next token as if it had to choose uniformly among $k$ equally likely options. Lower perplexity = better (more confident, accurate) predictions.

**Worked example.** Suppose a model assigns the true next word probability $0.5$ every single time step across a sequence (constant, for simplicity). Then $\text{PPL}=\exp(-\log 0.5) = \exp(0.693) = 2$ — exactly matching the intuition that "50% confidence in the right answer each step" behaves like uniformly guessing among 2 options.

**Relation to entropy and KL divergence.** Since cross-entropy $H(p,q)=H(p)+D_{KL}(p\|q)$, perplexity's floor (best achievable value, when the model exactly matches the true data distribution, $q=p$) is $e^{H(p)}$ — the intrinsic entropy of natural language itself (unknown but estimated in various corpora studies at roughly 1-1.5 bits/character or a handful of bits/word depending on context/model). No model can beat this floor; how close a model's perplexity gets to it (or to human-level baselines) is a standard language-model quality benchmark.

**Why it matters in ML/AI Engineering.** Perplexity is the standard intrinsic evaluation metric for language models (GPT-family, BERT-style, etc.) prior to/alongside downstream task benchmarks — used to compare tokenization schemes, model sizes/architectures (scaling laws plot perplexity/loss vs. compute/parameters/data), and to detect **distribution shift**/domain mismatch (a model perplexity-tuned on news text will show much higher, worse perplexity on, say, legal or medical text it wasn't trained on — a fast, label-free way to detect if a new corpus is "in-domain"). It's also used to score/filter web-scraped pretraining corpora (keep only low-perplexity-under-a-reference-model text as a crude quality filter).

**Pitfalls.** Perplexity is **not directly comparable across models with different vocabularies/tokenizers** — a byte-level tokenizer and a large sub-word vocabulary tokenizer will produce very different, non-comparable per-token perplexities on the same text since "per-token" means different things (this is exactly why cross-model LLM comparisons often normalize to *bits-per-byte* or *bits-per-character* instead of bits/token). Lower perplexity does not automatically mean better *downstream task performance* or better generation quality/coherence — a model can be well-calibrated in probability but still produce repetitive or factually poor generations; perplexity is a necessary-but-not-sufficient proxy.

### Interview Questions — Information Theory

**Q1 (basic).** Define entropy in your own words and give its formula.
**A.** Entropy $H(X)=-\sum_x p(x)\log p(x)$ measures the average uncertainty (or average information content / minimum expected code length in bits/nats) of a random variable's outcomes. It's maximized by a uniform distribution (maximum unpredictability) and is zero for a deterministic variable.

**Q2 (basic).** What's the difference between entropy and cross-entropy?
**A.** Entropy $H(p)=-\sum p(x)\log p(x)$ measures uncertainty under the *true* distribution using its own optimal code. Cross-entropy $H(p,q)=-\sum p(x)\log q(x)$ measures the average code length when encoding data from true distribution $p$ using a code optimized for a *different*, possibly mismatched, distribution $q$ — always $\ge H(p)$, with the gap being exactly the KL divergence.

**Q3 (conceptual).** Why is cross-entropy loss used for classification instead of MSE?
**A.** Cross-entropy is the natural loss under a maximum-likelihood framework for categorical outputs — minimizing it is equivalent to minimizing KL divergence between predicted and true class distributions. Its gradient w.r.t. logits ($p-y$ after softmax) stays well-scaled and doesn't vanish even for very wrong predictions, whereas MSE on probabilities (through a sigmoid/softmax) can produce very small gradients when predictions are confidently wrong (saturated activation regions), slowing learning. Cross-entropy also heavily penalizes confident wrong predictions ($-\log q\to\infty$ as $q\to0$), encouraging well-calibrated probability outputs.

**Q4 (conceptual, tricky).** Is KL divergence a valid distance metric? Justify with the specific property it violates.
**A.** No. It violates symmetry ($D_{KL}(p\|q)\neq D_{KL}(q\|p)$ in general) and the triangle inequality. It is non-negative and zero only when $p=q$, but that alone isn't sufficient to be a metric.

**Q5 (derivation).** Show that $D_{KL}(p\|q)\ge0$ for any distributions $p,q$.
**A.** By Jensen's inequality (since $-\log$ is convex, or $\log$ is concave): $D_{KL}(p\|q)=\sum_x p(x)\log\frac{p(x)}{q(x)} = -\sum_x p(x)\log\frac{q(x)}{p(x)} \ge -\log\left(\sum_x p(x)\frac{q(x)}{p(x)}\right) = -\log\left(\sum_x q(x)\right) = -\log(1) = 0$. Equality holds iff $q(x)/p(x)$ is constant everywhere $p(x)>0$, i.e., $p=q$.

**Q6 (applied).** How does minimizing cross-entropy loss during training relate to maximum likelihood estimation?
**A.** Cross-entropy between the empirical label distribution and model's predicted distribution, summed/averaged over training data, is exactly the negative log-likelihood (NLL) of the data under the model, up to a constant. So minimizing cross-entropy loss over parameters is precisely doing maximum likelihood estimation of those parameters.

**Q7 (conceptual, tricky).** Why does variational inference typically use reverse KL ($D_{KL}(q\|p)$) instead of forward KL, and what practical effect does that have?
**A.** In VI, the true posterior $p$ is intractable (can't sample from or evaluate it easily, e.g., can't even normalize it), while the approximating family $q$ is chosen to be tractable (e.g., factorized Gaussian) — reverse KL, $D_{KL}(q\|p)$, only requires expectations under the tractable $q$ (which you can sample from), making it computable via the ELBO, whereas forward KL would require expectations under the intractable $p$. Practically, reverse KL is "mode-seeking / zero-forcing" — $q$ tends to collapse onto a single mode of a multi-modal true posterior $p$ rather than spreading mass to cover all modes, which is why VAEs/mean-field VI often produce mode-collapsed, underdispersed posterior approximations.

**Q8 (applied, scenario).** You need to detect if incoming production data has drifted from training data. How could mutual information or KL divergence help?
**A.** Estimate the KL divergence (or its symmetrized version, JS divergence, for numerical stability) between the training feature distribution and a recent window of production feature distributions (e.g., via histogram-based or kernel density estimates per feature, or a multivariate density ratio estimator) — a rising divergence signals drift. Alternatively, train a discriminator model to distinguish "training" vs. "production" samples; its performance directly estimates the JS divergence (this is exactly the GAN-discriminator-as-divergence-estimator idea) — near-chance accuracy means no detectable drift, high accuracy means significant drift.

**Q9 (conceptual).** How does mutual information differ from Pearson correlation as a dependence measure, and when would you prefer it?
**A.** Pearson correlation only captures *linear* association and is exactly zero for many non-linear (but strongly dependent) relationships (e.g., $Y=X^2$ for symmetric $X$). Mutual information captures *any* form of statistical dependence (linear or non-linear) and is zero iff the variables are truly independent. Prefer MI for feature selection when you suspect non-linear or non-monotonic relationships between candidate features and the target, which is common with raw/unscaled/interaction-heavy features before feeding into non-linear models like trees or neural nets.

**Q10 (derivation, tricky).** Derive the relationship $H(X,Y) = H(X) + H(Y|X)$ and use it to derive mutual information's formula in terms of entropies.
**A.** $H(X,Y)=-\sum_{x,y}p(x,y)\log p(x,y) = -\sum_{x,y}p(x,y)\log[p(x)p(y|x)] = -\sum_{x,y}p(x,y)\log p(x) -\sum_{x,y}p(x,y)\log p(y|x) = H(X)+H(Y|X)$ (using $\sum_y p(x,y)=p(x)$ for the first term). Then $I(X;Y)=H(Y)-H(Y|X) = H(Y) - [H(X,Y)-H(X)] = H(X)+H(Y)-H(X,Y)$.

**Q11 (applied).** What does a perplexity of 20 mean intuitively for a language model, and why can't you compare it across two models with different tokenizers?
**A.** It means the model is, on average, as uncertain at each prediction step as if choosing uniformly among about 20 equally likely next tokens. It's not comparable across tokenizers because "per token" isn't a fixed unit of text — a model using large sub-word/whole-word tokens will naturally have lower per-token perplexity than a byte/character-level model simply because each of its "tokens" carries more information/spans more text, independent of true underlying model quality; fair cross-model comparison requires normalizing to a fixed unit like bits-per-character or bits-per-byte.

**Q12 (advanced).** Explain, using entropy/KL divergence, why label smoothing can improve calibration.
**A.** Standard one-hot cross-entropy training pushes the model toward assigning probability 1 to the correct class and 0 to all others — but achieving $q(\text{true})\to1$ requires logits to diverge to $+\infty$, which never fully happens but drives severe overconfidence/miscalibration and large gradients even very late in training. Label smoothing replaces the one-hot target with a softened target (e.g., $1-\epsilon$ on the true class, $\epsilon/(K-1)$ spread over others), which is now itself a valid distribution $p'$ with entropy $H(p')>0$; minimizing cross-entropy $H(p',q)$ instead has a finite, achievable optimum with finite logits, capping overconfidence and generally improving calibration (and often generalization) at a small cost in peak achievable log-likelihood.

**Q13 (advanced, tricky).** In RLHF fine-tuning, why is a KL penalty term added between the fine-tuned policy and the reference (SFT) model?
**A.** Fine-tuning against a learned reward model via RL (e.g., PPO) can find degenerate policies that exploit reward-model errors ("reward hacking") by drifting far from the distribution of text the reward model was actually trained/calibrated on. Adding $-\beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$ to the RL objective directly penalizes the policy for diverging too far (in distribution over generated text) from the trusted reference/SFT model, trading off reward maximization against staying "close" to known-good, fluent, in-distribution behavior — a direct, principled use of KL divergence as a regularizer/trust-region constraint in the RL objective itself.

**Q14 (advanced, tricky).** Prove that mutual information $I(X;Y)=0$ if and only if $X$ and $Y$ are independent.
**A.** $I(X;Y)=D_{KL}(p(x,y)\|p(x)p(y))$. From Q5's proof, $D_{KL}(\cdot\|\cdot)=0$ iff the two distributions being compared are identical almost everywhere. So $I(X;Y)=0 \iff p(x,y)=p(x)p(y)$ for all $x,y$ — which is exactly the definition of statistical independence. (Forward direction: if independent, $p(x,y)=p(x)p(y)$ trivially gives $I=0$ by direct substitution into the MI sum, each $\log(1)=0$ term.)

---

## Rapid-Fire Interview Q&A

1. **Q: What's the difference between a scalar, vector, matrix, and tensor?**
   A: Rank-0 (single number), rank-1 (1D array), rank-2 (2D array), rank-$n$ (general $n$-D array) objects respectively.

2. **Q: What does "linearly independent" mean?**
   A: No vector in the set can be written as a linear combination of the others; only the trivial (all-zero) combination sums to the zero vector.

3. **Q: What's the rank of a matrix?**
   A: The number of linearly independent rows (= number of linearly independent columns) = dimension of the column space.

4. **Q: Why does L1 regularization produce sparse weights?**
   A: Its constraint region has corners on the axes, and its penalty gradient has constant magnitude regardless of weight size, both of which favor solutions with exact zeros.

5. **Q: What is an eigenvector, informally?**
   A: A direction that a matrix only stretches/shrinks, never rotates.

6. **Q: How do eigenvalues relate to trace and determinant?**
   A: $\text{tr}(A)=\sum\lambda_i$; $\det(A)=\prod\lambda_i$.

7. **Q: What does SVD decompose a matrix into?**
   A: $A=U\Sigma V^T$ — orthogonal $U$, diagonal non-negative $\Sigma$ (singular values), orthogonal $V$.

8. **Q: How is PCA related to SVD?**
   A: PCA's principal components are the right singular vectors ($V$) of the (mean-centered) data matrix; eigenvalues of the covariance matrix equal $\sigma_i^2/(n-1)$.

9. **Q: What makes a matrix positive definite?**
   A: All eigenvalues $>0$ (equivalently, $\mathbf{x}^TA\mathbf{x}>0$ for all non-zero $\mathbf{x}$).

10. **Q: Why do we add $\lambda I$ in ridge regression?**
    A: To make $X^TX+\lambda I$ strictly positive definite (hence invertible) even when $X^TX$ is singular/ill-conditioned due to multicollinearity.

11. **Q: What's the gradient's geometric meaning?**
    A: Direction of steepest ascent; its negative is steepest descent.

12. **Q: What's the difference between a Jacobian and a gradient?**
    A: Gradient is for scalar-output functions (a vector); Jacobian generalizes this to vector-output functions (a matrix of all partial derivatives).

13. **Q: Why is the Hessian used to classify critical points?**
    A: At a stationary point ($\nabla f=0$), the Hessian's definiteness tells you if it's a local min (PD), local max (ND), or saddle (indefinite).

14. **Q: What is the chain rule's role in backpropagation?**
    A: It lets you compute the loss gradient w.r.t. any parameter by multiplying local derivatives along the computational path from that parameter to the loss output.

15. **Q: Why do vanishing gradients occur in deep/RNN networks?**
    A: Repeated multiplication by derivatives/weights with magnitude $<1$ across many layers/timesteps shrinks the gradient signal exponentially.

16. **Q: What's the difference between convex and non-convex optimization?**
    A: In convex problems every local minimum is global; in non-convex problems local minima/saddle points may not be globally optimal.

17. **Q: What test determines if a twice-differentiable function is convex?**
    A: Its Hessian is positive semi-definite everywhere on its domain.

18. **Q: What does momentum add to gradient descent?**
    A: An exponentially-weighted moving average of past gradients, smoothing the path and accelerating along consistent directions.

19. **Q: What's the key difference between AdaGrad and RMSProp?**
    A: AdaGrad accumulates squared gradients forever (learning rate monotonically shrinks); RMSProp uses an exponential moving average, so it doesn't decay to zero.

20. **Q: What two things does Adam combine?**
    A: Momentum (1st moment of gradients) and RMSProp-style adaptive scaling (2nd moment of gradients), plus bias correction.

21. **Q: What problem does AdamW fix vs. Adam+L2?**
    A: It decouples weight decay from the adaptive gradient scaling, applying uniform decay instead of decay inversely scaled by gradient history.

22. **Q: Why use learning-rate warmup?**
    A: Early optimizer statistics (e.g., Adam's second moment) are unreliable/noisy right after initialization; warmup avoids large, destabilizing steps before they stabilize.

23. **Q: What are the KKT conditions, in one line each?**
    A: Stationarity (gradient of Lagrangian = 0), primal feasibility (constraints satisfied), dual feasibility (inequality multipliers ≥ 0), complementary slackness (multiplier × constraint = 0).

24. **Q: Why are most SVM dual variables ($\alpha_i$) zero?**
    A: Complementary slackness forces $\alpha_i=0$ for any point not exactly on the margin boundary — only support vectors get non-zero $\alpha_i$.

25. **Q: What is entropy measuring?**
    A: The average uncertainty / minimum expected encoding length of a random variable's outcomes.

26. **Q: How is cross-entropy loss related to KL divergence?**
    A: $H(p,q)=H(p)+D_{KL}(p\|q)$; since $H(p)$ is fixed by the data, minimizing cross-entropy = minimizing KL divergence to the true distribution.

27. **Q: Is KL divergence symmetric?**
    A: No — $D_{KL}(p\|q)\neq D_{KL}(q\|p)$ in general; that's why it's not a true distance metric.

28. **Q: What's special about Jensen-Shannon divergence vs. KL?**
    A: JS is symmetric and bounded (between 0 and $\log2$), and its square root satisfies the triangle inequality (a true metric), unlike KL.

29. **Q: How does mutual information differ from correlation?**
    A: MI captures any statistical dependence (linear or non-linear) and is zero iff variables are truly independent; correlation only captures linear dependence and can be zero even for strongly (non-linearly) dependent variables.

30. **Q: What does perplexity intuitively represent?**
    A: The effective number of equally-likely choices the model is "choosing among" per token, on average — exponentiated average cross-entropy per token.

31. **Q: Why can't you compare perplexity across models with different tokenizers?**
    A: "Per token" means different amounts of underlying text/information depending on vocabulary/tokenization scheme, making raw token-level perplexity not directly comparable; normalize to bits-per-character/byte instead.

32. **Q: What's the relationship between maximum likelihood estimation and cross-entropy minimization?**
    A: They're equivalent — minimizing cross-entropy loss over model parameters is the same optimization as maximizing the log-likelihood of the observed data under the model.

33. **Q: Why is $1/\sqrt{d_k}$ scaling used in attention?**
    A: It counteracts the growth in variance of dot-product logits with dimension $d_k$, keeping softmax inputs well-scaled and gradients healthy.

34. **Q: What's the Frobenius norm, and how does it relate to eigenvalues/singular values?**
    A: $\|A\|_F=\sqrt{\sum A_{ij}^2}=\sqrt{\text{tr}(A^TA)}=\sqrt{\sum_i \sigma_i^2}$ — the Euclidean norm of the matrix's entries, equal to the root-sum-square of its singular values.

35. **Q: What's the difference between the pseudo-inverse and the regular inverse?**
    A: The regular inverse only exists for square, full-rank matrices; the Moore-Penrose pseudo-inverse (via SVD) always exists and gives the minimum-norm least-squares solution for any matrix, invertible or not.
