# Deep Learning — Interview Prep Syllabus

Deep Learning is the load-bearing wall of modern AI work. For a **Data Scientist**, it shows up in model selection, feature learning vs. manual feature engineering trade-offs, and knowing when a simpler model beats a deep net. For a **Machine Learning Engineer**, it is the daily job: architecting, training, debugging, scaling, and productionizing neural networks — you are expected to derive backprop on a whiteboard, diagnose a NaN loss in seconds, and reason about GPU memory and mixed precision without blinking. For an **AI Engineer**, deep learning (especially Transformers, attention, and training dynamics) is the substrate underneath every LLM/agent system you will build, fine-tune, or serve — you need to understand *why* transformers replaced RNNs, how positional encodings and normalization placement affect training stability, and how to reason about inference-time compute and hardware constraints. Interviewers at all three levels probe for the same triad: **math fluency** (can you derive it), **systems intuition** (can you debug it), and **practical judgment** (can you make the right trade-off under constraints).

---

## Table of Contents

1. [Neural Network Foundations](#neural-network-foundations)
2. [Training Deep Networks](#training-deep-networks)
3. [Convolutional Neural Networks](#convolutional-neural-networks)
4. [Sequence Models](#sequence-models)
5. [Attention and Transformers](#attention-and-transformers)
6. [Practical Deep Learning](#practical-deep-learning)
7. [Rapid-Fire Interview Q&A](#rapid-fire-interview-qa)

---

## Neural Network Foundations

### Perceptron, Multilayer Perceptron, Universal Approximation Theorem

**Perceptron.** The original perceptron (Rosenblatt, 1958) is a linear binary classifier:

$$
\hat{y} = \text{sign}(w^\top x + b)
$$

It updates weights only on misclassified examples: $w \leftarrow w + \eta \, y \, x$ when a point is misclassified. It can only represent **linearly separable** functions — famously it cannot learn XOR, which motivated multilayer architectures.

**Multilayer Perceptron (MLP).** Stacks of affine transforms + nonlinearities:

$$
h^{(1)} = \sigma(W^{(1)} x + b^{(1)}), \quad h^{(2)} = \sigma(W^{(2)} h^{(1)} + b^{(2)}), \quad \dots, \quad \hat{y} = f(W^{(L)} h^{(L-1)} + b^{(L)})
$$

Each layer is a linear map followed by a nonlinearity $\sigma$. Without the nonlinearity, stacking layers is pointless — a composition of affine maps is itself affine, collapsing to a single linear layer.

**Universal Approximation Theorem (concept).** A feedforward network with a single hidden layer of *sufficient width* and a non-polynomial (e.g., sigmoid) activation can approximate any continuous function on a compact domain to arbitrary precision. Key nuances for interviews:
- It is an **existence** proof, not a **learnability** or **efficiency** proof — it says nothing about how many neurons are needed, or whether gradient descent will find the right weights.
- In practice, **depth** is more parameter-efficient than width for representing complex, compositional functions (this is the empirical/theoretical motivation for "deep" learning rather than "wide shallow" learning).
- Universal approximation does not imply good generalization — an infinitely expressive model can still overfit badly.

**Pitfall:** Candidates sometimes claim UAT guarantees a network *will learn* any function. Clarify: it guarantees *representational capacity*, not trainability or sample efficiency.

### Activation Functions

Activations inject nonlinearity — without them, depth is useless. Key trade-off axes: **gradient behavior** (vanishing/exploding), **saturation**, **computational cost**, and **zero-centering**.

| Activation | Formula | Derivative | Range | Notes |
|---|---|---|---|---|
| Sigmoid | $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ | $(0,1)$ | Saturates for $\lvert x \rvert$ large → vanishing gradient. Not zero-centered → zig-zag updates. |
| Tanh | $\tanh(x) = \frac{e^x - e^{-x}}{e^x+e^{-x}}$ | $1 - \tanh^2(x)$ | $(-1,1)$ | Zero-centered (better than sigmoid), still saturates. |
| ReLU | $\max(0, x)$ | $1$ if $x>0$ else $0$ | $[0,\infty)$ | Cheap, no saturation for $x>0$. "Dying ReLU": neurons stuck at 0 forever if they enter the negative regime with a bad gradient. |
| Leaky ReLU | $x$ if $x>0$ else $\alpha x$ ($\alpha \approx 0.01$) | $1$ or $\alpha$ | $(-\infty,\infty)$ | Fixes dying ReLU by allowing small negative gradient. |
| PReLU | Same as Leaky ReLU but $\alpha$ is **learned** | $1$ or $\alpha$ (learned) | $(-\infty,\infty)$ | More flexible, slight overfitting risk, extra params. |
| ELU | $x$ if $x>0$ else $\alpha(e^x - 1)$ | $1$ or $\alpha e^x$ | $(-\alpha, \infty)$ | Smooth, negative saturation gives some noise robustness, pushes mean activation toward 0. |
| GELU | $x \cdot \Phi(x)$ ($\Phi$ = standard normal CDF) | smooth, no closed simple form | $(-\approx0.17,\infty)$ | Probabilistic "soft gate" — used in BERT/GPT/Transformers. Smooth everywhere (helps optimization vs ReLU's kink). |
| Swish/SiLU | $x \cdot \sigma(x)$ | $\sigma(x) + x\sigma(x)(1-\sigma(x))$ | $(-\approx0.28,\infty)$ | Self-gated, smooth, non-monotonic; used in EfficientNet. |

**Vanishing gradient relation:** Sigmoid/tanh derivatives are bounded by $0.25$ and $1.0$ respectively, and shrink to ~0 for large $\lvert x \rvert$. In a deep network, the backprop chain multiplies many such derivatives together — if each is $<1$, the product shrinks exponentially with depth, so early layers get almost no gradient signal ("vanishing gradient"). ReLU's derivative is exactly $1$ for active units, which is why ReLU-family activations mitigate (not eliminate) vanishing gradients — but they introduce the dying-unit failure mode instead.

```python
import torch
import torch.nn as nn

x = torch.linspace(-5, 5, 11)
print("sigmoid:", torch.sigmoid(x))
print("tanh:", torch.tanh(x))
print("relu:", torch.relu(x))
print("gelu:", nn.functional.gelu(x))
print("silu:", nn.functional.silu(x))          # Swish
print("leaky_relu:", nn.functional.leaky_relu(x, negative_slope=0.01))
```

**Practical tips:**
- Default for most feedforward/CNN hidden layers: ReLU (fast, works well) or GELU/Swish for Transformer-style and modern vision architectures.
- Output layer activation is dictated by the task: sigmoid for binary classification, softmax for multi-class, linear (no activation) for regression.
- Watch for dying ReLU when using high learning rates or poor initialization; Leaky ReLU/PReLU/ELU/GELU are common fixes.

### Loss Functions

| Loss | Formula | Use Case | Notes |
|---|---|---|---|
| MSE | $\frac{1}{N}\sum (y_i - \hat{y}_i)^2$ | Regression | Sensitive to outliers (squared penalty); assumes Gaussian noise (MLE under Gaussian likelihood). |
| MAE | $\frac{1}{N}\sum \lvert y_i - \hat{y}_i \rvert$ | Regression, robust to outliers | Non-smooth at 0, gradient is constant magnitude regardless of error size (slow convergence near optimum). |
| Huber | $\begin{cases}\frac12 (y-\hat y)^2 & \lvert y-\hat y\rvert \le \delta \\ \delta(\lvert y-\hat y\rvert - \frac12\delta) & \text{else}\end{cases}$ | Regression w/ outliers | Quadratic near 0 (smooth gradient), linear far away (robust) — best of MSE + MAE. |
| Binary Cross-Entropy | $-\frac{1}{N}\sum [y_i \log \hat y_i + (1-y_i)\log(1-\hat y_i)]$ | Binary classification | Pairs with sigmoid output; is the negative log-likelihood of a Bernoulli model. |
| Categorical Cross-Entropy | $-\frac{1}{N}\sum_i \sum_c y_{i,c} \log \hat y_{i,c}$ | Multi-class classification | Pairs with softmax output; NLL of a categorical/multinomial model. |
| Hinge Loss | $\max(0, 1 - y \cdot \hat y)$, $y \in \{-1,+1\}$ | Max-margin classifiers (SVM, some DL) | Zero loss once margin satisfied; encourages a decision boundary with margin, not just correct sign. |

**MLE connection (important interview point):** Minimizing MSE $\equiv$ maximizing likelihood under a Gaussian noise assumption with fixed variance. Minimizing cross-entropy $\equiv$ maximizing likelihood under a Bernoulli/categorical model. This is *why* cross-entropy — not MSE — is the natural loss for classification: MSE on softmax outputs produces much weaker gradients when predictions are very wrong (saturating sigmoid/softmax + squared error compounds the vanishing-gradient problem at the output layer).

```python
import torch.nn.functional as F

logits = torch.tensor([[2.0, 0.5, -1.0]])
target = torch.tensor([0])  # class index
loss = F.cross_entropy(logits, target)  # combines log-softmax + NLL, numerically stable

huber = F.huber_loss(torch.tensor([3.0]), torch.tensor([0.0]), delta=1.0)
```

**Pitfalls:**
- Applying `nn.CrossEntropyLoss` on already-softmaxed outputs (it expects raw logits and applies log-softmax internally) — this double-softmaxes and cripples training.
- Using MSE for classification tasks — slower convergence, poorer calibration.
- Class imbalance with BCE/CE: use class weights, focal loss, or resampling.

### Forward Propagation and Backpropagation — Full Derivation

Consider a 2-layer network (1 hidden layer) with input $x \in \mathbb{R}^{d}$, hidden width $h$, scalar output, sigmoid hidden activation, and MSE loss on a single example for clarity.

**Forward pass:**

$$
z^{(1)} = W^{(1)} x + b^{(1)} \in \mathbb{R}^h, \qquad a^{(1)} = \sigma(z^{(1)})
$$
$$
z^{(2)} = W^{(2)} a^{(1)} + b^{(2)} \in \mathbb{R}, \qquad \hat y = z^{(2)} \text{ (linear output)}
$$
$$
L = \frac{1}{2}(\hat y - y)^2
$$

**Backward pass (chain rule, layer by layer):**

Step 1 — gradient at the output:
$$
\frac{\partial L}{\partial \hat y} = (\hat y - y) \quad\Rightarrow\quad \delta^{(2)} \equiv \frac{\partial L}{\partial z^{(2)}} = (\hat y - y) \cdot 1
$$
(the "1" is $\partial \hat y/\partial z^{(2)}$ since output activation is identity)

Step 2 — gradients for layer 2 parameters:
$$
\frac{\partial L}{\partial W^{(2)}} = \delta^{(2)} \, (a^{(1)})^\top, \qquad \frac{\partial L}{\partial b^{(2)}} = \delta^{(2)}
$$

Step 3 — propagate error back into hidden layer:
$$
\frac{\partial L}{\partial a^{(1)}} = (W^{(2)})^\top \delta^{(2)}
$$
$$
\delta^{(1)} \equiv \frac{\partial L}{\partial z^{(1)}} = \frac{\partial L}{\partial a^{(1)}} \odot \sigma'(z^{(1)}) = \left[(W^{(2)})^\top \delta^{(2)}\right] \odot \sigma(z^{(1)})(1-\sigma(z^{(1)}))
$$
($\odot$ = elementwise product; this is the Hadamard product because each hidden unit's activation only affects the loss through its own scalar output, so the chain rule factorizes elementwise)

Step 4 — gradients for layer 1 parameters:
$$
\frac{\partial L}{\partial W^{(1)}} = \delta^{(1)} x^\top, \qquad \frac{\partial L}{\partial b^{(1)}} = \delta^{(1)}
$$

**General recursive rule (the essence of backprop):** define $\delta^{(l)} = \partial L / \partial z^{(l)}$. Then

$$
\delta^{(l)} = \left[(W^{(l+1)})^\top \delta^{(l+1)}\right] \odot \sigma'(z^{(l)}), \qquad \frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^\top, \qquad \frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}
$$

Backprop is just **reverse-mode automatic differentiation**: compute $\delta^{(L)}$ at the output, then recursively propagate backward, reusing $\delta^{(l+1)}$ to compute $\delta^{(l)}$ — this reuse is what makes it $O(\text{network size})$ instead of exponential in depth (as naive symbolic differentiation would be).

```python
import numpy as np

def forward_backward(x, y, W1, b1, W2, b2):
    z1 = W1 @ x + b1
    a1 = 1 / (1 + np.exp(-z1))          # sigmoid
    z2 = W2 @ a1 + b2
    yhat = z2                            # linear output
    loss = 0.5 * (yhat - y) ** 2

    # backward
    dL_dyhat = (yhat - y)
    delta2 = dL_dyhat                    # d(linear)/dz2 = 1
    dW2 = np.outer(delta2, a1)
    db2 = delta2
    da1 = W2.T @ delta2
    delta1 = da1 * (a1 * (1 - a1))       # sigmoid'
    dW1 = np.outer(delta1, x)
    db1 = delta1
    return loss, (dW1, db1, dW2, db2)
```

**Practical tip:** Always verify a hand-derived gradient with **numerical gradient checking**: $\frac{\partial L}{\partial \theta} \approx \frac{L(\theta+\epsilon) - L(\theta-\epsilon)}{2\epsilon}$ with $\epsilon \approx 10^{-4}$–$10^{-7}$, compared via relative error. This is a classic interview follow-up.

### Weight Initialization: Xavier/Glorot, He

**Why initialization matters:** If weights are too large, activations/gradients explode through layers; too small, they vanish. Also, if all weights are initialized identically (e.g., all zero), every neuron in a layer computes the same gradient and stays symmetric forever ("symmetry breaking" failure) — this is why we need *random*, not just *appropriately scaled*, initialization.

**Xavier/Glorot initialization** (for tanh/sigmoid, i.e., activations roughly linear near 0): choose variance so that the variance of activations is preserved forward and the variance of gradients is preserved backward. With $n_{in}$ fan-in and $n_{out}$ fan-out:

$$
\text{Var}(W) = \frac{2}{n_{in} + n_{out}} \quad\text{(commonly implemented as uniform } U\left[-\sqrt{\tfrac{6}{n_{in}+n_{out}}}, \sqrt{\tfrac{6}{n_{in}+n_{out}}}\right]\text{)}
$$

**He initialization** (for ReLU family): ReLU zeroes out roughly half the activations, halving variance, so compensate by doubling:

$$
\text{Var}(W) = \frac{2}{n_{in}}, \qquad W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)
$$

**Derivation intuition:** For a linear layer $z = Wx$ with $x_i$ i.i.d., $\text{Var}(z) = n_{in}\,\text{Var}(W)\,\text{Var}(x)$. To keep $\text{Var}(z) = \text{Var}(x)$ (preserve signal scale across layers), set $\text{Var}(W) = 1/n_{in}$. Xavier symmetrizes forward/backward by averaging fan-in and fan-out; He accounts for the factor-of-2 variance loss from ReLU's zeroing of negative inputs.

```python
import torch.nn as nn

linear = nn.Linear(256, 256)
nn.init.xavier_uniform_(linear.weight)     # for tanh/sigmoid nets
nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')  # He init, for ReLU nets
nn.init.zeros_(linear.bias)
```

**Pitfalls:**
- Using Xavier init with ReLU activations (mismatch — commonly causes slightly poorer early convergence, though modern normalization layers mask it).
- Zero-initializing all weights — breaks symmetry, network never learns beyond linear-combination-of-identical-neurons.
- Ignoring bias initialization for gates (e.g., LSTM forget gate biases are often initialized to 1, not 0, to default to "remember").

### Interview Questions

1. **Why can't a single-layer perceptron learn XOR?**
   XOR is not linearly separable — no single hyperplane separates the (0,0),(1,1) class from (0,1),(1,0). A perceptron computes $\text{sign}(w^\top x + b)$, a linear decision boundary, so it fundamentally cannot represent XOR. Adding a hidden layer (MLP) with nonlinear activations creates a piecewise-linear decision boundary that can represent XOR (e.g., 2 hidden ReLU/step units).

2. **State the Universal Approximation Theorem and its practical limitations.**
   A feedforward network with one hidden layer, given enough hidden units and a non-polynomial activation, can approximate any continuous function on a compact subset of $\mathbb{R}^n$ to arbitrary accuracy. Limitations: it's an existence result (doesn't say how many units, or how to find the weights via gradient descent); doesn't address generalization or sample efficiency; in practice deep narrow networks are far more parameter-efficient than shallow wide ones for compositional functions.

3. **Derive the backpropagation update for a 2-layer network (1 hidden layer, sigmoid activation, MSE loss).**
   See full derivation above: $\delta^{(2)} = (\hat y - y)$, $\nabla_{W^{(2)}} L = \delta^{(2)} (a^{(1)})^\top$, $\delta^{(1)} = [(W^{(2)})^\top \delta^{(2)}] \odot a^{(1)}(1-a^{(1)})$, $\nabla_{W^{(1)}} L = \delta^{(1)} x^\top$.

4. **Why does sigmoid cause vanishing gradients but ReLU (mostly) doesn't?**
   Sigmoid's derivative $\sigma(x)(1-\sigma(x))$ is maximized at $0.25$ and shrinks to ~0 as $|x|$ grows; multiplying many such small numbers across layers during backprop shrinks the gradient exponentially with depth. ReLU's derivative is exactly $1$ for all positive inputs (no shrinkage) and $0$ for negative inputs — so gradients pass through unattenuated for "active" paths, though it introduces the dying-ReLU issue instead.

5. **What is the "dying ReLU" problem and how do you fix it?**
   If a ReLU unit's input becomes negative for all training examples (often from a large negative bias update or high learning rate), its gradient is permanently 0 and the unit stops learning. Fixes: Leaky ReLU/PReLU/ELU (nonzero gradient for negative inputs), lower learning rate, better initialization (He init), batch normalization to keep pre-activations centered.

6. **Why is cross-entropy preferred over MSE for classification?**
   Cross-entropy is the negative log-likelihood under a categorical/Bernoulli model — it directly penalizes low probability assigned to the true class and its gradient w.r.t. logits (with softmax) is simply $\hat y - y$ (clean, non-vanishing). MSE on softmax/sigmoid outputs multiplies the error by the activation's derivative, which saturates near 0 or 1, causing weak gradients exactly when the model is most wrong — slow learning.

7. **Derive the gradient of softmax + cross-entropy loss w.r.t. the logits.**
   Given logits $z$, softmax $\hat y_c = e^{z_c}/\sum_k e^{z_k}$, and loss $L = -\sum_c y_c \log \hat y_c$ (one-hot $y$), the combined gradient simplifies beautifully to $\partial L/\partial z_c = \hat y_c - y_c$. This cancellation (softmax's Jacobian combined with cross-entropy's log) is why frameworks fuse `log_softmax` + `nll_loss` into one numerically stable op.

8. **Why do we initialize weights randomly instead of to zero?**
   Zero (or any identical) initialization makes every neuron in a layer receive identical gradients, so they update identically forever — the network can never break symmetry to learn diverse features, effectively collapsing a layer of width $h$ into a layer of width 1. Random initialization breaks this symmetry.

9. **Explain the intuition behind Xavier and He initialization. Why different formulas?**
   Both aim to keep the variance of activations (forward) and gradients (backward) roughly constant across layers, preventing vanishing/exploding signals. Xavier assumes activations are roughly linear near 0 (true for tanh/sigmoid) and uses $\text{Var}(W)=2/(n_{in}+n_{out})$. He accounts for ReLU zeroing ~half the units, halving variance, so it doubles the target variance: $\text{Var}(W) = 2/n_{in}$.

10. **What's the difference between a loss function and a cost function?** (Common terminology check.)
    Loss is typically per-example error; cost (or objective) is the aggregate (e.g., mean) loss over a batch/dataset that the optimizer actually minimizes. Many use the terms interchangeably in practice, but interviewers may probe whether you know the distinction.

11. **When would you use Huber loss over MSE or MAE?**
    When your regression targets have outliers you want to be robust to, but you still want a smooth, well-behaved gradient near the optimum (unlike MAE, whose gradient has constant magnitude and never shrinks, causing oscillation near the minimum). Huber is quadratic (MSE-like) inside a threshold $\delta$ and linear (MAE-like) outside it — you tune $\delta$ based on expected outlier scale.

12. **Why is hinge loss used in max-margin classifiers, and what's its relation to logistic loss?**
    Hinge loss $\max(0, 1-y\hat y)$ imposes zero penalty once a point is correctly classified with margin $\ge 1$, encouraging a decision boundary that maximizes margin (as in SVMs) rather than pushing probabilities to 0/1 indefinitely. Logistic loss (cross-entropy) never reaches zero gradient and keeps pushing confidence higher; hinge loss is more "lazy" once the margin is satisfied, and is not differentiable at the kink (subgradient methods used).

13. **You implement backprop by hand and gradients don't match `torch.autograd`. How do you debug?**
    Use numerical gradient checking: perturb each parameter by $\pm\epsilon$ (e.g., $10^{-5}$), recompute loss, estimate $\partial L/\partial\theta \approx (L(\theta+\epsilon)-L(\theta-\epsilon))/(2\epsilon)$, and compare relative error to the analytic gradient. Common bugs: wrong transpose in matrix gradients, forgetting to accumulate gradients over batch dimension correctly, wrong order of elementwise vs. matrix multiply, or using the wrong activation derivative.

14. **Why must the derivative of the activation function be nonzero for effective learning, and what activation issues cause "flat" loss landscapes early in training?**
    If $\sigma'(z) \approx 0$ (saturated sigmoid/tanh, or dead ReLU), the chain-rule product $\delta^{(l)}$ becomes ~0, so no gradient reaches earlier layers/weights — training stalls. This is exactly the vanishing gradient / dead unit issue; symptoms are a loss that plateaus immediately with near-zero weight updates in early layers.

15. **Explain why depth (many narrow layers) is often preferred over width (one wide layer) despite both being "universal approximators."**
    Empirically and theoretically, deep compositional architectures can represent certain function classes (e.g., hierarchical/compositional structure in images, language) with exponentially fewer parameters than a shallow network of equivalent expressiveness. Depth allows reuse of learned sub-features across layers (hierarchical feature composition), improving both parameter efficiency and often generalization, at the cost of harder optimization (vanishing/exploding gradients), which modern techniques (normalization, residuals, better init) address.

---

## Training Deep Networks

### Gradient Descent Optimizers in DL Context

**SGD + Momentum.** Vanilla SGD: $\theta \leftarrow \theta - \eta \nabla L(\theta)$. Momentum accumulates a velocity vector to smooth noisy gradients and accelerate through consistent directions:

$$
v_t = \beta v_{t-1} + (1-\beta)\nabla L(\theta_{t-1}) \quad\text{(or without the }(1-\beta)\text{ in the "classic" form)}, \qquad \theta_t = \theta_{t-1} - \eta v_t
$$

Intuition: think of a ball rolling downhill — momentum smooths out oscillations across steep, narrow ravines (common in loss landscapes) and speeds convergence along shallow, consistent directions.

**RMSProp.** Adapts the learning rate per-parameter using a moving average of squared gradients:

$$
s_t = \beta s_{t-1} + (1-\beta) (\nabla L)^2, \qquad \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{s_t}+\epsilon}\nabla L
$$

This divides the step by the RMS of recent gradients — large, noisy gradients get damped; small, consistent gradients get amplified relatively.

**Adam.** Combines momentum (1st moment) + RMSProp (2nd moment), with bias correction for the initial zero-init of the moving averages:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla L, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla L)^2
$$
$$
\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}, \qquad \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat v_t}+\epsilon}\hat m_t
$$

Typical defaults: $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$.

**AdamW.** Standard Adam applies weight decay as $L_2$ regularization baked *into the gradient* ($\nabla L + \lambda\theta$), which then gets rescaled by Adam's adaptive $1/\sqrt{\hat v}$ term — an unintended interaction that weakens regularization for parameters with large gradient variance. AdamW **decouples** weight decay from the gradient-based update:

$$
\theta_t = \theta_{t-1} - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\theta_{t-1}\right)
$$

AdamW is now the de facto standard for training Transformers.

**Learning rate schedules:**

| Schedule | Formula (sketch) | Use case |
|---|---|---|
| Step decay | $\eta_t = \eta_0 \cdot \gamma^{\lfloor t/T \rfloor}$ | Classic CNN training (drop LR every N epochs) |
| Cosine annealing | $\eta_t = \eta_{min} + \frac12(\eta_{max}-\eta_{min})(1+\cos(\pi t/T))$ | Smooth decay, popular for vision/Transformer training |
| Warmup (linear) | $\eta_t = \eta_{max}\cdot t/T_{warmup}$ for $t<T_{warmup}$, then decay | Stabilizes early Transformer training where Adam's variance estimates are noisy |
| Warmup + cosine | linear warmup → cosine decay | Standard modern recipe (BERT/GPT-style) |

**Why warmup matters for Transformers:** early in training, Adam's second-moment estimate $\hat v_t$ is based on very few samples and is noisy/underestimated, causing effectively huge, unstable early steps, especially interacting badly with layer norm. A linear LR warmup avoids large destabilizing updates before the moment estimates stabilize.

```python
import torch

model = torch.nn.Linear(10, 1)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=0.01)

# warmup + cosine schedule
from torch.optim.lr_scheduler import LambdaLR
import math

warmup_steps, total_steps = 500, 10000
def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = LambdaLR(opt, lr_lambda)
```

**Practical tips:**
- Adam/AdamW: good default, robust to LR choice, but can generalize slightly worse than well-tuned SGD+momentum on some CNN benchmarks.
- SGD+momentum: still common for CNNs (ResNet-style training) with step/cosine decay, often better final generalization with careful tuning.
- Always pair Adam-family optimizers with a warmup if training Transformers from scratch.

### Vanishing/Exploding Gradients

**Causes:** Deep chains of multiplied Jacobians (via backprop chain rule) — if each layer's Jacobian has singular values consistently $<1$ (or activations saturate), gradients vanish exponentially with depth; if consistently $>1$ (e.g., poor init, large weights, RNNs unrolled over long sequences), gradients explode.

**Remedies:**

| Remedy | Mechanism |
|---|---|
| Gradient clipping | Cap gradient norm: if $\lVert g \rVert > \tau$, rescale $g \leftarrow g \cdot \tau/\lVert g \rVert$. Directly prevents exploding updates. |
| Careful initialization | Xavier/He keep activation/gradient variance stable across layers at init. |
| Normalization (BatchNorm/LayerNorm) | Re-centers/rescales activations each layer, preventing drift toward saturation regions. |
| Residual/skip connections | Provide an "identity gradient highway" — $\partial(\text{x + F(x)})/\partial x = I + \partial F/\partial x$, so gradient has a direct path of magnitude $\approx 1$ regardless of how small $\partial F/\partial x$ becomes. |
| Better activations | ReLU/GELU avoid the extreme saturation of sigmoid/tanh. |
| LSTM/GRU gating | Additive cell-state update avoids repeated multiplication that plagues vanilla RNNs. |

```python
import torch.nn.utils as utils

loss.backward()
utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### Batch Normalization, Layer Normalization, Group Normalization

**Batch Normalization** normalizes activations across the **batch dimension** (per channel/feature), then applies a learned affine transform:

$$
\hat x_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y_i = \gamma \hat x_i + \beta
$$

where $\mu_B, \sigma_B^2$ are the mean/variance computed **over the batch** (and spatial dims for CNNs) for each channel. $\gamma,\beta$ are learned per-channel scale/shift, letting the network undo the normalization if beneficial.

*Why it helps:* Reduces internal covariate shift (the distribution of layer inputs shifting as earlier layers' weights update), which lets you use higher learning rates and reduces sensitivity to initialization. It also has a regularizing effect (batch statistics inject noise). At inference, running (exponential moving average) statistics from training are used instead of batch statistics, since batches may not be available/representative at inference.

*Backward pass note:* BatchNorm's backward pass must account for the fact that $\mu_B,\sigma_B^2$ are themselves functions of every example in the batch — so gradients flow through the batch statistics too, not just through $\hat x_i$ directly, coupling all examples in the batch during backprop.

**Layer Normalization** normalizes across the **feature dimension**, independently per example (not batch-dependent):

$$
\mu_i = \frac{1}{H}\sum_{j=1}^H x_{i,j}, \qquad \sigma_i^2 = \frac{1}{H}\sum_{j=1}^H (x_{i,j}-\mu_i)^2, \qquad \hat x_{i,j} = \frac{x_{i,j}-\mu_i}{\sqrt{\sigma_i^2+\epsilon}}, \qquad y_{i,j} = \gamma_j \hat x_{i,j} + \beta_j
$$

*Why Transformers use LayerNorm, not BatchNorm:* sequence lengths vary, batch sizes can be small (esp. in NLP with variable-length sequences and padding), and LayerNorm's per-example computation removes the batch-size dependency, making it stable regardless of batch composition and usable at inference with batch size 1 without needing running statistics.

**Group Normalization** divides channels into $G$ groups and normalizes within each group (per example, per group), independent of batch size:

$$
\hat x = \frac{x - \mu_{group}}{\sqrt{\sigma_{group}^2+\epsilon}}
$$

Useful when batch sizes are small (e.g., object detection/segmentation with high-res images, limited GPU memory) where BatchNorm statistics become noisy/unreliable.

| Norm type | Normalizes over | Batch-size dependent? | Common use |
|---|---|---|---|
| BatchNorm | batch (+ spatial) per channel | Yes | CNNs (image classification) |
| LayerNorm | features per example | No | Transformers, RNNs |
| GroupNorm | channel groups per example | No | Detection/segmentation, small-batch CNN training |
| InstanceNorm | spatial per channel per example | No | Style transfer |

```python
import torch.nn as nn

bn = nn.BatchNorm2d(num_features=64)     # for CNN feature maps (N, C, H, W)
ln = nn.LayerNorm(normalized_shape=512)  # for Transformer hidden states (..., 512)
gn = nn.GroupNorm(num_groups=8, num_channels=64)
```

**Pitfall:** Using BatchNorm with very small batch sizes (e.g., batch size 1–4) gives noisy statistics and hurts training — switch to GroupNorm or LayerNorm. Also remember `model.eval()` switches BatchNorm to use running stats — forgetting this at inference silently degrades results.

### Regularization in DL

**Dropout.** During training, randomly zero each activation with probability $p$ (independently), then scale surviving activations by $1/(1-p)$ (inverted dropout) so expected activation magnitude matches inference (no scaling at inference, since all units are active):

$$
\tilde h_i = \frac{m_i}{1-p} h_i, \qquad m_i \sim \text{Bernoulli}(1-p)
$$

*Why it works:* Prevents units from co-adapting/relying on specific other units being present (forces redundant, distributed representations); can be viewed as training an implicit ensemble of $2^n$ subnetworks that share weights, and at test time using all units approximates averaging over that ensemble.

```python
import torch.nn as nn
dropout = nn.Dropout(p=0.5)   # training: zeroes & scales; eval(): identity
```

**Weight decay ($L_2$ regularization).** Adds $\frac{\lambda}{2}\lVert\theta\rVert_2^2$ to the loss, penalizing large weights, encouraging smoother/simpler decision functions and improving generalization. Gradient contribution: $\lambda\theta$, added directly to parameter update (see AdamW discussion above for the subtlety of decoupling from adaptive optimizers).

**Data augmentation.** Synthetically expands training distribution (crops, flips, color jitter, rotation for images; back-translation, synonym replacement for text; mixup/cutmix). Reduces overfitting by exposing the model to label-preserving input variations it should be invariant to.

**Label smoothing.** Instead of hard one-hot targets, use soft targets: $y_c = 1-\epsilon$ for the true class, $\epsilon/(C-1)$ for others. Prevents the model from becoming overconfident (logits growing unboundedly to push softmax to exactly 1), improves calibration and sometimes generalization.

**Early stopping.** Monitor validation loss/metric during training; stop when it stops improving for $k$ epochs ("patience"). Effectively regularizes by limiting how long the model can fit training-set-specific noise, and avoids the cost of a separate regularization hyperparameter search.

| Technique | Regularizes by | Key hyperparameter | Typical values |
|---|---|---|---|
| Dropout | Preventing co-adaptation | $p$ | 0.1–0.5 |
| Weight decay | Penalizing weight magnitude | $\lambda$ | 1e-4 – 1e-2 (0.01 common for AdamW/Transformers) |
| Data augmentation | Expanding effective data distribution | augmentation strength | task-dependent |
| Label smoothing | Softening targets, reducing overconfidence | $\epsilon$ | 0.1 typical |
| Early stopping | Limiting training duration | patience | 3–10 epochs |

### Batch Size, Mixed Precision, Gradient Accumulation

**Batch size effects:** Larger batches give lower-variance gradient estimates (smoother optimization trajectory) but each step is more computationally expensive and (empirically) can generalize slightly worse without LR adjustment — the "generalization gap." Linear scaling rule: when increasing batch size by $k\times$, scale LR by $k\times$ too (with warmup) to roughly preserve training dynamics. Very small batches give noisy gradients (can act as implicit regularization, and are necessary when memory-constrained) but destabilize BatchNorm statistics.

**Mixed precision training (FP16/BF16):** Store/compute most operations in half precision (FP16 or BF16) to roughly halve memory usage and leverage hardware Tensor Cores for 2–8x speedup, while keeping a master copy of weights (and/or accumulating losses) in FP32 to preserve numerical precision where it matters. FP16 has a narrow exponent range (can underflow/overflow) — **loss scaling** (multiply the loss by a large constant before backward, then unscale gradients before the optimizer step) prevents small gradients from underflowing to zero in FP16. BF16 has the same exponent range as FP32 (fewer mantissa bits) so it's more robust to overflow/underflow without needing loss scaling, at slightly lower precision.

```python
import torch

scaler = torch.cuda.amp.GradScaler()
model, opt = ..., ...
for x, y in dataloader:
    opt.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = model(x)
        loss = criterion(out, y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
```

**Gradient accumulation:** Simulate a larger effective batch size on limited GPU memory by accumulating gradients over $k$ mini-batches before calling `optimizer.step()`, dividing the loss by $k$ so gradient magnitudes match what a true large-batch step would produce.

```python
accum_steps = 4
opt.zero_grad()
for i, (x, y) in enumerate(dataloader):
    out = model(x)
    loss = criterion(out, y) / accum_steps
    loss.backward()
    if (i + 1) % accum_steps == 0:
        opt.step()
        opt.zero_grad()
```

### Gradient Checkpointing (Activation Checkpointing)

**The memory problem:** standard backprop caches every layer's activations during the forward pass because the backward pass needs them to compute local gradients (e.g., $\sigma'(z^{(l)})$ requires $z^{(l)}$). Memory usage therefore scales roughly *linearly with depth* — for very deep networks or long Transformer stacks with large activations, this activation memory (not parameter memory) is often the binding constraint on how large a batch size, sequence length, or model you can fit on a GPU.

**The trade-off:** gradient checkpointing (a.k.a. activation checkpointing) discards most intermediate activations during the forward pass and **recomputes them on the fly** during the backward pass by re-running the forward computation for a segment of the network starting from the nearest saved checkpoint. This trades extra compute (roughly one additional partial forward pass, ~20-40% overhead) for a large reduction in peak activation memory — instead of storing every activation $a^{(1)},\dots,a^{(L)}$, you store only a subset of "checkpoint" activations and recompute what's needed between them during backward.

```python
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class CheckpointedBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        # recomputes self.block(x) during backward instead of caching its activations
        return checkpoint(self.block, x, use_reentrant=False)

# Typical usage: checkpoint every transformer layer (or every k layers) in a deep stack
for layer in model.layers:
    x = checkpoint(layer, x, use_reentrant=False)
```

**Practical tips:**
- Standard lever for fitting larger batch sizes, longer sequences, or bigger models when activation memory (not parameter memory) is the bottleneck — very common when fine-tuning large Transformers on limited GPU memory.
- Combine with mixed precision and gradient accumulation for maximum memory savings — the three techniques attack different parts of the memory budget (numeric precision, activation storage, and effective batch size respectively) and stack well together.
- Checkpointing granularity matters: checkpointing every layer maximizes memory savings but also maximizes recompute overhead; checkpointing every few layers is a common middle ground.
- **Pitfall:** blocks containing randomness (e.g., dropout) must recompute the *same* random mask during the backward recompute as they used in the (discarded) forward pass, or gradients will correspond to a different function than the one whose output was actually used downstream. PyTorch's `checkpoint` saves/restores RNG state automatically to handle this; custom implementations must replicate that RNG management explicitly.

### Interview Questions

1. **Derive the Adam update rule and explain the role of bias correction.**
   Adam maintains exponential moving averages of the gradient ($m_t$, 1st moment) and squared gradient ($v_t$, 2nd moment): $m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t$, $v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2$. Since $m_0=v_0=0$, early estimates are biased toward zero; bias correction divides by $(1-\beta_1^t)$ and $(1-\beta_2^t)$ respectively to produce unbiased estimates $\hat m_t,\hat v_t$, especially important in the first few steps where $\beta^t$ is not yet negligible. Final update: $\theta_t=\theta_{t-1}-\eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)$.

2. **Why does AdamW generalize better than Adam with L2 regularization added to the loss?**
   In vanilla Adam, adding $\lambda\theta$ to the gradient causes the weight decay term to be divided by $\sqrt{\hat v_t}$ along with the gradient — parameters with historically large gradients get *less* effective decay, which is not the intended uniform regularization. AdamW applies weight decay directly to the parameters outside the adaptive gradient update, decoupling it so every parameter gets the same proportional decay $\lambda\theta$, matching the original intent of L2 regularization / weight decay.

3. **Your training loss becomes NaN after a few hundred steps. What do you check, in order?**
   (a) Learning rate too high → try lowering by 10x. (b) Numerical instability from FP16 without loss scaling → check for underflow/overflow, enable/verify `GradScaler`. (c) Division by zero or log(0) in the loss (e.g., log of a probability that hit exactly 0) → add epsilon or use numerically stable ops (`log_softmax` instead of `log(softmax(...))`). (d) Exploding gradients → check gradient norms before clipping, add/verify gradient clipping. (e) Bad data — check for NaN/Inf in the input batch itself. (f) Uninitialized or corrupted weights from a bad checkpoint load. Systematically bisect: log loss and gradient norm every step, and use `torch.autograd.set_detect_anomaly(True)` to trace the exact op producing NaN.

4. **What's the difference between vanishing and exploding gradients, and give one architectural remedy for each that isn't gradient clipping.**
   Vanishing: gradients shrink toward 0 across layers, usually from saturating activations or repeated multiplication by small Jacobian singular values — architectural remedy: residual/skip connections (identity gradient path) or switching to ReLU/GELU. Exploding: gradients grow unboundedly, common in RNNs over long sequences or poor initialization — architectural remedy: LSTM/GRU gating (additive cell state update rather than repeated matrix multiplication) or careful (He/Xavier) initialization.

5. **Explain how BatchNorm's backward pass differs from a typical elementwise operation's backward pass.**
   Because $\mu_B$ and $\sigma_B^2$ are computed from *all* examples in the batch, each normalized output $\hat x_i$ depends on every other example in the batch (through the batch statistics), not just on $x_i$ itself. So the gradient w.r.t. $x_i$ has three terms: direct dependence through $\hat x_i$, plus indirect dependence through $\partial \mu_B/\partial x_i$ and $\partial \sigma_B^2/\partial x_i$ — the backward pass sums contributions from all three paths, coupling gradients across the batch dimension.

6. **Why do Transformers use LayerNorm instead of BatchNorm?**
   Transformers often deal with variable-length sequences (with padding) and can be trained/deployed with small or batch-size-1 settings; BatchNorm's statistics depend on the batch composition, which is unstable/undefined for small or size-1 batches and complicates handling of padded tokens. LayerNorm normalizes per example across the feature dimension, independent of batch size or other examples, making it far more stable for sequence models.

7. **Explain dropout's training vs. inference behavior and why inverted dropout scales by $1/(1-p)$.**
   During training, each unit is zeroed independently with probability $p$; to keep the *expected* value of each activation the same as it would be without dropout (so downstream layers see a consistent scale), surviving units are scaled up by $1/(1-p)$. At inference, all units are active and no scaling/dropout is applied (equivalent to averaging over the ensemble of dropout masks). Forgetting to disable dropout at inference silently degrades performance and makes outputs stochastic.

8. **Describe the linear LR scaling rule for large-batch training and its limitations.**
   When batch size increases by $k\times$, the gradient estimate's variance drops by $k\times$ (since it's an average over more samples), so you can afford proportionally larger, less noisy steps — scale LR by $k\times$ as well, usually combined with a warmup phase to avoid instability from the sudden large steps at the start of training. Limitation: this rule breaks down at very large batch sizes (diminishing returns / instability), and doesn't account for batch norm statistics or optimizer-specific adaptive behavior (Adam's adaptive scaling partially self-corrects, weakening the need for the rule).

9. **What's the purpose of learning rate warmup, especially for Transformer training?**
   Early in training, parameters are far from optimal and Adam's second-moment estimates are based on very few noisy samples (biased/underestimated), which combined with LayerNorm's sensitivity to input scale can cause large, destabilizing updates. Linearly ramping the LR from 0 (or near 0) up to the target value over the first N steps avoids this early instability, letting moment estimates and normalization statistics settle before applying full-strength updates.

10. **When would you choose GroupNorm over BatchNorm?**
    When batch size is small (e.g., 1–8, common in object detection/segmentation with high-resolution inputs and memory constraints), BatchNorm's batch statistics become noisy and unreliable, hurting both training stability and the train/inference statistics mismatch. GroupNorm normalizes within channel groups per individual example, so it's independent of batch size and remains stable in these regimes.

11. **How does gradient clipping work, and does it change the direction of the gradient?**
    Gradient clipping caps the *norm* of the gradient vector: if $\lVert g\rVert_2 > \tau$, rescale $g \leftarrow g \cdot \tau/\lVert g\rVert_2$. This preserves the *direction* of the gradient (it's a pure rescaling) while bounding step size, preventing occasional huge gradients (e.g., from a bad batch or unstable RNN unrolling) from blowing up the parameters. (Note: per-element/"clip by value" clipping does change direction, unlike norm clipping.)

12. **Explain mixed precision training and why loss scaling is necessary for FP16 but not (usually) for BF16.**
    Mixed precision runs most compute in FP16/BF16 (half the memory, faster on Tensor Cores) while keeping a FP32 master copy of weights/accumulators for precision-critical steps. FP16 has only 5 exponent bits — a narrow dynamic range — so small gradients can underflow to exactly 0, killing learning signal; loss scaling multiplies the loss by a large constant (e.g., 1024+) before backward so gradients land in FP16's representable range, then unscales before the optimizer step. BF16 keeps FP32's 8 exponent bits (same dynamic range, fewer mantissa bits), so it rarely underflows/overflows and typically doesn't need loss scaling, at a modest cost in precision.

13. **What is gradient accumulation and when would you use it?**
    Gradient accumulation simulates a larger effective batch size by running several forward/backward passes on smaller micro-batches, summing (or averaging) their gradients, and only calling `optimizer.step()` after $k$ micro-batches — used when the desired batch size doesn't fit in GPU memory. You must divide the loss (or gradients) by $k$ so the effective step matches what a true large batch would have produced, and be careful with BatchNorm (statistics are still computed per micro-batch, not over the full effective batch).

14. **Your model overfits: training loss keeps dropping but validation loss starts rising. List 5 concrete interventions, in likely order of trying them.**
    (1) Add/increase dropout. (2) Add or increase weight decay. (3) Add data augmentation. (4) Reduce model capacity (fewer params/layers) if data is very limited. (5) Early stopping based on validation loss/metric. Also consider: gather more training data, use label smoothing, or use a pretrained model + transfer learning to leverage external data.

15. **Your loss plateaus almost immediately at a high value and never improves. What are the top suspects?**
    Learning rate too low (steps too small to make progress) or too high (overshooting, effectively "bouncing" — check by trying both directions), bad initialization (e.g., all-zero weights, wrong init for activation), a bug in the loss/label pipeline (e.g., shuffled label misalignment, wrong loss function for the task), frozen parameters that shouldn't be frozen (`requires_grad=False` by mistake), or vanishing gradients from very deep/unnormalized architecture. First diagnostic: verify the model can overfit a tiny subset (e.g., 10 examples) — if it can't even memorize 10 examples, it's very likely a bug, not a legitimate optimization/capacity issue.

16. **What problem does gradient/activation checkpointing solve, and what's the fundamental trade-off?**
    Standard backprop caches every layer's activations from the forward pass to use during the backward pass, so memory scales roughly linearly with depth — for very deep/large models, activation memory (not parameter count) is often the binding GPU memory constraint. Gradient checkpointing discards most activations after the forward pass and recomputes them on demand during backward from the nearest saved checkpoint, trading extra compute (a partial re-forward pass, typically ~20-40% overhead) for significantly reduced peak memory usage, letting you fit larger batch sizes, longer sequences, or bigger models than would otherwise fit.

17. **How does gradient checkpointing interact with mixed precision and gradient accumulation — are they redundant or complementary?**
    Complementary, not redundant: mixed precision shrinks the memory footprint of each stored value (fewer bytes per activation/gradient), gradient checkpointing reduces *how many* activations are stored at all (regardless of precision), and gradient accumulation lets you simulate a larger effective batch size without holding the full batch's activations simultaneously. Combining all three is standard practice when training or fine-tuning large models under tight GPU memory budgets, since each addresses a different axis of the memory problem.

18. **What's a subtle correctness pitfall when applying gradient checkpointing to a block containing dropout or other stochastic operations?**
    Checkpointing recomputes the block's forward pass during backward — if the recomputed forward pass draws a *different* random dropout mask than the original (discarded) forward pass, the gradients computed correspond to a different function than the one whose output was actually used downstream, silently corrupting training. Frameworks like PyTorch handle this by saving and restoring the RNG state around the checkpointed segment so the recomputed forward pass is bit-for-bit identical to the original; custom/manual checkpointing implementations must replicate this RNG state management explicitly.

---

## Convolutional Neural Networks

### Convolution Operation, Filters/Kernels, Stride, Padding, Receptive Field, Parameter Sharing

**Convolution (cross-correlation in DL practice).** For a 2D input $X$ and kernel $K$ of size $k \times k$:

$$
Y_{i,j} = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} X_{i+m, j+n} \cdot K_{m,n} + b
$$

(Note: DL frameworks implement this as cross-correlation, not true mathematical convolution which flips the kernel — but the network learns whichever orientation is useful, so it's a non-issue in practice.)

**Output spatial size** given input size $W$, kernel size $F$, stride $S$, padding $P$:

$$
W_{out} = \left\lfloor \frac{W - F + 2P}{S} \right\rfloor + 1
$$

**Padding** ("same" vs "valid"): "valid" = no padding (output shrinks); "same" = pad so output spatial size equals input size (for stride 1), preserving resolution across layers, useful for tasks needing pixel-level output alignment (segmentation).

**Stride**: step size the kernel moves each application; stride $>1$ downsamples spatial resolution (alternative to pooling).

**Receptive field**: the region of the *original input* that a given output unit is sensitive to. It grows with depth and is affected by kernel size, stride, dilation, and pooling. For $L$ stacked $3\times3$ conv layers (stride 1), the receptive field grows roughly linearly: $RF_L = RF_{L-1} + (k-1)$, so $L$ layers of $3\times3$ convs give receptive field $2L+1$. Dilated convolutions expand receptive field without adding parameters or losing resolution, by inserting gaps between kernel elements.

**Parameter sharing**: the *same* kernel weights are applied at every spatial location. This gives (a) massive parameter reduction vs. a fully-connected layer (a $3\times3\times C_{in}\times C_{out}$ kernel has far fewer params than a dense layer over the full image), and (b) **translation equivariance** — a feature detector learned in one location generalizes to detecting that feature anywhere in the image.

```python
import torch.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
# output spatial size unchanged (same padding) for stride=1, kernel=3, padding=1
```

**Parameter count** for a conv layer: $(k \times k \times C_{in} + 1) \times C_{out}$ (the $+1$ for bias per output channel) — independent of input spatial resolution, unlike a fully-connected layer.

### Pooling Layers, 1x1 Convolutions

**Max pooling**: takes the max value in each pooling window — provides translation-invariance to small shifts and highlights the strongest activation (sharp features); no learnable parameters.

**Average pooling**: takes the mean — smoother, less prone to noise amplification, but can dilute sparse strong signals.

$$
\text{MaxPool}_{i,j} = \max_{(m,n)\in\text{window}} X_{i+m,j+n}
$$

Pooling reduces spatial resolution (downsampling), reducing compute for subsequent layers and enlarging effective receptive field cheaply, at the cost of losing precise spatial location information (partially why segmentation architectures need skip connections / upsampling paths to recover it, e.g., U-Net).

**1×1 convolutions**: don't mix spatial neighbors at all — they operate purely across the channel dimension, acting like a per-pixel fully-connected layer. Uses:
- **Channel dimensionality reduction/expansion** (bottleneck), reducing compute before expensive $3\times3$/$5\times5$ convs (used heavily in Inception, ResNet bottleneck blocks).
- Adding nonlinearity/depth cheaply without changing spatial size.
- Cross-channel feature mixing/pooling.

```python
bottleneck = nn.Conv2d(256, 64, kernel_size=1)  # reduce channels 256->64 cheaply before a 3x3 conv
```

### Classic Architectures

| Architecture | Key Innovation | Notes |
|---|---|---|
| **LeNet** (1998) | First practical CNN (conv + pooling + FC) for digit recognition | Small, shallow; proved conv+pool architecture works for images. |
| **AlexNet** (2012) | ReLU activations, dropout, GPU training, large-scale ImageNet win | Kickstarted the deep learning boom; showed depth + scale + GPUs win. |
| **VGG** (2014) | Very deep (16-19 layers) using *only* stacked 3×3 convs | Showed small kernels stacked deep can match/exceed larger kernels with fewer params per layer; simple, uniform design; but very parameter-heavy (huge FC layers) and slow. |
| **ResNet** (2015) | **Residual/skip connections**: $y = F(x) + x$ | Solved the degradation problem where very deep plain nets performed *worse* than shallower ones due to optimization difficulty (not just overfitting). Enabled 100+ layer networks. |
| **Inception (GoogLeNet)** (2014) | Multi-scale "Inception module": parallel 1×1, 3×3, 5×5 convs + pooling, concatenated | Captures features at multiple receptive field scales simultaneously; 1×1 convs used for cheap dimensionality reduction before expensive convs. |
| **EfficientNet** (2019) | **Compound scaling**: jointly scale depth, width, and resolution by a fixed ratio (found via neural architecture search) rather than scaling just one dimension | Achieves better accuracy/FLOPs trade-off than naively scaling only depth or only width. |

**ResNet math/intuition in depth.** A residual block computes $y = F(x, W) + x$ instead of $y = F(x, W)$ directly. Backprop gradient:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}\left(\frac{\partial F}{\partial x} + I\right) = \frac{\partial L}{\partial y}\frac{\partial F}{\partial x} + \frac{\partial L}{\partial y}
$$

The identity term $I$ guarantees a direct, undiminished gradient path back to $x$ regardless of how small/degenerate $\partial F/\partial x$ becomes — this is precisely why residual connections combat vanishing gradients in very deep networks and allow training of 50, 101, even 1000+ layer networks. Intuitively, the network only needs to learn a *residual correction* $F(x)$ on top of the identity, which is an easier optimization target than learning a full transformation from scratch (if the optimal function is close to identity, $F$ just needs to learn ≈0, which is easy for the optimizer to find, vs. a plain network needing to learn the identity mapping exactly through nonlinear layers, which is empirically hard).

```python
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity          # the residual/skip connection
        return self.relu(out)
```

**Practical tips:**
- Prefer ResNet-style skip connections in any custom deep architecture beyond ~10 layers.
- 1×1 conv bottlenecks are a cheap, standard way to control compute budget.
- EfficientNet-style compound scaling is a good mental model even outside vision: when scaling a model up, consider width/depth/resolution jointly rather than in isolation.

### Interview Questions

1. **Derive the output spatial dimension formula for a convolution layer and compute it for a 32×32 input, 5×5 kernel, stride 1, no padding.**
   Formula: $W_{out} = \lfloor (W - F + 2P)/S \rfloor + 1$. Plugging in $W=32, F=5, S=1, P=0$: $\lfloor(32-5+0)/1\rfloor+1 = 27+1=28$. Output is $28\times28$.

2. **Why does parameter sharing in CNNs help generalization compared to a fully-connected layer on the same image?**
   The same kernel weights slide across all spatial positions, so a feature detector (e.g., an edge detector) learned from one part of the image is automatically applied everywhere — this gives translation equivariance and drastically reduces the parameter count (kernel size × channels, independent of image resolution) compared to a fully-connected layer that would need a separate weight for every pixel-to-pixel connection, reducing overfitting risk and the amount of data needed to learn useful filters.

3. **What is the receptive field of a stack of three 3×3 conv layers (stride 1) compared to a single 7×7 conv layer? What's the trade-off?**
   Three stacked 3×3 convs give receptive field $3+2+2=7$ (same as one 7×7 conv) but with far fewer parameters ($3\times(3\times3\times C^2) = 27C^2$ vs. $7\times7\times C^2=49C^2$ for $C$ channels in/out) and two extra nonlinearities in between, increasing representational capacity/depth for the same receptive field. This is the core insight behind VGG's design.

4. **Explain the vanishing gradient / optimization problem that ResNet's skip connections solve, and derive why the gradient doesn't vanish through a residual block.**
   Very deep "plain" (non-residual) networks empirically get *harder to optimize* as depth increases beyond a point — training error can increase with added layers, not from overfitting but from optimization difficulty (the "degradation problem"). In a residual block $y=F(x)+x$, $\partial y/\partial x = \partial F/\partial x + I$. Even if $\partial F/\partial x \to 0$ (vanishing), the additive identity term ensures $\partial y/\partial x \to I$, so gradients propagate through the skip path with magnitude ≈1 regardless of depth, avoiding the exponential shrinkage seen in plain deep networks.

5. **What's the purpose of a 1×1 convolution? Give two distinct use cases.**
   A 1×1 conv mixes information purely across channels (no spatial mixing), acting as a per-pixel linear (or, with an activation, nonlinear) projection. Use cases: (1) dimensionality reduction/bottleneck — reduce channel count cheaply before an expensive 3×3/5×5 conv (ResNet bottleneck, Inception); (2) adding nonlinear depth/capacity without changing spatial resolution or incurring the cost of a larger kernel.

6. **Compare max pooling and average pooling — when would you prefer one over the other?**
   Max pooling picks the strongest activation in each window, good for detecting the *presence* of a sharp feature (e.g., edges) and provides small-shift invariance; it can be noise-sensitive (a single outlier spike dominates). Average pooling smooths the response, useful when the aggregate/average signal matters more than the single strongest activation, and is less prone to amplifying noise, though it can dilute sparse strong signals. Modern architectures often prefer strided convolutions or global average pooling (before the final classifier) over max pooling.

7. **What is the "degradation problem" that motivated ResNet, and is it caused by overfitting?**
   No — it's not overfitting. Before ResNet, researchers observed that simply stacking more layers in a *plain* CNN (no skip connections) beyond a certain depth caused *training* error (not just validation/test error) to increase, indicating an *optimization* difficulty (harder to find good weights via gradient descent in very deep plain networks), not a generalization/overfitting issue. Residual connections make it trivially easy for a block to learn "do nothing" (approximate identity by learning $F(x)\approx0$), so adding depth can only help (or be neutral), never hurt optimization the way plain stacking did.

8. **Why do we typically use ReLU (or GELU/Swish) rather than sigmoid/tanh in CNN hidden layers?**
   CNNs are typically very deep (tens to hundreds of layers); sigmoid/tanh saturate and cause vanishing gradients through the chain rule across so many layers, while ReLU's gradient is exactly 1 for positive inputs, propagating gradient signal without shrinkage through active paths, enabling much deeper networks to train effectively (combined with normalization and skip connections).

9. **Explain the Inception module's motivation. Why use multiple kernel sizes in parallel instead of picking one?**
   Different objects/features in an image appear at different scales — a single fixed kernel size is a strong prior that may not match all relevant feature scales in the data. The Inception module runs 1×1, 3×3, 5×5 convolutions (and pooling) in parallel on the same input and concatenates their outputs, letting the network learn which scales matter for each layer/task rather than committing to one scale by architectural choice; 1×1 convolutions are used before the expensive 3×3/5×5 branches to reduce channel dimensionality and control compute cost.

10. **What does EfficientNet's "compound scaling" mean, and why is it better than scaling only depth or only width?**
    Compound scaling jointly increases network depth, width (channels), and input resolution by a fixed set of ratios (derived via small-scale grid search / NAS) rather than scaling only one dimension. Scaling only depth risks vanishing gradients / diminishing returns per added layer; scaling only width risks a network that's wide but shallow, unable to capture higher-level abstractions; scaling only resolution without matching capacity wastes the extra input detail. Balancing all three empirically achieves a better accuracy-per-FLOP trade-off than any single-dimension scaling strategy.

11. **How would you reduce the spatial size of feature maps — compare pooling vs. strided convolution.**
    Pooling (e.g., max pool with stride 2) downsamples using a fixed, parameter-free aggregation rule; a strided convolution (stride 2) downsamples while *learning* the aggregation function via trainable kernel weights, potentially capturing more useful information during the downsampling itself at the cost of more parameters/compute. Many modern architectures (e.g., ResNet variants) replace pooling with strided convolutions for this reason.

12. **A CNN trained on 224×224 images is evaluated on 512×512 images and performance drops significantly. What might be happening, and how would you address it?**
    If the network has fully-connected layers expecting a fixed flattened spatial size, the architecture literally may not support the new resolution without modification. Even with a global-average-pool + FC head (resolution-agnostic), the effective receptive field / object-to-image scale ratio changes at test time, causing a train/test distribution mismatch (features learned at one scale don't transfer perfectly to objects appearing relatively smaller/larger). Fixes: use architectures with global average pooling instead of flatten+FC, fine-tune (or at least recalibrate BatchNorm statistics) at the target resolution, or use multi-scale training augmentation.

13. **What's the parameter count of a Conv2d layer with kernel 3×3, 64 input channels, 128 output channels (with bias)? Show the calculation.**
    $(3\times3\times64+1)\times128 = (576+1)\times128 = 577\times128 = 73{,}856$ parameters.

14. **Why is global average pooling often used at the end of a CNN instead of flattening + a large fully-connected layer?**
    Flattening a feature map (e.g., $7\times7\times512$) into a dense layer requires a huge weight matrix ($7\times7\times512\times\text{num\_classes}$), which is parameter-heavy and overfitting-prone, and hard-codes a fixed input resolution. Global average pooling reduces each channel to a single scalar (spatial average), producing a fixed-size vector regardless of input resolution, drastically cutting parameters and making the network resolution-agnostic (as used in ResNet, Inception, etc.).

15. **You're building a segmentation model and need pixel-precise output. Why can't you just stack conv+pool layers and predict at the final (small) resolution — what architecture change is needed?**
    Repeated pooling/downsampling discards spatial resolution and precise location information needed for pixel-level predictions. Segmentation architectures (e.g., U-Net, FCN) use an encoder-decoder structure with an **upsampling path** (transposed convolutions or interpolation) that restores resolution, combined with **skip connections** from corresponding encoder layers to reinject the fine-grained spatial detail lost during downsampling, giving pixel-precise output while still benefiting from the encoder's deep semantic features.

---

## Sequence Models

### RNN Basics, Backpropagation Through Time, Vanishing Gradient in RNNs

**Vanilla RNN.** At each timestep $t$, given input $x_t$ and previous hidden state $h_{t-1}$:

$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h), \qquad \hat y_t = W_{hy} h_t + b_y
$$

The **same weights** ($W_{hh}, W_{xh}, W_{hy}$) are reused at every timestep — this is parameter sharing across time, analogous to CNN spatial parameter sharing, letting the network handle variable-length sequences with a fixed parameter count.

**Backpropagation Through Time (BPTT).** "Unroll" the RNN across all $T$ timesteps into an equivalent deep feedforward computation graph, then apply standard backprop. The gradient of the loss at time $T$ w.r.t. an early hidden state $h_k$ ($k < T$) involves a product of Jacobians across every intermediate timestep:

$$
\frac{\partial L_T}{\partial h_k} = \frac{\partial L_T}{\partial h_T}\prod_{t=k+1}^{T} \frac{\partial h_t}{\partial h_{t-1}} = \frac{\partial L_T}{\partial h_T}\prod_{t=k+1}^{T} W_{hh}^\top \, \text{diag}(\tanh'(z_t))
$$

**Vanishing/exploding gradient in RNNs.** This product of $(T-k)$ Jacobians is exactly analogous to depth in a feedforward net, except here "depth" = sequence length, which can be hundreds or thousands of steps. If the largest singular value of $W_{hh}$ (scaled by $\tanh'$, which is $\le 1$) is $<1$, the product shrinks geometrically → vanishing gradient, meaning the network effectively **cannot learn long-range dependencies** (gradient from distant timesteps is numerically ≈0 by the time it reaches early timesteps). If $>1$, the product explodes. This structural issue — not just a training nuisance — directly motivated LSTM/GRU gating and, ultimately, attention/Transformers.

```python
import torch.nn as nn

rnn = nn.RNN(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
x = torch.randn(8, 20, 32)  # (batch, seq_len, input_size)
out, h_n = rnn(x)
```

### LSTM: Gates and Cell State — Full Mechanics

LSTM introduces a separate **cell state** $c_t$ (long-term memory) alongside the hidden state $h_t$, regulated by three gates, each a sigmoid-activated function of $[h_{t-1}, x_t]$:

$$
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{(forget gate)}
$$
$$
i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \quad \text{(input gate)}
$$
$$
o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \quad \text{(output gate)}
$$
$$
\tilde c_t = \tanh(W_c [h_{t-1}, x_t] + b_c) \quad \text{(candidate cell content)}
$$
$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t \quad \text{(cell state update — ADDITIVE, not multiplicative)}
$$
$$
h_t = o_t \odot \tanh(c_t) \quad \text{(hidden state / output)}
$$

**Why LSTM combats vanishing gradients:** The cell state update $c_t = f_t\odot c_{t-1} + i_t\odot\tilde c_t$ is **additive**. The gradient of $c_t$ w.r.t. $c_{t-1}$ is simply $f_t$ (elementwise), not a repeated matrix multiplication compounded with an activation derivative like in vanilla RNNs. When the forget gate $f_t \approx 1$, gradient flows through the cell state almost unchanged across many timesteps — this is often called the "constant error carousel." The gating mechanism lets the network *learn* when to preserve vs. overwrite memory, rather than being structurally forced to decay every step.

- **Forget gate** ($f_t$): decides what fraction of old cell state to keep.
- **Input gate** ($i_t$): decides how much new candidate information to write in.
- **Output gate** ($o_t$): decides how much of the (squashed) cell state to expose as the hidden state/output.

```python
import torch.nn as nn

lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
x = torch.randn(8, 20, 32)
out, (h_n, c_n) = lstm(x)
```

**Practical tip:** Initialize the forget gate bias to a positive value (e.g., 1.0) rather than 0 — this biases the network toward *remembering* by default early in training, which empirically improves learning of long-range dependencies (a well-known trick from Jozefowicz et al.).

### GRU vs. LSTM

GRU (Gated Recurrent Unit) simplifies LSTM: merges cell state and hidden state into one, and uses 2 gates instead of 3.

$$
z_t = \sigma(W_z[h_{t-1},x_t]) \quad \text{(update gate — combines LSTM's forget+input roles)}
$$
$$
r_t = \sigma(W_r[h_{t-1},x_t]) \quad \text{(reset gate)}
$$
$$
\tilde h_t = \tanh(W[r_t \odot h_{t-1}, x_t])
$$
$$
h_t = (1-z_t)\odot h_{t-1} + z_t \odot \tilde h_t
$$

| Aspect | LSTM | GRU |
|---|---|---|
| Gates | 3 (forget, input, output) | 2 (update, reset) |
| States | Separate cell state $c_t$ + hidden state $h_t$ | Single hidden state $h_t$ |
| Parameters | More (≈4× hidden×input matrices) | Fewer (≈3×), faster to train |
| Performance | Often slightly better on very long sequences / large datasets | Often comparable, sometimes better on smaller datasets, faster |
| When to prefer | Large data, need finer memory control | Faster iteration, limited compute/data, similar performance often achievable |

```python
gru = nn.GRU(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
```

**Practical tip:** In practice, try both — performance differences are often task-dependent and small; GRU's efficiency makes it a good first try, especially under compute constraints. Neither is commonly used from scratch for new large-scale sequence tasks today — Transformers have largely displaced both.

### Seq2Seq Architectures, Encoder-Decoder

**Encoder-decoder** framework: an **encoder** RNN/LSTM/GRU consumes the input sequence and compresses it into a fixed-size context vector (the final hidden state), and a **decoder** RNN generates the output sequence conditioned on that context vector (and its own previous outputs, autoregressively).

$$
\text{context} = h_T^{enc} \quad (\text{encoder's final hidden state})
$$
$$
h_t^{dec} = f(h_{t-1}^{dec}, y_{t-1}, \text{context}), \qquad \hat y_t = g(h_t^{dec})
$$

**Bottleneck problem:** Compressing an arbitrarily long input into a single fixed-size vector is a severe information bottleneck — for long sequences, early input tokens' influence on the context vector gets diluted/overwritten by the time the encoder finishes (exacerbated by the same vanishing-gradient dynamics as before). This motivated **attention** (covered next section): instead of relying on a single fixed context vector, the decoder can attend to *all* encoder hidden states at every decoding step, directly addressing the bottleneck and long-range dependency problem — this was the direct historical precursor to the Transformer.

```python
class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src, tgt):
        _, (h, c) = self.encoder(self.embed(src))
        dec_out, _ = self.decoder(self.embed(tgt), (h, c))
        return self.fc(dec_out)
```

### Interview Questions

1. **Derive why vanilla RNNs suffer from vanishing/exploding gradients over long sequences.**
   BPTT unrolls the RNN into $T$ layers sharing weights $W_{hh}$. The gradient at an early timestep $k$ from a loss at timestep $T$ involves $\prod_{t=k+1}^{T} W_{hh}^\top \text{diag}(\tanh'(z_t))$ — a product of $(T-k)$ Jacobian matrices. If the dominant singular values of these Jacobians are consistently $<1$ (common, since $\tanh'\le1$), the product shrinks exponentially with sequence length → vanishing gradient; if consistently $>1$, it explodes. This is structurally identical to the depth problem in feedforward nets, but "depth" is now sequence length, which can be very large.

2. **Explain the LSTM cell state update equation and why it mitigates vanishing gradients.**
   $c_t = f_t\odot c_{t-1} + i_t\odot\tilde c_t$. Because this is an *additive*, elementwise update rather than a repeated matrix multiplication, $\partial c_t/\partial c_{t-1} = f_t$ (elementwise, no matrix multiply, no activation-derivative compounding beyond the gate itself). When $f_t\approx1$, gradients pass through nearly unchanged across many timesteps, avoiding the geometric shrinkage that plagues vanilla RNN hidden-state recurrence.

3. **Walk through each of the LSTM's three gates and their role.**
   Forget gate $f_t=\sigma(W_f[h_{t-1},x_t])$ decides what fraction of the previous cell state to retain; input gate $i_t=\sigma(W_i[h_{t-1},x_t])$ decides how much new candidate content $\tilde c_t=\tanh(\cdot)$ to write into the cell state; output gate $o_t=\sigma(W_o[h_{t-1},x_t])$ decides how much of the (tanh-squashed) cell state to expose as the visible hidden state $h_t$. All three are learned sigmoid gates in $[0,1]$, allowing soft, differentiable "read/write/erase" control over memory.

4. **Compare GRU and LSTM — when might you choose one over the other in an interview-style trade-off discussion?**
   GRU merges the cell and hidden state and uses 2 gates instead of 3, giving fewer parameters and faster training/inference; LSTM's separate cell state and 3-gate mechanism gives finer-grained control over memory, which can matter for very long sequences or large-data regimes. In practice, performance is often comparable, so GRU is a reasonable default when compute/iteration speed matters, while LSTM is worth trying if the task specifically involves very long-range dependencies and there's compute budget to spare. (In modern practice, both are frequently superseded by attention-based/Transformer models.)

5. **What is the "bottleneck problem" in classic seq2seq encoder-decoder models, and how does attention address it?**
   The encoder compresses the entire input sequence into a single fixed-size context vector (final hidden state), which becomes an information bottleneck for long sequences — the decoder only has indirect, diluted access to early input tokens. Attention lets the decoder directly access *all* encoder hidden states at every decoding step (weighted by relevance), removing the fixed-size-vector bottleneck and providing a much shorter gradient path back to any input position.

6. **You have a sequence task with sequences of length 2000+. Would you use a vanilla RNN, LSTM, or Transformer, and why?**
   Vanilla RNN would almost certainly fail to learn long-range dependencies at this length due to vanishing gradients. LSTM handles moderately long sequences much better via gating, but even LSTMs degrade on extremely long sequences and are sequential (slow) to train. A Transformer (or a long-context variant with efficient attention) is generally preferred: attention gives a direct $O(1)$-hop path between any two positions regardless of distance (no vanishing-gradient chain across time), and training is fully parallelizable across the sequence, unlike the inherently sequential RNN/LSTM.

7. **Why must an RNN be processed sequentially (unlike a Transformer), and what's the practical training cost of this?**
   Each hidden state $h_t$ depends on $h_{t-1}$, creating a strict sequential dependency — you cannot compute $h_t$ before $h_{t-1}$ is known. This prevents parallelizing computation across the time dimension during training (though you can parallelize across the batch dimension), making RNN/LSTM training substantially slower on modern parallel hardware (GPUs/TPUs) for long sequences compared to Transformers, which compute attention over all positions in parallel.

8. **Derive the parameter count of an LSTM layer with input size $d$ and hidden size $h$.**
   Each of the 4 LSTM sub-computations (forget, input, output gates + candidate cell) has its own weight matrix operating on the concatenated $[h_{t-1}, x_t]$ vector (size $h+d$) mapping to size $h$, plus a bias of size $h$: total params $= 4 \times [(h+d)\times h + h] = 4h(h+d+1)$.

9. **What is teacher forcing in seq2seq training, and what problem can it cause at inference time?**
   Teacher forcing feeds the *ground-truth* previous token as decoder input during training (rather than the model's own previous prediction), which stabilizes and speeds up training since errors don't compound across timesteps. The problem: at inference, the model must condition on its *own* (possibly imperfect) previous predictions, creating a train/inference mismatch ("exposure bias") — small errors early in generation can compound since the model never learned to recover from its own mistakes during training. Mitigations: scheduled sampling (gradually mix in model predictions during training), or fully non-teacher-forced training schemes.

10. **A candidate says "GRUs always outperform LSTMs because they have fewer parameters." Critique this statement.**
    This is an overgeneralization. Fewer parameters means GRUs train faster and may generalize better on smaller datasets (less overfitting risk), but LSTM's extra gating flexibility (separate cell/hidden state, 3 gates) can outperform GRU on tasks needing finer memory control, particularly with large datasets and very long-range dependencies, where the additional capacity is beneficial rather than a liability. Empirically, neither uniformly dominates — the right choice is task- and data-dependent, and should be validated experimentally, not assumed from parameter count alone.

11. **What does "backpropagation through time" (BPTT) mean concretely, and how does truncated BPTT differ?**
    BPTT unrolls the RNN's recurrent computation across all $T$ timesteps into an equivalent feedforward computational graph, then applies standard reverse-mode backprop across the entire unrolled graph, accumulating gradients for the shared weights at every timestep. Truncated BPTT limits backpropagation to only the last $k$ timesteps (detaching/stopping gradient beyond that window) to bound memory/compute cost for very long sequences, trading off the ability to learn dependencies longer than $k$ steps for tractability.

12. **Why is $\tanh$ (not ReLU) traditionally used inside the recurrent update of vanilla RNNs/LSTMs?**
    $\tanh$ is bounded in $(-1,1)$, which helps keep the hidden/cell state values numerically bounded across many recurrent applications — unbounded activations like ReLU applied repeatedly across timesteps in the recurrence could cause values (and gradients) to grow without bound over long sequences. (Note: this doesn't fully eliminate vanishing gradients — $\tanh'\le1$ still contributes to shrinkage — but it addresses the exploding-value/numerical-stability side.)

---

## Attention and Transformers

### Attention Mechanism: Query/Key/Value, Scaled Dot-Product Attention

**Core idea:** for each output position, compute a weighted combination of all input representations, where the weights ("attention") reflect how *relevant* each input is to that output position — a direct, learnable content-based lookup, replacing the RNN's fixed sequential bottleneck.

**Query, Key, Value:** each input token's embedding is linearly projected into three vectors:

$$
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V
$$

Intuition: $Q$ is "what am I looking for," $K$ is "what do I contain" (used for matching against queries), $V$ is "what do I actually offer if selected" (the content that gets aggregated).

**Scaled dot-product attention:**

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

Steps: (1) compute similarity scores $QK^\top$ (dot product between every query and every key — higher dot product = more "aligned"/relevant); (2) scale by $1/\sqrt{d_k}$; (3) softmax over the key dimension to get a probability distribution (attention weights) per query; (4) weighted-sum the value vectors using these weights.

**Why scale by $\sqrt{d_k}$:** If $q, k$ have i.i.d. components with variance 1, their dot product $q\cdot k = \sum_{i=1}^{d_k} q_i k_i$ has variance $d_k$ (sum of $d_k$ terms each with variance ≈1) — so raw dot products grow with $d_k$. Large-magnitude logits pushed into softmax produce an extremely peaked (near one-hot) distribution, driving the softmax into its saturating regime where gradients vanish (softmax's Jacobian $\to 0$ when one input dominates). Dividing by $\sqrt{d_k}$ renormalizes the dot product back to unit variance, keeping softmax inputs in a well-conditioned range regardless of $d_k$, preserving healthy gradients.

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)   # (..., seq_q, seq_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights
```

### Self-Attention vs. Cross-Attention, Multi-Head Attention

**Self-attention:** $Q, K, V$ are all derived from the *same* sequence — every token attends to every other token (including itself) in the same input, capturing intra-sequence dependencies (e.g., syntactic/semantic relationships within a sentence) regardless of distance.

**Cross-attention:** $Q$ comes from one sequence (e.g., decoder), $K, V$ come from a *different* sequence (e.g., encoder output) — used in encoder-decoder Transformers (e.g., translation) so the decoder can attend to relevant parts of the source sequence when generating each output token. This is the direct successor to the RNN seq2seq attention mechanism.

**Multi-head attention:** Instead of one attention computation over the full $d_{model}$ dimension, split into $h$ heads, each operating on a lower-dimensional projection ($d_k = d_{model}/h$), run scaled dot-product attention independently per head, then concatenate and linearly project back:

$$
\text{head}_i = \text{Attention}(QW_Q^i, KW_K^i, VW_V^i), \qquad \text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)W_O
$$

**Why multiple heads:** Each head can specialize in different types of relationships (e.g., one head tracks syntactic dependency, another tracks coreference, another local n-gram patterns) — a single attention computation over the full dimensionality would average all these patterns together into one distribution per query, losing the ability to represent multiple distinct relationship types simultaneously. Empirically, multi-head attention consistently outperforms single-head attention of equivalent total parameter count.

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
x = torch.randn(2, 10, 512)  # (batch, seq_len, embed_dim)
out, attn_weights = mha(x, x, x)  # self-attention: Q=K=V=x
```

### Transformer Architecture

**Positional encoding.** Attention itself is permutation-invariant (no inherent notion of order — swapping two input tokens produces the same set of attention outputs, just reordered), so position information must be injected explicitly. Original Transformer uses fixed sinusoidal encodings:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \qquad PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

added elementwise to token embeddings. Different frequencies at different dimensions let the model learn to attend to relative positions (since $\sin/\cos$ of a sum can be expressed as a linear function of $\sin/\cos$ of the parts — enabling relative-position reasoning via linear combinations). Alternatives used in modern models: **learned absolute** positional embeddings, **relative positional encodings**, and **RoPE** (rotary position embedding, used in LLaMA/GPT-NeoX-style models) which rotates $Q,K$ vectors by an angle proportional to position, embedding relative position directly into the dot-product attention score.

**Encoder block:** Multi-head self-attention → residual add → LayerNorm → position-wise feed-forward network (FFN) → residual add → LayerNorm.

**Decoder block:** Masked multi-head self-attention (causal mask prevents attending to future tokens) → residual+LayerNorm → cross-attention over encoder output → residual+LayerNorm → FFN → residual+LayerNorm.

**Feed-forward sublayer:** Applied independently to each position (no cross-position mixing — that's attention's job):

$$
\text{FFN}(x) = \max(0, xW_1+b_1)W_2 + b_2 \quad \text{(or GELU instead of ReLU in modern variants)}
$$

Typically expands to $4\times d_{model}$ in the hidden layer, providing per-token nonlinear transformation capacity.

**Residual connections:** Every sublayer (attention, FFN) is wrapped as $x + \text{Sublayer}(x)$ — same motivation as ResNet: guarantees an unimpeded gradient path through very deep stacks (modern Transformers can have 50-100+ layers).

**Pre-norm vs. post-norm:**

| Variant | Formula | Property |
|---|---|---|
| Post-norm (original Transformer) | $x_{l+1} = \text{LayerNorm}(x_l + \text{Sublayer}(x_l))$ | Normalization applied *after* the residual add; empirically harder to train at large depth/without careful warmup — gradients through the residual path are not normalized, which can cause instability early in training. |
| Pre-norm (GPT-2/3, most modern LLMs) | $x_{l+1} = x_l + \text{Sublayer}(\text{LayerNorm}(x_l))$ | Normalization applied *before* the sublayer, so the residual (identity) path is completely clean/unnormalized, giving a direct gradient highway; much more stable at scale and typically removes the need for a long warmup, though can allow representation magnitude to grow across layers without a final norm. |

Most modern large-scale Transformers (GPT-family, LLaMA, etc.) use pre-norm (often plus a final LayerNorm at the very end of the stack) specifically because it makes training deep stacks far more stable.

```python
import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # pre-norm
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + self.dropout(attn_out)
        h2 = self.ln2(x)
        x = x + self.dropout(self.ffn(h2))
        return x
```

### Why Transformers Replaced RNNs for Most Sequence Tasks

| Dimension | RNN/LSTM | Transformer |
|---|---|---|
| Parallelization | Sequential over time — cannot compute $h_t$ before $h_{t-1}$; training is $O(T)$ sequential steps | Fully parallel across sequence positions during training (attention over all pairs computed simultaneously) — much better GPU/TPU utilization |
| Long-range dependencies | Gradient must pass through $T-k$ recurrent steps — vanishing gradient even with LSTM gating over very long sequences | Direct $O(1)$-hop connection between any two positions via attention — no gradient decay with distance |
| Computational complexity | $O(T)$ per layer (sequential), $O(T\cdot d^2)$ total compute | $O(T^2\cdot d)$ per layer (attention matrix) — worse asymptotic complexity for very long sequences, but far more parallelizable in practice on modern hardware |
| Inductive bias | Strong sequential/recency bias built in | Minimal built-in order bias (must be added via positional encoding) — more flexible, but needs more data to learn ordering patterns |

The key practical win is **parallelization**: Transformer training scales far better with modern accelerators since attention is a set of parallel matrix multiplications rather than a sequential recurrence, letting Transformers train efficiently on much larger datasets and model sizes — which combined with the removal of the vanishing-gradient-over-time problem is why they now dominate NLP, vision (ViT), and multimodal tasks. The trade-off is quadratic $O(T^2)$ compute/memory in sequence length, which motivates ongoing research into efficient/sparse/linear attention variants for very long contexts.

### Interview Questions

1. **Derive why dividing by $\sqrt{d_k}$ in scaled dot-product attention is necessary.**
   If query/key components are i.i.d. with mean 0, variance 1, the dot product $q\cdot k=\sum_{i=1}^{d_k}q_ik_i$ has variance $d_k$ (sum of $d_k$ independent unit-variance terms), so its magnitude grows with $\sqrt{d_k}$. Large-magnitude logits fed into softmax produce a very peaked, near-one-hot distribution, pushing softmax into a saturating regime where its Jacobian (and hence gradients) vanish. Dividing by $\sqrt{d_k}$ renormalizes the dot product back to unit variance regardless of dimensionality, keeping softmax well-conditioned and gradients healthy.

2. **What problem does positional encoding solve, and why is attention itself permutation-invariant?**
   Attention computes weighted sums over value vectors using softmax(QK^T) — if you permute the input token order, the same set of attention weights and outputs get permuted correspondingly, but the actual *values* and relationships computed are identical; there's no term in the attention formula that depends on absolute or relative token position. Without explicit positional information, a Transformer could not distinguish "the dog bit the man" from "the man bit the dog" (same set of tokens, same attention computation, different meaning) — positional encodings inject that missing order information into the token representations.

3. **Explain multi-head attention and why using 8 heads of dimension 64 is different from 1 head of dimension 512.**
   Multi-head attention runs $h$ independent attention computations in parallel, each in a lower-dimensional subspace, then concatenates and projects the results. A single head over the full dimensionality computes one softmax distribution per query — averaging together all types of relevant relationships into one weighting. Multiple smaller heads can each specialize (one head might learn syntactic dependencies, another positional/local patterns, another coreference), giving the model the capacity to represent several distinct relationship types simultaneously rather than forcing them into a single shared attention pattern.

4. **Derive the computational complexity of self-attention with respect to sequence length $T$ and model dimension $d$, and explain why this matters for very long sequences.**
   Computing $QK^\top$ is a $(T\times d)\times(d\times T)$ matrix multiply, costing $O(T^2 d)$; softmax and the subsequent weighted sum with $V$ are also $O(T^2 d)$. Total: $O(T^2 d)$ time and $O(T^2)$ memory (to store the attention matrix). This quadratic scaling in $T$ makes vanilla self-attention prohibitively expensive for very long sequences (e.g., $T=100{,}000$), motivating sparse/linear/windowed attention variants (e.g., Longformer, linear attention, FlashAttention for memory-efficient exact computation).

5. **What is the difference between self-attention and cross-attention? Where does each appear in a standard encoder-decoder Transformer?**
   Self-attention: $Q,K,V$ all come from the same sequence, used in both the encoder (attending over the source) and the decoder's first attention sublayer (masked, attending over previously generated tokens). Cross-attention: $Q$ comes from the decoder, $K,V$ come from the encoder's output, used in the decoder's second attention sublayer to let generated tokens attend to relevant parts of the source sequence — this is the direct mechanism that lets the decoder condition its output on the (encoded) input.

6. **Why is causal (look-ahead) masking needed in the decoder's self-attention, and how is it implemented?**
   During autoregressive generation, each output token can only depend on previously generated tokens, not future ones (which don't exist yet at inference, and would leak label information during training). The causal mask sets attention scores from position $i$ to any position $j>i$ to $-\infty$ before the softmax, so those positions receive zero attention weight — implemented as an upper-triangular mask added to the raw attention scores before softmax.

7. **Compare pre-norm and post-norm Transformer designs. Why do most modern large-scale models use pre-norm?**
   Post-norm applies LayerNorm *after* the residual addition ($\text{LN}(x+\text{Sublayer}(x))$), meaning the residual/identity path itself gets normalized/warped at every layer, which can make gradient flow through very deep stacks less stable, often requiring careful learning-rate warmup. Pre-norm applies LayerNorm *before* the sublayer ($x+\text{Sublayer}(\text{LN}(x))$), leaving the residual path completely clean, giving a direct, unimpeded gradient highway through arbitrarily many layers — this makes pre-norm substantially more stable to train at the depth/scale of modern LLMs (hence its near-universal adoption in GPT, LLaMA, etc.), at the minor cost of needing a final LayerNorm before the output head since representation magnitude can grow across layers.

8. **Why did Transformers largely replace RNNs/LSTMs for sequence modeling? Name the two most important reasons.**
   (1) Parallelization: RNNs process tokens sequentially (can't compute step $t$ before step $t-1$), while Transformer attention computes relationships between all token pairs simultaneously, enabling far better utilization of parallel hardware (GPUs/TPUs) and faster training at scale. (2) Long-range dependencies: RNN gradients must pass through $O(T)$ sequential steps (vanishing even with LSTM gating over very long sequences), while attention gives every pair of positions a direct, single-hop connection regardless of distance, with no gradient decay from sequence length.

9. **What is the role of the feed-forward sublayer in a Transformer block, given that attention already mixes information across positions?**
   The FFN is applied independently, position-by-position (no cross-position mixing) — it provides per-token nonlinear transformation capacity, effectively acting as the "computation" step after attention has done the "communication"/information-routing step. Attention alone (without an FFN) is a set of weighted averages (fundamentally limited in expressiveness — it's a convex combination of value vectors); the FFN's nonlinearity is what gives the block genuine representational power to transform features, not just re-weight and mix them.

10. **If you removed all residual connections from a deep Transformer, what would you expect to happen during training, and why?**
    Training would likely become very unstable or fail to converge for deep stacks — without the identity/skip path, gradients must flow purely through the composition of many attention+FFN+LayerNorm sublayers, each of which can attenuate or distort the gradient; deep stacks would suffer from the same optimization/degradation difficulties that motivated ResNet in CNNs (vanishing gradients, harder-to-optimize loss landscape), especially since Transformers used in practice are very deep (dozens to over a hundred layers).

11. **Explain RoPE (rotary position embedding) at a conceptual level and why it's preferred over absolute sinusoidal encodings in many modern LLMs.**
    RoPE encodes position by rotating the query and key vectors (treated as pairs of dimensions, like 2D coordinates) by an angle proportional to their absolute position, before computing the dot product. Because rotation is a linear, angle-additive operation, the dot product between a rotated query at position $m$ and rotated key at position $n$ depends only on their *relative* position $m-n$, not their absolute positions — this gives attention scores a built-in relative-position awareness "for free," which generalizes better to sequence lengths not seen during training and integrates directly into the attention computation without needing an added positional embedding vector.

12. **In a Transformer, what happens to attention weights if you forget the softmax's normalization dimension (e.g., normalize over the wrong axis)?**
    Softmax must be applied over the *key* dimension (so that, for each query, the attention weights across all keys sum to 1, forming a valid probability distribution used to weight-average the value vectors). If normalized over the wrong axis (e.g., over queries or over the feature dimension), the weights for a given query would no longer sum to 1, breaking the "weighted average" semantics — the output would no longer be a proper convex combination of value vectors and could have arbitrary, unintended scale, degrading or completely breaking the attention mechanism.

13. **Why does attention (unlike convolution) have effectively unlimited receptive field within a single layer?**
    A convolution's receptive field is bounded by kernel size and grows only linearly with depth (each layer adds a fixed amount). Self-attention computes a score between every pair of positions in the sequence in a single layer — every token can directly attend to every other token, regardless of distance, so the effective "receptive field" of a single self-attention layer spans the entire input sequence immediately, without needing depth to expand it.

14. **A Transformer trained on sequences up to length 512 is evaluated on sequences of length 2048 and performance degrades sharply. What's the likely cause, and what are two mitigation strategies?**
    Likely cause: positional encodings (especially learned absolute ones) were never trained on positions beyond 512, so the model has no meaningful representation for those positions — even sinusoidal encodings, while mathematically defined beyond the trained range, represent an out-of-distribution input the model never learned to use well. Mitigations: (1) use a relative or rotary (RoPE) positional scheme, which generalizes better to unseen lengths (and can be combined with position-interpolation/scaling tricks); (2) fine-tune (or continue pretraining) on longer sequences to adapt the model's positional and attention behavior to the extended range.

15. **What is the intuition for why the FFN hidden dimension is typically 4× the model dimension in standard Transformer blocks?**
    This is an empirically-tuned architectural choice (from the original Transformer paper) that balances representational capacity against compute/parameter cost: the FFN accounts for roughly two-thirds of a Transformer block's parameters, and expanding to $4\times d_{model}$ in the hidden layer before projecting back gives the position-wise transformation enough capacity to be a meaningful nonlinear feature transformer without making the FFN disproportionately dominate the model's total compute budget relative to the attention sublayers.

---

## Practical Deep Learning

### Transfer Learning and Fine-Tuning Strategies

**Transfer learning:** take a model pretrained on a large source dataset/task (e.g., ImageNet for vision, a large text corpus for LLMs) and adapt it to a target task with much less data, leveraging the general features/representations already learned.

**Strategies:**

| Strategy | What's frozen | What's trained | When to use |
|---|---|---|---|
| Feature extraction (frozen backbone) | Entire pretrained backbone | Only a new head (e.g., final linear classifier) | Very little target data; target task similar to source domain |
| Fine-tune last $k$ layers | Early/most layers | Last few layers + new head | Moderate target data; target domain moderately different |
| Full fine-tuning | Nothing | Entire network (often with a low LR) | Larger target dataset; target domain quite different from source |
| Parameter-efficient fine-tuning (LoRA, adapters, prompt tuning) | Nearly all original weights | Small added low-rank/adapter modules | Very large pretrained models (LLMs); limited compute/storage per task |

**Rationale for freezing early layers:** early layers in CNNs/Transformers tend to learn generic, low-level features (edges/textures in vision; general syntax/semantics in language) that transfer well across tasks/domains, while later layers learn increasingly task-specific, high-level representations that need to adapt to the new target task. Freezing early layers also reduces compute cost and overfitting risk when target data is limited.

```python
import torchvision.models as models
import torch.nn as nn

model = models.resnet50(weights="IMAGENET1K_V2")
for param in model.parameters():
    param.requires_grad = False        # freeze backbone

model.fc = nn.Linear(model.fc.in_features, num_classes)  # new trainable head
optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
```

**Practical tips:**
- Use a smaller learning rate for fine-tuned (pretrained) layers than for newly-initialized layers (they're already near a good optimum, don't want to destroy learned features — "differential/discriminative learning rates").
- Gradually unfreeze layers from the top down as training progresses ("progressive unfreezing") if full fine-tuning with limited data risks catastrophic forgetting.
- Watch for **catastrophic forgetting**: aggressive fine-tuning on a narrow target task can destroy general capabilities learned during pretraining — mitigate with lower LR, fewer epochs, or regularization toward the original weights (e.g., elastic weight consolidation, or simply early stopping).

### Self-Supervised and Contrastive Representation Learning (General Principles)

**Core idea:** self-supervised learning (SSL) trains a model on **pretext tasks** constructed automatically from unlabeled data itself (no human annotation), producing representations that transfer well to downstream tasks. This is a general training paradigm, not a specific architecture — pretext tasks range from predicting a masked/missing part of the input (masked modeling) to predicting the relative arrangement of parts, to the dominant modern approach: **contrastive learning**, which trains an encoder so that different "views" of the same underlying instance are pulled close together in embedding space, while views of different instances are pushed apart. (Vision-specific instantiations like SimCLR, MoCo, and MAE, and NLP-specific instantiations like BERT's masked-language-modeling and SBERT's contrastive sentence embeddings, are covered in their respective files — this section covers the shared, modality-agnostic principle.)

**General contrastive objective — InfoNCE:**

$$
L_i = -\log \frac{\exp(\text{sim}(z_i, z_i^+)/\tau)}{\exp(\text{sim}(z_i, z_i^+)/\tau) + \sum_{k=1}^{K}\exp(\text{sim}(z_i, z_k^-)/\tau)}
$$

where $z_i$ is an anchor embedding, $z_i^+$ is a "positive" (a different view/augmentation of the same underlying instance), $z_k^-$ are $K$ "negatives" (embeddings of different instances), $\text{sim}(\cdot,\cdot)$ is typically cosine similarity, and $\tau$ (temperature) controls how sharply the loss penalizes near-ties. This is exactly the softmax cross-entropy loss for a $(K{+}1)$-way classification problem: "which of these $K{+}1$ candidates is the true positive for this anchor?"

**Why it works (the intuition, not the architecture):**
- Solving "identify your own positive among many distractors" forces the encoder to extract features that are (a) **invariant** to the nuisance transformations used to create views (e.g., cropping, masking, paraphrasing — whatever defines "the same instance") and (b) **discriminative** enough to distinguish that instance from every other instance in the negative set. Representations satisfying both properties tend to align with semantically meaningful factors of variation, which is exactly what's useful for downstream tasks.
- Information-theoretically, minimizing InfoNCE is equivalent to maximizing a lower bound on the mutual information $I(z_i; z_i^+)$ between two views of the same instance — the tighter/higher this bound, the more the shared "content" between views is preserved in the representation while nuisance/view-specific detail is discarded.
- Contrastive learning needs no labels because the "label" (which sample is the positive) is generated automatically by the data augmentation/splitting procedure itself — this is what makes it "self"-supervised.

**Failure mode — representational collapse:** without negatives (or another anti-collapse mechanism), an encoder can trivially minimize a naive "pull positives together" objective by mapping *every* input to the same constant embedding (perfect similarity between any "positive" pair, zero useful information). This is why plain contrastive methods need negative samples (or techniques like a stop-gradient + predictor asymmetry, or explicit variance/covariance regularization terms in negative-free SSL methods) — the general point is that any SSL objective must be specifically designed to make the trivial constant/collapsed solution a *bad* one, not just a task that "sounds unsupervised."

```python
import torch
import torch.nn.functional as F

def info_nce_loss(z1, z2, temperature=0.5):
    """
    z1, z2: (batch, dim) embeddings of two augmented views of the same batch of instances.
    z1[i] and z2[i] are a positive pair; all other cross-batch pairs are negatives.
    Modality-agnostic: z1/z2 can come from an image encoder, text encoder, audio encoder, etc.
    """
    z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
    logits = z1 @ z2.T / temperature                     # (batch, batch) similarity matrix
    labels = torch.arange(z1.size(0), device=z1.device)  # diagonal = positive pairs
    loss = F.cross_entropy(logits, labels)                # each row: (K+1)-way classification
    return loss
```

**Practical tips / pitfalls:**
- The choice of "view-generating" transformation (augmentation) *defines* what the encoder becomes invariant to — pick transformations that destroy nuisance information but preserve label-relevant semantics, or you'll learn the wrong invariances.
- Temperature $\tau$ controls the effective "hardness" of the softmax: too low over-emphasizes the single hardest negative (noisy gradients); too high flattens the distribution and weakens the learning signal.
- Downstream evaluation is typically via **linear probing** (freeze the SSL-pretrained encoder, train only a linear classifier on top) — this isolates how good the *representation* is, independent of how much capacity the eval head adds.
- More negatives generally help (a tighter estimate of the contrastive gradient/mutual-information bound), which is why batch size / negative-queue size is a first-order hyperparameter for contrastive methods — a general point true for any modality, not just vision.

### Knowledge Distillation

**Core idea:** train a smaller, cheaper **student** model to reproduce the behavior of a larger, more accurate **teacher** model (or an ensemble), transferring much of the teacher's performance into a model that's faster and lighter to run. It's a general-purpose training technique, not tied to any modality — used to compress vision, NLP, and LLM-style models alike; the deployment-side compression pipeline (combining distillation with pruning/quantization for production serving) is covered in the MLOps file — this section covers the *training mechanics*.

**Soft-label distillation loss (Hinton et al., 2015):** rather than training the student only on hard ground-truth labels, also train it to match the teacher's full output *distribution*, using a temperature-raised softmax:

$$
p_i^{(T)} = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
$$

where $T>1$ "softens" the distribution — raising $T$ shrinks the gaps between logits, exposing more information about the teacher's *relative* confidence across all classes (which classes it considers plausible-but-wrong), not just the single argmax. The combined loss:

$$
L = \alpha \cdot L_{CE}(y, p_{\text{student}}) + (1-\alpha)\cdot T^2 \cdot L_{KL}\!\left(p_{\text{teacher}}^{(T)} \,\|\, p_{\text{student}}^{(T)}\right)
$$

- $L_{CE}$: standard hard-label cross-entropy against ground truth (keeps the student anchored to the true task).
- $L_{KL}$: KL divergence between the teacher's and student's softened output distributions (the actual "distillation" signal) — this transfers the teacher's learned notion of *class similarity structure* ("this image is mostly a cat, but has some dog-like features") that hard one-hot labels simply don't encode.
- The $T^2$ factor rescales the KL term's gradient magnitude to compensate for gradients through a softened softmax being scaled down by $\approx 1/T^2$, keeping the two loss terms' gradient magnitudes comparable.

**Why soft labels help beyond just "more training data":** a hard label only says "this is a 7"; a teacher's softened output might say "this is mostly a 7, somewhat like a 1, definitely not a 0" — this per-example, per-class relational information ("dark knowledge") is a richer training signal than a one-hot vector, especially valuable when the student has much less capacity and benefits from being told *which mistakes are more reasonable than others*.

```python
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, targets, T=4.0, alpha=0.5):
    hard_loss = F.cross_entropy(student_logits, targets)
    soft_teacher = F.softmax(teacher_logits / T, dim=-1)
    soft_student = F.log_softmax(student_logits / T, dim=-1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)
    return alpha * hard_loss + (1 - alpha) * soft_loss
```

**Practical tips / pitfalls:**
- The teacher is typically frozen (eval mode, no gradient) during distillation — only the student is trained.
- Common temperature range $T \in [2, 10]$; higher $T$ transfers more "dark knowledge" but can also transfer more noise if the teacher itself is poorly calibrated.
- Distillation works best when the student has enough capacity to approximate the teacher's decision boundary — an extremely small student can't recover a much larger teacher's full performance regardless of loss design.
- Response-based distillation (matching output logits, above) is the simplest form; feature-based distillation (matching intermediate activations/attention maps between teacher and student) can transfer more structure but requires architectural compatibility between teacher/student layers.

### Multi-Task Learning

**Core idea:** train a single model to perform multiple related tasks simultaneously, typically via a **shared backbone/encoder** feeding multiple **task-specific heads**:

$$
\hat y_1 = h_1(f_\theta(x)), \quad \hat y_2 = h_2(f_\theta(x)), \quad \dots, \quad \hat y_T = h_T(f_\theta(x))
$$

where $f_\theta$ (the shared trunk) learns representations useful across all tasks, and each lightweight head $h_t$ specializes to its own task's output space. Trained end-to-end with a combined loss:

$$
L = \sum_{t=1}^{T} w_t \, L_t
$$

**Why it can help (inductive bias / regularization argument):** tasks that share underlying structure act as mutual regularizers — features useful for task A (e.g., detecting edges/object boundaries) are often also useful for task B (e.g., depth estimation), so the shared backbone is pushed toward more generally useful representations than it would learn from any single task alone, and effectively sees more (task-diverse) training signal per parameter. This can improve sample efficiency, especially when individual tasks have limited labeled data.

**The core challenge — loss weighting / task balancing:** naively summing losses with $w_t=1$ almost never works well in practice, because:
- Tasks with different loss scales (e.g., a classification cross-entropy vs. a regression MSE with a large numeric range) will have their gradients dominated by whichever loss happens to have larger magnitude, regardless of which task actually matters more.
- Tasks can have **conflicting gradients** — the gradient direction that helps task A can actively hurt task B ("negative transfer"), especially when tasks are only weakly related.
- Tasks often learn at different *rates*, so a fixed weighting that's reasonable early in training can become wrong later (one task saturates while another is still underfitting).

**Common mitigations:**

| Approach | Mechanism |
|---|---|
| Manual/fixed loss weights | Simplest; requires hyperparameter search, doesn't adapt over training |
| Uncertainty weighting (Kendall et al.) | Learn a per-task weight from a learned task-noise/uncertainty parameter, so noisier/harder tasks are automatically down-weighted |
| GradNorm | Dynamically rescale each task's loss so its gradient norm relative to the shared backbone matches a target ratio across tasks, balancing learning speed |
| PCGrad / gradient surgery | Detect conflicting gradients (negative cosine similarity between two tasks' gradients) and project one onto the normal plane of the other to remove the conflicting component |
| Hard vs. soft parameter sharing | Hard: one shared trunk (as above) — max efficiency, max interference risk. Soft: separate per-task backbones with a cross-task regularization term (e.g., encouraging similar weights) — less interference, less sharing benefit |

```python
import torch.nn as nn

class MultiTaskModel(nn.Module):
    def __init__(self, backbone, task_output_dims):
        super().__init__()
        self.backbone = backbone                      # shared trunk
        self.heads = nn.ModuleDict({
            name: nn.Linear(backbone.out_dim, dim)
            for name, dim in task_output_dims.items()
        })

    def forward(self, x):
        features = self.backbone(x)
        return {name: head(features) for name, head in self.heads.items()}

# Fixed-weight combined loss (simplest baseline)
loss = sum(weights[t] * loss_fns[t](outputs[t], targets[t]) for t in tasks)
```

**Practical tips / pitfalls:**
- Start with fixed weights tuned by hand/grid search on a per-task validation metric before reaching for GradNorm/uncertainty-weighting machinery — simple weighting is a strong, cheap baseline.
- Watch for one "easy" task dominating training (its loss drops fast, gradients shrink, and the optimizer effectively stops paying attention to it while harder tasks are still undertrained) — or the reverse, if its loss scale is simply numerically larger.
- Group only genuinely related tasks on a shared backbone; forcing unrelated tasks to share a trunk risks negative transfer that hurts every task versus training them separately.
- Multi-task learning is closely related to (but distinct from) transfer learning and meta-learning — MTL trains on all tasks jointly and simultaneously, rather than sequentially (transfer learning) or learning-to-learn across a distribution of tasks (meta-learning).

### Debugging Training

| Symptom | Likely causes | What to check/do |
|---|---|---|
| **Loss is NaN** | LR too high; FP16 overflow/underflow without loss scaling; log(0)/div-by-0 in loss; exploding gradients; bad data (NaN/Inf inputs) | Lower LR; verify `GradScaler`/loss scaling; check loss implementation numerically stable (`log_softmax` not `log(softmax)`); add gradient clipping; use `torch.autograd.set_detect_anomaly(True)`; audit input batches for NaN/Inf |
| **Loss plateaus immediately / never decreases** | LR too low (no progress) or too high (bouncing/divergence); bad init; frozen params by mistake; label/data pipeline bug; vanishing gradients | Try overfitting a tiny (e.g., 10-example) subset as a sanity check — if that fails, it's a bug, not a tuning issue; sweep LR; verify `requires_grad`; inspect a raw batch and labels manually |
| **Loss decreases then plateaus at a mediocre value** | Model capacity insufficient; LR schedule decaying too fast/early; need more regularization tuning or more data | Increase model capacity; adjust LR schedule; check for underfitting vs. genuine task difficulty |
| **Training loss ↓, validation loss ↑ (overfitting)** | Model memorizing training set | Add dropout/weight decay/augmentation; reduce capacity; early stopping; get more data |
| **Loss oscillates wildly** | LR too high; batch size too small with high-variance gradients; unstable normalization (small batch + BatchNorm) | Lower LR; increase batch size or switch to GroupNorm/LayerNorm; add gradient clipping |
| **Training is very slow to start improving** | Poor initialization; no LR warmup with adaptive optimizer + LayerNorm; data loading bottleneck (not a modeling issue) | Add warmup; verify init scheme matches activation; profile data loader vs. GPU utilization |

```python
# Sanity check: can the model even memorize a tiny batch? (classic first debugging step)
tiny_x, tiny_y = next(iter(dataloader))
tiny_x, tiny_y = tiny_x[:8], tiny_y[:8]
for step in range(500):
    optimizer.zero_grad()
    loss = criterion(model(tiny_x), tiny_y)
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(step, loss.item())
# If loss doesn't go to ~0 on this tiny set, there is a bug (not a hyperparameter issue).
```

**Learning rate too high — symptoms:** loss decreases initially then diverges/oscillates/spikes to NaN; validation and training loss both erratic; gradient norms very large.
**Learning rate too low — symptoms:** loss decreases extremely slowly (barely visible over many epochs); training looks like it's "stuck" but a very long run does show slow improvement; gradient norms fine but tiny effective steps.

### Hardware Considerations

**GPU vs TPU basics:** GPUs are general-purpose massively parallel processors with flexible programmability (CUDA), optimized via Tensor Cores for mixed-precision matrix multiplication (the core op in DL — matmuls dominate compute in both conv and attention layers). TPUs (Google's custom ASICs) are purpose-built for exactly the matmul-heavy workloads of deep learning, offering high throughput per watt/dollar at scale, tightly integrated with frameworks like JAX/TensorFlow, but less flexible for custom/non-standard ops compared to GPUs.

**Mixed precision (FP16/BF16):** See "Training Deep Networks" section above — halves memory footprint, leverages specialized hardware (Tensor Cores) for 2-8x throughput gains, essential for training large models within memory/compute budgets. BF16 trades a bit of mantissa precision for FP32-equivalent dynamic range (no loss scaling typically needed); FP16 needs loss scaling to avoid gradient underflow.

**Distributed training — Data Parallel vs. Model Parallel:**

| Strategy | What's split across devices | Communication pattern | When needed |
|---|---|---|---|
| **Data Parallel (DP/DDP)** | Each device holds a full copy of the model; data batch is split across devices | All-reduce gradients across devices after each backward pass, then every device applies the same (averaged) update | Model fits on one device; want to scale throughput by processing more data in parallel |
| **Model Parallel** | Model itself is split across devices (e.g., different layers on different GPUs, or "tensor parallel" splitting individual weight matrices) | Activations/intermediate results passed between devices during forward/backward | Model too large to fit on a single device's memory |
| **Pipeline Parallel** | Different layers/stages on different devices, processed in a pipelined fashion across micro-batches | Sequential handoff of activations between stages, overlapped across micro-batches to keep devices busy | Very large models, combined with model parallelism to improve device utilization |
| **Fully Sharded Data Parallel (FSDP) / ZeRO** | Model *parameters, gradients, and optimizer states* are sharded across devices (not just data) | Gather/scatter parameter shards as needed during forward/backward | Very large models where even a single copy of parameters+optimizer state doesn't fit comfortably; combines memory savings of model parallelism with data-parallel-like usage pattern |

```python
# Minimal DDP sketch (conceptual, single-node multi-GPU)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")
model = MyModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])
# Each process gets a different data shard; DDP automatically all-reduces gradients
# after loss.backward(), so optimizer.step() applies identical updates on every replica.
```

**Practical tips:**
- Start with data parallelism (DDP) — simplest, most broadly applicable, use until the model no longer fits on one device.
- Move to model/pipeline/tensor parallelism (or FSDP/ZeRO) only when a single device can't hold the model + activations + optimizer state.
- Always profile: data loading is a common hidden bottleneck that looks like a "GPU is slow" problem but is actually a CPU/IO problem — check GPU utilization (`nvidia-smi`) during training; if it's well below 90%+, suspect the data pipeline, not the model or hardware.

### Interview Questions

1. **You fine-tune a pretrained ResNet on a small target dataset (500 images, 10 classes) and it overfits badly within a few epochs. What would you change?**
   Freeze most/all of the backbone and train only a new classification head (feature extraction rather than full fine-tuning) — with only 500 images, full fine-tuning has far too many trainable parameters relative to data. Add data augmentation (crops, flips, color jitter) to synthetically expand the effective dataset, add dropout/weight decay to the head, and use early stopping based on a held-out validation set. Consider progressively unfreezing only the last block if the target domain differs meaningfully from ImageNet.

2. **Why do we typically use a smaller learning rate for pretrained layers than for a newly initialized head during fine-tuning?**
   Pretrained layers already encode useful, near-optimal representations; a large learning rate risks quickly destroying this learned structure ("catastrophic forgetting") before the new head has had a chance to adapt to it. The new head starts randomly initialized and needs larger updates to move from random to useful quickly. Using differential/discriminative learning rates (smaller for early/pretrained layers, larger for the new head) balances preserving useful features against adapting to the new task.

3. **Your training loss is NaN starting from step ~200. Walk through your exact debugging process.**
   First, check if mixed precision (FP16) is enabled without proper loss scaling — a common and very frequent cause; verify `GradScaler` is used correctly. Next, check the learning rate — try reducing it by 10x to see if NaN disappears (indicates instability/exploding gradients). Inspect the loss function implementation for numerically unstable ops (e.g., manual `log(softmax(x))` instead of the fused, stable `log_softmax`; or division that could hit 0). Enable `torch.autograd.set_detect_anomaly(True)` to pinpoint exactly which op first produces a NaN/Inf. Check the input batch itself for NaN/Inf values (e.g., from a data preprocessing bug or corrupted file). Finally, add/verify gradient clipping as a safety net regardless of the root cause found.

4. **What are the visual signatures of overfitting on a loss/accuracy curve, and how would you distinguish overfitting from a simple LR-too-high problem?**
   Overfitting: training loss steadily decreases while validation loss decreases then starts *increasing* (or validation accuracy plateaus/declines) — a growing gap between train and validation curves over epochs, with training metrics still healthy/improving. LR-too-high: *both* training and validation loss are erratic/spiking/diverging, or training loss itself fails to decrease smoothly — the training curve itself looks unstable, not just diverging from validation. The key distinguishing signal is whether the *training* curve itself looks healthy (overfitting) or unhealthy/unstable (LR issue).

5. **Explain the difference between data parallelism and model parallelism, and how you'd decide which to use.**
   Data parallelism replicates the full model on every device and splits the *data batch* across devices, synchronizing gradients (e.g., via all-reduce) after each backward pass — used when the model fits comfortably on one device and you want to scale throughput. Model parallelism splits the *model itself* (layers or even individual weight matrices) across devices — necessary when the model is too large to fit in a single device's memory, regardless of batch size. Decision rule: use data parallelism by default; only add model/pipeline/tensor parallelism (or parameter-sharding approaches like FSDP/ZeRO) when the model no longer fits on one device.

6. **What is loss scaling in mixed-precision training, and why is it needed for FP16 but generally not for BF16?**
   FP16 has a narrow dynamic range (5 exponent bits), so small gradient values common in deep learning can underflow to exactly zero, silently killing the gradient signal for those parameters. Loss scaling multiplies the loss by a large constant (e.g., 1024, dynamically adjusted) before the backward pass, proportionally scaling up all gradients into FP16's representable range, then unscales (divides back down) before the optimizer step. BF16 retains FP32's 8 exponent bits (same dynamic range as FP32, fewer mantissa bits), so it rarely underflows/overflows and usually doesn't require loss scaling, trading a bit of precision for numerical robustness.

7. **You notice your GPU utilization is only 30% during training despite a large model and batch size. What's the likely cause and how do you fix it?**
   Low GPU utilization despite a large model/batch usually indicates a data loading/preprocessing bottleneck (CPU-bound), not a compute-bound model — the GPU is sitting idle waiting for the next batch. Fixes: increase `num_workers` in the DataLoader for parallel data loading, use `pin_memory=True`, prefetch batches, move expensive preprocessing (e.g., decoding, augmentation) to be more efficient or precomputed, and profile with tools like PyTorch Profiler or `nvidia-smi dmon` to confirm the GPU is stalling on data rather than being compute-limited.

8. **Explain gradient accumulation and describe a scenario where you'd need it.**
   Gradient accumulation runs several forward/backward passes on smaller micro-batches, summing their gradients, and only performs `optimizer.step()` after accumulating over $k$ micro-batches — this simulates training with an effective batch size $k\times$ larger than what fits in memory at once. Scenario: you want to fine-tune a large Transformer with an effective batch size of 256 for training stability, but your GPU can only fit a batch size of 32 in memory — use gradient accumulation over 8 steps to reach the effective batch size of 256 without an out-of-memory error.

9. **What's the risk of "catastrophic forgetting" during fine-tuning, and how would you mitigate it?**
   Aggressive fine-tuning (especially with a high learning rate, many epochs, or a narrow/small target dataset) can overwrite the general-purpose representations learned during pretraining, causing the model to lose broader capabilities while overfitting to the narrow target task. Mitigations: use a low learning rate for pretrained layers, freeze most of the backbone (only fine-tune a head or a small number of layers), use fewer training epochs with early stopping, or use regularization techniques that explicitly penalize deviation from the original pretrained weights (e.g., elastic weight consolidation) or parameter-efficient methods (LoRA/adapters) that inherently limit how much the original weights can shift.

10. **A colleague suggests switching from FP32 to FP16 training purely to "make training faster," without mentioning loss scaling or BF16. What would you ask them, and what risk are they overlooking?**
    Ask whether they're using automatic mixed precision with a loss scaler (e.g., `torch.cuda.amp.GradScaler`), and whether their hardware supports BF16 as an alternative. The overlooked risk: naive FP16 without loss scaling risks gradient underflow (small gradients rounding to exactly zero due to FP16's narrow dynamic range), which can silently stall learning for affected parameters or cause instability/NaNs, without necessarily throwing an obvious error — it's a subtle correctness risk, not just a performance lever.

11. **When would you choose full fine-tuning over parameter-efficient fine-tuning (e.g., LoRA) for adapting a large pretrained model?**
    Full fine-tuning makes sense when you have a large amount of target-task data, sufficient compute/memory budget, and the target task/domain is substantially different from the pretraining distribution (needing broad changes across the whole network). Parameter-efficient fine-tuning (LoRA/adapters) is preferable when compute/storage is constrained (especially when maintaining many task-specific variants of a huge base model), target data is limited (lower overfitting/catastrophic-forgetting risk since most weights stay frozen), or you need fast, cheap iteration across many downstream tasks sharing one frozen base model.

12. **Explain the InfoNCE contrastive loss and why it can be framed as a classification problem.**
    InfoNCE computes similarity between an anchor embedding and one positive (a different view of the same instance) plus $K$ negatives (other instances), then applies softmax cross-entropy over the $(K{+}1)$ candidates with the positive as the "correct class": $L=-\log\frac{\exp(\text{sim}(z,z^+)/\tau)}{\exp(\text{sim}(z,z^+)/\tau)+\sum_k\exp(\text{sim}(z,z_k^-)/\tau)}$. It's exactly a $(K{+}1)$-way classification cross-entropy where the "label" (which candidate is the true positive) is generated automatically by the data augmentation/pairing procedure, which is precisely what makes it self-supervised rather than requiring human annotation.

13. **Why does contrastive self-supervised learning need negative samples (or an equivalent anti-collapse mechanism), and what happens without them?**
    A naive objective that only pulls positive pairs together, with no term pushing anything apart, is trivially minimized by mapping every input to the same constant embedding — perfect similarity between any "positive" pair, but zero discriminative information ("representational collapse"). Negatives give the loss a reason to keep different instances' embeddings apart, which is what forces the encoder to retain instance-discriminative information; negative-free methods (e.g., BYOL/SimSiam-style approaches) instead use asymmetric mechanisms like a stop-gradient plus predictor network to make the collapsed solution unstable/unreachable without explicit negatives.

14. **At a representation level, what's the difference between what self-supervised pretraining learns and what fully supervised pretraining learns?**
    Supervised pretraining shapes representations specifically toward separating the predefined label classes used in training, which can discard information that's irrelevant to those particular labels but might matter for a different downstream task. Self-supervised pretraining (via reconstruction or contrastive pretext tasks) has no fixed label set to optimize toward, so it tends to preserve a broader range of general structure in the data (whatever is needed to solve the pretext task well), which often transfers more robustly across a wider variety of downstream tasks — at the cost of not being as directly optimized for any single one of them.

15. **Derive the knowledge distillation loss and explain the role of the temperature $T$.**
    The combined loss is $L=\alpha L_{CE}(y,p_{\text{student}}) + (1-\alpha)T^2 L_{KL}(p_{\text{teacher}}^{(T)}\|p_{\text{student}}^{(T)})$, where $p^{(T)}$ is a temperature-raised softmax $\exp(z_i/T)/\sum_j\exp(z_j/T)$. Raising $T>1$ softens the distribution, shrinking gaps between logits so the teacher's *relative* confidence across all classes (not just the argmax) becomes visible in gradient form; the $T^2$ multiplier compensates for the KL term's gradients being scaled down by $\approx 1/T^2$ due to the softened softmax, keeping the hard-label and soft-label loss terms' gradient magnitudes comparable.

16. **Why is a softened teacher distribution more informative to a student than the hard ground-truth label alone?**
    A one-hot hard label only states which class is correct; a softened teacher distribution additionally encodes *which* incorrect classes the teacher considers more or less plausible (e.g., "mostly a 7, somewhat like a 1, definitely not a 0") — this relational "dark knowledge" about class similarity structure is extra supervisory signal per example that a one-hot vector simply cannot represent, and it's especially valuable for a lower-capacity student that benefits from being told which mistakes are more reasonable than others.

17. **When does knowledge distillation fail to close the gap to the teacher's performance?**
    When the student's architecture/capacity is too limited to represent a decision boundary close to the teacher's, no amount of soft-label signal can make up the gap — distillation transfers *what* the teacher knows, not additional representational capacity the student doesn't have. It can also underperform if the teacher itself is poorly calibrated or systematically biased (the student inherits those flaws), or if the temperature/loss-weighting hyperparameters aren't tuned for the specific teacher-student capacity gap.

18. **Why doesn't naively summing per-task losses with equal weights usually work well in multi-task learning?**
    Different tasks' losses can have very different natural scales (e.g., cross-entropy vs. MSE with a large numeric range), so an equal-weight sum lets the largest-magnitude loss dominate the gradient regardless of which task actually matters most; tasks can also have gradients that point in genuinely conflicting directions ("negative transfer"), and different tasks learn at different rates, so a weighting that's reasonable at initialization can become wrong once one task saturates while another is still underfitting.

19. **What is "negative transfer" in multi-task learning, and how would you detect it?**
    Negative transfer is when training on an additional task actively *hurts* performance on another task sharing the same backbone, versus training that task alone — typically because the tasks require conflicting features or their gradients on the shared parameters point in opposing directions. Detect it by comparing each task's validation metric in the multi-task setup against a single-task baseline trained on that task alone; if the multi-task version is measurably worse for a given task, that task is experiencing negative transfer, and a technique like PCGrad (gradient surgery) or reweighting/decoupling that task onto a less-shared branch may be needed.

20. **Compare hard and soft parameter sharing in multi-task architectures.**
    Hard parameter sharing uses one shared backbone feeding multiple task-specific heads — maximal parameter efficiency and maximal opportunity for positive transfer, but also maximal risk that conflicting tasks interfere with each other's gradients on the shared trunk. Soft parameter sharing gives each task its own backbone but adds a regularization term encouraging the backbones' weights to stay similar (rather than forcing literal weight sharing) — this reduces interference risk between dissimilar tasks at the cost of losing most of the parameter-efficiency benefit, and is generally used when tasks are only loosely related.

---

## Rapid-Fire Interview Q&A

| # | Question | Answer |
|---|---|---|
| 1 | What does "backpropagation" fundamentally compute? | The gradient of the loss w.r.t. every parameter, via reverse-mode application of the chain rule, reusing intermediate results layer by layer. |
| 2 | Why does ReLU mitigate but not eliminate vanishing gradients? | Its derivative is 1 for positive inputs (no shrinkage there), but exactly 0 for negative inputs (dead units), so gradient flow depends on which units are "active." |
| 3 | What's the softmax + cross-entropy gradient w.r.t. logits? | $\hat y - y$ (predicted probability minus one-hot target) — simple and numerically well-behaved. |
| 4 | Xavier init targets which activations? He init targets which? | Xavier: tanh/sigmoid (roughly linear near 0). He: ReLU family (accounts for ~half of units being zeroed). |
| 5 | What does "batch" mean in Batch Normalization's statistics? | Mean/variance computed across the batch dimension (and spatial dims for CNNs), per channel/feature. |
| 6 | Why does LayerNorm suit Transformers better than BatchNorm? | It normalizes per example across features, independent of batch size/composition — critical for variable-length sequences and small/size-1 batches. |
| 7 | What is dropout doing at inference time? | Nothing — all units are active, no masking or scaling (inverted dropout already handles scale-matching during training). |
| 8 | What's the core difference between Adam and AdamW? | AdamW decouples weight decay from the adaptive gradient update rather than folding it into the gradient before the $1/\sqrt{\hat v}$ rescaling. |
| 9 | Why use LR warmup with Transformers? | Early Adam moment estimates are noisy/underestimated; warmup avoids large destabilizing updates before they settle. |
| 10 | What does gradient clipping preserve? | The gradient's direction (for norm-based clipping) while capping its magnitude. |
| 11 | Why do ResNets use $y=F(x)+x$ instead of just $y=F(x)$? | The identity term guarantees an undiminished gradient path regardless of how small $\partial F/\partial x$ becomes, easing optimization of very deep nets. |
| 12 | What's the receptive field of 3 stacked 3×3 convs? | 7×7 (same as one 7×7 conv, but with fewer parameters and more nonlinearity). |
| 13 | What does a 1×1 convolution NOT do? | It never mixes spatial neighbors — it only mixes across channels. |
| 14 | Max pooling vs average pooling — which is noise-sensitive? | Max pooling (dominated by a single strong/spurious activation). |
| 15 | What's the key architectural idea in Inception modules? | Parallel convolutions at multiple kernel sizes (multi-scale feature extraction), concatenated. |
| 16 | What does EfficientNet's compound scaling jointly scale? | Depth, width, and input resolution together (fixed ratio), not just one dimension. |
| 17 | Why do vanilla RNNs struggle with long sequences? | BPTT gradient is a product of many Jacobians across time — vanishes/explodes exponentially with sequence length. |
| 18 | What makes LSTM's cell-state update resistant to vanishing gradients? | It's additive ($c_t = f_t c_{t-1} + i_t\tilde c_t$), not repeated matrix multiplication — gradient w.r.t. $c_{t-1}$ is just $f_t$. |
| 19 | How many gates does a GRU have vs an LSTM? | GRU: 2 (update, reset). LSTM: 3 (forget, input, output). |
| 20 | What's the "bottleneck problem" in classic seq2seq? | Compressing the whole input into one fixed-size context vector loses information, especially for long inputs. |
| 21 | Why scale attention scores by $1/\sqrt{d_k}$? | Raw dot products grow with $\sqrt{d_k}$ in magnitude; unscaled, softmax saturates and gradients vanish. |
| 22 | What's the difference between self-attention and cross-attention? | Self-attention: Q,K,V from the same sequence. Cross-attention: Q from one sequence, K/V from another. |
| 23 | Why is a causal mask needed in decoder self-attention? | To prevent attending to future tokens during autoregressive training/generation. |
| 24 | Pre-norm vs post-norm — which is more stable for deep Transformers? | Pre-norm — keeps the residual/identity path clean and unnormalized, giving a direct gradient highway. |
| 25 | Why did Transformers replace RNNs for most sequence tasks? | Full parallelization across sequence positions + direct O(1)-hop attention between any two positions (no vanishing gradient over distance). |
| 26 | What's the time complexity of self-attention in sequence length $T$? | $O(T^2 d)$ — quadratic in sequence length. |
| 27 | What problem does RoPE solve better than absolute sinusoidal encoding? | It encodes *relative* position directly into the attention dot product, generalizing better to unseen sequence lengths. |
| 28 | Loss is NaN — first two things to check? | Learning rate too high, and FP16 training without proper loss scaling. |
| 29 | Model can't even memorize 10 training examples — what does that tell you? | There's very likely a bug (not a genuine capacity/optimization tuning issue) — always sanity-check on a tiny subset first. |
| 30 | Why freeze early layers during transfer learning? | Early layers learn generic, broadly-transferable features; freezing them reduces overfitting risk and compute cost when target data is limited. |
| 31 | Data parallel vs model parallel — the one-line distinction? | Data parallel splits the *data* across devices (full model copy each); model parallel splits the *model* across devices. |
| 32 | Why does BF16 usually skip loss scaling but FP16 doesn't? | BF16 keeps FP32's wide exponent range (less under/overflow risk); FP16's narrow exponent range makes small gradients prone to underflow to zero. |
| 33 | What is gradient accumulation simulating? | A larger effective batch size than fits in memory, by summing gradients over several micro-batches before stepping the optimizer. |
| 34 | GPU utilization is low despite a big model — likely cause? | A data loading/preprocessing (CPU/IO) bottleneck, not a compute-bound model. |
| 35 | What is catastrophic forgetting? | Fine-tuning overwriting/destroying general pretrained knowledge while adapting to a narrow target task. |
| 36 | Why does label smoothing help generalization/calibration? | It prevents the model from driving logits to push softmax probabilities to exactly 0/1, reducing overconfidence. |
| 37 | What's the practical symptom of "learning rate too low"? | Loss decreases, but extremely slowly, looking almost flat over normal training timescales. |
| 38 | Why is weight decay in Adam different from weight decay in SGD? | In Adam (not AdamW), the L2 term gets folded into the gradient and rescaled by the adaptive $1/\sqrt{\hat v}$ term, weakening decay for high-variance-gradient parameters — unlike plain SGD or AdamW's decoupled decay. |
| 39 | What does gradient checkpointing trade to save memory? | Extra compute (recomputing discarded activations during backward) in exchange for a much smaller peak activation memory footprint. |
| 40 | What is the InfoNCE loss doing, in one line? | Softmax cross-entropy over a positive plus $K$ negatives — "pick the true positive among all candidates." |
| 41 | Why can naive contrastive learning "collapse" without negatives? | The trivial constant-embedding solution perfectly minimizes a pull-only objective with zero useful information, unless negatives (or an equivalent anti-collapse mechanism) penalize it. |
| 42 | What does raising the temperature $T$ do in knowledge distillation? | Softens the teacher's output distribution, exposing more "dark knowledge" about relative class similarities beyond just the argmax. |
| 43 | Why is $T^2$ used to scale the KL term in the distillation loss? | To compensate for the softened softmax's gradients being scaled down by $\approx 1/T^2$, keeping it comparable in magnitude to the hard-label loss term. |
| 44 | In multi-task learning, what's "negative transfer"? | When training on an added task actively hurts another task's performance versus training that task alone, due to conflicting gradients on the shared backbone. |
| 45 | Hard vs. soft parameter sharing in multi-task learning — the key difference? | Hard sharing uses one literal shared backbone (max efficiency, max interference risk); soft sharing uses separate backbones regularized to stay similar (less interference, less efficiency). |
| 46 | Name one concrete symptom that multi-task loss weighting is off. | One task's loss drops to near-zero early and dominates gradients while other tasks stay underfit (or vice versa if its raw loss scale is simply numerically larger). |

