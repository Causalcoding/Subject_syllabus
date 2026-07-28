# Reinforcement Learning — Interview Prep Syllabus

Reinforcement Learning (RL) is the branch of machine learning concerned with sequential decision-making: an agent learns a behavior policy by interacting with an environment and receiving reward signals. Its interview relevance differs by role but is growing across all three:

- **Data Scientist**: RL rarely appears as a production tool in classic DS work, but interviewers use it to probe your understanding of sequential decision-making, exploration/exploitation, bandits for A/B testing and recommendation ranking, and Markov chains — all of which connect to causal inference and experimentation design.
- **Machine Learning Engineer**: You may be asked to design, train, and deploy RL agents (robotics, game AI, resource allocation, recommendation systems, ad bidding). Expect deep questions on DQN variants, policy gradients, actor-critic architectures, stability tricks, and productionization (reward shaping, simulators, offline RL).
- **AI Engineer**: RL has become *core* infrastructure knowledge because **RLHF (Reinforcement Learning from Human Feedback)** is the dominant technique for aligning large language models. PPO, reward models, KL penalties, and increasingly DPO-style offline alternatives are now standard interview topics for anyone working with LLMs, even without a classical RL background.

This syllabus goes from MDP foundations through tabular RL, deep RL, and finally the RLHF/LLM connection, with full derivations, pseudocode, pitfalls, and interview Q&A after every major section.

---

## Table of Contents

1. [Markov Decision Process Foundations](#markov-decision-process-foundations)
2. [Dynamic Programming and Tabular Methods](#dynamic-programming-and-tabular-methods)
3. [Deep Reinforcement Learning](#deep-reinforcement-learning)
4. [RL for LLMs (RLHF Connection)](#rl-for-llms-rlhf-connection)
5. [Rapid-Fire Interview Q&A](#rapid-fire-interview-qa)

---

## Markov Decision Process Foundations

### States, Actions, Rewards, Transition Probabilities, Policy, Discount Factor

An RL problem is formalized as a **Markov Decision Process (MDP)**, defined by the tuple:

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)
$$

| Symbol | Name | Meaning |
|---|---|---|
| $\mathcal{S}$ | State space | Set of all possible situations the agent can observe |
| $\mathcal{A}$ | Action space | Set of all possible actions the agent can take |
| $P(s'\mid s,a)$ | Transition probability | Probability of landing in state $s'$ after taking action $a$ in state $s$ |
| $R(s,a,s')$ or $R(s,a)$ | Reward function | Scalar feedback received for a transition |
| $\gamma \in [0,1]$ | Discount factor | Weighs future rewards relative to immediate ones |
| $\pi(a\mid s)$ | Policy | Agent's behavior — probability of choosing action $a$ in state $s$ |

**Markov property**: the future is conditionally independent of the past given the present state:

$$
P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots) = P(s_{t+1}\mid s_t, a_t)
$$

This is what makes the state a *sufficient statistic* — designing a good state representation (feature engineering, or in deep RL, the network's input) is often the hardest part of applying RL to a real problem.

**The RL objective**: find a policy $\pi$ that maximizes the expected discounted return

$$
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}, \qquad J(\pi) = \mathbb{E}_{\pi}\left[G_0\right]
$$

**Why discounting?**
- Mathematical: guarantees a finite sum for infinite-horizon problems when rewards are bounded (geometric series converges for $\gamma < 1$).
- Practical: encodes preference for immediate reward over delayed reward (like a financial discount rate), and reduces variance of the return estimate.
- $\gamma = 0$: fully myopic agent (only cares about immediate reward).
- $\gamma \to 1$: fully far-sighted agent; can cause slow/unstable learning because the return has high variance and DP/Bellman-based propagation of value takes longer to converge.

**Pitfall**: Confusing *episodic* tasks (natural terminal state, e.g., a chess game) with *continuing* tasks (no terminal state, e.g., a server load-balancer) — the latter *requires* $\gamma < 1$ or average-reward formulations to keep returns bounded.

### Value Functions: State-Value V(s), Action-Value Q(s,a)

**State-value function** — expected return starting from state $s$ and following policy $\pi$ thereafter:

$$
V^{\pi}(s) = \mathbb{E}_{\pi}\left[ G_t \mid S_t = s \right] = \mathbb{E}_{\pi}\left[\sum_{k=0}^{\infty}\gamma^k R_{t+k+1} \,\middle|\, S_t=s\right]
$$

**Action-value function (Q-function)** — expected return starting from state $s$, taking action $a$, then following $\pi$:

$$
Q^{\pi}(s,a) = \mathbb{E}_{\pi}\left[ G_t \mid S_t=s, A_t=a\right]
$$

**Relationship between V and Q**:

$$
V^{\pi}(s) = \sum_{a} \pi(a\mid s)\, Q^{\pi}(s,a) = \mathbb{E}_{a\sim\pi(\cdot|s)}\left[Q^{\pi}(s,a)\right]
$$

**Advantage function** (used heavily in policy-gradient/actor-critic methods later):

$$
A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s)
$$

Intuition: $A^\pi(s,a)$ tells you *how much better* action $a$ is than the average action the policy would take in state $s$. Positive advantage → reinforce that action; negative → suppress it.

**Pitfall**: $V^\pi$ and $Q^\pi$ are *policy-dependent* — a common interview trap is to ask for "the value of a state" without specifying which policy induces it.

### Bellman Expectation and Optimality Equations — Full Derivation

**Bellman Expectation Equation for $V^\pi$** — derived by unrolling one step of the return:

$$
V^{\pi}(s) = \mathbb{E}_\pi\Big[R_{t+1} + \gamma G_{t+1} \mid S_t = s\Big]
$$

Expand the expectation over the policy's action choice and the environment's transition:

$$
V^{\pi}(s) = \sum_{a}\pi(a\mid s) \sum_{s'} P(s'\mid s,a)\Big[ R(s,a,s') + \gamma V^{\pi}(s')\Big]
$$

**Bellman Expectation Equation for $Q^\pi$**:

$$
Q^{\pi}(s,a) = \sum_{s'} P(s'\mid s,a)\Big[R(s,a,s') + \gamma \sum_{a'}\pi(a'\mid s') Q^{\pi}(s',a')\Big]
$$

These are *linear* systems of equations (in $|\mathcal{S}|$ or $|\mathcal{S}|\times|\mathcal{A}|$ unknowns) that can, in principle, be solved exactly for small MDPs by matrix inversion:

$$
V^\pi = R^\pi + \gamma P^\pi V^\pi \;\;\Rightarrow\;\; V^\pi = (I - \gamma P^\pi)^{-1} R^\pi
$$

**Bellman Optimality Equation** — defines the *best possible* value functions, $V^* = \max_\pi V^\pi$ and $Q^* = \max_\pi Q^\pi$. Instead of averaging over the policy's action distribution, we take a **max**:

$$
V^{*}(s) = \max_{a} \sum_{s'} P(s'\mid s,a)\Big[R(s,a,s') + \gamma V^{*}(s')\Big]
$$

$$
Q^{*}(s,a) = \sum_{s'} P(s'\mid s,a)\Big[R(s,a,s') + \gamma \max_{a'} Q^{*}(s',a')\Big]
$$

**Derivation intuition**: If we know $Q^*$, the optimal policy is *greedy* with respect to it:

$$
\pi^*(s) = \arg\max_a Q^*(s,a)
$$

This is the crucial fact that underlies Q-learning: you don't need to explicitly represent a policy at all — just learn $Q^*$, and the optimal policy falls out for free by taking the argmax. The Bellman optimality equation is *nonlinear* (due to the max), so it generally cannot be solved in closed form — hence the need for iterative algorithms (value iteration, Q-learning, etc., covered next section).

**Bellman operator and contraction mapping (why iterative methods converge)**: define the Bellman backup operator $\mathcal{T}$ acting on any value function $V$:

$$
(\mathcal{T}V)(s) = \max_a \sum_{s'}P(s'|s,a)[R(s,a,s') + \gamma V(s')]
$$

$\mathcal{T}$ is a **$\gamma$-contraction** in the max-norm: $\|\mathcal{T}V_1 - \mathcal{T}V_2\|_\infty \le \gamma \|V_1 - V_2\|_\infty$. By the Banach fixed-point theorem, repeated application of $\mathcal{T}$ converges to the unique fixed point $V^*$ regardless of the initialization — this is the theoretical backbone of value iteration and Q-learning's convergence guarantees (in the tabular case).

### Exploration vs Exploitation Tradeoff, Epsilon-Greedy, UCB, Thompson Sampling

The **exploration–exploitation dilemma**: to maximize long-run reward, the agent must sometimes try actions it believes are suboptimal (explore) to gather information, rather than always choosing the currently best-known action (exploit).

**$\epsilon$-greedy**: with probability $\epsilon$ act randomly (explore), with probability $1-\epsilon$ act greedily (exploit):

$$
a_t = \begin{cases}\arg\max_a Q(s_t,a) & \text{w.p. } 1-\epsilon\\ \text{random action} & \text{w.p. } \epsilon\end{cases}
$$

Typically $\epsilon$ is annealed (decayed) over training, e.g., $\epsilon_t = \max(\epsilon_{\min}, \epsilon_0 \cdot d^{t})$.

- **Pitfall**: constant $\epsilon$ never fully converges to the optimal policy (always retains random noise); too-fast decay can cause premature convergence to a suboptimal action before the estimates are accurate.

**Upper Confidence Bound (UCB)** (classic multi-armed bandit algorithm, "optimism under uncertainty"): choose the action that maximizes an upper confidence bound on its estimated value:

$$
a_t = \arg\max_a \left[ \hat{Q}_t(a) + c\sqrt{\frac{\ln t}{N_t(a)}}\right]
$$

where $N_t(a)$ is the number of times action $a$ has been selected, and $c$ controls the exploration strength. Actions tried rarely have a wide confidence interval (large bonus term), so they get explored; as $N_t(a)$ grows the bonus shrinks toward zero and the algorithm exploits. UCB gives strong theoretical regret bounds ($O(\log T)$) in stationary stochastic bandits.

**Thompson Sampling** (Bayesian, probability matching): maintain a posterior distribution over the expected reward of each action (e.g., a Beta distribution for Bernoulli rewards), sample one value per action from its posterior each round, and pick the action with the highest sample:

$$
\theta_a \sim P(\theta_a \mid \text{history}), \qquad a_t = \arg\max_a \theta_a
$$

Intuition: actions are selected in proportion to the probability that they are actually optimal — naturally balances exploration (wide posteriors get sampled with more variance, occasionally winning) and exploitation (narrow, high-mean posteriors usually win). Empirically outperforms UCB and $\epsilon$-greedy in many bandit benchmarks (e.g., ad-serving, recommendation) and is widely used in industry A/B testing / multi-armed bandit systems.

| Method | Style | Pros | Cons |
|---|---|---|---|
| $\epsilon$-greedy | Random exploration | Trivial to implement | Explores uniformly at random — wasteful |
| UCB | Optimism under uncertainty | Strong regret guarantees, no randomness | Needs confidence bound formula; harder in non-stationary or huge action spaces |
| Thompson Sampling | Bayesian posterior sampling | Great empirical performance, naturally handles non-stationarity with sliding posteriors | Requires a (possibly approximate) posterior model |

**The Multi-Armed Bandit (MAB) setting — bandits as "RL without state"**: A multi-armed bandit is the simplest instance of sequential decision-making under uncertainty: there is a single state (or no state at all), $K$ actions ("arms"), and at each round $t$ the agent picks an arm $a_t$ and observes a stochastic reward $r_t \sim P(r\mid a_t)$ — crucially, **the chosen action does not affect the state/context for the next round** (no transition dynamics to learn or plan through). This is what makes bandits a strict special case of the full MDP framework ($|\mathcal{S}|=1$, $\gamma$ irrelevant since there's no future state to discount into) — the exploration-exploitation dilemma is the *entire* problem, uncomplicated by credit assignment across a sequence of state transitions.

**Regret** — the standard way bandit algorithms are evaluated: the cumulative gap between the reward the agent could have earned by always pulling the best arm and what it actually earned:

$$
\text{Regret}(T) = \sum_{t=1}^{T}\big(\mu^{*} - \mu_{a_t}\big) = T\mu^{*} - \sum_{t=1}^{T}\mu_{a_t}
$$

where $\mu^* = \max_a \mu_a$ is the true mean reward of the best arm. A "good" bandit algorithm has *sub-linear* regret in $T$ (regret/$T \to 0$) — UCB and Thompson Sampling both achieve the optimal $O(\log T)$ regret in the stochastic stationary-reward setting, while pure random exploration incurs *linear* regret.

**Thompson Sampling — concrete mechanics for Bernoulli rewards**: for a reward in $\{0,1\}$ (e.g., click/no-click), maintain a $\text{Beta}(\alpha_a,\beta_a)$ posterior per arm, initialized $\text{Beta}(1,1)$ (uniform prior):

$$
\theta_a \sim \text{Beta}(\alpha_a,\beta_a), \qquad a_t=\arg\max_a \theta_a, \qquad \text{after observing } r_t\in\{0,1\}:\;\; \alpha_{a_t}\leftarrow\alpha_{a_t}+r_t,\;\; \beta_{a_t}\leftarrow\beta_{a_t}+(1-r_t)
$$

Beta is the conjugate prior for a Bernoulli likelihood, so the posterior update is a simple closed-form increment — no approximate inference needed, which is why Thompson Sampling is so cheap to run in production ad-serving/recommendation bandits.

**Contextual bandits — the bridge to full RL**: a contextual bandit adds a per-round *context/state* $s_t$ (e.g., user features) that affects the reward distribution ($r_t\sim P(r\mid s_t,a_t)$) but — critically — is drawn i.i.d. each round rather than transitioning as a consequence of the chosen action. This is the natural stepping stone between plain bandits (no state) and full MDPs/RL (state transitions depend on the action, so actions have long-term consequences); contextual bandits are widely used for recommendation ranking and ad targeting precisely because they capture personalization without needing to model any long-horizon consequences of an action.

**Pitfall**: all of the above (UCB regret bounds, Thompson Sampling's clean conjugate updates) assume **stationary** reward distributions; if the true reward distribution per arm drifts over time (common in real ad-serving/recommendation systems), vanilla UCB/Thompson Sampling estimates become stale and biased — production systems typically use sliding-window estimates, discounted posteriors, or explicit change-detection to handle non-stationarity.

### Interview Questions

**Q1: What is the Markov property, and why does it matter for RL?**
A: The Markov property states that the conditional distribution of the next state depends only on the current state and action, not on the full history: $P(s_{t+1}|s_t,a_t) = P(s_{t+1}|s_t,a_t,s_{t-1},a_{t-1},\dots)$. It matters because it lets us define value functions and Bellman equations recursively in terms of the *current* state alone, making dynamic programming and TD learning tractable. If the true environment is non-Markovian (partially observed), we typically augment the state (e.g., stack frames, use an RNN/LSTM) to restore an approximate Markov property.

**Q2: Define the return $G_t$ and explain the role of the discount factor $\gamma$.**
A: $G_t = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$ is the total discounted future reward from time $t$. $\gamma \in [0,1)$ ensures the (possibly infinite) sum converges when rewards are bounded, encodes a preference for sooner rewards, and controls the effective planning horizon ($\frac{1}{1-\gamma}$ is roughly the number of future steps that matter).

**Q3: What's the difference between $V^\pi(s)$ and $Q^\pi(s,a)$?**
A: $V^\pi(s)$ is the expected return from state $s$ following policy $\pi$ for *all* subsequent actions. $Q^\pi(s,a)$ is the expected return from state $s$ taking a *specific* action $a$ first, then following $\pi$. $V^\pi(s) = \mathbb{E}_{a\sim\pi}[Q^\pi(s,a)]$.

**Q4: Derive the Bellman expectation equation for $V^\pi$.**
A: Start from $V^\pi(s)=\mathbb{E}_\pi[G_t|S_t=s]$. Split off the first reward: $G_t = R_{t+1}+\gamma G_{t+1}$. Taking expectation over the action (via $\pi$) and the resulting next state (via $P$): $V^\pi(s) = \sum_a \pi(a|s)\sum_{s'}P(s'|s,a)[R(s,a,s')+\gamma V^\pi(s')]$. This expresses value at $s$ recursively in terms of values at successor states.

**Q5: Why is the Bellman optimality equation nonlinear while the Bellman expectation equation is linear?**
A: The expectation equation averages over actions using the fixed policy distribution $\pi(a|s)$ — a linear operation in $V$. The optimality equation replaces that average with a $\max_a$, and max is a nonlinear operator, so the system of equations for $V^*$ cannot be solved by simple linear algebra and requires iterative methods.

**Q6: If you have $Q^*$, how do you get the optimal policy, and why don't you need $V^*$ explicitly?**
A: $\pi^*(s) = \arg\max_a Q^*(s,a)$. Because $Q^*$ already bakes in the value of taking each action and behaving optimally afterward, you can act optimally by simply comparing $Q^*$ values across actions — no model of the environment's transition dynamics is needed at decision time (this is why Q-learning is "model-free").

**Q7: What is the exploration-exploitation tradeoff? Give a real-world analogy.**
A: It's the tension between exploiting the best-known option now versus exploring lesser-known options to potentially discover something better later. Analogy: choosing a restaurant — going to your favorite restaurant (exploit) vs. trying a new one that might be even better or might be worse (explore).

**Q8: Compare $\epsilon$-greedy, UCB, and Thompson Sampling.**
A: $\epsilon$-greedy explores uniformly at random with fixed/decaying probability $\epsilon$ — simple but wasteful since it doesn't prioritize promising-but-uncertain actions. UCB explores by adding an uncertainty bonus that shrinks as an action is tried more, giving strong theoretical regret guarantees. Thompson Sampling maintains a Bayesian posterior per action and samples from it, naturally balancing exploration/exploitation according to the probability each action is optimal; it tends to perform best empirically and adapts naturally to non-stationarity.

**Q9: Why must $\gamma < 1$ for many continuing (non-episodic) tasks?**
A: Without episode termination, an infinite-horizon sum of non-discounted rewards ($\gamma=1$) can diverge to infinity even for bounded per-step rewards, making the value function ill-defined. $\gamma<1$ guarantees convergence of the geometric-like series (bounded rewards $\Rightarrow$ $|V|\le R_{max}/(1-\gamma)$).

**Q10: What is the advantage function and why is it useful?**
A: $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$ measures how much better (or worse) action $a$ is compared to the policy's average action in state $s$. It's central to variance reduction in policy-gradient methods because it centers the learning signal around zero — actions better than average get positive updates, worse-than-average actions get suppressed, without needing the absolute scale of returns.

**Q11: Can reward be a function of the next state, not just the current state and action? Does this break the MDP formalism?**
A: No — it's standard to write $R(s,a,s')$; the MDP formalism handles this naturally since $s'$ is drawn from a known distribution $P(s'|s,a)$, so $\mathbb{E}[R(s,a,s')]$ is well defined given $(s,a)$.

**Q12: What is "reward hacking" and how does it relate to the reward function design in MDPs?**
A: Reward hacking occurs when an agent finds an unintended way to maximize the specified reward signal without accomplishing the intended task (e.g., a cleaning robot that hides mess instead of cleaning it, because the reward only penalizes visible mess). It highlights that the reward function is an approximation of the designer's true intent, and misspecification can be exploited — a critical concern later in RLHF as well.

**Q13: In UCB, why does the exploration bonus term include $\ln t$ in the numerator and $N_t(a)$ in the denominator?**
A: $N_t(a)$ in the denominator shrinks the bonus as an action accumulates more pulls (more certainty → less need to explore it). $\ln t$ in the numerator grows (slowly) over time so that, in principle, every action is still explored infinitely often as $t\to\infty$ (ensuring the confidence bound stays valid, i.e., doesn't shrink too fast for actions that just haven't been tried recently), giving the $O(\log T)$ regret guarantee.

**Q14: What does it mean for the Bellman operator to be a contraction mapping, and why does that matter practically?**
A: It means applying the Bellman backup to any two value functions strictly reduces their max-norm distance by a factor of $\gamma < 1$. By the Banach fixed-point theorem this guarantees a *unique* fixed point ($V^*$) and that repeated application from *any* starting guess converges to it — this is exactly what justifies why value iteration and (in the tabular case) Q-learning are guaranteed to converge.

**Q15: Give an example of a real MDP and identify $\mathcal{S}, \mathcal{A}, R, P, \gamma$.**
A: Example — inventory management: $\mathcal{S}$ = current stock level, $\mathcal{A}$ = order quantity, $P(s'|s,a)$ = distribution over next stock level given random demand, $R(s,a,s')$ = revenue minus holding/ordering costs, $\gamma$ = discount reflecting the cost of capital / time value of money.

**Q16: What makes a multi-armed bandit a special case of an MDP rather than a full RL problem?**
A: A bandit has effectively one state (or a context that doesn't transition as a consequence of the action) and a single-step reward — the chosen action has no effect on any future state, so there's no credit-assignment-across-time or planning problem, only the exploration-exploitation problem in isolation. Formally it's an MDP with $|\mathcal{S}|=1$ and no transition dynamics to learn.

**Q17: Define regret in the bandit setting, and what regret growth rate do UCB and Thompson Sampling achieve?**
A: $\text{Regret}(T)=T\mu^*-\sum_{t=1}^T \mu_{a_t}$, the gap between always playing the best arm and the agent's actual cumulative expected reward. Both UCB and Thompson Sampling achieve the optimal $O(\log T)$ regret in stationary stochastic bandits — sub-linear, so average per-round regret $\to 0$ as $T\to\infty$.

**Q18: Walk through the concrete Thompson Sampling update for a Bernoulli bandit.**
A: Each arm $a$ has a $\text{Beta}(\alpha_a,\beta_a)$ posterior, starting at $\text{Beta}(1,1)$. Each round: sample $\theta_a\sim\text{Beta}(\alpha_a,\beta_a)$ for every arm, pull $a_t=\arg\max_a\theta_a$, observe reward $r_t\in\{0,1\}$, then update only the pulled arm's posterior: $\alpha_{a_t}\mathrel{+}=r_t$, $\beta_{a_t}\mathrel{+}=(1-r_t)$. Beta being conjugate to the Bernoulli likelihood is what keeps this update in closed form.

**Q19: What is a contextual bandit, and how does it differ from both a plain bandit and a full MDP?**
A: A contextual bandit adds a state/context per round that influences the reward distribution ($r_t\sim P(r|s_t,a_t)$), unlike a plain bandit which has no state. But unlike a full MDP, the context is drawn independently each round rather than transitioning as a consequence of the action taken — so there's still no long-horizon credit assignment, only per-round personalized exploration-exploitation. It's the standard formalism for recommendation ranking and ad targeting.

---

## Dynamic Programming and Tabular Methods

Dynamic Programming (DP) methods assume a **known** model of the environment ($P$ and $R$ are given) and compute exact solutions via the Bellman equations. Tabular methods (MC, TD, SARSA, Q-learning) relax this assumption and *learn from experience/samples* — they store values in a lookup table (one entry per state or state-action pair), which only scales to small/discrete state spaces (motivating deep RL in the next section).

### Policy Evaluation, Policy Iteration, Value Iteration

**Policy Evaluation** — compute $V^\pi$ for a fixed policy $\pi$ by iterating the Bellman expectation backup until convergence:

$$
V_{k+1}(s) \leftarrow \sum_a \pi(a|s)\sum_{s'}P(s'|s,a)\big[R(s,a,s') + \gamma V_k(s')\big]
$$

```
function POLICY_EVALUATION(pi, P, R, gamma, theta):
    V = zeros(|S|)
    repeat:
        delta = 0
        for each s in S:
            v = V[s]
            V[s] = sum over a of pi(a|s) * sum over s' of P(s'|s,a) * (R(s,a,s') + gamma * V[s'])
            delta = max(delta, |v - V[s]|)
    until delta < theta
    return V
```
Converges because this is repeated application of a $\gamma$-contraction operator (fixed policy version).

**Policy Improvement** — given $V^\pi$, construct a new, weakly-better policy by acting greedily w.r.t. $Q^\pi$:

$$
\pi'(s) = \arg\max_a \sum_{s'}P(s'|s,a)\big[R(s,a,s') + \gamma V^\pi(s')\big]
$$

The **Policy Improvement Theorem** guarantees $V^{\pi'}(s) \ge V^{\pi}(s)$ for all $s$, with strict improvement unless $\pi$ is already optimal.

**Policy Iteration** — alternate evaluation and improvement until the policy stops changing:

```
function POLICY_ITERATION():
    initialize pi arbitrarily
    repeat:
        V = POLICY_EVALUATION(pi)          # full evaluation to convergence
        pi_new = greedy policy w.r.t. V
        if pi_new == pi: break
        pi = pi_new
    return pi, V
```
Converges in a *finite* number of iterations for finite MDPs, since there are finitely many deterministic policies and each iteration strictly improves (or terminates).

**Value Iteration** — instead of fully evaluating each policy, combine evaluation and improvement into a single Bellman *optimality* backup, truncating evaluation to one sweep:

$$
V_{k+1}(s) \leftarrow \max_a \sum_{s'}P(s'|s,a)\big[R(s,a,s') + \gamma V_k(s')\big]
$$

```
function VALUE_ITERATION(P, R, gamma, theta):
    V = zeros(|S|)
    repeat:
        delta = 0
        for each s in S:
            v = V[s]
            V[s] = max over a of sum over s' of P(s'|s,a) * (R(s,a,s') + gamma * V[s'])
            delta = max(delta, |v - V[s]|)
    until delta < theta
    pi = greedy policy w.r.t. final V
    return pi, V
```

| | Policy Iteration | Value Iteration |
|---|---|---|
| Inner loop | Full policy evaluation to convergence, then improve | One Bellman-optimality sweep per iteration |
| Iterations to converge | Fewer outer iterations | More iterations, but each is cheaper |
| Guarantee | Converges in finite steps (finite MDP) | Converges asymptotically (contraction mapping) to $V^*$ |
| Cost per iteration | Higher (nested loop) | Lower |

**Pitfall**: Both require a *known, complete* transition model $P(s'|s,a)$ — unrealistic for most real-world problems; this is exactly the gap that model-free methods (MC, TD, Q-learning) address.

### Monte Carlo Methods: First-Visit vs Every-Visit MC

Monte Carlo (MC) methods learn value functions purely from **sampled complete episodes**, without needing a model of $P$ or $R$ — they estimate $V^\pi(s)$ as the empirical average of observed returns following visits to $s$.

**First-visit MC**: for each episode, only the *first* time state $s$ is visited contributes a sample of the return $G_t$ to the average.

**Every-visit MC**: *every* time state $s$ is visited within an episode, that occurrence's return is added to the average.

```
function FIRST_VISIT_MC_PREDICTION(policy, num_episodes):
    Returns = defaultdict(list)
    for episode in range(num_episodes):
        trajectory = generate_episode(policy)     # [(s0,a0,r1), (s1,a1,r2), ...]
        G = 0
        visited = set()
        for t in reversed(range(len(trajectory))):
            s_t, a_t, r_t1 = trajectory[t]
            G = r_t1 + gamma * G
            if s_t not in visited:                 # first-visit check
                visited.add(s_t)
                Returns[s_t].append(G)
    V = {s: mean(Returns[s]) for s in Returns}
    return V
```

- First-visit MC gives an **unbiased**, independent estimator of $V^\pi(s)$ (i.i.d. samples across episodes) and has cleaner convergence proofs (law of large numbers applies directly).
- Every-visit MC is slightly **biased** in finite samples (returns from the same episode are correlated) but is a consistent estimator (bias $\to 0$ as episodes $\to \infty$) and often has lower variance in practice since it uses more data per episode.

**Key properties of MC methods**:
- **Model-free**: only needs the ability to sample episodes.
- **No bootstrapping**: updates use the *actual* full return $G_t$, not an estimate of future value — unbiased but high variance.
- **Requires episodic tasks**: must wait until an episode terminates before any update.

**Pitfall**: MC cannot be used for continuing (non-terminating) tasks in its basic form, and has high variance for long episodes, making learning slow compared to TD methods.

### Temporal Difference Learning: TD(0), TD Error, Eligibility Traces / TD(λ)

TD learning combines ideas from MC (learn from raw experience, model-free) and DP (bootstrap — use current value estimates to update other value estimates), giving lower variance than MC at the cost of some bias.

**TD(0) update rule** for state-value prediction:

$$
V(S_t) \leftarrow V(S_t) + \alpha \Big[\underbrace{R_{t+1} + \gamma V(S_{t+1}) - V(S_t)}_{\text{TD error }\delta_t}\Big]
$$

- $\alpha$: learning rate/step size.
- $\delta_t = R_{t+1}+\gamma V(S_{t+1}) - V(S_t)$: the **TD error** — the difference between the "bootstrapped target" $R_{t+1}+\gamma V(S_{t+1})$ and the current estimate $V(S_t)$.

**Intuition**: unlike MC, which waits for the full episode return, TD(0) updates immediately after a *single* step using its own current estimate of the successor state's value as a stand-in for the rest of the return ("bootstrapping"). This is why TD can learn online, incrementally, and even in non-terminating tasks.

| | Monte Carlo | TD(0) |
|---|---|---|
| Needs episode to terminate? | Yes | No (updates every step) |
| Bootstraps? | No | Yes |
| Bias | Unbiased | Biased (uses estimate $V(S_{t+1})$) |
| Variance | High (depends on all future randomness) | Lower (only one-step randomness) |
| Works online / continuing tasks | No | Yes |

**Eligibility traces / TD($\lambda$)**: a unifying mechanism that interpolates between TD(0) ($\lambda=0$) and MC ($\lambda=1$) by giving credit to states visited in the recent past, decayed by $\lambda\gamma$ each step:

$$
E_t(s) = \gamma\lambda E_{t-1}(s) + \mathbb{1}[S_t = s]
$$

$$
V(s) \leftarrow V(s) + \alpha\, \delta_t\, E_t(s) \quad \text{for all } s
$$

**Intuition**: instead of updating only the *current* state with the TD error, the eligibility trace remembers *how recently and how often* each state was visited, and distributes credit for the current TD error backward across recently visited states — a computationally efficient way to combine multi-step returns without literally waiting to accumulate them. The **forward view** (n-step / $\lambda$-return $G_t^\lambda = (1-\lambda)\sum_{n=1}^\infty \lambda^{n-1}G_t^{(n)}$) and the **backward view** (eligibility traces) are mathematically equivalent for offline updates.

**Pitfall**: choosing $\lambda$ trades bias vs. variance — $\lambda$ close to 0 is low-variance/high-bias (like TD(0)); $\lambda$ close to 1 is high-variance/low-bias (like MC). Also, eligibility traces require care with function approximation (accumulating vs. replacing traces) to avoid divergence.

### SARSA (On-Policy) vs Q-Learning (Off-Policy)

Both are **TD control** algorithms that learn $Q(s,a)$ directly (rather than $V(s)$), enabling model-free control (no need for $P$ to derive a greedy policy from $Q$).

**SARSA** ("State-Action-Reward-State-Action") — **on-policy**: the target uses the action *actually taken* by the current behavior policy in the next state.

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha\Big[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t,A_t)\Big]
$$

Note the update requires the quintuple $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$ — hence the name — and $A_{t+1}$ is chosen by the *same* exploratory policy (e.g., $\epsilon$-greedy) that generated the trajectory.

**Q-learning** — **off-policy**: the target uses the *maximum* Q-value over next actions, regardless of what action the behavior policy would actually pick next.

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha\Big[R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t,A_t)\Big]
$$

```
# Q-learning
function Q_LEARNING(env, num_episodes, alpha, gamma, epsilon):
    Q = zeros(|S|, |A|)
    for episode in range(num_episodes):
        s = env.reset()
        done = False
        while not done:
            a = epsilon_greedy(Q, s, epsilon)
            s_next, r, done = env.step(a)
            target = r + gamma * max(Q[s_next, :]) * (0 if done else 1)
            Q[s, a] += alpha * (target - Q[s, a])
            s = s_next
    return Q

# SARSA
function SARSA(env, num_episodes, alpha, gamma, epsilon):
    Q = zeros(|S|, |A|)
    for episode in range(num_episodes):
        s = env.reset()
        a = epsilon_greedy(Q, s, epsilon)
        done = False
        while not done:
            s_next, r, done = env.step(a)
            a_next = epsilon_greedy(Q, s_next, epsilon)
            target = r + gamma * Q[s_next, a_next] * (0 if done else 1)
            Q[s, a] += alpha * (target - Q[s, a])
            s, a = s_next, a_next
    return Q
```

**On-policy vs. off-policy — the core distinction**:
- **On-policy** (SARSA): learns the value of the policy it is *currently executing* (including its exploratory actions). The learned $Q$ reflects the actual (exploration-inclusive) behavior.
- **Off-policy** (Q-learning): learns the value of the *optimal greedy* policy while following a different (exploratory) behavior policy. This is possible because the target uses $\max_{a'}$ instead of the actually-sampled next action, decoupling "the policy being learned about" (target/greedy policy) from "the policy generating data" (behavior policy).

**Classic illustrative difference — Cliff Walking**: SARSA learns a *safer* path away from a cliff edge because it accounts for the exploratory (occasionally random) actions of its own behavior policy near the cliff; Q-learning learns the *optimal* (riskier, shorter) path along the cliff edge because it assumes greedy behavior at evaluation time even though it explores during training. This is a favorite interview illustration of on- vs off-policy behavior.

| | SARSA | Q-learning |
|---|---|---|
| Policy type | On-policy | Off-policy |
| Update target | $R + \gamma Q(S', A')$, $A'$ sampled from behavior policy | $R+\gamma\max_{a'}Q(S',a')$ |
| Learns value of | The policy being executed (incl. exploration) | The optimal greedy policy |
| Risk-taking behavior | More conservative (accounts for own exploration risk) | Can appear to learn "optimal but risky" paths |
| Convergence (tabular) | Converges to $Q^\pi$ for the behavior policy, and to $Q^*$ if exploration is annealed appropriately (GLIE conditions) | Converges to $Q^*$ under standard step-size and exploration conditions |

**Pitfalls**:
- Q-learning's off-policy max operator introduces **maximization bias / overestimation bias** — since $\max_a \hat Q(s,a)$ is a biased (upward) estimator of $\max_a Q(s,a)$ when $\hat Q$ has estimation noise — this motivates Double Q-learning / Double DQN (see next section).
- Both require sufficient exploration (every state-action pair visited infinitely often, in the limit) and appropriately decaying step sizes for convergence guarantees to hold — purely greedy behavior would never explore alternatives.
- Both are tabular — they don't scale beyond small, discrete state-action spaces, motivating function approximation (Deep RL).

### Model-Based vs Model-Free RL

This distinction cuts across everything covered so far in this section, and is a favorite standalone interview question: **model-based** RL learns (or is given) a model of the environment's dynamics — an approximation of $P(s'|s,a)$ and $R(s,a,s')$ — and can use that model to *plan* (simulate rollouts, look ahead, do tree search) without needing to interact with the real environment for every update. **Model-free** RL skips modeling the dynamics entirely and learns a value function and/or policy directly from sampled experience.

| | Model-Based | Model-Free |
|---|---|---|
| Needs $P, R$? | Learns or is given a model of dynamics | Never models dynamics explicitly |
| Examples so far | DP (policy/value iteration) — model given | MC, TD, SARSA, Q-learning — model-free |
| Sample efficiency | Usually higher — can generate extra "imagined" experience from the model | Usually lower — every update needs a real environment sample |
| Asymptotic performance | Capped by model accuracy (model bias compounds over multi-step rollouts) | Not limited by model bias — learns directly from ground truth |
| Planning capability | Can look ahead / search (e.g., MCTS) before acting | Cannot plan; must have interacted with (or replayed) the relevant experience to know a value |

**Dyna-Q** — the classic tabular hybrid that learns a model *from real experience* and interleaves real Q-learning updates with extra "simulated" Q-learning updates sampled from that learned model:

```
function DYNA_Q(env, num_steps, alpha, gamma, epsilon, n_planning_steps):
    Q = zeros(|S|, |A|)
    Model = {}                                  # learned model: (s,a) -> (r, s')
    s = env.reset()
    for t in range(num_steps):
        a = epsilon_greedy(Q, s, epsilon)
        s_next, r, done = env.step(a)
        Q[s, a] += alpha * (r + gamma * max(Q[s_next, :]) - Q[s, a])   # real update
        Model[(s, a)] = (r, s_next)                                    # update learned model
        for _ in range(n_planning_steps):                              # "imagined" planning updates
            s_sim, a_sim = sample_previously_visited(Model)
            r_sim, s_next_sim = Model[(s_sim, a_sim)]
            Q[s_sim, a_sim] += alpha * (r_sim + gamma * max(Q[s_next_sim, :]) - Q[s_sim, a_sim])
        s = env.reset() if done else s_next
    return Q
```

**Beyond tabular Dyna-Q**: modern model-based deep RL learns a *neural* dynamics model (e.g., an ensemble of networks predicting $s'$ and $r$) and either (a) uses it purely for planning/rollout generation to augment a model-free learner's data (e.g., MBPO, World Models), or (b) plans directly through the model at decision time via tree search (e.g., **MCTS** in AlphaZero, or a *learned* model in MuZero, which plans through a learned latent dynamics model without even needing the true environment rules).

**Pitfall — model (compounding) error**: a learned model is never perfect; small one-step prediction errors compound multiplicatively over a multi-step simulated rollout, so long imagined rollouts can drift far from what the real environment would actually produce, quietly corrupting the value/policy updates trained on them. Mitigations include short rollout horizons, model ensembles with uncertainty estimates (down-weighting or truncating rollouts where models disagree), and periodically re-grounding in real environment data.

**Interview framing**: model-based RL trades *asymptotic performance risk* (model bias) for *sample efficiency* (can plan/simulate instead of only learning from real, often expensive/dangerous environment interaction) — this tradeoff is why robotics (where real rollouts are slow/costly/risky) leans model-based, while domains with cheap simulators (many Atari/game benchmarks) often do fine model-free.

### Interview Questions

**Q1: What is the difference between policy iteration and value iteration?**
A: Policy iteration alternates between fully evaluating the current policy (iterating the Bellman *expectation* backup to convergence) and then improving it greedily; value iteration merges evaluation and improvement into a single Bellman *optimality* backup applied once per sweep. Policy iteration needs fewer, more expensive iterations; value iteration needs more, cheaper iterations. Both converge to $\pi^*, V^*$ for finite MDPs.

**Q2: Why does greedy policy improvement guarantee a non-decreasing value function?**
A: By the Policy Improvement Theorem: if $\pi'(s) = \arg\max_a Q^\pi(s,a)$, then $Q^\pi(s,\pi'(s)) \ge V^\pi(s)$ for all $s$. Applying this inequality recursively through the Bellman equation shows $V^{\pi'}(s) \ge V^\pi(s)$ for all $s$, with equality only when $\pi$ is already optimal.

**Q3: What assumption do DP methods make that MC and TD methods don't need?**
A: DP requires a fully known model of the environment's dynamics — the transition probabilities $P(s'|s,a)$ and reward function $R(s,a,s')$. MC and TD are model-free: they learn purely from sampled trajectories/transitions without ever needing $P$ or $R$ explicitly.

**Q4: Explain first-visit vs. every-visit Monte Carlo and their bias/variance properties.**
A: First-visit MC averages the return only from the first occurrence of a state per episode, giving i.i.d. samples and an unbiased estimator. Every-visit MC averages returns from every occurrence of the state per episode; because multiple visits within an episode are correlated, this is a biased estimator in finite samples but consistent (bias vanishes asymptotically), and in practice often has lower variance since it uses more data.

**Q5: Why can't standard Monte Carlo be used for continuing (non-episodic) tasks?**
A: MC requires waiting until an episode terminates to compute the actual return $G_t$ used as the update target. In a continuing task there is no terminal state, so the return would never be fully observed — TD methods solve this by bootstrapping off of estimated future values instead of waiting for the real outcome.

**Q6: Write the TD(0) update rule and identify the TD error. Why is TD called a "bootstrapping" method?**
A: $V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1}+\gamma V(S_{t+1}) - V(S_t)]$; the bracketed term is the TD error $\delta_t$. It's bootstrapping because the update target uses the model's own current estimate $V(S_{t+1})$ of a future value, rather than the true, fully-realized future return.

**Q7: What is the bias-variance tradeoff between MC and TD(0)?**
A: MC targets ($G_t$) are unbiased estimates of $V^\pi(s)$ but have high variance (they depend on the full stochastic trajectory of rewards and transitions to episode end). TD(0) targets ($R_{t+1}+\gamma V(S_{t+1})$) have lower variance (only one step of randomness) but are biased because $V(S_{t+1})$ is itself an imperfect estimate.

**Q8: What does TD($\lambda$) interpolate between, and what do $\lambda=0$ and $\lambda=1$ correspond to?**
A: TD($\lambda$) interpolates between TD(0) ($\lambda=0$, pure one-step bootstrapping) and Monte Carlo ($\lambda=1$, full-return, no bootstrapping) by weighting n-step returns with weight $(1-\lambda)\lambda^{n-1}$, or equivalently propagating credit backward via decaying eligibility traces $E_t(s)=\gamma\lambda E_{t-1}(s)+\mathbb{1}[S_t=s]$.

**Q9: Derive/explain the SARSA update rule and state why it's called "on-policy."**
A: $Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha[R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)]$, where $A_{t+1}$ is the action actually chosen (e.g., via $\epsilon$-greedy) by the same policy generating the trajectory. It's on-policy because the TD target evaluates the behavior policy's own next action, so SARSA learns the value of the policy it's actually following, exploration included.

**Q10: Derive/explain the Q-learning update rule and state why it's called "off-policy."**
A: $Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha[R_{t+1}+\gamma \max_{a'}Q(S_{t+1},a')-Q(S_t,A_t)]$. It's off-policy because the target uses the greedy max over next actions regardless of what the (exploratory) behavior policy would actually do next — so Q-learning learns the value of the optimal policy while following a different, exploring behavior policy.

**Q11: In the Cliff Walking example, why does SARSA learn a safer path than Q-learning?**
A: SARSA's update incorporates the actual next action chosen by its exploratory ($\epsilon$-greedy) policy, so if that policy occasionally takes a random action near a cliff (falling off), SARSA "feels" that risk in its value estimates and learns to stay away from cliff-adjacent states. Q-learning's target always assumes the best (greedy) next action regardless of the exploration that generated the data, so it learns the objectively optimal — but riskier under exploration — path along the cliff edge.

**Q12: What causes maximization bias in Q-learning, and how would you address it?**
A: The $\max_a \hat Q(s,a)$ operator is applied to noisy estimates $\hat Q$; since $\mathbb{E}[\max_a \hat Q(s,a)] \ge \max_a \mathbb{E}[\hat Q(s,a)]$ (Jensen's inequality for the convex max function), Q-learning systematically overestimates action values. Double Q-learning (and later Double DQN) fixes this by decoupling action *selection* and action *evaluation* into two separate value estimators.

**Q13: What conditions are required for tabular Q-learning to provably converge to $Q^*$?**
A: (1) Every state-action pair is visited infinitely often (sufficient exploration, e.g., GLIE — Greedy in the Limit with Infinite Exploration); (2) the learning rate $\alpha_t$ satisfies the Robbins-Monro conditions, $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$ (e.g., $\alpha_t = 1/t$); (3) rewards are bounded.

**Q14: Give a practical scenario where you'd prefer SARSA over Q-learning.**
A: In safety-critical online learning scenarios where the agent must keep exploring while training (e.g., a real physical robot, autonomous driving simulator with real risk, or a live production recommender that's still exploring) — SARSA's more conservative, exploration-aware policy is preferable to Q-learning's policy which optimizes for a hypothetical fully-greedy future that isn't actually being executed during training.

**Q15: What's the difference between "on-policy" and "off-policy" in general (not just SARSA/Q-learning), and why does off-policy learning matter for scalability/data efficiency?**
A: On-policy methods can only learn from data generated by the exact policy currently being evaluated/improved — every policy update invalidates old data. Off-policy methods can learn about a target policy using data generated by any (sufficiently exploratory) behavior policy — enabling reuse of old experience (experience replay), learning from demonstrations/logs, and learning about multiple policies from one stream of data — a major reason off-policy methods like Q-learning/DQN are more data-efficient and practical at scale.

**Q16: What is the core difference between model-based and model-free RL?**
A: Model-based RL learns or is given an approximation of the environment's transition/reward dynamics ($\hat P, \hat R$) and can use it to plan or simulate additional experience; model-free RL (MC, TD, SARSA, Q-learning) learns a value function/policy directly from real sampled experience without ever representing the dynamics explicitly.

**Q17: What is Dyna-Q, and what problem does it solve?**
A: Dyna-Q is a tabular hybrid that learns a model of the environment from real transitions as it goes, then performs extra Q-learning updates using *simulated* transitions sampled from that learned model in addition to the real one — squeezing more learning signal out of each real environment interaction, improving sample efficiency versus pure model-free Q-learning.

**Q18: What is the main risk of planning with a learned (imperfect) model, and how is it mitigated?**
A: Small one-step prediction errors in a learned model compound over multi-step simulated rollouts, so long imagined trajectories can diverge from what the real environment would produce, corrupting downstream value/policy updates ("model bias" / compounding error). Mitigations include keeping simulated rollouts short, using model ensembles to estimate and down-weight uncertain predictions, and periodically re-grounding training in real environment data.

**Q19: Give one real example each of model-based and model-free RL success, and explain why the model-based one could use a model.**
A: Model-based: AlphaZero/MuZero use tree search (MCTS) through a known (AlphaZero) or learned (MuZero) dynamics model to plan moves — feasible because board games have either perfectly known or easily learnable, low-dimensional deterministic dynamics. Model-free: DQN/PPO on Atari/robotics learn directly from pixels/sensor experience without any explicit dynamics model — necessary when the true dynamics are too complex or costly to model accurately (e.g., raw high-dimensional visual environments).

---

## Deep Reinforcement Learning

Tabular methods fail once the state (or action) space is large or continuous — you cannot store a table indexed by every possible state. Deep RL replaces the table with a parameterized function approximator (a neural network) that generalizes across similar states.

### Deep Q-Networks (DQN)

**Function approximation for Q-values**: instead of a table $Q(s,a)$, learn a neural network $Q(s,a;\theta)$ (parameters $\theta$) that takes a state (e.g., raw pixels) and outputs a Q-value for every action.

**Loss function** (regression toward the Bellman-optimal target, treating it as supervised learning with a moving target):

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')\sim \mathcal{D}}\Big[\big(r + \gamma \max_{a'} Q(s',a';\theta^{-}) - Q(s,a;\theta)\big)^2\Big]
$$

where $\theta^-$ denotes the **target network** parameters (explained below) and $\mathcal{D}$ is a **replay buffer** of past transitions.

**Why naive Q-learning with neural nets is unstable** — three compounding problems:
1. **Correlated, non-stationary data**: consecutive samples from a trajectory are highly correlated (violates the i.i.d. assumption of SGD), and the data distribution itself shifts as the policy improves, causing catastrophic forgetting / oscillation.
2. **Moving target**: the regression target $r+\gamma\max_{a'}Q(s',a';\theta)$ depends on the *same* parameters $\theta$ being updated — every gradient step changes the target, creating a feedback loop that can diverge (chasing a moving target).
3. **Small changes in $\theta$ → large changes in the greedy policy → large changes in the visited-state distribution**, compounding instability further.

**Fix 1 — Experience Replay**: store transitions $(s,a,r,s')$ in a large buffer $\mathcal{D}$ and train on **random mini-batches** sampled from it, rather than sequential online transitions.
- Breaks temporal correlation (near-i.i.d. batches).
- Reuses data multiple times → better sample efficiency.
- Smooths over the changing data distribution.

**Fix 2 — Target Network**: maintain a second copy of the network, $\theta^-$, used *only* to compute the TD target, and update it slowly (e.g., copy $\theta \to \theta^-$ every $C$ steps, or via Polyak/soft update $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$).
- Freezes the regression target for a while, converting the problem into something closer to stable supervised regression, breaking the destabilizing feedback loop between updating $\theta$ and immediately changing the target that depends on $\theta$.

```
function DQN_TRAIN(env, num_steps, buffer_size, batch_size, C, gamma, epsilon):
    Q = init_network()
    Q_target = copy(Q)               # theta^-
    D = ReplayBuffer(buffer_size)
    s = env.reset()
    for t in range(num_steps):
        a = epsilon_greedy(Q, s, epsilon)
        s_next, r, done = env.step(a)
        D.push(s, a, r, s_next, done)
        s = env.reset() if done else s_next

        batch = D.sample(batch_size)
        for (s_i, a_i, r_i, s_next_i, done_i) in batch:
            target = r_i if done_i else r_i + gamma * max(Q_target(s_next_i))
            loss += (target - Q(s_i, a_i)) ** 2
        gradient_step(Q.params, loss)

        if t % C == 0:
            Q_target = copy(Q)       # hard update; or soft: Q_target = tau*Q + (1-tau)*Q_target
    return Q
```

**Pitfalls**:
- DQN is designed for **discrete** action spaces (the $\max_a$ requires enumerating actions) — doesn't directly extend to continuous control (motivates actor-critic / policy-gradient methods, e.g., DDPG, SAC).
- Overestimation bias from the $\max$ operator persists (see Double DQN below).
- Replay buffer size and update frequency are sensitive hyperparameters; too-small a buffer reintroduces correlation, too infrequent target updates slow learning, too frequent reintroduces instability.

### DQN Improvements: Double DQN, Dueling DQN, Prioritized Experience Replay

**Double DQN** — fixes maximization bias by **decoupling action selection from action evaluation**: use the online network $\theta$ to *select* the best next action, but the target network $\theta^-$ to *evaluate* it.

$$
y = r + \gamma\, Q\big(s', \arg\max_{a'} Q(s',a';\theta);\ \theta^{-}\big)
$$

Contrast with vanilla DQN's target $y = r+\gamma\max_{a'}Q(s',a';\theta^-)$, which both selects *and* evaluates using $\theta^-$ — reusing the same noisy estimates twice compounds the overestimation. By using two different (decorrelated-ish) networks for selection vs. evaluation, the positive bias is substantially reduced.

**Dueling DQN** — splits the Q-network into two streams that share a feature extractor but separately estimate the **state-value** $V(s;\theta,\beta)$ and the **advantage** $A(s,a;\theta,\alpha)$, then recombine:

$$
Q(s,a) = V(s) + \Big(A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'}A(s,a')\Big)
$$

The mean-subtraction term is a normalization trick for identifiability (otherwise $V$ and $A$ could shift arbitrarily and still sum to the same $Q$). **Intuition**: in many states, the value of the state matters far more than which specific action is taken (e.g., "about to crash" — all actions are bad regardless), so separately learning $V(s)$ lets the network generalize across actions in states where the action choice barely matters, learning faster than forcing every action's Q-value to be estimated independently from scratch.

**Prioritized Experience Replay (PER)** — instead of sampling transitions uniformly from the replay buffer, sample transitions with probability proportional to the magnitude of their TD error (i.e., how "surprising"/informative they were):

$$
P(i) = \frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}}, \qquad p_i = |\delta_i| + \epsilon
$$

Because this biases the sampling distribution, **importance-sampling (IS) weights** are used to correct the gradient update:

$$
w_i = \Big(\frac{1}{N \cdot P(i)}\Big)^{\beta}
$$

**Intuition**: transitions with large TD error indicate the network's prediction is far from the target — these are the most valuable to learn from, so training focuses computational effort where the network is most wrong (similar spirit to boosting / hard-example mining), improving sample efficiency and convergence speed, at some extra implementation complexity (typically implemented with a sum-tree for $O(\log N)$ sampling/updating).

| Improvement | Problem addressed | Mechanism |
|---|---|---|
| Double DQN | Overestimation bias from $\max$ | Decouple action selection (online net) from evaluation (target net) |
| Dueling DQN | Inefficient learning when action choice doesn't matter much | Separate $V(s)$ and $A(s,a)$ streams, recombine with mean-normalization |
| Prioritized Experience Replay | Uniform sampling wastes time on "already learned" transitions | Sample $\propto$ TD-error magnitude, correct bias with importance-sampling weights |

### Policy Gradient Methods: REINFORCE and Baselines

Instead of learning value functions and deriving a policy indirectly, **policy gradient methods directly parameterize the policy** $\pi_\theta(a|s)$ (e.g., a softmax over network logits for discrete actions, or a Gaussian with network-predicted mean/std for continuous actions) and optimize $\theta$ via gradient ascent on the expected return $J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[G(\tau)]$.

**Policy Gradient Theorem** (derivation sketch): We want $\nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}_{\tau\sim\pi_\theta}[G(\tau)]$. Using the log-derivative trick $\nabla_\theta \pi_\theta(\tau) = \pi_\theta(\tau)\nabla_\theta \log\pi_\theta(\tau)$:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\Big[G(\tau)\, \nabla_\theta \log \pi_\theta(\tau)\Big]
$$

Since $\pi_\theta(\tau) = \prod_t \pi_\theta(a_t|s_t)\, P(s_{t+1}|s_t,a_t)$ and the transition dynamics don't depend on $\theta$, the gradient of the log-trajectory-probability reduces to a sum over the policy terms only:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\Big[\sum_{t=0}^{T} G_t\, \nabla_\theta \log \pi_\theta(a_t|s_t)\Big]
$$

This is the **REINFORCE** algorithm (Williams, 1992):

```
function REINFORCE(env, policy_net, num_episodes, alpha, gamma):
    for episode in range(num_episodes):
        trajectory = generate_episode(policy_net)     # (s0,a0,r1), (s1,a1,r2), ...
        returns = compute_discounted_returns(trajectory, gamma)   # G_t for each t
        loss = 0
        for t, (s_t, a_t, _) in enumerate(trajectory):
            log_prob = policy_net.log_prob(a_t, s_t)
            loss -= log_prob * returns[t]              # negative for gradient ASCENT via descent optimizer
        gradient_step(policy_net.params, loss)
```

**Intuition**: increase the log-probability of actions that led to high return, decrease it for actions that led to low return — the update is literally "gradient ascent weighted by how good the outcome was."

**Problem — high variance**: raw Monte Carlo returns $G_t$ can vary hugely across episodes/trajectories, making the gradient estimate very noisy and training slow/unstable.

**Variance reduction via baselines**: subtract a baseline $b(s_t)$ (any function *not* depending on $a_t$) from the return without introducing bias:

$$
\nabla_\theta J(\theta) = \mathbb{E}\Big[\sum_t \big(G_t - b(s_t)\big)\nabla_\theta\log\pi_\theta(a_t|s_t)\Big]
$$

**Why this is unbiased**: $\mathbb{E}_{a_t\sim\pi_\theta}[b(s_t)\nabla_\theta \log\pi_\theta(a_t|s_t)] = b(s_t)\nabla_\theta\sum_a \pi_\theta(a|s_t) = b(s_t)\nabla_\theta(1) = 0$, since probabilities always sum to 1 regardless of $\theta$. So subtracting any state-dependent baseline changes variance but not the expected gradient.

The most common (near-optimal) baseline choice is $b(s_t) = V^\pi(s_t)$ (learned via a separate value-function estimator), which turns $G_t - V(s_t)$ into an approximation of the **advantage function** $A(s_t,a_t)$ — this is precisely the bridge to Actor-Critic methods below.

**Pitfalls**:
- REINFORCE is a Monte Carlo method: high variance, sample-inefficient, requires full episode rollouts before any update.
- Choosing a poor/no baseline can make training impractically slow.
- Sensitive to reward scaling; typically rewards are normalized in practice.

### Actor-Critic Methods: Advantage Function, A2C/A3C

**Actor-Critic** combines policy gradients (the "**actor**," $\pi_\theta$, which selects actions) with a learned value function (the "**critic**," $V_\phi$ or $Q_\phi$, which evaluates actions) to reduce variance further and enable online, per-step updates instead of waiting for full episodes.

**Advantage Actor-Critic update**: the actor is updated using the advantage estimate (from the critic) instead of the raw Monte Carlo return:

$$
\nabla_\theta J(\theta) \approx \mathbb{E}\Big[\sum_t A_\phi(s_t,a_t)\, \nabla_\theta \log \pi_\theta(a_t|s_t)\Big]
$$

A simple, practical **TD-based advantage estimator** (1-step):

$$
A_\phi(s_t,a_t) \approx \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

The critic $V_\phi$ is trained simultaneously via TD/regression toward $r_t+\gamma V_\phi(s_{t+1})$ (minimizing squared TD error), same as ordinary TD learning.

```
function ACTOR_CRITIC_STEP(s, a, r, s_next, done, actor, critic, alpha_actor, alpha_critic, gamma):
    v_s = critic(s)
    v_s_next = 0 if done else critic(s_next)
    td_error = r + gamma * v_s_next - v_s          # this IS the advantage estimate

    # critic update (minimize td_error^2)
    critic.params += alpha_critic * td_error * grad(critic(s))

    # actor update (policy gradient using advantage)
    actor.params += alpha_actor * td_error * grad(log actor.log_prob(a, s))
```

**A2C (Advantage Actor-Critic)**: synchronous version — a batch of parallel environment workers collect experience, gradients are averaged/aggregated synchronously across workers, then a single update is applied.

**A3C (Asynchronous Advantage Actor-Critic)**: multiple worker threads/processes each maintain their own copy of the environment and interact **asynchronously** with a shared global network — each worker computes gradients locally (often using n-step returns) and asynchronously applies them to the shared parameters, then pulls the updated shared parameters back down.

**Why asynchronous parallel workers help**: different workers explore different parts of the state space simultaneously (natural decorrelation of experience, similar in spirit to what experience replay achieves in DQN, but without needing a replay buffer), stabilizing training and speeding up wall-clock time; this was historically important before GPU-parallelized synchronous variants (A2C, PPO with vectorized environments) became more common.

| | REINFORCE | Actor-Critic (A2C/A3C) |
|---|---|---|
| Update frequency | End of episode (needs full return $G_t$) | Every step / n-steps (bootstrapped) |
| Variance | High | Lower (bootstrapped critic estimate) |
| Bias | Unbiased | Slightly biased (critic is imperfect) |
| Needs value function? | No (or just a simple baseline) | Yes — the critic is a core learned component |

**Pitfall**: Actor-critic introduces a second source of approximation error (the critic itself is learned and imperfect), and training two interacting, non-stationary-target networks (actor influences the data the critic sees; critic influences the actor's gradient) can be unstable without careful tuning (learning rates, entropy bonuses for exploration, gradient clipping).

### Proximal Policy Optimization (PPO): Clipped Objective

**Motivation**: naive policy gradient methods take a step in parameter space based on a *local* gradient estimate, but a large step can change the policy so much that the new data collected is no longer representative of the policy that generated the gradient — leading to a performance collapse that's hard to recover from ("falling off a cliff"). We'd like to take the *largest safe step* — improving as much as possible without changing the policy too drastically.

**Importance sampling ratio** (lets us reuse data collected under an older policy $\pi_{\theta_{old}}$ to estimate the objective for a new candidate $\pi_\theta$):

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

**Surrogate objective** (without clipping — this is the basis of vanilla policy gradient / TRPO's objective):

$$
L^{CPI}(\theta) = \mathbb{E}_t\big[r_t(\theta)\, \hat A_t\big]
$$

This objective alone is unbounded — if $r_t(\theta)$ grows very large or very small, the objective can be pushed arbitrarily, encouraging destructively large policy updates.

**PPO's clipped surrogate objective**:

$$
L^{CLIP}(\theta) = \mathbb{E}_t\Big[\min\big(r_t(\theta)\hat A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat A_t\big)\Big]
$$

where $\epsilon$ is a small hyperparameter (e.g., 0.2).

**Why clipping stabilizes training — intuition**:
- If $\hat A_t > 0$ (action was better than expected): the objective wants to increase $r_t(\theta)$ (increase probability of that action). Clipping caps the benefit of increasing $r_t(\theta)$ beyond $1+\epsilon$ — i.e., once the policy has already increased this action's probability by more than $\epsilon$, there's no further incentive from this term to push it even higher. This prevents excessively large, greedy updates.
- If $\hat A_t < 0$ (action was worse than expected): the objective wants to decrease $r_t(\theta)$. Clipping caps this at $1-\epsilon$, similarly preventing the probability from being driven down too aggressively in a single update.
- The **min** in the objective ensures the clip only ever makes the objective *more pessimistic* (a lower bound on the unclipped objective) — so clipping never gives a false incentive to move $r_t(\theta)$ *further* outside the trust region than it already is; it only removes the incentive to move further.

**Full PPO objective** (in practice, combined with a value-function loss and an entropy bonus for exploration):

$$
L^{PPO}(\theta) = \mathbb{E}_t\Big[L^{CLIP}(\theta) - c_1 (V_\phi(s_t) - V_t^{target})^2 + c_2\, \mathcal{H}[\pi_\theta](s_t)\Big]
$$

```
function PPO_UPDATE(trajectories, actor, critic, epsilon, epochs, c1, c2):
    advantages = compute_GAE(trajectories, critic)          # generalized advantage estimation
    old_log_probs = actor_old.log_prob(actions, states)     # frozen snapshot before updates
    for epoch in range(epochs):                             # multiple epochs of minibatch SGD on SAME data
        for batch in minibatches(trajectories):
            new_log_probs = actor.log_prob(batch.actions, batch.states)
            ratio = exp(new_log_probs - old_log_probs[batch])
            unclipped = ratio * batch.advantages
            clipped = clip(ratio, 1-epsilon, 1+epsilon) * batch.advantages
            policy_loss = -mean(min(unclipped, clipped))
            value_loss = mean((critic(batch.states) - batch.returns) ** 2)
            entropy_bonus = mean(actor.entropy(batch.states))
            loss = policy_loss + c1 * value_loss - c2 * entropy_bonus
            gradient_step(actor.params + critic.params, loss)
```

**Why PPO is popular / practical**: it's a first-order method (only needs standard gradient descent — no second-order/conjugate-gradient machinery like TRPO), allows multiple epochs of minibatch updates on the same batch of collected data (much more sample-efficient than one-gradient-step-per-batch REINFORCE/A2C), and is comparatively simple to implement and tune, while achieving similar or better stability/performance than TRPO in most benchmarks. This combination of simplicity + stability is exactly why **PPO became the default RL algorithm for RLHF** (see the next section).

**Pitfalls**: PPO still has hyperparameters that matter a lot (clip range $\epsilon$, number of epochs per batch, GAE $\lambda$, entropy coefficient); too many epochs on stale data can still overfit to an old advantage estimate; clipping bounds *how much* a single update can move the policy but doesn't guarantee monotonic improvement in the same rigorous theoretical sense TRPO's trust region does.

### Trust Region Policy Optimization (TRPO) vs PPO

**TRPO's approach**: explicitly constrain the policy update to stay within a **trust region** defined by KL-divergence between old and new policy, solving a constrained optimization problem at every update:

$$
\max_\theta\ \mathbb{E}_t\big[r_t(\theta)\hat A_t\big] \quad \text{subject to}\quad \mathbb{E}_t\big[D_{KL}(\pi_{\theta_{old}}(\cdot|s_t)\ \|\ \pi_\theta(\cdot|s_t))\big] \le \delta
$$

This constrained problem is solved approximately at each step using a second-order method: conjugate gradient to approximately compute the natural-gradient direction using the Fisher Information Matrix, plus a line search to ensure the KL constraint and the surrogate objective's improvement both hold.

**TRPO gives strong theoretical guarantees** (monotonic improvement bound) but is:
- Computationally expensive (Hessian-vector products, conjugate gradient, line search per update).
- Complex to implement correctly.
- Harder to combine with things like shared actor-critic parameters, recurrent networks, etc.

**PPO's key insight**: you can get *almost all* of TRPO's stability benefits with a much simpler **first-order** method by directly clipping the objective's incentive to move too far, instead of solving a constrained optimization problem exactly.

| | TRPO | PPO |
|---|---|---|
| Constraint mechanism | Explicit KL-divergence constraint (hard constraint) | Implicit, via clipped surrogate objective |
| Optimization | Second-order (conjugate gradient, Fisher-vector products, line search) | First-order (plain SGD/Adam) |
| Theoretical guarantee | Monotonic improvement bound (formal) | Empirical stability (no formal monotonic guarantee) |
| Implementation complexity | High | Low |
| Sample reuse (multiple epochs per batch) | Limited | Yes — a key practical advantage |
| Practical adoption | Rare today | Extremely widespread (control, games, RLHF) |

**Pitfall**: some PPO implementations also add a KL-penalty variant (as an alternative or supplement to clipping) — this exact "KL-controlled" flavor of PPO is what's typically used in RLHF, where an explicit KL penalty against a frozen *reference* policy is added for a different reason (staying close to the pretrained/SFT model, not just the previous iterate) — don't confuse the RLHF reference-model KL penalty with TRPO's trust-region KL constraint; they constrain against different "old" policies for different purposes.

### Offline RL (Batch RL)

**Definition**: offline RL (also called batch RL) learns a policy purely from a **fixed, previously-collected dataset** of transitions $\mathcal{D}=\{(s,a,r,s')\}$ gathered by some behavior policy (possibly unknown, possibly suboptimal, possibly a mix of policies/logged production traffic) — with **no further interaction with the environment at all during training**. This is a step beyond ordinary off-policy RL (e.g., DQN with a replay buffer): DQN is off-policy but still *online* — it keeps collecting fresh experience from the environment throughout training. Offline RL removes that entirely: the dataset is frozen upfront, which matters whenever real interaction is expensive, slow, or unsafe (healthcare, robotics, recommendation systems using only historical logs).

**Core challenge — distributional shift / extrapolation error**: the policy being learned, $\pi_\theta$, will naturally drift toward actions with high estimated $Q$-values — but for state-action pairs that are rare or absent in $\mathcal{D}$, $Q(s,a)$ is an unconstrained extrapolation with no data to correct it. Because TD bootstrapping propagates the target $r+\gamma\max_{a'}Q(s',a')$ through the network, these erroneous, typically over-optimistic out-of-distribution (OOD) estimates get bootstrapped into *other* Q-values too, compounding across training and yielding a policy that looks excellent under its own (badly miscalibrated) $Q$-function but performs poorly in the real environment. Naively running vanilla DQN/actor-critic on a static logged dataset (no offline-specific correction) is a classic failure mode for exactly this reason.

**Mitigations** (representative, not exhaustive):
- **Policy constraint**: keep $\pi_\theta$ close to the behavior policy that generated $\mathcal{D}$ (e.g., **BCQ** — Batch-Constrained Q-learning — restricts action selection to actions a generative model deems likely under the behavior policy).
- **Conservative value estimation**: penalize $Q$-values on actions not well-supported by the data instead of letting them float upward unchecked. **CQL** (Conservative Q-Learning) adds a regularizer that pushes down $Q$ on distribution-wide/OOD actions while pushing up $Q$ on the actions actually seen in $\mathcal{D}$:
$$
\mathcal{L}_{CQL}(Q) = \alpha\Big(\mathbb{E}_{s\sim\mathcal{D}}\big[\log\textstyle\sum_a \exp Q(s,a)\big] - \mathbb{E}_{(s,a)\sim\mathcal{D}}[Q(s,a)]\Big) + \mathcal{L}_{Bellman}(Q)
$$
- **Off-policy evaluation (OPE)**: before ever deploying a candidate policy, estimate its real-world performance from the logged data alone (e.g., importance sampling, doubly robust estimators) — critical in offline settings since you can't just "try it and see" in the real environment.

**Connection to RLHF/DPO**: DPO (previous chapter) is, in this exact technical sense, an **offline** method — it trains on a fixed set of human preference pairs with no further sampling/rollouts from the policy during training. The DPO pitfall noted earlier — that it's "bounded by the coverage of the fixed preference dataset" and "cannot actively explore" — is precisely the classical offline RL distributional-shift concern, applied to preference data instead of transition data. This is *why* RLHF-style alignment has bifurcated into fully-online (PPO, can explore beyond the dataset but pays the cost of a live RL loop) versus fully-offline (DPO and kin, cheap and stable but capped by dataset coverage) camps.

**Pitfall**: candidates often propose "just run Q-learning/DQN on our logged data" as if it were free — the interviewer is listening for whether you flag distributional shift/OOD extrapolation as the reason this typically fails without offline-specific corrections (conservative estimation or policy constraints).

### Multi-Agent RL (Brief)

**Setting**: multiple agents act simultaneously in a shared environment, each with (possibly) its own observations, actions, and reward — formalized as a **Markov/Stochastic Game**, a generalization of an MDP where the transition and each agent's reward depend on the **joint** action of all agents, not just one agent's own action.

**What makes MARL fundamentally harder than single-agent RL**:
- **Non-stationarity**: from any single agent's point of view, the environment appears non-stationary because the other agents' policies are simultaneously changing during learning — the Markov property w.r.t. a single agent's own state/action no longer holds cleanly since the effective "dynamics" include other learning agents.
- **Credit assignment**: in cooperative settings with a single shared/team reward, it's hard to attribute which agent's action was responsible for the outcome.
- **Scalability**: the joint action space grows exponentially with the number of agents.
- **No single "optimal policy"**: in competitive/mixed settings, the right solution concept is often a **Nash equilibrium** (no agent can unilaterally improve by deviating), not a single best policy, since "optimal" depends on what other agents do.

**Common paradigms**:
- **Independent learners**: each agent just runs single-agent RL (e.g., independent Q-learning), ignoring the others — simple, but suffers badly from non-stationarity.
- **Centralized Training, Decentralized Execution (CTDE)**: during training, a centralized critic sees the global state and all agents' actions (reducing non-stationarity for value estimation), but each agent's actor only conditions on its own local observation at execution/deployment time (e.g., **MADDPG**, **QMIX**) — a practical compromise between fully centralized and fully independent learning.
- **Self-play**: an agent trains against copies of itself (current and/or past versions), turning a multi-agent competitive problem into a curriculum of increasingly strong opponents (e.g., AlphaGo/AlphaZero, OpenAI Five, StarCraft II agents).

**Pitfall**: don't confuse cooperative MARL (shared/team reward, credit assignment is the core issue) with competitive/mixed MARL (individual rewards, equilibrium concepts and non-stationarity are the core issue) — interviewers may ask you to distinguish which challenge dominates in a given scenario.

### Interview Questions

**Q1: Why is naive Q-learning with a neural network function approximator unstable, and what two mechanisms does DQN introduce to fix this?**
A: Instability comes from (1) correlated, non-i.i.d. sequential training data and a non-stationary data distribution as the policy changes, and (2) a moving regression target that depends on the same parameters being updated, creating a destabilizing feedback loop. DQN fixes these with (1) experience replay — training on random minibatches from a large buffer of past transitions to decorrelate data and improve sample reuse, and (2) a target network — a slowly-updated copy of the network used only to compute TD targets, which freezes the target for a while and breaks the feedback loop.

**Q2: What is the difference in how vanilla DQN and Double DQN compute the TD target?**
A: Vanilla DQN uses the target network for both selecting and evaluating the best next action: $y=r+\gamma\max_{a'}Q(s',a';\theta^-)$. Double DQN decouples selection and evaluation: it uses the *online* network to pick the argmax action, but the *target* network to evaluate that action's value: $y = r+\gamma Q(s', \arg\max_{a'}Q(s',a';\theta); \theta^-)$. This reduces the systematic overestimation bias caused by reusing the same noisy max twice.

**Q3: Explain the architecture and intuition behind Dueling DQN.**
A: Dueling DQN splits the network into two heads after a shared feature extractor: one estimates $V(s)$ (a scalar), the other estimates $A(s,a)$ (one value per action), recombined as $Q(s,a)=V(s)+(A(s,a)-\frac{1}{|A|}\sum_{a'}A(s,a'))$. The intuition is that in many states the identity of the best action barely matters (all actions lead to similar outcomes), so separately learning a state's overall value lets the network generalize/learn faster instead of estimating every action's absolute Q-value independently.

**Q4: How does Prioritized Experience Replay change the sampling distribution, and how is the resulting bias corrected?**
A: Instead of sampling transitions uniformly, PER samples proportional to $|TD\ error|^\alpha$, prioritizing transitions the network currently predicts poorly (more informative for learning). Because this changes the effective training distribution away from the "true" replay distribution, importance-sampling weights $w_i = (1/(N\cdot P(i)))^\beta$ are applied to the loss to correct for the sampling bias.

**Q5: Derive the REINFORCE gradient estimator starting from $J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[G(\tau)]$.**
A: Using $\nabla_\theta \pi_\theta(\tau) = \pi_\theta(\tau)\nabla_\theta\log\pi_\theta(\tau)$ (log-derivative trick), $\nabla_\theta J(\theta) = \int \nabla_\theta\pi_\theta(\tau) G(\tau) d\tau = \mathbb{E}_{\tau}[G(\tau)\nabla_\theta\log\pi_\theta(\tau)]$. Since $\log\pi_\theta(\tau)=\sum_t\log\pi_\theta(a_t|s_t) + \text{(dynamics terms independent of }\theta\text{)}$, this becomes $\mathbb{E}[\sum_t G_t \nabla_\theta\log\pi_\theta(a_t|s_t)]$ — increase log-probability of actions in proportion to the return that followed them.

**Q6: Why does subtracting a baseline from the return not bias the policy gradient?**
A: Because $\mathbb{E}_{a\sim\pi_\theta}[b(s)\nabla_\theta\log\pi_\theta(a|s)] = b(s)\nabla_\theta\sum_a\pi_\theta(a|s) = b(s)\nabla_\theta(1) = 0$ for any baseline $b(s)$ that doesn't depend on the action — the expectation of that extra term is identically zero regardless of $\theta$, so it can only change the variance of the gradient estimator, not its expected value.

**Q7: What is the actor-critic architecture, and how does the critic reduce variance compared to REINFORCE?**
A: The actor is the policy $\pi_\theta$ being trained; the critic is a learned value function ($V_\phi$ or $Q_\phi$) that estimates expected returns. Instead of using the full, high-variance Monte Carlo return $G_t$ as the learning signal, actor-critic uses a bootstrapped estimate (e.g., the TD error $\delta_t = r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$, an estimate of the advantage), which has much lower variance because it depends on only one step of environment randomness plus the critic's (biased but lower-variance) estimate of the rest.

**Q8: What is the difference between A2C and A3C?**
A: A2C (synchronous) runs multiple environment copies in parallel, collects a batch of experience from all of them, and performs one synchronized gradient update using the averaged/aggregated batch. A3C (asynchronous) runs multiple independent workers, each interacting with its own environment and asynchronously pushing gradients to (and pulling parameters from) a shared global network without waiting for other workers — historically useful for CPU-parallel training without needing a GPU-friendly synchronous batch.

**Q9: Explain the PPO clipped surrogate objective and why it stabilizes training.**
A: PPO optimizes $L^{CLIP}(\theta)=\mathbb{E}_t[\min(r_t(\theta)\hat A_t, \text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t)]$ where $r_t(\theta)=\pi_\theta(a_t|s_t)/\pi_{\theta_{old}}(a_t|s_t)$. The clip removes the incentive for the objective to keep increasing (for positive advantage) or decreasing (for negative advantage) the probability ratio once it moves outside $[1-\epsilon,1+\epsilon]$, and the outer $\min$ ensures the clipped term only ever acts as a pessimistic lower bound — so a single update can't be pushed to drastically overhaul the policy based on a possibly-stale advantage estimate, preventing destructive large policy updates while still allowing multiple epochs of reuse on the same batch of data.

**Q10: What problem does the importance sampling ratio $r_t(\theta)$ solve, and why is it necessary in PPO?**
A: On-policy policy-gradient methods traditionally require fresh data from the exact current policy for every update. The importance-sampling ratio lets PPO reweight data collected under an older policy $\pi_{\theta_{old}}$ to still validly (approximately) estimate the objective for the current candidate $\pi_\theta$, enabling multiple gradient epochs per batch of collected rollouts — a major sample-efficiency win over vanilla REINFORCE/A2C which effectively discard the batch after one gradient step.

**Q11: How does TRPO enforce a "trust region," and what is the tradeoff versus PPO's approach?**
A: TRPO explicitly constrains the KL-divergence between the old and new policy to be below a threshold $\delta$, and solves this constrained optimization via conjugate gradient and a line search using the Fisher Information Matrix (a second-order, natural-gradient approach). This gives a formal, theoretically justified monotonic-improvement guarantee, but at high computational/implementation cost. PPO approximates the same "don't move too far" effect via a much simpler first-order clipped objective, at the cost of losing the formal guarantee — in practice PPO is nearly as stable and vastly simpler/cheaper, which is why it displaced TRPO in most applications.

**Q12: What role does the entropy bonus play in actor-critic / PPO objectives?**
A: Adding $c_2\,\mathcal{H}[\pi_\theta](s)$ (entropy of the action distribution) to the objective encourages the policy to remain stochastic/exploratory rather than collapsing prematurely to a deterministic policy, preventing premature convergence to a suboptimal, overly-confident policy and maintaining exploration throughout training.

**Q13: Why can't DQN directly handle continuous action spaces, and what family of algorithms addresses this?**
A: DQN's action-selection step requires computing $\max_{a} Q(s,a)$, which is only tractable by exhaustive enumeration over a discrete, typically small action set; over a continuous action space this max has no closed form. Continuous-control algorithms like DDPG, TD3, SAC, and policy-gradient/actor-critic methods (PPO, A2C/A3C) instead learn a parameterized policy that directly outputs (or samples) a continuous action, avoiding the need to maximize over an infinite action set at each step.

**Q14: What is Generalized Advantage Estimation (GAE), and why is it used in PPO?**
A: GAE computes the advantage estimate as an exponentially-weighted average of n-step TD advantage estimates (analogous to TD($\lambda$) but for the advantage function), controlled by a parameter $\lambda_{GAE}$: $\hat A_t^{GAE} = \sum_{l=0}^\infty (\gamma\lambda_{GAE})^l \delta_{t+l}$. It provides a tunable bias-variance tradeoff for the advantage estimate used in the policy gradient — low $\lambda_{GAE}$ is lower variance/more biased (closer to 1-step TD), high $\lambda_{GAE}$ is closer to the unbiased Monte Carlo advantage — and is standard practice in PPO implementations for more stable, lower-variance policy updates.

**Q15: If you were choosing between DQN-family methods and PPO for a new RL problem, what factors would drive your choice?**
A: Key factors: (1) Action space — discrete favors DQN-family (or PPO with a categorical policy head); continuous strongly favors PPO/actor-critic/DDPG/SAC. (2) Sample efficiency needs — off-policy methods (DQN + replay) reuse data more efficiently than on-policy PPO, valuable when environment interaction is expensive (e.g., real robots). (3) Stability/simplicity of tuning — PPO is generally considered more robust to hyperparameters "out of the box" across diverse tasks. (4) Whether you need a stochastic policy (PPO naturally supports this, useful for exploration and for downstream applications like RLHF where a distribution over actions/tokens is required).

**Q16: What is offline RL, and how does it differ from ordinary off-policy RL like DQN with a replay buffer?**
A: Offline RL learns entirely from a fixed, previously-collected dataset with zero further environment interaction during training. Ordinary off-policy RL (e.g., DQN) is off-policy in the sense that its target policy differs from the exact behavior policy, but it's still online — it continues collecting fresh experience from the environment throughout training and adding it to the replay buffer. Offline RL removes that entirely; the dataset is frozen upfront.

**Q17: What is distributional shift in offline RL, and why does it cause value overestimation?**
A: The policy being trained naturally drifts toward actions with high estimated $Q$-values, but for state-action pairs rare/absent in the fixed dataset, $Q(s,a)$ is an unconstrained extrapolation with no data to correct errors. Because Bellman bootstrapping propagates $\max_{a'}Q(s',a')$ targets through training, these erroneous out-of-distribution estimates compound, producing a policy that looks great under its own miscalibrated $Q$-function but performs poorly when actually deployed.

**Q18: Name one algorithm designed specifically for offline RL and briefly describe its fix.**
A: CQL (Conservative Q-Learning) adds a regularizer to the standard Bellman loss that pushes down $Q$-values on actions not well-supported by the dataset (via a log-sum-exp term over all actions) while pushing up $Q$ on the actions actually observed, preventing the unconstrained overestimation that causes offline value-based methods to fail. (BCQ is another valid answer — it constrains action selection to actions a generative model deems likely under the behavior policy.)

**Q19: How does DPO's "offline" nature relate to the classical offline RL distributional-shift problem?**
A: DPO trains purely on a fixed set of human preference pairs with no further policy rollouts/sampling during training — technically an offline method. Its well-known limitation (can't discover or get credit for novel completions beyond the dataset) is the preference-data analogue of classical offline RL's distributional-shift problem: the model can't be reliably evaluated or improved on preference judgments it never saw, just as a $Q$-function can't be reliably evaluated on state-action pairs absent from a static transition dataset.

**Q20: What makes multi-agent RL fundamentally harder than single-agent RL?**
A: Chiefly non-stationarity — from any one agent's perspective the environment's effective dynamics keep changing because other agents are simultaneously learning/updating their own policies, breaking the stationary-MDP assumptions that single-agent convergence guarantees rely on. Secondary challenges include credit assignment under shared rewards, exponential blow-up of the joint action space, and the fact that "optimal" often means a Nash equilibrium rather than a single best policy.

**Q21: What is Centralized Training with Decentralized Execution (CTDE), and why is it a popular MARL compromise?**
A: During training, a centralized critic has access to the global state and all agents' actions, giving it a stable (non-moving-target) view for value estimation despite other agents changing; at execution/deployment time, each agent's actor acts using only its own local observations (no need for centralized information in production, which is often unavailable or too slow). It's popular because it mitigates the non-stationarity problem during learning without requiring centralized coordination at deployment (e.g., MADDPG, QMIX).

---

## RL for LLMs (RLHF Connection)

### Framing Text Generation as an RL Problem

**RLHF (Reinforcement Learning from Human Feedback)** casts language model fine-tuning as an RL problem by mapping LLM concepts onto MDP concepts:

| RL concept | RLHF mapping |
|---|---|
| Policy $\pi_\theta$ | The language model itself — $\pi_\theta(\text{token}_t \mid \text{prompt}, \text{tokens}_{<t})$ |
| State $s_t$ | The prompt plus all tokens generated so far |
| Action $a_t$ | The next token chosen from the vocabulary |
| Episode | One full generated response (from prompt to end-of-sequence token) |
| Reward $R$ | A scalar from a learned **reward model** $r_\phi(\text{prompt}, \text{response})$, typically given only at the *end* of the episode (sparse, terminal reward) |
| Reference policy $\pi_{ref}$ | The original (SFT-tuned) LLM before RL fine-tuning — used to regularize the RL policy |

**Pipeline overview**:
1. **Supervised fine-tuning (SFT)**: pretrain/fine-tune the base LLM on high-quality demonstration data to get an initial policy $\pi_{SFT}$.
2. **Reward model training**: collect pairs of model outputs ranked by human preference (\"response A is better than response B\"), and train a reward model $r_\phi$ (often another LLM with a scalar output head) using a pairwise/Bradley-Terry loss:
$$
\mathcal{L}(\phi) = -\mathbb{E}_{(x,y_w,y_l)}\Big[\log \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\Big]
$$
where $y_w$ is the human-preferred ("winning") response and $y_l$ is the "losing" response.
3. **RL fine-tuning**: use the reward model to score generations from the policy, and update the policy with an RL algorithm (PPO) to maximize expected reward, **regularized by a KL penalty** to the reference policy:
$$
\max_\theta\ \mathbb{E}_{x\sim\mathcal{D},\, y\sim\pi_\theta(\cdot|x)}\Big[r_\phi(x,y)\Big] - \beta\, \mathbb{E}_{x}\Big[D_{KL}\big(\pi_\theta(\cdot|x)\ \|\ \pi_{ref}(\cdot|x)\big)\Big]
$$

**Why the KL penalty against a reference policy?**
- **Prevents reward hacking / degeneration**: the reward model is an imperfect, learned proxy for "true" human preference; without a constraint, the policy can drift arbitrarily far to exploit quirks/blind spots of the reward model (e.g., producing repetitive, unnatural, or gibberish text that happens to score highly), a phenomenon called **reward hacking** or **reward over-optimization**.
- **Preserves language quality/capabilities**: keeps the policy's output distribution close to the well-calibrated, fluent, broadly capable distribution learned during pretraining/SFT, avoiding catastrophic forgetting of general language ability.
- **Practical implementation**: the KL term is typically implemented as a per-token reward penalty added directly into the RL reward signal rather than as a separate loss term:
$$
r_t^{total} = r_\phi(x,y)\cdot \mathbb{1}[t=T] \;-\; \beta \log\frac{\pi_\theta(a_t|s_t)}{\pi_{ref}(a_t|s_t)}
$$
so the "reward" fed into PPO already includes a per-token KL penalty at every timestep, plus the terminal reward-model score at the end of the sequence.

**Pitfall**: too-large $\beta$ over-constrains the policy (barely any improvement over SFT); too-small $\beta$ allows reward hacking / mode collapse (degenerate outputs that game the reward model). Choosing/annealing $\beta$ (or using adaptive KL-controllers that target a specific KL budget) is a key practical tuning challenge.

### Why PPO Is Used in RLHF, and Practical Challenges

**Why PPO specifically (and not e.g. vanilla REINFORCE or Q-learning)**:
- **Stability with a learned, imperfect reward signal**: the clipped objective prevents any single batch of (noisy, reward-model-scored) rollouts from causing a catastrophically large policy update — critical when the reward signal itself is an approximation subject to exploitation.
- **Sample efficiency via minibatch reuse**: generating rollouts from an LLM (sampling full responses token-by-token) is expensive; PPO's ability to do multiple epochs of gradient updates on the same batch of generations is valuable when rollout generation dominates the compute cost.
- **Naturally supports a stochastic policy over a huge discrete action space** (the vocabulary) — the same machinery that works for discrete-action robotics/game problems transfers directly to token-level action spaces.
- Historically, PPO was already the most battle-tested, robust general-purpose policy-gradient algorithm at the time RLHF was popularized (InstructGPT/ChatGPT lineage), so it was the natural default choice.

**Practical challenges of PPO-based RLHF**:

| Challenge | Description |
|---|---|
| **Reward hacking** | The policy finds ways to score highly on the learned reward model without actually satisfying the true underlying human preference it was meant to proxy (e.g., excessive hedging, sycophancy, repetitive filler, exploiting reward-model blind spots) |
| **Mode collapse / reduced diversity** | RL optimization pressure can push the policy toward a narrow set of high-reward response patterns, reducing output diversity and creativity compared to the SFT model |
| **Reward model miscalibration / distribution shift** | The reward model is trained on a fixed preference dataset; as the policy evolves during RL, it generates outputs increasingly out-of-distribution for the reward model, whose scores become less reliable for these novel outputs |
| **High variance / sample inefficiency** | Sparse, terminal-only rewards (one scalar per full generated sequence) give a weak training signal per token; combined with the general high variance of policy-gradient methods, this requires large batch sizes/significant compute |
| **Complex, multi-model training infrastructure** | PPO-based RLHF requires simultaneously running/updating (at least) 4 models in memory — policy (actor), value/critic, reward model, and frozen reference model — a significant engineering and memory/compute burden |
| **Hyperparameter sensitivity** | KL coefficient $\beta$, PPO clip range, learning rates, batch sizes, and reward normalization/whitening all require careful tuning; small misconfigurations can cause training collapse |
| **Instability / training collapse** | RL fine-tuning of LLMs is empirically less stable/more finicky than supervised fine-tuning; runs can diverge or degrade output quality if not carefully monitored |

**Pitfall (interview-relevant)**: candidates often forget that in RLHF, PPO's "environment" has *no real transition dynamics to learn* — the state transition (appending the next generated token) is deterministic and known, so there's no model uncertainty to handle; the *only* source of learning difficulty is the reward signal (learned, imperfect, sparse) and the enormous discrete action space (vocabulary size) — this distinguishes RLHF from classic RL domains like robotics/games where environment dynamics themselves are complex/stochastic and unknown.

### Alternatives: DPO and Other Offline Preference Optimization Methods

**Motivation for alternatives**: full PPO-based RLHF is complex (multiple models, unstable training, heavy compute, many hyperparameters). A family of methods asks: *can we get the same alignment effect without running an actual RL loop at all?*

**Direct Preference Optimization (DPO)** — key insight: for the exact KL-constrained reward maximization objective used in RLHF, the *optimal* policy has a closed-form relationship to the reward function:

$$
r(x,y) = \beta \log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} + \beta\log Z(x)
$$

Substituting this relationship into the pairwise Bradley-Terry preference-modeling loss (used to train the reward model) **eliminates the need for a separate reward model entirely** — you can optimize the policy directly on human preference pairs $(x, y_w, y_l)$:

$$
\mathcal{L}_{DPO}(\theta) = -\mathbb{E}_{(x,y_w,y_l)}\left[\log \sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]
$$

**Why this avoids the "full RL loop"**:
- No reward model needs to be trained/hosted separately.
- No online sampling/rollout generation from the policy during training is required — DPO is trained like ordinary supervised learning directly on a fixed, static dataset of preference pairs (an **offline** method).
- No value/critic network, no PPO clipping mechanics, no advantage estimation — just a classification-style loss over log-probability ratios, computed with simple backpropagation.
- Only 2 models are needed in memory during training (the policy being trained and a frozen reference copy), versus 4 for PPO-RLHF.

**Intuition**: DPO directly increases the *relative* log-probability the model assigns to preferred vs. dispreferred responses (weighted by how confidently, via $\beta$, this should be enforced relative to the reference model), implicitly performing the same reward-maximizing, KL-regularized optimization that PPO performs explicitly and iteratively — but by exploiting the closed-form optimal-policy/reward relationship, it collapses two training stages (reward modeling + RL) and an iterative/online sampling procedure into a single, static, supervised-learning-style objective.

**Other notable offline/simplified alternatives**:

| Method | Key idea |
|---|---|
| **DPO** | Closed-form reward-policy relationship; single supervised-style loss on preference pairs, no reward model, no online rollouts |
| **IPO (Identity Preference Optimization)** | Addresses DPO's tendency to overfit/saturate on preference pairs by using a different (bounded) loss less prone to pushing log-ratios to extremes |
| **KTO (Kahneman-Tversky Optimization)** | Optimizes directly from binary "good/bad" labels (not necessarily paired comparisons), inspired by prospect theory's asymmetric treatment of gains/losses |
| **RLAIF** | Same RL/DPO machinery, but preference labels come from another AI model instead of (or supplementing) human annotators — addresses the cost/scalability of human labeling, not the algorithmic RL-vs-offline question itself |
| **Rejection sampling / Best-of-N fine-tuning** | Sample N responses per prompt from the current policy, keep only the reward-model-best one(s), and further SFT on those — a simple, RL-free way to distill reward-model preferences into the policy |

**Tradeoffs — PPO/RLHF vs. DPO-style offline methods**:

| Aspect | PPO-based RLHF | DPO-style offline methods |
|---|---|---|
| Needs separate reward model | Yes | No (reward is implicit in the policy's own log-ratios) |
| Needs online rollout sampling during training | Yes | No — trains directly on a static preference dataset |
| Number of models in memory | ~4 (policy, critic, reward model, reference) | ~2 (policy, reference) |
| Training stability | Harder — classic RL instabilities apply | Easier — behaves like standard supervised fine-tuning |
| Can improve using out-of-distribution self-generated samples during training | Yes (online exploration can discover novel high-reward behaviors) | No — limited to the coverage/quality of the static preference dataset |
| Compute cost | Higher (generation + multiple models) | Lower |
| Empirical quality (as of current literature) | Often still marginally ahead when reward model and RL loop are well-tuned, and best for on-policy exploration | Very competitive, and much simpler/cheaper — widely adopted for its favorable cost/quality tradeoff |

**Pitfall (interview-relevant)**: DPO does *not* eliminate the need for good preference data or a good reference model — its quality is fundamentally bounded by the coverage and quality of the fixed offline preference dataset, and (being an offline method) it cannot actively explore/generate novel completions the way an online RL loop with a live reward model can; this is the central tradeoff to articulate when asked "why would you still ever use PPO/RLHF instead of DPO?"

### Interview Questions

**Q1: Map the standard RL components (state, action, reward, policy) onto the RLHF setting for LLMs.**
A: Policy = the LLM, $\pi_\theta(\text{token}|prompt, \text{tokens so far})$. State = the prompt concatenated with tokens generated so far. Action = choosing the next token from the vocabulary. Episode = one full generated response. Reward = a scalar score from a learned reward model, typically given only at the end of the sequence (sparse/terminal reward).

**Q2: Why is a KL-divergence penalty against a reference policy included in the RLHF objective?**
A: It regularizes the RL-tuned policy to stay close to the original SFT/reference model's output distribution, preventing reward hacking (exploiting blind spots/quirks of an imperfect learned reward model) and preserving the fluency/general capability learned during pretraining and SFT, at the cost of limiting how far the policy can move from the reference behavior.

**Q3: How is the reward model typically trained in RLHF?**
A: On pairs of model outputs ranked by human annotators (preferred $y_w$ vs. dispreferred $y_l$ for the same prompt $x$), using a pairwise Bradley-Terry-style loss: $\mathcal{L}(\phi) = -\mathbb{E}[\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))]$, which trains the reward model to assign a higher scalar score to the response humans preferred.

**Q4: Why is PPO specifically favored for the RL step of RLHF rather than plain REINFORCE?**
A: PPO's clipped surrogate objective prevents any single batch of noisy, reward-model-scored rollouts from causing a destructively large policy update — important because the reward signal is itself an imperfect learned proxy. PPO also allows multiple epochs of minibatch gradient updates per batch of (expensive-to-generate) rollouts, giving much better sample efficiency than REINFORCE/A2C, which effectively discard the batch after a single gradient step.

**Q5: What is "reward hacking" in the context of RLHF, and give a concrete example.**
A: Reward hacking is when the policy learns to exploit flaws/blind spots in the learned reward model to achieve high reward scores without actually producing genuinely higher-quality (per true human preference) outputs. Example: a model learning to produce longer, more verbose responses because the reward model has a length bias, or a model becoming excessively sycophantic/agreeable because annotators (and thus the reward model) tend to rate agreeable responses higher, independent of correctness.

**Q6: What is mode collapse in the RLHF context, and what causes it?**
A: Mode collapse refers to a loss of output diversity — the RL-tuned policy converges toward a narrow set of high-reward response patterns/phrasings rather than maintaining the broader diversity of the SFT model. It's driven by the optimization pressure of RL fine-tuning consistently reinforcing whatever patterns the reward model scores highest, especially if the KL penalty is too weak to hold the policy near the more diverse reference distribution.

**Q7: Derive (at a high level) how DPO avoids training a separate reward model.**
A: The KL-constrained RL objective used in RLHF has a known closed-form solution relating the optimal policy to the reward function: $r(x,y) = \beta\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta\log Z(x)$. Substituting this expression for $r$ into the Bradley-Terry preference loss (which only ever compares reward *differences* between a preferred and dispreferred response, so the intractable partition function $Z(x)$ cancels out) yields a loss expressed purely in terms of the policy's own log-probability ratios — so you can train $\pi_\theta$ directly on preference pairs without ever materializing a separate reward model.

**Q8: What are the practical resource/engineering advantages of DPO over PPO-based RLHF?**
A: DPO requires only 2 models in memory during training (the policy and a frozen reference copy) versus roughly 4 for PPO-RLHF (policy/actor, critic/value, reward model, reference model); it requires no online rollout sampling/generation loop during training (trains like standard supervised fine-tuning on a static dataset), and avoids all PPO-specific stability machinery (clipping, advantage estimation, value-function training) — making it substantially simpler, cheaper, and more stable to train.

**Q9: What is a key limitation of DPO compared to full online RLHF with PPO?**
A: DPO is an offline method — it can only learn from the fixed, static preference dataset it's given and cannot actively generate and evaluate novel self-generated completions during training. PPO-based RLHF, being online, can explore genuinely new high-reward behaviors the current policy discovers via its own sampling and get them scored by an up-to-date reward model, potentially discovering better behaviors than what's represented in a fixed offline dataset.

**Q10: In the RLHF reward formulation, why is the per-token KL penalty typically folded directly into the reward signal rather than kept as a separate loss term?**
A: Because PPO (and policy gradient methods generally) is set up to consume a per-timestep reward signal from the environment; by defining $r_t^{total} = -\beta\log\frac{\pi_\theta(a_t|s_t)}{\pi_{ref}(a_t|s_t)}$ at every token (plus the terminal reward-model score at sequence end), the KL regularization is naturally absorbed into the standard RL machinery (returns, advantages, value function targets) without needing any special-cased loss term outside of the normal PPO objective.

**Q11: What does it mean that the "environment" in RLHF has no real transition dynamics to learn, and why does this matter?**
A: In text generation, the "next state" (prompt + generated tokens so far, with the new token appended) is a deterministic, fully known function of the current state and the chosen action (token) — there's no external environment stochasticity/uncertainty to model, unlike robotics or games. This means RLHF's difficulty comes entirely from the reward signal (learned, imperfect, sparse) and the enormous discrete action space (the vocabulary), not from any need to learn or plan through unknown environment dynamics — an important distinguishing point from classical RL domains.

**Q12: What is KTO, and how does it differ fundamentally from DPO in terms of the data it requires?**
A: KTO (Kahneman-Tversky Optimization) trains directly from binary "desirable/undesirable" labels on individual responses, rather than requiring *paired* preference comparisons (response A vs. B for the same prompt) as DPO does. This makes it applicable to a broader/easier-to-collect class of feedback data (e.g., simple thumbs up/down signals) and is motivated by prospect theory's asymmetric human sensitivity to gains vs. losses.

**Q13: Why might reward model quality degrade over the course of PPO-based RLHF training, and what is this phenomenon called?**
A: The reward model is trained once on a fixed preference dataset reflecting the *initial* (SFT) policy's output distribution. As RL fine-tuning proceeds, the policy's generations drift and become increasingly out-of-distribution relative to what the reward model was trained on, so the reward model's scores become progressively less reliable/calibrated for the policy's newer outputs — a form of distribution shift sometimes discussed as reward model staleness or over-optimization, and a core motivation for periodically re-collecting preference data or retraining the reward model during long RLHF runs.

**Q14: When would you still choose PPO/full RLHF over DPO despite its added complexity?**
A: When you need the model to be able to actively explore and be rewarded for genuinely novel behaviors beyond what's represented in a fixed offline preference dataset (online exploration advantage), when you have the infrastructure/compute to support multiple models and stable large-scale RL training, or when empirical benchmarks for your specific task/domain show PPO-tuned models achieving meaningfully higher quality than DPO-tuned ones after careful tuning — situations where the marginal quality gain justifies the substantially higher engineering and compute cost.

**Q15: How does RLAIF relate to RLHF/DPO, and what problem does it solve?**
A: RLAIF (RL from AI Feedback) uses another AI model (rather than, or in addition to, human annotators) to generate the preference labels/rankings used to train the reward model (in PPO-based RLHF) or directly as preference pairs (in DPO). It addresses the cost and scalability bottleneck of collecting large volumes of human preference annotations, not the underlying algorithmic choice between online RL (PPO) and offline preference optimization (DPO) — RLAIF can be combined with either.

---

## Rapid-Fire Interview Q&A

| # | Question | Answer |
|---|---|---|
| 1 | What are the 5 components of an MDP? | States $\mathcal{S}$, actions $\mathcal{A}$, transition probabilities $P$, reward function $R$, discount factor $\gamma$ |
| 2 | What does the Markov property assume? | Next state depends only on the current state and action, not the full history |
| 3 | Formula for discounted return $G_t$? | $G_t=\sum_{k=0}^\infty \gamma^k R_{t+k+1}$ |
| 4 | Relationship between $V^\pi$ and $Q^\pi$? | $V^\pi(s)=\mathbb{E}_{a\sim\pi}[Q^\pi(s,a)]$ |
| 5 | What is the advantage function? | $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$ |
| 6 | Is the Bellman optimality equation linear or nonlinear? Why? | Nonlinear — because of the $\max_a$ operator |
| 7 | What guarantees convergence of value iteration? | The Bellman optimality backup is a $\gamma$-contraction (Banach fixed-point theorem) |
| 8 | Policy iteration vs value iteration — which needs full policy evaluation each step? | Policy iteration |
| 9 | Model-free vs model-based RL — key difference? | Model-free learns directly from experience without knowing/learning $P,R$; model-based learns or uses a model of the environment to plan |
| 10 | MC vs TD — which bootstraps? | TD |
| 11 | MC vs TD — which has lower variance? | TD (uses only one step of real randomness) |
| 12 | What does TD($\lambda=1$) reduce to? | Monte Carlo |
| 13 | What does TD($\lambda=0$) reduce to? | TD(0) |
| 14 | SARSA is on-policy or off-policy? | On-policy |
| 15 | Q-learning is on-policy or off-policy? | Off-policy |
| 16 | Q-learning's target uses which operator over next actions? | $\max_{a'}$ |
| 17 | SARSA's target uses which action for the next state? | The action actually chosen by the current behavior policy |
| 18 | What bias does the max operator in Q-learning introduce? | Overestimation / maximization bias |
| 19 | Which algorithm fixes Q-learning's overestimation bias by decoupling selection and evaluation? | Double Q-learning / Double DQN |
| 20 | Two key stabilization tricks in DQN? | Experience replay and target networks |
| 21 | Why is experience replay needed for DQN? | Breaks correlation between consecutive samples and reuses data, stabilizing SGD-based training |
| 22 | What does Dueling DQN separately estimate? | State-value $V(s)$ and advantage $A(s,a)$ |
| 23 | What does Prioritized Experience Replay prioritize sampling? | Transitions with larger TD error magnitude |
| 24 | What correction is needed when using prioritized (non-uniform) replay sampling? | Importance-sampling weights |
| 25 | What's the log-derivative trick used for in policy gradients? | Rewriting $\nabla_\theta\pi_\theta(\tau)$ as $\pi_\theta(\tau)\nabla_\theta\log\pi_\theta(\tau)$ to derive the policy gradient estimator |
| 26 | Why can you subtract a baseline from the return without introducing bias? | Because $\mathbb{E}_a[b(s)\nabla_\theta\log\pi_\theta(a|s)]=0$ for any action-independent baseline |
| 27 | What is REINFORCE's biggest weakness? | High variance (Monte Carlo estimator) and needing full episodes before updating |
| 28 | What role does the critic play in actor-critic methods? | Provides a lower-variance, bootstrapped estimate of value/advantage to reduce policy-gradient variance |
| 29 | A2C vs A3C — synchronous or asynchronous? | A2C synchronous, A3C asynchronous |
| 30 | What does PPO clip? | The probability ratio $r_t(\theta)=\pi_\theta(a_t|s_t)/\pi_{\theta_{old}}(a_t|s_t)$ |
| 31 | What is the purpose of PPO's clipping? | Prevent overly large, destabilizing policy updates in a single step |
| 32 | TRPO uses what kind of constraint? | An explicit KL-divergence trust-region constraint |
| 33 | Is TRPO first-order or second-order optimization? | Second-order (uses Fisher Information Matrix / conjugate gradient) |
| 34 | Why is PPO preferred over TRPO in practice? | Similar stability with far simpler, cheaper first-order optimization |
| 35 | In RLHF, what plays the role of the "policy"? | The language model |
| 36 | In RLHF, what plays the role of "action"? | The next generated token |
| 37 | In RLHF, what is the reward typically based on? | The output of a learned reward model trained on human preference comparisons |
| 38 | Why is a KL penalty added in RLHF? | To keep the policy close to the reference/SFT model and prevent reward hacking |
| 39 | What loss is typically used to train the reward model? | A pairwise Bradley-Terry-style preference loss |
| 40 | Why is PPO favored for the RL stage of RLHF? | Stability (via clipping) with a noisy learned reward, plus sample-efficient minibatch reuse |
| 41 | What does DPO eliminate the need for? | A separately trained reward model and an online RL sampling loop |
| 42 | Is DPO an online or offline method? | Offline — trains on a static preference dataset |
| 43 | What is the main limitation of DPO vs. PPO-based RLHF? | Cannot explore/learn from novel self-generated completions beyond the fixed offline dataset |
| 44 | What does KTO use instead of paired preference comparisons? | Binary desirable/undesirable labels on individual responses |
| 45 | What is reward hacking? | The policy exploiting reward model flaws to score highly without truly satisfying the intended objective |
| 46 | What is mode collapse in RLHF? | Loss of output diversity as the policy converges to a narrow set of high-reward response patterns |
| 47 | Epsilon-greedy, UCB, Thompson Sampling — which is Bayesian? | Thompson Sampling |
| 48 | Which exploration strategy adds an uncertainty bonus that shrinks with visit count? | UCB |
| 49 | What is the GLIE condition? | Greedy in the Limit with Infinite Exploration — needed for tabular Q-learning/SARSA convergence guarantees |
| 50 | How many models are typically needed in memory for PPO-based RLHF vs. DPO? | ~4 (policy, critic, reward model, reference) vs. ~2 (policy, reference) |
| 51 | What is the regret of a bandit algorithm? | $\text{Regret}(T)=T\mu^*-\sum_{t=1}^T\mu_{a_t}$, the gap vs. always playing the best arm |
| 52 | What regret rate do UCB/Thompson Sampling achieve in stationary bandits? | $O(\log T)$ — sub-linear |
| 53 | What's the difference between a plain bandit and a contextual bandit? | Contextual bandits add a per-round state/context that affects the reward, but it doesn't transition as a result of the action |
| 54 | Model-based vs model-free — which learns/uses $P,R$ explicitly? | Model-based |
| 55 | What does Dyna-Q add to plain Q-learning? | Extra Q-updates from simulated transitions sampled from a learned model |
| 56 | What causes value overestimation in offline RL? | Bootstrapping through out-of-distribution (OOD) actions with no data to correct the extrapolation |
| 57 | What does CQL do to fix offline RL overestimation? | Penalizes/regularizes $Q$-values on OOD actions instead of letting them float upward unchecked |
| 58 | Why is DPO considered an offline method? | It trains on a fixed preference dataset with no further policy rollouts/sampling during training |
| 59 | What's the main source of difficulty added by multi-agent RL? | Non-stationarity — other agents' policies keep changing during learning |
| 60 | What does CTDE stand for in MARL? | Centralized Training, Decentralized Execution |
