# Probability and Statistics for Data Scientist, ML Engineer, and AI Engineer Interviews

Probability and statistics are the mathematical backbone of every data-driven discipline. **Data Scientists** are tested on this material more heavily than any other topic — expect deep-dive questions on hypothesis testing, A/B testing, estimator theory, and Bayesian reasoning, often as full case-study interviews. **Machine Learning Engineers** need this foundation to understand loss functions (most are negative log-likelihoods), regularization (MAP estimation), model evaluation under uncertainty, calibration, and why algorithms like Naive Bayes, Gaussian Mixture Models, and bootstrapped ensembles work. **AI Engineers** rely on it for understanding sampling in generative models (temperature, top-p, diffusion noise schedules), evaluating LLM outputs statistically (confidence in benchmark deltas, A/B testing prompts), and reasoning about uncertainty/hallucination rates. Every role should expect at least one probability brain-teaser and one statistics scenario question in a typical interview loop.

## Table of Contents

1. [Probability Foundations](#probability-foundations)
2. [Random Variables and Distributions](#random-variables-and-distributions)
3. [Statistical Inference](#statistical-inference)
4. [Bayesian Statistics](#bayesian-statistics)
5. [Sampling and Experimental Design](#sampling-and-experimental-design)
6. [Rapid-Fire Interview Q&A](#rapid-fire-interview-qa)

---

## Probability Foundations

*Relevance: Data Scientist (core, tested constantly), ML Engineer (moderate — needed for probabilistic models), AI Engineer (moderate — needed for reasoning about model outputs/uncertainty).*

### Sample Space, Events, and Axioms of Probability

**Definitions**

- **Sample space (Ω):** the set of all possible outcomes of a random experiment. E.g., rolling a die: Ω = {1,2,3,4,5,6}.
- **Event (A):** any subset of the sample space, A ⊆ Ω. E.g., "rolling an even number" = {2,4,6}.
- **Probability measure P:** a function mapping events to real numbers satisfying the **Kolmogorov axioms**:
  1. **Non-negativity:** P(A) ≥ 0 for all events A.
  2. **Normalization:** P(Ω) = 1.
  3. **Countable additivity:** for mutually exclusive events A₁, A₂, …, P(⋃ᵢ Aᵢ) = Σᵢ P(Aᵢ).

**Derived properties** (all provable from the axioms):

| Property | Formula |
|---|---|
| Complement rule | P(Aᶜ) = 1 − P(A) |
| Empty set | P(∅) = 0 |
| Monotonicity | If A ⊆ B, then P(A) ≤ P(B) |
| Inclusion-exclusion (2 events) | P(A ∪ B) = P(A) + P(B) − P(A ∩ B) |
| Inclusion-exclusion (3 events) | P(A∪B∪C) = P(A)+P(B)+P(C) − P(A∩B) − P(A∩C) − P(B∩C) + P(A∩B∩C) |
| Union bound (Boole's inequality) | P(⋃ᵢAᵢ) ≤ Σᵢ P(Aᵢ) |

**Intuition:** think of probability as *mass* spread over the sample space that must sum to exactly 1. Events are regions carved out of that space; probability just measures how much mass each region holds.

**Worked example:** In a deck of 52 cards, let A = "draw a heart" (13 cards), B = "draw a face card" (12 cards: J,Q,K × 4 suits). A ∩ B = "heart face card" = 3 cards. So P(A∪B) = 13/52 + 12/52 − 3/52 = 22/52 ≈ 0.423.

**Common pitfalls:**
- Confusing P(A ∪ B) with P(A) + P(B) when events are not mutually exclusive (must subtract the overlap).
- Treating a probability model's axioms as "just formulas" instead of understanding they encode consistency requirements — many brain teasers exploit violations of these axioms in naive reasoning.

### Conditional Probability and Independence

**Conditional probability:**
$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

**Intuition:** restrict the sample space to B, then ask what fraction of that restricted space is also in A.

**Independence:** A and B are independent iff
$$P(A \cap B) = P(A)\,P(B) \iff P(A\mid B) = P(A)$$

**Conditional independence:** A and B are conditionally independent given C if P(A ∩ B | C) = P(A|C)P(B|C). This does **not** imply (and is not implied by) marginal independence — a classic interview trap.

**Chain rule (multiplication rule):**
$$P(A_1 \cap A_2 \cap \dots \cap A_n) = P(A_1) P(A_2\mid A_1) P(A_3 \mid A_1,A_2)\cdots P(A_n \mid A_1,\dots,A_{n-1})$$

**Worked example:** Two fair dice are rolled. Let A = "sum is 8", B = "first die is 5". P(B) = 1/6. A∩B = "first die 5, second die 3" = 1/36. P(A|B) = (1/36)/(1/6) = 1/6. Compare to marginal P(A) = 5/36. Since 1/6 ≠ 5/36, A and B are **not** independent.

**Common pitfalls:**
- Mixing up P(A|B) with P(B|A) — the single most common statistics interview trap (see Bayes' theorem below).
- Assuming pairwise independence implies mutual independence for 3+ events (false in general — need the joint factorization to hold for every subset).
- Forgetting independence and conditional independence are distinct concepts (e.g., two symptoms may be independent overall but become dependent once you condition on a disease, or vice versa — "explaining away").

### Bayes' Theorem

**Derivation:** From the definition of conditional probability, P(A∩B) = P(A|B)P(B) = P(B|A)P(A). Rearranging:
$$P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}$$

Expanding P(B) via the law of total probability gives the full form:
$$P(A \mid B) = \frac{P(B\mid A) P(A)}{P(B\mid A)P(A) + P(B\mid A^c)P(A^c)}$$

**Terminology:** P(A) = prior, P(B|A) = likelihood, P(A|B) = posterior, P(B) = evidence/marginal likelihood.

**Worked example 1 — Medical testing (the classic):** A disease affects 1% of a population. A test has 95% sensitivity (true positive rate) and 90% specificity (true negative rate, so 10% false positive rate). Given a positive test, what's P(disease | positive)?

- P(D) = 0.01, P(¬D) = 0.99
- P(+|D) = 0.95, P(+|¬D) = 0.10
- P(+) = 0.95×0.01 + 0.10×0.99 = 0.0095 + 0.099 = 0.1085
- P(D|+) = 0.0095 / 0.1085 ≈ **0.0876 (≈ 8.8%)**

**Insight/pitfall:** Despite a "95% accurate" test, a positive result only means ~9% chance of disease because the disease is rare (base-rate fallacy). Interviewers love this because most candidates instinctively answer "~95%" or "~90%."

**Worked example 2 — Spam filter (Naive Bayes flavor):** P(spam) = 0.4. The word "free" appears in 60% of spam emails and 5% of non-spam emails. Given an email contains "free," what's P(spam | "free")?
- P("free") = 0.6×0.4 + 0.05×0.6 = 0.24 + 0.03 = 0.27
- P(spam|"free") = 0.24/0.27 ≈ **0.889**

This is exactly the mechanism behind **Naive Bayes classifiers**: multiply likelihood ratios across features (assuming conditional independence given the class) and combine with the prior.

**Odds form (useful in practice):** 
$$\frac{P(A\mid B)}{P(A^c \mid B)} = \frac{P(B\mid A)}{P(B\mid A^c)} \times \frac{P(A)}{P(A^c)}$$
i.e., posterior odds = likelihood ratio × prior odds. This is how you chain evidence sequentially (each new independent piece of evidence multiplies the odds).

**Common pitfalls:**
- **Base-rate fallacy** — ignoring the prior when it's extreme (rare disease, rare fraud).
- **Prosecutor's fallacy** — equating P(evidence|innocent) with P(innocent|evidence).
- Forgetting to renormalize by the full marginal P(B), not just P(B|A)P(A).

### Law of Total Probability

**Statement:** If B₁, …, Bₙ partition the sample space (mutually exclusive, exhaustive), then for any event A:
$$P(A) = \sum_{i=1}^n P(A \mid B_i) P(B_i)$$

**Intuition:** decompose a complicated event into simpler conditional pieces weighted by how likely each condition is — "average over all the ways it could happen."

**Worked example:** Three factories supply light bulbs: Factory 1 (50% of stock, 2% defective), Factory 2 (30% of stock, 5% defective), Factory 3 (20% of stock, 10% defective). What fraction of all bulbs are defective?
$$P(\text{defective}) = 0.5(0.02) + 0.3(0.05) + 0.2(0.10) = 0.01+0.015+0.02 = 0.045 = 4.5\%$$
This is also the denominator computation inside every Bayes' theorem problem.

### Combinatorics Basics

| Concept | Formula | When to use |
|---|---|---|
| Permutations of n distinct items | n! | Arrange all items in order |
| Permutations of n items taken r at a time | P(n,r) = n!/(n−r)! | Order matters, subset of size r |
| Combinations ("n choose r") | C(n,r) = n!/(r!(n−r)!) | Order doesn't matter |
| Permutations with repetition | n!/(n₁!n₂!⋯nₖ!) | Multiset arrangements (e.g., letters in "MISSISSIPPI") |
| Combinations with replacement | C(n+r−1, r) | Choosing r items from n types, repeats allowed, order doesn't matter |
| Binomial theorem link | Σᵣ C(n,r) = 2ⁿ | Total subsets of an n-set |

**Worked example (birthday problem):** Probability that among 23 people, at least two share a birthday (365 days, ignore leap years):
$$P(\text{no shared}) = \prod_{i=0}^{22}\frac{365-i}{365} \approx 0.4927 \implies P(\text{shared}) \approx 0.5073$$
Famous because the answer (>50% with just 23 people) is wildly counter-intuitive — a staple probability brain teaser.

**Worked example (poker hand):** P(exactly one pair in 5-card hand) = [C(13,1)·C(4,2)·C(12,3)·4³] / C(52,5) ≈ 0.4226 — illustrates multiplying independent combinatorial choices (rank of pair, suits of pair, ranks of other 3 cards, suits of those cards) then dividing by the total sample space size.

**Common pitfalls:**
- Double-counting when order doesn't matter but you used a permutation formula (or vice-versa).
- Forgetting to divide by symmetry factors (identical items, e.g., anagram counting).
- In "at least one" problems, forgetting the complement trick: P(at least one) = 1 − P(none) — almost always easier.

### Interview Questions

**Q1. State the three axioms of probability and derive P(Aᶜ) = 1 − P(A) from them.**
A: Axioms: P(A)≥0, P(Ω)=1, countable additivity for disjoint events. Since A and Aᶜ are disjoint and A∪Aᶜ=Ω, additivity gives P(A)+P(Aᶜ)=P(Ω)=1, so P(Aᶜ)=1−P(A).

**Q2. Two events A and B have P(A)=0.3, P(B)=0.4, P(A∩B)=0.1. Are A and B independent? Find P(A∪B) and P(A|B).**
A: Independent would require P(A∩B)=P(A)P(B)=0.12≠0.1, so **not independent**. P(A∪B)=0.3+0.4−0.1=0.6. P(A|B)=0.1/0.4=0.25.

**Q3. Explain the difference between mutually exclusive and independent events.**
A: Mutually exclusive: P(A∩B)=0 (can't both happen). Independent: P(A∩B)=P(A)P(B) (occurrence of one doesn't change probability of the other). If both P(A),P(B)>0, mutually exclusive events cannot be independent (since P(A∩B)=0≠P(A)P(B)>0).

**Q4. Derive Bayes' theorem from first principles.**
A: P(A∩B) can be written two ways: P(A|B)P(B) = P(B|A)P(A). Equate and solve: P(A|B) = P(B|A)P(A)/P(B).

**Q5. A rare disease affects 1 in 1,000 people. A test is 99% sensitive and 99% specific. If a random person tests positive, what is the probability they actually have the disease?**
A: P(D)=0.001. P(+|D)=0.99, P(+|¬D)=0.01. P(+)=0.99(0.001)+0.01(0.999)=0.00099+0.00999=0.01098. P(D|+)=0.00099/0.01098≈**9.02%**. Despite 99% accuracy, most positives are false positives because the disease is rare.

**Q6. You have 3 boxes: Box A has 2 gold, 3 silver coins; Box B has 4 gold, 1 silver; Box C has 1 gold, 4 silver. You pick a box at random, then a coin at random, and get gold. What's the probability it came from Box B?**
A: P(gold) = (1/3)(2/5)+(1/3)(4/5)+(1/3)(1/5) = (1/3)(7/5)=7/15. P(B|gold)=P(gold|B)P(B)/P(gold) = [(4/5)(1/3)]/(7/15) = (4/15)/(7/15)=**4/7≈0.571**.

**Q7. Monty Hall problem: 3 doors, one has a car. You pick door 1. Host opens door 3 (goat). Should you switch to door 2?**
A: Yes — switching gives 2/3 probability, staying gives 1/3. Initially each door has 1/3. If you switch, you win whenever your original pick was wrong (2/3 of the time), because the host's action (always revealing a goat) is informative and non-random. Formal Bayes: P(car behind 2 | host opens 3, you picked 1) = P(open 3|car@2)P(car@2)/P(open3) = (1×1/3)/(1/2) = 2/3.

**Q8. What is the birthday paradox and why does it surprise people?**
A: With 23 people, P(at least two share a birthday) ≈ 50.7%. It surprises people because they intuitively compare 23 to 365 (small ratio) rather than realizing there are C(23,2)=253 *pairs* to check, each with a small chance of matching, and these compound quickly. Complement method: multiply P(no match) sequentially: (365/365)(364/365)…(343/365).

**Q9. If P(A|B) = P(A), what does that tell you about B's effect on A? Does this imply P(B|A) = P(B)?**
A: It means A and B are independent (B gives no information about A). By symmetry of the independence definition (P(A∩B)=P(A)P(B)), yes, this also implies P(B|A)=P(B) — independence is symmetric.

**Q10. Explain "explaining away" (a case where conditional dependence arises despite marginal independence).**
A: Two independent causes (e.g., battery dead, alternator dead) both explain the same effect (car won't start). Marginally, cause1 ⟂ cause2. But condition on the effect happening: learning that cause1 is true makes cause2 less likely (since cause1 already explains the effect) — so they become dependent given the effect. This is common in Bayesian networks ("V-structure" / collider).

**Q11 (tricky). A family has two children. Given at least one is a girl, what's the probability both are girls? Now suppose instead you're told "the older child is a girl" — does the answer change?**
A: Case 1 (at least one girl): sample space {BB,BG,GB,GG} each 1/4, condition removes BB → {BG,GB,GG}, P(both girls)=1/3. Case 2 (older is girl): sample space restricted to {GB,GG}, P(both girls)=1/2. The two conditioning statements are different events, producing different answers — a classic demonstration that how information is obtained matters, not just what is known.

**Q12. Derive the inclusion-exclusion formula for P(A∪B∪C).**
A: Start from P(A∪B∪C) = P(A)+P(B∪C) − P(A∩(B∪C)), expand P(B∪C)=P(B)+P(C)−P(B∩C), and P(A∩(B∪C)) = P(A∩B)+P(A∩C)−P(A∩B∩C) (distributing intersection over union). Combine terms to get P(A)+P(B)+P(C)−P(A∩B)−P(A∩C)−P(B∩C)+P(A∩B∩C).

**Q13 (brain-teaser). You roll a fair die repeatedly. What is the expected number of rolls until you see a 6?**
A: This is a Geometric(p=1/6) waiting time; E[rolls] = 1/p = **6**.

**Q14 (scenario). In an A/B test funnel, click-through and conversion are correlated features you condition on sequentially. How would you use the chain rule to decompose P(purchase) into a funnel model?**
A: P(purchase) = P(view)·P(click|view)·P(add-to-cart|click)·P(purchase|add-to-cart). Each stage's conditional probability is measurable from data and multiplying gives the overall conversion rate — this is exactly the multiplication rule applied to a funnel.

---

## Random Variables and Distributions

*Relevance: Data Scientist (core), ML Engineer (core — likelihoods, loss functions, generative models), AI Engineer (core — sampling, temperature, token probabilities).*

### Discrete Distributions

| Distribution | PMF | Mean | Variance | Use case |
|---|---|---|---|---|
| Bernoulli(p) | P(X=1)=p, P(X=0)=1−p | p | p(1−p) | Single binary trial (click/no-click) |
| Binomial(n,p) | C(n,k)pᵏ(1−p)ⁿ⁻ᵏ | np | np(1−p) | Count of successes in n independent trials |
| Geometric(p) | (1−p)ᵏ⁻¹p, k=1,2,… | 1/p | (1−p)/p² | Number of trials until first success |
| Poisson(λ) | e⁻λλᵏ/k! | λ | λ | Count of rare events in fixed interval/area |
| Hypergeometric(N,K,n) | C(K,k)C(N−K,n−k)/C(N,n) | nK/N | n(K/N)(1−K/N)(N−n)/(N−1) | Sampling without replacement |

**Bernoulli/Binomial:** Binomial is the sum of n i.i.d. Bernoulli(p) trials. Use case: number of users who convert out of n shown an ad.

**Worked example (Binomial):** n=10 emails sent, each opened independently with p=0.3. P(exactly 3 opened) = C(10,3)(0.3)³(0.7)⁷ = 120 × 0.027 × 0.0823543 ≈ **0.2668**.

**Geometric — memorylessness:** P(X > m+n | X > m) = P(X > n). The distribution "forgets" how many failures already occurred. Common pitfall: confusing the "number of trials" version (support starts at 1) with the "number of failures before success" version (support starts at 0, mean (1−p)/p) — always clarify which parameterization is being used.

**Poisson:** Arises as the limit of Binomial(n,p) as n→∞, p→0, with np=λ fixed (law of rare events). Key property: Poisson is closed under summation — sum of independent Poisson(λ₁) and Poisson(λ₂) is Poisson(λ₁+λ₂). Use cases: number of website hits per minute, number of defects per unit area, call-center arrivals.

**Worked example (Poisson):** A server averages λ=4 requests/second. P(exactly 6 requests in a second) = e⁻⁴4⁶/6! = e⁻⁴(4096)/720 ≈ 0.0183×5.689 ≈ **0.1042**.

**Hypergeometric vs Binomial:** Hypergeometric = sampling *without* replacement from a finite population (successes deplete the pool); Binomial assumes independence (with replacement, or infinite population). When N is much larger than n, Hypergeometric ≈ Binomial. Classic use: quality control (K defective items in a batch of N, draw n without replacement, count defects).

**Common pitfalls:**
- Using Binomial when sampling is without replacement from a small population (should be Hypergeometric).
- Forgetting Poisson requires events to occur independently at a constant average rate (not valid if events cluster/have contagion, e.g., viral events).
- Off-by-one errors in Geometric distribution support/parameterization.

### Continuous Distributions

| Distribution | PDF | Mean | Variance | Use case |
|---|---|---|---|---|
| Uniform(a,b) | 1/(b−a), a≤x≤b | (a+b)/2 | (b−a)²/12 | Complete ignorance over a range; random seeds |
| Normal(μ,σ²) | (1/√(2πσ²))e^(−(x−μ)²/2σ²) | μ | σ² | Errors, aggregated sums (via CLT), heights |
| Exponential(λ) | λe^(−λx), x≥0 | 1/λ | 1/λ² | Time between events (waiting time), survival |
| Gamma(k,θ) | x^(k−1)e^(−x/θ)/(Γ(k)θᵏ) | kθ | kθ² | Sum of k i.i.d. exponentials; skewed positive data |
| Beta(α,β) | x^(α−1)(1−x)^(β−1)/B(α,β), 0≤x≤1 | α/(α+β) | αβ/[(α+β)²(α+β+1)] | Modeling probabilities/proportions; Bayesian prior for p |
| Log-normal(μ,σ²) | (1/(xσ√2π))e^(−(ln x−μ)²/2σ²) | e^(μ+σ²/2) | (e^σ²−1)e^(2μ+σ²) | Multiplicative processes: income, stock prices, city sizes |

**Uniform:** Every sub-interval of equal length has equal probability. Basis for inverse-CDF sampling: if U~Uniform(0,1), then X=F⁻¹(U) has CDF F. This is *the* core trick behind generating samples from any distribution in code.

**Normal/Gaussian:** Fully described by μ, σ². 68-95-99.7 rule: ~68% of mass within 1σ, ~95% within 2σ, ~99.7% within 3σ. Central to statistics because of the CLT (sums/averages of many independent effects tend toward Normal) — this is *why* it appears everywhere in nature and in statistical theory (sampling distributions, residuals).

**Exponential — memorylessness:** Like the geometric distribution's continuous analogue: P(X>s+t | X>s) = P(X>t). This means "waiting time to the next event doesn't depend on how long you've already waited" — valid for Poisson-process arrivals but a common misapplied assumption (e.g., misapplied to human behaviors that actually have "aging," like machine wear-out, where Weibull is more appropriate).

**Gamma:** Generalizes Exponential (Gamma(1,θ)=Exponential(1/θ)) and is the sum of k i.i.d. Exponential(λ) random variables (Erlang distribution when k is an integer). Widely used as a conjugate prior for the Poisson rate λ and for modeling skewed non-negative data (e.g., insurance claim sizes, rainfall amounts).

**Beta:** Defined on [0,1], making it the natural distribution for modeling an unknown probability. It's the conjugate prior for the Bernoulli/Binomial likelihood parameter p, central to Bayesian A/B testing (see Bayesian Statistics section). α,β can be thought of as "pseudo-counts" of prior successes/failures.

**Log-normal:** If ln(X) ~ Normal(μ,σ²), then X ~ Log-normal. Arises from multiplicative (not additive) processes — products of many independent positive factors. Common pitfall: applying arithmetic-mean intuition to log-normal data (e.g., average income) when the median is far more representative because of the heavy right skew.

**Worked example (Exponential):** Average time between customer arrivals is 5 minutes (λ=1/5 per minute). P(next customer arrives within 2 minutes) = 1 − e^(−λt) = 1 − e^(−2/5) ≈ 1 − 0.6703 = **0.3297**.

**Worked example (Normal, z-score):** X~N(100,15²) (IQ scores). P(X>130)? z=(130−100)/15=2.0. P(Z>2.0) ≈ 0.0228 (from standard normal table), so about **2.28%**.

### Censoring and Survival Analysis Basics

**Concept:** Survival analysis models "time until an event" (churn, death, equipment failure, conversion). **Censoring** occurs when the event has not yet been observed for some subjects by the end of the study window — treating those subjects as if they never experience the event (or dropping them) biases the estimate.
- **Right-censoring** (by far the most common case): we know a subject survived at least until time t, but not the exact event time (e.g., a subscriber still active when the observation period ends).

**Hazard and survival functions:** the hazard h(t) is the instantaneous event rate at time t given survival up to t; the survival function S(t)=P(T>t)=exp(−∫₀ᵗh(u)du) gives the probability of "surviving" past time t. Exponential survival = constant hazard; Weibull allows increasing/decreasing hazard over time (see the churn Exponential-vs-Weibull discussion above).

**Kaplan-Meier estimator (intuition):** a non-parametric step-function estimate of S(t) that correctly folds in censored subjects — each subject contributes to the "at-risk" set for every event time it survives past, without requiring its (unknown) final event time.

**Worked example:** 10 subscribers are tracked; 6 churn during the study (known tenure), 4 are still active at the study's end (right-censored, tenure so far is a lower bound only). Simply averaging the tenure of the 6 who churned underestimates typical tenure, since the 4 censored subscribers have already survived at least as long and would likely pull the average up further. Kaplan-Meier correctly uses the censored subjects' partial "at risk" time instead of discarding them or treating their current tenure as a completed observation.

**Common pitfalls:**
- Treating censored subjects as if the event never occurs (overstates survival) or discarding them entirely (wastes information).
- Confusing the hazard rate (instantaneous conditional risk) with the cumulative probability of the event occurring by time t.
- Defaulting to Exponential (constant-hazard) survival models for processes with clearly time-varying risk (e.g., early-tenure churn spikes) — Weibull or Cox proportional-hazards models are more appropriate.

### Poisson Process

**Definition:** A Poisson process counts events N(t) occurring in continuous time, characterized by: N(0)=0; independent increments (counts in disjoint intervals are independent); stationary increments (the distribution of counts in an interval depends only on its length); and events occur at a constant average rate λ per unit time, with no two events simultaneous.

**Key results:**
- N(t) ~ Poisson(λt) — the count in any interval of length t is Poisson with mean λt.
- Interarrival times (gaps between consecutive events) are i.i.d. Exponential(λ) — the bridge between the Poisson (counts) and Exponential (waiting-time) distributions.
- The waiting time until the k-th event is Gamma(k, 1/λ) (Erlang distribution) — the sum of k i.i.d. Exponential(λ) interarrival times.
- **Superposition:** merging two independent Poisson processes of rates λ₁, λ₂ gives a Poisson process of rate λ₁+λ₂ (consistent with sums of independent Poissons being Poisson).
- **Thinning:** independently keeping each event with probability p yields a Poisson process of rate λp.

**Worked example:** Store arrivals form a Poisson process, λ=3/hour.
- Arrivals in 2 hours ~ Poisson(6); P(exactly 4) = e⁻⁶6⁴/4! ≈ **0.1339**.
- Time until the 3rd arrival ~ Gamma(3, 1/3), mean = 3/λ = **1 hour**.
- Time between consecutive arrivals ~ Exponential(3), mean = 1/λ = **20 minutes**.

**Common pitfalls:**
- Confusing the rate λ (per unit time) with the mean count over a specific window (λt) — always rescale by the window length.
- Assuming real arrival processes are Poisson when they actually cluster/exhibit contagion (viral spikes, bursty error logs) — Poisson processes assume independent, non-clustering increments.
- Missing the interarrival-time ↔ Erlang-waiting-time connection, usually the key insight needed to solve these problems quickly.

### Markov Chains

**Definition:** A discrete-time Markov chain is a sequence X₀,X₁,X₂,… over a finite (or countable) state space satisfying the **Markov property** — the future depends on the past only through the present state:
$$P(X_{n+1}=j \mid X_n=i, X_{n-1},\dots,X_0) = P(X_{n+1}=j \mid X_n=i)$$

**Transition matrix:** P, with Pᵢⱼ = P(X_{n+1}=j | X_n=i); each row sums to 1. **n-step transitions** are given by matrix powers: P(X_n=j | X_0=i) = (Pⁿ)ᵢⱼ (Chapman-Kolmogorov equations).

**Stationary distribution:** a row vector π (Σπᵢ=1) satisfying
$$\pi P = \pi$$
For an irreducible (every state reachable from every other) and aperiodic (no rigid cyclic pattern) finite chain, π exists, is unique, and every row of Pⁿ converges to π as n→∞ regardless of the starting state.

**Worked example:** Weather states {Sunny, Rainy} with transition matrix (row=today, column=tomorrow):
$$P = \begin{pmatrix} 0.8 & 0.2 \\ 0.4 & 0.6 \end{pmatrix}$$
Solve π_S = 0.8π_S + 0.4π_R with π_S+π_R=1: 0.2π_S = 0.4π_R ⟹ π_S=2π_R ⟹ π_R=1/3, π_S=2/3. Long-run, it's sunny 2/3 of days regardless of today's weather.

**Common pitfalls:**
- Mistaking the Markov property for full independence across time — it's conditional independence of the future from the past *given the present*, not unconditional independence.
- Confusing the stationary distribution (long-run marginal behavior) with the transition matrix (one-step conditional dynamics).
- Assuming every chain converges to a stationary distribution — periodic or reducible chains may not; irreducibility and aperiodicity are required.
- This machinery underlies MCMC sampling and, with rewards/actions added, Markov Decision Processes in reinforcement learning (see the RL syllabus file for the full MDP treatment).

### Joint, Marginal, and Conditional Distributions; Covariance and Correlation

**Joint PMF/PDF:** f(x,y) describes probability over pairs. 
**Marginal:** f_X(x) = Σ_y f(x,y) (discrete) or ∫ f(x,y) dy (continuous) — "integrate/sum out" the other variable.
**Conditional:** f(y|x) = f(x,y)/f_X(x).

**Independence of random variables:** X ⟂ Y iff f(x,y) = f_X(x)f_Y(y) for all x,y.

**Covariance:**
$$\text{Cov}(X,Y) = E[(X-\mu_X)(Y-\mu_Y)] = E[XY] - E[X]E[Y]$$

**Correlation (Pearson):**
$$\rho_{X,Y} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}, \quad -1 \le \rho \le 1$$

**Key properties:**
- Var(X+Y) = Var(X) + Var(Y) + 2Cov(X,Y); if independent, Cov=0 so Var(X+Y)=Var(X)+Var(Y).
- Independence ⟹ Cov=0, but Cov=0 does **not** imply independence (classic pitfall — e.g., X~Uniform(−1,1), Y=X² are uncorrelated but perfectly dependent).
- Correlation only captures *linear* association; strong non-linear relationships can have ρ near 0.
- Cov(aX+b, cY+d) = ac·Cov(X,Y) (scale/shift invariance of covariance up to scaling factor).

**Worked example:** X = temperature, Y = ice-cream sales, both driven by a common cause (season). Even after "controlling," spurious residual correlation can appear if confounders aren't fully modeled — motivates checking for confounders before causal claims (see Sampling & Experimental Design section).

**Common pitfalls:**
- "Correlation implies causation" — the single most repeated interview trap.
- Treating ρ=0 as proof of no relationship (could be non-linear, e.g., quadratic).
- Confusing covariance's *scale-dependence* (depends on units) with correlation's *scale invariance* (unitless, in [-1,1]).

### Expectation, Variance, Moments, and Moment Generating Functions

**Expectation (mean):**
$$E[X] = \sum_x x\,P(X=x) \quad \text{(discrete)}, \qquad E[X] = \int x f(x)\,dx \quad \text{(continuous)}$$

**Linearity of expectation:** E[aX+bY+c] = aE[X]+bE[Y]+c — holds *regardless of independence*, a frequently tested fact.

**Variance:**
$$\text{Var}(X) = E[(X-E[X])^2] = E[X^2] - (E[X])^2$$

**Properties:** Var(aX+b) = a²Var(X); Var(X+Y) = Var(X)+Var(Y)+2Cov(X,Y).

**Moments:** The k-th moment is E[Xᵏ]; the k-th central moment is E[(X−μ)ᵏ]. Skewness = 3rd standardized moment (asymmetry); Kurtosis = 4th standardized moment (tail heaviness, "peakedness"; Normal has kurtosis 3, "excess kurtosis" = kurtosis−3).

**Moment generating function (MGF):**
$$M_X(t) = E[e^{tX}]$$
Properties:
- M_X'(0) = E[X], M_X''(0) = E[X²], etc. — derivatives at 0 generate moments (hence the name).
- If X,Y independent, M_{X+Y}(t) = M_X(t)M_Y(t) — makes MGFs the tool of choice for finding the distribution of sums of independent random variables.
- Two distributions with the same MGF (in a neighborhood of 0) are identical — useful for proving distributional results (e.g., sum of independent Normals is Normal; sum of independent Poissons is Poisson).

**Worked example (MGF of Exponential):** M_X(t) = λ/(λ−t) for t<λ. M'(t) = λ/(λ−t)², M'(0)=1/λ=E[X]. ✓.

**Common pitfalls:**
- Assuming E[XY] = E[X]E[Y] in general — only true under independence.
- Forgetting Var(X−Y) = Var(X)+Var(Y)−2Cov(X,Y) (minus sign for covariance term when subtracting).
- Confusing sample variance's (n−1) divisor (Bessel's correction, unbiased) with population variance's n divisor.

### Markov's and Chebyshev's Inequalities

**Markov's inequality:** For a non-negative random variable X and any a>0:
$$P(X \ge a) \le \frac{E[X]}{a}$$
**Derivation:** E[X] = ∫x f(x)dx ≥ ∫_{x≥a} x f(x) dx ≥ a·P(X≥a); rearranging gives the bound. It requires only that X ≥ 0 and E[X] be finite — no distributional shape assumption at all.

**Chebyshev's inequality:** For any random variable X with mean μ and finite variance σ², and any k>0:
$$P(|X-\mu| \ge k\sigma) \le \frac{1}{k^2} \quad\Big(\text{equivalently } P(|X-\mu|\ge a) \le \sigma^2/a^2\Big)$$
**Derivation:** Apply Markov's inequality to the non-negative random variable Y=(X−μ)². P(Y ≥ a²) ≤ E[Y]/a² = σ²/a². Since {Y≥a²} = {|X−μ|≥a}, this gives P(|X−μ|≥a) ≤ σ²/a²; setting a=kσ yields the k² form.

**Intuition:** Markov's is a very weak, universal tail bound using only the mean. Chebyshev's tightens it using variance, giving a *distribution-free* guarantee like "at least 75% of any distribution's mass lies within 2 SD of the mean" — compare this to the Normal-specific 68-95-99.7 rule, which is much tighter but only valid for Gaussian data.

**Worked example (Markov):** Average employee salary is $70,000. What's the maximum possible fraction of employees earning ≥$200,000? P(X≥200000) ≤ 70000/200000 = **0.35** — at most 35% can earn that much, regardless of the salary distribution's shape.

**Worked example (Chebyshev):** A distribution has mean 50, SD 5 (shape unknown). Guaranteed lower bound on P(40<X<60)? This range is μ±2σ, so by Chebyshev P(|X−50|≥10) ≤ 1/2² = 0.25, hence P(40<X<60) ≥ 1−0.25 = **75%** (versus ~95% if we knew it were Normal — Chebyshev is distribution-free and therefore much looser).

**Common pitfalls:**
- Applying Markov's inequality to a variable that can be negative without first shifting/transforming it to be non-negative.
- Treating Chebyshev's bound as a tight, typical-case estimate — it's a worst-case guarantee over *all* distributions sharing that mean/variance, so real-world tail probabilities are usually far smaller.
- Forgetting these inequalities trade specificity for universality: Markov needs only a finite mean, Chebyshev needs only a finite variance — neither requires knowing the distribution's family.

### Central Limit Theorem (CLT)

**Statement:** If X₁,…,Xₙ are i.i.d. with mean μ and finite variance σ², then as n→∞:
$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} N(0,1)$$
Equivalently, $\bar{X}_n \approx N(\mu, \sigma^2/n)$ for large n, **regardless of the shape of the original distribution** (as long as it has finite variance).

**Intuition:** Averaging many independent sources of randomness cancels out idiosyncratic noise, leaving a bell-shaped distribution of the average — this is why so many natural phenomena (measurement errors, aggregated sums) look Normal even when the underlying process is not.

**Why it matters in practice:**
- Justifies using z-tests/t-tests and Normal-based confidence intervals for the sample mean even when the population distribution is unknown or skewed, provided n is reasonably large (rule of thumb: n≥30, though highly skewed data needs larger n).
- Underlies A/B testing: the difference in sample means between two groups is approximately Normal, enabling z-tests for conversion-rate differences.
- Explains why sums of many small independent effects (e.g., portfolio returns aggregated over many independent bets) trend toward Normal.

**Worked example:** Roll a fair die (uniform, definitely not Normal) 100 times and look at the sample mean. Individual roll: μ=3.5, σ²=35/12≈2.917. By CLT, sample mean X̄₁₀₀ ≈ N(3.5, 2.917/100) = N(3.5, 0.02917), so SD of the mean ≈ 0.171 — even though a single die roll is uniform, the average of 100 rolls is closely Normal.

**Common pitfalls:**
- CLT is about the *sampling distribution of the mean/sum*, not about the raw data becoming Normal.
- CLT requires finite variance — fails for heavy-tailed distributions like Cauchy (infinite variance), where sample means do not converge to Normal.
- "Large enough n" is context-dependent — highly skewed or heavy-tailed data needs much larger n than symmetric data.

### Law of Large Numbers (LLN)

**Weak LLN:** For i.i.d. X₁,…,Xₙ with mean μ, the sample mean X̄ₙ converges in probability to μ: for any ε>0, P(|X̄ₙ − μ| > ε) → 0 as n→∞.

**Strong LLN:** X̄ₙ → μ almost surely (a stronger mode of convergence).

**Intuition vs CLT:** LLN tells you *where* the sample mean is heading (converges to the true mean); CLT tells you the *rate and shape* of fluctuations around that limit (Normal, shrinking at rate 1/√n). They answer different questions and are frequently mixed up in interviews.

**Worked example:** Flip a fair coin n times, track the running proportion of heads. LLN guarantees this proportion → 0.5 as n→∞. It does **not** guarantee "corrections" will happen soon after a streak of heads (see Gambler's Fallacy below) — each flip is still independent; LLN is purely an asymptotic long-run statement.

**Common pitfalls / the Gambler's Fallacy:** Believing that after a run of heads, tails is "due" to balance things out — LLN operates through the accumulation of many future trials diluting past deviations, not through any compensating mechanism. Each individual flip remains independent and unaffected by history.

### Interview Questions

**Q1. State the PMF and give the mean/variance of a Binomial(n,p). Derive the mean using linearity of expectation.**
A: PMF: C(n,k)pᵏ(1−p)ⁿ⁻ᵏ. Write X=ΣᵢXᵢ where Xᵢ~Bernoulli(p) i.i.d. E[X]=ΣE[Xᵢ]=np (linearity holds regardless of independence). Var(X)=ΣVar(Xᵢ) (independence needed here)=np(1−p).

**Q2. When would you use a Poisson distribution instead of a Binomial? Derive the Poisson as a limiting case.**
A: Poisson models counts of rare independent events over a continuum (time/space) without a fixed number of trials n. It's the limit of Binomial(n,p) as n→∞, p→0 with np=λ fixed: C(n,k)pᵏ(1−p)ⁿ⁻ᵏ → e⁻λλᵏ/k!.

**Q3. Explain the memoryless property. Which distributions have it?**
A: P(X>s+t | X>s)=P(X>t) — the remaining waiting time doesn't depend on elapsed time. Only the Exponential (continuous) and Geometric (discrete) distributions have this property among common distributions.

**Q4. If X and Y are uncorrelated, are they independent? Give a counterexample.**
A: Not necessarily. Let X~Uniform(−1,1), Y=X². Cov(X,Y)=E[X³]−E[X]E[X²]=0−0=0 (X³ is odd, integrates to 0 over symmetric interval), so ρ=0, yet Y is a deterministic (fully dependent) function of X.

**Q5. Derive Var(X) = E[X²] − (E[X])².**
A: Var(X)=E[(X−μ)²]=E[X²−2μX+μ²]=E[X²]−2μE[X]+μ²=E[X²]−2μ²+μ²=E[X²]−μ² (using μ=E[X]).

**Q6. State the Central Limit Theorem. Why is it important for A/B testing?**
A: For i.i.d. samples with finite variance, the standardized sample mean converges to N(0,1) as n→∞, regardless of the underlying distribution's shape. In A/B testing, it justifies treating the difference in sample means (e.g., conversion rate difference) as approximately Normal, enabling z-tests/t-tests and confidence intervals even for skewed underlying metrics (like revenue per user), as long as sample sizes are large enough.

**Q7. What's the difference between the Law of Large Numbers and the Central Limit Theorem?**
A: LLN says the sample mean converges to the true mean as n→∞ (a statement about the *location* of the limit). CLT describes the *distribution of fluctuations* around that mean at rate 1/√n, showing they're approximately Normal. LLN is about convergence; CLT is about the shape/rate of convergence.

**Q8. Given X~N(0,1), what is E[X⁴]? (Use MGF or known Normal moment facts.)**
A: For standard Normal, E[X^(2k)] = (2k−1)!! (double factorial). E[X⁴] = 3!! = 3×1 = **3**. (Also derivable via MGF M(t)=e^(t²/2), M⁗(0)=3.)

**Q9. Two random variables X, Y are independent. Derive Var(X−Y) in terms of Var(X) and Var(Y).**
A: Var(X−Y)=Var(X)+Var(−Y)+2Cov(X,−Y)=Var(X)+Var(Y)−2Cov(X,Y). With independence, Cov(X,Y)=0, so Var(X−Y)=Var(X)+Var(Y) (note it's a *sum*, not a difference, even though we're subtracting the variables — a common trap).

**Q10. A website gets Poisson-distributed traffic with average 10 visits/minute. What's the probability of zero visits in a given 30-second window?**
A: λ for 30 sec = 5. P(X=0) = e⁻⁵(5⁰)/0! = e⁻⁵ ≈ **0.0067**.

**Q11 (scenario). You're modeling time-to-churn for subscribers. Would you use Exponential or Weibull, and why does it matter?**
A: Exponential assumes a constant hazard rate (memoryless — a subscriber who's stayed 2 years is just as likely to churn next month as a brand-new subscriber), which is often unrealistic. Weibull generalizes Exponential with a shape parameter that allows increasing or decreasing hazard over time (e.g., higher churn risk early on, or increasing loyalty over time), better matching real churn/survival patterns.

**Q12 (derivation). Prove that the sum of two independent Poisson random variables is Poisson.**
A: Using MGFs: M_X(t)=e^(λ₁(e^t−1)), M_Y(t)=e^(λ₂(e^t−1)). Since X⟂Y, M_{X+Y}(t)=M_X(t)M_Y(t)=e^((λ₁+λ₂)(e^t−1)), which is exactly the MGF of Poisson(λ₁+λ₂). By uniqueness of MGFs, X+Y~Poisson(λ₁+λ₂).

**Q13 (brain-teaser). You have two independent Uniform(0,1) random variables X and Y. What is P(X > Y)?**
A: By symmetry, P(X>Y)=P(Y>X), and P(X=Y)=0 (continuous), so each equals **1/2**.

**Q14 (tricky). If Cov(X,Y)=0 for all pairs among X₁,…,Xₙ but they are NOT mutually independent, does Var(ΣXᵢ)=ΣVar(Xᵢ) still hold?**
A: Yes — Var(ΣXᵢ) = ΣVar(Xᵢ) + 2Σᵢ<ⱼ Cov(Xᵢ,Xⱼ). This formula only requires pairwise covariances to vanish, not full independence, so the variance-additivity result still holds even without independence, as long as all pairwise covariances are zero.

**Q15 (scenario). Why might a heavy-tailed metric like revenue-per-user break naive application of CLT-based confidence intervals in an A/B test with modest sample size?**
A: CLT convergence rate depends on how quickly the distribution's tail behavior "averages out" — extremely skewed/heavy-tailed data (e.g., a few whales driving most revenue) requires much larger n before the sampling distribution of the mean looks Normal. With modest n, the actual sampling distribution can remain skewed, causing z/t-test p-values and CIs to be unreliable; practitioners often log-transform, winsorize/cap outliers, use the bootstrap, or use non-parametric tests (Mann-Whitney) instead.

**Q16. State Markov's inequality and derive Chebyshev's inequality from it.**
A: Markov: for X≥0, P(X≥a) ≤ E[X]/a. To get Chebyshev, apply Markov to Y=(X−μ)² (non-negative): P(Y≥a²) ≤ E[Y]/a² = σ²/a². Since {Y≥a²}={|X−μ|≥a}, this gives P(|X−μ|≥a) ≤ σ²/a², i.e., P(|X−μ|≥kσ) ≤ 1/k² when a=kσ.

**Q17. A distribution (unknown shape) has mean 40 and standard deviation 4. Give a lower bound on the probability the value falls between 32 and 48.**
A: 32 and 48 are μ±2σ. By Chebyshev, P(|X−40|≥8) ≤ 1/2²=0.25, so P(32<X<48) ≥ 1−0.25 = **75%**. This bound holds regardless of the distribution's actual shape, unlike the Normal-specific 95% figure.

**Q18. Derive the fact that interarrival times of a Poisson process with rate λ are Exponential(λ).**
A: Let T be the time until the first event. P(T>t) = P(no events in [0,t]) = P(N(t)=0) = e^(−λt) (from the Poisson(λt) count distribution). So the survival function of T is e^(−λt), which is exactly the Exponential(λ) survival function — hence T~Exponential(λ), and by stationary/independent increments, every subsequent interarrival time has this same distribution independently.

**Q19. Derive the stationary distribution of a 2-state Markov chain with transition matrix P=[[0.8,0.2],[0.4,0.6]].**
A: Solve πP=π with π_S+π_R=1: π_S=0.8π_S+0.4π_R ⟹ 0.2π_S=0.4π_R ⟹ π_S=2π_R. Substituting: 2π_R+π_R=1 ⟹ π_R=1/3, π_S=2/3. The chain spends 2/3 of the long run in state S regardless of its starting state.

**Q20. What conditions guarantee a Markov chain converges to a unique stationary distribution, and what happens if they're violated?**
A: The chain must be irreducible (every state reachable from every other, so it doesn't get "stuck" in a subset of states) and aperiodic (doesn't cycle through states in a rigid, deterministic pattern). If reducible, different starting states can converge to different limiting distributions (or fail to mix across components); if periodic (e.g., a chain that alternates deterministically between two states), Pⁿ never settles to a single limiting distribution even though a stationary π satisfying πP=π may still exist.

**Q21 (scenario). You're estimating average subscriber tenure, but 40% of subscribers are still active (haven't churned) when your data snapshot is taken. Why is simply averaging the tenure of only the churned subscribers wrong, and what's the right approach?**
A: This ignores right-censoring: the still-active subscribers have already survived at least their current tenure and would likely extend the average upward if followed further, so restricting to churned subscribers systematically underestimates true average tenure. The right approach is a survival-analysis method (e.g., Kaplan-Meier for the survival curve, or a parametric hazard model like Exponential/Weibull/Cox) that incorporates the censored subjects' partial at-risk time rather than dropping or misclassifying them.

---

## Statistical Inference

*Relevance: Data Scientist (core — the single most heavily tested section), ML Engineer (core — MLE underlies most loss functions; bias-variance is central to model selection), AI Engineer (moderate — evaluating model/benchmark differences statistically).*

### Point Estimation: MLE and MAP

**Maximum Likelihood Estimation (MLE):** Choose the parameter θ that maximizes the probability of observing the data:
$$\hat\theta_{MLE} = \arg\max_\theta \; L(\theta) = \arg\max_\theta \prod_{i=1}^n f(x_i;\theta)$$
In practice, maximize the log-likelihood ℓ(θ) = Σ log f(xᵢ;θ) (monotonic transform, turns products into sums, numerically stable).

**Worked derivation — Bernoulli MLE:** Data x₁,…,xₙ ∈{0,1}, iid Bernoulli(p). Likelihood: L(p)=p^(Σxᵢ)(1−p)^(n−Σxᵢ). Log-likelihood: ℓ(p)=(Σxᵢ)ln p + (n−Σxᵢ)ln(1−p). Take derivative, set to 0:
$$\frac{d\ell}{dp} = \frac{\sum x_i}{p} - \frac{n-\sum x_i}{1-p} = 0 \implies \hat p_{MLE} = \frac{1}{n}\sum x_i = \bar{x}$$
The MLE of p is simply the sample proportion — matches intuition.

**Worked derivation — Normal MLE:** Data iid N(μ,σ²). 
$$\hat\mu_{MLE} = \bar{x}, \qquad \hat\sigma^2_{MLE} = \frac{1}{n}\sum(x_i-\bar{x})^2$$
Note: this variance estimator is **biased** (divides by n, not n−1) — E[σ̂²_MLE] = ((n−1)/n)σ², a frequently asked follow-up.

**Maximum a Posteriori (MAP):** Incorporate a prior p(θ) and maximize the posterior:
$$\hat\theta_{MAP} = \arg\max_\theta \; p(\theta \mid x) = \arg\max_\theta \; f(x\mid \theta) p(\theta)$$
MAP = MLE + a regularization term from the prior (in log space, log-prior acts as a penalty).

**Connection to ML regularization (important for ML Engineer interviews):**
- MAP with a **Gaussian prior** on weights ⟺ **L2 regularization / Ridge regression**.
- MAP with a **Laplace prior** on weights ⟺ **L1 regularization / Lasso**.
- As prior becomes flat (uninformative), MAP → MLE.

**Worked example:** Coin flip with 8 heads out of 10 flips.
- MLE: p̂ = 8/10 = 0.8.
- MAP with Beta(2,2) prior (mild prior belief toward fairness): posterior ∝ p^(8+2−1)(1−p)^(2+2−1) = p⁹(1−p)³, i.e., posterior is Beta(10,4). Mode = (α−1)/(α+β−2) = (10−1)/(10+4−2) = 9/12 = **0.75**. So MAP pulls the estimate toward the prior mean (0.5), giving 0.75 instead of MLE's 0.8.

**Common pitfalls:**
- Confusing MLE (no prior, "let data speak") with MAP (prior regularizes, especially valuable with small data).
- Forgetting the MLE of variance is biased (n divisor); the unbiased sample variance uses n−1.
- Assuming MLE is always consistent/unbiased — it's asymptotically consistent and often asymptotically unbiased, but can be biased in finite samples (as shown above).

### Maximum Likelihood for the Exponential Family (General Form)

**Definition:** A distribution belongs to the exponential family if its density/PMF can be written as
$$f(x;\theta) = h(x)\exp\big(\eta(\theta)^\top T(x) - A(\theta)\big)$$
where η(θ) is the natural parameter, T(x) is the sufficient statistic, A(θ) is the log-partition (normalizing) function, and h(x) is a base measure. Bernoulli, Binomial, Poisson, Exponential, Normal, Gamma, Beta, and Multinomial are all members — this single form unifies the "named" distributions used throughout GLMs.

**The general MLE result:** because E_θ[T(X)] = ∇_η A(η) (a standard exponential-family identity) and the log-likelihood's gradient w.r.t. η is Σᵢ T(xᵢ) − nA(η), setting the gradient to zero gives one elegant "moment-matching" rule for *any* exponential-family member:
$$\nabla_\eta A(\hat\eta_{MLE}) = \frac{1}{n}\sum_{i=1}^n T(x_i)$$
i.e., **the MLE sets the model's expected sufficient statistic equal to the observed average sufficient statistic.** This is why Bernoulli's MLE is the sample mean, Poisson's MLE is the sample mean, and Normal's MLE is the sample mean/variance — all are instances of one underlying mechanic, and it's the reason GLM log-likelihoods (linear, logistic, Poisson regression) are all concave with a unique MLE fit via the same iteratively-reweighted-least-squares machinery.

**Worked example (Poisson via the general form):** f(x;λ)=e^(−λ)λˣ/x! = (1/x!)exp(x ln λ − λ). Here T(x)=x, η=ln λ, A(η)=λ=e^η. The rule matches E[T(X)]=∇_ηA(η)=e^η=λ to the sample mean, giving λ̂_MLE = x̄ — the familiar Poisson MLE, recovered directly from the exponential-family identity instead of a fresh derivative-and-solve.

**Common pitfalls:**
- Assuming the moment-matching shortcut applies to any distribution — it only holds for genuine exponential-family members (not, e.g., mixture models or the Student-t distribution).
- Confusing "exponential family" (the broad class of distributions with this factorized form) with "the Exponential distribution" (one specific member) — a common naming trap.
- Forgetting the practical payoff: this is precisely why the "link function" formalism in GLMs works, and why GLM coefficient estimation is numerically well-behaved.

### Bias, Variance, and Consistency of Estimators; Bias-Variance Tradeoff

**Bias:** Bias(θ̂) = E[θ̂] − θ. An estimator is **unbiased** if Bias=0 for all θ.

**Variance:** Var(θ̂) measures the spread of the estimator across different samples.

**Consistency:** θ̂ₙ is consistent if θ̂ₙ → θ in probability as n→∞ (regardless of finite-sample bias).

**Mean Squared Error (MSE) decomposition** — the central identity:
$$MSE(\hat\theta) = E[(\hat\theta-\theta)^2] = \text{Var}(\hat\theta) + \text{Bias}(\hat\theta)^2$$

**Worked derivation:**
$$E[(\hat\theta-\theta)^2] = E[(\hat\theta - E[\hat\theta] + E[\hat\theta]-\theta)^2] = \text{Var}(\hat\theta) + 2E[\hat\theta-E[\hat\theta]]\cdot(E[\hat\theta]-\theta) + \text{Bias}^2$$
The cross term vanishes since E[θ̂−E[θ̂]]=0, leaving Var+Bias².

**Bias-variance tradeoff (statistical view, generalizes to ML):** You cannot always minimize both bias and variance simultaneously; the best MSE-minimizing estimator often accepts a small amount of bias for a large reduction in variance (this is precisely the statistical justification for regularization / shrinkage estimators like Ridge, and for MAP estimation with informative priors).

**Worked example:** Sample variance with n divisor (MLE) vs n−1 divisor (unbiased, "Bessel's correction"):
- E[S²_MLE] = ((n−1)/n)σ² → biased (underestimates), but lower variance.
- E[S²_unbiased] = σ² → unbiased, but slightly higher variance.
For small n, the n-divisor version can have *lower MSE* despite being biased — a concrete numeric illustration of the tradeoff.

**Efficiency:** Among unbiased estimators, the one with the smallest variance is most efficient. The **Cramér-Rao Lower Bound** gives the theoretical minimum variance any unbiased estimator can achieve (1/Fisher Information) — MLE achieves this bound asymptotically (asymptotic efficiency).

**Common pitfalls:**
- Assuming "unbiased" automatically means "better" — a biased estimator can have lower MSE.
- Confusing bias-variance tradeoff in *estimator theory* with the ML *model complexity* framing (underfitting=high bias, overfitting=high variance) — same math, different context; interviewers often want you to connect the two.
- Believing consistency implies unbiasedness (false — a consistent estimator can be biased in every finite sample as long as bias→0).

### Confidence Intervals

**Definition:** A C% confidence interval is a range constructed by a *procedure* such that, if the experiment were repeated many times, C% of the intervals so constructed would contain the true parameter.

**Formula (mean, known σ, or large n via CLT):**
$$\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$
For unknown σ (typical case), replace σ with sample std s and z with the t-distribution critical value:
$$\bar{x} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}$$

| Confidence level | z_{α/2} |
|---|---|
| 90% | 1.645 |
| 95% | 1.96 |
| 99% | 2.576 |

**Worked example:** Sample of n=100, x̄=50, s=10. 95% CI = 50 ± 1.96×(10/√100) = 50 ± 1.96 = **[48.04, 51.96]**.

**Correct interpretation (heavily tested — the #1 CI misinterpretation trap):** "We are 95% confident the true mean lies in [48.04, 51.96]" means: *if we repeated this sampling procedure many times, 95% of the resulting intervals would contain the true parameter.* It does **not** mean "there's a 95% probability the true mean is in this specific interval" (in frequentist terms, the true mean is fixed, not random — the interval is random). This subtle distinction is the classic tripwire between frequentist CIs and Bayesian credible intervals.

**Width factors:** CI width ∝ σ/√n (shrinks with more data, at rate 1/√n — diminishing returns) and grows with confidence level (99% CI is wider than 90% CI, all else equal).

**Bootstrap confidence intervals:** Resample the data (with replacement) B times, compute the statistic on each resample, and use the empirical percentiles (e.g., 2.5th and 97.5th) as the CI bounds. Useful when the sampling distribution is unknown or the statistic has no closed-form CI (e.g., median, correlation coefficient, complex ML metrics).

**Common pitfalls:**
- Misinterpreting CI as "probability the parameter is in the interval" (frequentist parameters are fixed, not random).
- Using z instead of t for small samples with unknown σ (underestimates uncertainty, CI too narrow).
- Assuming a non-overlapping CI between two groups is required to claim significance, or that overlapping CIs necessarily mean "not significant" (partial overlap can still be significant — always run the actual hypothesis test).

### Hypothesis Testing

**Framework:**
- **Null hypothesis (H₀):** the default/no-effect claim (e.g., "new design has the same conversion rate as old").
- **Alternative hypothesis (H₁/Hₐ):** what you're trying to find evidence for.
- **Test statistic:** a function of the data used to decide between H₀ and H₁.
- **p-value:** the probability, *assuming H₀ is true*, of observing a test statistic at least as extreme as the one observed.
- **Significance level (α):** the pre-chosen threshold (commonly 0.05) below which you reject H₀.

**Decision rule:** Reject H₀ if p-value < α.

**Error types:**

| | H₀ True | H₀ False |
|---|---|---|
| **Reject H₀** | Type I Error (α) — false positive | Correct (Power = 1−β) |
| **Fail to reject H₀** | Correct | Type II Error (β) — false negative |

- **Type I error rate = α** (significance level) — probability of a false positive.
- **Type II error rate = β** — probability of a false negative.
- **Statistical power = 1 − β** — probability of correctly detecting a true effect. Power increases with larger sample size, larger true effect size, lower variance, and higher α (a tradeoff — raising α to boost power raises false-positive risk too).

**Common pitfall on p-values (heavily tested):** A p-value of 0.03 does **not** mean "3% chance H₀ is true" or "97% chance H₁ is true." It means: *if H₀ were true*, you'd see data this extreme or more only 3% of the time. p-values say nothing directly about P(H₀|data) — that requires Bayesian reasoning with a prior.

**Worked numeric example (one-sample z-test):** A factory claims average bolt length is 10cm. Sample of n=49, x̄=10.3, σ=1.4 (known). H₀: μ=10, H₁: μ≠10.
$$z = \frac{\bar x - \mu_0}{\sigma/\sqrt n} = \frac{10.3-10}{1.4/7} = \frac{0.3}{0.2} = 1.5$$
Two-tailed p-value = 2×P(Z>1.5) ≈ 2×0.0668 = **0.1336**. Since 0.1336 > 0.05, **fail to reject H₀** — insufficient evidence the mean differs from 10cm.

**Practical vs statistical significance:** With huge sample sizes, even trivially small (practically meaningless) effects become statistically significant. Always report effect size (e.g., Cohen's d, absolute lift) alongside p-values.

### Effect Size and Practical Significance (Cohen's d)

**Concept:** A p-value only tells you an effect is unlikely to be pure chance; it says nothing about whether the effect is *large enough to matter*. **Effect size** quantifies the magnitude of a difference in a standardized, sample-size-independent way, and should always be reported alongside the p-value.

**Cohen's d (standardized mean difference):**
$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{pooled}}, \qquad s_{pooled} = \sqrt{\frac{(n_1-1)s_1^2+(n_2-1)s_2^2}{n_1+n_2-2}}$$
This uses the same pooled SD as the two-sample t-test denominator — d and the t-statistic are directly related: t = d·√(n₁n₂/(n₁+n₂)). Notice t grows with n even if d stays fixed, which is exactly why large-n studies achieve significance for tiny effect sizes.

**Rule-of-thumb interpretation (Cohen's conventions):** |d|≈0.2 small, ≈0.5 medium, ≈0.8 large. These are heuristics, not laws — a "small" d can still be highly consequential at scale (e.g., a billion-impression ad platform).

**Worked example:** Control mean session time = 300s (s₁=50), treatment mean = 315s (s₂=55), n₁=n₂=10,000. Pooled SD ≈ 52.5. d = 15/52.5 ≈ **0.286** (small-to-medium). With n=10,000/group this difference will almost certainly be statistically significant (p<0.001) despite the modest effect size — a textbook illustration of why both numbers belong together in any report.

**Other effect-size measures worth knowing:** odds ratio / relative risk (binary outcomes), Cramér's V (effect size for chi-square test of independence), r²/η² (variance explained, regression/ANOVA), and absolute/relative lift (the business-friendly framing common in A/B testing).

**Common pitfalls:**
- Reporting only a p-value with no effect size — p<0.0001 from a huge sample can correspond to a practically meaningless d=0.02.
- Assuming a large effect-size estimate is automatically "real" — small samples can produce large, noisy effect-size estimates that fail to replicate (effect sizes have their own sampling variability and should ideally be reported with a CI).
- Note: statistical tests for comparing *ML models'* metrics on held-out data (paired t-test, McNemar's test, etc.) are covered in the Feature Engineering & Model Evaluation material — this section covers the general statistical-significance-vs-effect-size concept.

### Common Statistical Tests

| Test | Use case | Assumptions |
|---|---|---|
| **Z-test** | Compare mean(s) to a value, or two means, when σ known or n large | Normal (or CLT via large n), known variance |
| **One-sample t-test** | Compare sample mean to hypothesized value, σ unknown | Approx. Normal, unknown σ |
| **Two-sample t-test (independent)** | Compare means of two independent groups | Approx. Normal, (assume equal or unequal variance — Welch's t-test for unequal) |
| **Paired t-test** | Compare means of paired/matched observations (before/after) | Differences approx. Normal |
| **Chi-square goodness-of-fit** | Test if observed categorical frequencies match expected distribution | Expected counts ≥5 per cell (rule of thumb) |
| **Chi-square test of independence** | Test if two categorical variables are associated | Expected counts ≥5 per cell |
| **ANOVA (F-test)** | Compare means across 3+ groups | Normality, homogeneity of variance (homoscedasticity) |
| **Mann-Whitney U** | Compare distributions of two independent groups (non-parametric alternative to t-test) | Ordinal/continuous data, no normality assumption |
| **Wilcoxon signed-rank** | Compare paired samples (non-parametric alternative to paired t-test) | Symmetric distribution of differences |
| **Kolmogorov-Smirnov (KS)** | Compare a sample to a reference distribution, or two samples' distributions | Continuous distributions |

**t-test formula (two independent samples, equal variance assumed):**
$$t = \frac{\bar x_1 - \bar x_2}{s_p\sqrt{1/n_1+1/n_2}}, \quad s_p^2 = \frac{(n_1-1)s_1^2+(n_2-1)s_2^2}{n_1+n_2-2}$$
Degrees of freedom = n₁+n₂−2.

**Paired t-test:** Reduces to a one-sample t-test on the differences dᵢ = xᵢ − yᵢ: t = d̄/(s_d/√n). Use when observations are naturally linked (same user before/after, twin studies) — paired tests have more power than unpaired because they remove between-subject variance.

**Chi-square statistic:**
$$\chi^2 = \sum_i \frac{(O_i - E_i)^2}{E_i}$$
where Oᵢ = observed count, Eᵢ = expected count under H₀.

**Worked example (chi-square goodness-of-fit):** A die is rolled 60 times; observed counts: [8,9,12,11,7,13] for faces 1-6. Under fairness, expected each = 10.
$$\chi^2 = \frac{(8-10)^2}{10}+\frac{(9-10)^2}{10}+\frac{(12-10)^2}{10}+\frac{(11-10)^2}{10}+\frac{(7-10)^2}{10}+\frac{(13-10)^2}{10}$$
$$= 0.4+0.1+0.4+0.1+0.9+0.9 = 2.8$$
df=5. Critical value at α=0.05 is 11.07 — since 2.8 < 11.07, **fail to reject H₀** (no evidence die is unfair).

**Chi-square test of independence example:** 2×2 contingency table testing if gender is associated with product preference. Compute expected counts as (row total × column total)/grand total per cell, then apply the same χ² formula; df=(rows−1)(cols−1).

**ANOVA:** Tests H₀: μ₁=μ₂=...=μₖ using the F-statistic = (between-group variance)/(within-group variance). A significant F-test tells you *at least one* group differs, but not *which* — requires post-hoc tests (Tukey's HSD, Bonferroni-corrected pairwise t-tests) to localize the difference.

**Non-parametric tests:** Use when normality assumption is violated or data is ordinal/ranked.
- **Mann-Whitney U:** rank-based test comparing two independent groups' distributions (tests if one distribution stochastically dominates the other, not strictly means).
- **Wilcoxon signed-rank:** paired analogue, using ranks of the differences.
- **KS test:** compares empirical CDFs; useful for testing normality or comparing two samples' full distributions (sensitive to any type of distributional difference: location, scale, shape).

**Common pitfalls:**
- Using a two-sample t-test with unequal variances without switching to Welch's correction (can inflate Type I error).
- Applying chi-square with small expected cell counts (<5) — violates the chi-square approximation; use Fisher's exact test instead.
- Running ANOVA then interpreting significance as "all groups differ" — it only says "not all are equal."
- Using a t-test when data is heavily skewed/ordinal with small n — non-parametric alternative is safer.

### Multiple Testing Correction

**The problem:** Running m independent hypothesis tests at α=0.05 each inflates the family-wise Type I error rate: P(at least one false positive) = 1−(1−α)^m, which grows quickly (e.g., m=20 tests → ~64% chance of at least one false positive by chance alone).

**Bonferroni correction:** Use adjusted significance level α' = α/m (or equivalently, multiply each p-value by m before comparing to α). Simple, conservative — controls family-wise error rate (FWER) but sacrifices power, especially as m grows large.

**Benjamini-Hochberg (FDR control):** Controls the **False Discovery Rate** (expected proportion of false positives *among rejected hypotheses*) rather than the probability of any false positive. Procedure:
1. Sort m p-values ascending: p₁≤p₂≤...≤p_m.
2. Find the largest k such that p_k ≤ (k/m)×q (q = desired FDR level, e.g., 0.05).
3. Reject H₀ for all tests with p ≤ p_k.

**Worked example (BH procedure):** 5 p-values: 0.01, 0.02, 0.03, 0.04, 0.20, at q=0.05. Thresholds: (1/5)(0.05)=0.01, (2/5)(0.05)=0.02, (3/5)(0.05)=0.03, (4/5)(0.05)=0.04, (5/5)(0.05)=0.05. Compare each pᵢ to its threshold: 0.01≤0.01 ✓, 0.02≤0.02 ✓, 0.03≤0.03 ✓, 0.04≤0.04 ✓, 0.20≤0.05 ✗. Largest k satisfying the condition is k=4, so reject the first 4 hypotheses (p ≤ 0.04).

**When to use which:** Bonferroni for a small number of tests or when any single false positive is very costly (e.g., clinical trials with a handful of endpoints). BH/FDR for large-scale testing (e.g., thousands of A/B test metrics, genomics with thousands of genes) where some false positives are tolerable in exchange for much greater power.

**Common pitfalls:**
- Not correcting at all when running many simultaneous tests ("p-hacking" via multiple comparisons).
- Confusing FWER control (Bonferroni: probability of *any* false positive) with FDR control (BH: *expected proportion* of false positives among discoveries) — they answer different questions.
- Applying Bonferroni to hundreds/thousands of tests, resulting in near-zero power (should use BH instead).

### Bootstrap Resampling and Permutation Tests

**Bootstrap resampling:** Given an observed sample of size n, draw B resamples *with replacement*, each of size n, from the original data, and compute the statistic of interest on each resample to build an empirical sampling distribution. Used for standard errors, confidence intervals (see above), and bias estimation whenever no closed-form sampling distribution exists (e.g., median, correlation coefficient, ratio of two means, custom business metrics).

**Bootstrap SE and bias estimate:**
$$\widehat{SE}_{boot} = \sqrt{\frac{1}{B-1}\sum_{b=1}^B \big(\hat\theta^{*(b)} - \bar{\hat\theta}^*\big)^2}, \qquad \widehat{Bias}_{boot} = \bar{\hat\theta}^* - \hat\theta$$
where θ̂*(b) is the statistic on the b-th resample and θ̂ is the statistic on the original sample.

**Worked example:** Original sample (n=5) of purchase amounts [10, 12, 15, 100, 11], median=12. Draw B=1,000 bootstrap resamples of size 5 (with replacement), compute the median of each; the empirical distribution of these 1,000 medians directly gives a bootstrap SE and a percentile-based 95% CI for the true median — without deriving the analytically awkward sampling distribution of the median.

**Permutation test (randomization test):** To test H₀: "two groups come from the same distribution" (no effect of the group label), pool all observations, then repeatedly (1) randomly shuffle the group labels, (2) recompute the test statistic (e.g., difference in means) on the shuffled data. The p-value is the fraction of shuffled-label statistics at least as extreme as the one actually observed — this simulates the null distribution directly, requiring only *exchangeability* under H₀ (no Normality assumption).

**Worked example:** Group A (n=5): [23,25,21,30,22], mean=24.2. Group B (n=5): [30,28,35,27,31], mean=30.2. Observed difference = 6.0. Pool all 10 values and repeatedly draw random 5/5 splits (or enumerate all C(10,5)=252 splits exactly), computing the mean difference each time; p-value = fraction of permuted differences with |difference| ≥ 6.0. With groups this well-separated, the p-value will typically come out small, agreeing directionally with a standard two-sample t-test — permutation and t-tests usually agree when t-test assumptions hold, but the permutation test stays valid even when they don't.

**Scope note:** these are general-purpose inference tools applicable to any statistic or hypothesis about a data-generating process. When the "two groups" being compared are two competing *ML models'* predictions on the same test set (e.g., paired bootstrap for model-metric CIs, or McNemar's test for classifier comparison), that application-specific material lives in the Feature Engineering & Model Evaluation syllabus file.

**Common pitfalls:**
- Confusing the bootstrap (resampling *the data* to approximate the sampling distribution under the actual population) with the permutation test (reshuffling *labels* to build the null distribution assuming H₀) — different questions, not interchangeable.
- Using too few resamples/permutations (B<1,000 or so) — inflates Monte Carlo noise in the estimated SE/CI/p-value.
- Believing the bootstrap fixes a small or biased original sample — it only approximates sampling variability *given* that sample; it can't manufacture information the sample doesn't contain.
- Naively shuffling data with inherent structure (e.g., autocorrelated time series) — this destroys structure that should be preserved and invalidates the exchangeability assumption.

### Interview Questions

**Q1. Derive the MLE for the parameter p of a Bernoulli distribution given n i.i.d. observations.**
A: ℓ(p)=Σxᵢ ln p + (n−Σxᵢ)ln(1−p). dℓ/dp = Σxᵢ/p − (n−Σxᵢ)/(1−p) = 0 ⟹ p̂=(1/n)Σxᵢ=x̄, the sample mean/proportion.

**Q2. What's the difference between MLE and MAP? When do they coincide?**
A: MLE maximizes the likelihood P(data|θ); MAP maximizes the posterior P(θ|data) ∝ P(data|θ)P(θ), incorporating a prior. They coincide when the prior is uniform/flat (uninformative), since then the posterior is proportional to the likelihood alone.

**Q3. Explain bias, variance, and the bias-variance decomposition of MSE.**
A: Bias = E[θ̂]−θ (systematic error); Variance = spread of θ̂ across samples. MSE(θ̂)=Var(θ̂)+Bias(θ̂)². A good estimator balances both; sometimes accepting bias lowers overall MSE by reducing variance more.

**Q4. Why does the MLE of variance use n instead of n−1, and why is the "corrected" sample variance considered better?**
A: MLE of σ² is (1/n)Σ(xᵢ−x̄)², derived by maximizing the Normal likelihood — but this uses x̄ estimated from the same data, "using up" a degree of freedom, causing systematic underestimation: E[σ̂²_MLE]=((n−1)/n)σ². Dividing by n−1 instead (Bessel's correction) makes the estimator exactly unbiased, E[S²]=σ².

**Q5. Explain what a 95% confidence interval actually means, and give a common misinterpretation.**
A: Correct: if you repeated the sampling/estimation procedure many times, 95% of the constructed intervals would contain the true (fixed) parameter. Misinterpretation: "there's a 95% probability the true parameter lies in this specific interval" — wrong under frequentist logic because the parameter is fixed, not random; only the interval (from sample to sample) is random.

**Q6. Define Type I and Type II error and statistical power. How are they related?**
A: Type I error (α) = rejecting a true H₀ (false positive). Type II error (β) = failing to reject a false H₀ (false negative). Power = 1−β = probability of correctly detecting a real effect. For fixed sample size, reducing α (stricter threshold) tends to increase β (lower power); increasing sample size can reduce β without raising α.

**Q7. A p-value of 0.04 is obtained. Explain precisely what this number means, and what it does NOT mean.**
A: It means: assuming H₀ is true, the probability of observing a test statistic as extreme as (or more extreme than) the one observed is 4%. It does NOT mean "4% probability H₀ is true" nor "96% probability the effect is real" — p-values are computed under the assumption H₀ holds, they don't give P(H₀|data).

**Q8. When should you use a paired t-test instead of an independent two-sample t-test?**
A: When observations are naturally linked/matched (e.g., same subjects measured before and after treatment). Paired tests analyze the within-subject differences, removing between-subject variability, which increases statistical power compared to treating the two samples as independent.

**Q9. Explain the difference between a chi-square goodness-of-fit test and a chi-square test of independence.**
A: Goodness-of-fit compares one categorical variable's observed distribution to a hypothesized/theoretical distribution (e.g., is this die fair?). Test of independence uses a contingency table to test whether two categorical variables are associated (e.g., is gender related to product preference?). Both use the same χ²=Σ(O−E)²/E statistic but differ in how expected counts are derived and degrees of freedom.

**Q10. Why do we need multiple testing correction? Compare Bonferroni and Benjamini-Hochberg.**
A: Running many tests at α=0.05 each inflates the chance of at least one false positive (family-wise error rate = 1−(1−α)^m). Bonferroni divides α by the number of tests m, controlling FWER (probability of any false positive) but is conservative, losing power as m grows. Benjamini-Hochberg controls the False Discovery Rate (expected proportion of false discoveries among rejections) via a step-up procedure on sorted p-values, retaining much more power for large-scale testing (e.g., genomics, many simultaneous A/B metrics).

**Q11 (scenario). You run an A/B test and get p=0.06 with n=500. Your PM says "let's just collect more data until it's significant." What's wrong with this?**
A: This is "p-hacking via optional stopping" — repeatedly checking and continuing to sample until reaching p<0.05 inflates the true Type I error rate far above 5%, because you're effectively running many sequential tests without correction. The correct approach is to determine sample size *before* the test (power analysis) and stick to it, or use a formal sequential-testing method (e.g., alpha-spending / SPRT) designed to control error rates under repeated peeking.

**Q12 (derivation/scenario). Derive the two-sample t-statistic for testing equality of means and explain each term.**
A: t = (x̄₁−x̄₂)/(s_p√(1/n₁+1/n₂)), where s_p² = [(n₁−1)s₁²+(n₂−1)s₂²]/(n₁+n₂−2) is the pooled variance estimate combining both samples' variability (assumes equal population variances). The numerator is the observed mean difference; the denominator is the standard error of that difference under H₀: μ₁=μ₂. Compare |t| to the t-distribution with n₁+n₂−2 degrees of freedom.

**Q13 (tricky). Two groups have significantly overlapping 95% confidence intervals for their means. Does this mean the difference between them is not statistically significant?**
A: Not necessarily — overlapping CIs don't automatically mean the difference is non-significant; the correct test is a direct two-sample test on the difference (which has its own, generally smaller, standard error than the sum of individual CI widths would suggest). Conversely, non-overlapping CIs generally do imply significance, but overlapping CIs can still yield p<0.05 in a direct comparison test. Always run the actual test rather than eyeballing CI overlap.

**Q14 (scenario). Why might a chi-square test give unreliable results in a table with several cells having expected counts of 2?**
A: The chi-square test statistic's distribution is only well-approximated by the χ² distribution asymptotically; with small expected cell counts (rule of thumb <5), the approximation breaks down and p-values become unreliable (often too liberal). Use Fisher's exact test instead for small samples/sparse tables.

**Q15 (brain-teaser). If you flip a coin 100 times and observe 60 heads, is the coin biased? Set up and perform the hypothesis test.**
A: H₀: p=0.5 vs H₁: p≠0.5. Under H₀, X~Binomial(100,0.5), approx Normal with mean 50, SD=√(100×0.5×0.5)=5. z=(60−50)/5=2.0. Two-tailed p-value=2×P(Z>2.0)≈2×0.0228=0.0456. Since 0.0456<0.05, **reject H₀ at the 5% level** — mild evidence of bias, though it's a borderline result (would not survive a stricter α=0.01 threshold or multiple-testing correction).

**Q16. Write the general exponential-family form of a distribution and state the resulting general rule for its MLE.**
A: f(x;θ)=h(x)exp(η(θ)ᵀT(x)−A(θ)), with η the natural parameter, T(x) the sufficient statistic, and A(θ) the log-partition function. The MLE solves ∇_ηA(η̂)=(1/n)Σ T(xᵢ), i.e., it matches the model's expected sufficient statistic to the observed average sufficient statistic — the single mechanism underlying the Bernoulli, Poisson, and Normal MLE derivations shown earlier in this section.

**Q17. What's the difference between bootstrap resampling and a permutation test? Give a scenario for each.**
A: Bootstrap resampling draws samples *with replacement from the observed data* to approximate the sampling distribution of a statistic under the actual (unknown) population — e.g., building a CI for a median. A permutation test *shuffles group labels* to build the null distribution assuming no group effect, then compares the observed statistic to that null — e.g., testing whether two groups' means differ without assuming Normality. Bootstrap answers "how uncertain is my estimate?"; permutation testing answers "how surprising is my observed difference under H₀?"

**Q18. You want a 95% CI for the median of a skewed dataset with no known closed-form sampling distribution. How would you construct one?**
A: Use the bootstrap: draw B (e.g., 1,000–10,000) resamples with replacement of the same size as the original data, compute the median on each resample, and take the 2.5th and 97.5th percentiles of the resulting distribution of medians as the CI bounds (the percentile bootstrap method). This works because it empirically approximates the median's sampling distribution without requiring a closed-form formula.

**Q19. Compute Cohen's d for two groups: Group 1 mean=52, SD=8, n=40; Group 2 mean=48, SD=9, n=40. Interpret the result.**
A: Pooled SD = √[((39)(64)+(39)(81))/(78)] = √[(2496+3159)/78] = √72.5 ≈ 8.51. d = (52−48)/8.51 ≈ **0.47**, a small-to-medium effect by Cohen's conventions. Whether this is "significant" would still require a formal t-test on top of this — d alone doesn't give a p-value, and the p-value alone wouldn't tell you the effect is this small.

**Q20 (scenario). Your experiment finds p=0.001 for a new checkout flow, but Cohen's d is only 0.03. How would you advise the team?**
A: The result is statistically significant but the effect size is negligible — with a large enough sample, even a practically meaningless difference becomes detectable. Before shipping, weigh the (tiny) measured benefit against implementation/maintenance cost, guardrail-metric risk, and whether the effect would even be perceptible to users or move a business-relevant KPI; statistical significance alone is not sufficient justification for a launch decision.

---

## Bayesian Statistics

*Relevance: Data Scientist (core, especially for A/B testing and priors-driven modeling), ML Engineer (core — regularization-as-prior, Bayesian model averaging, uncertainty quantification), AI Engineer (moderate — Bayesian reasoning underlies calibration and uncertainty estimates in generative models).*

### Prior, Likelihood, Posterior; Conjugate Priors

**Bayesian updating:**
$$\underbrace{P(\theta\mid x)}_{\text{posterior}} = \frac{\overbrace{P(x\mid\theta)}^{\text{likelihood}}\;\overbrace{P(\theta)}^{\text{prior}}}{\underbrace{P(x)}_{\text{evidence}}} \propto P(x\mid\theta)P(\theta)$$

**Conjugate prior:** A prior distribution family that, when combined with a given likelihood, yields a posterior in the *same* family — making Bayesian updating analytically tractable (closed-form, no numerical integration needed).

**Beta-Binomial (the canonical example for A/B testing):** 
- Prior: θ ~ Beta(α, β) (belief about conversion rate).
- Likelihood: observe k successes out of n Bernoulli trials.
- Posterior: θ | data ~ **Beta(α+k, β+n−k)**.

Interpretation: α, β act as "pseudo-counts" of prior successes/failures; the posterior simply adds the observed successes/failures to these pseudo-counts.

**Worked example:** Prior belief on a website's conversion rate: Beta(2,2) (weakly informative, centered at 0.5). Observe 30 conversions out of 100 visitors. Posterior: Beta(2+30, 2+70) = Beta(32,72). Posterior mean = 32/(32+72) = 32/104 ≈ **0.3077** (very close to the raw MLE 0.30, since the prior is weak relative to n=100 data points).

**Normal-Normal (known variance) conjugacy:**
- Prior: μ ~ N(μ₀, σ₀²).
- Likelihood: x₁,…,xₙ iid N(μ, σ²) (σ² known).
- Posterior: μ | data ~ N(μ_post, σ_post²), where
$$\mu_{post} = \frac{\frac{\mu_0}{\sigma_0^2}+\frac{n\bar x}{\sigma^2}}{\frac{1}{\sigma_0^2}+\frac{n}{\sigma^2}}, \qquad \sigma_{post}^2 = \left(\frac{1}{\sigma_0^2}+\frac{n}{\sigma^2}\right)^{-1}$$
This is a **precision-weighted average** of the prior mean and the sample mean — a beautiful and frequently-asked result: more data (higher n) or lower likelihood variance shifts the posterior mean toward the data; a tighter prior (lower σ₀²) keeps it closer to the prior belief.

**Other common conjugate pairs (good to know exist):** Gamma prior for Poisson rate λ; Gamma prior for Exponential rate; Dirichlet prior for Multinomial probabilities (generalizes Beta-Binomial to k>2 categories); Inverse-Gamma prior for Normal variance (unknown σ²).

**Common pitfalls:**
- Forgetting the posterior mean is a *weighted average* pulled toward the prior when data is scarce, and toward the MLE as data grows (prior's influence vanishes asymptotically — Bayesian and frequentist estimates converge with enough data).
- Choosing an overly strong/informative prior that dominates the posterior even with substantial contrary data (a common real-world Bayesian pitfall in production A/B testing systems).
- Believing conjugacy is required for Bayesian inference — it's a convenience; non-conjugate models use MCMC/variational inference instead.

### Bayesian vs Frequentist Inference

| Aspect | Frequentist | Bayesian |
|---|---|---|
| Parameter view | Fixed, unknown constant | Random variable with a distribution |
| Probability interpretation | Long-run frequency over repeated trials | Degree of belief, can be updated |
| Uses prior info? | No | Yes, explicitly via prior distribution |
| Key output | Point estimate + p-value/CI | Full posterior distribution |
| Interval | Confidence interval (property of the *procedure*) | Credible interval (property of the parameter's belief distribution) |
| Sequential/peeking | Requires correction (multiple testing) | Naturally supports continuous monitoring (with proper priors) |
| Example question answered | "How often would this procedure be right?" | "Given this data, what do I believe about θ?" |

**Practical differences that come up in interviews:**
- Bayesian methods let you directly answer "What's the probability variant B is better than A?" — a question frequentist p-values *cannot* directly answer (p-values only quantify evidence against H₀ under repeated sampling).
- Bayesian A/B testing naturally handles continuous monitoring/peeking (with appropriate priors or methods like always-valid inference), whereas naive frequentist peeking inflates Type I error.
- Frequentist methods are often computationally simpler and don't require specifying a (potentially subjective) prior; Bayesian methods require prior elicitation, which can be a source of criticism/bias if done carelessly, but shine with small data or when incorporating domain expertise is valuable.
- As sample size grows, Bayesian posteriors (with reasonably weak priors) and frequentist estimates typically converge (Bernstein-von Mises theorem, informally).

### Credible Intervals vs Confidence Intervals

**Credible interval:** A range [a,b] such that P(a ≤ θ ≤ b | data) = C%, computed directly from the posterior distribution. This is a genuine probability statement about the parameter, given the observed data and the prior.

**Confidence interval:** A range constructed via a procedure that captures the true (fixed) parameter C% of the time *across repeated experiments* — it is **not** a probability statement about the parameter for any single observed interval.

**Key interview distinction:** "There's a 95% probability the true parameter is in this interval" is **valid language for a Bayesian credible interval**, but **technically incorrect for a frequentist confidence interval** (common trap — many practitioners casually (mis)use CI language as if it were a credible interval).

**Worked example:** Beta(32,72) posterior from above. A 95% credible interval can be computed from the 2.5th and 97.5th percentiles of Beta(32,72), e.g., approximately [0.222, 0.402] (exact values require the Beta quantile function) — and we can correctly say "there's a 95% probability the true conversion rate lies in this range, given our data and prior."

**When they roughly agree:** With large data and weak/flat priors, credible intervals and confidence intervals often produce numerically very similar ranges — but their *interpretation* remains philosophically distinct.

### Interview Questions

**Q1. Explain Bayesian updating and conjugate priors in your own words. Why are conjugate priors useful?**
A: Bayesian updating combines a prior belief with observed data (through the likelihood) to produce an updated belief (posterior), via posterior ∝ likelihood × prior. A conjugate prior is one where this multiplication yields a posterior in the same distributional family as the prior, giving closed-form updates (e.g., Beta prior + Binomial likelihood → Beta posterior) — this avoids numerical integration and makes sequential updating trivial (just update the pseudo-counts).

**Q2. Derive the posterior distribution for a Beta(α,β) prior with a Binomial(n,k) likelihood.**
A: Posterior ∝ [pᵏ(1−p)ⁿ⁻ᵏ] × [p^(α−1)(1−p)^(β−1)] = p^(α+k−1)(1−p)^(β+n−k−1), which is the kernel of a Beta(α+k, β+n−k) distribution. So posterior ~ Beta(α+k, β+n−k).

**Q3. What's the difference between a confidence interval and a credible interval?**
A: A confidence interval is a frequentist construct: a procedure that captures the true fixed parameter C% of the time over repeated sampling; a single realized interval either does or doesn't contain the parameter, and you can't attach a probability to it. A credible interval is a Bayesian construct: given the observed data (and prior), it's an actual probability statement — there's a C% probability the parameter lies in that range, conditional on the data.

**Q4. Explain the philosophical difference between Bayesian and frequentist views of probability.**
A: Frequentists treat probability as long-run frequency of repeatable events and parameters as fixed unknown constants (not random). Bayesians treat probability as a degree of belief that can apply to parameters themselves, updating that belief via observed data using Bayes' theorem, allowing direct probabilistic statements about hypotheses/parameters.

**Q5. In an A/B test, why might a Bayesian approach be preferred if the team wants to monitor results continuously ("peek") without inflating false-positive rates?**
A: Frequentist p-values assume a fixed, pre-specified sample size; repeatedly checking and stopping early when p<0.05 ("peeking") inflates the true Type I error rate well beyond the nominal α. Bayesian approaches instead track how the posterior probability of "B is better than A" evolves, and (with well-chosen priors or appropriately designed stopping rules) can support principled continuous monitoring — though naive Bayesian peeking without care can also have issues, so purpose-built methods (e.g., always-valid inference, sequential testing) are used in practice.

**Q6. As sample size grows very large, what typically happens to the influence of the prior on the posterior?**
A: The influence of the prior diminishes — the likelihood (data) dominates, and the posterior converges toward the MLE regardless of the (reasonable, non-degenerate) prior chosen. This is why Bayesian and frequentist point estimates tend to agree asymptotically.

**Q7 (derivation). Given a Normal-Normal conjugate setup with prior N(0, 1) and observing a single data point x=5 from N(μ, 4) (known variance 4), find the posterior mean.**
A: μ_post = (μ₀/σ₀² + nx̄/σ²)/(1/σ₀² + n/σ²) = (0/1 + (1)(5)/4)/(1/1 + 1/4) = 1.25/1.25 = **1.0**. This is a precision-weighted average of the prior mean (0) and the data (5): the prior precision (1/σ₀²=1) and the data's precision (n/σ²=1/4) set the weights (an 80/20 split toward the prior), so despite observing x=5, the relatively tight prior (variance 1) pulls the posterior mean strongly toward 0, landing at 1.0 rather than close to 5.

**Q8 (scenario). A colleague says, "We got p=0.03, so there's a 97% chance our hypothesis is true." How would you correct this, and how would a Bayesian frame the same question correctly?**
A: This misinterprets the p-value — p-values are computed *under H₀* and don't directly give P(H₁|data) or P(H₀|data). The frequentist correction: p=0.03 means "if H₀ were true, we'd see data this extreme 3% of the time" — nothing more. A Bayesian would instead compute the actual posterior probability P(H₁|data) using Bayes' theorem, which requires specifying priors on H₀ and H₁ and would generally give a different (and directly interpretable) number.

**Q9 (tricky). Two analysts use different (reasonable) priors on the same data and get different posteriors. Is this a flaw in Bayesian inference?**
A: Not necessarily a flaw — it reflects that Bayesian inference explicitly encodes and is influenced by prior beliefs, which is a feature (allows incorporating domain knowledge) as well as a critique (subjectivity). With more data, if both priors are non-degenerate (assign nonzero density everywhere plausible), the posteriors should converge toward each other and toward the likelihood-dominated estimate. Sensitivity analysis (checking how much results change under different reasonable priors) is standard practice to address this concern.

---

## Sampling and Experimental Design

*Relevance: Data Scientist (core — A/B testing is a top interview topic), ML Engineer (moderate — sampling bias affects training data quality), AI Engineer (moderate — evaluating and comparing model/prompt variants).*

### Sampling Methods

| Method | Description | Pros | Cons |
|---|---|---|---|
| **Simple random sampling** | Every individual has equal probability of selection | Unbiased, simple | May not represent small subgroups well |
| **Stratified sampling** | Divide population into homogeneous strata (e.g., by region/age), sample within each | Reduces variance, ensures subgroup representation | Requires knowing strata membership in advance |
| **Cluster sampling** | Divide population into clusters (e.g., geographic), randomly select whole clusters | Cost-efficient when population is naturally clustered | Higher variance if clusters are internally homogeneous but different from each other |
| **Systematic sampling** | Select every k-th individual from an ordered list | Simple to implement, spreads sample evenly | Risk of hidden periodicity in the list aligning with k |

**Sampling bias:** Occurs when the sampling method systematically favors certain outcomes/subgroups, so the sample is not representative of the population.

**Classic examples:**
- **Survivorship bias:** analyzing only "survivors" (e.g., successful companies, WWII planes that returned) ignoring those that failed/were lost, leading to wrong conclusions about what drives success.
- **Selection bias in A/B testing:** e.g., only measuring engagement among users who completed onboarding, ignoring those who dropped out due to the treatment itself.
- **Non-response bias:** survey respondents differ systematically from non-respondents.
- **Self-selection bias:** users who opt-in to a beta feature are not representative of the general population.

**Common pitfalls:**
- Confusing stratified sampling (proportional representation of known subgroups) with cluster sampling (sampling whole groups, often for logistical convenience, at potential cost of representativeness).
- Ignoring that systematic sampling can badly fail if the population list has periodic patterns matching the sampling interval.
- Believing a large sample size fixes sampling bias — it does not; bias is about *who* is sampled, not *how many*.

### A/B Testing Fundamentals

**Core design steps:**
1. Define a single primary metric (and guardrail metrics) tied to the business question.
2. State H₀ (no difference) and H₁, choose α (Type I error tolerance, usually 0.05) and desired power (usually 0.8).
3. Determine minimum detectable effect (MDE) — smallest effect size worth detecting.
4. Compute required sample size via power analysis.
5. Randomize users into control/treatment (ensuring no leakage/contamination between groups).
6. Run for a pre-determined duration/sample size; avoid early stopping based on interim significance.
7. Analyze with the pre-specified test; check for guardrail metric regressions.

**Sample size / power analysis formula (two-proportion z-test, simplified):**
$$n \approx \frac{(z_{\alpha/2}+z_\beta)^2 \cdot [p_1(1-p_1)+p_2(1-p_2)]}{(p_1-p_2)^2}$$
where p₁ is the baseline conversion rate, p₂ = p₁ + MDE is the target rate, z_{α/2} corresponds to significance level (1.96 for α=0.05), z_β corresponds to desired power (0.84 for 80% power).

**Worked example:** Baseline conversion p₁=10%, want to detect an absolute lift to p₂=12% (MDE=2pp), α=0.05 (two-tailed, z=1.96), power=80% (z_β=0.84).
$$n \approx \frac{(1.96+0.84)^2[0.10(0.9)+0.12(0.88)]}{(0.02)^2} = \frac{(2.8)^2 \times [0.09+0.1056]}{0.0004} = \frac{7.84 \times 0.1956}{0.0004} \approx \frac{1.534}{0.0004} \approx 3835$$
So roughly **~3,835 users per group** are needed — illustrating the inverse-square relationship between required sample size and MDE (halving the MDE quadruples the needed sample size).

**Key relationships in power analysis (memorize the qualitative directions):**

| Increase this | Effect on required sample size |
|---|---|
| Smaller MDE (subtler effect) | Sample size increases (∝ 1/MDE²) |
| Higher desired power | Sample size increases |
| Lower α (stricter significance) | Sample size increases |
| Higher baseline variance | Sample size increases |

**Novelty and primacy effects:**
- **Novelty effect:** users react positively to a change simply because it's new/different, inflating short-term metrics that fade over time — risk of overestimating a treatment's long-run effect if the test runs too short.
- **Primacy effect:** existing users initially resist/underperform on a new experience due to unfamiliarity (habit disruption), understating the treatment's true long-run effect.
- **Mitigation:** run tests long enough to let novelty/primacy wash out, or segment analysis by new vs. existing users, or hold out a "long-term holdback" group to measure durable effects.

**The peeking problem:** Continuously monitoring a test and stopping as soon as p<0.05 is observed inflates the actual Type I error rate far beyond the nominal α (each additional look is like running another hypothesis test without correction). Standard fixes: pre-register a fixed sample size/duration and don't stop early; or use formal sequential testing frameworks (e.g., alpha-spending functions, Bayesian always-valid inference, mSPRT) explicitly designed to allow continuous monitoring while controlling error rates.

**Other important A/B testing concepts:**
- **Network effects / interference:** in social or marketplace products, treatment can spill over to control users (e.g., a treated seller affects a control buyer) — violates the independence (SUTVA) assumption; mitigated via cluster-randomization (randomize by cluster/market instead of by individual user).
- **Sample Ratio Mismatch (SRM):** if the observed split between control/treatment deviates significantly from the intended ratio (e.g., 48/52 instead of 50/50), it signals a bug in randomization/logging — always check with a chi-square test before trusting results.
- **Novelty vs seasonality confounds:** always ensure the test runs across full weekly cycles (avoid day-of-week effects) and ideally captures any relevant seasonal patterns.

### Causal Inference Basics

**Correlation vs causation:** Correlation measures statistical association; causation implies that changing one variable *produces* a change in another. Correlation can arise from: (1) true causation, (2) reverse causation, (3) confounding (common cause), (4) coincidence/chance, (5) selection bias.

**Confounders:** A variable that influences both the treatment/exposure and the outcome, creating a spurious association if not controlled for.

**Worked example:** Ice cream sales and drowning deaths are positively correlated. Confounder: hot weather increases both ice cream sales and swimming (hence drowning risk) — no causal link between ice cream and drowning.

**Simpson's Paradox:** A trend that appears in aggregated data reverses (or disappears) when the data is broken down by a confounding subgroup.

**Worked numeric example (Simpson's Paradox — classic treatment comparison):**

| | Treatment A | Treatment B |
|---|---|---|
| Small stones: success rate | 93% (81/87) | 87% (234/270) |
| Large stones: success rate | 73% (192/263) | 69% (55/80) |
| **Overall success rate** | **78% (273/350)** | **83% (289/350)** |

Treatment A is better for *both* small stones and large stones individually, yet Treatment B looks better overall — because Treatment A was disproportionately used on harder (large-stone) cases. The confounder (stone size, correlated with which treatment was chosen) reverses the aggregate conclusion. Lesson: always check for lurking/confounding variables and consider stratified analysis, not just pooled aggregates.

**Propensity score matching (brief):** In observational (non-randomized) studies, directly comparing treated vs. untreated groups is biased because treatment assignment isn't random (confounded by covariates that influence both treatment choice and outcome). Propensity score matching estimates P(treatment=1 | covariates) for each unit (the "propensity score"), then matches treated units to untreated units with similar propensity scores, approximating what a randomized experiment would have looked like — reducing (but not eliminating, since it only addresses *observed* confounders) bias from confounding.

**Other causal inference tools worth knowing exist (brief mention):**
- **Randomized Controlled Trials (RCTs)/A-B tests:** the gold standard — randomization breaks the link between confounders and treatment assignment by construction.
- **Instrumental variables:** use a variable that affects treatment but not the outcome except through treatment, to estimate causal effects despite unobserved confounding.
- **Difference-in-differences:** compares the change over time in an outcome between a treated and untreated group, controlling for time-invariant confounders and common trends.
- **Regression discontinuity:** exploits a threshold-based treatment assignment rule to compare units just above/below the cutoff as a quasi-random experiment.

**Common pitfalls:**
- Assuming any observed correlation implies a causal mechanism — always ask "what could confound this?"
- Ignoring Simpson's Paradox risk when aggregating data across heterogeneous subgroups (e.g., combining data across very different user segments/markets).
- Believing propensity score matching fully solves confounding — it only adjusts for *measured* confounders; unmeasured confounding can still bias results (a key limitation to mention in interviews).

### Interview Questions

**Q1. What's the difference between stratified and cluster sampling? Give an example of when you'd use each.**
A: Stratified sampling divides the population into homogeneous subgroups (strata) and samples from *within each* stratum, ensuring representation (e.g., sampling proportionally by age group for a survey). Cluster sampling divides the population into heterogeneous clusters (often geographically) and randomly selects *entire clusters* to sample fully, mainly for cost/logistical efficiency (e.g., randomly selecting a few schools and surveying all students in them, rather than traveling to sample individuals from many schools).

**Q2. How would you calculate the required sample size for an A/B test?**
A: Specify: baseline conversion rate (p₁), minimum detectable effect (MDE, giving p₂), significance level α, and desired power (1−β). Plug into the two-proportion sample size formula: n ≈ (z_{α/2}+z_β)²[p₁(1−p₁)+p₂(1−p₂)]/(p₁−p₂)². Smaller MDE, higher power, or lower α all increase the required n.

**Q3. What is the "peeking problem" in A/B testing, and how do you avoid it?**
A: Continuously checking p-values during an ongoing test and stopping as soon as p<0.05 is observed inflates the true Type I error rate far above the nominal level, because each additional look is effectively an additional (uncorrected) hypothesis test. Avoid by fixing the sample size/duration in advance (power analysis) and not stopping early, or by using sequential testing methods (alpha-spending, always-valid p-values, Bayesian sequential monitoring) explicitly built to control error rates under continuous monitoring.

**Q4. Explain novelty effect and primacy effect in A/B testing, and how you'd detect/mitigate them.**
A: Novelty effect: users engage more with a change simply because it's new, inflating short-term metrics that decay over time. Primacy effect: existing users temporarily underperform due to disruption of established habits, understating long-run benefit. Detect by segmenting results over time (is the effect shrinking/growing across weeks?) and by new vs. existing users; mitigate by running the test long enough for the effect to stabilize, or using a long-term holdout group.

**Q5. What is Simpson's Paradox? Give a real example.**
A: A phenomenon where a trend present in aggregated data reverses or vanishes when the data is disaggregated by a confounding subgroup variable. Classic example: kidney stone treatment where Treatment A outperforms Treatment B within both small-stone and large-stone subgroups individually, but B looks better in the combined/aggregate data — because Treatment A was used disproportionately on harder (large-stone) cases.

**Q6. Why is correlation not sufficient evidence for causation? Name the main alternative explanations for an observed correlation.**
A: Correlation only measures statistical co-movement, not the underlying mechanism. Alternative explanations: (1) reverse causation (Y causes X, not X causes Y), (2) confounding (a third variable causes both), (3) coincidence/chance (especially in multiple comparisons), (4) selection bias in how the data was collected.

**Q7. What is a confounding variable? How does propensity score matching help address it in observational studies?**
A: A confounder influences both the treatment assignment and the outcome, creating a spurious association between them if uncontrolled. Propensity score matching estimates each unit's probability of receiving treatment given observed covariates, then matches treated and untreated units with similar propensity scores — approximating a randomized comparison with respect to *observed* covariates (though unobserved confounders remain a limitation).

**Q8. Your A/B test shows a 55/45 split between control and treatment instead of the intended 50/50. What should you check, and why does it matter?**
A: This is a potential Sample Ratio Mismatch (SRM) — check with a chi-square goodness-of-fit test against the expected 50/50 split. An SRM often indicates a bug in randomization, logging, or a differential exclusion/crash rate between arms, which can badly bias the metric comparison; results from a test with a significant SRM should not be trusted until the root cause is found and fixed.

**Q9 (scenario). You want to A/B test a new feature on a social network where users interact with each other (e.g., messaging). Why might standard user-level randomization be problematic, and what would you do instead?**
A: User-level randomization can violate the independence assumption (SUTVA) because treated users can interact with and influence control users (network interference/spillover), contaminating the control group's behavior and biasing the estimated treatment effect. Mitigation: cluster-randomize at a level that captures most interactions (e.g., by geographic market, or by graph-cluster/community), or use specialized designs like ego-network randomization to isolate spillover effects.

**Q10 (tricky). A company observes that stores with more employees have higher sales, and concludes hiring more staff increases sales. What's the likely flaw?**
A: Likely reverse causation and/or confounding: busier/higher-traffic stores (higher expected sales due to location, foot traffic) may simply need and hire more staff, rather than staff count driving sales. Without a controlled/randomized staffing experiment (or careful causal design like instrumental variables or difference-in-differences), this observational correlation cannot establish that hiring more staff *causes* higher sales.

**Q11 (scenario/derivation). Explain why halving the minimum detectable effect in an A/B test roughly quadruples the required sample size.**
A: In the sample size formula n ∝ (p₁−p₂)⁻² = MDE⁻², sample size scales with the *inverse square* of the effect size you want to detect. Halving MDE means dividing the denominator by 4 (since (MDE/2)² = MDE²/4), so n must roughly quadruple to maintain the same power — detecting subtler effects requires disproportionately more data.

**Q12 (brain-teaser). You're told two independently-run 95% confidence A/B tests both showed "no significant difference." Can you conclude the feature has no effect?**
A: No — "fail to reject H₀" is not the same as "H₀ is true" (absence of evidence isn't evidence of absence). The test may simply have been underpowered to detect the true effect size (Type II error), especially if the true effect is small relative to the MDE the test was powered for. You'd want to check the achieved power/CI width, and potentially combine studies (meta-analysis) or run a properly powered test before concluding no effect exists.

---

## Rapid-Fire Interview Q&A

1. **Q: What are the three axioms of probability?**
   A: Non-negativity (P(A)≥0), normalization (P(Ω)=1), countable additivity for disjoint events.

2. **Q: State Bayes' theorem.**
   A: P(A|B) = P(B|A)P(A)/P(B).

3. **Q: What does "i.i.d." mean?**
   A: Independent and identically distributed — samples are drawn independently from the same distribution.

4. **Q: What is the expected value of a fair 6-sided die?**
   A: 3.5, i.e., (1+2+3+4+5+6)/6.

5. **Q: Var(X) formula in terms of moments?**
   A: Var(X) = E[X²] − (E[X])².

6. **Q: What distribution models the number of trials until the first success?**
   A: Geometric distribution.

7. **Q: What distribution models rare event counts over a fixed interval?**
   A: Poisson distribution.

8. **Q: What's the key property that both Exponential and Geometric distributions share?**
   A: Memorylessness.

9. **Q: State the Central Limit Theorem in one sentence.**
   A: The sampling distribution of the mean of i.i.d. random variables approaches a Normal distribution as sample size grows, regardless of the population's original distribution shape (given finite variance).

10. **Q: Does zero correlation imply independence?**
    A: No — zero correlation only rules out *linear* association; non-linear dependence can still exist.

11. **Q: What's the unbiased estimator of population variance, and why n−1?**
    A: S² = Σ(xᵢ−x̄)²/(n−1); n−1 (Bessel's correction) accounts for the degree of freedom used up estimating x̄ from the same data, making E[S²]=σ².

12. **Q: Define p-value in one sentence.**
    A: The probability of observing data at least as extreme as what was observed, assuming the null hypothesis is true.

13. **Q: What is Type I error vs Type II error?**
    A: Type I = false positive (rejecting a true H₀); Type II = false negative (failing to reject a false H₀).

14. **Q: What increases statistical power?**
    A: Larger sample size, larger true effect size, lower variance, higher α.

15. **Q: MLE vs MAP — what's the key difference?**
    A: MAP incorporates a prior distribution on the parameter; MLE does not (equivalent to MAP with a flat/uninformative prior).

16. **Q: What is a conjugate prior?**
    A: A prior that, combined with a given likelihood, produces a posterior in the same distribution family (e.g., Beta prior + Binomial likelihood → Beta posterior).

17. **Q: Confidence interval vs credible interval — one-line distinction?**
    A: A confidence interval is a property of the estimation *procedure* over repeated sampling (frequentist); a credible interval is a direct probability statement about the parameter given the data (Bayesian).

18. **Q: What does Bonferroni correction do?**
    A: Divides the significance threshold by the number of tests (α/m) to control the family-wise error rate under multiple comparisons.

19. **Q: What does Benjamini-Hochberg control, and how does it differ from Bonferroni?**
    A: Controls the False Discovery Rate (expected proportion of false positives among rejected hypotheses); less conservative than Bonferroni, yielding more power for large numbers of tests.

20. **Q: What is Simpson's Paradox?**
    A: A trend seen in aggregated data reverses when the data is split by a confounding subgroup variable.

21. **Q: Name a non-parametric alternative to the two-sample t-test.**
    A: Mann-Whitney U test.

22. **Q: What is the "peeking problem" in A/B testing?**
    A: Repeatedly checking significance during an ongoing experiment and stopping early inflates the true Type I error rate beyond the nominal α.

23. **Q: What is a Sample Ratio Mismatch (SRM)?**
    A: When the observed traffic split between test arms deviates significantly from the intended allocation, usually signaling a randomization/logging bug.

24. **Q: What does the Law of Large Numbers guarantee?**
    A: The sample mean converges to the true population mean as sample size grows to infinity.

25. **Q: What is the Gambler's Fallacy?**
    A: The mistaken belief that past independent outcomes (e.g., a coin-flip streak) affect the probability of future independent outcomes.

26. **Q: What test would you use to compare two categorical variables for association?**
    A: Chi-square test of independence.

27. **Q: What is statistical power formally?**
    A: 1 − β, the probability of correctly rejecting a false null hypothesis (detecting a true effect).

28. **Q: What's the relationship between Gamma and Exponential distributions?**
    A: Gamma(k=1, θ) = Exponential(1/θ); Gamma is the sum of k i.i.d. Exponential random variables.

29. **Q: Why is the Beta distribution commonly used in Bayesian A/B testing?**
    A: It's the conjugate prior for a Bernoulli/Binomial likelihood and is naturally bounded on [0,1], matching the support of a probability/conversion rate parameter.

30. **Q: What's a propensity score used for?**
    A: Estimating the probability of receiving treatment given covariates, to match treated and untreated units in observational studies and reduce confounding bias.

31. **Q: MAP with a Gaussian prior on model weights corresponds to which regularization?**
    A: L2 regularization (Ridge).

32. **Q: MAP with a Laplace prior on model weights corresponds to which regularization?**
    A: L1 regularization (Lasso).

33. **Q: What's the difference between statistical significance and practical significance?**
    A: Statistical significance means the observed effect is unlikely due to chance (low p-value); practical significance means the effect size is large enough to matter for real-world decisions — large samples can make trivially small effects statistically significant.

34. **Q: What is Cov(X,Y) in terms of expectations?**
    A: Cov(X,Y) = E[XY] − E[X]E[Y].

35. **Q: When is a paired t-test preferred over an independent two-sample t-test?**
    A: When observations are naturally matched/linked (e.g., before/after on the same subjects), reducing variance and increasing power by controlling for between-subject differences.

36. **Q: State Markov's inequality in one sentence.**
    A: For a non-negative random variable X, P(X≥a) ≤ E[X]/a — a distribution-free tail bound using only the mean.

37. **Q: State Chebyshev's inequality in one sentence.**
    A: P(|X−μ|≥kσ) ≤ 1/k² — a distribution-free bound on how far a random variable can stray from its mean, using only the variance.

38. **Q: What's the relationship between a Poisson process's event counts and its interarrival times?**
    A: Counts in an interval of length t are Poisson(λt); the interarrival times between consecutive events are i.i.d. Exponential(λ).

39. **Q: What equation defines a Markov chain's stationary distribution?**
    A: πP = π, where π is a row vector of state probabilities summing to 1 and P is the transition matrix.

40. **Q: In one sentence, what does bootstrap resampling do?**
    A: Resamples the observed data with replacement many times to empirically approximate the sampling distribution of a statistic.

41. **Q: In one sentence, what does a permutation test do?**
    A: Repeatedly shuffles group labels to build the null distribution of a test statistic under "no effect," then compares the observed statistic to it.

42. **Q: What does Cohen's d measure?**
    A: The standardized mean difference between two groups (mean difference divided by pooled standard deviation) — an effect-size measure independent of sample size.

43. **Q: What's the general MLE rule for any exponential-family distribution?**
    A: Set the model's expected sufficient statistic equal to the observed average sufficient statistic (moment matching via ∇A(η)).

44. **Q: What is right-censoring in survival analysis?**
    A: When a subject is known to have survived at least until the end of observation, but its exact event time is unknown.

