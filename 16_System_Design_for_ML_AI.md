# System Design for ML/AI Systems

System design interviews for ML roles test whether a candidate can turn an ambiguous product goal into a working, scalable, maintainable pipeline — not just pick an algorithm. **Data Scientists** are expected to justify metric choices, evaluation design, and offline/online experiment strategy. **Machine Learning Engineers** are expected to own the production architecture: feature pipelines, training/serving infrastructure, latency, and monitoring. **AI Engineers** (LLM/GenAI-focused) are expected to design retrieval-augmented generation, agentic, and guardrail systems on top of foundation models, with sharp instincts about latency, cost, and safety. This topic is one of the highest-leverage areas at the senior/staff level because it signals end-to-end ownership — the ability to reason about a system from a cold user request all the way to a dashboard alert three months after launch.

## Table of Contents

1. [A Framework for ML System Design Interviews](#a-framework-for-ml-system-design-interviews)
2. [Data and Feature Pipeline Design](#data-and-feature-pipeline-design)
3. [Case Study: Design a Recommendation System](#case-study-design-a-recommendation-system)
4. [Case Study: Design a Short-Form Video Feed Ranking System](#case-study-design-a-short-form-video-feed-ranking-system)
5. [Case Study: Design a Search Ranking System](#case-study-design-a-search-ranking-system)
6. [Case Study: Design an Autocomplete and Search-Suggestion System](#case-study-design-an-autocomplete-and-search-suggestion-system)
7. [Case Study: Design a Fraud/Anomaly Detection System](#case-study-design-a-fraudanomaly-detection-system)
8. [Case Study: Design a Chatbot / RAG-based Q&A System](#case-study-design-a-chatbot--rag-based-qa-system)
9. [Case Study: Design a Content Moderation / Toxicity Detection System](#case-study-design-a-content-moderation--toxicity-detection-system)
10. [Case Study: Design an Ad Click-Through-Rate Prediction System](#case-study-design-an-ad-click-through-rate-prediction-system)
11. [Case Study: Design an ML Platform from Scratch](#case-study-design-an-ml-platform-from-scratch)
12. [Rapid-Fire Interview Q&A](#rapid-fire-interview-qa)

---

## A Framework for ML System Design Interviews

### Why a Framework Matters

ML system design questions are deliberately underspecified ("design a system that recommends videos to users"). Interviewers are grading your *process* as much as your final answer: do you ask clarifying questions, do you quantify constraints before designing, do you make and justify tradeoffs out loud, do you circle back to earlier decisions when new constraints appear. A memorized architecture without the surrounding reasoning reads as rehearsed and scores poorly at senior levels.

### The Structured Approach (8-Step Loop)

```
 1. CLARIFY REQUIREMENTS  -->  2. DEFINE METRICS  -->  3. DATA
        (functional +              (offline/online         (sources, labels,
         non-functional)            success metrics)         freshness, scale)
        |                                                          |
        v                                                          v
 8. TRADEOFFS/ITERATE <-- 7. MONITORING <-- 6. SERVING <-- 5. TRAINING <-- 4. FEATURES
   (revisit any step        (drift, alerts,   (latency,      (pipeline,      (online vs
    as constraints           feedback loop)    scaling,       retraining      offline,
    change)                                    caching)       cadence)        skew)
```

Treat this as a loop, not a line: senior candidates explicitly say "given the 50ms latency budget we just established, let's revisit the model choice in step 4" — that back-reference is what differentiates a senior signal from a junior one.

### Step 1 — Clarifying Requirements

**Functional requirements** — what does the system actually do, for whom, in what context?

| Question | Why it matters |
|---|---|
| Who are the users / consumers of this system (end user, another ML system, an internal team)? | Determines API contract and latency tolerance |
| What is the core prediction/decision task (classification, ranking, generation, regression)? | Shapes model family and evaluation metric |
| What does a "good" output look like? Give 2-3 examples. | Prevents solving the wrong problem |
| What is explicitly out of scope? | Bounds interview time and signals prioritization skill |
| Is this a new system or an extension/re-architecture of an existing one? | Changes migration and backward-compatibility concerns |

**Non-functional requirements checklist** — this is the list senior interviewers listen for explicitly:

| Dimension | Key questions | Typical senior answer pattern |
|---|---|---|
| **Latency budget** | p50/p95/p99 end-to-end? Synchronous or async? | "150ms p99 total, meaning ~30ms retrieval, ~80ms model, ~40ms overhead" |
| **Throughput / QPS** | Peak vs average QPS? Regional distribution? | "10K QPS average, 5x spike on sale days — must design for burst" |
| **Scale** | # users, # items, data volume/day, model size? | "500M users, 100M items → embedding store must be approximate NN, not exact" |
| **Data freshness** | How stale can features/model be? | "Fraud needs sub-second features; content recs tolerate hourly batch" |
| **Cold start** | New users/items/queries with no history? | Explicitly plan a fallback path |
| **Accuracy targets** | Business-relevant metric and floor (e.g., AUC ≥ 0.85, recall@10 ≥ 0.4)? | Ties model choice to a measurable target, not vibes |
| **Fairness / bias** | Protected groups, demographic parity constraints? | Especially critical in lending, hiring, moderation, medical |
| **Explainability** | Regulatory need to explain individual decisions? | Rules out black-box models in some domains (credit, health) |
| **Cost constraints** | $/1000 predictions, GPU budget, storage budget? | LLM systems: token cost dominates; recsys: storage/serving cost dominates |
| **Availability / reliability** | SLA (99.9% vs 99.99%)? Fallback on model failure? | Always have a non-ML fallback (cached result, rule-based heuristic) |
| **Privacy / compliance** | PII handling, GDPR "right to be forgotten," data residency? | Shapes storage architecture and retraining pipeline |
| **Consistency of feedback loop** | Does the model's own output influence future training data (feedback loops, exposure bias)? | Recommendation and fraud systems especially prone to this |

**Pitfall:** jumping straight to "I'd use a transformer" before establishing scale/latency. Interviewers often penalize candidates who never ask about QPS or latency — it suggests they've never shipped anything.

### Step 2 — Defining Success Metrics

Separate three metric layers explicitly:

| Layer | Purpose | Examples |
|---|---|---|
| **Model/offline metrics** | Measure model quality on held-out data | AUC, log loss, RMSE, precision/recall@k, NDCG, BLEU/ROUGE, perplexity |
| **Online/business metrics** | Measure real-world impact | CTR, conversion rate, revenue/user, session length, task completion rate, false-positive review cost |
| **System/operational metrics** | Measure engineering health | p99 latency, throughput, error rate, model staleness, feature pipeline lag, cost/prediction |

**Framework — "North Star + Guardrails":** pick one north-star metric the model optimizes for, and 2-4 guardrail metrics that must not regress (e.g., optimize engagement, but guardrail on diversity, latency, and complaint rate). State this explicitly in every case study below.

### Step 3 — High-Level Architecture Sketch

Before diving into any single component, draw the generic ML system skeleton and annotate it with the constraints from Step 1:

```
             +-------------------+
 Raw Events  |   Data Ingestion  |  (Kafka/Kinesis, batch ETL, CDC)
 ----------> |   & Logging       |
             +---------+---------+
                       |
                       v
             +-------------------+        +--------------------+
             |  Feature Store     |<------>|  Offline Feature   |
             |  (online + offline)|        |  Pipeline (Spark/  |
             +---------+---------+        |  Airflow)          |
                       |                   +--------------------+
                       v
             +-------------------+
             |  Training Pipeline |----> Model Registry ----> Approval/Eval Gate
             +-------------------+
                       |
                       v
             +-------------------+       +-------------------+
 Request --->|  Serving Layer     |<----->|  Online Feature    |
             |  (API, batch, or   |       |  Store (low-latency)|
             |  streaming infer.) |       +-------------------+
             +---------+---------+
                       |
                       v
             +-------------------+
             |  Monitoring &      | ---> Alerts, dashboards
             |  Feedback Capture  | ---> Retraining trigger
             +-------------------+
```

Use this as the "spine" and drill into whichever box the interviewer steers toward.

### Step 4 — Discussing Tradeoffs Explicitly

Senior-level answers make tradeoffs *visible*, not implicit. Use a simple tradeoff table any time you choose between two approaches:

| Option A | Option B | When A wins | When B wins |
|---|---|---|---|
| Simple/interpretable model | Complex/high-capacity model | Low data, need explainability, tight latency | High data volume, latency budget allows, marginal accuracy matters commercially |
| Batch precompute | Real-time compute | Predictable request patterns, cost sensitive | Highly personalized/context-dependent, freshness critical |
| Single-stage model | Multi-stage (candidate gen + ranking) | Small candidate space | Huge item/action space (millions+) |

### Interview Questions

**Q1: How do you decide what's in scope in the first two minutes of an ambiguous prompt like "design Netflix's recommendation system"?**
A: State an assumed scope out loud ("I'll focus on the homepage row recommendations for logged-in users, not search or cold-start onboarding") and ask the interviewer to correct it. This converts an open-ended prompt into a boundable problem in seconds.

**Q2: What's the difference between a latency budget and a latency SLA?**
A: The budget is an internal engineering allocation across sub-components (e.g., 20ms retrieval + 60ms model + 20ms overhead = 100ms) that you design to; the SLA is the external commitment (e.g., p99 < 150ms) that includes margin for network/variance on top of the budget.

**Q3: Why do interviewers care about QPS before model architecture?**
A: QPS determines whether you can afford a heavy model synchronously, must precompute predictions in batch, or need caching/approximate methods (e.g., ANN instead of exact kNN). Model choice without a load estimate is unconstrained and often unrealistic.

**Q4: How would you handle a case where the offline metric improves but you expect the online metric to regress?**
A: Flag the mismatch as a hypothesis before shipping (e.g., "higher accuracy but lower diversity → engagement might drop") and propose an A/B test with the business metric as the primary readout and offline metric only as a leading indicator.

**Q5: What's a "north star + guardrail" metric framework and why use it?**
A: One primary optimization target avoids diffused effort across too many metrics; guardrails (latency, diversity, complaint rate, fairness) prevent the optimizer from gaming the north star at the expense of things users/business actually also care about.

**Q6: Give an example where a non-functional requirement completely changes the model choice.**
A: A fraud system needing a sub-50ms decision at authorization time rules out heavy ensemble/deep models requiring feature joins across multiple slow stores; you'd choose a lightweight gradient-boosted model with precomputed aggregate features instead.

**Q7: How do you incorporate fairness requirements into a design without a dedicated fairness metric being given?**
A: Ask which protected attributes are legally/ethically relevant in this domain, propose slicing the eval set by those attributes, and report parity metrics (e.g., equal opportunity difference) alongside aggregate metrics as a guardrail.

**Q8: What's the risk of designing the system as a straight pipeline instead of a loop?**
A: You miss feedback effects — e.g., a recommender that only shows popular items generates training data biased toward those items, reinforcing popularity bias over retrain cycles (feedback loop/exposure bias).

**Q9: How would you size infrastructure for "cold start day one" vs "steady state at scale"?**
A: Explicitly design two operating points: MVP with simpler heuristics/rules for launch validation, and a scaling plan (sharding, caching, approximate retrieval) triggered by defined load thresholds — mention this proactively to show forward planning.

**Q10: When should you propose a non-ML fallback in your architecture?**
A: Always, for anything customer-facing or safety-critical — cached last-good response, rule-based heuristic, or "most popular" default when the model service times out or its confidence is low, to protect availability and avoid embarrassing failures.

**Q11: How do you handle a request for "the most accurate model" when interviewer gives no other constraints?**
A: Push back: "accurate" needs a metric and a cost context — most accurate on offline AUC may violate a latency budget in production; propose 2-3 candidate operating points (fast/cheap vs slow/accurate) and let stated constraints pick one.

**Q12: What signals seniority in how you present tradeoffs verbally?**
A: Stating both sides of a tradeoff with a concrete "when X wins vs when Y wins" rule, rather than declaring a single "best" choice, and tying the decision back to the requirements gathered in step 1.

---

## Data and Feature Pipeline Design

### Designing Data Collection & Labeling Pipelines

**Collection sources:**

| Source type | Examples | Characteristics |
|---|---|---|
| Implicit behavioral logs | Clicks, dwell time, scroll depth, purchases | High volume, noisy, biased by UI position/exposure |
| Explicit feedback | Ratings, thumbs up/down, survey | Low volume, higher signal, response bias |
| Human annotation | Labeled toxicity, relevance judgments | Expensive, slow, needs QA, guideline drift |
| Weak/programmatic labels | Heuristics, distant supervision, LLM-as-labeler | Cheap, scalable, noisier — combine with a small gold set |
| Third-party/purchased data | Demographic enrichment, external fraud lists | Licensing/compliance overhead |

**Labeling pipeline framework:**

```
Raw sample --> Sampling strategy --> Task queue (annotation tool)
                                          |
                                          v
                              Multiple annotators + adjudication
                                          |
                                          v
                          Inter-annotator agreement check (Cohen's kappa)
                                          |
                             pass  <------+------>  fail (relabel / refine guidelines)
                                          |
                                          v
                                  Gold-labeled dataset
```

**Handling label noise:**

| Technique | Mechanism | Tradeoff |
|---|---|---|
| Multiple annotators + majority vote | Reduces individual annotator noise | Cost multiplies by # annotators |
| Confidence-weighted loss | Down-weight low-agreement samples during training | Requires per-sample agreement scores |
| Label smoothing | Soften one-hot targets to reduce overfitting to noisy labels | Small, may not fix systematic bias |
| Noise-robust loss functions (e.g., generalized cross-entropy) | Explicit statistical modeling of noise | More complex to tune |
| Active learning to re-review uncertain/borderline cases | Focuses expensive review effort where it matters most | Requires a working model in the loop already |

**Handling delayed labels** (e.g., "did this loan default," "was this transaction actually fraud," "did the user churn"):

- **Label maturation window**: define a fixed delay (e.g., 90 days) after which a label is considered final; train only on matured examples, and track "labeling lag" as a monitored metric.
- **Proxy/early labels**: use a fast, noisy proxy signal (e.g., chargeback flag within 24h) to train a fast-reacting model, and a slower model retrained on mature labels for calibration.
- **Snapshotting features at prediction time**: always store the feature vector *as it existed at prediction time*, not recomputed later, to avoid leaking future information into the joined label.
- **Sample-and-hold correction (survivorship bias)**: for fraud/spam, flagged-and-blocked cases never generate a "did they actually commit fraud" ground truth — occasionally let a small random sample through to measure true label distribution (unbiased evaluation).

### Online vs Offline Feature Computation

| Property | Batch/Offline | Streaming/Near-real-time | Online/On-demand |
|---|---|---|---|
| Computed by | Spark/Hive nightly jobs | Flink/Kafka Streams | Request-time service call |
| Freshness | Hours to a day stale | Seconds to minutes stale | Milliseconds stale |
| Example features | 30-day purchase count, user embedding from last retrain | Rolling 5-min transaction count, session click count | Current cart contents, real-time inventory |
| Cost | Cheap, high throughput | Moderate, needs stream infra | Expensive per request, adds request latency |
| Failure mode | Stale features if job fails silently | Backpressure/lag under load spikes | Timeout risk blowing the latency budget |

**Feature Store Architecture (the standard senior-level answer):**

```
                     +-----------------------+
                     |     Feature Registry   |  (feature definitions, ownership, schema)
                     +-----------+-----------+
                                 |
        +------------------------+------------------------+
        |                                                  |
        v                                                  v
+----------------+                               +--------------------+
| Offline Store   |  (Hive/BigQuery/Parquet)      | Online Store        |  (Redis/DynamoDB/
| - training      |------- materialize ---------->| - low-latency serve |   Cassandra)
| - backfill      |                               | - point lookups     |
+----------------+                               +--------------------+
        ^                                                  ^
        |                                                  |
   Batch pipeline                                    Streaming pipeline
   (Spark/Airflow)                                   (Flink/Kafka Streams)
```

The **same transformation logic** must generate both the offline (training) and online (serving) values — this is the single most important design principle in this section.

### Avoiding Training-Serving Skew

Training-serving skew is the #1 silent production-accuracy killer in ML systems. Root causes and mitigations:

| Root cause | Example | Mitigation |
|---|---|---|
| Different code paths for feature computation | Python feature logic in a notebook vs Java logic in the serving service | Use a **shared feature transformation library**, or a feature store with unified definitions compiled once |
| Time-travel/leakage | Training joins today's feature value against yesterday's label | Point-in-time correct joins; store feature snapshots keyed by event time |
| Missing-value handling divergence | Training fills NaN with column mean; serving fills with 0 | Encode the imputation logic itself as a versioned, shared artifact |
| Distribution shift between train window and serving time | Model trained on last year's holiday season traffic | Continuous monitoring + scheduled retraining cadence |
| Feature not available at serving time | Feature engineered from data only available post-hoc (e.g., "did the user click," used as a feature to predict clicks) | Explicit "what is known at t=0" audit for every feature before use |

**Framework: the "Point-in-Time Correctness" check** — for every feature, ask: *"If I froze the system at the exact moment of prediction, would this exact value have been available?"* If the answer requires future information, it's leakage.

### Handling Data at Scale

**Sharding strategies:**

| Shard key | Use case | Risk |
|---|---|---|
| User ID | Per-user personalization features | Hot users (celebrities) create skewed shards |
| Item ID | Item-level aggregate stats | Popular items create hot shards |
| Time-based | Log storage, time-series features | Query patterns often need cross-shard scans |
| Geographic | Latency-sensitive regional serving | Uneven regional traffic |

**Sampling strategies for training data at scale:**

| Strategy | Mechanism | When to use |
|---|---|---|
| Random uniform sampling | Simple i.i.d. sample of logs | Baseline, class-balanced problems |
| Stratified sampling | Preserve class/segment ratios | Multi-segment products (e.g., per-country models) |
| Negative downsampling + reweighting | Keep all positives, sample negatives, correct via calibration offset | Extreme class imbalance (CTR, fraud) |
| Reservoir sampling | Fixed-size sample from unbounded stream | Streaming pipelines without full materialization |
| Importance sampling | Weight samples by how "surprising"/high-loss they are | Active learning, hard example mining |
| Time-windowed sampling | Only recent N days, with decay weighting | Non-stationary environments (trends, seasonality) |

**Pitfalls:**
- Sampling logs *after* a ranking/filtering step introduces selection bias (e.g., only training on items that were already shown, biasing the model to reinforce the existing ranker).
- Deduplicating too aggressively can remove legitimate repeated signal (e.g., a fraud pattern that recurs).
- Sharding by a skewed key without a hot-key mitigation (consistent hashing + virtual nodes, or a dedicated hot-shard cache) causes p99 latency cliffs.

### Interview Questions

**Q1: What's the difference between training-serving skew and concept drift?**
A: Skew is an engineering bug — same "true" feature computed two different ways in two code paths. Drift is a statistical reality — the true relationship between features and label changes over time. Skew is fixed by unifying pipelines; drift is fixed by retraining/monitoring.

**Q2: How would you detect training-serving skew in production without a labeled dataset yet?**
A: Log the feature vector actually used at serving time, and separately recompute the same features offline for the same requests; diff the two distributions (e.g., via population stability index or basic summary stats) to catch divergence before waiting for label-based accuracy drops.

**Q3: Why is point-in-time-correct joining hard at scale, and what's a common way to implement it?**
A: Naive joins on user/item ID pull the *latest* feature value regardless of event time, leaking future info. Implementations timestamp every feature write and use an as-of / point-in-time join (e.g., Spark's asof join or feature-store-native support) keyed on event time, not wall-clock latest.

**Q4: How do you decide the label maturation window for a delayed-label problem?**
A: Empirically plot cumulative % of eventual labels resolved vs. days elapsed; choose a window covering e.g. 95% of eventual label flips, balancing model freshness (shorter window trains faster) against label completeness.

**Q5: What's "sample-and-hold" and why do fraud/spam systems need it?**
A: Deliberately letting through a small random % of flagged cases (instead of blocking 100%) to observe true outcomes, because blocking removes the ability to know whether a blocked case was truly fraud — without this, your evaluation set is entirely conditioned on your own model's past decisions.

**Q6: When would you choose streaming feature computation over pure batch, given the added infra cost?**
A: When the feature's predictive value decays within the batch refresh window — e.g., "transactions in the last 5 minutes" is nearly useless if only refreshed nightly, so the added Flink/Kafka Streams infra cost is justified by the freshness requirement from Step 1 of the design framework.

**Q7: How do you handle a categorical feature whose cardinality grows unboundedly over time (e.g., new item IDs)?**
A: Hashing trick (feature hashing) to bound dimensionality, or maintain an embedding table with an out-of-vocabulary bucket and periodic re-training/expansion; monitor OOV rate as a drift signal.

**Q8: What's a practical way to keep a feature store's online and offline values consistent without duplicating code?**
A: Define transformations once in a framework (e.g., a DSL or shared UDF library) that compiles to both the batch engine (Spark) and the streaming/online engine, so the same logic executes in both places instead of being reimplemented per environment.

**Q9: Your negative-sampling ratio for a CTR model changed between two training runs — what breaks, and how do you fix it?**
A: Predicted probabilities become miscalibrated relative to the true positive rate (since negatives were downsampled). Fix by recalibrating with a known correction formula: `p_calibrated = p / (p + (1-p)/w)` where `w` is the negative sampling rate, or refit a calibration layer (Platt scaling/isotonic regression) post-hoc.

**Q10: How would you design a labeling pipeline for a task where human annotators frequently disagree?**
A: Route disagreement cases to a senior/adjudicator tier, track inter-annotator agreement (Cohen's/Fleiss' kappa) per annotator and per task category, retrain/clarify guidelines when agreement drops, and consider modeling label uncertainty directly (soft labels) rather than forcing a single hard label.

**Q11: What's the danger of training on data sampled only from users who opted into a feature?**
A: Selection bias — the model learns a skewed population (opt-in users often differ systematically from the general population), producing poor generalization once the feature rolls out broadly; mitigate with a randomized holdout/pilot cohort instead of pure opt-in sampling.

**Q12: How do you scale a feature pipeline when a single "hot" entity (e.g., a viral item) dominates traffic?**
A: Detect hot keys via traffic monitoring, apply request coalescing/caching for that specific key, or shard the hot key's data across multiple partitions with an aggregation step, rather than letting one partition absorb disproportionate load.

---

## Case Study: Design a Recommendation System

### Problem Framing

*"Design a system that recommends items (videos/products/posts) to users on a homepage feed."*

**Clarify:** logged-in vs anonymous users, feed vs email vs push, personalization vs editorial mix, business goal (engagement, revenue, retention), scale (users, items), latency (typically <200ms for a page load).

**North star metric:** e.g., session engagement (watch time / clicks / purchases). **Guardrails:** diversity, latency, fairness across content creators, complaint rate.

### Two-Stage Architecture: Candidate Generation + Ranking

Recommending over tens of millions of items in real time is infeasible with a single heavy model scoring every item per request — hence the near-universal two-stage design:

```
                         User request (user_id, context)
                                    |
                                    v
                     +-----------------------------+
                     |     CANDIDATE GENERATION     |   goal: recall, cheap, fast
                     |  (recall ~ hundreds-thousands |
                     |   from millions/billions)     |
                     +---------------+---------------+
                                     |
                +--------------------+--------------------+
                |                    |                     |
        Collaborative           Content-based        Popularity/
        filtering /             embedding             trending
        embedding ANN           similarity             heuristics
                |                    |                     |
                +--------------------+--------------------+
                                     |
                                     v
                     +-----------------------------+
                     |          RANKING             |   goal: precision, personalized
                     |  (score ~hundreds -> top-N)   |   score, heavier model
                     +---------------+---------------+
                                     |
                                     v
                     +-----------------------------+
                     |   RE-RANKING / BUSINESS LOGIC|   diversity, freshness,
                     |   (dedup, diversity, ads mix) |   policy filters, ad slots
                     +---------------+---------------+
                                     |
                                     v
                              Final feed returned
```

| Stage | Goal | Latency budget (typical) | Model complexity |
|---|---|---|---|
| Candidate generation | High recall, cheap to run over huge corpus | ~10-30ms | Simple embeddings + ANN, or matrix factorization |
| Ranking | High precision, personalized fine score | ~30-80ms | GBDT, deep neural net (wide & deep, DeepFM, two-tower) |
| Re-ranking | Business constraints, diversity | ~5-15ms | Rule-based / lightweight MMR-style re-ranking |

### Candidate Generation: Approaches

| Approach | Mechanism | Pros | Cons |
|---|---|---|---|
| **Collaborative filtering (CF)** | User-item interaction matrix factorization (e.g., ALS) or neighborhood-based | Captures latent taste patterns without content features | Cold-start for new users/items; needs interaction history |
| **Content-based filtering** | Similarity between item content embeddings (text/image/metadata) and user's historical item embeddings | Works for new items immediately (no interaction history needed) | Limited serendipity; needs good content features |
| **Embedding-based retrieval (two-tower model)** | Separate user-tower and item-tower neural nets mapping to a shared embedding space; retrieve via approximate nearest neighbor (ANN) | Scales to huge catalogs; captures complex non-linear similarity | Requires ANN infra (FAISS, ScaNN, HNSW); retraining/re-indexing needed as embeddings drift |
| **Hybrid** | Combine CF + content-based + popularity, often via a weighted blend or multiple candidate sources merged before ranking | Balances coverage and cold-start handling | More moving parts, harder to debug which source contributed a candidate |

**Two-tower architecture (worked detail):**

```
   User features                         Item features
 (history, demo,                      (metadata, content
  context)                             embeddings, stats)
       |                                      |
       v                                      v
  +----------+                          +----------+
  | User     |                          | Item     |
  | Tower    |                          | Tower    |
  | (DNN)    |                          | (DNN)    |
  +----+-----+                          +-----+----+
       |                                      |
       v                                      v
  user_embedding (d-dim)             item_embedding (d-dim)
       |                                      |
       +------------------+-------------------+
                          |
                dot product / cosine
                (trained with in-batch
                 or sampled softmax loss)
```

At serving time, item embeddings are precomputed and indexed in an ANN structure; only the user embedding is computed at request time and used to query the index — this is what makes sub-30ms retrieval over billions of items feasible.

### Cold-Start Problem

| Cold-start type | Symptom | Mitigation strategies |
|---|---|---|
| New user | No interaction history for CF | Ask onboarding preferences; use demographic/contextual priors; popularity-based defaults; content-based on any early signal (search query, first click) |
| New item | No interaction history for CF or embeddings trained on interactions | Content-based candidate generation from metadata; exploration boost/quota so new items get impressions; use a separate "explore" bucket of traffic |
| New user AND new item (marketplace launch) | Both sides cold | Editorial/heuristic bootstrap; rapid A/B exploration; transfer learning from a similar existing market/vertical |

### Exploration Strategies (Bandits)

Pure exploitation (always show the highest-predicted-score item) causes a feedback loop: items never shown never accumulate data, so their true value is never learned (a "rich get richer" bias).

| Strategy | Mechanism | Pros | Cons |
|---|---|---|---|
| ε-greedy | With probability ε, show a random/under-explored item instead of the top-ranked one | Simple to implement | Wastes exploration budget uniformly, even on obviously bad items |
| Upper Confidence Bound (UCB) | Rank by score + a bonus proportional to uncertainty (fewer impressions → higher bonus) | Explores efficiently, more shown to promising uncertain items | Needs a well-calibrated uncertainty estimate |
| Thompson Sampling | Sample from the posterior distribution of each item's value; rank by the sample | Naturally balances explore/exploit; strong empirical performance | Requires maintaining a posterior (e.g., Beta distribution for CTR) |
| Contextual bandits (e.g., LinUCB) | Incorporate context (user/item features) into the exploration decision, not just item-level stats | Personalizes the explore/exploit tradeoff | More complex to implement and evaluate offline |

**Framework tip:** state that exploration is typically confined to a small, controlled slice of traffic (e.g., 2-5%) or a dedicated slot in the feed, so exploration cost is bounded and measurable.

### Ranking Model: Features and Position Bias

**Typical ranking feature groups:**

| Group | Examples |
|---|---|
| User features | Demographics, long-term taste embedding, recent activity, device/context |
| Item features | Content embedding, category, freshness, creator quality score, popularity |
| User-item interaction features | Past interactions with this item/creator, similarity of item to user's history |
| Context features | Time of day, device, page/slot position (used carefully — see below), session state |
| Cross features | User segment x item category, historical CTR of similar user-item pairs |

**Position bias:** items shown in higher slots get more clicks *regardless of relevance*, purely due to position. If the ranking model is trained on raw click logs, it partly learns "position predicts click" rather than "relevance predicts click," reinforcing whatever the current ranker already does.

Mitigations:

| Technique | Mechanism |
|---|---|
| Position as a model feature at training time, set to a fixed/marginalized constant at inference time | Model learns to factor out position, then predicts as if item were in a neutral position |
| Inverse propensity weighting | Weight training examples by inverse probability of being shown/clicked at that position, deconfounding the position effect |
| Randomized position experiments (small traffic slice) | Directly measure position-click curves, used to build a position bias correction model |
| Counterfactual learning-to-rank | Explicitly models the logging policy and corrects for it during offline training |

### Offline vs Online Evaluation

**Offline evaluation:**

| Metric | Measures |
|---|---|
| Precision@k / Recall@k | Relevant items retrieved in top-k |
| NDCG@k | Ranking quality with position discount |
| MAP (Mean Average Precision) | Precision across ranks, averaged over queries/users |
| Coverage / diversity metrics | Catalog coverage, intra-list diversity |
| Calibration (predicted CTR vs actual CTR) | Needed if scores feed into downstream systems like ad auctions |

**Online evaluation — A/B test design:**

```
                 Traffic splitter (consistent hashing on user_id)
                        |                          |
                        v                          v
                 Control (existing              Treatment (new
                 ranking model)                  candidate model)
                        |                          |
                        v                          v
                 Log engagement metrics    Log engagement metrics
                        |                          |
                        +------------+-------------+
                                     v
                    Statistical significance test
                    (t-test/ sequential testing,
                     guardrail metric checks,
                     segment-level analysis)
```

**Key A/B considerations for recsys:**
- **Network/interference effects**: recommendations can affect the pool of items other users see (e.g., marketplace supply); may need item-level or cluster-level randomization instead of pure user-level.
- **Novelty effects**: short-term lifts from "new UI" curiosity fade — run tests long enough to see steady-state behavior.
- **Guardrail dashboards**: latency, error rate, diversity, and complaint rate monitored throughout, not just at test end.
- **Sample ratio mismatch** checks to catch broken randomization before trusting any metric delta.

### Interview Questions

**Q1: Why not just use one large model to rank all items directly instead of a two-stage pipeline?**
A: Scoring a heavy, highly personalized ranking model over millions/billions of items per request is computationally infeasible within a page-load latency budget; the two-stage design trades a cheap high-recall filter (candidate generation) for a small pool that a slower, more precise ranker can then afford to score.

**Q2: How would you pick the size of the candidate set passed from generation to ranking?**
A: Balance recall (larger set catches more relevant items) against ranking-stage latency budget; typically tuned empirically by plotting offline recall@k of the final ranked output as a function of candidate set size, choosing the knee of the curve within the latency ceiling.

**Q3: A new content creator uploads their first video — walk through how it gets any views at all.**
A: Content-based candidate generation surfaces it via metadata/embedding similarity to users' known interests even with zero interactions; an exploration quota/boost ensures it gets some guaranteed impressions; and it is fast-tracked into the online feature/embedding pipeline as soon as any interaction data accrues.

**Q4: How do you know if position bias correction is actually working?**
A: Run a randomized "swap test" — randomly shuffle a small fraction of top positions and check whether the model's predicted relevance still correlates with actual engagement independent of the shown position; also track whether items promoted purely by the correction later show consistent (not one-off) engagement lifts.

**Q5: Why might increasing offline NDCG not translate into an online engagement lift?**
A: Offline eval uses historical logged interactions which are themselves biased by the previous ranking policy (missing counterfactual data for unseen items), and doesn't capture behavioral effects like habituation, diversity fatigue, or cross-item cannibalization that only show up in a live test.

**Q6: How would you detect and mitigate a feedback loop where the recommender increasingly favors already-popular items?**
A: Monitor Gini coefficient/concentration of impressions across the catalog over time; mitigate via exploration bandits, diversity-aware re-ranking (e.g., MMR), and periodically retraining on a de-biased or exploration-only slice of data.

**Q7: What's the tradeoff between collaborative filtering and content-based candidate generation?**
A: CF captures latent taste correlations invisible to content features (serendipity) but fails on cold-start items/users; content-based works immediately for new items but tends toward narrower, less serendipitous recommendations — most production systems blend both.

**Q8: How would you evaluate a bandit-based exploration policy before fully trusting it in production?**
A: Offline policy evaluation via importance sampling/off-policy estimators using logged bandit feedback, then a small online pilot with a bounded traffic percentage and tight guardrail monitoring before a full rollout.

**Q9: If your latency budget is cut in half, what's your first architectural lever?**
A: Reduce candidate set size passed to ranking, move more feature computation to precomputed/batch rather than online, simplify the ranking model (e.g., distill a smaller model), and increase caching of repeat-request results.

**Q10: How would you design the system to serve both a mobile app and a smart-TV app with very different latency/UX constraints from the same backend?**
A: Share the candidate generation and feature store layers, but allow separate re-ranking/business-logic layers per surface (different diversity/format constraints), and set per-surface latency SLAs/timeouts with surface-specific fallbacks.

**Q11: What's "exposure bias" and how does it differ from position bias?**
A: Exposure bias is the broader phenomenon that the model only ever learns from items it (or a prior policy) chose to show, biasing training data toward its own past decisions; position bias is a specific instance of it caused by slot order within a shown list.

**Q12: How would you decide between an A/B test at the user level vs. the item/marketplace level?**
A: If treatment can affect the shared pool available to other users (e.g., limited inventory, marketplace supply, or social network effects), user-level randomization causes interference/contamination between arms — use cluster-level (e.g., geographic region or item-side) randomization instead.

**Q13: How would you detect training-serving skew specifically in the two-tower retrieval model?**
A: Compare the distribution of retrieved candidates when scoring with the exact same user embedding offline (batch recompute) vs. what the online ANN index actually returns for that same embedding — divergence indicates a stale index, quantization mismatch, or feature pipeline inconsistency.

**Q14: How often should item embeddings be re-indexed into the ANN store, and what determines that cadence?**
A: Determined by how fast item embeddings drift (new interaction data, content changes) balanced against re-indexing cost; typically daily-to-hourly batch re-indexing with an incremental/delta index update path for newly added items to avoid full-catalog re-indexing latency.

**Q15: How do you handle a business requirement to guarantee a minimum exposure share for smaller/new creators (fairness to supply side)?**
A: Introduce an explicit diversity/fairness constraint in the re-ranking stage (e.g., a quota-based or Lagrangian-penalty re-ranker) that trades a small amount of predicted engagement for guaranteed exposure, monitored as a guardrail metric rather than left to emerge from the ranking model alone.

---

## Case Study: Design a Short-Form Video Feed Ranking System

### Problem Framing

*"Design a system that ranks an infinite-scroll feed of short-form videos (TikTok/Reels/YouTube Shorts-style)."*

This looks like a variant of the recommendation case study above, and shares the two-stage candidate-generation-plus-ranking skeleton — but a rigorous interviewer expects you to name what's *structurally different* here, not re-derive the same architecture:

| Dimension | Homepage/e-commerce recsys | Short-form video feed |
|---|---|---|
| Session shape | Minutes, several distinct "sessions" a day | Long, dense sessions (tens-hundreds of items per sitting) |
| Feedback latency | Click/purchase, often minutes-days later | Sub-second implicit signal (skip in <1s, replay, watch %) on nearly every item |
| Content velocity | New items added at moderate, predictable rate | Millions of new uploads/day; a huge fraction of impressions go to content with near-zero interaction history |
| Ranking granularity | Rank a page/row once per request | Effectively one ranking decision per swipe — re-rank *within* the session as feedback arrives |
| Label richness | Mostly binary (click/no-click, purchase/no-purchase) | Multi-signal per view: % watched, replays, like, share, follow, comment, hide/report |

**Clarify:** autoplay vs tap-to-play, infinite scroll vs discrete pages, single-objective (watch time) vs explicit multi-objective business goal, creator ecosystem health as a stated goal, latency per next-video decision (typically <100-200ms so the next video is ready before the current one ends).

**North star:** typically a composite "session value" score, *not* raw watch time alone (see below). **Guardrails:** creator/content diversity, well-being/time-well-spent complaint rate, misinformation-adjacent guardrails (cross-reference: fairness/well-being tradeoffs of engagement optimization are covered in depth in the Responsible AI/Ethics file — this section stays at the architecture level).

### Why Raw Watch Time Is the Wrong Label

Optimizing directly for watch-time-per-video is gameable and misleading: a 60-second video that plays to completion because it's engaging looks identical, by raw seconds watched, to a 5-minute video a user autoplayed into and skipped from at 60 seconds. Two corrections senior candidates should name explicitly:

- **Normalize by duration**: use **completion rate / % watched** (and replay count) rather than raw seconds, so short and long videos are compared fairly.
- **Multi-task value model**: predict several engagement heads independently, then combine into a single ranking score with business-tuned weights, rather than training on one blended label from the start.

```
                 Candidate video
                       |
                       v
        +-------------------------------+
        |     Shared representation      |  (video content/audio/caption
        |     (video + user + context)   |   embeddings, user history, session state)
        +---------------+-----------------+
                         |
      +-------+-------+--------+--------+---------+
      v       v       v        v        v         v
  P(watch   P(like)  P(share) P(follow  P(comment) P(hide/
  >=X%)                        creator)             report)  <- negative head
      |       |       |        |        |         |
      +-------+-------+--------+--------+---------+
                         |
                         v
          value = w1*P(watch) + w2*P(like) + w3*P(share)
                + w4*P(follow) + w5*P(comment) - w6*P(hide)
                         |
                         v
                  Final ranking score
```

| Design choice | Why |
|---|---|
| Separate heads per engagement type | Each signal has different sparsity/reliability (likes are sparse but high-signal; watch % is dense but noisier); a single blended label loses this structure |
| Explicit negative head (hide/report/fast-skip) | Without it, the model only ever learns "what maximizes positive engagement," never "what actively repels users" — the two are not simply inverses |
| Business-tuned combination weights, not learned end-to-end from one label | Lets product/policy teams directly adjust the tradeoff (e.g., dial down watch-time weight, raise diversity/well-being weight) without retraining the underlying model |

### Session-Level Real-Time Re-Ranking

Because feedback arrives within seconds and sessions are long, the system re-ranks *within* a session using signal from videos the user just saw — not just a daily-refreshed user embedding.

```
Video N shown --> implicit feedback (skip time, watch %, like)
                            |
                            v
                 Session state store (in-memory,
                 keyed by session_id, TTL'd)
                            |
                            v
        Candidate re-scoring for video N+1 uses:
        - static user embedding (daily/hourly refresh)
        - session state (last K videos' categories/creators
          and the user's reaction to them)
                            |
                            v
                     Video N+1 served
```

This is the same "point-in-time correctness + freshness" principle from the data pipeline section, taken to its logical extreme: the relevant "recent history" window is seconds old, not hours old, so session state lives in a fast in-memory store rather than the standard online feature store's normal refresh cadence.

### Cold Start at Extreme Content Velocity

Unlike the general recsys case study where cold start is an edge case, here it is close to the *default* state for a meaningful fraction of traffic, because upload volume vastly outpaces the rate at which any single video accumulates interactions:

| Lever | Mechanism |
|---|---|
| Content-based candidate generation | Video/audio/caption/visual-frame embeddings let a brand-new upload be retrieved by similarity before it has any engagement data |
| Creator-side priors | Creator's historical average performance, follower count, and past video quality substitute for the specific video's (currently absent) engagement history |
| Guaranteed exploration slots | A small, bounded fraction of feed slots reserved for new/low-impression content, same principle as the bandit exploration budget in the general recsys case study |
| Fast feedback-to-index loop | Engagement signal on a new video should update its candidate-generation eligibility within minutes, not the next daily batch cycle, given how quickly a video's relevance window can close |

### Diversity and Filter-Bubble Guardrails

A pure engagement-value ranker will, left unchecked, converge the feed toward a narrow band of highly-engaging content/creators — the same popularity feedback loop discussed in the general recsys case study, but faster and more visible here because of session density. Mitigations mirror that section (exploration bandits, re-ranking-stage diversity constraints/quotas) with one addition specific to this domain: **category/creator interleaving constraints** in the re-ranking stage (e.g., "no more than 2 consecutive videos from the same creator or topic cluster") applied as a hard constraint after the value-model score, not left to emerge from the model.

**Pitfalls:**
- Training on raw watch-time seconds without duration normalization silently biases the ranker toward longer content regardless of quality.
- Treating "no explicit negative feedback" as equivalent to satisfaction — fast skip/scroll-past behavior is a strong negative signal even without an explicit hide/report action, and should be modeled as such.
- Refreshing user/session state on the same cadence as a generic recsys (hourly/daily) fails to capture in-session taste shifts that matter within a single sitting.
- Ignoring diversity guardrails because the north-star metric looks great short-term — this is precisely the failure mode "north star + guardrails" is designed to catch.

### Interview Questions

**Q1: Why is a multi-task value model generally preferred over training directly on a single blended engagement label?**
A: Each engagement signal (watch %, like, share, follow, hide) has different sparsity and reliability characteristics, and blending them into one label at data-creation time bakes in a fixed tradeoff that can't be adjusted later; predicting each independently and combining with tunable weights lets product/policy teams adjust the tradeoff without retraining the base model.

**Q2: Why is raw watch time a poor primary training label for short-form video ranking?**
A: It's confounded with video duration (a longer video passively accumulates more seconds watched even at equal or lower quality) and doesn't capture negative engagement; completion rate/% watched, combined with an explicit negative (skip/hide) signal, better isolates true content quality from duration artifacts.

**Q3: How does session-level re-ranking differ architecturally from the daily-refreshed user embeddings used in typical recsys?**
A: Session-level re-ranking reads and writes a fast, short-TTL in-memory state keyed by session ID that captures the last few videos' reactions, feeding the *next* candidate scoring within the same sitting; this operates on a seconds-level freshness cadence that a normal online feature store (minutes-to-hours refresh) isn't designed for.

**Q4: A new creator uploads a video with zero views — trace how it can still get meaningfully ranked in the first hour.**
A: Content-based candidate generation retrieves it via embedding similarity to users' known interests with no interaction history required; creator-level priors (past video performance, follower count) substitute for the specific video's absent signal; and a bounded exploration quota guarantees it some impressions to begin accumulating real engagement data.

**Q5: How would you detect that your feed ranker is over-optimizing engagement at the expense of content diversity?**
A: Track concentration/Gini-style metrics on impression share across creators and content categories over time, and monitor the well-being/complaint-rate guardrail explicitly — a north-star metric that's climbing while a diversity or complaint guardrail is degrading is exactly the pattern the guardrail framework exists to catch.

**Q6: Why is fast-skip (scrolling past within a second, no explicit hide) treated as a negative signal rather than neutral/missing?**
A: If fast-skip is treated as neutral, the model has no way to distinguish "content the user actively disliked" from "content the user simply hasn't seen yet," which understates how negative that class of feedback actually is and biases the ranker toward tolerating low-quality-but-not-explicitly-reported content.

**Q7: How would you enforce that no more than two consecutive videos come from the same creator, and where in the pipeline does that belong?**
A: As a hard constraint applied in the re-ranking/business-logic stage after the value-model score is computed, similar to the diversity re-ranking used in the general recommendation case study — it should not be left for the ranking model itself to learn implicitly, since nothing in a pointwise/session value objective guarantees it.

**Q8: How do you evaluate a change to the multi-task value model's combination weights before a full rollout?**
A: Offline, sanity-check how the new weights reshuffle a sample of historical rankings against the old weights to catch obviously wrong tradeoffs; online, run a bounded A/B test tracking the north-star composite alongside every individual engagement head and the diversity/complaint guardrails, since a weight change can improve the composite while quietly regressing one head (e.g., trading away follows for watch time).

**Q9: Why does this system need a tighter next-item latency budget than a typical homepage feed?**
A: With autoplay and infinite scroll, the next video must be selected and ready before the current one finishes playing, so the effective request rate per active user is far higher (potentially one ranking decision every few seconds) than a page-load-triggered homepage feed, making per-item latency a much tighter constraint.

**Q10: How would you distinguish concept drift in this system from a content-velocity cold-start problem?**
A: Concept drift is a shift in what an *existing*, previously-well-modeled item/user relationship looks like over time; the cold-start problem here is structural — a large share of impressions are, at any given moment, going to content or creators with inherently little accumulated history, which requires content-based and creator-prior fallbacks regardless of how fresh the model is.

**Q11: What's the risk of using the same exploration bucket size as a lower-velocity recommendation system?**
A: Given how much of the catalog is perpetually "new" here, an exploration budget sized for a slower-moving catalog (e.g., 2-5%) may under-serve the volume of genuinely unproven content, starving new creators of the impressions needed to ever accumulate a signal — the exploration quota should be sized relative to the platform's actual content velocity, not copied from a generic recsys default.

---

## Case Study: Design a Search Ranking System

### Problem Framing

*"Design a search engine that ranks documents/products for a user query."*

**Clarify:** query types (navigational, informational, transactional), corpus size, latency (typically <100-300ms), personalization needs, freshness of index (news vs static docs).

**North star:** relevance-driven engagement (click-through on top results, task success, dwell time on result). **Guardrails:** latency, diversity of sources, freshness for time-sensitive queries.

### End-to-End Architecture

```
   Query -> [Query Understanding] -> [Retrieval] -> [Ranking] -> [Re-ranking/Blending] -> Results
                    |                     |               |
             spell-correct,         inverted index   pointwise/pairwise/
             intent classify,       (BM25) + dense    listwise LTR model
             query expansion        embedding ANN
             (synonyms, NER)        (hybrid retrieval)
```

### Query Understanding

| Component | Purpose | Techniques |
|---|---|---|
| Spelling correction | Handle typos | Edit-distance, learned seq2seq correction models |
| Query segmentation/NER | Identify entities, intents (e.g., "cheap flights nyc to sf" → intent=flight_search) | CRF/transformer-based NER, intent classifiers |
| Query expansion | Add synonyms/related terms | WordNet, learned expansion embeddings, click-log co-occurrence |
| Query rewriting | Normalize variants to a canonical form | Rule-based + learned rewriting models |

### Retrieval: Inverted Index (BM25) + Embeddings Hybrid

**Lexical retrieval (BM25 over an inverted index):**

```
Term -> Posting list (doc_id, term_freq, positions)
"running" -> [(doc_42, tf=3), (doc_108, tf=1), ...]
```
BM25 scores documents by term frequency, inverse document frequency, and document length normalization. Strength: exact-match precision, interpretable, cheap. Weakness: no semantic understanding ("car" won't match "automobile").

**Dense/embedding retrieval:** encode query and documents into a shared embedding space (often a two-tower/bi-encoder transformer), retrieve via ANN. Strength: semantic matching, synonym/paraphrase robustness. Weakness: can miss exact rare-term matches (e.g., product codes, exact names), expensive to keep indexes fresh.

**Hybrid retrieval — the standard production answer:**

```
      Query
        |
  +-----+------+
  |             |
  v             v
BM25 top-K   Embedding ANN top-K
  |             |
  +-----+------+
        |
   Union / merge (dedupe by doc_id)
        |
   Combined candidate set -> Ranking stage
```

Merging strategies: simple score union with normalization, reciprocal rank fusion (RRF), or a learned fusion model that takes both scores as ranking-stage features.

| Approach | Pros | Cons |
|---|---|---|
| Lexical (BM25) only | Fast, exact-match precision, no training data needed | Misses semantic/paraphrase matches |
| Dense embedding only | Captures semantic similarity | Weak on rare terms, IDs, exact phrase queries; index freshness lag |
| Hybrid | Best of both, standard in modern search (web, e-commerce) | More infra complexity, need a fusion/ranking layer |

### Ranking: Pointwise, Pairwise, Listwise Learning-to-Rank

| Approach | Loss formulation | Example algorithms | Pros | Cons |
|---|---|---|---|---|
| **Pointwise** | Predict a relevance score per document independently (regression/classification) | Logistic regression, GBDT regression | Simple, easy to train/debug | Ignores relative ordering; doesn't directly optimize ranking metrics |
| **Pairwise** | Learn to correctly order pairs of documents (which is more relevant) | RankNet, LambdaMART | Directly models relative preference | Pairwise comparisons scale O(n²) with list length; still an approximation of listwise metrics |
| **Listwise** | Optimize a ranking metric (NDCG, MAP) directly over the whole list | LambdaRank, ListNet, SoftRank | Directly optimizes what you actually care about | More complex loss/training; harder to debug |

**LambdaMART worked intuition:** it's a pairwise gradient-boosted tree model where the gradient for each pair is scaled ("lambda") by how much swapping that pair would change NDCG — effectively injecting listwise metric-awareness into a pairwise training framework. This is why it remains an extremely common production baseline even in the deep-learning era.

**Ranking features (search-specific):**

| Group | Examples |
|---|---|
| Query-document match | BM25 score, embedding cosine similarity, exact/partial term match counts |
| Document quality | PageRank-like authority score, freshness, spam score, click-through history |
| Query features | Query length, intent classification, historical query popularity |
| User/personalization | User's historical clicks, location, language, device |
| Contextual | Time of day, seasonality (e.g., "world cup" query during the tournament) |

### Relevance Feedback Loop

```
User query -> Results shown -> Click/dwell/no-click logged
                                        |
                                        v
                          Relevance judgment inference
                          (implicit: click models like DBN/click-through rate;
                           explicit: human relevance raters)
                                        |
                                        v
                          Training data for next LTR model iteration
                                        |
                                        v
                          Offline eval (NDCG on held-out judged set)
                          -> Online A/B test -> Ship
```

**Click models** (e.g., Dynamic Bayesian Network, Cascade model) are used because raw clicks are confounded by position bias and "last click wins" effects — a click model estimates the *probability a document is actually relevant* given the observed click pattern across the whole result list, not just a raw click/no-click label.

**Human relevance judgments:** used for a smaller, high-quality gold set (graded relevance, e.g., 0-4 scale) to calibrate/validate that click-derived labels correlate with true relevance, and to evaluate rare/high-stakes query categories where click data is sparse or ambiguous (e.g., medical queries).

### Interview Questions

**Q1: Why do modern search systems use hybrid retrieval instead of pure embedding-based retrieval, given embeddings capture semantics better?**
A: Pure dense retrieval systematically underperforms on exact-match needs — product codes, rare named entities, precise phrase queries — where lexical overlap is the actual signal; hybrid retrieval covers both failure modes and is empirically more robust across query type distributions.

**Q2: Why is listwise learning-to-rank harder to train than pointwise, and why bother?**
A: The ranking metrics we actually care about (NDCG, MAP) are non-differentiable and depend on the sort order of the whole list, not independent point predictions; listwise/LambdaRank-style approaches approximate metric-aware gradients, which measurably improves ranking quality over pointwise regression, at the cost of more complex training and debugging.

**Q3: How would you address position bias when constructing training labels for a learning-to-rank model from click logs?**
A: Use a click model (e.g., cascade or DBN model) that explicitly separates "probability of being examined" (a function of position) from "probability of being relevant given examined," or use randomized result interleaving experiments to get position-debiased relevance signal.

**Q4: A query has zero exact lexical matches in the inverted index but the user clearly has a valid intent — what happens, and how do you handle it?**
A: The embedding-based retrieval leg of the hybrid system should still surface semantically related documents even with zero lexical overlap; if using BM25-only retrieval, this is exactly the failure case that motivates adding a dense retrieval path.

**Q5: How would you evaluate a change to the retrieval stage (before ranking) in isolation?**
A: Measure recall@k of the retrieval stage against a human-judged relevant-document set (did the truly relevant docs make it into the candidate pool at all), independent of final ranking order — a ranking-stage improvement can't fix a retrieval-stage recall failure.

**Q6: What's the danger of over-relying on click-through rate as the online metric for search quality?**
A: Clickbait/misleading titles can inflate CTR without satisfying the underlying information need; better online signals combine CTR with post-click engagement (dwell time, no immediate "pogo-sticking" back to search results, task completion).

**Q7: How do you keep a search index fresh for time-sensitive content (e.g., breaking news) without re-indexing the entire corpus constantly?**
A: Maintain a small, frequently-updated "fresh" index/tier alongside the large, less-frequently-updated main index, and merge results across both tiers at query time, with freshness-aware ranking features boosting the fresh tier for time-sensitive query intents.

**Q8: Why might a pairwise ranking model still not directly optimize NDCG well, and how does LambdaMART address it?**
A: A naive pairwise loss treats all incorrectly-ordered pairs equally, but swapping a pair at rank 1-2 affects NDCG far more than a pair at rank 50-51; LambdaMART scales each pair's gradient ("lambda") by its actual impact on the target ranking metric, directly injecting metric sensitivity into an otherwise pairwise objective.

**Q9: How would you handle personalization in ranking without hurting head-query latency?**
A: Precompute/cache user-level personalization features (embeddings, historical preferences) offline or in a fast online feature store, so the ranking model only needs a cheap lookup rather than an expensive on-the-fly personalization computation per query.

**Q10: What offline signal would you use to catch a ranking regression before it reaches an A/B test?**
A: A held-out, human-judged relevance set (query, document, graded relevance label) scored with NDCG/MAP on every model candidate as a pre-deployment gate, plus regression tests on a fixed "golden query" set known to be sensitive to ranking changes.

**Q11: How do query understanding errors (e.g., wrong intent classification) propagate downstream, and how would you contain the blast radius?**
A: Wrong intent skews retrieval toward the wrong document types (e.g., treating a navigational query as informational), degrading recall for the true intent; contain it by having retrieval consider multiple candidate intents in parallel with confidence-weighted blending rather than a hard single-intent gate.

**Q12: Would you use the same ranking model for all query types (navigational, informational, transactional), or separate models?**
A: Often a shared base model with intent as a strong feature works well and is easier to maintain; but for very distinct behavior patterns (e.g., transactional/shopping queries needing price/inventory features irrelevant to informational queries) a specialized secondary ranker or feature set per intent segment is common in mature systems.

---

## Case Study: Design an Autocomplete and Search-Suggestion System

### Problem Framing

*"Design the typeahead/autocomplete system that suggests completions as a user types into a search box."*

This is a distinct system-design problem from search ranking (previous section), not a smaller version of it: the request fires on **every keystroke**, the acceptable latency is far tighter, and most of the "intelligence" lives in a specialized prefix-retrieval data structure rather than a heavy ranking model.

**Clarify:** perceived latency target (typically sub-100ms, often targeted at <20-50ms for the suggestion list to feel instant), personalization (recent/frequent searches, location), scale (fires far more often than full search — every keystroke of every user), multi-language/script support, safety (suggestions must never surface policy-violating or offensive completions), whether suggestions should reflect trending/breaking-news queries.

**North star:** suggestion acceptance rate (user selects a suggestion vs. typing the full query) or downstream search success rate when a suggestion is used. **Guardrails:** latency, offensive/policy-violating suggestion rate, freshness lag for trending queries.

### Why This Isn't Just "Search Ranking, but Shorter"

| Property | Full search ranking | Autocomplete |
|---|---|---|
| Request trigger | One request per submitted query | One request per keystroke (many requests per query eventually typed) |
| Latency budget | ~100-300ms acceptable | ~10-50ms perceived as "instant"; users notice lag on every keypress |
| Primary retrieval structure | Inverted index (BM25) + ANN | Prefix index (trie/FST) over a fixed vocabulary of historical queries |
| Ranking model weight | Heavy LTR model is standard | Lightweight scoring pass over a small pre-filtered candidate set; heavy per-request ML inference is usually too slow/expensive at this request volume |
| Query understanding | Full NER/intent classification | Mostly prefix matching + light typo tolerance; full query is often incomplete/ambiguous by definition |

### Architecture

```
  Keystroke --> [Client-side debounce/cache] --> Suggestion request
                                                         |
                                                         v
                                          +---------------------------+
                                          |   Prefix Index Lookup      |  trie / FST over
                                          |   (candidate generation)   |  historical query log,
                                          +--------------+-------------+  precomputed top-K per node
                                                         |
                                                         v
                                          +---------------------------+
                                          |  Lightweight Ranking       |  popularity + personalization
                                          |  (blend + personalize)     |  + recency/trending boost
                                          +--------------+-------------+
                                                         |
                                                         v
                                          +---------------------------+
                                          |  Safety / Policy Filter    |  precomputed denylist +
                                          +--------------+-------------+  classifier, applied at
                                                         |               index-build time too
                                                         v
                                                 Top-N suggestions
                                                 returned to client
```

### Prefix Retrieval: Trie / FST Design

The core data-structure decision: a **trie** (or a more memory-efficient **finite state transducer / DAWG**) built offline from the historical query log, where each node stores the **precomputed top-K most popular completions** for that prefix rather than requiring a live traversal-and-rank of the entire subtree per request.

```
        root
       / | \
      c  d  ...
     /|
    a t
    | |
    r  <- node for prefix "cat"
    (precomputed top-5 completions cached at this node:
     "cats", "catalog", "category", "catamaran", "cat food")
```

| Design choice | Why |
|---|---|
| Precompute top-K per node offline, not at request time | Traversing and sorting an entire subtree per keystroke would blow the latency budget at this request volume; precomputation trades index-build cost (batch, offline) for O(prefix length) request-time lookup |
| Rebuild/update index incrementally, not only in full nightly batches | Query popularity — especially trending/breaking-news queries — shifts within minutes; a purely nightly-rebuilt trie misses same-day trends entirely |
| FST/DAWG over a naive trie for large vocabularies | Shares suffixes across many words, dramatically reducing memory footprint at the scale of a real production query vocabulary (tens-hundreds of millions of distinct historical queries) |
| Fuzzy/typo-tolerant traversal as a secondary path | A pure exact-prefix trie returns nothing for a misspelled prefix; merge in a small edit-distance-tolerant candidate set (or a separate spelling-correction index) the same way search ranking merges lexical and dense retrieval |

### Ranking and Personalization

Once the trie returns a candidate set (typically already ordered by global popularity), a lightweight pass blends in:

| Signal | Purpose |
|---|---|
| Global popularity / historical frequency | The base signal already baked into the precomputed trie ordering |
| Personalization (user's own recent/frequent searches) | A user who has searched "python pandas" repeatedly should see it surface faster on "py" than a generic popular unrelated term |
| Recency/trending boost | Streaming pipeline detects rapid query-volume spikes (breaking news, live events) and boosts affected prefixes ahead of their "normal" historical popularity rank |
| Context (location, app surface, time of day) | E.g., a food-delivery app's autocomplete should weight local restaurant names differently by the user's city |

This ranking pass is deliberately kept cheap (simple weighted blend or a small model, not a full LTR pass) precisely because it runs on every keystroke — pushing personalization signal to be **precomputed per user** (a small cached profile fetched in one lookup) rather than computed fresh per request is the standard mitigation, mirroring the "precompute personalization features to protect head-query latency" answer from the search ranking case study.

### Latency Budget (Worked Example)

| Stage | Budget | Notes |
|---|---|---|
| Client debounce | ~50-100ms wait after last keystroke, or request cancellation on new keystroke | Prevents firing a full request cascade on every single character; also the single highest-leverage lever for reducing backend load |
| Network round trip | ~10-30ms | Often served from an edge/CDN cache for very common prefixes |
| Trie/FST lookup | ~1-5ms | O(prefix length); the entire reason for precomputing top-K per node |
| Ranking/personalization blend | ~2-10ms | Cheap lookup + weighted blend, not a heavy model |
| Safety filter | ~1-3ms | Precomputed at index-build time wherever possible; request-time check is a fast lookup, not a fresh classifier call |

**Framework tip:** because the *client-side* debounce/cancel behavior directly determines backend load (an undebounced client can generate one request per character typed, i.e., 5-10x the necessary QPS), this is one of the few case studies in this file where client-side design decisions are as load-bearing as backend architecture — worth calling out explicitly to an interviewer.

### Scale Considerations

- **Zipfian traffic distribution**: a small fraction of prefixes account for the large majority of requests, so an edge/CDN cache in front of the ranking service achieves a very high hit rate — mention this explicitly as a cost/latency lever.
- **Sharding the trie**: shard by prefix range or hash across serving nodes once the vocabulary exceeds single-node memory; route requests to the correct shard by the first character(s) of the prefix.
- **Case/diacritic/locale normalization**: must be handled consistently between index-build time and query time (a specific instance of the training-serving-skew principle — same normalization logic must run in both the offline index build and the online request path).

**Pitfalls:**
- Computing a full learning-to-rank pass per keystroke — the request volume at this layer makes anything heavier than a cheap blend infeasible at reasonable cost/latency.
- No client-side debounce or request cancellation, causing an explosion of in-flight requests (most of which become stale before the user finishes typing).
- Applying the safety/policy filter only at request time instead of also at index-build time, so a policy-violating query can still transiently surface before the filter catches it.
- Rebuilding the trie only in a nightly batch, missing same-day trending queries entirely — treat this the same as the "streaming vs batch" freshness tradeoff from the data pipeline section.

### Interview Questions

**Q1: Why use a trie/FST with precomputed top-K completions instead of a live database query ranked at request time?**
A: At the request volume autocomplete generates (potentially one request per keystroke across all users), traversing and ranking a full candidate set live would blow the sub-50ms latency budget; precomputing the top-K completions per prefix offline trades a batch-time cost for an O(prefix-length) request-time lookup.

**Q2: How would you keep autocomplete suggestions fresh for a breaking-news query that didn't exist an hour ago?**
A: A streaming pipeline monitors query-volume velocity and boosts rapidly-trending prefixes into the served candidate set ahead of their "normal" historical-popularity rank, incrementally updating the relevant part of the index rather than waiting for the next full nightly rebuild.

**Q3: Why is client-side debounce called out as a backend design lever rather than purely a frontend concern?**
A: An undebounced client fires a new backend request on every keystroke, multiplying request volume several-fold over what's actually needed to serve a useful suggestion list; since this system's backend load is dominated by request *count* rather than per-request complexity, the client-side trigger policy directly determines backend capacity requirements.

**Q4: A user types a prefix with a typo — what happens in a naive trie-only design, and how do you fix it?**
A: A pure exact-prefix trie traversal returns nothing once the typed characters diverge from any real prefix in the index; fix by merging in a secondary edit-distance-tolerant candidate path (or a dedicated spelling-correction index) alongside the exact-prefix path, the same hybrid-retrieval principle used for lexical+dense search.

**Q5: How would you personalize autocomplete without adding meaningful per-request latency?**
A: Precompute and cache a small personalization profile (the user's own recent/frequent queries) that can be fetched in a single fast lookup and blended into the ranking pass, rather than computing any personalization signal fresh per keystroke.

**Q6: Why might a trie be replaced with a finite state transducer (FST) at large vocabulary scale?**
A: An FST/DAWG shares common suffixes across many entries, giving substantially better memory efficiency than a naive trie at the scale of a real historical query vocabulary (tens to hundreds of millions of distinct queries), which matters directly for how much of the index can be held in memory per serving node.

**Q7: How do you prevent a policy-violating query from ever appearing as a suggestion?**
A: Apply the safety/policy filter at index-build time (so violating queries are never inserted into the served trie in the first place), not only as a request-time check, since a request-time-only filter still allows a transient window where a newly-popular violating query could surface before the filter catches up.

**Q8: How would you shard the prefix index once the vocabulary no longer fits on one serving node?**
A: Shard by prefix range or a hash of the first character(s), routing each incoming request to the correct shard based on the prefix typed so far; this keeps the lookup itself O(prefix length) per shard while distributing memory footprint across nodes.

**Q9: Why is a heavy learning-to-rank model, appropriate for full search ranking, generally the wrong choice for the autocomplete ranking pass?**
A: Autocomplete's request volume (one request per keystroke, across all active users) is far higher than full search's per-submitted-query volume, so a ranking pass that's affordable in search ranking's latency/cost budget would be prohibitively expensive here; autocomplete instead relies on a cheap blend over an already-pre-filtered, precomputed candidate set.

**Q10: What metric would tell you your autocomplete system is technically fast but not actually useful?**
A: Low suggestion acceptance rate (users ignoring suggestions and typing the full query themselves) despite low latency indicates the *relevance* of suggestions is the problem, not speed — this is why the north-star metric is acceptance/downstream-search-success rate, not latency alone (latency belongs in guardrails).

**Q11: How would caching interact with the Zipfian nature of prefix request traffic?**
A: Because a small number of common prefixes account for a large share of total request volume, an edge/CDN cache in front of the ranking service can absorb the majority of traffic with a small cache footprint, substantially reducing load on the origin ranking/trie service — a high-leverage, low-complexity scaling lever specific to this traffic shape.

**Q12: How would normalization (case, diacritics, locale) inconsistencies between index-build and query time manifest as a bug?**
A: A query typed with different casing/accents than how the index was built would fail to match an otherwise-valid prefix entry, silently degrading suggestion quality for affected locales/scripts — this is a training-serving-skew-style bug, and the fix is the same: normalize with one shared, versioned piece of logic used identically at index-build and request time.

---

## Case Study: Design a Fraud/Anomaly Detection System

### Problem Framing

*"Design a system to detect fraudulent transactions in real time."*

**Clarify:** transaction type (card payment, account takeover, wire transfer), decision point (pre-authorization vs post-hoc review), latency (usually <100-300ms if blocking authorization), cost asymmetry (missed fraud $$ vs false-positive customer friction), regulatory constraints.

**North star:** minimize fraud loss $ net of prevented-transaction revenue loss. **Guardrails:** false-positive rate (customer friction/churn), latency, explainability for disputes/regulators.

### Class Imbalance at Extreme Scale

Fraud rates are typically 0.01%-1% of transactions — a massively imbalanced problem.

| Technique | Mechanism | Tradeoff |
|---|---|---|
| Undersampling majority class | Randomly drop non-fraud examples during training | Loses information; can hurt calibration |
| Oversampling minority class (SMOTE, replication) | Synthesize/replicate fraud examples | Risk of overfitting to synthetic patterns |
| Class-weighted loss | Penalize misclassifying minority class more heavily | No data loss, but hyperparameter (weight) needs tuning |
| Anomaly detection framing | Model "normal" behavior distribution, flag deviations (unsupervised/semi-supervised) | Doesn't need many labeled fraud examples; higher false-positive risk |
| Precision-Recall (not ROC) focused evaluation | ROC-AUC is misleadingly optimistic under extreme imbalance | Requires stakeholder education on PR-AUC/precision@threshold |
| Calibration post-training | Correct probability estimates skewed by resampling | Needed if score feeds into a cost-based decision threshold |

**Framework: cost-based thresholding.** Instead of a default 0.5 probability cutoff, set the decision threshold by explicit business cost:

```
Expected cost(block) = P(legit) * cost_of_false_positive (friction, lost revenue)
Expected cost(allow) = P(fraud) * cost_of_false_negative (fraud loss)

Decision: block if Expected cost(allow) > Expected cost(block)
       => threshold derived from cost ratio, not arbitrary 0.5
```

### Real-Time Scoring Latency Constraints

```
Transaction request
        |
        v
+-----------------+     +----------------------+
| Feature lookup   | --> | Precomputed aggregate |  (Redis/online store:
| (real-time)      |     | features (rolling      |   rolling txn count,
|                  |     | windows, velocity)     |   avg amount, device
+-----------------+     +----------------------+   reputation, etc.)
        |
        v
+-----------------+
| Lightweight model |  (GBDT / logistic regression — deep models rare
| scoring (<20-50ms)|   here due to latency + explainability needs)
+-----------------+
        |
        v
   Score + threshold ---> ALLOW / BLOCK / STEP-UP CHALLENGE (2FA)
                                          |
                                          v
                              (if borderline) Route to human
                              review queue (async, doesn't
                              block the transaction)
```

Because the model sits directly in the critical authorization path, latency budgets are tight (often tens of milliseconds) — this rules out expensive joint feature computation across many slow data stores, favoring precomputed rolling aggregates refreshed by a streaming pipeline.

### Feature Engineering from Transaction Streams

| Feature category | Examples |
|---|---|
| Velocity features | # transactions in last 1min/1hr/1day, $ sum in rolling windows, per card/device/IP |
| Behavioral deviation | Deviation from user's historical average transaction amount/merchant category |
| Graph/network features | Shared device/IP/card across multiple accounts (fraud rings), graph degree centrality |
| Device/session features | Device fingerprint reputation, new device flag, IP geolocation mismatch vs billing address |
| Merchant features | Merchant risk score, merchant category code, chargeback history |
| Sequence features | Time-since-last-transaction, transaction sequence embeddings (RNN/transformer-derived) |

Streaming computation of velocity/rolling-window features is typically implemented via a windowed aggregation engine (Flink/Kafka Streams) writing into a low-latency online store, refreshed continuously as new transactions arrive.

### Feedback Loop with Delayed/Adversarial Labels

This is the hardest part of fraud system design and a favorite senior-level probe:

| Challenge | Explanation | Mitigation |
|---|---|---|
| **Delayed labels** | True fraud/chargeback confirmation can take 30-90+ days | Fast proxy labels (early chargebacks, user disputes) for rapid model + slow "mature label" model for calibration; explicitly track/report on labeling lag |
| **Sample-and-hold / survivorship bias** | Blocked transactions never generate a genuine outcome label | Randomly allow a small % of borderline-flagged transactions through to observe true outcome, feeding an unbiased evaluation set |
| **Adversarial adaptation** | Fraudsters actively probe and adapt to evade the current model (concept drift is *adversarial*, not just natural) | Frequent retraining cadence, ensemble of diverse models/rules to raise attacker cost, anomaly-based components that don't rely solely on known fraud patterns, rate-limit probing behavior |
| **Feature leakage from investigation process** | Features accidentally encode "this was already flagged for review" | Strict separation of investigation-stage metadata from model input features |

### Human-in-the-Loop Review Queues

```
     Model score
         |
   +-----+-----+-----------------+
   |     |                       |
 Low   Medium                  High
(auto  (queued for              (auto-block,
allow) human review,            optional step-
        priority-ranked         up auth)
        by score * $amount)
   |     |                       |
   v     v                       v
Allow  Human analyst          Block/Challenge
       decision -> label
       feeds back into
       training data
```

**Review queue design considerations:**
- **Prioritization**: rank the queue by expected value at risk (score × transaction amount), not just raw score, so limited analyst time targets the highest-cost cases first.
- **Analyst feedback capture**: structured labels (confirmed fraud / confirmed legitimate / inconclusive) feed directly back into training data, closing the loop.
- **Queue capacity constraints**: if review volume exceeds analyst throughput, the system must have a policy for what happens to excess borderline cases (auto-allow with monitoring vs auto-block with appeal path) — an explicit tradeoff to raise with the interviewer.
- **Analyst-model disagreement tracking**: monitor cases where human review overturns the model decision as a targeted error-analysis and retraining signal.

### Interview Questions

**Q1: Why is ROC-AUC a misleading primary metric for a fraud model, and what would you use instead?**
A: Under 0.1% fraud prevalence, ROC-AUC can look excellent while precision at any operational threshold is still poor (huge false-positive volume in absolute terms) because ROC-AUC is insensitive to the extreme class skew; precision-recall AUC and precision@fixed-recall (or @fixed-review-capacity) reflect operational reality far better.

**Q2: How do you set the fraud score threshold, and how does it differ across merchants or transaction types?**
A: Derive the threshold from the relative cost of a false positive (customer friction, lost revenue, support cost) vs. false negative (fraud loss amount); this ratio differs by merchant/transaction size, so thresholds are often tuned per-segment rather than globally fixed.

**Q3: How would you design an evaluation strategy given that most blocked transactions never get a confirmed ground-truth label?**
A: Sample-and-hold — deliberately allow a small random fraction of would-be-blocked transactions through, observe their true outcome, and use that unbiased sample to estimate true precision/recall of the full policy, correcting for the sampling rate.

**Q4: Why might a deep neural network be a poor choice for the final production fraud-scoring model despite potentially higher offline accuracy?**
A: Latency constraints in the authorization path, the need for case-level explainability (regulatory disputes, chargebacks, analyst review), and easier real-time feature engineering compatibility often favor GBDTs/logistic regression with well-engineered features — though DNNs may still be used upstream for representation learning (e.g., sequence embeddings) feeding into a lighter final scorer.

**Q5: How do you defend against a fraud model becoming stale due to adversarial adaptation, distinct from ordinary data drift?**
A: Treat drift monitoring as continuous and frequent (not just periodic), diversify signal sources (rules + anomaly detection + supervised model, so an attacker evading one layer still hits another), and shorten retraining cycles specifically because the "true function" is actively adapting against you, unlike passive drift.

**Q6: What's a graph-based feature and why is it powerful for fraud specifically?**
A: Fraud often involves rings/networks sharing devices, IPs, or payment instruments across many "unrelated" accounts; graph features (shared-entity counts, connected-component size, centrality) surface these coordinated patterns that transaction-level tabular features alone would miss.

**Q7: How would you prioritize a human review queue when there are more flagged cases than analyst capacity?**
A: Rank by expected monetary value at risk (score × transaction amount, or expected loss), not raw model score alone, so review time concentrates on the highest-impact cases; also consider SLA-based prioritization for time-sensitive cases (e.g., before a transaction settles).

**Q8: What happens to your feature pipeline design if the latency budget drops from 200ms to 50ms?**
A: Push more features to be fully precomputed (streaming rolling aggregates written to a fast online store) rather than computed on-demand, simplify or distill the model, and consider a tiered approach — a very fast first-pass filter followed by a slower secondary check only for borderline scores.

**Q9: How would you incorporate step-up authentication (e.g., 2FA challenge) into the decision framework instead of a binary allow/block?**
A: Add a third decision band for borderline scores where the cost-benefit of an added friction step (challenge) is better than either extreme; model this explicitly as a 3-way (or continuous) decision policy with its own cost terms (friction cost of a challenge vs. block/allow costs).

**Q10: How would you detect that a specific fraud pattern is a new, previously-unseen attack vector rather than one your supervised model already covers?**
A: Run an unsupervised/semi-supervised anomaly detection layer in parallel with the supervised classifier specifically to catch out-of-distribution patterns the supervised model (trained only on historically labeled fraud) has never seen, then route detected anomalies to analysts for new-pattern investigation and labeling.

**Q11: Why is it risky to let investigation/review metadata leak into your model's features?**
A: If a feature encodes "flagged for manual review" or similar investigation-stage signals, the model effectively learns to predict its own past decisions rather than true fraud risk, creating a circular, non-generalizable signal that will fail on new patterns and can mask true model quality in evaluation.

**Q12: How would you measure whether your fraud model is contributing to unfair outcomes across demographic or geographic segments?**
A: Slice false-positive rate (customer friction) and false-negative rate by relevant segments (geography, demographic proxies where legally appropriate) as an explicit fairness guardrail, and investigate/mitigate systematic disparities (e.g., higher false-positive rates for certain regions due to sparse training data there).

---

## Case Study: Design a Chatbot / RAG-based Q&A System

### Problem Framing

*"Design a chatbot that answers questions grounded in a company's internal/external documents."*

**Clarify:** knowledge domain and freshness, latency target (typically 1-5s for chat, streaming perceived latency matters more than total), multi-turn vs single-turn, need for citations, safety/guardrail requirements, expected QPS/concurrency, cost budget (token-based).

**North star:** answer quality (helpfulness + correctness/groundedness). **Guardrails:** latency, hallucination rate, safety violations, cost per query.

### End-to-End RAG Architecture

```
  Documents --> [Ingestion Pipeline] --> [Vector Store + Metadata Store]
                                                  |
  User Query --> [Query Processing] --> [Retrieval] <--------+
                        |                    |
                        |                    v
                        |          [Re-ranking of retrieved chunks]
                        |                    |
                        v                    v
                  [Prompt Construction: system prompt + retrieved context + query + history]
                        |
                        v
                  [LLM Generation] --> [Guardrails/Safety Filter] --> Response to user
                        |                                                   |
                        v                                                   v
                  [Citation attachment]                          [Feedback capture: thumbs up/down,
                                                                    follow-up question, escalation]
                                                                             |
                                                                             v
                                                                  [Feedback store -> retrieval/prompt
                                                                   tuning, eval set curation]
```

### Ingestion Pipeline

| Step | Purpose | Key decisions |
|---|---|---|
| Document parsing | Extract text from PDFs, HTML, Office docs, etc. | Handle tables/images; OCR for scanned docs |
| Chunking | Split documents into retrievable units | Chunk size (typically 200-800 tokens), overlap, structure-aware splitting (by heading/section) |
| Embedding | Convert chunks to dense vectors | Choice of embedding model, dimensionality, domain fine-tuning |
| Indexing | Store vectors + metadata for retrieval | ANN index type (HNSW, IVF), metadata filters (date, source, access permissions) |
| Incremental updates | Keep index fresh as documents change | Change-data-capture on source docs, delta re-indexing, versioning/deletion handling |

**Chunking pitfalls:** chunks too small lose context (a fact split across chunk boundaries becomes unretrievable together); chunks too large dilute retrieval precision and waste context-window budget. Structure-aware chunking (respecting headings/sections/tables) generally outperforms naive fixed-size splitting.

### Retrieval

Same hybrid principle as search ranking (see Search Ranking case study): combine dense embedding similarity with lexical/keyword matching (BM25) for robustness on exact terms (product names, IDs, codes) that embeddings can miss. Add a **re-ranking** step (often a cross-encoder) over the top-k retrieved chunks before passing to the LLM, since initial retrieval optimizes for recall while re-ranking optimizes for precision within the smaller candidate set — the same two-stage principle as recommendation/search, applied inside RAG.

**Metadata filtering** is critical in enterprise RAG: retrieval must respect access-control permissions (a user should never receive context from documents they aren't authorized to see), recency filters, and source-type filters — implemented as pre-filters on the vector search, not as a post-hoc check after generation.

### Generation and Prompt Construction

```
System prompt: role, tone, constraints, citation format instructions
     +
Retrieved context: top-k re-ranked chunks (with source metadata)
     +
Conversation history: prior turns (summarized/truncated if long)
     +
User query
     =
Final prompt sent to LLM (must fit context window budget)
```

**Key design decisions:**
- **Context window budgeting**: allocate a fixed token budget across system prompt, retrieved context, history, and query; truncate/summarize history first typically, since retrieved context is the primary grounding signal.
- **Grounding instructions**: explicit system-prompt instructions to answer only from provided context and say "I don't know" rather than fabricate, plus citation formatting requirements.
- **Structured output**: for downstream parsing (e.g., citations, confidence), request structured (JSON) output where feasible.

### Guardrails

| Guardrail type | Purpose | Implementation |
|---|---|---|
| Input guardrails | Block prompt injection, jailbreak attempts, PII in queries | Classifier or rule-based pre-filter before hitting the LLM |
| Output guardrails | Block toxic, biased, or policy-violating generations | Post-generation classifier, or a second "critic" LLM pass |
| Groundedness/hallucination check | Verify the answer is actually supported by retrieved context | NLI-style entailment check between answer and context, or a citation-verification step |
| Topic/scope guardrails | Keep the bot within its intended domain | Intent classifier routing off-topic queries to a refusal/redirect path |
| Rate limiting / abuse prevention | Prevent cost blowups and abuse | Per-user quotas, anomaly detection on usage patterns |

### Feedback Capture and Continuous Improvement

```
Response shown -> User signal (thumbs up/down, follow-up rephrase,
                   escalation to human agent, explicit correction)
                              |
                              v
                    Feedback store (query, retrieved context,
                    generated answer, signal, timestamp)
                              |
              +---------------+----------------+
              |                                |
    Curate hard/failure cases          Aggregate metrics dashboard
    into eval set for regression       (thumbs-up rate, escalation
    testing and retrieval tuning       rate, latency, cost/query)
```

Since fine-tuning the base LLM is often unnecessary or infeasible, the primary levers for continuous improvement in RAG systems are: retrieval quality (chunking, embedding model, re-ranker), prompt refinement, and guardrail tuning — informed directly by curated feedback failure cases.

### Scaling Considerations for High QPS

| Bottleneck | Mitigation |
|---|---|
| LLM inference cost/latency at scale | Batching requests, model distillation/smaller models for simpler queries, response caching for repeated/similar queries (semantic cache) |
| Vector search latency at scale | Sharded ANN indexes, approximate search tuning (recall vs. speed knob), read replicas |
| Token cost at scale | Prompt compression, context truncation, tiered model routing (cheap model for easy queries, expensive model for hard ones) |
| Streaming UX under load | Token-streaming responses to reduce perceived latency even if total generation time is unchanged |

**Semantic caching**: cache responses keyed by embedding similarity of the query (not just exact string match), so paraphrased-but-equivalent repeated questions hit the cache — a major cost/latency lever for high-traffic RAG systems with a "long tail of repeated intent."

### Latency Budget Breakdown (Worked Example)

For a target end-to-end p95 of 3 seconds on a streamed chat response:

| Stage | Budget | Notes |
|---|---|---|
| Query processing (intent/guardrail pre-check) | ~50-100ms | Lightweight classifier |
| Retrieval (hybrid search + metadata filter) | ~100-300ms | ANN index lookup + BM25 |
| Re-ranking of retrieved chunks | ~50-150ms | Cross-encoder over top-k, k typically 20-50 |
| Prompt construction | ~<10ms | String assembly, negligible |
| LLM generation (time-to-first-token) | ~300-800ms | Perceived latency; streaming starts here |
| LLM generation (full completion) | ~1.5-3s+ | Depends on output length; mitigated by streaming so user doesn't wait for full completion |
| Output guardrail check | ~50-150ms | Can run on partial/streamed output or post-hoc |

**Key insight for interviews:** time-to-first-token (TTFT) matters more for perceived latency than total generation time once streaming is used — architect the guardrail/generation pipeline to start streaming as early as safely possible, deferring heavier output-safety checks to run concurrently or on stream chunks rather than blocking the entire response.

### Interview Questions

**Q1: Why is chunking strategy one of the highest-leverage design decisions in a RAG system?**
A: Retrieval quality is bounded by whether a chunk contains a complete, self-sufficient answer unit — too-small chunks fragment context across boundaries (unretrievable together), too-large chunks dilute embedding specificity and waste the LLM's context budget, so chunking directly determines both recall and generation grounding quality.

**Q2: How would you detect hallucination in a RAG system's outputs at scale, without human review of every response?**
A: An automated groundedness/entailment check comparing the generated answer against the retrieved context (e.g., NLI model or an LLM-as-judge prompt asking "is this claim supported by this context?"), flagging low-entailment responses for sampling-based human review and dashboarding a hallucination-rate metric over time.

**Q3: Why add a re-ranking step after initial vector retrieval instead of just retrieving more chunks directly into the LLM prompt?**
A: Initial retrieval (especially ANN-based) optimizes for recall over a large corpus cheaply but is less precise; a heavier cross-encoder re-ranker over a smaller candidate set (e.g., top 50) achieves much higher precision at acceptable added latency, and feeding fewer but more relevant chunks into the LLM improves both grounding quality and cost (fewer tokens).

**Q4: How does your architecture change for a multi-turn conversation vs. single-turn Q&A?**
A: Must manage conversation history within the token budget (summarization/truncation strategy), potentially rewrite/contextualize the current query using prior turns before retrieval (since a follow-up like "what about last year?" is meaningless to a retriever without context), and track/carry state (e.g., previously cited sources) across turns.

**Q5: What's the danger of relying solely on thumbs-up/down feedback to improve the system?**
A: It's sparse (low response rate), biased toward extreme experiences, and doesn't diagnose *why* a response failed (retrieval miss vs. generation error vs. guardrail false positive) — supplement with structured failure analysis on a curated sample and implicit signals (rephrased follow-ups, escalation to a human agent) as stronger, denser feedback signals.

**Q6: How would you enforce document-level access control in a RAG system serving multiple user permission tiers from one shared knowledge base?**
A: Apply permission metadata as a hard pre-filter at the vector search stage (not a post-hoc filter on retrieved results), so restricted documents are never even considered as retrieval candidates for unauthorized users, and audit that filter enforcement rather than trusting prompt-level instructions to withhold restricted content.

**Q7: When would you choose to fine-tune the underlying LLM rather than rely purely on RAG with prompting?**
A: When the need is stylistic/format/behavioral adaptation (consistent tone, domain-specific reasoning patterns, structured output reliability) rather than factual knowledge injection — RAG is generally the right tool for injecting up-to-date or proprietary factual knowledge, while fine-tuning is better for consistently changing *how* the model responds.

**Q8: How would you reduce cost for a high-QPS RAG deployment without materially hurting answer quality?**
A: Semantic caching of repeated/similar queries, tiered model routing (route simple/FAQ-like queries to a cheaper/smaller model, escalate complex queries to a larger model), prompt/context compression, and reducing re-ranker candidate set size where retrieval precision is already high.

**Q9: How do you test a RAG system for prompt injection where a malicious retrieved document tries to hijack the LLM's instructions?**
A: Treat retrieved content as untrusted data, not instructions — use prompt structuring/delimiters that clearly separate system instructions from retrieved context, run an input/output guardrail classifier specifically trained on injection patterns, and red-team the pipeline with adversarial documents designed to embed fake instructions.

**Q10: What's the tradeoff between a bigger top-k in retrieval (more context) and the LLM's context window/cost budget?**
A: More retrieved chunks increase the chance the true answer is present (better recall) but increase token cost/latency and can dilute the LLM's attention across irrelevant chunks (sometimes degrading answer quality) — the re-ranking stage exists precisely to let you retrieve broadly then pass only the most relevant few chunks forward.

**Q11: How would you design an evaluation set for a RAG system before it has real user traffic?**
A: Curate a set of representative questions with known correct answers and known supporting source documents (a "golden set"), and measure retrieval recall@k, groundedness of generated answers against the golden context, and answer correctness via human or LLM-as-judge scoring — before any real users interact with the system.

**Q12: Streaming is enabled, but a user still complains the bot "feels slow." What would you investigate?**
A: Check time-to-first-token specifically (not total completion time) — if TTFT is high, the bottleneck is likely retrieval/re-ranking/guardrail pre-checks blocking the start of generation; optimize or parallelize those stages, or begin streaming before slower guardrail checks complete (checking output guardrails on the stream rather than gating the start of the stream).

**Q13: How would you keep the vector index fresh when source documents are updated or deleted frequently?**
A: Use change-data-capture or webhook-driven incremental re-indexing tied to the source system, version chunks so stale versions can be invalidated/removed rather than accumulating duplicates, and include a "last updated" metadata filter/boost in retrieval so freshness is factored into ranking, not just presence in the index.

---

## Case Study: Design a Content Moderation / Toxicity Detection System

### Problem Framing

*"Design a system to detect and act on toxic/policy-violating content at platform scale."*

**Clarify:** content types (text, image, video, multi-modal), real-time (pre-publish) vs post-hoc (already published) moderation, policy categories (hate speech, harassment, spam, CSAM, misinformation — each with very different risk profiles), scale (posts/sec), regional/legal variation in policy definitions.

**North star:** minimize platform harm (weighted by severity) subject to acceptable user friction. **Guardrails:** false-positive rate (over-censorship/free-expression complaints), review latency SLA for high-severity categories, appeal overturn rate.

### Multi-Label Classification at Scale

Content moderation is fundamentally **multi-label**, not single-label — a single post can simultaneously be spam, harassment, and contain a policy-violating link.

```
                     Content item
                          |
                          v
        +----------------------------------+
        |     Feature extraction            |
        |  text embeddings, image/video      |
        |  embeddings, user history,         |
        |  engagement velocity, network      |
        |  signals (who's sharing it)        |
        +-----------------+------------------+
                          |
                          v
        +----------------------------------+
        |   Multi-label classifier(s)        |
        |  per-category score:               |
        |  hate_speech, harassment, spam,     |
        |  violence, sexual_content, ...      |
        +-----------------+------------------+
                          |
                          v
        +----------------------------------+
        |   Per-category thresholding &      |
        |   severity-weighted policy engine  |
        +-----------------+------------------+
                          |
        +--------+--------+--------+
        v                 v                v
    Auto-allow      Human review        Auto-remove /
    (low score       queue (medium      restrict (high
    all categories)  confidence /       confidence +
                      high severity)     high severity)
```

**Why separate models/heads per category (rather than one generic "toxic" score) is usually preferred:**

| Approach | Pros | Cons |
|---|---|---|
| Single generic toxicity score | Simple, fast to build | Conflates very different harms (spam vs. CSAM vs. political misinformation) with very different cost/action policies |
| Per-category classifiers (shared backbone, multiple heads) | Each category can have its own threshold/action policy matched to its severity and error-cost profile | More complex to maintain; needs category-specific labeled data |

### Human Review Integration

| Tier | Trigger | SLA |
|---|---|---|
| Auto-allow | All category scores well below threshold | N/A — no human touch |
| Priority human review (high severity categories: self-harm, CSAM, imminent violence) | Any non-trivial score in these categories | Minutes, 24/7 staffing, often escalates to specialized/legal teams |
| Standard human review queue | Medium-confidence scores in standard categories (spam, harassment, misinformation) | Hours, prioritized by severity × reach/virality |
| Auto-remove | High-confidence, high-severity score | Immediate; paired with a user appeal path |

**Prioritization within the review queue:** rank by `severity_weight × predicted_probability × estimated_reach` (a viral post with medium confidence may warrant faster review than a low-reach post with slightly higher confidence) — this mirrors the "expected value at risk" prioritization principle from the fraud case study.

**Reviewer feedback loop:** every human decision (uphold flag / overturn flag / escalate) becomes a labeled training example; track **model-reviewer agreement rate** per category as a core quality metric, and route disagreement cases into targeted retraining/error analysis, exactly as in the fraud review queue.

**Reviewer well-being and calibration:** for high-severity content (violent/graphic content review), rotate reviewers, provide psychological support resources, and regularly calibrate reviewers against gold-standard cases to prevent guideline drift over time — a dimension unique to this domain that senior candidates should mention.

### Precision/Recall Tradeoffs and Business Cost Asymmetry

The cost of a false positive vs. false negative varies wildly *by category* — a critical point that must be surfaced explicitly:

| Category | Cost of false negative (missed violation) | Cost of false positive (wrongly flagged) | Typical policy stance |
|---|---|---|---|
| CSAM / imminent violence | Severe (legal, safety, PR catastrophe) | Moderate (user friction, but usually escalated to human anyway) | Bias heavily toward recall; near-zero tolerance for misses |
| Hate speech / harassment | High (user harm, platform trust, regulatory risk) | Moderate-high (free expression complaints, chilling effect) | Balanced, often region/policy-dependent thresholds |
| Spam | Low-moderate (nuisance) | Low (minor friction) | Optimize for high precision at reasonable recall; err toward fewer false positives |
| Misinformation | Context-dependent (varies by topic sensitivity, e.g., health vs. general) | High (censorship concerns, political sensitivity) | Often paired with labeling/friction (interstitials) rather than outright removal, to reduce false-positive cost |

**Framework: explicit cost-weighted threshold selection**, same structure as the fraud case study:

```
For each category, define:
  cost_FN (missed violation)  -- varies hugely by category severity
  cost_FP (wrongly flagged)   -- includes appeal-handling cost + user trust cost

Threshold chosen to minimize expected cost, not to maximize a
generic F1/accuracy number blind to category-specific cost asymmetry.
```

**Pitfall:** using one global threshold or one global F1 target across all categories ignores this asymmetry and is a common wrong answer — flag this explicitly in your interview response, as interviewers are specifically testing for it.

### Interview Questions

**Q1: Why shouldn't you use a single toxicity score and threshold across all policy violation types?**
A: Different categories carry radically different cost asymmetries (a missed CSAM case is catastrophic; a false-positive spam flag is trivial), so a single global threshold either over-removes low-severity content or under-catches high-severity content — per-category models/thresholds calibrated to each category's cost profile is the correct design.

**Q2: How would you prioritize an overloaded human review queue across many simultaneously-flagged posts?**
A: Rank by an expected-harm score combining severity weight, model confidence, and estimated reach/virality (a viral post accruing views every minute warrants faster review than a low-reach post at similar confidence), mirroring expected-value-at-risk prioritization used in fraud review queues.

**Q3: How do you handle policy differences across regions/jurisdictions (e.g., speech laws differing by country)?**
A: Maintain region-aware policy configuration as a layer on top of (not baked into) the core classifier — the model outputs category scores, and a separate, region-specific policy/threshold engine maps scores to actions, allowing legal/policy changes without retraining the underlying model.

**Q4: What's the risk of training your moderation model purely on past human-reviewer decisions?**
A: You inherit and amplify any systematic reviewer bias or guideline-drift error, and you cap the model's ceiling at historical human accuracy — mitigate with periodic gold-standard calibration sets, reviewer agreement audits, and explicit tracking of appeal-overturn rate as an independent quality signal.

**Q5: How would you measure whether your moderation system is disproportionately flagging certain user groups or content styles (e.g., dialect, satire)?**
A: Slice precision/recall and flag rates by relevant content/user segments (language variety, region, content format) as a fairness guardrail dashboard, and run targeted red-team/adversarial test sets (e.g., known reclaimed/in-group language vs. genuine slurs) to catch systematic misclassification patterns.

**Q6: Why might "friction" (labels, interstitials, reduced distribution) be preferable to outright removal for some categories like misinformation?**
A: Outright removal carries a high false-positive cost (censorship perception, especially on contested/evolving topics) while still needing to reduce harm from spread — friction-based interventions reduce reach/impact without the binary all-or-nothing cost of removal, better matching the asymmetric and evolving cost profile of that category.

**Q7: How would you design the system to catch novel harmful content patterns your classifier has never seen (e.g., a new coordinated harassment tactic)?**
A: Combine supervised per-category classifiers with anomaly/network-based signals (unusual engagement velocity, coordinated posting patterns across many accounts) that don't require the pattern to already exist in labeled training data, and route detected anomalies to human review/policy teams for new-pattern labeling.

**Q8: What SLA differences would you design between a self-harm-risk flag and a spam flag, and why?**
A: Self-harm/imminent-safety flags require near-immediate (minutes, 24/7-staffed, specialized) review given potential real-world urgency, while spam can tolerate an hours-long queue with lower staffing intensity — SLA design must be explicitly tied to real-world severity and time-sensitivity, not a single uniform review latency target.

**Q9: How do you evaluate a multi-label moderation model given that many categories have very few positive examples?**
A: Report per-category precision/recall/PR-AUC rather than a single aggregate accuracy (which would be dominated by the "clean content" majority class), and specifically monitor rare, high-severity categories with dedicated small gold-evaluation sets even if that data is too sparse for standalone model training.

**Q10: How would an appeal process feed back into model improvement?**
A: Every successful appeal (overturned auto-removal or upheld-then-overturned review decision) is a labeled model/reviewer error; track appeal-overturn rate per category as a core quality metric, and route overturned cases into targeted error analysis and retraining data curation, closing the loop similarly to the fraud/review feedback pattern.

**Q11: Should image/video moderation share infrastructure with text moderation, or be entirely separate pipelines?**
A: Share the overall architecture pattern (multi-label scoring → severity-weighted policy engine → tiered human review) but use modality-specific feature extraction (vision/video embeddings vs. text embeddings); many pieces of content are multi-modal (image + caption), so late-stage fusion of modality-specific scores is often needed rather than fully separate, non-communicating pipelines.

---

## Case Study: Design an Ad Click-Through-Rate Prediction System

### Problem Framing

*"Design a system that predicts the probability a user clicks a given ad, for use in an ad auction."*

**Clarify:** where the score is used (ranking ads vs. computing bid price in an auction — the latter demands well-calibrated probabilities, not just correct ranking), latency (typically <10-50ms, extremely tight since it runs inside a real-time auction), scale (billions of ad-impression opportunities/day), feature cardinality (huge categorical spaces: user IDs, ad IDs, publisher IDs).

**North star:** revenue (often via expected value = predicted CTR × bid), subject to advertiser ROI and user experience guardrails. **Guardrails:** latency, ad relevance/quality (avoid pure clickbait optimization), calibration accuracy (critical for auction fairness/pricing).

### Why Calibration Matters More Here Than Almost Anywhere Else

In most classification problems, ranking quality (AUC) is what matters. In ad CTR prediction feeding a **second-price or generalized-second-price auction**, the *actual probability value* is used directly to compute expected value and thus the bid/price:

```
Expected Value (for ranking/pricing) = predicted_CTR * bid_amount

If predicted_CTR is systematically too high or too low (miscalibrated),
advertisers are charged incorrectly and/or the auction ranks ads
in the wrong order relative to true expected value -- this is a direct
revenue and trust problem, not just a minor accuracy issue.
```

Calibration is measured via **calibration curves/reliability diagrams** (predicted probability bucket vs. observed empirical CTR in that bucket) and summarized with metrics like **Expected Calibration Error (ECE)** or the simple **calibration ratio** (predicted CTR / actual CTR, target ≈ 1.0). Any resampling done during training (near-universal here, given extreme class imbalance) must be corrected with an explicit calibration step post-training (see Data Pipeline section's negative-sampling correction formula).

### Feature Crosses

Individual features (e.g., `user_age`, `ad_category`) are often weak predictors alone; **interactions** between features carry most of the signal (e.g., "users aged 18-24 respond well to gaming ads" is a cross between `user_age_bucket` and `ad_category`, not visible from either feature alone).

| Approach | Mechanism | Pros | Cons |
|---|---|---|---|
| Manual/explicit feature crosses | Hand-engineer `feature_A x feature_B` combinations | Interpretable, easy to reason about which crosses matter | Doesn't scale — combinatorial explosion, misses unknown useful crosses |
| Factorization Machines (FM) | Learn low-rank latent factors per feature, model all pairwise interactions implicitly via dot products | Captures all pairwise crosses without manual enumeration, handles sparse categorical data well | Limited to pairwise (2nd-order) interactions by default |
| Wide & Deep | "Wide" linear component with manual crosses (memorization) + "Deep" neural component learning implicit higher-order interactions (generalization) | Combines interpretable memorization with generalization | More complex architecture; two components to tune |
| DeepFM / xDeepFM | Combine FM-style explicit interaction modeling with a deep network, sharing embeddings between both | Captures both low-order explicit and high-order implicit interactions in one model | Higher model complexity, more compute at training and often serving |
| Gradient Boosted Trees (GBDT) | Tree splits naturally capture non-linear feature interactions | Strong baseline, less feature-engineering burden than linear models, interpretable via feature importance | Less naturally suited to very high-cardinality sparse categorical/embeddings compared to neural approaches |

### Embeddings for Massive-Cardinality Categorical Features

Ad systems have some of the highest-cardinality categorical features in all of ML: user IDs (hundreds of millions to billions), ad/creative IDs (millions, high churn as campaigns launch/end), publisher/placement IDs.

```
 High-cardinality ID (e.g., ad_id: 50M distinct values)
             |
             v
     Embedding lookup table (ad_id -> dense vector, e.g., dim=32)
             |
             v
     Concatenated with other feature embeddings --> Model input
```

| Challenge | Mitigation |
|---|---|
| Embedding table memory footprint at billions of IDs | Hashing trick (bucket collisions accepted as a tradeoff), embedding table sharding across serving nodes |
| New IDs appearing constantly (new ads/campaigns launch continuously) | Reserve an out-of-vocabulary/default embedding bucket; fall back to content-based features (advertiser, category) for brand-new ad IDs until enough data accrues |
| ID churn (old ads retiring) | Periodic embedding table pruning/eviction of stale/low-traffic IDs, re-training cadence to reflect current active ad inventory |
| Embedding staleness between retrains | Frequent retraining or online/incremental embedding updates for highly dynamic ID spaces |

### Online Learning Considerations

Ad CTR is one of the most non-stationary prediction problems in production ML — new ad creatives launch hourly, trends shift within a day, and a model trained on last week's data can already be meaningfully stale.

```
   Streaming impression/click events
              |
              v
     Online feature/label join (delayed click attribution window,
     e.g., a click can arrive up to N minutes/hours after impression)
              |
              v
     Incremental model update (e.g., online logistic regression /
     FTRL-Proximal, or frequent mini-batch retraining of a
     neural model) -----------------> Updated model pushed to serving
              |                        (canary rollout + shadow eval)
              v
     Continuous calibration monitoring
```

| Approach | Mechanism | Pros | Cons |
|---|---|---|---|
| Full batch retraining (e.g., daily) | Retrain from scratch or fine-tune on full recent window | Simple, stable, easy to validate before deploy | Staleness within the retrain window; misses intra-day shifts |
| Incremental/online learning (e.g., FTRL, online SGD) | Continuously update model weights as new labeled data streams in | Captures fast-moving trends, adapts within hours | Harder to validate/rollback safely; risk of feedback-loop instability; needs careful monitoring |
| Warm-start frequent retraining | Retrain frequently (e.g., hourly) but initialize from the previous model's weights rather than from scratch | Balance of freshness and stability, common production compromise | Requires robust automated eval gating to catch regressions before each frequent push |

**Delayed label handling in this context** mirrors the fraud case study's delayed-label problem: a click can be attributed to an impression with a lag; the join/attribution window must be fixed and monitored, and models must be evaluated only on matured (fully-attributed) data to avoid training bias from incomplete labels.

### Real-Time Serving: Eligibility Filtering, Auction Mechanics, and Budget Pacing

Everything above is the *modeling* problem (predicting pCTR accurately and well-calibrated). A distinct, equally interview-relevant problem is the *surrounding real-time system* that must run an actual auction, across potentially many advertisers/exchanges, inside an extremely tight round-trip latency budget — this is the angle a senior interviewer probes with "now design the system that actually serves this in a live auction."

**Context:** in real-time bidding (RTB), a bid request typically must be answered within an end-to-end budget of roughly 100ms across the whole chain (publisher/exchange -> DSP -> exchange -> publisher), which leaves only a slice of that — often 10-50ms — for everything the DSP-side system does internally, including the CTR scoring discussed above.

```
   Bid request arrives (user, context, ad slot)
              |
              v
   +--------------------------+
   |  Eligibility / targeting  |  budget remaining? geo/demo targeting
   |  filter (cheap, first)    |  match? frequency cap not exceeded?
   +-------------+--------------+
              |  (cheap filter runs BEFORE any scoring —
              |   same "filter cheaply before ranking expensively"
              |   principle as candidate generation elsewhere in this file)
              v
   +--------------------------+
   |  Candidate ad retrieval    |  which eligible ads/campaigns
   |  (multi-stage, if large)   |  are in play for this slot
   +-------------+--------------+
              |
              v
   +--------------------------+
   |  pCTR scoring              |  the model discussed above
   +-------------+--------------+
              |
              v
   +--------------------------+
   |  Expected value / bid      |  EV = pCTR * advertiser value;
   |  computation                |  bid shading applied here in
   +-------------+--------------+  first-price auctions (see below)
              |
              v
   +--------------------------+
   |  Auction                   |  2nd-price (GSP) or 1st-price
   +-------------+--------------+
              |
              v
   +--------------------------+
   |  Budget pacing check       |  would this spend push the
   +-------------+--------------+  campaign off its pacing curve?
              |
              v
        Bid response
```

**Budget pacing.** Advertisers set a finite budget over a time window (e.g., a daily budget); naively bidding on every eligible opportunity as fast as the model would allow front-loads spend early in the window and starves the campaign of budget for potentially higher-value opportunities later (e.g., evening traffic that converts better for some verticals).

| Pacing strategy | Mechanism | Tradeoff |
|---|---|---|
| Uniform/probabilistic throttling | Randomly skip a fraction of otherwise-eligible bid opportunities so spend rate roughly matches `budget / time remaining` | Simple, but ignores that some remaining opportunities are worth more than others |
| PID-controller-based pacing | Continuously adjust a throttle/multiplier based on the gap between actual and target cumulative spend | Reacts smoothly to under/over-spend without hard cutoffs; standard production approach |
| Predicted-traffic-shape pacing | Use a forecast of the day's traffic/value distribution to bias spend toward historically higher-value time windows within the budget constraint | Most value-efficient, but depends on a reasonably accurate traffic/value forecast model — another adjacent ML problem |

**Bid shading.** In a second-price auction, bidding your true expected value is optimal (you only ever pay the second-highest bid). In a **first-price auction** — now the dominant mechanism on many exchanges — the winner pays *exactly* what they bid, so bidding full expected value systematically overpays whenever the second-highest bid is meaningfully lower. Bid-shading models predict the minimum bid likely needed to win a given opportunity and shade the bid down toward that estimate, capturing more of the surplus between value and clearing price. This is a distinct modeling problem from CTR prediction — it predicts *competitive landscape/clearing price*, not click probability — worth naming explicitly as a separate model in the stack if an interviewer pushes on auction mechanics.

**Pitfalls:**
- Running the full pCTR model over every technically-eligible ad instead of filtering cheaply on budget/targeting/frequency-cap first — the same candidate-generation-before-ranking principle applies to eligibility filtering here.
- Ignoring budget pacing entirely and treating "spend the full budget" as the only constraint, which silently sacrifices higher-value later-window opportunities to lower-value early ones.
- Applying second-price bidding logic (bid = true value) unchanged in a first-price auction context, which overpays every time the true clearing price is below the bid.

### Interview Questions

**Q1: Why is calibration a first-class requirement for CTR prediction but often secondary for, say, a recommendation ranking model?**
A: The CTR prediction's raw probability is used directly in auction pricing/expected-value computation (predicted_CTR × bid), so a miscalibrated score directly distorts advertiser billing and ad ranking fairness — a recommendation ranker typically only needs relative ordering to be correct, not the absolute probability value, since nothing downstream multiplies the raw score into a price.

**Q2: How would you correct for calibration distortion introduced by negative downsampling during training?**
A: Apply the known correction formula relating the sampled and true positive rates (`p_calibrated = p / (p + (1-p)/w)` where `w` is the negative sampling rate) or fit a post-hoc calibration layer (Platt scaling/isotonic regression) on a held-out, non-downsampled validation set before deploying scores into the auction.

**Q3: Why can't you just manually engineer all useful feature crosses instead of using FM/deep models?**
A: The space of potentially useful pairwise (and higher-order) interactions among hundreds of categorical features grows combinatorially, and many valuable crosses are non-obvious a priori — automated interaction modeling (FM, deep networks) discovers useful crosses at a scale manual engineering cannot match, while a few high-value manually-engineered crosses often still get added to a "wide" memorization component for interpretability and strong baseline performance.

**Q4: How do you serve a model with an embedding table too large to fit on a single machine?**
A: Shard the embedding table across multiple serving nodes (e.g., by hashing the ID), with a request-routing layer that fetches the relevant embedding shard(s) per request, or use parameter-server architectures designed for distributed embedding lookup at serving time.

**Q5: A brand-new ad campaign launches with a new ad_id and zero historical clicks — how does your model score it reasonably?**
A: Fall back to an out-of-vocabulary/default embedding combined with available content-level features (advertiser ID, category, creative attributes) that generalize across ads, rather than relying purely on an ad_id embedding that has no learned signal yet — analogous to cold-start handling in the recommendation case study.

**Q6: What's the risk of retraining a CTR model too infrequently in this domain specifically?**
A: Ad inventory and trends shift within hours (campaign launches/ends, seasonal/news-driven demand spikes), so infrequent retraining causes the model to systematically mis-score fresh inventory, directly costing ad revenue and degrading advertiser trust in pricing — much faster staleness impact than in slower-moving domains like general content recommendation.

**Q7: How would you decide between full batch retraining vs. true online/incremental learning?**
A: Weigh the freshness gain of online learning against its operational risk (harder rollback, feedback-loop instability, more complex monitoring); a common middle ground is frequent warm-started batch retraining (e.g., hourly, initialized from the prior model) with strong automated evaluation gates before each push, reserving full online learning for cases where even hourly retraining can't keep pace with drift.

**Q8: How do delayed click attributions bias your training data if not handled carefully?**
A: If you train on data before a fixed attribution window closes, some impressions that will eventually be clicked are still labeled as non-clicks (because the click hasn't arrived yet), systematically underestimating true CTR for recently-logged impressions — exactly mirroring the delayed-label problem in fraud detection; fix with a defined maturation window and training only on fully-attributed data.

**Q9: Why would a business insist on ad quality/relevance guardrails rather than pure CTR-revenue optimization?**
A: Pure CTR optimization can be gamed by clickbait-style ad creative that generates clicks without genuine user interest or advertiser value, degrading long-term user trust and advertiser ROI/retention even as short-term click revenue rises — hence guardrails on post-click engagement/conversion and user complaint/hide rates alongside the CTR north star.

**Q10: How would embedding table staleness manifest as a production bug, and how would you catch it?**
A: Ads/users with IDs added after the last embedding table refresh get default/OOV embeddings and are systematically mis-scored (usually under-scored, losing them auction impressions/revenue) until the next retrain; catch it by monitoring the OOV/default-embedding hit rate over time as a leading indicator, alerting if it climbs above a normal baseline.

**Q11: How is the "wide" component of a Wide & Deep model different in purpose from the "deep" component, and why keep both?**
A: The wide linear component with explicit hand-crafted crosses excels at memorizing specific, sparse, highly predictive feature combinations seen frequently in training data (exploitation of known patterns), while the deep component generalizes to unseen feature combinations via learned dense representations (exploration of latent structure) — combining both captures benefits neither achieves alone.

**Q12: What monitoring would specifically catch a calibration regression in production before it causes billing/pricing problems?**
A: Continuous tracking of the calibration ratio (aggregate predicted CTR / observed actual CTR) sliced by key segments (advertiser, ad format, traffic source) with automated alerting on drift beyond a tolerance band, since a global-average calibration ratio near 1.0 can still mask segment-level miscalibration that matters for specific advertisers' pricing fairness.

**Q13: Why must eligibility/targeting filtering happen before pCTR scoring rather than after?**
A: Scoring every technically-in-the-catalog ad with a full model and only then discarding ones that violate budget/targeting/frequency caps wastes the majority of your tight per-request compute budget on ads that could never have won anyway; filtering cheaply first (the same principle as candidate generation before ranking) leaves the latency budget for scoring only the ads that are actually eligible to win.

**Q14: What breaks if you deploy a bidding strategy tuned for a second-price auction into a first-price auction environment unchanged?**
A: Bidding your full expected value is optimal under second-price rules (you only ever pay the next-highest bid) but is systematically overpaying under first-price rules (you pay exactly what you bid); without a bid-shading adjustment, the campaign burns budget faster than necessary and gets a worse effective ROI per dollar spent.

**Q15: How would you tell whether a campaign's poor performance is a modeling problem (bad pCTR) or a pacing problem (bad budget allocation across the day)?**
A: Compare calibration and ranking quality (pCTR vs. actual CTR, AUC) against the campaign's spend curve over the day; a well-calibrated model with spend concentrated in a clearly suboptimal time window points to a pacing issue, while miscalibration or poor ranking regardless of time-of-day points to the model itself.

**Q16: Why is a PID-controller-style pacing approach generally preferred over simple uniform throttling in production?**
A: Uniform/random throttling targets an average spend rate but reacts slowly to actual over/under-spend as the day progresses; a PID-style controller continuously measures the gap between actual and target cumulative spend and adjusts the throttle multiplier accordingly, smoothing out both early bursts and late-day budget exhaustion without hard cutoffs.

**Q17: What's the risk of building a bid-shading model that reuses the same features and training signal as the pCTR model?**
A: Bid shading is predicting a fundamentally different target — the likely clearing price / minimum winning bid in a competitive auction — not click probability; reusing pCTR's feature set and labels conflates two different prediction problems and is likely to produce a poorly-fit shading model even if the underlying pCTR model is excellent.

**Q18: How would frequency capping interact with the eligibility filter, and why does it need to be checked before scoring, not after?**
A: Frequency capping (limiting how many times a given user sees ads from the same campaign in a window) is a hard eligibility constraint like budget or targeting — checking it before scoring avoids wasting compute on an ad that would be rejected regardless of its predicted CTR, and avoids the correctness risk of a race condition where the cap check happens too late relative to concurrent bid requests for the same user.

**Q19: Within a ~100ms end-to-end OpenRTB round trip, how would you allocate the internal DSP-side latency budget across eligibility filtering, scoring, and the auction/pacing check?**
A: Order stages from cheapest-and-highest-elimination-rate to most expensive: eligibility/targeting/frequency-cap filtering first (a few ms, eliminates the majority of the catalog cheaply), candidate retrieval and pCTR scoring next (the bulk of the remaining internal budget), and the auction/pacing check last (a fast, cheap lookup), leaving margin for the network hops that consume the rest of the ~100ms external SLA.

**Q20: Why can't budget pacing be solved purely by the pCTR/value model itself?**
A: The value model scores a single opportunity in isolation; pacing is a sequential resource-allocation problem across the *entire remaining time window* of a finite budget, which requires tracking cumulative spend state and a control mechanism (throttle) that adjusts moment-to-moment — it's a different problem shape than per-request prediction and is implemented as a separate service/control loop layered on top of the scoring model's output.

---

## Case Study: Design an ML Platform from Scratch

### Problem Framing

*"Design an ML platform to support many teams at your company building and deploying models."*

This is a fundamentally different question from every case study above: those ask you to design **one system** that solves **one product problem**. This one asks you to design the **meta-infrastructure** that many teams use to build and ship *their* systems — the interviewer is probing platform/infra judgment, self-serve thinking, and governance, not any single model's accuracy. Staff-level ML infra and MLE interviews use this prompt specifically because it exposes whether a candidate has only ever consumed platform infrastructure or has also had to design/own it.

**Clarify:** how many teams and roughly how many models/use-cases today and at a 1-2 year horizon; mix of batch vs. real-time serving needs across teams; current state (greenfield vs. migrating off ad hoc notebooks/scripts); build vs. buy appetite (managed cloud ML platforms vs. custom); compliance/data-residency constraints; how much of this the platform team owns vs. what individual model teams still own themselves.

**North star:** typically **time from "model works in a notebook" to "model is safely serving production traffic"** (dev velocity/lead time) — a platform's entire value proposition is compressing this. **Guardrails:** cost (compute + storage, often the single biggest line item), reproducibility (can any past model be rebuilt from its recorded lineage?), security/access control, and adoption (a platform nobody uses has failed regardless of its technical elegance).

### Why This Isn't the Feature Store Section, Bigger

The Data and Feature Pipeline Design section earlier in this file describes the **feature store** as one component within a single system's data pipeline. This case study is broader: the feature store is *one box* in a much larger platform that also has to solve training orchestration, experiment tracking, model versioning/lineage, deployment automation, and cross-team governance — treat the feature store architecture from that section as a given building block here rather than re-deriving it.

### Platform Layers and Architecture

```
   +-----------------------------------------------------------------+
   |                      Governance & Access Control                 |
   |     (data access policy, PII handling, audit trail, cost/usage    |
   |      attribution per team)                                       |
   +-----------------------------------------------------------------+
              |                    |                     |
              v                    v                     v
   +------------------+  +-------------------+  +----------------------+
   | Data Layer         |  | Feature Store      |  | Experimentation /     |
   | (warehouse/lake,   |  | (offline + online,  |  | Notebook Layer         |
   | CDC ingestion)     |  | as in Section 2)   |  | (dev sandboxes,        |
   +--------+-----------+  +---------+----------+  | experiment tracking)   |
            |                        |               +-----------+-----------+
            +------------+-----------+                           |
                         v                                        v
              +-------------------------+          +-------------------------+
              | Training Infrastructure  |<-------->| Model Registry          |
              | (job orchestration, GPU  |          | (versioning, lineage,   |
              |  scheduling, distributed |          |  automated eval gate)   |
              |  training, HPO service)  |          +------------+-------------+
              +-------------------------+                        |
                                                                   v
                                                       +-------------------------+
                                                       | CI/CD for ML            |
                                                       | (validation, shadow/    |
                                                       |  canary deploy)         |
                                                       +------------+-------------+
                                                                    |
                                                                    v
                                                       +-------------------------+
                                                       | Serving Infrastructure  |
                                                       | (online low-latency,    |
                                                       |  batch scoring, multi-  |
                                                       |  model hosting)         |
                                                       +------------+-------------+
                                                                    |
                                                                    v
                                                       +-------------------------+
                                                       | Monitoring/Observability|
                                                       | (drift, cost, latency,  |
                                                       |  per-team dashboards)   |
                                                       +-------------------------+
```

| Layer | Core responsibility | Common building blocks named in interviews |
|---|---|---|
| Data layer | Reliable, governed access to raw and processed data | Data warehouse/lakehouse, CDC pipelines |
| Feature store | Consistent offline/online feature definitions across teams | As detailed in the Data and Feature Pipeline Design section |
| Experimentation layer | Reproducible dev environment + experiment metadata tracking | Managed notebooks, MLflow/Weights&Biases-style experiment tracking |
| Training infrastructure | Scalable, schedulable compute for training jobs | Job orchestrator (Kubeflow/Airflow-style), GPU scheduling, distributed training framework, hyperparameter tuning service |
| Model registry | Versioned artifacts with lineage and an approval gate | Registry storing model + training-data version + code version + eval results, with a promotion workflow |
| CI/CD for ML | Automated validation before any production exposure | Automated eval-gate tests, shadow traffic, canary rollout |
| Serving infrastructure | Low-latency online serving and/or batch scoring, shared across teams | Multi-model serving containers, autoscaling, per-model resource isolation |
| Monitoring/observability | Drift, performance, and cost visibility per model/team | Feature/prediction drift dashboards, latency/error dashboards, cost attribution |
| Governance | Access control, compliance, and organizational guardrails | Data access policy engine, PII/audit logging, feature/model ownership registry |

### Model Registry, Lineage, and Approval Gates

A model artifact alone is not reproducible or auditable without recording **what produced it**:

```
Model version record:
  - model artifact (weights/binary)
  - training code version (git commit)
  - training data version / feature-store snapshot version
  - hyperparameters + training job config
  - offline eval results (metrics + guardrail checks)
  - approval status (pending / approved / rejected) + approver
```

The **automated eval gate** — every candidate model version is scored against the same held-out set and guardrail checks as the currently-deployed model before it's eligible for promotion — is the platform-level generalization of the "offline eval as a pre-deployment gate" pattern seen throughout this file's case studies (search ranking regression tests, RAG golden sets, fraud/CTR calibration checks); the platform's job is to make that gate a **standard, enforced step for every team**, not something each team reinvents inconsistently.

### Multi-Tenancy and Governance

| Concern | Design answer |
|---|---|
| Feature/model sprawl (duplicate, inconsistently-named features across teams) | A feature registry with enforced naming/ownership conventions and a discovery UI, so teams reuse rather than re-derive existing features |
| Resource contention (one team's training job starving another's) | Per-team compute quotas and priority scheduling in the training job orchestrator |
| Cost attribution | Tag all compute/storage usage by team/project for chargeback and to make the platform's own cost visible and defensible |
| Differing latency/scale needs across teams | Serving infrastructure exposes multiple serving tiers (e.g., shared low-QPS multi-model hosting vs. dedicated high-QPS deployment) rather than one-size-fits-all |
| Access control to sensitive data/features | Governance layer enforces data access policy at the feature-store and data-layer level, not left to individual teams' discretion |

### Build vs. Buy

| Option | Pros | Cons |
|---|---|---|
| Fully managed platform (SageMaker, Vertex AI, Databricks) | Fast to stand up, less operational burden, vendor-maintained | Less flexibility for non-standard workflows, potential vendor lock-in, cost at scale can exceed custom infra |
| Open-source components assembled in-house (Kubeflow, MLflow, Feast, etc.) | More control/customization, avoids lock-in | Significant integration and ongoing-maintenance engineering cost, slower initial time-to-value |
| Fully custom in-house platform | Maximum fit to internal workflows/scale | Highest build and maintenance cost; only justified at large scale with genuinely non-standard requirements |

**Framework tip:** state explicitly that the right default answer for most companies below a certain scale is "start with managed/open-source building blocks, build custom only where a specific bottleneck justifies it" — proposing a fully custom platform on day one for a company with 3 ML use-cases is a common overengineering mistake interviewers watch for, the same "MVP vs. steady-state" instinct from the framework section applied at the platform level.

**Pitfalls:**
- Building generalized platform abstractions before there are 2-3 concrete real use-cases to generalize from, producing a platform that fits no one's actual workflow well.
- Underinvesting in feature-store governance (naming/ownership conventions, discovery), leading to duplicate and subtly-inconsistent feature definitions across teams — reintroducing the exact training-serving-skew and consistency problems the feature store was built to prevent, just at the cross-team level.
- Treating the model registry as a passive artifact store rather than an enforced gate — if promotion to production doesn't require passing the eval gate, the registry provides auditability but not actual safety.
- Optimizing purely for platform-team convenience (fewer supported configurations) at the expense of adoption — a platform that's technically clean but doesn't fit real teams' workflows gets bypassed via shadow infrastructure.

### Interview Questions

**Q1: How is "design an ML platform" different from "design a recommendation/fraud/search system," and how should that change your first two minutes?**
A: Every other case study asks you to design one system solving one product problem with its own users and metrics; this asks you to design shared infrastructure serving *many* teams' different systems, so the first clarifying questions should be about scale of adoption (how many teams/use-cases), current maturity, and build-vs-buy appetite, rather than about a single product's latency/QPS.

**Q2: What's the platform-level north star metric, and why is it different from any single model's accuracy?**
A: Typically time-to-production for a new model (dev velocity/lead time) or overall model reliability/uptime across the platform — the platform's value is measured by how much it accelerates and de-risks *every* team's work, not by any individual model's offline metric.

**Q3: Why shouldn't you design a fully custom platform from scratch for a company with only a handful of ML use-cases?**
A: The engineering cost of building and maintaining custom training/serving/registry infrastructure is large and mostly fixed regardless of use-case count, so at low use-case volume that cost isn't justified relative to assembling managed/open-source building blocks; custom investment becomes justified once a specific, identified bottleneck (cost, latency, a workflow no existing tool supports) demands it.

**Q4: What's the risk of building a model registry that stores artifacts but doesn't enforce an eval gate before promotion?**
A: It gives you auditability (you can see what was deployed and when) but not actual safety — nothing prevents a team from promoting a regressed or miscalibrated model straight to production, which defeats a primary reason platforms centralize deployment in the first place.

**Q5: How would you prevent every team from re-deriving slightly different versions of the same underlying feature (e.g., "user 30-day purchase count")?**
A: Enforce feature registry governance — required naming conventions, ownership metadata, and a discovery/search UI so teams can find and reuse existing feature definitions before creating new ones — without this, a platform's shared feature store degenerates into the same skew/inconsistency problems it was built to solve, just duplicated across teams instead of within one pipeline.

**Q6: A training job from one team is starving another team's job of GPU resources — how does the platform prevent this?**
A: Per-team compute quotas and priority-based scheduling in the training job orchestrator, so one team's workload can't unboundedly consume shared capacity; this is a multi-tenancy concern specific to platform design that doesn't arise when designing a single team's system in isolation.

**Q7: How do you decide what belongs in the shared platform vs. what stays owned by individual model teams?**
A: Centralize what benefits from consistency and reuse across teams (feature definitions, the eval-gate/promotion process, core serving infra, monitoring conventions) and leave what's genuinely product-specific (model architecture choice, feature engineering logic for a specific model, business-logic re-ranking) to individual teams — centralizing everything creates a rigid platform that can't fit varied use-cases.

**Q8: What does "reproducibility" mean concretely for a model trained a year ago, and how does the platform guarantee it?**
A: Being able to reconstruct the exact model given only its registry record — its training code version, the exact feature-store/data snapshot version used, and hyperparameters — which requires the platform to version not just code but also data/feature snapshots (tying back to point-in-time correctness) and store that lineage alongside every registered model version, not just the final weights.

**Q9: How would you introduce this platform to an organization that already has several teams shipping models with ad hoc, team-specific infrastructure?**
A: Start with the highest-friction, most broadly-shared pain point (commonly the feature store, since inconsistent feature pipelines cause the most cross-team bugs) as an MVP, prove adoption and value with 1-2 pilot teams, and expand platform scope incrementally rather than mandating a full-stack migration on day one — mirroring the "MVP then scale" instinct from the general framework, applied organizationally.

**Q10: How do you measure whether the platform is actually succeeding, beyond "we built it"?**
A: Track adoption rate (fraction of new models launched using the platform vs. bypassing it via shadow infrastructure), the north-star dev-velocity metric (time from model-ready to production), and cost per model served — low adoption despite a technically complete platform is itself a critical failure signal, not just a nice-to-have metric.

**Q11: Why is cost attribution per team/project a first-class platform requirement rather than an afterthought?**
A: Shared infrastructure obscures who is actually consuming compute/storage; without per-team cost tagging, the platform team can neither defend its own budget nor give individual teams the feedback needed to notice and fix inefficient training/serving usage, and cost overruns get discovered too late to act on.

**Q12: How does the model registry's "approval gate" concept generalize the pre-deployment checks seen in earlier single-system case studies?**
A: Search ranking's golden-query regression test, RAG's golden eval set, and the fraud/CTR calibration checks are all instances of the same underlying pattern — score every candidate model against a fixed, held-out benchmark and guardrails before it can replace the current production model; the platform's job is to make this a standardized, enforced step available to every team by default, instead of each team building (or forgetting to build) their own version of it.

---

## Rapid-Fire Interview Q&A

**Q1: What are the two stages in a typical large-scale recommendation/search system, and why split them?**
A: Candidate generation (high recall, cheap, over the full corpus) and ranking (high precision, expensive, over a small candidate set) — splitting avoids running an expensive personalized model over millions/billions of items within a tight latency budget.

**Q2: What is training-serving skew?**
A: A mismatch between feature values computed during training (offline) and at inference (online), usually from divergent code paths, causing production accuracy to underperform offline evaluation.

**Q3: What is point-in-time correctness in feature engineering?**
A: The guarantee that a feature's value used to train a model only reflects information that would genuinely have been available at the moment of prediction, preventing label leakage from future data.

**Q4: Name three cold-start scenarios in a recommender system.**
A: New user (no interaction history), new item (no interaction history), and new user+item simultaneously (e.g., marketplace launch).

**Q5: What is position bias and why is it a problem for ranking models trained on click logs?**
A: Higher-ranked items get more clicks regardless of true relevance; training naively on such logs teaches the model to partly predict position rather than relevance, reinforcing the existing ranking policy.

**Q6: What is a two-tower model used for?**
A: Embedding-based retrieval — separately encoding user/query and item into a shared vector space so similarity search (ANN) can retrieve relevant candidates from huge catalogs cheaply.

**Q7: Why is ROC-AUC often a poor primary metric for extremely imbalanced problems like fraud detection?**
A: It can look excellent even when precision at any usable operating threshold is poor, because it's insensitive to the base rate; precision-recall metrics reflect the operational reality of extreme imbalance better.

**Q8: What is sample-and-hold, and which case study is it central to?**
A: Deliberately letting a random fraction of flagged (would-be-blocked) cases through to observe true outcomes, avoiding survivorship bias in evaluation; central to fraud detection (and applicable to content moderation).

**Q9: What's the difference between pointwise, pairwise, and listwise learning-to-rank?**
A: Pointwise scores documents independently; pairwise learns relative ordering between pairs; listwise directly optimizes a whole-list ranking metric (e.g., NDCG) — listwise is the most metric-aligned but hardest to train.

**Q10: Why do modern search/RAG systems use hybrid retrieval (lexical + dense) instead of just embeddings?**
A: Dense embeddings capture semantic similarity but miss exact-match needs (IDs, rare terms, precise phrases) that lexical/BM25 retrieval handles well; combining both covers more failure modes.

**Q11: What is a feature store, and what two "sides" does it typically have?**
A: A system that stores feature definitions and values consistently for both training (offline store) and serving (online, low-latency store), using shared transformation logic to prevent skew.

**Q12: Why is calibration especially critical in ad CTR prediction for auctions?**
A: The raw predicted probability is multiplied directly into expected-value/bid calculations used for ranking and pricing ads — miscalibration directly distorts revenue and advertiser billing, not just ranking order.

**Q13: What's the standard correction for calibration distortion caused by negative downsampling?**
A: Rescale predicted probabilities with the known sampling-rate correction formula, or fit a post-hoc calibration layer (Platt scaling/isotonic regression) on non-downsampled validation data.

**Q14: What is a contextual bandit, and why is it used in recommendation exploration?**
A: An algorithm that balances exploring uncertain items/actions against exploiting known-good ones, using contextual features to personalize the explore/exploit decision, preventing popularity feedback loops.

**Q15: What is time-to-first-token (TTFT), and why does it matter more than total generation time for perceived chatbot latency?**
A: The time before the first output token is streamed to the user; with streaming UIs, users perceive responsiveness based on TTFT, not total completion time, so optimizing pipeline stages before generation (retrieval, guardrails) is high-leverage for perceived latency.

**Q16: In content moderation, why use per-category classifiers/thresholds instead of one global toxicity score?**
A: Different violation categories (CSAM vs. spam) carry wildly different false-positive/false-negative cost asymmetries; a single global threshold can't correctly balance costs that differ by orders of magnitude across categories.

**Q17: What's the difference between concept drift and adversarial drift, using fraud detection as the example?**
A: Concept drift is a passive, natural change in the underlying data distribution over time; adversarial drift (as in fraud) is an active, intentional adaptation by bad actors specifically to evade the current model, requiring faster/more defensive countermeasures.

**Q18: Why is chunking strategy so important in RAG systems?**
A: It directly determines whether a complete, self-sufficient answer unit is retrievable — too-small chunks fragment needed context across boundaries, too-large chunks dilute retrieval precision and waste the LLM's context budget.

**Q19: What is a semantic cache and why is it useful for high-QPS LLM systems?**
A: A cache keyed by embedding similarity of the query (not exact string match) so paraphrased-but-equivalent repeated questions can reuse a cached response, cutting cost and latency for high-traffic systems with repeated intent.

**Q20: What is the "north star + guardrails" metric framework?**
A: Optimizing a single primary business metric (north star) while monitoring several other metrics (guardrails — latency, fairness, diversity, cost) that must not regress, preventing the optimizer from gaming the north star at the expense of other important outcomes.

**Q21: Why should retrieved documents in a RAG pipeline be treated as untrusted data rather than trusted instructions?**
A: A malicious or compromised document could contain embedded text designed to hijack the LLM's behavior (prompt injection); structurally separating instructions from retrieved content and running injection-detection guardrails mitigates this risk.

**Q22: What's the key architectural reason embeddings for extremely high-cardinality IDs (ad_id, user_id) need special handling at serving time?**
A: The full embedding table can be too large for a single machine's memory, requiring sharding across serving nodes, plus explicit handling (OOV bucket, content-based fallback) for constantly-appearing new IDs that have no learned embedding yet.

**Q23: What's the danger of designing an ML system as a one-way pipeline instead of a closed loop?**
A: You miss feedback effects — the model's own outputs shape future training data (e.g., exposure bias in recommenders, self-reinforcing decisions in fraud/moderation review), and without an explicit feedback/monitoring loop, these effects go undetected until they cause visible harm.

**Q24: Why do senior candidates explicitly quantify latency/QPS/scale before proposing a model architecture?**
A: Because these constraints determine what's even feasible (batch vs. real-time compute, model size, caching strategy, single-stage vs. multi-stage architecture) — proposing a model before establishing constraints risks designing something technically infeasible or wildly over/under-engineered for the actual requirements.

**Q25: What is inverse propensity weighting used for in ranking system design?**
A: Correcting training data for the bias introduced by the logging/serving policy itself (e.g., position bias, exposure bias) by weighting examples inversely to their probability of being observed/shown, producing less biased offline training signal.

**Q26: Why is raw watch time a poor training label for a short-form video feed ranker?**
A: It's confounded with video duration — longer videos passively accumulate more watched seconds even at equal or lower quality — so completion rate/% watched combined with an explicit negative (skip/hide) signal isolates true content quality far better than raw seconds watched.

**Q27: Why does short-form video feed ranking need session-level, near-real-time re-ranking that a typical homepage recsys doesn't?**
A: Sessions are long and dense with sub-second implicit feedback on nearly every item, so the value of "what the user just reacted to a few seconds ago" decays too fast for an hourly/daily-refreshed user embedding to capture — the system needs a fast, short-TTL session state store feeding the next item's scoring within the same sitting.

**Q28: Why is a full learning-to-rank model typically the wrong fit for autocomplete's per-keystroke ranking pass?**
A: Autocomplete fires on every keystroke across all active users, a far higher request volume than full search's per-submitted-query rate, so anything heavier than a cheap precomputed-candidate blend would blow both the latency budget and the cost budget at that volume.

**Q29: What's the core data structure behind most production autocomplete systems, and why precompute results at each node?**
A: A trie or finite-state transducer (FST) over the historical query vocabulary, with the top-K completions precomputed and cached at each prefix node — precomputation trades an offline batch cost for an O(prefix-length) request-time lookup, which is what makes sub-50ms suggestion latency feasible.

**Q30: What's the difference between designing "a recommendation system" and designing "an ML platform," and why do interviewers ask both?**
A: The former designs one system solving one product's prediction problem; the latter designs the shared infrastructure (feature store, training orchestration, model registry, serving, monitoring, governance) that many teams use to build *their* systems — interviewers use the platform question specifically to probe infra/staff-level judgment (self-serve design, multi-tenancy, build-vs-buy) that a single-system question doesn't surface.

**Q31: In real-time ad bidding, what problem does budget pacing solve that the pCTR/value model alone cannot?**
A: The value model scores one bid opportunity in isolation, while pacing is a sequential resource-allocation problem across an advertiser's entire remaining budget window — without a separate pacing control loop (e.g., PID-based throttling), naive full-speed bidding front-loads spend early and starves potentially higher-value later opportunities of budget.

**Q32: Why does bid shading matter more in first-price auctions than in second-price auctions?**
A: In a second-price auction the winner pays the next-highest bid, so bidding true expected value is already optimal; in a first-price auction the winner pays exactly what they bid, so bidding full value systematically overpays whenever the true clearing price is lower — bid shading predicts that clearing price and shades the bid down to capture more surplus.
