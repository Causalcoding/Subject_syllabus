# Interview Prep Syllabus — Data Scientist / ML Engineer / AI Engineer

An extremely thorough, self-contained interview-prep syllabus split across 18 Markdown files. Each file covers one subject end-to-end: core topics → subtopics → concepts (definitions, math, code, pitfalls) → a dedicated "Interview Questions" set per topic → a closing "Rapid-Fire Q&A" for last-minute review.

Built for three overlapping tracks — **Data Scientist (DS)**, **Machine Learning Engineer (MLE)**, **AI Engineer (AIE)** — each file's intro paragraph flags which track(s) weight that subject most heavily.

## How the files are organized

Files are numbered in a rough learning order (foundations → classical ML → deep learning → modern AI → systems/production → non-technical), but each is self-contained — jump to whatever you need.

| # | File | Covers | Weighted toward |
|---|---|---|---|
| 01 | [Mathematics_for_ML_and_AI](01_Mathematics_for_ML_and_AI.md) | Linear algebra, calculus/optimization, information theory | All three |
| 02 | [Probability_and_Statistics](02_Probability_and_Statistics.md) | Probability, distributions, inference, Bayesian stats, sampling/A-B testing | DS |
| 03 | [Python_Programming_and_DSA](03_Python_Programming_and_DSA.md) | Python fundamentals, pandas/numpy, complexity, data structures, algorithms | MLE / AIE |
| 04 | [SQL_and_Databases](04_SQL_and_Databases.md) | SQL fundamentals/advanced, database design/theory, NoSQL | DS / MLE |
| 05 | [Data_Wrangling_EDA_and_Visualization](05_Data_Wrangling_EDA_and_Visualization.md) | Cleaning, EDA, visualization | DS |
| 06 | [Feature_Engineering_and_Model_Evaluation](06_Feature_Engineering_and_Model_Evaluation.md) | Encoding, scaling, feature selection, dimensionality reduction, imbalanced data, leakage, CV/splitting, metrics, calibration, hyperparameter tuning | DS / MLE |
| 07 | [Machine_Learning_Fundamentals](07_Machine_Learning_Fundamentals.md) | Classical ML algorithms — regression, classification, ensembles, clustering, association rules, interpretability | All three |
| 08 | [Deep_Learning](08_Deep_Learning.md) | Neural net foundations, training, CNNs, RNNs/LSTMs, attention/Transformers, practical DL | MLE / AIE |
| 09 | [Natural_Language_Processing](09_Natural_Language_Processing.md) | Text preprocessing, representations, classical + transformer-era NLP tasks | DS / AIE |
| 10 | [Computer_Vision](10_Computer_Vision.md) | Image fundamentals, CNN architectures, detection, segmentation, ViT, video/face/OCR/3D | MLE / AIE |
| 11 | [Generative_AI_and_LLMs](11_Generative_AI_and_LLMs.md) | VAE/GAN/diffusion, Transformer/LLM architecture, fine-tuning/RLHF/DPO, decoding, evaluation, MoE, multimodal | AIE |
| 12 | [AI_Agents_RAG_and_LLM_Applications](12_AI_Agents_RAG_and_LLM_Applications.md) | Prompt engineering, embeddings/vector search, RAG, agents, LLM app engineering | AIE |
| 13 | [Reinforcement_Learning](13_Reinforcement_Learning.md) | MDPs, DP/tabular methods, deep RL, bandits, offline RL, RL for LLMs (RLHF) | AIE (niche for DS/MLE) |
| 14 | [MLOps_and_Model_Deployment](14_MLOps_and_Model_Deployment.md) | Experiment tracking, CI/CD, serving, monitoring/drift, feature stores, incident response | MLE |
| 15 | [Big_Data_Distributed_Systems_and_Cloud](15_Big_Data_Distributed_Systems_and_Cloud.md) | Distributed computing, storage, streaming, cloud ML platforms | MLE |
| 16 | [System_Design_for_ML_AI](16_System_Design_for_ML_AI.md) | Structured framework + case studies (recsys, search, fraud, RAG chatbot, ad CTR, ML platform, etc.) | MLE / AIE (senior) |
| 17 | [Responsible_AI_Ethics_and_Fairness](17_Responsible_AI_Ethics_and_Fairness.md) | Bias/fairness, privacy, explainability/governance, AI safety, copyright/liability | All three |
| 18 | [Behavioral_Case_Studies_and_Product_Sense](18_Behavioral_Case_Studies_and_Product_Sense.md) | Behavioral frameworks, product-sense/metrics cases, communication, negotiation | DS (behavioral applies to all) |

## Cross-file scoping (avoid confusion)

Some topics could plausibly live in more than one file. To avoid duplicating the same content twice, each topic has one home:

- **Model evaluation, cross-validation, dimensionality reduction (PCA/t-SNE/UMAP), imbalanced-data handling, feature encoding, hyperparameter tuning** → all in **06**, not 07. File 07 covers only the algorithms themselves.
- **RLHF/PPO/DPO general RL mechanics** → owned by **13**; file 11 applies it to LLMs specifically without re-deriving it.
- **RAG system design at the architecture/case-study level** → **16**; component-level depth (chunking, reranking, vector DB internals) → **12**.
- **Offline SHAP/LIME interpretability** → **07**; **production-time** explainability/audit logging → **14**.
- **Prompt injection / reward hacking / hallucination** appear in three files at three different angles: safety-and-policy framing (**17**), architectural defenses (**12**), and RL/model-quality mechanics (**13**/**11**) — read the relevant one for your angle, minor overlap is intentional.

## How to use this

1. Pick a file for the day/session — each is long enough to be a standalone study block.
2. Read the concept sections first; don't skip straight to Q&A — the concept explanations contain the "why," which is what interviewers actually probe.
3. Use each topic's "Interview Questions" subsection to self-test after reading that topic.
4. Use the "Rapid-Fire Q&A" at the end of each file for spaced-repetition review closer to an actual interview.
5. For system-design and behavioral rounds (files 16 and 18), practice saying the answers out loud, not just reading them.

## Provenance / quality notes

Every file went through two passes:
1. **Initial write** — full topic coverage generated per the scope above.
2. **Audit pass** — each file was independently re-read in full, checked for technical correctness (formulas, code, claims), and patched for gaps against what a rigorous interviewer would expect, without duplicating content that belongs in another file per the scoping table above.

If you spot an error or a gap, it's worth fixing directly in the file — treat this as a living document, not a fixed reference.
