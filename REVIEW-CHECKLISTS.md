# Review Checklists — Directing & Reviewing AI-Generated ML Code

These checklists operationalize the stance in the README: AI writes the code; you own the **problem definition, the principles, and the review**. Use them *before* accepting AI-generated code in each domain. They are questions to ask — of the AI, of the data, and of yourself.

Format per category: the business questions it answers → the principles that gate correctness → the review questions to ask before trusting AI output.

---

## Machine Learning Frameworks (ml-frameworks)

**Answers:** "I have a prediction/optimization problem — what computational substrate do I build on?"

**Gating principles:** generalization vs. memorization, bias-variance, train/serve skew.

**Review before trusting AI code:**
- [ ] What exactly is being predicted, and does the label exist at prediction time in production? (If not: target leakage.)
- [ ] How is the data split — random, temporal, grouped by entity? Does the split match how the system will be used?
- [ ] Which metric is optimized, and how does it map to the business outcome? (AUC ≠ revenue ≠ user retention.)
- [ ] Is the same preprocessing applied identically at train and inference time?
- [ ] What baseline does this beat — a constant, a rule, last week's value?

## LLM Frameworks & Toolkits (llm-frameworks)

**Answers:** "I need to access, fine-tune, quantize, or align a large language model."

**Gating principles:** what fine-tuning actually changes (behavior, not knowledge), quantization's accuracy/cost curve, eval-before-and-after.

**Review before trusting AI code:**
- [ ] Why fine-tuning instead of prompting or RAG? Can the AI justify the choice in terms of the failure being fixed?
- [ ] What eval set proves the fine-tune/quantize helped — measured on your tasks, not the vendor's benchmarks?
- [ ] How much accuracy is lost at the chosen quantization level, on *your* distribution?
- [ ] Is training data deduplicated and license-compatible for your use?

## Agent & Orchestration Frameworks (agent-frameworks)

**Answers:** "I need LLMs to take multi-step actions, call tools, or coordinate."

**Gating principles:** agents compound errors; tool boundaries are security boundaries; observability before autonomy.

**Review before trusting AI code:**
- [ ] What happens when a step fails or hallucinates a tool call — retry, escalate to a human, or silently continue?
- [ ] Which actions are reversible? Who/what approves side-effecting operations (writes, purchases, emails)?
- [ ] Is there a trace of every decision path for post-hoc debugging?
- [ ] Is this actually an agent problem, or would a deterministic workflow suffice? (AI defaults to agents too readily.)
- [ ] What is the cost/latency ceiling, and is it enforced?

## RAG & Vector Databases (rag-vectordb)

**Answers:** "I need answers grounded in my own documents/data."

**Gating principles:** retrieval quality caps generation quality; chunking and embeddings shape what is findable; recall vs. precision is a business choice.

**Review before trusting AI code:**
- [ ] How is retrieval quality measured separately from generation quality (hit rate, MRR on a labeled query set)?
- [ ] What chunking strategy was chosen and why — how does it handle tables, long sections, cross-references?
- [ ] Is the embedding model appropriate for the language/domain of the corpus?
- [ ] What happens when retrieval returns nothing or garbage — does the system admit it or confabulate?
- [ ] Are permissions/freshness of the source data respected at query time?

## LLM Inference & Serving (inference-serving)

**Answers:** "I need to serve model inference at throughput/latency targets."

**Gating principles:** batching vs. latency tradeoff, memory bandwidth as the real bottleneck, SLOs before benchmarks.

**Review before trusting AI code:**
- [ ] What are the actual SLOs (p50/p99 latency, throughput, cost per request)? Decisions without SLOs are decoration.
- [ ] Are the benchmarks run on your model, your sequence-length mix, your hardware?
- [ ] What happens under load spike — queue, shed, scale? Is backpressure explicit?
- [ ] How are failures isolated so one bad request cannot poison the worker pool?

## Hyperparameter Optimization & AutoML (hyperopt)

**Answers:** "I want to search configuration space systematically instead of by hand."

**Gating principles:** search amplifies whatever the objective measures — including leakage; validation budget is a resource.

**Review before trusting AI code:**
- [ ] What is the objective being optimized, and is the validation protocol leak-proof (nested splits for hyperparameter search)?
- [ ] Does the search budget reflect the variance of the metric? (Small gains within noise = nothing.)
- [ ] Are early results on the same distribution as late results — or did the search overfit the validation set?
- [ ] Is a human-readable report of searched space and findings produced?

## Reinforcement Learning (reinforcement-learning)

**Answers:** "My problem is sequential decision-making with delayed rewards."

**Gating principles:** RL is sample-hungry and brittle; sim-to-real and off-policy correctness are the usual killers.

**Review before trusting AI code:**
- [ ] Why RL rather than a supervised proxy or a heuristic policy? What is the justification?
- [ ] Does the simulator's dynamics match reality closely enough for the learned policy to transfer?
- [ ] How is exploration constrained away from unacceptable states (safety, cost, brand)?
- [ ] What is the offline evaluation story before any online deployment?

## Model Interpretability (interpretability)

**Answers:** "I must explain/debug/validate why a model decides."

**Gating principles:** explanation fidelity vs. plausibility; explanations are model-level evidence, not causal truth.

**Review before trusting AI code:**
- [ ] Does the explanation method's assumption match the model class (SHAP for trees ≠ for deep nets)?
- [ ] Are explanations validated against known behavior (sanity checks, perturbation tests)?
- [ ] Is the audience considered — regulator, engineer, end user need different explanations?
- [ ] Could the explanation be gamed if exposed?

## Distributed Machine Learning (distributed-ml)

**Answers:** "My model/data doesn't fit one machine, or training is too slow."

**Gating principles:** communication dominates; scaling efficiency vs. absolute speed; failure recovery.

**Review before trusting AI code:**
- [ ] Is the bottleneck actually compute, or data loading / communication? (Profile before scaling.)
- [ ] What parallelism strategy (data/model/pipeline) matches the model and cluster?
- [ ] How are checkpoints and preemption handled — can a 3-day run survive a node failure?
- [ ] What is the throughput scaling curve — measured, not assumed?

## Time Series Data (time-series-data)

**Answers:** "I need to forecast or detect anomalies in ordered, temporal data."

**Gating principles:** temporal causality; leakage through future information; seasonal/structural change.

**Review before trusting AI code:**
- [ ] Is every feature computed only from data available at forecast time? (Lagging features are the classic leak.)
- [ ] Does the validation use rolling-origin splits, never random shuffling?
- [ ] What baseline does it beat — seasonal naive, ARIMA? Strong baselines humble fancy models.
- [ ] How does the system detect regime change (holiday, outage, product launch) vs. retrain on stale patterns?

## Recommender Systems (recommender-systems)

**Answers:** "I need to rank/suggest items to users."

**Gating principles:** offline ranking metrics ≠ business outcomes; feedback loops bias training data.

**Review before trusting AI code:**
- [ ] Is the training data biased by what the previous system showed (survivorship/position bias)? Any correction (IPS, counterfactual)?
- [ ] Does offline NDCG/Recall@K correlate with online A/B results — has that been established?
- [ ] Beyond accuracy: how are diversity, novelty, and exploitation of popularity handled?
- [ ] Are cold-start users/items handled explicitly?

## Adversarial Robustness (adversarial)

**Answers:** "My model faces malicious inputs; I need to test/verify robustness."

**Gating principles:** robustness is threat-model-specific; undefended models fail quietly.

**Review before trusting AI code:**
- [ ] What is the threat model — white-box, black-box, perturbation budget, physical-world constraints?
- [ ] Are evaluations against adaptive attacks, not just fixed attack suites? (Defense claims need adaptive evaluation.)
- [ ] Is robustness tested on the deployed pipeline (preprocessing included), not the model alone?

## Privacy Machine Learning (privacy-ml)

**Answers:** "I must train/infer under privacy constraints — federated, encrypted, or differentially private."

**Gating principles:** privacy has a budget and a proof, not a vibe; utility loss is the cost of the guarantee.

**Review before trusting AI code:**
- [ ] What is the formal guarantee — DP epsilon/delta per release, or just "federated"? (Federation alone is not privacy.)
- [ ] Is the privacy budget accounted cumulatively across queries/releases?
- [ ] What utility loss does the privacy mechanism cause, measured on the real task?
- [ ] Are there leakage channels outside the model (gradients, telemetry, logs) that the mechanism doesn't cover?

---

## Categories where the review is mostly engineering hygiene

For data plumbing and infrastructure categories — **tabular, data-containers, data-loading, data-pipelines, web-scraping, db-clients, gpu-utilities, model-serialisation, ml-experiments, data-viz, ocr, distributed-ml tooling** — the AI-coding risks are less about ML principles and more about: correctness of schemas/transforms, idempotency and retry semantics, cost of compute, and silent data loss. Ask: *what proves this transform is correct on the full distribution, not just the sample I tested?*

Domain-data categories — **image, audio, graph, geospatial, financial, medical, chinese-nlp, nlp** — inherit the review questions of the modeling category they feed (usually ml-frameworks or llm-frameworks), plus domain-specific legality: licensing of scraped data, PHI handling, regulatory constraints on financial signals.

Framework-extension categories — **tensorflow-utils, jax-utils, sklearn-utils, pytorch-utils** — inherit the **ml-frameworks** checklist plus their framework's own contract semantics (noted per-category in the list).

The remaining specialized categories — **probabilistics, nn-search, model-hub, chinese-nlp** — carry their own gating principles in their subtitles above; treat those as the checklist until dedicated sections are warranted.
