<!-- markdownlint-disable -->
<h1 align="center">
    Best-of AI &amp; Machine Learning with Python
    <br>
</h1>

<p align="center">
    <strong>🏆&nbsp; A ranked list of awesome Python libraries for machine learning and the AI era — LLMs, agents, RAG, and inference. Updated weekly.</strong>
</p>

<p align="center">
    <a href="https://github.com/best-of-lists/best-of" title="Best-of-badge"><img src="http://bit.ly/3o3EHNN"></a>
    <a href="#Contents" title="Project Count"><img src="https://img.shields.io/badge/projects-{project_count}-blue.svg?color=5ac4bf"></a>
    <a href="#Contribution" title="Contributions are welcome"><img src="https://img.shields.io/badge/contributions-welcome-green.svg"></a>
    <a href="https://github.com/binbinao/best-of-ml-python/releases" title="Best-of Updates"><img src="https://img.shields.io/github/release-date/binbinao/best-of-ml-python?color=green&label=updated"></a>
</p>

This curated list contains {project_count} awesome open-source projects with a total of {stars_count} stars grouped into {category_count} categories — covering everything from classic ML frameworks to LLM frameworks, agent orchestration, RAG &amp; vector databases, and LLM inference &amp; serving. All projects are ranked by a project-quality score, which is calculated based on various metrics automatically collected from GitHub and different package managers. If you like to add or update projects, feel free to open an [issue](https://github.com/binbinao/best-of-ml-python/issues/new/choose), submit a [pull request](https://github.com/binbinao/best-of-ml-python/pulls), or directly edit the [projects.yaml](https://github.com/binbinao/best-of-ml-python/edit/main/projects.yaml). Contributions are very welcome!

---

<p align="center">
     🧙‍♂️&nbsp; Discover other <a href="https://best-of.org">best-of lists</a> or create <a href="https://github.com/best-of-lists/best-of/blob/main/create-best-of-list.md">your own</a>.<br>
    🙏&nbsp; Based on <a href="https://github.com/ml-tooling/best-of-ml-python">ml-tooling/best-of-ml-python</a> — many thanks to the original maintainers.
</p>

---



## How to Use This List in the AI-Coding Era

AI can now write most of the code in this list from a one-paragraph prompt. What it cannot do is decide **what the business problem actually is**, or catch the failure mode where generated code runs perfectly, passes every metric, and still solves the wrong problem. This list is organized accordingly: it is a **map of solution paradigms and the principles you must hold in order to direct and review AI-generated ML code** — not an API directory.

**Practical posture per category:**

1. **Start from the business question, not the library.** Each category below states the business problems it answers. Pick the paradigm before letting an AI pick the tool.
2. **Know the principle that gates the category.** Every category names the concepts you need to review AI output competently — e.g. target leakage in modeling, distribution shift in deployment, recall/precision tradeoffs in retrieval.
3. **Use the review checklists before accepting AI code.** For the core categories, [REVIEW-CHECKLISTS.md](REVIEW-CHECKLISTS.md) lists the questions to ask an AI assistant *before* its code is trusted: How was the data split? What does the metric optimize versus what the business needs? Where could leakage hide?
4. **Mechanical correctness is machine-gated here.** Repository-level checks (`scripts/validate.py`, CI) catch structural errors so that human attention stays on judgment: problem framing, data legitimacy, and whether the evaluation measures the business outcome.

The ranked ordering within categories still reflects project-quality scores (stars, downloads, contributors) — a proxy for ecosystem maturity, not a statement of fitness for your problem. Fitness is your call; that is the point.
