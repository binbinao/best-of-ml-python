# AI 时代策展批次审核清单(Task 5 Step 1-2)

- **生成日期:** 2026-10-03
- **数据日期:** 2026-10-03(星标、许可证、活跃度均当日核实)
- **数据来源:** GitHub REST API `repos/*` 与 GraphQL `licenseInfo.spdxId`(经已认证 `gh` CLI;后者即 best-of-generator 实际读取的许可证字段);PyPI 包存在性经 `pypi.org/pypi/<id>/json`;conda 包存在性经 `api.anaconda.org/package/conda-forge/<id>`
- **用途:** 用户逐条审核。勾选 = 同意该条处置;待审核条目勾选 = 同意按所附 YAML 入库。**未获批准前不入库任何条目;本文件不改动 `projects.yaml`。**
- **门槛与隐藏规则(Task 4 结论 B + 生成器行为核实):**
  - 全局 `min_stars: 300` —— 星标 <300 的候选 EXCLUDE(threshold);
  - 生成器另会隐藏:许可证缺失或 GitHub 报 NOASSERTION 的项目(`require_license` 默认 True,`projects.yaml` 未改写即生效);许可证 SPDX 不在 `allowed_licenses` 白名单(Unlicense/Apache-2.0/MIT/BSD-3-Clause/BSD-2-Clause/ISC/MPL-2.0/LGPL-2.1)的项目;最后提交超 12 个月(`project_dead_months` 默认 12)的项目。
- **候选计数核对:** 任务书标题称「38 候选」,但 spec §3.3 与任务书实际枚举为 **31 个**(推理/服务 4 + Agent/编排 8 + RAG/向量库 7 + 模型/训练 10 + SDK 2)。本清单处置全部 31 个枚举候选,无虚构补充。

## 处置汇总

| 状态 | 数量 | 候选 |
|---|---|---|
| SKIP(已在库,零迁移) | 5 | litellm、faiss、milvus、haystack、deepspeed |
| EXCLUDE | 2 | autogen(许可证+维护模式)、weaviate(许可证不可探测+Python 客户端<300) |
| 待审核 | 24 | inference-serving 4 · agent-frameworks 4 · rag-vectordb 4 · llm-frameworks 12 · model-hub 0 |

全部 31 候选星标均 ≥300;2 个 EXCLUDE 均非门槛原因,而是生成器许可证过滤/生态归属原因(详见条目)。

### 汇总表

| # | 候选 | github_id | ⭐(2026-10-03) | 状态 | 拟归类 | 一句话理由 |
|---|---|---|---|---|---|---|
| 1 | vLLM | vllm-project/vllm | 93,095 | 待审核 | inference-serving | 高吞吐 LLM 推理与服务引擎 |
| 2 | SGLang | sgl-project/sglang | 36,742 | 待审核 | inference-serving | 高性能 LLM 推理引擎+服务框架 |
| 3 | llama.cpp | ggml-org/llama.cpp | 130,191 | 待审核 | inference-serving | 本地 LLM 推理引擎;pypi 挂 Python 绑定 |
| 4 | Ollama | ollama/ollama | 182,080 | 待审核 | inference-serving | 本地模型运行器;pypi 挂官方 Python 客户端 |
| 5 | LangChain | langchain-ai/langchain | 147,396 | 待审核 | agent-frameworks | LLM 应用编排/代理框架 |
| 6 | LlamaIndex | run-llama/llama_index | 52,392 | 待审核 | rag-vectordb | 数据接入/索引/检索为主的数据框架 |
| 7 | AutoGen | microsoft/autogen | 61,248 | EXCLUDE | — | CC-BY-4.0 不在白名单;maintenance mode |
| 8 | CrewAI | crewAIInc/crewAI | 59,303 | 待审核 | agent-frameworks | 多智能体编排框架 |
| 9 | smolagents | huggingface/smolagents | 29,660 | 待审核 | agent-frameworks | HF 极简 code-agent 框架 |
| 10 | litellm | BerriAI/litellm | 60,074 | SKIP | — | 已在库(nlp);⚠️ 许可证检测已变 NOASSERTION |
| 11 | instructor | 567-labs/instructor | 13,971 | 待审核 | llm-frameworks | LLM 结构化输出库(模型访问层) |
| 12 | pydantic-ai | pydantic/pydantic-ai | 20,378 | 待审核 | agent-frameworks | 基于 Pydantic 的代理框架 |
| 13 | chromadb | chroma-core/chroma | 29,430 | 待审核 | rag-vectordb | AI 原生嵌入式向量数据库 |
| 14 | faiss | facebookresearch/faiss | 41,020 | SKIP | — | 已在库(nn-search);⚠️ 现有 pypi/conda_id 均坏 |
| 15 | weaviate | weaviate/weaviate | 16,861 | EXCLUDE | — | 许可证 NOASSERTION 会被隐藏;Python 客户端 228★ |
| 16 | milvus | milvus-io/milvus | 46,310 | SKIP | — | 已在库(nn-search) |
| 17 | LanceDB | lancedb/lancedb | 11,588 | 待审核 | rag-vectordb | 面向 ML 的向量数据库(Lance 列存) |
| 18 | txtai | neuml/txtai | 12,990 | 待审核 | rag-vectordb | 嵌入驱动 RAG/语义搜索一体化管线 |
| 19 | haystack | deepset-ai/haystack | 26,646 | SKIP | — | 已在库(nlp);⚠️ pypi_id 指向已弃用 v1 包 |
| 20 | diffusers | huggingface/diffusers | 34,646 | 待审核 | llm-frameworks | 扩散模型访问/微调(审核点:非 LLM) |
| 21 | Whisper | openai/whisper | 109,897 | 待审核 | llm-frameworks | 语音识别基础模型库(审核点:非 LLM) |
| 22 | faster-whisper | SYSTRAN/faster-whisper | 25,681 | 待审核 | llm-frameworks | Whisper 高速重实现(审核点:非 LLM) |
| 23 | FunASR | modelscope/FunASR | 20,572 | 待审核 | llm-frameworks | 语音识别工具包(审核点:非 LLM) |
| 24 | MLX | ml-explore/mlx | 28,637 | 待审核 | llm-frameworks | Apple silicon 数组框架(审核点:或归 ml-frameworks) |
| 25 | unsloth | unslothai/unsloth | 77,158 | 待审核 | llm-frameworks | LLM 微调加速 |
| 26 | peft | huggingface/peft | 21,750 | 待审核 | llm-frameworks | 参数高效微调(LoRA 等) |
| 27 | trl | huggingface/trl | 19,440 | 待审核 | llm-frameworks | LLM 对齐/RLHF 训练 |
| 28 | DeepSpeed | microsoft/DeepSpeed | 43,174 | SKIP | — | 已在库(distributed-ml) |
| 29 | bitsandbytes | bitsandbytes-foundation/bitsandbytes | 8,510 | 待审核 | llm-frameworks | k-bit 量化(PyTorch) |
| 30 | openai-python | openai/openai-python | 31,737 | 待审核 | llm-frameworks | OpenAI 官方 Python SDK |
| 31 | anthropic-sdk-python | anthropics/anthropic-sdk-python | 3,943 | 待审核 | llm-frameworks | Anthropic 官方 Python SDK |

---

## inference-serving(4 待审核)

- [ ] **vLLM** — 待审核
  - `vllm-project/vllm` · ⭐ 93,095(2026-10-03,GitHub API)· Apache-2.0 · 活跃(最后推送 2026-10-03)
  - 包核实:pypi `vllm` ✓ · conda `conda-forge/vllm` ✓
  - 归类理由:PagedAttention 高吞吐 LLM 推理与服务引擎,类目副旨「高吞吐推理引擎、服务化部署」的直接实例。

```yaml
  - name: vLLM
    github_id: vllm-project/vllm
    pypi_id: vllm
    conda_id: conda-forge/vllm
    category: inference-serving
    labels: ["inference", "llm"]
```

- [ ] **SGLang** — 待审核
  - `sgl-project/sglang` · ⭐ 36,742 · Apache-2.0 · 活跃(2026-10-03)
  - 包核实:pypi `sglang` ✓ · conda 无对应包(conda-forge/sglang 404,不设 conda_id)
  - 归类理由:RadixAttention 高性能 LLM 推理引擎 + 服务框架。

```yaml
  - name: SGLang
    github_id: sgl-project/sglang
    pypi_id: sglang
    category: inference-serving
    labels: ["inference", "llm"]
```

- [ ] **llama.cpp** — 待审核
  - `ggml-org/llama.cpp` · ⭐ 130,191 · MIT · 活跃(2026-10-03)
  - 包核实:pypi `llama-cpp-python` ✓(Python 绑定,仓库 abetlen/llama-cpp-python,10,638★)· conda `conda-forge/llama-cpp-python` ✓
  - 归类理由:本地 CPU/GPU LLM 推理引擎。**跨类/归属说明:** 核心为 C/C++,沿用库内 Milvus 先例(github_id 指核心仓库、pypi_id 指 Python 客户端);备选方案是 github_id 直接指 `abetlen/llama-cpp-python`,请审核定夺。

```yaml
  - name: llama.cpp
    github_id: ggml-org/llama.cpp
    pypi_id: llama-cpp-python
    conda_id: conda-forge/llama-cpp-python
    category: inference-serving
    labels: ["inference", "llm"]
```

- [ ] **Ollama** — 待审核
  - `ollama/ollama` · ⭐ 182,080 · MIT · 活跃(2026-10-03)
  - 包核实:pypi `ollama` ✓(官方 Python 客户端,仓库 ollama/ollama-python,10,569★)· conda `conda-forge/ollama` 为 Go CLI 包,与 Python 客户端语义不符,不设 conda_id
  - 归类理由:本地模型运行器与服务,Python 生态经官方客户端访问;同 Milvus 先例挂核心仓库。

```yaml
  - name: Ollama
    github_id: ollama/ollama
    pypi_id: ollama
    category: inference-serving
    labels: ["inference", "llm"]
```

## agent-frameworks(4 待审核,1 EXCLUDE)

- [ ] **LangChain** — 待审核
  - `langchain-ai/langchain` · ⭐ 147,396 · MIT · 活跃(2026-10-03)
  - 包核实:pypi `langchain` ✓ · conda `conda-forge/langchain` ✓
  - **跨类说明:** 也可归 llm-frameworks;按现代主用例(编排、LangGraph 代理)归 agent-frameworks。

```yaml
  - name: LangChain
    github_id: langchain-ai/langchain
    pypi_id: langchain
    conda_id: conda-forge/langchain
    category: agent-frameworks
    labels: ["agent", "llm"]
```

- [ ] **CrewAI** — 待审核
  - `crewAIInc/crewAI` · ⭐ 59,303 · MIT · 活跃(2026-10-03)
  - 包核实:pypi `crewai` ✓ · conda `conda-forge/crewai` ✓
  - 归类理由:角色化多智能体编排框架,类目主旨直接实例。

```yaml
  - name: CrewAI
    github_id: crewAIInc/crewAI
    pypi_id: crewai
    conda_id: conda-forge/crewai
    category: agent-frameworks
    labels: ["agent"]
```

- [ ] **smolagents** — 待审核
  - `huggingface/smolagents` · ⭐ 29,660 · Apache-2.0 · 活跃(2026-09-30)
  - 包核实:pypi `smolagents` ✓ · conda `conda-forge/smolagents` ✓
  - 归类理由:Hugging Face 极简 code-agent 框架。

```yaml
  - name: smolagents
    github_id: huggingface/smolagents
    pypi_id: smolagents
    conda_id: conda-forge/smolagents
    category: agent-frameworks
    labels: ["agent", "huggingface"]
```

- [ ] **pydantic-ai** — 待审核
  - `pydantic/pydantic-ai` · ⭐ 20,378 · MIT · 活跃(2026-10-03)
  - 包核实:pypi `pydantic-ai` ✓ · conda `conda-forge/pydantic-ai` ✓
  - 归类理由:基于 Pydantic 的类型安全代理框架。

```yaml
  - name: pydantic-ai
    github_id: pydantic/pydantic-ai
    pypi_id: pydantic-ai
    conda_id: conda-forge/pydantic-ai
    category: agent-frameworks
    labels: ["agent", "llm"]
```

- [ ] **AutoGen** — EXCLUDE(许可证 + 维护模式)
  - `microsoft/autogen` · ⭐ 61,248 · **CC-BY-4.0**(GraphQL licenseInfo 核实)· 最后推送 2026-04-15
  - 理由:① CC-BY-4.0 不在 `projects.yaml` 的 `allowed_licenses` 白名单,生成器会置 `show=False`,入库也不会展示;② 官方 README 已宣布 maintenance mode(不再新增功能,社区维护),官方指定后继为 `microsoft/agent-framework`(MIT,13,920★,2026-10-03 数据)——可作为未来批次候选,本批不扩容。
  - 若用户仍想收录,需先扩充 `allowed_licenses` 并接受 maintenance-mode 项目,属配置决策,超出本批。

## rag-vectordb(4 待审核,1 EXCLUDE,3 SKIP)

- [ ] **LlamaIndex** — 待审核
  - `run-llama/llama_index` · ⭐ 52,392 · MIT · 活跃(2026-10-01)
  - 包核实:pypi `llama-index` ✓ · conda `conda-forge/llama-index` ✓
  - **跨类说明:** 具备代理能力,但主功能是数据接入/索引/检索(RAG 数据框架),归 rag-vectordb。

```yaml
  - name: LlamaIndex
    github_id: run-llama/llama_index
    pypi_id: llama-index
    conda_id: conda-forge/llama-index
    category: rag-vectordb
    labels: ["rag", "llm"]
```

- [ ] **chromadb** — 待审核
  - `chroma-core/chroma` · ⭐ 29,430 · Apache-2.0 · 活跃(2026-10-02)
  - 包核实:pypi `chromadb` ✓ · conda `conda-forge/chromadb` ✓
  - 归类理由:AI 原生嵌入式向量数据库,RAG 管线标配存储。

```yaml
  - name: chromadb
    github_id: chroma-core/chroma
    pypi_id: chromadb
    conda_id: conda-forge/chromadb
    category: rag-vectordb
    labels: ["rag"]
```

- [ ] **LanceDB** — 待审核
  - `lancedb/lancedb` · ⭐ 11,588 · Apache-2.0 · 活跃(2026-10-03)
  - 包核实:pypi `lancedb` ✓ · conda `conda-forge/lancedb` ✓
  - 归类理由:面向 ML 的向量数据库(Lance 列式存储,磁盘级向量检索)。

```yaml
  - name: LanceDB
    github_id: lancedb/lancedb
    pypi_id: lancedb
    conda_id: conda-forge/lancedb
    category: rag-vectordb
    labels: ["rag"]
```

- [ ] **txtai** — 待审核
  - `neuml/txtai` · ⭐ 12,990 · Apache-2.0 · 活跃(2026-10-02)
  - 包核实:pypi `txtai` ✓ · conda `conda-forge/txtai` ✓
  - 归类理由:嵌入驱动的 RAG/语义搜索/LLM 工作流一体化管线。

```yaml
  - name: txtai
    github_id: neuml/txtai
    pypi_id: txtai
    conda_id: conda-forge/txtai
    category: rag-vectordb
    labels: ["rag"]
```

- [ ] **weaviate** — EXCLUDE(许可证不可探测 + Python 客户端低于门槛)
  - `weaviate/weaviate` · ⭐ 16,861 · **NOASSERTION**(GraphQL licenseInfo 核实,同生成器读取路径)· 活跃(2026-10-02)
  - 理由:① 生成器对 NOASSERTION 许可证直接跳过赋值(`github_integration.py`:「if licenses is noassertion, then it is not provided」)→ license 为空 → `require_license` 默认 True → `show=False`,入库也不会展示;② 核心为 Go 服务,其 Python 客户端仓库 `weaviate/weaviate-python-client` 仅 228★(<300,且 PyPI 元数据声明的 BSD-3-Clause 只在客户端仓库生效,核心仓库许可证仍不可探测)。若未来 Weaviate 修复 GitHub 许可证探测,可重新纳入。

- [ ] **faiss** — SKIP(已在库)
  - `facebookresearch/faiss` 已在 `projects.yaml:2412`,当前 category: **nn-search**。零迁移原则,不入库。
  - ⚠️ 附带发现(不在本批范围,建议列入后续数据修复):现有条目 `pypi_id: pymilvus` 为复制粘贴错误(应为 `faiss-cpu`),`conda_id: conda-forge/faiss` 包不存在(应为 `conda-forge/faiss-cpu`,两者均已于 2026-10-03 核实)。

- [ ] **milvus** — SKIP(已在库)
  - `milvus-io/milvus` 已在 `projects.yaml:2407`,当前 category: **nn-search**。零迁移原则,不入库。

- [ ] **haystack** — SKIP(已在库)
  - `deepset-ai/haystack` 已在 `projects.yaml:3295`,当前 category: **nlp**。零迁移原则,不入库。
  - ⚠️ 附带发现:`pypi_id: haystack` 指向已弃用的 v1 包(现行包为 `haystack-ai`),建议后续修复。

## llm-frameworks(12 待审核,2 SKIP)

- [ ] **instructor** — 待审核
  - `567-labs/instructor` · ⭐ 13,971 · MIT · 活跃(2026-10-01)
  - 包核实:pypi `instructor` ✓ · conda `conda-forge/instructor` ✓
  - **跨类/ID 说明:** 任务书给定 `jxnl/instructor` 已迁移,现行规范 ID 为 `567-labs/instructor`(旧 ID 经 GitHub 重定向可达,查重已按两个 ID 核对,均不在库)。功能为 LLM 结构化输出(Pydantic 模型直出),属模型访问层而非编排框架,故归 llm-frameworks 而非 agent-frameworks。

```yaml
  - name: instructor
    github_id: 567-labs/instructor
    pypi_id: instructor
    conda_id: conda-forge/instructor
    category: llm-frameworks
    labels: ["llm"]
```

- [ ] **diffusers** — 待审核
  - `huggingface/diffusers` · ⭐ 34,646 · Apache-2.0 · 活跃(2026-10-02)
  - 包核实:pypi `diffusers` ✓ · conda `conda-forge/diffusers` ✓
  - **审核点:** 类目副标题为「large language models」,而 diffusers 是图像/视频扩散模型库——归入依据是「模型访问/微调」主功能与 Hugging Face 生态聚合;如不接受,可改归现有 `image` 类目或不收,请定夺。

```yaml
  - name: diffusers
    github_id: huggingface/diffusers
    pypi_id: diffusers
    conda_id: conda-forge/diffusers
    category: llm-frameworks
    labels: ["huggingface"]
```

- [ ] **Whisper** — 待审核
  - `openai/whisper` · ⭐ 109,897 · MIT · 活跃(2026-08-31)
  - 包核实:pypi `openai-whisper` ✓ · conda `conda-forge/openai-whisper` ✓
  - **审核点:** 按任务书「模型访问/微调」主功能归 llm-frameworks;严格说 ASR 基础模型库并非 LLM,同 diffusers 一并请定夺。

```yaml
  - name: Whisper
    github_id: openai/whisper
    pypi_id: openai-whisper
    conda_id: conda-forge/openai-whisper
    category: llm-frameworks
```

- [ ] **faster-whisper** — 待审核
  - `SYSTRAN/faster-whisper` · ⭐ 25,681 · MIT · 活跃(2026-10-01)
  - 包核实:pypi `faster-whisper` ✓ · conda 无对应包(404,不设 conda_id)
  - **审核点:** 同 Whisper(CTranslate2 高速重实现,模型访问主功能)。

```yaml
  - name: faster-whisper
    github_id: SYSTRAN/faster-whisper
    pypi_id: faster-whisper
    category: llm-frameworks
```

- [ ] **FunASR** — 待审核
  - `modelscope/FunASR` · ⭐ 20,572 · MIT · 活跃(2026-10-02)
  - 包核实:pypi `funasr` ✓ · conda 无对应包(404,不设 conda_id)
  - **审核点:** 语音识别工具包,同 Whisper 一并请定夺。

```yaml
  - name: FunASR
    github_id: modelscope/FunASR
    pypi_id: funasr
    category: llm-frameworks
```

- [ ] **MLX** — 待审核
  - `ml-explore/mlx` · ⭐ 28,637 · MIT · 活跃(2026-10-02)
  - 包核实:pypi `mlx` ✓ · conda `conda-forge/mlx` ✓
  - **跨类/ID 说明:** 任务书给定 `ml-exp/mlx` 不存在(404),GitHub org 实为 `ml-explore`,已改用规范 ID(查重核对不在库)。MLX 是 Apple silicon 数组框架(NumPy 风格 + ML),是本地 LLM 训练/推理(mlx-lm 等)的底座;**更精确的归类或是现有 `ml-frameworks`**(通用 ML 框架),本清单按「新类目优先」原则归 llm-frameworks,请审核定夺。

```yaml
  - name: MLX
    github_id: ml-explore/mlx
    pypi_id: mlx
    conda_id: conda-forge/mlx
    category: llm-frameworks
```

- [ ] **unsloth** — 待审核
  - `unslothai/unsloth` · ⭐ 77,158 · Apache-2.0 · 活跃(2026-10-03)
  - 包核实:pypi `unsloth` ✓ · conda `conda-forge/unsloth` ✓
  - 归类理由:LLM 微调加速(显存减半/速度翻倍),「微调」主旨直接实例。

```yaml
  - name: unsloth
    github_id: unslothai/unsloth
    pypi_id: unsloth
    conda_id: conda-forge/unsloth
    category: llm-frameworks
    labels: ["llm"]
```

- [ ] **peft** — 待审核
  - `huggingface/peft` · ⭐ 21,750 · Apache-2.0 · 活跃(2026-10-02)
  - 包核实:pypi `peft` ✓ · conda `conda-forge/peft` ✓
  - 归类理由:参数高效微调(LoRA/QLoRA/Prefix-tuning),「微调」主旨直接实例。

```yaml
  - name: peft
    github_id: huggingface/peft
    pypi_id: peft
    conda_id: conda-forge/peft
    category: llm-frameworks
    labels: ["huggingface", "llm"]
```

- [ ] **trl** — 待审核
  - `huggingface/trl` · ⭐ 19,440 · Apache-2.0 · 活跃(2026-10-03)
  - 包核实:pypi `trl` ✓ · conda `conda-forge/trl` ✓
  - 归类理由:LLM 对齐/RLHF 后训练库,「对齐」主旨直接实例。

```yaml
  - name: trl
    github_id: huggingface/trl
    pypi_id: trl
    conda_id: conda-forge/trl
    category: llm-frameworks
    labels: ["huggingface", "llm"]
```

- [ ] **bitsandbytes** — 待审核
  - `bitsandbytes-foundation/bitsandbytes` · ⭐ 8,510 · MIT · 活跃(2026-09-07)
  - 包核实:pypi `bitsandbytes` ✓ · conda `conda-forge/bitsandbytes` ✓
  - **ID 说明:** 任务书给定 `bitsandbytes/bitsandbytes` 已迁移至 `bitsandbytes-foundation/bitsandbytes`(旧 ID 重定向可达,查重已按两个 ID 核对,均不在库)。k-bit 量化库,「量化」主旨直接实例。

```yaml
  - name: bitsandbytes
    github_id: bitsandbytes-foundation/bitsandbytes
    pypi_id: bitsandbytes
    conda_id: conda-forge/bitsandbytes
    category: llm-frameworks
    labels: ["pytorch", "llm"]
```

- [ ] **openai-python** — 待审核
  - `openai/openai-python` · ⭐ 31,737 · Apache-2.0 · 活跃(2026-10-02)
  - 包核实:pypi `openai` ✓ · conda `conda-forge/openai` ✓
  - **归类说明:** 五个新类目无 SDK 专用类;官方模型访问 SDK 是「模型访问」层的最直接实例,归 llm-frameworks。

```yaml
  - name: openai-python
    github_id: openai/openai-python
    pypi_id: openai
    conda_id: conda-forge/openai
    category: llm-frameworks
    labels: ["llm"]
```

- [ ] **anthropic-sdk-python** — 待审核
  - `anthropics/anthropic-sdk-python` · ⭐ 3,943 · MIT · 活跃(2026-09-30)
  - 包核实:pypi `anthropic` ✓ · conda `conda-forge/anthropic` ✓
  - 归类理由:同 openai-python(官方模型访问 SDK)。

```yaml
  - name: anthropic-sdk-python
    github_id: anthropics/anthropic-sdk-python
    pypi_id: anthropic
    conda_id: conda-forge/anthropic
    category: llm-frameworks
    labels: ["llm"]
```

- [ ] **litellm** — SKIP(已在库)
  - `BerriAI/litellm` 已在 `projects.yaml:3605`,当前 category: **nlp**,labels: []。零迁移原则,不入库。
  - ⚠️ 附带发现(重要):其 GitHub 许可证探测已从 MIT(2025-10-30 历史快照 `history/2025-10-30_projects.csv` 记录 license=MIT)变为 **NOASSERTION**(2026-10-03 经 GraphQL 核实,同生成器读取路径)——下次生成器运行时该现有条目可能被隐藏。属既有条目处置问题,超出本批范围,建议单独决策。

- [ ] **DeepSpeed** — SKIP(已在库)
  - `microsoft/DeepSpeed` 已在 `projects.yaml:2217`,当前 category: **distributed-ml**,labels: ["pytorch"]。零迁移原则,不入库。
  - 附注:仓库已迁移至 `deepspeedai/DeepSpeed`(43,174★,旧 ID 重定向可达);现有条目沿用旧 ID 仍有效,如需规范化属后续数据修复。

## model-hub(0 条,说明)

本批 31 个候选无一归属 model-hub。spec §3.2 已预案:现有 `huggingface_hub`(`huggingface/huggingface_hub`,已在 `projects.yaml:4115`,category: **model-serialisation**)按零迁移原则不移动,故 model-hub 首批空置,类目定义保留待未来批次使用。

---

## 审核操作说明

1. 逐条勾选表示同意该处置;不同意者在条目旁批注修改意见(改类目/标签/ID 直接改 YAML 文本)。
2. 全部审核完成后,勾选的待审核条目按上方类目顺序拼接,追加到 `projects.yaml` 的 `projects:` 列表末尾(neurolink 之后),再跑 `python scripts/validate.py` 验收(Task 5 Step 3-5)。
3. 四个「审核点」条目(diffusers、Whisper 系、MLX)集中涉及「llm-frameworks 副标题写 LLM 但项目非 LLM」的归类张力,如需批量改判请明确指示目标类目。
