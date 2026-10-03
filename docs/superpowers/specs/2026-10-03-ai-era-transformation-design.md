# best-of-ml-python → best-of-ai-python 改造设计

日期:2026-10-03
状态:已批准(对话中逐项确认)
路径:架构级(三阶段,每阶段独立可交付、可停)

## 1. 定位与身份

- Fork `ml-tooling/best-of-ml-python`,重命名为 **`best-of-ai-python`**,定位「AI 时代的 Python ML 生态学习清单」。
- 保留 `README.md`/清单内容的 **CC-BY-SA-4.0** 许可与对上游的致谢;致谢文字落在 `config/footer.md`(手改合法,属模板而非生成物)。
- Fork、改名、开启 Actions 等 GitHub 侧操作由用户执行;本设计的所有仓库内改动在下述三阶段内完成。

## 2. 目标与非目标

**目标**
1. 覆盖 2023-2026 AI 生态:新增 5 个类目 + 约 40 个头部项目。
2. 建立 `projects.yaml` 校验层,坏数据在 PR 阶段被机器拦截,而非下周四的生成器运行时。
3. 工具链现代化:移除已归档/弃用的 CI 依赖。
4. 激活 `history/`(250 周 × 925 项目 × ~50 列)为可查询的学习导航数据层。

**非目标**
- 不重写 best-of-generator;生成管道继续用 `best-of-lists/best-of-update-action@v0.8.5`。
- 不迁移现有项目到新类目(阶段一零迁移;后续可另行决策)。
- 不做生态分析型全量表设计(阶段三只做学习导航场景)。
- 不向上游 ml-tooling 反哺本设计的结构性改动(数据修复类 PR 可另行单独评估,不在本设计内)。

## 3. 阶段一:数据焕新 + 校验脚本

### 3.1 修复已知数据缺陷(先于一切扩充)

| 缺陷 | 位置 | 修法 |
|---|---|---|
| neurolink 条目缺 `name:` | `projects.yaml:4668` | 补 `name: neurolink`(与 PR #422 标题一致) |
| `labels: ["spacy"]` 引用未定义标签 | `projects.yaml:561,4106` | 在 `labels:` 块**新增** spacy 标签定义(图标用 spacy 官方 favicon,格式对齐现有条目) |
| `labels: ["huggingface"]` 引用未定义标签 | `projects.yaml:4300,4341,4353` | 同上,新增 huggingface 标签定义 |

选择「补定义」而非「删引用」:两个标签表达真实的生态归属,且 huggingface 已被 3 处使用。

### 3.2 新增 5 个平行类目

在 `categories:` 块末尾(`others` 之前)追加:

| category id | title | subtitle 要点 |
|---|---|---|
| `llm-frameworks` | LLM Frameworks & Toolkits | 模型访问、微调、量化、对齐工具 |
| `agent-frameworks` | Agent & Orchestration Frameworks | 多智能体编排、工具调用、规划 |
| `rag-vectordb` | RAG & Vector Databases | 检索增强、向量存储、嵌入管线 |
| `inference-serving` | LLM Inference & Serving | 高吞吐推理引擎、服务化部署 |
| `model-hub` | Model Hubs & Registries | 模型托管、权重分发、注册表 |

`model-hub` 依赖阶段一的验证结论(见 3.4):若 huggingface_hub(22k★)等现有项目迁移被明确排除在非目标外,则 model-hub 首批仅收新条目。

### 3.3 策展批次(约 40 个项目)

- 清单由 AI 起草、**用户逐条审核后**一次性入库;每条含 `name`/`github_id`/`pypi_id`/`category`/`labels`(新标签:spacy、huggingface 已补;另新增 `llm`、`agent`、`rag`、`inference` 四个标签定义,镜像现有标签块的格式)。
- 候选池(星标为起草时估算,入库前逐一核实与查重):**推理/服务**:vLLM、sglang、llama.cpp、Ollama;**Agent/编排**:LangChain、LlamaIndex、AutoGen、CrewAI、smolagents、litellm、instructor、pydantic-ai;**RAG/向量库**:chromadb、faiss、weaviate、milvus、lancedb、txtai、haystack;**模型/训练**:diffusers、whisper、faster-whisper、funasr、mlx、unsloth、peft、trl、deepspeed、bitsandbytes;**SDK**:openai-python、anthropic-sdk-python。排除非 Python 或非库类:text-generation-webui、LM Studio。
- 查重规则:凡 `github_id` 已存在于 `projects.yaml` 的候选一律跳过(零迁移原则)。
- 入库位置:追加到 `projects:` 列表末尾(生成器按分数自动排序,无需手工定位)。

### 3.4 分类目门槛(已验证)

- **已验证:best-of-generator(0.8.6,与上游 master 一致)不支持 per-category `min_stars`,仅全局 `configuration.min_stars` 生效**(结论 B)。
- 证据(源码:`pip download best-of==0.8.6` 解包 + 上游 master 核对,两处一致;行号以 master 为准):
  - `src/best_of/generator.py:38-46`:`min_stars` 只从 `projects.yaml` 顶层 `configuration` 块经 `prepare_configuration()` 读入;`categories` 条目经 `prepare_categories()` 原样转 `Dict` 全量透传,装配期不做属性过滤。
  - `src/best_of/projects_collection.py:427-433`(`apply_filters()`):星标过滤唯一读取点为全局 `configuration.min_stars`(`int(project_info.star_count) < int(configuration.min_stars) → project_info.show = False`);该函数签名只接收全局 `configuration`,不接收 `categories`,且全包无任何代码读取 `categories[].min_stars`——阻塞机制是**装配期全量透传、下游无消费路径**,而非装配过滤。
  - `src/best_of/default_config.py:28-29`:`prepare_configuration()` 的缺省兜底(`config.min_stars = 100`)同样只作用于全局配置对象。
  - 上游文档(README Configuration/Category Properties 章节):`min_stars` 仅列于 `configuration` 属性表;Category 属性表仅含 `category`/`title`/`subtitle`/`ignore`,无 `min_stars`。
- 阶段一处置:维持全局 `min_stars: 300` 不变,策展批次仅收星标 ≥300 的候选(头部候选 vLLM/Ollama/LangChain 等均远超 300);「5 个新类目放宽至 100」的差异化门槛转为「后续给上游(best-of-lists/best-of-generator)提 patch」待办,不阻塞本阶段。Task 5 准入过滤按 ≥300 执行。

### 3.5 校验脚本 `scripts/validate.py`

- 无第三方依赖仅标准库 + PyYAML(`yaml.safe_load`)。
- 检查项(每项独立错误计数,退出码非 0 即失败):
  1. YAML 可解析;
  2. 每个项目有非空 `name`,且全局唯一;
  3. `category`(若设)存在于 `categories:` 定义的 ID 集合(含 `others` 兜底语义:未设 category 合法);
  4. `labels`(若设)全部存在于 `labels:` 定义的 ID 集合;
  5. `github_id` 匹配 `^[^/\s]+/[^/\s]+$`;
  6. `configuration`/`categories`/`labels`/`projects` 四个顶层键存在。
- 输出格式:`projects.yaml: project '<name>': <错误描述>`(以项目名定位,不带行号——CI 门禁只依赖退出码;人类排查时用项目名 grep 定位),人类与 CI 双友好。
- 验收:对修复前快照跑出 ≥6 处错误(1 缺 name + 5 未定义标签引用);修复后 0 错误。

## 4. 阶段二:CI + 工具链现代化

改 `.github/workflows/update-best-of-list.yml`(基于上游 v0.8.5 模板):

1. **前置校验 step**(checkout 之后、生成之前):`pip install pyyaml && python scripts/validate.py`——坏数据在生成前拦截。
2. **`create-pull-request`**:弃用已归档的 `hub`(2.14.2,经 curl 安装),改用 `gh pr create -b main -h update/${VERSION} --title "Best-of update: ${VERSION}" --body-file <(模板正文)`;`gh` 预装于 ubuntu-latest,无需安装 step。
3. **`create-release`**:弃用 `actions/create-release@v1`(已弃用),改 `gh release create ${VERSION} latest-changes.md --draft --title "Update: ${VERSION}"`。
4. `.github/.gitattributes` 追加生成物标记:
   ```
   README.md linguist-generated=true
   latest-changes.md linguist-generated=true
   history/** linguist-generated=true
   ```
   保留既有 `* linguist-language=Python`。
5. 不改 cron 节奏(周四 14:00 UTC)、不改分支/tag/PR 命名约定。

## 5. 阶段三:SQLite 查询层(学习导航优先)

- `scripts/import_history.py`:遍历 `history/*_projects.csv` → 建表 `projects_history(week TEXT, <CSV 列>)`,主键 `(week, github_id)`;写入 `data/history.db`。`data/` 入 `.gitignore`(库按需重建,源 CSV 已在库)。
- `scripts/analyze.py`(纯标准库 `sqlite3`),首批查询:
  1. `top <category> <N>` — 某类目按 projectrank 的 top-N;
  2. `trend <name>` — 单项目周度星标/projectrank 轨迹;
  3. `recent <N>` — 最近入库项目;
  4. `status <name>` — 活跃状态判定(最近一次快照的 commit 活跃度字段)。
- `AGENTS.md` 同步更新:登记脚本用法,使 AI 助手可直接执行查询。

## 6. 验证方式(汇总)

- 阶段一:`scripts/validate.py` 修复前后对照(6→0);策展批次经用户审核;`python -c "import yaml; yaml.safe_load(open('projects.yaml'))"` 通过。
- 阶段二:workflow YAML 语法校验(`actionlint` 若可用,否则 `python -c` yaml 解析 + 人工审读 diff);改动保持与上游模板 diff 最小。
- 阶段三:`import_history.py` 全量导入无错;`analyze.py` 四个查询各跑一条真实数据并人工核对结果。

## 7. 交付顺序与停点

阶段一 → 阶段二 → 阶段三,严格串行;每个阶段结束即独立可用,可在任意阶段间停下。每阶段完成后更新 `AGENTS.md` 对应章节。
