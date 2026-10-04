# 《机器学习审阅者指南》Part I 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付教程 Part I 五章(ch01-05),每章遵循「业务问题 → 手写原理 → AI 工程版 → 审阅对照 → 仓库实战」五段结构,代码全部可运行,审阅题目引用 REVIEW-CHECKLISTS.md,导航节命令真实可执行。

**Architecture:** Markdown 教程正文 `docs/tutorial/chXX-<slug>.md` + 每章独立代码目录 `docs/tutorial/code/chXX/`。手写实现零第三方依赖(纯 stdlib/numpy),AI 工程版限主流库(sklearn)。每章独立提交,互不依赖;ch01 是立场章无代码。

**Tech Stack:** Python 3.13(/opt/homebrew/bin/python3),numpy(仅 ch05 手写梯度下降与 ch04 需要时),scikit-learn(工程版),pytest(代码可运行性验收)。

**Spec:** `docs/superpowers/specs/2026-10-03-ml-reviewer-tutorial-design.md`

## Global Constraints

- 正文语言:中文;代码标识符、库名、术语英文
- 每章五段结构齐全,缺一不可
- 手写实现:纯 Python 标准库,最多允许 numpy;**禁止** sklearn/pandas 等第三方库
- AI 工程版:依赖限主流库(scikit-learn 等),须带 `# 生成记录` 注释(提示词摘要 + 日期 2026-10-03 + 说明系 AI 产出后经审阅修正)
- 每章末尾必须含「仓库导航」节,`analyze.py` 命令逐字可执行(exit 0),类目 ID 必须真实存在于 projects.yaml
- 审阅对照引用 REVIEW-CHECKLISTS.md 的节标题须与该文件实际标题逐字一致(现有 15 个 `## ` 节,见各章引用)
- 验收命令统一用 `/opt/homebrew/bin/python3`(仓库默认 `python` 无 numpy/pytest)
- 提交粒度:每章一提交,消息格式 `docs(tutorial): chXX <slug>`
- 导航命令前置条件:`data/history.db` 不存在时先跑 `/opt/homebrew/bin/python3 scripts/import_history.py`(约 6 秒,187,281 行)

---

### Task 1: 教程骨架与 ch01 立场章

**Files:**
- Create: `docs/tutorial/README.md`(教程索引与阅读方式)
- Create: `docs/tutorial/ch01-engine-not-catalog.md`
- Modify: `AGENTS.md`(登记教程目录,让 AI 助手知道它是仓库资产)

**Interfaces:**
- Consumes: `config/header.md` 立场段、`REVIEW-CHECKLISTS.md` 引言、本仓 git 历史(ch01 案例素材)
- Produces: 教程目录结构与阅读约定,供 ch02-05 引用(`docs/tutorial/README.md` 的章节表);ch01 被 ch02-05 引用为"立场章"

- [ ] **Step 1: 创建 docs/tutorial/README.md 索引**

```markdown
# 机器学习审阅者指南

AI 写代码的时代,你的价值是定义业务问题、理解原理门槛、审阅 AI 产出。这本书教你获得"审阅资格"。

**读者**:会 Python、懂基础统计、未系统学过 ML 的工程师。

**每章结构**:业务问题 → 原理门槛(手写最小实现) → AI 工程版(真实生成后审阅修正) → 审阅对照 → 仓库导航。

## 章节

| 章 | 标题 | 审阅门槛主题 | 对应类目 |
|---|---|---|---|
| [ch01](ch01-engine-not-catalog.md) | 引擎而非目录 | 人机分工总论 | 全部 |
| [ch02](ch02-leakage.md) | 泄漏 | 预测与优化问题 | ml-frameworks |
| [ch03](ch03-splits.md) | 切分 | 预测与优化问题 | ml-frameworks |
| [ch04](ch04-metrics.md) | 指标 | 预测与优化问题 | ml-frameworks |
| [ch05](ch05-tabular-modeling.md) | 表格建模全景 | 预测与优化问题 | tabular, ml-frameworks |

Part II(数据与领域)、Part III(LLM 时代)、Part IV(可信与规模)待续。

## 仓库实战

导航节命令依赖历史数据库,首次使用先构建:

```bash
/opt/homebrew/bin/python3 scripts/import_history.py
```

审阅题目来源:[REVIEW-CHECKLISTS.md](../../REVIEW-CHECKLISTS.md)
类目与项目数据:[projects.yaml](../../projects.yaml)
```

- [ ] **Step 2: 写 ch01 立场章正文**

内容骨架(每节展开为 400-800 字):
1. **开篇**:一个具体场景——把业务需求交给 AI,AI 半小时交付了完整管线,指标全绿,上线后业务指标纹丝不动。问题出在哪。
2. **价值曲线倒转**:贬值的(手写胶水代码)与升值的(问题定义、原理门槛、审阅能力)三件事;为什么贬值的部分成本趋近于零。
3. **危险的失效模式**:代码完全正确、指标完全达标、做错了事。举三个具体形态:target leakage、指标代理失真、时序穿越。每个形态一句话点明"不懂原理看不出来,因为程序能跑"。
4. **本清单的用法**:它是解法范式与审阅原理的地图,不是 API 目录。四个姿态(先业务问题后工具、握住把关原理、用审阅清单、机器管机械正确性)。
5. **本仓的实演**:引用本仓 git 历史作为案例——2026-10-03 的改造中,11 处数据缺陷(重复 name、未定义标签)若无校验层将在一周后的定时生成中才暴露;litellm 的裸字符串 labels 让校验器产生 6 个假阳性,这类"机械检查的误报"本身也是需要理解的原理。
6. **五条工作原则**:定义问题先于选工具;手写一次最小实现;AI 产出必须过审阅清单;指标必须翻译到业务语言;上线后要有监控假设。
7. **仓库导航节**:

```bash
/opt/homebrew/bin/python3 scripts/import_history.py   # 首次:187,281 行
/opt/homebrew/bin/python3 scripts/analyze.py data/history.db recent 5
```

- [ ] **Step 3: AGENTS.md 登记教程目录**

在 Key Directories 表增一行:| `docs/tutorial/` | 教程正文与可运行示例(chXX 代码目录,验收:`python code/chXX/chXX_minimal.py`)|

- [ ] **Step 4: 验收**

Run: `grep -c '^## ' docs/tutorial/ch01-engine-not-catalog.md`(应 ≥5,含仓库导航节)+ 逐字执行 Step 2 的两条命令确认 exit 0
Expected: 节数达标,命令成功

- [ ] **Step 5: 提交**

```bash
git add docs/tutorial/README.md docs/tutorial/ch01-engine-not-catalog.md AGENTS.md
git commit -m "docs(tutorial): ch01 engine-not-catalog stance chapter"
```

---

### Task 2: ch02 泄漏章

**Files:**
- Create: `docs/tutorial/ch02-leakage.md`
- Create: `docs/tutorial/code/ch02/leakage_minimal.py`
- Create: `docs/tutorial/code/ch02/leakage_engineered.py`

**Interfaces:**
- Consumes: REVIEW-CHECKLISTS.md 的 `## Machine Learning Frameworks (ml-frameworks)` 节;类目 `ml-frameworks`
- Produces: 可复用的 `leakage_minimal.py` 演示数据集(后续章可 import);ch02 被 ch03 引用("泄漏的孪生兄弟是切分")

- [ ] **Step 1: 写 ch02_leakage_minimal.py(手写原理,纯 stdlib)**

一个可运行的演示:构造"客户流失预测"数据集,其中**故意**埋三种泄漏——(a) 一个在预测时点才产生的字段 `cancellation_date`(未来信息);(b) 一个由标签派生的字段 `cancelled`(直接泄漏);(c) 训练后才写入的字段 `tenure_months_at_snapshot` 实际用了当前值而非快照值。用纯 Python(无 sklearn)训练逻辑回归(手写 sigmoid + 梯度下降),打印三组 AUC,展示泄漏模型接近满分且泄漏系数主导。

```python
#!/usr/bin/env python3
"""Leakage demo: a model that scores perfectly by cheating. Pure stdlib."""
import math, random

# 生成 2000 样本;真实信号弱,泄漏字段强
# (a) cancellation_date: 预测时点之后才有 -> 直接包含答案
# (b) cancelled: 标签的副本
# (c) snapshot_tenure: 用了当前值而非历史快照 -> 近泄漏
# (d) age, plan, usage: 真实弱信号
...
```

实现要求:手写 sigmoid、二元交叉熵损失、批量梯度下降;固定随机种子;输出每个特征学到的权重(泄漏特征权重应显著大于真实特征)。

- [ ] **Step 2: 运行确认可运行**

Run: `/opt/homebrew/bin/python3 docs/tutorial/code/ch02/leakage_minimal.py`
Expected: 打印三组 AUC(泄漏组 >0.99、真实特征组 ~0.6-0.7)与特征权重表

- [ ] **Step 3: 写 ch02_leakage_engineered.py(AI 工程版)**

AI 生成后审阅修正的版本,用 sklearn(LogisticRegression + Pipeline)。**故意保留一处真实陷阱**并在代码注释与正文标注:先做了标准化再切分(标准化用了全量数据的均值方差——轻度泄漏)。文件头必须含:

```python
"""AI-generated engineered version, reviewed and corrected.
生成记录: 2026-10-03,提示词摘要="写一个客户流失预测的 sklearn 管线,含标准化、逻辑回归、ROC-AUC 评估"
审阅发现: 原始产出在切分前对全量数据 fit 了 StandardScaler(泄漏),已改为在 Pipeline 内先切分后 fit。
"""
```

- [ ] **Step 4: 运行确认可运行**

Run: `/opt/homebrew/bin/python3 docs/tutorial/code/ch02/leakage_engineered.py`
Expected: 打印 AUC + 系数表;注释中的审阅发现与实际代码一致

- [ ] **Step 5: 写 ch02-leakage.md 正文**

五段结构:业务问题(信贷欺诈/流失预测中"模型太好用"的真实案例)→ 原理门槛(引用 leakage_minimal.py,解释三种泄漏的机制与权重证据)→ AI 工程版(引用 engineered.py,展示那处标准化泄漏)→ 审阅对照(引 REVIEW-CHECKLISTS.md ml-frameworks 节的第一、二条问题:标签在预测时点是否存在、预处理是否训练/推理一致)→ 仓库导航节。

导航节:

```bash
/opt/homebrew/bin/python3 scripts/import_history.py
/opt/homebrew/bin/python3 scripts/analyze.py data/history.db top ml-frameworks 5
```

对应类目:`ml-frameworks`

- [ ] **Step 6: 验收**

Run: 两条代码均 exit 0 + `grep -c 'REVIEW-CHECKLISTS.md' docs/tutorial/ch02-leakage.md` ≥1 + 导航命令 exit 0
Expected: 全通过

- [ ] **Step 7: 提交**

```bash
git add docs/tutorial/ch02-leakage.md docs/tutorial/code/ch02/
git commit -m "docs(tutorial): ch02 leakage"
```

---

### Task 3: ch03 切分章

**Files:**
- Create: `docs/tutorial/ch03-splits.md`
- Create: `docs/tutorial/code/ch03/splits_minimal.py`
- Create: `docs/tutorial/code/ch03/splits_engineered.py`

**Interfaces:**
- Consumes: ch02 的泄漏主题(交叉引用);REVIEW-CHECKLISTS.md `## Machine Learning Frameworks (ml-frameworks)`;类目 `ml-frameworks`、`time-series-data`
- Produces: `splits_minimal.py` 的三种切分函数,ch06 时序章可复用

- [ ] **Step 1: 写 ch03_splits_minimal.py(手写三种切分,纯 stdlib)**

实现 `random_split` / `time_split` / `group_split` 三个函数(纯 Python,接受 list of dict + 索引数组)。构造一个"设备故障预测"数据集:同一设备的多条记录构成 group、时间有序;演示随机切分让同一设备同时出现在训练与测试(乐观偏差),group 切分消除之;时间切分在时序场景下的必要性。每种切分后计算"训练-测试分布差异"的简单统计量(某特征的均值差)。

- [ ] **Step 2: 运行确认可运行**

Run: `/opt/homebrew/bin/python3 docs/tutorial/code/ch03/splits_minimal.py`
Expected: 打印三种切分下的 group 重叠数与均值差,随机切分显示重叠 >0、group 切分显示 0

- [ ] **Step 3: 写 ch03_splits_engineered.py(AI 工程版)**

sklearn 的 train_test_split + GroupShuffleSplit + TimeSeriesSplit 对照。**陷阱**:AI 常产出 `train_test_split(df, test_size=0.2, random_state=42)` 单一随机切分,注释标注"业务数据有 group 结构与时间顺序,此切分会泄漏"。文件头含生成记录 + 审阅发现。

- [ ] **Step 4: 写 ch03-splits.md 正文**

五段结构。核心论点:AI 默认给随机切分,因为随机切分是代码示例里最常见的;但业务的 group 结构(同一用户/设备/客户)与时间顺序让随机切分系统性乐观。重点讲"泄漏的孪生兄弟":ch02 泄漏来自字段,ch03 泄漏来自切分边界。

审阅对照引 ml-frameworks 节:"How is the data split — random, temporal, grouped by entity? Does the split match how the system will be used?"

导航节:

```bash
/opt/homebrew/bin/python3 scripts/analyze.py data/history.db top time-series-data 5
```

对应类目:`ml-frameworks`、`time-series-data`

- [ ] **Step 5: 验收 + 提交**

Run: 代码 exit 0 + 正文五段结构齐全 + 导航命令 exit 0
```bash
git add docs/tutorial/ch03-splits.md docs/tutorial/code/ch03/
git commit -m "docs(tutorial): ch03 splits"
```

---

### Task 4: ch04 指标章

**Files:**
- Create: `docs/tutorial/ch04-metrics.md`
- Create: `docs/tutorial/code/ch04/metrics_minimal.py`
- Create: `docs/tutorial/code/ch04/metrics_engineered.py`

**Interfaces:**
- Consumes: ch02/ch03 的评估语境;REVIEW-CHECKLISTS.md `## Machine Learning Frameworks (ml-frameworks)` 与 `## Time Series Data (time-series-data)` 的指标相关条目;类目 `ml-frameworks`
- Produces: 代价敏感阈值选择的可复用逻辑

- [ ] **Step 1: 写 ch04_metrics_minimal.py(手写混淆矩阵与代价,纯 stdlib)**

实现混淆矩阵、precision/recall/阈值扫描;给定一个业务场景(信贷:漏批坏贷款损失 1000 元、误批好客户损失 100 元),用纯 Python 计算不同阈值下的期望成本,展示"最高准确率的阈值"与"最低业务成本的阈值"相差甚远;AUC 高但按业务成本算不划算的反例。

- [ ] **Step 2: 写 ch04_metrics_engineered.py(AI 工程版)**

sklearn 的 roc_auc_score / precision_recall_curve + 成本最优阈值选择。陷阱:AI 常直接用 `accuracy_score` 报喜(在 1:99 正样本比例下 accuracy 0.99 但模型什么都没学到),注释标注。

- [ ] **Step 3: 写 ch04-metrics.md 正文**

五段结构。核心论点:指标是业务的代理,代理会失真;AUC 衡量排序能力,业务关心的是"按这个排序行动后的净收益"。给出阈值选择的标准做法(用业务成本而非模型指标选阈值)与常见代理失真形态(代理指标被优化博弈、离线-在线不一致)。

审阅对照引 ml-frameworks 节:"Which metric is optimized, and how does it map to the business outcome?"

导航节:

```bash
/opt/homebrew/bin/python3 scripts/analyze.py data/history.db trend Tensorflow 2>/dev/null | head -5
```

(演示 rank 与星标随时间的轨迹——一个指标的轨迹读法示范)
对应类目:`ml-frameworks`

- [ ] **Step 4: 验收 + 提交**

Run: 代码 exit 0 + 导航命令 exit 0
```bash
git add docs/tutorial/ch04-metrics.md docs/tutorial/code/ch04/
git commit -m "docs(tutorial): ch04 metrics"
```

---

### Task 5: ch05 表格建模全景章

**Files:**
- Create: `docs/tutorial/ch05-tabular-modeling.md`
- Create: `docs/tutorial/code/ch05/tabular_minimal.py`
- Create: `docs/tutorial/code/ch05/tabular_engineered.py`

**Interfaces:**
- Consumes: ch02(ch04)的评估闭环(切分+指标在此收束成完整管线);REVIEW-CHECKLISTS.md `## Machine Learning Frameworks (ml-frameworks)`;类目 `tabular`、`ml-frameworks`
- Produces: Part I 的收束章,ch01-05 叙事闭环

- [ ] **Step 1: 写 ch05_tabular_minimal.py(手写逻辑回归全流程,纯 stdlib)**

纯 Python 实现逻辑回归的完整训练:sigmoid、标准化(手写)、L2 正则、批量梯度下降、学习率衰减;在"用户流失"合成数据集上训练,手写 AUC 计算。与 ch02 的差异:此处实现完整(正则、标准化),ch02 是最小演示。输出训练/验证曲线与最终指标。

- [ ] **Step 2: 写 ch05_tabular_engineered.py(AI 工程版)**

sklearn Pipeline(StandardScaler + LogisticRegression(solver='lbfgs', C 网格)) + cross_val_score。陷阱:AI 常在网格搜索里用默认 CV(分层 K 折),对时序或分组数据是错的;注释标注此陷阱并指出正确做法(GroupKFold/TimeSeriesSplit)。

- [ ] **Step 3: 写 ch05-tabular-modeling.md 正文**

五段结构,收束 Part I:把 ch02(泄漏)、ch03(切分)、ch04(指标)拼成一条完整管线,展示一份"正确的手写版"与"典型的 AI 版"在每个环节的分歧点。结尾节「Part I 总结:审阅者的四个动作」——定义问题(业务语言)、握住门槛(原理)、对照清单(逐问)、翻译指标(回业务)。

审阅对照引 ml-frameworks 节全部五条。

导航节:

```bash
/opt/homebrew/bin/python3 scripts/analyze.py data/history.db top tabular 5
```

对应类目:`tabular`、`ml-frameworks`

- [ ] **Step 4: 验收**

Run: 两条代码 exit 0 + `grep -c '^## ' docs/tutorial/ch0[2-5]*.md` 每章 ≥5 + 全导航命令 exit 0 + `/opt/homebrew/bin/python3 scripts/validate.py` OK + `pytest tests/ -q` 14 passed(确认未破坏仓库资产)
Expected: 全通过

- [ ] **Step 5: 提交**

```bash
git add docs/tutorial/ch05-tabular-modeling.md docs/tutorial/code/ch05/
git commit -m "docs(tutorial): ch05 tabular modeling"
```

---

### Task 6: Part I 交叉审查与收尾

**Files:**
- Modify: `docs/tutorial/README.md`(补 Part I 完成标记与 Part II 预告)
- Create: `docs/tutorial/PART1-REVIEW.md`(审查记录)

**Interfaces:**
- Consumes: ch01-05 全部产出
- Produces: Part I 交付状态与用户审阅入口

- [ ] **Step 1: 自查五项验收(全 Part I 范围)**

逐项执行并记录结果到 `docs/tutorial/PART1-REVIEW.md`:
1. 5 章五段结构齐全(每章 `## ` 节数 ≥5)
2. 8 个代码文件全部 exit 0 运行
3. 审阅引用闭合:每章的 REVIEW-CHECKLISTS.md 节标题与该文件实际标题逐字一致(grep 核对)
4. 导航命令全部 exit 0(5 条:top ml-frameworks / top time-series-data / trend Tensorflow / top tabular / recent 5)
5. 类目 ID 全部存在于 projects.yaml(ml-frameworks, time-series-data, tabular)

- [ ] **Step 2: 记录 AI 素材生产实况**

每章的 `生成记录` 注释汇总到 PART1-REVIEW.md:实际提示词、生成日期、审阅发现的陷阱条目。这是 spec §7 风险对策的兑现。

- [ ] **Step 3: README 补完成标记**

在章节表下加一行:`**Part I 状态**:已完成,待审阅。Part II- IV 待续。`

- [ ] **Step 4: 提交并推送**

```bash
git add docs/tutorial/README.md docs/tutorial/PART1-REVIEW.md
git commit -m "docs(tutorial): Part I cross-review and wrap-up"
git push origin main
```

## Self-Review 记录

- Spec 覆盖:§2 五段结构→每章 Task 2-5 的 Step「写正文」;§3 章节骨架 Part I→Task 1-5;§4 工程安排(载体/素材/联动/验收)→Global Constraints + 各 Task 验收步;§5 非目标→计划不含英文版/Notebook/RL;§6 资产关系→ch01 Step 2(立场/清单/git 历史)与各章审阅引用;§7 风险→Task 6 Step 2(素材实况记录)。无缺口。
- 占位符扫描:每章代码有明确实现要求与输出期望,无"TBD/类似 Task N";AI 工程版代码在 Task 2 给出完整文件头范式,Task 3-5 沿用同一范式(已声明),不重复全文。
- 类型/名称一致性:`analyze.py <db> <cmd>` 参数序(取自 AGENTS.md 修正后的 synopsis);类目 ID 与 projects.yaml 核实一致;REVIEW-CHECKLISTS 节标题与实际 `## ` 标题核实一致;代码文件名 `chXX_{minimal,engineered}.py` 在 README 索引与各 Task 间一致。
