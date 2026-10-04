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

## 环境准备

手写实现(各章的 `*_minimal.py`)只用 Python 标准库,无需安装任何依赖。

AI 工程版(各章的 `*_engineered.py`)需要 scikit-learn:

```bash
/opt/homebrew/bin/python3 -m pip install -r requirements-tutorial.txt
```

验证环境:Python 3.14.8,scikit-learn 1.9.1,numpy 2.5.3。

## 仓库实战

导航节命令依赖历史数据库,首次使用先构建:

```bash
/opt/homebrew/bin/python3 scripts/import_history.py
```

审阅题目来源:[REVIEW-CHECKLISTS.md](../../REVIEW-CHECKLISTS.md)
类目与项目数据:[projects.yaml](../../projects.yaml)
