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

**Part I 状态**:已完成并通过逐章审查(44 项发现全部处置),详见 [PART1-REVIEW.md](PART1-REVIEW.md)。

Part II(数据与领域:时序 / 文本 / 图像 / 特征工程与管道)、Part III(LLM 时代)、Part IV(可信与规模)待续。

## 每章检查动作

读完每章,先别急着认可代码——带着这三个问题回看本章的 AI 工程版:

1. **这个数字回表核对过吗?** ch04 教过:FP 列 1183 不是总拒绝数(是 1419);ch05 教过:0.6145 vs 0.6145 相减不是 0.0019。
2. **这个方向验证过吗?** ch04 教过:`ORDER BY projectrank DESC` 里 44→56 是上升不是下降;ch05 教过:`gap=1` 的单位是样本数不是月份。
3. **这是构造的还是测出来的?** ch02 的近泄漏字段、ch03 的 0.5000 恒等式、ch04 的 1:1 成本口径、ch05 的 partner_channel——每一个都曾被当成实证发现,实际上都是人为设定。

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
