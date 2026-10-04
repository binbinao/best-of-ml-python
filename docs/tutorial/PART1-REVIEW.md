# Part I 交付审查记录

日期:2026-10-03
范围:ch01-ch05 + 代码 + 与仓库资产的联动
结论:**Part I 完成,可交付审阅**

## 1. 五项验收自查(主 agent 直接执行,非子代理)

| # | 项 | 结果 |
|---|---|---|
| 1 | 五段结构(`## ` 节数 ≥5) | ch01=8 · ch02=6 · ch03=8 · ch04=6 · ch05=7 ✅ |
| 2 | 八个代码文件可运行 | 8/8 exit 0 ✅ |
| 3 | 手写版无第三方依赖 | 4 份手写版顶层仅 stdlib(math/random/sys/time/fractions)✅;`metrics_minimal.py:359` 的 sklearn 导入是 `try/except ImportError` 包裹的**可选软依赖**,专用于 ch04 建立的跨库逐位对照,sklearn 缺失时优雅降级——不是违规 |
| 4 | 审阅引用闭合 | ch02-05 各 1 处 `Machine Learning Frameworks (ml-frameworks)`,与 REVIEW-CHECKLISTS.md 标题逐字一致;ch01 为立场章不引清单 ✅ |
| 5 | 导航命令 | 5 条实测全 exit 0;全部为 `analyze.py data/history.db <cmd>` 形式(db 必填第一位置)✅ |
| 6 | 类目 ID 真实存在 | 引用 ml-frameworks / tabular / time-series-data / probabilistics / chinese-nlp,全部存在于 projects.yaml,无效 ID: NONE ✅ |
| 7 | 仓库资产未破坏 | `validate.py` OK;`pytest tests/ -q` 14 passed ✅ |
| 8 | 残留临时物 | `__pycache__` 已被 .gitignore 覆盖并清除;无 `_*.py` 原型残留 ✅ |

## 2. 交付物规模

| 章 | 正文 | 手写版 | 工程版 |
|---|---|---|---|
| ch01 引擎而非目录 | 8 节 | — | — |
| ch02 泄漏 | 6 节 | leakage_minimal.py | leakage_engineered.py |
| ch03 切分 | 8 节 | splits_minimal.py | splits_engineered.py |
| ch04 指标 | 6 节 | metrics_minimal.py | metrics_engineered.py |
| ch05 表格建模 | 7 节 | tabular_minimal.py | tabular_engineered.py |

## 3. AI 素材生产实况(spec §7 风险对策的兑现)

每章工程版的"生成记录"取自实现过程中真实发生的审阅事件,不是事后编造:

| 章 | 审阅发现的真实陷阱 |
|---|---|
| ch02 | 切分前 fit StandardScaler(轻泄漏);实测证明它**不抬 AUC**(AUC 是纯排序指标),真正破坏的是持久化 scaler 污染测试集 + L2 在错误坐标系惩罚 → 教学点变成"标签泄漏扭曲指标、预处理泄漏不动指标,只盯指标抓一半" |
| ch03 | 单一 `train_test_split` 忽略 group 与时间结构;**实测否证了"group 切分必然抬分"**——单种子 +0.0433 经 300 种子重采样后均值 −0.0123、CI 跨零 |
| ch04 | `accuracy_score` 在 1:99 正样本下报 0.99;成本最优阈值与准确率最优阈值差 1.80 倍 |
| ch05 | 网格搜索默认分层 K 折对时序/分组数据错误 |

**审查过程中被实证推翻的计划预设**(4 处):ch02 的近泄漏字段实为确定性标签函数、ch03 的 group 泄漏乐观偏差、ch04 的预处理泄漏抬 AUC、ch05 的"过拟合"实为 −0.0129(没过拟合)。每一次都改为按实况重写,而非让数字迁就预设。

## 4. 跨章沉淀的 7 条硬标准(Part II 续用)

1. 数字必须实测,不得编造
2. 不得把单点值当因果叙述(方向性判断须重采样报 CI)
3. 不得把构造性恒等式当测量证据
4. 不得把可读性重构称作 bug 修复
5. 正文引用的输出块须与实跑逐字一致
6. 构造性设定必须显式标注为构造/反事实
7. 数据源读法必须先验证方向(DESC vs ASC)

## 5. 审查发现统计

| 章 | 首审发现 | 修复轮次 | 复审 |
|---|---|---|---|
| ch01 | 6(2×P1) | 1 轮 | correct / 0.96 |
| ch02 | 11(3×P1) | 2 轮 | correct / 0.95 |
| ch03 | 10(6×P1) | 1 轮 + 2 处直改 | correct / 0.92 |
| ch04 | 9(1×P1) | 1 轮 | correct / 0.96 |
| ch05 | 8(1×P1) | 2 轮 | correct(2 项新表述问题已修) |

合计 44 项发现全部处置。**两处模式值得记录**:
- **"看起来勤勉、其实没分开"**:ch04 以为 `| head` 能断管(实测 PIPESTATUS=0 0)、ch05 以为 `gap=1` 能隔月(单位是样本数)——两处都是正文在教读者防范的失效形态,自己却踩了。已在 ch05 收束节点名。
- **"自承与交付物不符"**:ch04 实现者报告称"已标注 1:1 为反事实",但 `grep 反事实` 在交付物中零命中。承诺须以交付物验证。

## 6. 已知限制(交付给读者的诚实说明)

- 成本侧无跨库验证:sklearn 无"成本最优阈值"API,ch04 的成本数字全部来自手写实现,两个手写实现互为验证(能抓笔误,抓不住共同系统性误解)
- ch05 两个最优阈值是单点量,未 bootstrap(数据与种子固定时重跑同值;换数据需重扫)
- ch04/ch05 的类目跨度数字取自不同周次(ch04 是 2020-11-30,ch05 是最新一周),各章已显式说明不可直接搬
- ch05"既要 gap 又要足够折内样本"在真实项目可能无解(gap=300 + 5 折需 1800 行,sklearn 拒绝,须收 test_size),本章未展开该取舍
- README.md 仍是旧渲染(920 projects / 34 categories),新叙事需等下次生成器运行

## 7. 后续

Part II(数据与领域:ch06 时序 / ch07 文本 / ch08 图像 / ch09 特征工程与管道)待续。
