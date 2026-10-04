# ch02 泄漏

> 本章是 Part I 第一份「代码对照」章。读完你应该能回答一个问题:**看到 AUC 0.99,你怎么知道它是真的?**

## 1. 业务问题:模型太好用的时候

ch01 第 1 节那个流失预测的失败案例,值得再拆开看一次。真实项目里,"模型太好用"几乎从不出现在需求评审会上,它出现在**上线三个月后的复盘会上**。

信贷欺诈场景是另一个样子的失败。风控团队要一个反欺诈模型,三个月后拿到一张表:模型在验证集上的 AUC 是 0.78,不功不过。但同一批人做的交叉验证给出了 0.94。差异出现的原因是:某个特征记录的是"这张卡过去 90 天内被拒付的次数"——而**拒付事件本身发生在预测时点之后**。风控系统每天凌晨跑批,用昨天为止的数据给今天的交易打分,而这一列要到交易发生后 24 小时才同步进数据仓库。训练表是从线上库直接导出的,所以它安静地躺在特征列表里,列名干净,类型正确,没有任何缺失。

上线之后的表现是:前三个月的通过率一路下滑到 0.6%,坏账率却纹丝不动。团队以为模型失效了,开始调阈值。真正的问题在别处——他们优化的是一个"事后会发生拒付"的排序,所以排在最前面的交易,恰恰是那些几天后真的会被拒付的交易。而欺诈交易早就不该通过了,把它们排在前面只是把风险往后推,一步都没减少。

流失场景的失败形态不一样,但根子是同一个。ch01 讲的那个版本是**字段时点**错了:字段在预测时点还不存在。信贷这个版本更隐蔽一层:**字段存在,时点也对,但它记录的是你想预测的那个事件本身。** 前者读代码时能看出来,后者你得先想清楚业务因果链才看得出来。

这两类失败有一个共同点,而且它比"模型精度不够"严重得多:**指标不会报警。** AUC 0.99 会照常出现在周报里,数据管线的健康检查一片绿,模型监控面板上所有曲线都正常。泄漏不是让系统崩掉,是让它**以一种看起来成功的方式做错事**,并且在做错的路上越走越有成就感。

判断一份产出可不可信,靠的不是看它的指标有多漂亮,而是逐字段问一句话:**做预测的那一刻,这个值拿得到吗?**

这一章要把这句话变成能跑的数字。下面两份代码是同一份数据集、同一组随机种子,你可以把它们并排放着看,数字直接对得上。

## 2. 原理门槛:手写一个靠作弊拿满分的模型

```bash
/opt/homebrew/bin/python3 docs/tutorial/code/ch02/leakage_minimal.py
```

[leakage_minimal.py](code/ch02/leakage_minimal.py) 是一个纯标准库的实现,只有 `math`、`random`、`sys`、`time` 四个 import——sigmoid、二元交叉熵、批量梯度下降、AUC(秩和公式)、ROC 曲线全部手写。2000 个样本、6 个特征、60 轮梯度下降,跑完 0.3 秒。教学代码以可读性优先,没有任何性能优化。

数据集是构造的"客户流失预测",故意埋了三类泄漏:

| 特征 | 类型 | 为什么拿不到 |
|---|---|---|
| `cancellation_date_after_snapshot` | 目标泄漏(未来信息) | 取消日期在快照之后才写入,预测时点这一列是空的 |
| `cancelled_is_label_copy` | 目标泄漏(标签副本) | 它就是标签本身,只是换了列名 |
| `snapshot_tenure_current_value` | 近泄漏(时点错配) | 用了流失发生**之后**才确定的 tenure |
| `age` / `plan_tier` / `usage_score` | 真实弱信号 | 预测时点确实拿得到 |

跑出来三个模型:

```
模型 A  全特征   (3 个真实 + 3 个泄漏) : AUC = 1.0000
模型 B  真实特征 (age/plan_tier/usage) : AUC = 0.6767
模型 C  真实 + 近泄漏 tenure          : AUC = 0.9995
```

模型 B 的 0.6767 落在一个现实区间里——**真实世界里绝大多数流失模型就在这个位置**。这是这份数据真正的上限,也是唯一一个你可以拿去跟业务讨论的数字。模型 A 的 1.0000 什么都不是。

### 权重表是证据的核心

AUC 只告诉你"有东西不对",权重表才告诉你"是什么"。模型 A 的权重表(标准化后):

```
  特征                                         权重     |权重|排名
  --------------------------------------------------------
  cancelled_is_label_copy                1.5493          1  <- 泄漏
  cancellation_date_after_snapshot       0.6773          2  <- 泄漏
  snapshot_tenure_current_value         -0.6668          3  <- 泄漏
  plan_tier                              0.1306          4
  age                                    0.0306          5
  usage_score                            0.0192          6
  泄漏 |w| 合计 2.8934 / 真实 |w| 合计 0.1803 = 16.0x
```

三个泄漏特征拿走了 94% 的权重总量。真实信号被挤成了配角。这不是"模型有点过拟合",这是模型在告诉你:**它找到了一个捷径,而这个捷径的终点就是标签本身。**

训练过程同样值得看:二元交叉熵从 0.6931 降到 0.0331,收敛得干干净净。

```
epoch       BCE 损失
    1       0.693147
   30       0.064798
   60       0.033102
```

**损失曲线漂亮恰恰是坏消息。** 损失函数开心,是因为泄漏特征让这个任务变得平凡。真实信号能产生的损失有一个下限,泄漏把这个下限击穿了。审阅时看到损失一路向下,正确的反应不是放心,是去看它到底学到了什么。

### 模型 C:最难发现的那种

三个模型里,模型 C 最值得单独说。它只用三个正常字段加一个 `tenure`,特征表里**没有任何一列叫 cancelled,没有任何一列长得像标签**。而它的 AUC 是 0.9995。

```python
if cancelled:
    tenure_now = rng.uniform(0.5, 22.0)
else:
    tenure_now = rng.uniform(24.0, 55.0)
```

近泄漏不需要长得像答案,它只需要在**答案发生之后**才被确定。`snapshot_tenure_current_value` 满足这个条件:客户流失之后,他们的 tenure 就此定格;而一个还留着的客户,tenure 还在正常增长。这个字段和标签的相关性极强,但它在预测时点上根本不存在——那一刻你只能看到上个月的 tenure。

模型 C 的权重表里,`snapshot_tenure_current_value` 拿走了 `-1.9992`,压过全部三个真实特征加起来。

这就是近泄漏在现场的样子:**它不会触发任何警报,不会让任何指标变形,你只能靠"这个字段的值,在那一天真的拿得到吗"这个问题抓住它。**

## 3. AI 工程版:sklearn 管线,和一处更隐蔽的泄漏

```bash
/opt/homebrew/bin/python3 docs/tutorial/code/ch02/leakage_engineered.py
```

[leakage_engineered.py](code/ch02/leakage_engineered.py) 是同一份数据集的 sklearn 版本,文件头带生成记录(2026-10-03,提示词摘要写在 docstring 里)。两份代码共用同一个随机种子和同一套生成逻辑,所以数字可以直接对照。

它同时**故意保留了一处真实陷阱**:原始 AI 产出在切分前对全量数据 `fit` 了 `StandardScaler`,审阅者发现了、记录了,但代码没改——因为本章需要的正是让它跑起来的样子。

```python
# 原始产出:标准化在切分之前,fit 了全量 X(含测试集)
X_scaled, full_scaler = fit_transform_scale_scaler(X)
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_scaled, y, ...)
```

正确写法就多两行,把 scaler 放进 `Pipeline`,`fit` 只会作用在训练集上。

### 这处泄漏不会让 AUC 变高

跑出来:

```
路径 1  先 fit 标准化器再切分(原始产出,有泄漏) : AUC = 1.0000
路径 2  Pipeline 内先切分后 fit(已修正)       : AUC = 1.0000
路径 3  剔除全部泄漏字段,只用真实特征        : AUC = 0.6501
```

路径 1 和路径 2 的数据、切分方式、随机种子完全相同,只差 scaler `fit` 的时机。**AUC 一模一样。** 我在真实特征下单独验证过,也是逐位相同(0.6501)。

这不是 bug,这是本章最反直觉的一课:**AUC 是纯排序指标,只关心谁排在谁前面,对特征整体缩放完全不敏感。** 所以"先标准化再切分"这种泄漏**不会**在 AUC 上留下任何痕迹。你不可能靠"分数不对"发现它——分数根本不会不对。

### 那它到底破坏了什么

两处可验证的东西:

**第一,模型工件里嵌入了测试集统计量。**

```
  全量 fit 出来的 mean_ : [46.321   1.489   0.5011  0.1785  0.1255 36.0409]
  仅训练集 fit 的 mean_ : [46.2244  1.4888  0.5022  0.1756  0.1219 36.0642]
```

存下去的 `StandardScaler` 里有 20% 的样本根本属于测试集。你的"训练产物"从定义上就不再是纯训练产物。

**第二,L2 正则在错误的坐标系里生效。** sklearn 的 `C=1.0` 作用在标准化后的空间上;`fit` 全量数据时每个特征拿到的缩放系数变了,等于给不同特征施加了不同的正则强度:

```
   age                                  0.1019 ->   0.1005 (+1.34%)
   plan_tier                            0.2242 ->   0.2223 (+0.83%)
   usage_score                          0.0375 ->   0.0375 (+0.04%)
```

差异很小——但方向是对的:**污染确实存在,只是小到离线指标看不见。** 真正的风险在下游:线上换一个批次重新 fit 标准化器,或者数据分布随时间漂移,这个模型就对不上了,而你的离线指标永远看不到信号。

这一节的结论比结论本身更重要:**有两类泄漏,一类会让指标变形(标签泄漏),另一类不会(预处理泄漏)。** 只盯指标,你只能抓到前一半。剩下那一半只能靠读代码——看 `fit` 发生在哪一步、参数从哪个数据集来。

### 系数表:sklearn 版一样骗不了人

```
全特征
  cancelled_is_label_copy                2.6056          1  <- 泄漏
  snapshot_tenure_current_value         -1.2796          2  <- 泄漏
  cancellation_date_after_snapshot       1.1232          3  <- 泄漏
  plan_tier                              0.2223          4
  age                                    0.1005          5
  usage_score                            0.0375          6
  泄漏 |w| 合计 5.0083 / 真实 |w| 合计 0.3604 = 13.9x

仅真实特征
  plan_tier                              0.6137          1
  age                                    0.2299          2
  usage_score                            0.1720          3
```

换个库不改变任何事。sklearn 一样老实——它把泄漏字段的系数排在了第一、第二、第三位。手写实现和工程实现在这件事上给出完全一致的证据:**这是数据的问题,不是框架的问题。**

## 4. 审阅对照:拿清单去问这两份代码

打开 [REVIEW-CHECKLISTS.md](../../REVIEW-CHECKLISTS.md),找到这一节:

> ## Machine Learning Frameworks (ml-frameworks)
>
> **Gating principles:** generalization vs. memorization, bias-variance, train/serve skew.

这一章的三个泄漏例子刚好各打中一条:标签泄漏和近泄漏是 memorization,标准化泄漏是 train/serve skew。逐条对着这两份代码过一遍。

**"What exactly is being predicted, and does the label exist at prediction time in production? (If not: target leakage.)"**

第一条就拦下两个特征。`cancelled_is_label_copy` 是标签副本,`cancellation_date_after_snapshot` 在预测时点为空。**答案:不存在,这是 target leakage,两个字段必须划掉。**

这条问题是整套清单里投入产出比最高的一条,因为它不依赖任何工具、不需要跑任何代码,只需要你花三十秒逐字段问一遍。它也是唯一一条能在**代码还没写完的时候**就拦住泄漏的检查——特征列表一列出来,就该问这个问题。

**"How is the data split — random, temporal, grouped by entity? Does the split match how the system will be used?"**

两份代码都用的是时序切分(`split_by_time` / `shuffle=False`),这一条通过。但通过的原因值得说清楚:**测试集里同样带着那三列泄漏字段**,所以模型 A 的测试 AUC 一样是 1.0000。换一份测试集救不了你——那还是同一种泄漏,只是换了个切片。

这也是"测试 AUC 高"这个说法最危险的地方:它默认了"测试集能验证你的假设"。当泄漏存在于**表结构**里而不是**样本分布**里时,任何切分方式都验证不了它。切分的细节留给 ch03(泄漏的孪生兄弟),这里只记住一句:**切分解决不了字段时点问题。**

**"Is the same preprocessing applied identically at train and inference time?"**

这一条抓到 engineered 版的 `fit_transform_scale_scaler`。训练与推理的预处理不一致,就是 train/serve skew 的定义。sklearn 给的答案是 `Pipeline`:把 scaler 放进去,`fit` 只会作用在训练集,上线时整个 Pipeline 一起 pickle,训练和推理走同一条代码路径。

这一条之所以能抓到前面 AUC 抓不到的东西,是因为它问的是**代码结构**,不是指标。指标是结果,结构是原因——审阅要问的是原因。

**"Which metric is optimized, and how does it map to the business outcome? (AUC ≠ revenue ≠ user retention.)"**

1.0000 换算不出任何业务量。AUC 0.65 的真实模型意味着"名单前 10% 的客户里,实际流失率比总体高 0.65 倍对应的差距"——这是可以拿去和运营团队讨论的。1.0000 只能换算出一句话:名单最前面的人,就是那些即将在快照之后留下取消记录的人。这个"能力"没有业务价值,因为这些人本来就知道自己会流失。

**"What baseline does this beat — a constant, a rule, last week's value?"**

模型 B 的 0.6767 是唯一能和基线对话的数字。模型 A 的 1.0000 连基线都不需要——它已经满分了,而"满分"在这里恰恰是警报。

## 5. 仓库导航

先把历史数据导进本地库,再用查询 CLI 看 `ml-frameworks` 这个类目在 208 周里装了什么:

```bash
/opt/homebrew/bin/python3 scripts/import_history.py
/opt/homebrew/bin/python3 scripts/analyze.py data/history.db top ml-frameworks 5
```

第二条命令的输出:

```
PyTorch: rank=56.00 stars=94399 — Tensors and Dynamic neural networks in Python with strong GPU acceleration.
Tensorflow: rank=56.00 stars=195075 — An Open Source Machine Learning Framework for Everyone.
scikit-learn: rank=53.00 stars=63864 — scikit-learn: machine learning in Python.
Keras: rank=50.00 stars=63519 — Deep Learning for humans.
PaddlePaddle: rank=46.00 stars=23363 — PArallel Distributed Deep LEearning: Machine Learning Framework from Industrial Practice &.
```

`ml-frameworks` 是本清单里 rank 最高的一类。它回答的是 ch01 第 4 节那个问题的后半段:先定"X 到底该是什么",再决定要不要车。这一类里的所有项目——PyTorch、Tensorflow、scikit-learn、Keras——都在回答同一个问题:**当泄漏已经被排掉之后,你用哪个框架把剩下的信号挖出来。**

顺序不能反。用 Tensorflow 不会让 1.0000 变成 0.65;只有先承认 0.65 才是这份数据的上限,选 Tensorflow 才有意义。

想换个方向看,同一套命令把类目 ID 换掉即可:

```bash
/opt/homebrew/bin/python3 scripts/analyze.py data/history.db top time-series-data 5
/opt/homebrew/bin/python3 scripts/analyze.py data/history.db top tabular 5
```

时序数据那一条是 ch03 的入口,表格建模那一条是 ch05 的入口。

## 本章检查动作

回到 ch01 原则二说的那句:从 ch02 起,每章的手写实现都放在 `docs/tutorial/code/<章号>/<章号>_minimal.py`,用 `/opt/homebrew/bin/python3` 直接跑通,并打印出你要用来对照的量。这一章打印的是**权重表**。

审阅一份 AI 产出的流失/欺诈/风控模型时,按这个顺序做:

- **逐字段问预测时点。** 列出全部特征,逐个回答"做预测的那一刻,这个值拿得到吗"。这一条要在写代码之前做。
- **看权重表,不只看 AUC。** 权重排前三的字段,如果业务上讲不出为什么,它就是泄漏。
- **读 `fit` 的位置。** 标准化、填充、编码,凡是 fit 的对象,确认它只见过训练集。
- **切分解决不了字段时点问题。** 测试集里带着泄漏字段时,测试 AUC 和训练 AUC 一样漂亮。

下一章是 ch03,讲泄漏的孪生兄弟:**切分**。同一个失效模式,来源不同——ch02 的泄漏来自字段,ch03 的泄漏来自边界。
