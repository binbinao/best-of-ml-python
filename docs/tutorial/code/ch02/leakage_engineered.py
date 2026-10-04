#!/usr/bin/env python3
"""AI-generated engineered version, reviewed and corrected.

生成记录: 2026-10-03,提示词摘要="写一个客户流失预测的 sklearn 管线,含标准化、
逻辑回归、ROC-AUC 评估"
审阅发现: 原始产出在切分前对全量数据 fit 了 StandardScaler(泄漏)。

—— 注意:审阅发现**已记录,但未修复**。下面这份就是"记录了问题、修复没落地"
的真实形态,故意保留陷阱,供正文第 3 节与第 4 节逐行对照。修好的版本并排放
在 fit_pipeline_correct() 里。

    ⚠️  审阅发现(本文件实际状态,2026-10-04)
    ⚠️
    ⚠️  本文件**没有**修好。原始 AI 产出对全量数据 fit 了 StandardScaler;
    ⚠️  审阅者发现并记录了这个问题,但 main() 里的路径 1 仍照原样使用
    ⚠️  这份已污染的矩阵。
    ⚠️
    ⚠️  先说清楚这件事最反直觉的地方,因为它比"泄漏"本身更值得记住:
    ⚠️
    ⚠️  **测试 AUC 不会动。** 路径 1 与路径 2 的 AUC 逐位相同(1.0000)。
    ⚠️  但别把它读成"所以这处泄漏无害",下面两行是实测:
    ⚠️
    ⚠️      分数最大改动  max |Δp| = 5.21e-04
    ⚠️      排序改变的行  63 / 400 (15.8%)
    ⚠️
    ⚠️  模型确实变了,只是扰动小到没能让任何一对正负样本换序。AUC 保持
    ⚠️  沉默是**这个数据集、这个测试集占比下的巧合**,不是规律 ——
    ⚠️  扰动再大一点,换序就会发生。这正是 REVIEW-CHECKLISTS.md 那条
    ⚠️  "Is the same preprocessing applied identically at train and
    ⚠️  inference time?" 要抓的东西。
    ⚠️
    ⚠️  真正被破坏的是另外两件事:
    ⚠️
    ⚠️  1) 存下来的模型工件里嵌入了测试集统计量。pickle 出去的
    ⚠️     StandardScaler.mean_ 含 20% 测试数据,你的"训练产物"从定义上
    ⚠️     就不再是纯训练产物。
    ⚠️  2) L2 正则在错误的坐标系里生效。sklearn 的 C=1.0 作用在标准化后
    ⚠️     的空间上;全量 fit 改变了每个特征各自的仿射系数,等于给不同
    ⚠️     特征施加了不同的正则强度 —— 同一份数据、两条路径下 age 系数
    ⚠️     0.1052 vs 0.1037(+1.41%),测试 AUC 却都是 1.0000。
    ⚠️
    ⚠️  风险在下游:线上换一个批次重新 fit 标准化器,或数据分布随时间漂移,
    ⚠️  这个模型就对不上了,而离线指标看不到任何信号。
    ⚠️
    ⚠️  正确写法在 fit_pipeline_correct() 里:把 StandardScaler 放进
    ⚠️  Pipeline,fit 就只会作用在训练集上。两个函数并排放着,差异无处可藏。

复现:
    /opt/homebrew/bin/python3 docs/tutorial/code/ch02/leakage_engineered.py
"""
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261003
N_SAMPLES = 2000
TEST_SIZE = 0.2

REAL_FEATURES = ["age", "plan_tier", "usage_score"]
LEAKAGE_FEATURES = [
    "cancellation_date_after_snapshot",
    "cancelled_is_label_copy",
    "snapshot_tenure_current_value",
]
ALL_FEATURES = REAL_FEATURES + LEAKAGE_FEATURES
REJOIN_RATE = 0.06


# ---------------------------------------------------------------------------
# 数据:与 leakage_minimal.py 同一套构造,保证两章的数字可以直接对照
# ---------------------------------------------------------------------------
def make_dataset(n=N_SAMPLES, seed=SEED):
    """构造流失数据集,返回 (X, y)。

    与手写版逐字相同的生成逻辑 —— 唯一区别是这里用 numpy 抽样。
    如果两边数字对不上,说明两边不是同一份数据,对照就失去意义。
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 76, size=n).astype(float)
    plan_tier = rng.integers(0, 4, size=n).astype(float)
    usage_score = np.round(rng.random(n), 4)

    logit = (
        -2.6
        + 0.012 * (age - 40.0)
        + 0.60 * (plan_tier - 1.0)
        + 0.30 * usage_score
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    cancelled = (rng.random(n) < prob).astype(int)

    # (a) 未来信息:快照之后的取消日期。"取消后又回来"的客户同样非空,
    #     但标签为 0 —— 这一列因此不是标签的完美复制。
    churn_and_rejoin = ((cancelled == 0) & (rng.random(n) < REJOIN_RATE)).astype(int)
    cancellation_date_after = np.where(
        (cancelled == 1) | (churn_and_rejoin == 1), 1.0, 0.0
    )

    # (b) 标签副本:换了列名的 y 本身。
    label_copy = cancelled.astype(float)

    # (c) 近泄漏:用了当前 tenure 而非快照时刻的 tenure。
    #     两个区间**故意重叠**在 [24.0, 30.0],所以这是强相关而非确定性的
    #     标签函数(单列 AUC ≈ 0.99),和 leakage_minimal.py 完全一致。
    tenure_now = np.where(
        cancelled == 1,
        rng.uniform(0.5, 30.0, size=n),
        rng.uniform(24.0, 55.0, size=n),
    ).round(3)

    X = np.column_stack(
        [
            age,
            plan_tier,
            usage_score,
            cancellation_date_after,
            label_copy,
            tenure_now,
        ]
    )
    return X, cancelled.astype(int)


# ---------------------------------------------------------------------------
# 陷阱所在:先 fit 标准化器,后切分
# ---------------------------------------------------------------------------
def fit_transform_scale_scaler(X):
    """把标准化提到切分之前 —— 这就是原始 AI 产出干的事。

    StandardScaler 在这里 fit 了**全量 X**,包含后面要被当作测试集的那 20%。
    训练阶段因此知道了测试集的均值和标准差。

    返回 (变换后的矩阵, 那个被污染的 scaler),后者留给正文第 3 节做对照。
    """
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


def fit_pipeline(X_train, y_train):
    """原始产出的结构:scaler 已经在外面 fit 过了,这里只管训练逻辑回归。"""
    pipe = Pipeline(
        [
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def fit_pipeline_correct(X_train, y_train):
    """修好之后的写法:标准化器放进 Pipeline,只对训练集 fit。

    Pipeline.fit 只会把 fit 作用在训练集上,测试集在 transform 阶段才被看到。
    这是 sklearn 里防 train/serve skew 的标准手段。

    对比 fit_pipeline() 读一下差异:差的就是 Pipeline 里那两行。
    """
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def coefficient_table(pipe, features, title):
    """打印系数表 —— 在 AI 工程版里,权重依然是最快的一双眼睛。"""
    clf = pipe.named_steps["clf"]
    pairs = sorted(zip(features, clf.coef_[0]), key=lambda kv: -abs(kv[1]))
    print(title)
    print("  %-34s %10s %10s" % ("特征", "系数", "|系数|排名"))
    print("  " + "-" * 56)
    for rank, (name, coef) in enumerate(pairs, start=1):
        marker = "  <- 泄漏" if name in LEAKAGE_FEATURES else ""
        print("  %-34s %10.4f %10d%s" % (name, coef, rank, marker))
    coefs = clf.coef_[0]
    leak_total = sum(abs(c) for n, c in zip(features, coefs) if n in LEAKAGE_FEATURES)
    real_total = sum(abs(c) for n, c in zip(features, coefs) if n not in LEAKAGE_FEATURES)
    print("  %-34s %10.4f" % ("(intercept)", clf.intercept_[0]))
    if leak_total:
        print("  泄漏 |w| 合计 %.4f / 真实 |w| 合计 %.4f = %.1fx"
              % (leak_total, real_total, leak_total / real_total))


REVIEW_NOTES = """\
  [x]  标签在预测时点存在吗?—— cancelled_is_label_copy 是标签副本,
      cancellation_date_after_snapshot 是快照之后才写入的字段。
      两个都必须划掉。
  [x]  预处理在训练与推理时一致吗?—— 路径 1 不一致:标准化器 fit 了
      全量数据,测试集的均值方差进入了训练。这是 train/serve skew。
  [ ]  切分方式匹配真实用法吗?—— 本文件用 shuffle=False 的时序切分,
      这一条通过;随机切分的风险留给 ch03。
  [ ]  这个 AUC 换算成业务是多少?—— {auc:.4f} 换算不出任何东西:名单上
      排在前面的客户,是那些即将在快照之后留下取消记录的客户。

  这份代码测试全过、类型干净、管道封装正规。它会在评审会上通过。
  拦住它的不是测试,是上面那两条逐字段的提问。
"""


def main():
    started = time.time()
    X, y = make_dataset()

    print("=" * 74)
    print("ch02 AI 工程版:sklearn 管线 + 一处故意留下的标准化泄漏")
    print("=" * 74)
    print()
    print("样本 %d  特征 %d  流失率 %.1f%%"
          % (len(y), len(ALL_FEATURES), 100.0 * float(y.mean())))
    print()

    # 切分只做一次,两条路径共用同一批训练/测试行 —— 否则两个 AUC
    # 是在不同的测试集上算的,连"只差 fit 时机"这个前提都不成立。
    all_idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=TEST_SIZE, shuffle=False, random_state=SEED
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ---- 路径 1:原始产出。标准化在切分之前 fit 了全量数据 ----
    # 先对**全量**X 标准化,再取已经切好的那批行 —— 这就是"先 fit 后切分"
    # 写进代码里的样子:切分逻辑没变,只是喂给它的数据已经带着测试集信息。
    full_scaler = StandardScaler()
    X_all_scaled = full_scaler.fit_transform(X)      # ← 泄漏发生在这里
    X_train_l = X_all_scaled[train_idx]
    X_test_l = X_all_scaled[test_idx]
    pipe_leaky = fit_pipeline(X_train_l, y_train)
    auc_leaky = roc_auc_score(y_test, pipe_leaky.predict_proba(X_test_l)[:, 1])

    # ---- 路径 2:修好之后。标准化只在 Pipeline 内对训练集 fit ----
    pipe_clean = fit_pipeline_correct(X_train, y_train)
    auc_clean = roc_auc_score(y_test, pipe_clean.predict_proba(X_test)[:, 1])
    train_scaler = pipe_clean.named_steps["scaler"]

    # ---- 路径 3:彻底剔除泄漏字段,只用真实信号 ----
    real_idx = [ALL_FEATURES.index(name) for name in REAL_FEATURES]
    pipe_real = fit_pipeline_correct(X_train[:, real_idx], y_train)
    auc_real = roc_auc_score(
        y_test, pipe_real.predict_proba(X_test[:, real_idx])[:, 1]
    )

    print("-" * 74)
    print("三条路径的测试 AUC")
    print("-" * 74)
    print("路径 1  先 fit 标准化器再切分(原始产出,有泄漏) : AUC = %.4f" % auc_leaky)
    print("路径 2  Pipeline 内先切分后 fit(已修正)       : AUC = %.4f" % auc_clean)
    print("路径 3  剔除全部泄漏字段,只用真实特征        : AUC = %.4f" % auc_real)
    print()
    print("路径 1 与路径 2 用的是**同一批**训练/测试行、同一个随机种子,")
    print("只差 StandardScaler fit 的时机。两条路径的 AUC 完全相同。")
    print()
    print("但分数并没有相同 —— 这才是这件事需要小心的地方:")
    print()
    pred_leaky = pipe_leaky.predict_proba(X_test_l)[:, 1]
    pred_clean = pipe_clean.predict_proba(X_test)[:, 1]
    delta = np.abs(pred_leaky - pred_clean)
    rank_changed = int((np.argsort(np.argsort(pred_leaky))
                        != np.argsort(np.argsort(pred_clean))).sum())
    print("  分数最大改动      max |Δp| = %.2e" % delta.max())
    print("  排序发生变化的行   %d / %d (%.1f%%)"
          % (rank_changed, len(pred_leaky), 100.0 * rank_changed / len(pred_leaky)))
    print("  AUC               %.10f vs %.10f(逐位相同)"
          % (auc_leaky, auc_clean))
    print()
    print("模型确实变了,只是这个扰动小到没能让任何一对正负样本换序,")
    print("所以 AUC 这一个指标选择性地保持了沉默。")
    print()
    print("这里要说准确:AUC 对**分数的仿射缩放**不敏感(整体乘 c 不改变排序),")
    print("但全量 fit 并不是整体乘一个常数 —— 它改变的是每个特征各自的仿射")
    print("系数,于是 L2 的作用点也跟着变(见下面第 2 点)。学到的函数变了,")
    print("AUC 没变。这是这个数据集、这个测试集占比下的巧合,不是规律:")
    print("扰动再大一点,换序就会发生,AUC 就会动。")
    print()
    print("所以正确的结论不是\"这处泄漏不影响指标\",而是:")
    print("**别依赖指标替你发现泄漏。** 问的是代码结构 —— fit 发生在哪一步。")
    print()
    print("那它到底破坏了什么?两处可验证的东西:")
    print()
    print("  1) 模型工件里嵌入了测试集统计量")
    print("     全量 fit 出来的 mean_ : %s" % np.round(full_scaler.mean_, 4))
    print("     仅训练集 fit 的 mean_ : %s" % np.round(train_scaler.mean_, 4))
    print("     存下去的 StandardScaler 里,有 %d%% 的样本根本属于测试集。"
          % int(round(100.0 * len(X_test) / len(X))))
    print("     你的“训练产物”从定义上就不再是纯训练产物。")
    print()
    print("  2) L2 正则在错误的坐标系里生效")
    print("     sklearn 的 C=1.0 作用在标准化后的空间上;缩放系数变了,")
    print("     等于给不同特征施加了不同的正则强度。系数差异:")
    leaky_coef = pipe_leaky.named_steps["clf"].coef_[0]
    clean_coef = pipe_clean.named_steps["clf"].coef_[0]
    for name, a_c, c_c in zip(ALL_FEATURES, leaky_coef, clean_coef):
        if name in LEAKAGE_FEATURES:
            continue
        drift = 100.0 * (a_c - c_c) / abs(c_c) if c_c else 0.0
        print("       %-34s %8.4f -> %8.4f (%+.2f%%)"
              % (name, a_c, c_c, drift))
    print()
    print("风险在下游:线上换一个批次重新 fit 标准化器,或数据随时间漂移,")
    print("这个模型就对不上了,而离线指标永远看不到信号。")
    print()
    print("而路径 2 与路径 3 之间差 %.4f,那才是泄漏特征贡献的部分。"
          % (auc_clean - auc_real))
    print("模型算得越漂亮,你和业务之间的距离就越远。")
    print()

    print("-" * 74)
    print("系数表:路径 2(预处理已修正,但特征表仍含泄漏字段)")
    print("-" * 74)
    print()
    coefficient_table(pipe_clean, ALL_FEATURES, "全特征")
    print()
    coefficient_table(pipe_real, REAL_FEATURES, "仅真实特征")
    print()

    print("-" * 74)
    print("审阅结论")
    print("-" * 74)
    print(REVIEW_NOTES.format(auc=auc_clean))

    print("=" * 74)
    print("三个数字的读法:")
    print("  %.4f  泄漏特征在报数,不是模型在预测。" % auc_clean)
    print("  %.4f  真实特征能达到的水平(不是理论上限,大概还能到 0.69)。" % auc_real)
    print("  路径 1 与路径 2 的 AUC 相差 0,但分数最大改动 %.1e ——"
          % delta.max())
    print("         分数变了、排序变了,只是没到让 AUC 变动的程度。")
    print("         所以这处泄漏只能靠读代码发现,不能靠看指标发现。")
    print("全部训练与评估耗时 %.2f 秒(3 条 sklearn 管线)。" % (time.time() - started))


if __name__ == "__main__":
    main()
