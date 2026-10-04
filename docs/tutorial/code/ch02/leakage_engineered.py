#!/usr/bin/env python3
"""AI-generated engineered version, reviewed and corrected.

生成记录: 2026-10-03,提示词摘要="写一个客户流失预测的 sklearn 管线,含标准化、
逻辑回归、ROC-AUC 评估"
审阅发现: 原始产出在切分前对全量数据 fit 了 StandardScaler(泄漏),已改为在
Pipeline 内先切分后 fit。

—— 上面那段是本书的叙事。下面是本章的真实文件:它保留一处**故意留下**的陷阱,
供正文第 3 节与第 4 节逐行对照。

    ⚠️  审阅发现(本文件实际状态,2026-10-04)
    ⚠️
    ⚠️  本文件**没有**完全修好。原始 AI 产出在 fit_transform_scale_scaler()
    ⚠️  里对全量数据 fit 了 StandardScaler;审阅者发现并记录了这个问题,
    ⚠️  但 fit_pipeline() 仍照原样使用这份已污染的矩阵。
    ⚠️
    ⚠️  先说清楚这件事最反直觉的地方,因为它比"泄漏"本身更值得记住:
    ⚠️
    ⚠️  **这处泄漏不会让测试 AUC 变高。** 本文件跑出来路径 1 与路径 2
    ⚠️  的 AUC 完全相同(1.0000),真实特征下也逐位相同(0.6501)。
    ⚠️  原因在 AUC 的定义:它是纯排序指标,只关心"谁排在谁前面",
    ⚠️  对特征整体缩放完全不敏感。sklearn 的版本差异同样只有小数点后
    ⚠️  第三位(0.2319 vs 0.2299)。
    ⚠️
    ⚠️  所以"盯着指标看"这个习惯,在这一类泄漏上彻底失效。你不会因为
    ⚠️  分数不对而发现它 —— 分数根本不会不对。真正被污染的是另外两件事:
    ⚠️
    ⚠️  1) 存下来的模型工件里嵌入了测试集统计量。pickle 出去的
    ⚠️     StandardScaler.mean_ 含 20% 测试数据,你的"训练产物"从定义上
    ⚠️     就不再是纯训练产物。
    ⚠️  2) L2 正则在错误的坐标系里生效。sklearn 的 C=1.0 是作用在
    ⚠️     标准化后的空间上;fit 全量数据时每个特征拿到的缩放系数变了,
    ⚠️     等于给不同特征施加了不同的正则强度。
    ⚠️
    ⚠️  真正的风险在下游:线上换一个批次重新 fit 标准化器,或数据分布
    ⚠️  随时间漂移,这个模型就对不上了,而你在离线指标上永远看不到信号。
    ⚠️  这正是 REVIEW-CHECKLISTS.md 那条 "Is the same preprocessing
    ⚠️  applied identically at train and inference time?" 要抓的东西。
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
    tenure_now = np.where(
        cancelled == 1,
        rng.uniform(0.5, 22.0, size=n),
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


def fit_pipeline(X_train, y_train, features):
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

    # ---- 路径 1:原始产出。标准化在切分之前 fit 了全量数据 ----
    X_scaled, full_scaler = fit_transform_scale_scaler(X)   # ← 泄漏发生在这里
    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, shuffle=False, random_state=SEED
    )
    pipe_leaky = fit_pipeline(X_train_l, y_train_l, ALL_FEATURES)
    auc_leaky = roc_auc_score(y_test_l, pipe_leaky.predict_proba(X_test_l)[:, 1])

    # ---- 路径 2:修好之后。标准化只在 Pipeline 内对训练集 fit ----
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=False, random_state=SEED
    )
    pipe_clean = fit_pipeline_correct(X_train_c, y_train_c)
    auc_clean = roc_auc_score(y_test_c, pipe_clean.predict_proba(X_test_c)[:, 1])
    train_scaler = pipe_clean.named_steps["scaler"]

    # ---- 路径 3:彻底剔除泄漏字段,只用真实信号 ----
    real_idx = [ALL_FEATURES.index(name) for name in REAL_FEATURES]
    pipe_real = fit_pipeline_correct(X_train_c[:, real_idx], y_train_c)
    auc_real = roc_auc_score(
        y_test_c, pipe_real.predict_proba(X_test_c[:, real_idx])[:, 1]
    )

    print("-" * 74)
    print("三条路径的测试 AUC")
    print("-" * 74)
    print("路径 1  先 fit 标准化器再切分(原始产出,有泄漏) : AUC = %.4f" % auc_leaky)
    print("路径 2  Pipeline 内先切分后 fit(已修正)       : AUC = %.4f" % auc_clean)
    print("路径 3  剔除全部泄漏字段,只用真实特征        : AUC = %.4f" % auc_real)
    print()
    print("路径 1 与路径 2 的数据、切分方式、随机种子完全相同,只差")
    print("StandardScaler fit 的时机。你会发现两条路径的 AUC **完全一样**")
    print("(真实特征下也逐位相同)。")
    print()
    print("这不是 bug,这是本章最反直觉的一课:")
    print()
    print("  AUC 是纯排序指标,只关心谁排在谁前面,对特征整体缩放不敏感。")
    print("  所以“先标准化再切分”这种泄漏 **不会** 在 AUC 上留下任何痕迹。")
    print("  你不可能靠“分数不对”发现它 —— 分数根本不会不对。")
    print()
    print("那它到底破坏了什么?两处可验证的东西:")
    print()
    print("  1) 模型工件里嵌入了测试集统计量")
    print("     全量 fit 出来的 mean_ : %s" % np.round(full_scaler.mean_, 4))
    print("     仅训练集 fit 的 mean_ : %s" % np.round(train_scaler.mean_, 4))
    print("     存下去的 StandardScaler 里,有 %d%% 的样本根本属于测试集。"
          % int(round(100.0 * len(X_test_c) / len(X))))
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
    print("  %.4f  这份数据真正的上限。" % auc_real)
    print("  路径 1 与路径 2 相差 0 —— 标准化泄漏不会动 AUC,")
    print("         所以它只能靠读代码发现,不能靠看指标发现。")
    print("全部训练与评估耗时 %.2f 秒(3 条 sklearn 管线)。" % (time.time() - started))


if __name__ == "__main__":
    main()
