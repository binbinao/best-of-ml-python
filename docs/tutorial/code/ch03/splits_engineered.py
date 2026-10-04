#!/usr/bin/env python3
"""AI-generated engineered version, reviewed and corrected.

生成记录: 2026-10-03,提示词摘要="写一个设备故障预测的 sklearn 训练脚本,
用 train_test_split 切分,训练逻辑回归并输出 AUC"
审阅发现: 原始产出只有一行 `train_test_split(df, test_size=0.2,
random_state=42)`,没有任何关于实体结构和时间顺序的考虑。

    ⚠️  审阅发现(本文件实际状态,2026-10-04)
    ⚠️
    ⚠️  main() 里的**路径 1 就是那行原始产出**,一字未改。它仍然是这份
    ⚠️  脚本里跑出来的最高分,而它之所以高分,是因为 60 台设备全部同时
    ⚠️  出现在训练集和测试集里。
    ⚠️
    ⚠️  说准确一点,免得下结论走偏:**一半的指标会动,而且幅度大得刺眼。**
    ⚠️  同一份数据、同一组种子,只把切分换成 GroupShuffleSplit:
    ⚠️
    ⚠️      设备先验估计器  0.8048 -> 0.5000   (-0.3048)  <- 塌成常数
    ⚠️      特征逻辑回归    0.8006 -> 0.8439   (+0.0433)  <- 不但没掉,还涨了
    ⚠️
    ⚠️  第二行是这份文件最反直觉的地方,也是最容易讲错的地方:
    ⚠️  **不要把它总结成「group 切分更保守所以更差」。** 逻辑回归的
    ⚠️  系数在两条路径下几乎不变(见 main() 输出),模型没变,变的是测试集。
    ⚠️  随机切分的测试集里,每台设备同时出现在训练和测试,同类样本被
    ⚠️  稀释;group 切分顺手消掉了这一点,是净收益。
    ⚠️
    ⚠️  诚实的结论:**切分泄漏不是对任何模型都抬分,它只对会记住实体的**
    ⚠️  **模型抬分。** 而你事先通常不知道线上跑的是哪种模型 —— 这才是
    ⚠️  默认切分不能是随机的理由,也正是它比 ch02 的字段泄漏更难发现的原因:
    ⚠️  ch02 的泄漏能从权重表上看出来,这一章的看不出来。
    ⚠️
    ⚠️  修好之后路径 2/3/4 用的是 sklearn 的 GroupShuffleSplit 与
    ⚠️  TimeSeriesSplit,重叠压到 0。**但别把"重叠 = 0"读成"切分对了"**:
    ⚠️  路径 3 的组重叠是 16/20,它只修了时间方向,没修实体那条线。

    用法:
        /opt/homebrew/bin/python3 docs/tutorial/code/ch03/splits_engineered.py
"""
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261003
TEST_SIZE = 0.2

FEATURES = ["hours_z", "vibration_rms", "ambient"]


# ---------------------------------------------------------------------------
# 数据:与 splits_minimal.py 同一套构造(那个文件里的 make_dataset 原样搬过来),
# 保证两章的数字可以直接对照。
# ---------------------------------------------------------------------------
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from splits_minimal import FEATURES as _F, make_dataset  # noqa: E402

FEATURES = _F


def to_arrays(records):
    """list[dict] -> (X, y, groups, 时间, factory_groups)。"""
    X = np.array([[r[k] for k in FEATURES] for r in records], dtype=float)
    y = np.array([r["failed"] for r in records], dtype=int)
    groups = np.array([r["device"] for r in records])
    factory_groups = np.array([r["factory"] for r in records])
    time_key = np.array([r["abs_day"] for r in records])
    return X, y, groups, factory_groups, time_key


# ---------------------------------------------------------------------------
# 切分:四个函数,全部返回 (train_idx, test_idx)
# ---------------------------------------------------------------------------
def ai_default_split(X, y, groups, factory_groups, time_key):
    """⚠️ 陷阱所在:原始 AI 产出,一字未改。

    这是代码示例里最常见的一行切分。它没问过两件事:
      1. 数据里有没有实体(同一台设备的多条记录)?
      2. 数据是不是按时间排的?
    两个问题的答案在这份数据里都是"有"。而这一行不打算知道。
    """
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=42)
    return train_idx, test_idx


def group_shuffle_split(X, y, groups, factory_groups, time_key):
    """sklearn 的 GroupShuffleSplit:整台设备留出,等价于最小版的 group_split。"""
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return train_idx, test_idx


def time_series_split_cv(X, y, groups, factory_groups, time_key):
    """sklearn 的 TimeSeriesSplit:单条时间轴上的滚动窗口。

    本章用它做两件事:(a) 取最后一个折当作"未来期测试集";
    (b) 打印前面几折的规模,说明它和 GroupShuffleSplit 解决的是不同问题。
    它**按行号切**,所以前提是 X 已经按时间排好 —— 本文件用 np.argsort
    保证这一点,现实里这一步经常被忘掉:数据从数据库捞出来时通常按主键
    排,不是按时间排,而 TimeSeriesSplit 不检查、不警告,照切不误。
    """
    order = np.argsort(time_key, kind="stable")
    Xs, ys = X[order], y[order]
    tss = TimeSeriesSplit(n_splits=4)
    folds = list(tss.split(Xs))
    train_rel, test_rel = folds[-1]
    return order[train_rel], order[test_rel], folds, Xs


def group_split_by_factory(X, y, groups, factory_groups, time_key):
    """按工厂留出,留环境应力最高的那座。sklearn 换 groups 即可,不需要新类。

    单独列出来是因为它是本文件里**唯一测出干净分布偏移**的切分:
    留出整座工厂 = 训练集里根本没有那个环境条件,重叠同样是 0。
    与路径 2 的对照说明:**重叠 = 0 只是及格线,不是及格。**
    """
    # GroupShuffleSplit 换 groups 就能按工厂切,但它随机挑工厂,拿到的
    # 偏移量取决于运气。这里显式留出 ambient 最高的那座,和
    # splits_minimal.py 的 group_split(level="factory") 保持一致。
    stress = {f: X[factory_groups == f, FEATURES.index("ambient")][0] for f in set(factory_groups)}
    hottest = max(stress, key=stress.get)
    test_idx = np.where(factory_groups == hottest)[0]
    train_idx = np.where(factory_groups != hottest)[0]
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# 两个估计器
# ---------------------------------------------------------------------------
def device_prior_scores(records, train_idx, test_idx):
    """设备先验:测试行属于哪台设备就用那台的训练故障率;没见过的退回总体值。"""
    pos, tot, n, p = {}, {}, 0, 0
    for i in train_idx:
        r = records[i]
        dev = r["device"]
        tot[dev] = tot.get(dev, 0) + 1
        pos[dev] = pos.get(dev, 0) + r["failed"]
        n += 1
        p += r["failed"]
    overall = p / n
    scores = np.empty(len(test_idx))
    unseen = 0
    for k, i in enumerate(test_idx):
        dev = records[i]["device"]
        if dev in tot:
            scores[k] = pos[dev] / tot[dev]
        else:
            scores[k] = overall
            unseen += 1
    return scores, unseen


def fit_logreg(X_train, y_train, X_test):
    """标准化放进 Pipeline —— fit 只见过训练集(这一点不要学路径 1 之外的任何写法)。"""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe.predict_proba(X_test)[:, 1], pipe


def main():
    started = time.time()
    records = make_dataset()
    X, y, groups, factory_groups, time_key = to_arrays(records)
    print("=== 数据 ===")
    print("  %d 行 / %d 台设备 / %d 座工厂,总体故障率 %.2f%%"
          % (len(y), len(set(groups)), len(set(factory_groups)), 100.0 * y.mean()))
    print()

    results = []

    # --- 路径 1:原始 AI 产出,不改 ---
    tr, te = ai_default_split(X, y, groups, factory_groups, time_key)
    print("--- 路径 1:AI 默认产出 train_test_split(test_size=0.2, random_state=42) ---")
    print("    切完就知道它是错的,但指标不会告诉你:")
    print("      训练/测试里同时出现的设备: %d / %d"
          % (len(set(groups[tr]) & set(groups[te])), len(set(groups[te]))))
    future = sum(1 for i in te if (time_key[tr][groups[tr] == groups[i]] > time_key[i]).any())
    print("      测试行读到同设备『更晚』训练行的: %d / %d (%.1f%%)"
          % (future, len(te), 100.0 * future / len(te)))
    print()

    # --- 路径 2/3/4 ---
    tr2, te2 = group_shuffle_split(X, y, groups, factory_groups, time_key)
    tr3, te3, folds, Xs = time_series_split_cv(X, y, groups, factory_groups, time_key)
    tr4, te4 = group_split_by_factory(X, y, groups, factory_groups, time_key)

    for name, tr_i, te_i in [
        ("路径 1  train_test_split(AI 默认)", tr, te),
        ("路径 2  GroupShuffleSplit(按设备)", tr2, te2),
        ("路径 3  TimeSeriesSplit(最后折)", tr3, te3),
        ("路径 4  GroupShuffleSplit(按工厂)", tr4, te4),
    ]:
        overlap = len(set(groups[tr_i]) & set(groups[te_i]))
        test_groups = len(set(groups[te_i]))
        future = sum(1 for i in te_i if (time_key[tr_i][groups[tr_i] == groups[i]] > time_key[i]).any())
        d_hours = X[tr_i, 0].mean() - X[te_i, 0].mean()
        d_ambient = X[tr_i, 2].mean() - X[te_i, 2].mean()

        scores_a, unseen = device_prior_scores(records, tr_i, te_i)
        auc_a = roc_auc_score(y[te_i], scores_a)
        scores_b, pipe = fit_logreg(X[tr_i], y[tr_i], X[te_i])
        auc_b = roc_auc_score(y[te_i], scores_b)

        results.append(
            dict(name=name, overlap=overlap, test_groups=test_groups,
                 future=future, n_test=len(te_i), d_hours=d_hours, d_ambient=d_ambient,
                 auc_a=auc_a, auc_b=auc_b, unseen=unseen, coef=pipe.named_steps["clf"].coef_[0])
        )

    print("=== TimeSeriesSplit 的折规模(说明它和 GroupShuffleSplit 解决不同问题) ===")
    for k, (a, b) in enumerate(folds):
        print("    折 %d  训练 %3d 行 / 测试 %3d 行" % (k + 1, len(a), len(b)))
    print("    它按行号切,前提是 X 已按时间排好;本文件用 np.argsort 保证。")
    print("    现实里这一步经常被忘掉 —— 数据从库里捞出来通常按主键排,不是按")
    print("    时间排,而 TimeSeriesSplit 不检查、不警告,照切不误。")
    print()

    print("=== 切分质量 ===")
    print("  切分                              组重叠      未来行泄漏     hours_z 均值差  ambient 均值差")
    print("  " + "-" * 98)
    for r in results:
        print("  %-32s %2d/%-2d %5.1f%%  %4d (%5.1f%%)  %+14.4f  %+14.4f"
              % (r["name"], r["overlap"], r["test_groups"],
                 100.0 * r["overlap"] / r["test_groups"],
                 r["future"], 100.0 * r["future"] / r["n_test"],
                 r["d_hours"], r["d_ambient"]))
    print()

    print("=== 测试 AUC ===")
    print("  A 设备先验(记实体)   B 特征逻辑回归(只认特征)")
    print("  " + "-" * 66)
    for r in results:
        print("  %-32s A = %.4f        B = %.4f" % (r["name"], r["auc_a"], r["auc_b"]))
    print()

    for r in results:
        if r["unseen"]:
            print("  %-32s A 有 %d/%d 条测试行退回总体故障率(设备没在训练集里)"
                  % (r["name"], r["unseen"], r["n_test"]))
    print()

    print("=== 路径 1 vs 路径 2:只把切分换成 GroupShuffleSplit,模型结构一字未改 ===")
    r1, r2 = results[0], results[1]
    print("  设备先验   AUC %.4f -> %.4f  (%+.4f)" % (r1["auc_a"], r2["auc_a"], r2["auc_a"] - r1["auc_a"]))
    print("  逻辑回归   AUC %.4f -> %.4f  (%+.4f)" % (r1["auc_b"], r2["auc_b"], r2["auc_b"] - r1["auc_b"]))
    print()
    print("  逻辑回归系数(路径 1 训练 / 路径 2 训练,同一个 Pipeline 结构):")
    for j, f in enumerate(FEATURES):
        print("    %-14s %+9.4f -> %+9.4f" % (f, r1["coef"][j], r2["coef"][j]))
    print()
    print("  三个数字,三个不同的事实,别混成一句:")
    print("    1) 系数基本没动。**泄漏不改变模型,只改变测试集。** 所以 ch02")
    print("       那套『看权重表』的检查在这一章抓不到任何东西 ——")
    print("       泄漏不在字段里,在边界上。")
    print("    2) 会记实体的设备先验从 %.4f 掉到 %.4f。它需要的每一条"
          % (r1["auc_a"], r2["auc_a"]))
    print("       测试信息,在 GroupShuffleSplit 下它一条都拿不到,于是退回")
    print("       常数,AUC 就是 0.5000。**这是真能力,不是 bug。**")
    print("    3) 只认特征的逻辑回归不但没掉,%+.4f。**别把这读成『group 切分"
          % (r2["auc_b"] - r1["auc_b"]))
    print("       更保守所以更差』—— 它变好是因为 group 切分顺手干掉了另一件")
    print("       好事:测试设备不再在训练集里出现,同类样本不再被稀释。")
    print("       两种效应方向相反,净结果是正的。**所以在本数据集上,随机")
    print("       切分相对 group 切分对逻辑回归并没有系统性乐观。**")
    print()
    print("  诚实的结论:切分泄漏**不是**对任何模型都抬分。它只对会记住")
    print("  实体的模型抬分。而你事先通常不知道上线跑的是哪种 —— 这就是")
    print("  默认切分不能是随机的理由,而不是『随机一定让指标虚高』。")
    print()
    print("  跑完 %.2f 秒(4 条 sklearn 路径)。" % (time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
