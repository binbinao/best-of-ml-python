#!/usr/bin/env python3
"""AI-generated engineered version, reviewed and corrected.

生成记录: 2026-10-03,提示词摘要="写一个设备故障预测的 sklearn 训练脚本,
用 train_test_split 切分,训练逻辑回归并输出 AUC"
审阅发现: 原始产出只有一行 `train_test_split(df, test_size=0.2,
random_state=42)`,没有任何关于实体结构和时间顺序的考虑。

    ⚠️  审阅发现(本文件实际状态,2026-10-04)
    ⚠️
    ⚠️  main() 里的**路径 1 就是那行原始产出**,一字未改。它跑出来
    ⚠️  的分数明显高于其余三条路径,而它之所以高分,是因为 60 台设备
    ⚠️  全部同时出现在训练集和测试集里。
    ⚠️
    ⚠️  说准确一点,免得下结论走偏 —— 这里有两个数字,量级完全不同:
    ⚠️
    ⚠️      设备先验估计器  0.8048 -> 0.5000   (-0.3048)
    ⚠️      特征逻辑回归    0.8006 -> 0.8439   (+0.0433)
    ⚠️
    ⚠️  第一行是**恒等式,不是测量值**:group 切分下设备交叠恒为 0,
    ⚠️  设备先验对每条测试行都退回常数,常数预测的 AUC 就是 0.5。
    ⚠️  拿它和 0.8048 比差额,是拿构造性常数和随机变量作比较。
    ⚠️
    ⚠️  第二行是本文件最容易讲错的地方。**+0.0433 不能当结论**:
    ⚠️  切分抽样本身带来 sd≈0.05 的波动,这个差值在噪声量级以内。
    ⚠️  30 组种子重采样(同数据集、同 Pipeline 结构,见 splits_minimal.py
    ⚠️  的 seed_sweep)给出:
    ⚠️
    ⚠️      AUC(group) - AUC(random) 均值 -0.0058
    ⚠️                               95% CI [-0.0228, +0.0112]
    ⚠️
    ⚠️  **区间跨零**,group 优于 random 的种子只占 43.3%,而单次那个
    ⚠️  +0.0433 落在分布第 57 百分位附近 —— 换一次切分抽样符号就可能反转。
    ⚠️  所以既不能说「group 切分更保守所以更差」,也不能说「随机切分虚高」。
    ⚠️
    ⚠️  代价真正可测的是**记实体**的估计器,测法见 splits_minimal.py 的
    ⚠️  seen_device_subset():"设备见过、未来没见过"的 160 条干净测试行上,
    ⚠️  设备先验只有 0.5963,而随机切分给它 0.7794 —— 0.5963 落在该估计器
    ⚠️  30 组随机切分分布的最小值(0.7321)之下,远超抽样噪声。
    ⚠️
    ⚠️  ⚠️  系数**不是**"基本没动":hours_z -11.79%、ambient -11.96%。
    ⚠️  ⚠️  权重表在这里**有信号,但不能定位原因** —— 系数变了,既可能
    ⚠️  ⚠️  是训练集构成变了,也可能是别的。ch02 那套"看权重表"的检查
    ⚠️  ⚠️  在这一章**不足以定论**,详见正文第 5 节。
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
        ("路径 4  显式留出最热工厂", tr4, te4),
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
        d = r2["coef"][j] - r1["coef"][j]
        pct = 100.0 * d / abs(r1["coef"][j]) if r1["coef"][j] else float("nan")
        print("    %-14s %+9.4f -> %+9.4f  (%+.4f, %+.2f%%)" % (f, r1["coef"][j], r2["coef"][j], d, pct))
    print()
    print("  三件事,别混成一句:")
    print("    1) 系数**动了**,hours_z %+.2f%%、ambient %+.2f%%。所以 ch02 那句"
          % (100.0 * (r2["coef"][0] - r1["coef"][0]) / abs(r1["coef"][0]),
             100.0 * (r2["coef"][2] - r1["coef"][2]) / abs(r1["coef"][2])))
    print("       『泄漏能从权重表上看出来』在这里**不成立**:权重表有信号,")
    print("       但信号本身**不能定位原因** —— 系数变了既可能是边界画错了,")
    print("       也可能只是训练集构成变了。**要定论必须回去看组重叠。**")
    print("    2) 设备先验 %.4f -> %.4f 是**恒等式不是测量值**:交叠恒为 0,"
          % (r1["auc_a"], r2["auc_a"]))
    print("       每条测试行都退回常数。别拿它和 %.4f 比差额。" % r1["auc_a"])
    print("    3) 逻辑回归 %+.4f **不能当结论**:切分抽样本身 sd≈0.05。"
          % (r2["auc_b"] - r1["auc_b"]))
    print("       30 组种子重采样(见 splits_minimal.py 的 seed_sweep)给出均值")
    print("       -0.0058、95% CI [-0.0228, +0.0112]:**区间跨零**,")
    print("       group 优于 random 的种子只占 43.3%。")
    print("       **既不能说『group 更保守所以更差』,也不能说『随机虚高』。**")
    print()
    print("  诚实的结论:切分泄漏**不是**对任何模型都抬分。它只对会记住")
    print("  实体的模型抬分 —— 而那正是**恒等式**之外真正能测到的那一类,")
    print("  测法见 splits_minimal.py 的 seen_device_subset()。")
    print("  你事先通常不知道上线跑的是哪种,这就是默认切分不能是随机的理由。")
    print()
    print("  跑完 %.2f 秒(4 条 sklearn 路径)。" % (time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
