#!/usr/bin/env python3
"""Leakage demo: a model that scores perfectly by cheating.

纯 Python 标准库实现逻辑回归,把"泄漏"从概念变成一张可看见的权重表。
无 sklearn / numpy / pandas;固定随机种子,输出可复现。

构造的客户流失数据集里故意埋了三类泄漏:

  (a) cancellation_date_after_snapshot —— 目标泄漏(未来信息)
      快照之后才写入的取消日期。非流失样本该字段为空,记 0。
  (b) cancelled_is_label_copy          —— 目标泄漏(标签副本)
      标签本身换了列名。教科书级的泄漏,也是唯一能把 AUC 顶到 1.0 的那种。
  (c) snapshot_tenure_current_value    —— 近泄漏(时点错配)
      用了"当前"tenure 而不是快照时刻的 tenure。与标签强相关,
      但这个值在流失发生之后才最终确定,预测时点拿不到。

真实弱信号(预测时点确实拿得到):
  (d) age, plan_tier, usage_score

三段式对比:训练曲线 / AUC / 权重表,再加一张 ASCII ROC 曲线。
样本量 2000、6 特征、60 轮批量梯度下降,纯 Python 约 2 秒内跑完;
教学代码以可读性优先,不做任何性能优化。

用法:
    /opt/homebrew/bin/python3 docs/tutorial/code/ch02/leakage_minimal.py
"""
import math
import random
import sys
import time

SEED = 20261003
N_SAMPLES = 2000
EPOCHS = 60
LEARNING_RATE = 0.5
TEST_FRACTION = 0.2  # 前 80% 训练 / 后 20% 测试

LEAKAGE_FEATURES = [
    "cancellation_date_after_snapshot",  # (a) 未来信息
    "cancelled_is_label_copy",            # (b) 标签副本
    "snapshot_tenure_current_value",      # (c) 近泄漏
]
REAL_FEATURES = ["age", "plan_tier", "usage_score"]
ALL_FEATURES = REAL_FEATURES + LEAKAGE_FEATURES
# churn_and_rejoin 表示"历史上取消过又回来了"的客户:他们的取消日期早于快照,
# 在 (a) 那一列里同样非空,于是 (a) 不再与标签完全重合,AUC 从 1.0 掉到 0.97 左右。
REJOIN_RATE = 0.06


# ---------------------------------------------------------------------------
# 1. 数据生成:真实信号弱,泄漏字段强
# ---------------------------------------------------------------------------
def sigmoid(z):
    """数值稳定的 sigmoid,避免 math.exp 溢出。"""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def make_dataset(n=N_SAMPLES, seed=SEED):
    """构造流失预测数据集,返回 (X, y)。

    X 的列顺序固定为 ALL_FEATURES;y 为 0/1 流失标签。
    行按快照日期升序(生成顺序即时间顺序),供后面的时序切分使用。

    真实信号来自线性打分:
        logit = -2.6 + 0.020*(age-40) + 0.85*(plan_tier-1) + 0.45*usage_score
    系数刻意调小,让"只用真实特征"的 AUC 落在 0.6-0.7 这个现实区间。
    """
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        age = rng.randint(18, 75)
        plan_tier = rng.choice([0, 1, 2, 3])
        usage_score = rng.uniform(0.0, 1.0)

        logit = (
            -2.6
            + 0.012 * (age - 40)
            + 0.60 * (plan_tier - 1)
            + 0.30 * usage_score
        )
        cancelled = 1 if rng.random() < sigmoid(logit) else 0

        # (a) 未来信息:快照之后的取消日期。
        #     流失样本必有;未流失样本里有一小部分是"取消后又回来了"的,
        #     他们的历史取消日期也落在这一列里 -> 列非空但标签为 0。
        churn_and_rejoin = 1 if (not cancelled and rng.random() < REJOIN_RATE) else 0
        cancellation_date_after = 1 if (cancelled or churn_and_rejoin) else 0

        # (b) 标签副本:换个列名的 y 本身。
        label_copy = cancelled

        # (c) 近泄漏:用了当前 tenure 而非快照时 tenure。
        #     刚流失的客户 tenure 普遍更短,与标签相关,但不是确定关系,
        #     也完全不可用于预测时点。
        if cancelled:
            tenure_now = rng.uniform(0.5, 22.0)
        else:
            tenure_now = rng.uniform(24.0, 55.0)

        rows.append(
            {
                "age": float(age),
                "plan_tier": float(plan_tier),
                "usage_score": round(usage_score, 4),
                "cancellation_date_after_snapshot": float(cancellation_date_after),
                "cancelled_is_label_copy": float(label_copy),
                "snapshot_tenure_current_value": round(tenure_now, 3),
            }
        )

    X = [[r[name] for name in ALL_FEATURES] for r in rows]
    y = [int(r["cancelled_is_label_copy"]) for r in rows]
    return X, y


def column_moments(X):
    """返回每列的 (均值, 标准差),供标准化使用。"""
    n_cols = len(X[0])
    n_rows = len(X)
    means, stds = [], []
    for j in range(n_cols):
        col = [row[j] for row in X]
        mean = sum(col) / n_rows
        var = sum((v - mean) ** 2 for v in col) / n_rows
        means.append(mean)
        stds.append(math.sqrt(var) if var > 1e-12 else 1.0)
    return means, stds


def standardize(X, means, stds):
    """按给定均值/标准差做 z-score。均值标准差来自训练集,不是全量数据。"""
    return [[(row[j] - means[j]) / stds[j] for j in range(len(row))] for row in X]


def split_by_time(X, y, test_fraction=TEST_FRACTION):
    """时序切分:前 test_fraction 比例的行留给测试集。

    这是 ch03 的主题,这里只用最朴素的形式。它已经足以说明一件事:
    泄漏字段在测试集里同样非空,所以测试 AUC 一样漂亮 —— 测试集救不了你。
    """
    split = int(len(X) * (1 - test_fraction))
    return X[:split], y[:split], X[split:], y[split:]


# ---------------------------------------------------------------------------
# 2. 手写逻辑回归:sigmoid + 二元交叉熵 + 批量梯度下降
# ---------------------------------------------------------------------------
def bce_loss(y_true, y_pred):
    """二元交叉熵,对样本取平均。

    概率先 clip 到 [1e-12, 1-1e-12],避免 log(0) 产生 -inf。
    """
    total = 0.0
    for target, p in zip(y_true, y_pred):
        p = min(max(p, 1e-12), 1.0 - 1e-12)
        total += -(target * math.log(p) + (1 - target) * math.log(1 - p))
    return total / len(y_true)


def predict_proba(X, w, b):
    """线性打分 z = w·x + b,再过 sigmoid。"""
    return [sigmoid(sum(xi * wi for xi, wi in zip(row, w)) + b) for row in X]


def train_logistic_regression(X, y, epochs=EPOCHS, lr=LEARNING_RATE):
    """批量梯度下降。返回 (w, b, loss_history)。

    梯度:
        dw_j = (1/m) * Σ (p_i - y_i) * x_ij
        db   = (1/m) * Σ (p_i - y_i)
    没有 L2 正则 —— 这里要的是把"权重被谁主导"暴露得尽可能干净,
    正则化会压低泄漏特征的权重,把证据抹平。
    """
    m = len(X)
    n_features = len(X[0])
    w = [0.0] * n_features
    b = 0.0
    history = []

    for _ in range(epochs):
        preds = predict_proba(X, w, b)
        history.append(bce_loss(y, preds))

        grad_w = [0.0] * n_features
        grad_b = 0.0
        for row, target, p in zip(X, y, preds):
            err = p - target
            for j in range(n_features):
                grad_w[j] += err * row[j]
            grad_b += err

        for j in range(n_features):
            w[j] -= lr * (grad_w[j] / m)
        b -= lr * (grad_b / m)

    return w, b, history


# ---------------------------------------------------------------------------
# 3. 手写 AUC 与 ROC 曲线(秩和公式,不依赖任何库)
# ---------------------------------------------------------------------------
def roc_auc(y_true, y_pred):
    """Mann-Whitney U 形式的 AUC。并列分数按平均秩处理。

        AUC = (Σ rank(positive) - n_pos(n_pos+1)/2) / (n_pos * n_neg)
    """
    pairs = sorted(zip(y_pred, y_true), key=lambda item: item[0])

    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based 平均秩
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    n_pos = sum(1 for target in y_true if target == 1)
    n_neg = len(y_true) - n_pos
    rank_sum_pos = sum(r for r, (_, target) in zip(ranks, pairs) if target == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def roc_curve_points(y_true, y_pred):
    """按预测概率降序扫描,返回 (FPR, TPR) 点列表(含起点 (0,0))。"""
    pairs = sorted(zip(y_pred, y_true), key=lambda item: -item[0])
    n_pos = sum(1 for target in y_true if target == 1)
    n_neg = len(y_true) - n_pos
    tp = fp = 0
    points = [(0.0, 0.0)]
    for _, target in pairs:
        if target == 1:
            tp += 1
        else:
            fp += 1
        points.append((fp / n_neg, tp / n_pos))
    return points


def print_roc_ascii(points, width=45):
    """把 ROC 点画成 ASCII 图,直观展示"贴着左上角"意味着什么。"""
    grid = [[" "] * width for _ in range(11)]
    for fpr, tpr in points:
        col = min(width - 1, int(fpr * (width - 1) + 0.5))
        row = min(10, int(tpr * 10 + 0.5))
        grid[10 - row][col] = "*"
    grid[10][0] = "."
    for idx, line in enumerate(grid):
        tpr_value = 1.0 - idx * 0.1
        label = "%4.1f |" % tpr_value if idx % 2 == 0 else "     |"
        print("%s%s" % (label, "".join(line)))
    print("     +" + "-" * width)
    print("      0.0" + " " * (width - 8) + "1.0   FPR")


def print_weight_table(title, names, w, b):
    """按 |权重| 降序打印权重表 —— 这是整个演示的证据核心。"""
    print(title)
    header = "  %-34s %10s %10s" % ("特征", "权重", "|权重|排名")
    print(header)
    print("  " + "-" * (len(header) - 2))

    order = sorted(range(len(w)), key=lambda j: -abs(w[j]))
    for rank, j in enumerate(order, start=1):
        marker = "  <- 泄漏" if names[j] in LEAKAGE_FEATURES else ""
        print("  %-34s %10.4f %10d%s" % (names[j], w[j], rank, marker))

    leak_w = [w[j] for j, name in enumerate(names) if name in LEAKAGE_FEATURES]
    if leak_w:
        leak_total = sum(abs(v) for v in leak_w)
        real_total = sum(
            abs(w[j]) for j, name in enumerate(names) if name not in LEAKAGE_FEATURES
        )
        ratio = leak_total / real_total if real_total else float("inf")
        print("  %-34s %10.4f" % ("(bias)", b))
        print("  泄漏 |w| 合计 %.4f / 真实 |w| 合计 %.4f = %.1fx"
              % (leak_total, real_total, ratio))


# ---------------------------------------------------------------------------
# 4. 主体:训练两组模型并对比
# ---------------------------------------------------------------------------
def main():
    started = time.time()

    X, y = make_dataset()
    n_pos = sum(y)
    print("=" * 74)
    print("ch02 泄漏演示:同一个数据集,两组特征,两套 AUC")
    print("=" * 74)
    print()
    print("样本 %d  特征 %d  流失率 %.1f%%" % (len(X), len(ALL_FEATURES), 100.0 * n_pos / len(y)))
    print("特征清单: 真实 = %s" % ", ".join(REAL_FEATURES))
    print("          泄漏 = %s" % ", ".join(LEAKAGE_FEATURES))

    # 标准化:只用训练集的均值/标准差。这本身是正确做法 ——
    # ch02 第三节会说明"先切分再 fit"一旦被破坏会怎样。
    X_train_raw, y_train, X_test_raw, y_test = split_by_time(X, y)
    means, stds = column_moments(X_train_raw)
    X_train = standardize(X_train_raw, means, stds)
    X_test = standardize(X_test_raw, means, stds)
    print("切分   时序切分,前 %d 行训练 / 后 %d 行测试"
          % (len(X_train), len(X_test)))
    print("标准化 用训练集均值/标准差做 z-score(权重可比,前提是 fit 只看训练集)")
    print()

    # ---- 模型 A:全部特征(含三列泄漏) ----
    w_all, b_all, history_all = train_logistic_regression(X_train, y_train)
    pred_all = predict_proba(X_test, w_all, b_all)
    auc_all = roc_auc(y_test, pred_all)

    # ---- 模型 B:只留真实特征 ----
    real_idx = [ALL_FEATURES.index(name) for name in REAL_FEATURES]
    X_train_real = [[row[j] for j in real_idx] for row in X_train]
    X_test_real = [[row[j] for j in real_idx] for row in X_test]
    w_real, b_real, history_real = train_logistic_regression(X_train_real, y_train)
    pred_real = predict_proba(X_test_real, w_real, b_real)
    auc_real = roc_auc(y_test, pred_real)

    # ---- 模型 C:真实特征 + 单独隔离出来的近泄漏 tenure ----
    #     这一组是现场里最容易漏掉的情形:表里没有任何一列长得像标签,
    #     权重也不夸张,但它同样不可用于预测时点。
    near_idx = real_idx + [ALL_FEATURES.index("snapshot_tenure_current_value")]
    X_train_near = [[row[j] for j in near_idx] for row in X_train]
    X_test_near = [[row[j] for j in near_idx] for row in X_test]
    w_near, b_near, _ = train_logistic_regression(X_train_near, y_train)
    auc_near = roc_auc(y_test, predict_proba(X_test_near, w_near, b_near))

    print("-" * 74)
    print("第 1 段  训练过程(全特征组):二元交叉熵逐轮下降")
    print("-" * 74)
    print("epoch       BCE 损失")
    for i, loss in enumerate(history_all, start=1):
        if i == 1 or i % 10 == 0 or i == len(history_all):
            print("%5d       %.6f" % (i, loss))
    print("损失从 %.4f 降到 %.4f —— 收敛得很漂亮。"
          % (history_all[0], history_all[-1]))
    print("收敛本身不说明任何问题:损失函数快乐,是因为泄漏特征让任务变得平凡。")
    print()

    print("-" * 74)
    print("第 2 段  AUC 对比")
    print("-" * 74)
    print("模型 A  全特征   (3 个真实 + 3 个泄漏) : AUC = %.4f" % auc_all)
    print("模型 B  真实特征 (age/plan_tier/usage) : AUC = %.4f" % auc_real)
    print("模型 C  真实 + 近泄漏 tenure          : AUC = %.4f" % auc_near)
    print()
    print("A 比 B 高 %+.4f,这个分数没有任何业务含义:它的权重大多建立在"
          % (auc_all - auc_real))
    print("预测时点不存在的字段上。")
    print()
    print("更值得看的是 C。特征表里只有三个正常字段加一个 tenure,")
    print("没有任何一列长得像标签,没有任何一列叫 cancelled。")
    print("AUC 却从 %.4f 一步跳到 %.4f —— 近泄漏就是这样:"
          % (auc_real, auc_near))
    print("它不需要长得像答案,只需要在流失发生之后才被确定。")
    print()
    print("还要注意:测试集里同样带着这些列,所以模型 A 的测试 AUC 也一样漂亮。")
    print("换一份测试集救不了你 —— 那还是同一种泄漏。")
    print()

    print("-" * 74)
    print("第 3 段  学到的权重表:证据核心")
    print("-" * 74)
    print()
    print_weight_table("模型 A —— 全特征(标准化后)", ALL_FEATURES, w_all, b_all)
    print()
    print_weight_table("模型 B —— 只用真实特征(标准化后)", REAL_FEATURES, w_real, b_real)
    print()
    near_names = REAL_FEATURES + ["snapshot_tenure_current_value"]
    print_weight_table("模型 C —— 真实特征 + 近泄漏 tenure", near_names, w_near, b_near)
    print()
    print("读法:标签副本 cancelled_is_label_copy 拿走了模型 A 的最大权重,")
    print("未来信息 cancellation_date_after_snapshot 紧随其后,而三个真实特征")
    print("挤在后面。模型 B 的权重温和得多 —— 那才是这份数据真正的排序能力。")
    print()
    print("再看模型 C:近泄漏 tenure 的权重压过了全部三个真实特征,")
    print("可这一列表格里既没有标签、也没有任何一眼可疑的名字。")
    print("你在 feature importance 面板里看到的往往就是这种东西。")
    print()

    print("-" * 74)
    print("第 4 段  ROC 曲线:两组特征画在同一坐标系里")
    print("-" * 74)
    print()
    print("模型 A(全特征)—— 贴着左上角,形状上无可挑剔:")
    print_roc_ascii(roc_curve_points(y_test, pred_all))
    print()
    print("模型 B(真实特征)—— 明显偏离对角线,但这才是模型真实的排序能力:")
    print_roc_ascii(roc_curve_points(y_test, pred_real))

    print()
    print("=" * 74)
    print("读完这张权重表该记住的一句话")
    print("=" * 74)
    print("AUC %.4f 是泄漏字段在报数,不是模型在预测。" % auc_all)
    print("判断方法只有一个,没有任何工具能替你做:逐字段问 ——")
    print("做预测的那一刻,这个值拿得到吗?拿不到的字段,无论 AUC 多高都要划掉。")
    print("而模型 B 的 %.4f 才是这份数据真正的上限。" % auc_real)
    print()
    print("全部训练与评估耗时 %.2f 秒(3 个逻辑回归,纯 Python,无任何第三方库)。"
          % (time.time() - started))


if __name__ == "__main__":
    sys.exit(main())
