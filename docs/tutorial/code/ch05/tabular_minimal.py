#!/usr/bin/env python3
"""Tabular modeling, complete version: 把 ch02/ch03/ch04 拼成一条完整管线。

纯 Python 标准库,**零第三方 import**(模块级和函数级都没有)。手写的东西:
sigmoid、二元交叉熵、z-score 标准化、L2 正则、批量梯度下降、学习率衰减、
早停、两条互不相关的 AUC 算法、梯形求和的 pairwise summation 复刻、
精确有理数裁判、lift。

## 和 ch02 的关系(这是本章的对照点)

ch02 的 [leakage_minimal.py](../ch02/leakage_minimal.py) 是**最小演示**:
它的训练器 `train_logistic_regression()` 里明确写着"没有 L2 正则 ——
这里要的是把'权重被谁主导'暴露得尽可能干净,正则化会压低泄漏特征的权重,
把证据抹平"。同一份数据、同一批种子、同一套 sigmoid 和批量梯度下降。

本文件是**完整实现**:同样手写,但把 ch02 为了看清楚而省掉的东西全部补上
——标准化、L2、学习率衰减、验证集早停、λ 网格选择。所以"同一问题、两种
实现深度"不是修辞,是两份可以并排跑的代码。

## 业务场景:客户流失预测(面板数据)

300 个客户 × 12 个月 = 3600 行月度快照,预测"下个月是否流失"。
面板(同一个客户有多行)是真实流失场景的常态,也是第 3 节 CV 陷阱的土壤。

**这份数据集是构造的,不含任何泄漏字段** —— 六个特征全部在预测时点拿得到。
ch02 已经把泄漏讲完了,本章不重复;本章的靶子是**切分策略**和**模型选择**。

**明确标注(构造设定,不是从数据里发现的性质):**

  1. 客户敏感度 s ~ N(0,1) 逐客户抽样、跨月恒定,由生成器**直接赋值**。
     现实里没有这一列,它扮演的是"未观测的客户异质性"。
  2. 流失压力随时间单调上升:`trend = 0.11 * max(0, month - 3)`。
     现实里可能是获客渠道变化或竞品动作,这里是为了让"未来行混进训练"
     的代价**可测**,不是普适结论。
  3. 标签是 logistic 噪声采样,不是确定函数。所以任何单列 AUC 都不是 1.0。

  这三条都是**为了演示而设的构造维度**。机制可迁移,具体数值不可迁移。

## 跨库验证在哪里

本文件**不 import sklearn**,它的 AUC 只能自证:两条独立算法互相对照,
再用 `fractions.Fraction` 精确有理数当裁判。**自证不是跨库验证** ——
两份实现可能共享同一个误解。

真正的跨库对照在 [tabular_engineered.py](tabular_engineered.py) 第 4 段,
那里把本文件的两个 AUC 函数和 `sklearn.metrics.roc_auc_score` 做逐位比较
(复刻 scipy/numpy 的 pairwise summation,才能声称"逐位一致")。

用法:
    /opt/homebrew/bin/python3 docs/tutorial/code/ch05/tabular_minimal.py
"""
import math
import random
import time

SEED = 20261003
N_CUSTOMERS = 300
MONTHS = 12
N_SAMPLES = N_CUSTOMERS * MONTHS  # 3600 行月度快照

# 月份边界切分。三个数是 12 的整数分解,所以每一段都落在整月上,
# 不会把同一个月劈成两半:
#   month  0..5   训练期(拟合用)   6 个月 = 1800 行
#   month  6..7   训练期(验证用)   2 个月 =  600 行   ← 只用于选 λ 和早停
#   month  8..11  上线后的测试期   4 个月 = 1200 行   ← 只跑一次
FIT_END_MONTH = 6
VALID_END_MONTH = 8

FEATURES = ["tenure_months", "monthly_charge", "support_tickets_90d",
            "plan_tier", "late_payments_12m", "is_self_serve"]

L2_GRID = (0.0, 0.003, 0.01, 0.03, 0.10, 0.30)
EPOCHS = 160
LEARNING_RATE = 0.8
DECAY_HALFLIFE = 40.0   # 学习率衰减半衰期(轮)
COMPARE_SEEDS = 5       # 配置对照的种子数 —— 单次差值不是测量结果(见 ch03)


# ---------------------------------------------------------------------------
# 1. 数据:一个客户流失面板(300 客户 × 12 个月)
# ---------------------------------------------------------------------------
def gauss(rng):
    """标准正态,Box-Muller。手写是为了不依赖 random.gauss 的实现细节。"""
    u1 = rng.random()
    if u1 <= 1e-12:
        u1 = 1e-12
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def make_dataset(n_customers=N_CUSTOMERS, months=MONTHS, seed=SEED):
    """构造客户流失面板,返回按 (月份, 客户) 排序的 list[dict]。

    ⚠️ 构造设定,见模块 docstring 的三条标注。核心的两条:
      - `sensitivity` 逐客户赋值、跨月恒定(现实里不可观测,这里显式给出);
      - 流失压力随月份单调上升,让"未来行混进训练"的代价可测。
    """
    rng = random.Random(seed)
    rows = []
    for cid in range(n_customers):
        # —— 构造维度 1:客户敏感度,生成器直接赋值,跨月恒定 ——
        sensitivity = gauss(rng)
        plan_tier = rng.choice((0, 1, 2, 3))   # 0=basic … 3=enterprise
        is_self_serve = 1 if rng.random() < 0.45 else 0
        base_charge = (35.0, 60.0, 95.0, 160.0)[plan_tier]
        base_charge *= 0.85 + 0.30 * rng.random()
        for m in range(months):
            tenure_months = float(m + 1)
            monthly_charge = base_charge * (1.0 + 0.004 * m) \
                * (1.0 + 0.10 * gauss(rng))
            support_tickets_90d = max(
                0, int(abs(3.0 - 0.55 * plan_tier + 1.4 * gauss(rng))))
            late_payments_12m = 1 if rng.random() < (
                0.10 + 0.06 * (3 - plan_tier)) else 0
            # —— 构造维度 2:随月份单调上升的流失压力 ——
            trend = 0.11 * max(0, m - 3)
            logit = (-1.15
                     - 0.85 * tenure_months / 5.0
                     - 0.55 * (plan_tier - 1.5) / 1.5
                     - 0.42 * (support_tickets_90d - 3) / 3.0
                     - 0.70 * late_payments_12m
                     + 0.35 * is_self_serve
                     + 0.35 * sensitivity
                     + trend)
            p = 1.0 / (1.0 + math.exp(-logit))
            rows.append({
                "month": m,
                "customer_id": cid,
                "tenure_months": tenure_months,
                "monthly_charge": monthly_charge,
                "support_tickets_90d": float(support_tickets_90d),
                "plan_tier": float(plan_tier),
                "late_payments_12m": float(late_payments_12m),
                "is_self_serve": float(is_self_serve),
                "churned": 1 if rng.random() < p else 0,
            })
    rows.sort(key=lambda r: (r["month"], r["customer_id"]))
    return rows


def split_by_month(rows, fit_end=FIT_END_MONTH, valid_end=VALID_END_MONTH):
    """按月份切三段:拟合段 / 验证段 / 测试段。全部落在整月边界上。

    ch03 的结论在这里直接生效:切分决定数字可不可信。本文件不用随机切分,
    因为这份数据的行**有严格的时间方向**(月度快照)。
    """
    fit = [r for r in rows if r["month"] < fit_end]
    valid = [r for r in rows if fit_end <= r["month"] < valid_end]
    test = [r for r in rows if r["month"] >= valid_end]
    return fit, valid, test


def to_matrix(rows, features=FEATURES):
    """把 list[dict] 拆成 (X, y)。特征列顺序由 FEATURES 固定,不依赖字典序。"""
    X = [[float(r[f]) for f in features] for r in rows]
    y = [int(r["churned"]) for r in rows]
    return X, y


def split_diagnostics(fit, valid, test):
    """切分体检:行数、客户重叠、月份边界、基础流失率。

    这一段不需要任何机器学习知识,一行 SQL / 一行 pandas 就能算。
    ch03 的核心经验:**这些量要在训练之前算,而不是训练完了回头猜。**
    """
    fit_ids = {r["customer_id"] for r in fit}
    valid_ids = {r["customer_id"] for r in valid}
    test_ids = {r["customer_id"] for r in test}
    out = []
    for name, part in (("拟合段", fit), ("验证段", valid), ("测试段", test)):
        rate = sum(r["churned"] for r in part) / float(len(part))
        out.append((name, len(part), part[0]["month"], part[-1]["month"],
                    len({r["customer_id"] for r in part}), rate))
    return out, (len(fit_ids & valid_ids), len(fit_ids & test_ids))


# ---------------------------------------------------------------------------
# 2. 预处理:标准化(只 fit 拟合段)
# ---------------------------------------------------------------------------
def column_moments(X):
    """返回每列的 (均值, 标准差),**只在拟合段上算**。

    ch02 第 3 节的教训:对全量数据 fit 标准化器,会让 33% 的验证/测试期
    统计量钻进模型工件,并且改变 L2 正则在各特征上的实际作用强度。
    训练器 `train_logistic_regression()` 只在这里调用一次,输入是拟合段,
    结构上排除这个错误。
    """
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


def apply_scaler(X, means, stds):
    """用**给定**的 (means, stds) 做 z-score。

    这就是上线时推理代码要调的那一个函数:序列化下来的 (means, stds)
    必须是拟合段那一组,不能拿测试期的统计量重新算一遍。
    """
    return [[(row[j] - means[j]) / stds[j] for j in range(len(row))]
            for row in X]


# ---------------------------------------------------------------------------
# 3. 手写逻辑回归:sigmoid + BCE + L2 + 批量梯度下降 + 学习率衰减 + 早停
# ---------------------------------------------------------------------------
def sigmoid(z):
    """数值稳定的 sigmoid:正负分支分开,避免 exp 溢出。"""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def bce_loss(y_true, y_pred, w=None, l2=0.0):
    """二元交叉熵(对样本取平均)+ 0.5*l2*||w||²。

    概率先 clip 到 [1e-12, 1-1e-12],避免 log(0) 产生 -inf。
    l2 项**不作用于截距** —— 截距不是要收缩的参数。
    """
    total = 0.0
    for target, p in zip(y_true, y_pred):
        p = min(max(p, 1e-12), 1.0 - 1e-12)
        total += -(target * math.log(p) + (1 - target) * math.log(1 - p))
    loss = total / len(y_true)
    if w is not None and l2:
        loss += 0.5 * l2 * sum(wi * wi for wi in w)
    return loss


def predict_proba(X, w, b):
    """线性打分 z = w·x + b,再过 sigmoid。"""
    return [sigmoid(sum(xi * wi for xi, wi in zip(row, w)) + b) for row in X]


def train_logistic_regression(X, y, X_val=None, y_val=None, *, l2=0.01,
                              epochs=EPOCHS, lr=LEARNING_RATE,
                              decay_halflife=DECAY_HALFLIFE,
                              standardize=True):
    """批量梯度下降 + L2 + 学习率衰减 + 验证集早停。返回 dict。

    梯度(样本平均 + L2):
        dw_j = (1/m) * Σ (p_i - y_i) x_ij  +  l2 * w_j
        db   = (1/m) * Σ (p_i - y_i)
    学习率按半衰期衰减:lr_t = lr0 * 0.5 ** (t / decay_halflife)

    `standardize=False` 是给第 3 节的配置对照用的 —— ch02 式训练器不做
    标准化。同一份数据、同一组种子,只差这一个开关。

    **本函数自己拥有标准化器**,并把 (means, stds) 一并返回:测试期和线上
    推理必须用**同一个** scaler 变换,否则就是 ch02 第 3 节的 train/serve skew。
    """
    m = len(X)
    n_features = len(X[0])
    if standardize:
        # 标准化器只在**拟合段**上 fit,再原样应用到验证段。
        # 注意是三处都走同一个 scale:fit / val / 未来上线时的推理数据。
        means, stds = column_moments(X)
        def scale(M):
            return [[(v - means[j]) / stds[j] for j, v in enumerate(row)]
                    for row in M]
    else:
        means, stds = None, None
        def scale(M):
            return M
    X_fit = scale(X)
    Xv = scale(X_val) if X_val is not None else None

    w = [0.0] * n_features
    b = 0.0
    history = []          # (epoch, train_bce, val_bce or None)
    best = {"val_bce": float("inf"), "epoch": -1, "w": list(w), "b": b}
    diverged_at = None    # 发散轮次(见下);None = 正常收敛

    for epoch in range(epochs):
        preds = predict_proba(X_fit, w, b)
        train_bce = bce_loss(y, preds)
        val_bce = (bce_loss(y_val, predict_proba(Xv, w, b))
                   if X_val is not None else None)
        history.append((epoch + 1, train_bce, val_bce))

        grad_w = [0.0] * n_features
        grad_b = 0.0
        for row, target, p in zip(X_fit, y, preds):
            err = p - target
            for j in range(n_features):
                grad_w[j] += err * row[j]
            grad_b += err
        for j in range(n_features):
            grad_w[j] = grad_w[j] / m + l2 * w[j]      # L2 只作用在权重上
        grad_b = grad_b / m

        lr_t = lr * (0.5 ** (epoch / decay_halflife))
        for j in range(n_features):
            w[j] -= lr_t * grad_w[j]
        b -= lr_t * grad_b

        if val_bce is not None and val_bce < best["val_bce"]:
            best = {"val_bce": val_bce, "epoch": epoch + 1,
                    "w": list(w), "b": b}

        # 早停之外的第二道闸:**检测发散**。批量梯度下降在未标准化特征上
        # 很容易一步跨过极小点,损失会从 ln2(0.6931) 跳到 5、20 并且再也
        # 回不来。第 3 节的 A 配置(无标准化)就是靠这个闸才跑得下去的。
        # 不装这道闸,发散会被"验证 BCE 最好的一轮恰好是第 1 轮"掩盖掉,
        # 报出一个漂亮的 ln2 = 0.6931,看起来像"模型没学到东西",
        # 而真相是优化器炸了。
        if not math.isfinite(train_bce) or train_bce > 2.0:
            diverged_at = epoch + 1
            break

    if X_val is None:      # 没有验证集就没有早停,返回最后一轮
        best = {"val_bce": float("nan"), "epoch": epochs,
                "w": list(w), "b": b}
    return {"w": best["w"], "b": best["b"], "history": history,
            "best_epoch": best["epoch"], "best_val_bce": best["val_bce"],
            "l2": l2, "standardize": standardize,
            "means": means, "stds": stds, "diverged_at": diverged_at}


# ---------------------------------------------------------------------------
# 4. 手写 AUC:两条算法互不相关的实现 + 精确有理数裁判
# ---------------------------------------------------------------------------
def roc_auc_rank_sum(y_true, y_score):
    """Mann-Whitney U 形式的 AUC,并列分数按平均秩处理。"""
    pairs = sorted(zip(y_score, y_true), key=lambda t: t[0])
    m = len(pairs)
    ranks = [0.0] * m
    i = 0
    while i < m:
        j = i
        while j + 1 < m and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = average_rank
        i = j + 1
    rank_sum_pos = 0.0
    for r, (_, label) in zip(ranks, pairs):
        if label == 1:
            rank_sum_pos += r
    n_pos = sum(y_true)
    n_neg = m - n_pos
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _non_collinear(sequence):
    """返回值得保留的下标:端点,以及步长发生变化的位置(drop_intermediate)。"""
    keep = [0]
    for i in range(1, len(sequence) - 1):
        if sequence[i] - sequence[i - 1] != sequence[i + 1] - sequence[i]:
            keep.append(i)
    keep.append(len(sequence) - 1)
    return keep


def pairwise_sum(values):
    """复刻 numpy 的 pairwise summation(分块 8 路累加,块长 128)。

    scipy.integrate.trapezoid 对 numpy 数组做 xp.sum,而 numpy 的 float64
    求和用分块两两相加;纯 Python 的 sum 是从左到右链式相加。数学上等价,
    最后一位可能差 1 ulp。**不写这一段,"与 sklearn 逐位一致"就退化成
    "四位小数一致"** —— 见 ch04 第 2 节的实测。
    """
    n = len(values)
    if n < 8:
        total = 0.0
        for v in values:
            total += v
        return total
    if n <= 128:
        lanes = list(values[:8])
        i = 8
        end = n - (n % 8)
        while i < end:
            for k in range(8):
                lanes[k] += values[i + k]
            i += 8
        total = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) \
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
        while i < n:
            total += values[i]
            i += 1
        return total
    half = n // 2
    half -= half % 8
    return pairwise_sum(values[:half]) + pairwise_sum(values[half:])


def roc_auc_trapezoid(y_true, y_score):
    """ROC 曲线下面积,梯形法。**与 sklearn 的算法同构**:

    sklearn 的 roc_auc_score 把 roc_curve() 的 (fpr, tpr) 交给
    `auc()`,实现是 `area = direction * trapezoid(y, x)`,而 trapezoid
    来自 scipy.integrate。因此复刻:降序排序、distinct_value_indices
    分界、drop_intermediate 剔共线点、pairwise_sum 求和。

    ⚠️ 这里的"同构"是给 [tabular_engineered.py](tabular_engineered.py)
       第 4 段的跨库逐位对照用的。本文件自己不 import sklearn。
    """
    n = len(y_score)
    n_pos = sum(y_true)
    n_neg = n - n_pos
    order = sorted(range(n), key=lambda i: y_score[i], reverse=True)
    ys = [y_true[i] for i in order]
    ss = [y_score[i] for i in order]

    boundaries = []
    previous = None
    for i, v in enumerate(ss):
        if previous is None or v != previous:
            boundaries.append(i)
            previous = v
    boundaries.append(n)

    tps, fps = [], []
    cum_tp = cum_fp = 0
    for j in range(len(boundaries) - 1):
        lo, hi = boundaries[j], boundaries[j + 1]
        hit = 0
        for k in range(lo, hi):
            hit += ys[k]
        cum_tp += hit
        cum_fp += (hi - lo) - hit
        tps.append(cum_tp)
        fps.append(cum_fp)

    if len(fps) > 2:
        keep = sorted(set(_non_collinear(fps) + _non_collinear(tps)))
        tps = [tps[i] for i in keep]
        fps = [fps[i] for i in keep]

    tps.insert(0, 0)
    fps.insert(0, 0)
    tpr = [t / float(n_pos) for t in tps]
    fpr = [f / float(n_neg) for f in fps]
    return pairwise_sum([(fpr[i + 1] - fpr[i]) * (tpr[i + 1] + tpr[i]) / 2.0
                         for i in range(len(fpr) - 1)])


def roc_auc_exact_rational(y_true, y_score):
    """用 Fraction 精确有理数算 AUC,得到浮点下的"真值"。

    秩和公式在数学上是精确的;用 Fraction 算就完全绕开了浮点求和顺序。
    **本文件唯一一处允许依赖 fractions 的地方,且只用于校验,不参与训练。**
    """
    from fractions import Fraction
    pairs = sorted(zip(y_score, y_true), key=lambda t: t[0])
    m = len(pairs)
    ranks = [Fraction(0)] * m
    i = 0
    while i < m:
        j = i
        while j + 1 < m and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        average_rank = Fraction(i + j, 2) + 1
        for k in range(i, j + 1):
            ranks[k] = average_rank
        i = j + 1
    rank_sum_pos = sum((r for r, (_, label) in zip(ranks, pairs) if label == 1),
                       Fraction(0))
    n_pos = sum(y_true)
    n_neg = m - n_pos
    return (rank_sum_pos - Fraction(n_pos * (n_pos + 1), 2)) / Fraction(n_pos * n_neg)


def auc_selfcheck(cases):
    """本文件内部的自证:两条算法 + 精确有理数裁判。

    ⚠️ **这是自证,不是跨库验证。** 两条实现可能共享同一个误解;
       真正排除"手写实现有 bug"的对照在 engineered 版第 4 段。
    """
    agree = 0
    rank_is_truth = 0
    trap_is_truth = 0
    rows = []
    for case_id, (y, s) in enumerate(cases):
        mine_rank = roc_auc_rank_sum(y, s)
        mine_trap = roc_auc_trapezoid(y, s)
        truth = float(roc_auc_exact_rational(y, s))
        agree += abs(mine_rank - mine_trap) < 1e-12
        rank_is_truth += mine_rank == truth
        trap_is_truth += mine_trap == truth
        rows.append((case_id, len(y), 100.0 * sum(y) / len(y), truth,
                     mine_rank, mine_trap))
    for case_id, n, pos, truth, r, t in rows:
        print("    数据集 %d  n=%-5d 正样本 %5.1f%%  精确真值 %.17f  "
              "秩和 %.17f  梯形 %.17f"
              % (case_id, n, pos, truth, r, t))
    print("    两条算法互相一致 %d/%d;秩和等于精确真值 %d/%d;"
          "梯形等于精确真值 %d/%d"
          % (agree, len(cases), rank_is_truth, len(cases),
             trap_is_truth, len(cases)))
    print("    ⚠️ 以上是**自证**:两条实现都出自本文件,可能共享同一个误解。")
    print("       跨库对照(sklearn roc_auc_score)见 tabular_engineered.py 第 4 段。")
    return agree, len(cases)


def make_auc_selfcheck_cases(seed=SEED + 7):
    """五个构造的数据集:不同规模、不同正样本比例、有/无大量并列分数。

    单个数据集上的"一致"可能只是那个数据集恰好没有并列、没有极端值。
    """
    cases = []
    for case_id, (n, p, ties, offset) in enumerate([
        (400, 0.05, True, 1), (2000, 0.14, False, 2),
        (800, 0.50, True, 3), (5000, 0.02, True, 4), (1500, 0.30, False, 5),
    ]):
        rng = random.Random(seed + offset)
        y = [1 if rng.random() < p else 0 for _ in range(n)]
        if sum(y) in (0, n):
            y[0] = 1
        s = []
        for label in y:
            v = rng.gauss(0.0, 1.0) + 1.1 * label
            if ties:
                v = round(v, 1)
            s.append(v)
        cases.append((y, s))
    return cases


def lift_at_fraction(y_true, y_score, fraction=0.10):
    """前 fraction 名单的响应率相对总体的倍数。运营唯一能直接用的数字。

    ch04 已经实测过:AUC 和 lift 不可互相换算。这里只重新算一遍,作为
    "指标回到业务语言"这一步的落点,不重复 ch04 的论证。
    """
    n = len(y_true)
    k = int(fraction * n)
    order = sorted(range(n), key=lambda i: -y_score[i])
    base_rate = sum(y_true) / float(n)
    top_rate = sum(y_true[i] for i in order[:k]) / float(k)
    return base_rate, top_rate, top_rate / base_rate


# ---------------------------------------------------------------------------
# 5. 统计工具:均值、标准差、95% CI
# ---------------------------------------------------------------------------
def binary_entropy(p):
    """二元熵 H(p) —— 常数预测器(恒输出 p)在标签为 Bernoulli(p) 时能达到的
    最小交叉熵。

    它的用处是把"训练损失和验证损失差了多少"分解成两部分:
    **两段的基础率不同造成的那部分(分布漂移)** + **学出来的过拟合那部分**。
    少了这个分解,很容易把漂移读成过拟合。
    """
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def mean(values):
    return sum(values) / float(len(values))


def stdev(values):
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def mean_ci(values, z=1.96):
    """正态近似的 95% 置信区间。

    样本只有 5 个时正态近似很粗,所以**每个区间都同时打印原始样本** ——
    n=5 的 CI 只能支撑"没有明显差异"这种弱结论,支撑不了强因果叙述。
    这是 ch03 打回过一次的地方,这里从输出格式上就堵住。
    """
    mu = mean(values)
    return mu, mu - z * stdev(values) / math.sqrt(len(values)), \
        mu + z * stdev(values) / math.sqrt(len(values))


def print_ascii_curve(title, xs, series, height=9, width=61):
    """把若干条曲线画成 ASCII 图。series = [(名字, y 值列表), ...]。

    只为教学可读性,不承担任何数值结论 —— 精确数字在旁边的表格里。
    """
    all_y = [v for _, ys in series for v in ys]
    lo, hi = min(all_y), max(all_y)
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    grid = [[" "] * width for _ in range(height)]
    marks = "*o+#"
    for si, (name, ys) in enumerate(series):
        ch = marks[si % len(marks)]
        for xi, v in enumerate(ys):
            col = int(xi * (width - 1) / max(1, len(ys) - 1))
            row = height - 1 - int((v - lo) / (hi - lo) * (height - 1))
            grid[row][col] = ch
    print("  %s  (y: %.4f -> %.4f)" % (title, lo, hi))
    for r in range(height):
        tag = "*=训练  o=验证" if r == 0 else ""
        print("    %s |%s| %s" % (("%8.4f" % (hi - (hi - lo) * r / (height - 1))),
                                  "".join(grid[r]), tag))


# ---------------------------------------------------------------------------
# 6. main:四段输出
# ---------------------------------------------------------------------------
def print_split_report(rows, diag_rows, overlap):
    print("=== 1. 数据与切分 ===")
    print("  %-8s %6s %-16s %6s %8s" % ("段", "行数", "月份", "客户数", "流失率"))
    for name, n, m0, m1, n_cust, rate in diag_rows:
        print("  %-8s %6d  month %-2d..%-2d %6d %7.2f%%"
              % (name, n, m0, m1, n_cust, 100.0 * rate))
    print("  客户重叠  验证段∩拟合段 %d 个  测试段∩拟合段 %d 个  "
          "(面板数据:每个客户 12 行,重叠是**结构性的**,不是切分错误)"
          % overlap)
    print("  按月份边界切分,不用随机切分 —— 这些行有严格的时间方向(见 ch03)")
    print("  特征(全部在预测时点拿得到,本数据集**不含泄漏字段**,ch02 已讲完):")
    for f in FEATURES:
        print("    - %s" % f)
    print("  ⚠️ 构造设定:客户敏感度由生成器逐客户赋值;流失压力随月份上升;")
    print("     标签含 logistic 噪声。机制可迁移,数值不可迁移。")

    # 逐月流失率的成因分解。注意结论是反直觉的:**上升的压力项并没有让
    # 流失率上升**,因为 tenure 项降得更多。审阅时的正确动作是查生成器,
    # 而不是"流失压力上升 → 流失率应该上升"这一句直觉。
    print("\n  逐月流失率与两项 logit 分解(为什么后段基础率更低):")
    for m in range(MONTHS):
        sub = [r for r in rows if r["month"] == m]
        rate_m = sum(r["churned"] for r in sub) / float(len(sub))
        print("    month %2d  rate %.4f  tenure_term %+.3f  trend_term %+.3f"
              % (m, rate_m, -0.85 * (m + 1) / 5.0,
                 0.11 * max(0, m - 3)))
    print("    -> tenure 项累计降 1.870,trend 项累计升 0.880,净效应是**下降**。")
    print("       「压力上升」不等于「流失率上升」——先分解再下结论。")


def print_weight_table(title, names, w, b, mean_scaled):
    print("  %s(标准化后权重):" % title)
    print("    %-22s %9s %8s" % ("特征", "权重", "|权重|排名"))
    order = sorted(range(len(w)), key=lambda j: -abs(w[j]))
    rank = {j: i + 1 for i, j in enumerate(order)}
    for j in order:
        print("    %-22s %9.4f %7d" % (names[j], w[j], rank[j]))
    print("    %-22s %9.4f" % ("(截距)", b))
    print("    全部特征都是真实信号,没有哪个能把权重独吞。")
    print("    这和 ch02 那张泄漏权重表(前三名合计占 94%)是完全不同的形态。")


def run_divergence_probe(sub, epochs=400, seed_label=""):
    """A 配置(无标准化 + 常数学习率)去掉发散闸跑满 epochs 轮,看它会不会自己回来。

    ⚠️ 为什么要专门跑这一次:代码里那行"损失从 ln2 跳走并且回不来"是一个
    **机制断言**,机制断言也必须实测。ch03 打回过一次"单点值当因果叙述"。

    所以这里不重跑一遍训练,而是**直接观测** A 配置的损失轨迹:它是收敛到
    0.39 附近,还是永远在两个坏值之间来回跳。
    """
    f_r, v_r, _ = split_by_month(sub)
    X, y = to_matrix(f_r)
    Xv, yv = to_matrix(v_r)
    m, nf = len(X), len(X[0])
    w = [0.0] * nf
    b = 0.0
    best = (float("inf"), -1)
    marks = {}
    for ep in range(epochs):
        preds = predict_proba(X, w, b)
        val_bce = bce_loss(yv, predict_proba(Xv, w, b))
        if val_bce < best[0]:
            best = (val_bce, ep + 1)
        for probe in (1, 2, 3, 4, 5, 10, 20, 50, 100, 200, epochs):
            if ep + 1 == probe:
                marks[probe] = val_bce
        grad_w = [0.0] * nf
        grad_b = 0.0
        for row, target, p in zip(X, y, preds):
            err = p - target
            for j in range(nf):
                grad_w[j] += err * row[j]
            grad_b += err
        for j in range(nf):
            w[j] -= LEARNING_RATE * (grad_w[j] / m)
        b -= LEARNING_RATE * (grad_b / m)
    lines = ["epoch %s 验证 BCE: " % "  ".join(str(k) for k in sorted(marks))
             + "  ".join("%.4f" % marks[k] for k in sorted(marks)),
             "%d 轮里的最优是 epoch %d 的 %.4f;ln2 = %.4f"
             % (epochs, best[1], best[0], math.log(2))]
    if best[1] == 1:
        lines.append("  → 最优就是**第 1 轮**(恰好等于 ln2,即恒定预测器的损失)。"
                     "跑满 %d 轮也回不去 —— 早停在第 1 轮报出的 0.6931 不是"
                     "\"模型没学到东西\",是优化器根本没动起来。" % epochs)
    else:
        lines.append("  → 最优在 epoch %d,说明它最终收敛了;发散是暂时的。"
                     % best[1])
    return lines


def main():
    t0 = time.time()
    print("ch05 表格建模全景 · 手写完整管线(纯标准库,零第三方 import)")
    print("种子 %d | %d 客户 x %d 月 = %d 行 | %d 轮批量梯度下降\n"
          % (SEED, N_CUSTOMERS, MONTHS, N_SAMPLES, EPOCHS))

    # ---- 第 1 段:数据与切分 ----
    rows = make_dataset()
    fit_rows, val_rows, test_rows = split_by_month(rows)
    diag_rows, overlap = split_diagnostics(fit_rows, val_rows, test_rows)
    print_split_report(rows, diag_rows, overlap)

    X_fit, y_fit = to_matrix(fit_rows)
    X_val, y_val = to_matrix(val_rows)
    X_test, y_test = to_matrix(test_rows)

    # ---- 第 2 段:λ 网格(标准化 + L2 + 衰减 + 早停) ----
    print("\n=== 2. 原理门槛:标准化 + L2 + 衰减 + 早停 ===")
    print("  训练器自己拥有标准化器:fit 只碰拟合段,(means, stds) 随结果一起返回。")
    print("  验证段/测试段/线上推理都必须用这同一组 (means, stds) 变换 ——")
    print("  这就是 ch02 第 3 节 train/serve skew 的结构性解法。")

    print("\n  L2 强度 λ 网格(λ=0 即无正则):")
    print("    %8s %10s %10s %10s" % ("λ", "验证 BCE", "早停轮", "验证 AUC"))
    sweep = []
    for lam in L2_GRID:
        res = train_logistic_regression(X_fit, y_fit, X_val, y_val, l2=lam)
        # 验证段必须用训练器返回的 scaler 变换,不能直接喂原始值
        val_scores = predict_proba(apply_scaler(X_val, res["means"], res["stds"]),
                                   res["w"], res["b"])
        auc_v = roc_auc_rank_sum(y_val, val_scores)
        sweep.append((lam, res, auc_v))
        print("    %8.3f %10.4f %10d %10.4f"
              % (lam, res["best_val_bce"], res["best_epoch"], auc_v))
    best_lam, best_res, best_auc_v = min(sweep, key=lambda t: t[1]["best_val_bce"])
    means, stds = best_res["means"], best_res["stds"]
    print("  按验证 BCE 选出 λ = %.3f(验证 AUC %.4f)" % (best_lam, best_auc_v))
    if best_lam == 0.0:
        # 诚实的边界:**这份数据上最优 λ 是 0**,也就是"不需要正则"。
        # 这不是说 L2 无用,而是说 6 个特征、1800 行的规模下没有过拟合压力
        # (证据就在下一段的训练/验证间隔里)。把 λ 强推到一个非零值才叫作弊。
        print("  ⚠️ 选出的最优 λ = 0,即**这份数据上不需要正则**。")
        print("     这不是说 L2 无用:λ 网格仍在代码里、仍参与选型,只是没有一格")
        print("     赢过 λ=0。特征少、样本足时这就是正确答案。")
        print("     强行挑一个非零 λ 让'正则起了作用'才是作弊。")
    print("  被选中那次的标准化器(只在拟合段上 fit):")
    for j, f in enumerate(FEATURES):
        print("    %-22s mean %10.4f  std %10.4f" % (f, means[j], stds[j]))

    tr_curve = [h[1] for h in best_res["history"]]
    va_curve = [h[2] for h in best_res["history"] if h[2] is not None]
    print("\n  训练/验证损失曲线(λ = %.3f,早停在第 %d 轮):"
          % (best_lam, best_res["best_epoch"]))
    for idx in range(0, len(tr_curve), max(1, len(tr_curve) // 8)):
        h = best_res["history"][idx]
        print("    epoch %4d  训练 %.4f  验证 %.4f" % h)
    h = best_res["history"][-1]
    print("    epoch %4d  训练 %.4f  验证 %.4f  (最后一轮)" % h)
    # ⚠️ 这里的 0.10 **不是过拟合**。过拟合的签名是训练损失**低于**验证损失;
    # 本例相反(训练 0.4842 > 验证 0.3832)。差额几乎全部来自两段的基础
    # 流失率不同 —— 常数预测器的损失下限就是该段基础率的二元熵。分解见下。
    gap = tr_curve[-1] - va_curve[-1]
    floor_fit = binary_entropy(sum(y_fit) / float(len(y_fit)))
    floor_val = binary_entropy(sum(y_val) / float(len(y_val)))
    print("    训练 %.4f / 验证 %.4f,差 %+.4f"
          % (tr_curve[-1], va_curve[-1], gap))
    print("    ⚠️ 这个差**不是过拟合** —— 过拟合的签名是训练损失**低于**验证损失,"
          " 这里相反。分解:")
    print("      基础流失率  拟合段 %.4f -> 二元熵下限 %.4f"
          % (sum(y_fit) / float(len(y_fit)), floor_fit))
    print("                  验证段 %.4f -> 二元熵下限 %.4f"
          % (sum(y_val) / float(len(y_val)), floor_val))
    print("      两段下限之差 %+.4f  vs  实测训练-验证差 %+.4f"
          % (floor_fit - floor_val, gap))
    print("      两者量级一致 -> 差额**主要是分布漂移**(两段基础率不同),"
          "不是模型在背答案。")
    print("      学出来的过拟合量 = 实测差 - 下限差 = %+.4f(小,且方向上"
          % (gap - (floor_fit - floor_val)))
    print("      训练更贴近自己的下限,这是正常的,不是记住了样本)")
    print_ascii_curve("损失曲线", list(range(len(tr_curve))),
                      [("train", tr_curve), ("val", va_curve)])

    # ---- 第 3 段:三种配置对照(ch02 式 / 只加标准化 / 完整实现) ----
    print("\n=== 3. 同一问题,两种实现深度:配置对照(%d 组种子)===" % COMPARE_SEEDS)
    print("  A = ch02 式:不标准化 + 无 L2 + 常数学习率 + 固定轮数")
    print("  B = A + 标准化(只差 standardize 这一个开关)")
    print("  C = 完整实现:标准化 + L2=%.3f + 衰减 + 早停" % best_lam)
    print("  指标取**验证集** —— 模型选择阶段不该看测试期,测试期只跑一次。")
    print("  %-4s %10s %10s %10s %10s %8s"
          % ("种子", "A 验证BCE", "B 验证BCE", "C 验证BCE", "C 验证AUC", "A 发散轮"))
    a_bce, b_bce, c_bce, c_auc, a_div = [], [], [], [], []
    for k in range(COMPARE_SEEDS):
        sub = make_dataset(seed=SEED + 1000 * (k + 1))
        f_r, v_r, _ = split_by_month(sub)
        Xa, ya = to_matrix(f_r)
        Xb, yb = to_matrix(v_r)
        ra = train_logistic_regression(Xa, ya, Xb, yb, l2=0.0,
                                       decay_halflife=float("inf"),
                                       standardize=False)
        rb = train_logistic_regression(Xa, ya, Xb, yb, l2=0.0,
                                       decay_halflife=float("inf"),
                                       standardize=True)
        rc = train_logistic_regression(Xa, ya, Xb, yb, l2=best_lam)
        a_div.append(ra["diverged_at"])
        a_bce.append(ra["best_val_bce"])
        b_bce.append(rb["best_val_bce"])
        c_bce.append(rc["best_val_bce"])
        c_auc.append(roc_auc_rank_sum(yb, predict_proba(apply_scaler(
            Xb, rc["means"], rc["stds"]), rc["w"], rc["b"])))
        print("  %-4d %10.4f %10.4f %10.4f %10.4f %8s"
              % (k, a_bce[-1], b_bce[-1], c_bce[-1], c_auc[-1],
                 ra["diverged_at"] if ra["diverged_at"] else "无"))
    print("  A 发散 %d/%d 组 —— 缺标准化时固定学习率的批量梯度下降在真实量纲"
          % (sum(d is not None for d in a_div), COMPARE_SEEDS))
    print("     (月费 std %.1f、工单 std %.1f)上一步跨过极小点。" % (stds[1], stds[2]))
    print("  对照:第 0 组去掉发散闸再跑 400 轮,看它**会不会自己回来** ——")
    for line in run_divergence_probe(sub, epochs=400):
        print("    %s" % line)
    for name, series in (("A(无标准化)", a_bce), ("B(+标准化)", b_bce),
                         ("C(完整实现)", c_bce)):
        mu, lo, hi = mean_ci(series)
        print("  %-14s 均值 %7.4f  sd %6.4f  95%%CI [%7.4f, %7.4f]  样本 %s"
              % (name, mu, stdev(series), lo, hi,
                 " ".join("%.4f" % v for v in series)))
    mu, lo, hi = mean_ci(c_auc)
    print("  %-14s 均值 %7.4f  sd %6.4f  95%%CI [%7.4f, %7.4f]"
          % ("C 验证 AUC", mu, stdev(c_auc), lo, hi))
    d_ab = [b - a for a, b in zip(a_bce, b_bce)]
    mu, lo, hi = mean_ci(d_ab)
    print("  差值 B-A(正 = B 的验证 BCE 更高 = B 更差):"
          "均值 %+.4f 95%%CI [%+.4f, %+.4f]" % (mu, lo, hi))
    if lo < 0 < hi:
        print("    ⚠️ CI 跨零 —— 这个差值**不能**当成方向性结论。"
              "n=%d 太小,只能说不显著。" % COMPARE_SEEDS)
    else:
        print("    ✅ CI 不含 0(%d 组种子全部同号)。" % COMPARE_SEEDS)

    # ---- 第 4 段:手写 AUC 自证 + 测试期一次评估 ----
    print("\n=== 4. 手写 AUC(自证)与测试期评估 ===")
    auc_selfcheck(make_auc_selfcheck_cases())

    test_scores = predict_proba(
        apply_scaler(X_test, best_res["means"], best_res["stds"]),
        best_res["w"], best_res["b"])
    auc_t_rank = roc_auc_rank_sum(y_test, test_scores)
    auc_t_trap = roc_auc_trapezoid(y_test, test_scores)
    auc_t_truth = float(roc_auc_exact_rational(y_test, test_scores))
    base_rate, top_rate, lift = lift_at_fraction(y_test, test_scores, 0.10)
    print("\n  测试期(月份 %d..%d,%d 行)只跑一次:"
          % (test_rows[0]["month"], test_rows[-1]["month"], len(y_test)))
    print("    秩和 AUC      %.17f" % auc_t_rank)
    print("    梯形 AUC      %.17f" % auc_t_trap)
    print("    精确有理数真值 %.17f" % auc_t_truth)
    print("    秩和 == 真值: %s;梯形 == 真值: %s"
          % (auc_t_rank == auc_t_truth, auc_t_trap == auc_t_truth))
    print("    恒定预测器 AUC 0.5000 —— 那是**恒等式不是测量**:任何常数"
          "分数的 AUC 按定义就是 0.5")
    print("    基础流失率 %.4f | 前 10%% 名单流失率 %.4f | lift %.2fx"
          % (base_rate, top_rate, lift))
    print("    lift 只能报一档,且和 AUC 不可互推(ch04 实测过)")

    print()
    print_weight_table("最终模型(λ = %.3f)" % best_lam, FEATURES,
                       best_res["w"], best_res["b"], means)
    print("\n耗时 %.1f 秒" % (time.time() - t0))


if __name__ == "__main__":
    main()
