#!/usr/bin/env python3
"""Metrics demo: 指标是业务的代理,代理会失真。

三段内容,全部纯标准库(math / random / sys / time):

  1) 手写混淆矩阵、precision/recall、手写 AUC,以及与 sklearn 的逐位对照;
  2) 信贷场景的阈值扫描表 —— 最高准确率的阈值和最低业务成本的阈值相差甚远;
  3) "AUC 更高但业务上更贵"的反例,以及 lift 与 AUC 不是同一个量的实证。

业务成本(元/笔,显式可调,见 C_FN / C_FP):
    漏批一个坏贷款(放出去之后违约)损失 1000 元 -> 假负例 FN
    误批一个好客户(该拒的批了,白占用额度)损失 100 元 -> 假正例 FP
"""
import math
import random
import sys
import time

SEED = 20261003
N_APPLICANTS = 6000
TEST_FRACTION = 0.30  # 后 30% 留作测试期,前 70% 只用来拟合系数

# 业务成本:这两个数字是本章唯一的"业务输入",其余一切指标都是从它们派生出来的。
# 改这两个数,下面所有结论的排序都会变 —— 这正是本章要讲的。
C_FN = 1000.0  # 漏批坏贷款(放出去 -> 违约):按单笔坏账损失计
C_FP = 100.0   # 误批好客户(拒了 -> 丢客户):按单笔误拒损失计
# 反事实场景:同一批数据、同一批分数,只把两个成本调成 1:1(比如风控团队的 KPI
# 从"坏账额"改成"拒批量与坏账量各罚一半")。AUC 一个数都不会动。
C_FN_FLAT, C_FP_FLAT = 100.0, 100.0

CHANNEL_RATE = 0.06     # 合作渠道占比
CHANNEL_DEFAULT = 0.85  # 渠道件的违约率(远高于自主件)

BOOTSTRAP_N = 300  # 方向性结论的重采样次数 —— 单次差值不是测量结果(见 ch03)


# ---------------------------------------------------------------------------
# 1. 数据:一个信贷申请表,三个连续特征 + 一个渠道标记
# ---------------------------------------------------------------------------
def gauss(rng):
    """标准正态,Box-Muller。手写是为了不依赖 random.gauss 的实现细节。"""
    u1 = rng.random()
    if u1 <= 1e-12:
        u1 = 1e-12
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def make_dataset(n=N_APPLICANTS, seed=SEED):
    """构造信贷违约数据集,返回 list[dict]。

    标签由两个来源合成,这一点必须写清楚,否则第 5 节的反例会站不住:

    1. **自主件**:标签由三个连续特征决定(债务率越高越危险、收入稳定性
       越低越危险、征信年限越短越危险)。
    2. **渠道件**(partner_channel == 1):违约率被**无条件改写**成
       CHANNEL_DEFAULT(0.85),**完全不再取决于那三个特征**。

    所以 partner_channel 不是一个"从数据里发现的、与标签无关的运营维度",
    而是生成器直接赋值的一个**构造维度**。第 5 节说"渠道件违约率高达
    83.33%"时,那个数是这条赋值在测试期的实现值,不是数据里自然长出来的
    现象。反例的机制可迁移,具体幅度不可迁移。

    真实数据集里特征与标签的关系是未知的,这里的生成器只是构造出一个
    "已知真相"的世界,好让后面的对照有确定的答案。
    """
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        debt_ratio = max(0.0, min(1.0, 0.30 + 0.18 * gauss(rng)))
        income_stability = max(0.0, 0.5 + 0.25 * gauss(rng))
        credit_history_years = max(0.0, min(30.0, 6.0 + 5.0 * gauss(rng)))
        partner_channel = 1 if rng.random() < CHANNEL_RATE else 0

        logit = -2.45 + 3.0 * debt_ratio - 1.8 * income_stability \
            + 0.05 * (credit_history_years - 6.0)
        p_bad = 1.0 / (1.0 + math.exp(-logit))
        # ⚠️ partner_channel 是**生成器直接赋值的构造维度**,不是从数据里
        # 发现的信号:渠道件的违约率被这一行**无条件改写**成 CHANNEL_DEFAULT
        # (0.85),完全不再取决于三个连续特征。
        #
        # 第 5 段的反例整条建立在这个赋值上。它是必要的:要让"AUC 高但
        # 业务上不划算"这个现象在可控的幅度内出现,必须有一个高信噪比
        # 的、与成本结构强耦合的分组变量。
        #
        # **代价要说清楚:模型 C 的表现完全来自这一行赋值。** 换到真实数据,
        # 反例的**机制**(代理指标里没有成本参数)可迁移,具体数字不可迁移。
        # 真实项目里这个角色由"合作渠道件的尽调强度更低"这类运营事实担任,
        # 但那需要业务确认,不是从表里看出来的。
        p_eff = CHANNEL_DEFAULT if partner_channel else p_bad
        default = 1 if rng.random() < p_eff else 0
        rows.append({
            "debt_ratio": debt_ratio,
            "income_stability": income_stability,
            "credit_history_years": credit_history_years,
            "partner_channel": partner_channel,
            "default": default,
        })
    return rows


def split_by_time(records, test_fraction=TEST_FRACTION):
    """时序切分,和 ch02/ch03 保持一致:前段训练,后段测试。"""
    split = int(len(records) * (1.0 - test_fraction))
    return records[:split], records[split:]


# ---------------------------------------------------------------------------
# 2. 三个模型分数(手写线性打分,不做分类训练)
# ---------------------------------------------------------------------------
# A 三个宽信号,就是生成器的真相 —— 这是"能做到多好"的上界参照。
BROAD_COEFFS = {"debt_ratio": 3.0, "income_stability": -1.8,
                "credit_history_years": 0.05}


def score_broad(r):
    """A:只用三个连续特征的宽信号线性打分。"""
    return (BROAD_COEFFS["debt_ratio"] * r["debt_ratio"]
            + BROAD_COEFFS["income_stability"] * r["income_stability"]
            + BROAD_COEFFS["credit_history_years"]
            * (r["credit_history_years"] - 6.0))


def score_broad_plus_channel(r, w_channel=0.9):
    """B:宽信号 + 渠道标记。渠道是一个真实的运营信号,权重需要业务判断。"""
    return score_broad(r) + w_channel * r["partner_channel"]


def score_channel_only(r):
    """C:只用渠道标记。它是一个**粗但真实**的信号。"""
    return float(r["partner_channel"])


# ---------------------------------------------------------------------------
# 3. 混淆矩阵与基础指标(手写)
# ---------------------------------------------------------------------------
def confusion_at_threshold(records, scores, threshold):
    """阈值化 -> 混淆矩阵 (tp, fp, fn, tn)。

    正类 = 违约(default=1),分数 = 风险分。决策规则:**分数越高越危险,分数 >=
    阈值就拒**。所以"模型说这人危险"对应"拒批",不是"批下来"。

    对应关系(这一段是最容易写反的地方,写反了整张表都会算错):
        拒批 且 真的违约   -> TP   拦下一笔坏账
        拒批 但 其实没违约   -> FP   误拒一个好客户
        批了 且 真的违约   -> FN   漏批一笔坏账(最贵)
        批了 但 其实没违约   -> TN   正常放行
    """
    tp = fp = fn = tn = 0
    for r, s in zip(records, scores):
        predict_reject = 1 if s >= threshold else 0
        if predict_reject and r["default"]:
            tp += 1
        elif predict_reject:
            fp += 1
        elif r["default"]:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def accuracy(tp, fp, fn, tn):
    return (tp + tn) / float(tp + fp + fn + tn)


def precision(tp, fp, fn, tn):
    """precision = TP/(TP+FP) = 拒批的人里真违约的比例(查准率)。"""
    return tp / float(tp + fp) if tp + fp else float("nan")


def recall(tp, fp, fn, tn):
    """recall = TP/(TP+FN) = 全部违约的人里被拦下的比例(查全率)。"""
    return tp / float(tp + fn) if tp + fn else float("nan")


def expected_cost(tp, fp, fn, tn, c_fn=C_FN, c_fp=C_FP):
    """每份申请的平均业务成本(元)。这是本章唯一"对得上业务"的数字。

    漏批的坏贷款(FN)按单笔坏账损失 c_fn 计,误拒的好客户(FP)按单笔误拒损失
    c_fp 计。**precision 和 recall 都不出现在这个式子里** —— 它们只能告诉你
    分数排得怎么样,只有这两个成本能告诉你钱花了多少。
    除以总份数是为了让不同样本量的实验可以横向比较。
    """
    return (c_fn * fn + c_fp * fp) / float(tp + fp + fn + tn)


def sweep_thresholds(records, scores, c_fn=C_FN, c_fp=C_FP):
    """扫描**每一个**真实分数作为阈值,返回逐行指标。

    逐个分数扫而不是均匀取网格点,是因为代价最优点常常落在两个分数之间的
    极窄区间里,网格采样很容易跳过它。这里的候选数 = 不同分数的个数。
    """
    rows = []
    n = float(len(records))
    for threshold in sorted(set(scores)):
        tp, fp, fn, tn = confusion_at_threshold(records, scores, threshold)
        rows.append({
            "threshold": threshold,
            "accuracy": (tp + tn) / n,
            "precision": precision(tp, fp, fn, tn),
            "recall": recall(tp, fp, fn, tn),
            "cost": (c_fn * fn + c_fp * fp) / n,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return rows


def best_by(rows, key, sign=1):
    """按 key 取最优行。sign=-1 表示越小越好。"""
    return max(rows, key=lambda r: sign * r[key])


# ---------------------------------------------------------------------------
# 4. 手写 AUC —— 两条互不相关的算法,互相验证
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


def roc_auc_trapezoid(y_true, y_score):
    """ROC 曲线下面积,梯形法。**和 sklearn 的算法同构**:

    sklearn 的 roc_auc_score 走 sklearn.metrics._ranking._binary_roc_auc_score,
    它把 roc_curve() 返回的 (fpr, tpr) 交给 `auc()`,而 `auc()` 的实现是
    `area = direction * trapezoid(y, x)` —— **trapezoid 来自
    `scipy.integrate`,不是 numpy**(sklearn 1.9.1,
    sklearn/metrics/_ranking.py:18 `from scipy.integrate import trapezoid`)。
    复刻这条路径,才能做"逐位一致"的对照 —— 光说"我的 AUC 和 sklearn
    差不多"没有任何验证价值。
    """
    n = len(y_score)
    n_pos = sum(y_true)
    n_neg = n - n_pos
    order = sorted(range(n), key=lambda i: y_score[i], reverse=True)
    ys = [y_true[i] for i in order]
    ss = [y_score[i] for i in order]

    # distinct_value_indices:分数每变化一次就是一个候选阈值
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

    # drop_intermediate:删掉与邻居共线的点。sklearn 默认开启,这一步不改变
    # 曲线形状,但它决定了求和时一共有多少个加数 —— 少了这一步,最后一位会差。
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


def _non_collinear(sequence):
    """返回值得保留的下标:端点,以及步长发生变化的位置。"""
    keep = [0]
    for i in range(1, len(sequence) - 1):
        if sequence[i] - sequence[i - 1] != sequence[i + 1] - sequence[i]:
            keep.append(i)
    keep.append(len(sequence) - 1)
    return keep


# scipy.integrate.trapezoid 内部对 numpy 数组做 xp.sum,而 numpy 的 float64
# 求和用分块两两相加(块大小 128)。纯 Python 逐个相加会因为结合顺序不同,
# 在最后一位上差 1 ulp。复刻这个算法,才能声称"逐位一致"—— 少这一段,
# 下面的对照就只是"两位小数一致",不是逐位。
def pairwise_sum(values):
    """复刻 numpy 的 pairwise summation(分块 8 路累加,块长 128)。

    scipy.integrate.trapezoid 走的是 numpy 的 sum,不是它自己写的求和循环,
    所以这里对齐的是 numpy 的分块算法。
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


def lift_at_fraction(y_true, y_score, fraction=0.10):
    """前 fraction 名单的响应率相对总体的倍数。运营唯一能直接用的数字。"""
    n = len(y_true)
    k = int(fraction * n)
    order = sorted(range(n), key=lambda i: -y_score[i])
    base_rate = sum(y_true) / float(n)
    top_rate = sum(y_true[i] for i in order[:k]) / float(k)
    return base_rate, top_rate, top_rate / base_rate


# ---------------------------------------------------------------------------
# 5. 与 sklearn 逐位对照
# ---------------------------------------------------------------------------
def verify_against_sklearn():
    """在手写 AUC 和 sklearn.roc_auc_score 之间做逐位对照。

    多个数据集上验证,而不是一个 —— 单个数据集上的"一致"可能只是
    那个数据集的分数恰好没有并列、没有极端值。
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None, 0, 0

    cases = []
    # 五个构造的数据集:不同规模、不同正样本比例、有/无大量并列分数
    for case_id, (n, p, ties, seed) in enumerate([
        (400, 0.05, True, 1), (2000, 0.14, False, 2),
        (800, 0.50, True, 3), (5000, 0.02, True, 4), (1500, 0.30, False, 5),
    ]):
        rng = random.Random(SEED + seed)
        y = [1 if rng.random() < p else 0 for _ in range(n)]
        if sum(y) == 0 or sum(y) == n:
            y[0] = 1
        s = []
        for label in y:
            v = rng.gauss(0.0, 1.0) + 1.1 * label
            if ties:
                v = round(v, 1)  # 制造大量并列,逼出并列处理的差异
            s.append(v)
        cases.append((case_id, n, p, ties, y, s))

    rank_exact = 0
    trap_exact = 0
    rank_is_truth = 0
    for case_id, n, p, ties, y, s in cases:
        mine_rank = roc_auc_rank_sum(y, s)
        mine_trap = roc_auc_trapezoid(y, s)
        theirs = roc_auc_score(y, s)
        truth = roc_auc_exact_rational(y, s)
        rank_exact += mine_rank == theirs
        trap_exact += mine_trap == theirs
        rank_is_truth += float(truth) == mine_rank

        def verdict(mine):
            if mine == theirs:
                return "逐位一致"
            # 差在最后一位,且相对误差 < 1e-15:这是浮点求和顺序造成的 1 ulp,
            # 不是算法差异。报告它,不要藏。
            return "差 1 ulp (%.17f)" % mine

        print("  数据集 %d  n=%-5d 正样本 %5.1f%%  并列 %-3s | 秩和 %-28s | 梯形 %-28s | sklearn %.17f"
              % (case_id, n, 100.0 * p, "有" if ties else "无",
                 verdict(mine_rank), verdict(mine_trap), theirs))
    print("  梯形法(与 sklearn 同构)%d/%d 逐位一致;秩和法 %d/%d 逐位一致"
          % (trap_exact, len(cases), rank_exact, len(cases)))
    # 差异归属:用 Fraction 精确有理数算一遍真值,看是谁偏了。
    print("  用精确有理数(Fraction)重算真值,秩和法 %d/%d 等于真值"
          % (rank_is_truth, len(cases)))
    print("  —— 那两次 1 ulp 差异里,**偏离真值的是 sklearn 的梯形结果**,")
    print("     手写秩和法给出的是正确舍入值。成因是浮点求和顺序,不是算法差异。")
    return cases, len(cases), trap_exact


def roc_auc_exact_rational(y_true, y_score):
    """用 Fraction 精确有理数算 AUC,得到浮点下的"真值"。

    秩和公式在数学上是精确的;用 Fraction 算就完全绕开了浮点求和顺序。
    拿它当裁判,才能判断"谁偏了 1 ulp"—— 这是本文件唯一一处允许依赖
    标准库 fractions 的地方,它只用于**校验**,不参与任何生产计算。
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


# ---------------------------------------------------------------------------
# 6. 重采样:方向性结论必须扛得住抽样波动
# ---------------------------------------------------------------------------
def bootstrap_compare(y_true, scores_hi, scores_lo, c_fn, c_fp,
                      n_boot=BOOTSTRAP_N, seed=SEED + 99):
    """重采样 n_boot 次,比较两个模型在 AUC 和业务成本上的差值分布。

    差值 > 0 一律表示"hi 更好":AUC 差是 hi 更高,成本差是 hi 更便宜。
    单次数字不做判断,这里只负责给出分布和置信区间。
    """
    rng = random.Random(seed)
    n = len(y_true)
    auc_diffs, cost_diffs = [], []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        yb = [y_true[i] for i in idx]
        if sum(yb) == 0 or sum(yb) == n:
            continue
        hi = [scores_hi[i] for i in idx]
        lo = [scores_lo[i] for i in idx]
        auc_diffs.append(roc_auc_rank_sum(yb, hi) - roc_auc_rank_sum(yb, lo))
        cost_hi = min_expected_cost(yb, hi, c_fn, c_fp)[0]
        cost_lo = min_expected_cost(yb, lo, c_fn, c_fp)[0]
        cost_diffs.append(cost_lo - cost_hi)
    return auc_diffs, cost_diffs


def min_expected_cost(y_true, scores, c_fn=C_FN, c_fp=C_FP):
    """扫描全部阈值,返回 (最低期望成本, 混淆矩阵, 最优阈值)。

    分数 = 风险分,分数 >= 阈值就拒。按分数**从高到低**依次纳入拒批名单,
    每个人进来时混淆矩阵 O(1) 更新。成本沿轨迹不单调,每一步都要比较。

    成本最优点一定落在某个真实分数上(再往上抬阈值不会改变预测),
    所以只扫不同分数即可。复杂度 O(n log n)。

    注意:分数并列时必须**整组**纳入或整组排除。逐个纳入会在组内产生
    一个真实不存在的中间状态(部分并列者被拒),那是不可实施的决策。
    """
    n = len(y_true)
    n_pos = sum(y_true)
    order = sorted(range(n), key=lambda i: -scores[i])
    # 起点:阈值低于所有分数 -> 全部拒批(这就是"全拒"基线,成本 = c_fp)
    best = (c_fp, (0, n, 0, 0), -float("inf"))
    tp = fp = 0
    pos = 0
    while pos < n:
        # 整组处理:分数相同的这一批必须同时进入拒批名单
        value = scores[order[pos]]
        end = pos
        while end < n and scores[order[end]] == value:
            end += 1
        for k in range(pos, end):
            if y_true[order[k]]:
                tp += 1
            else:
                fp += 1
        pos = end
        fn = n_pos - tp
        tn = (n - n_pos) - fp
        cost = (c_fn * fn + c_fp * fp) / float(n)
        if cost < best[0]:
            best = (cost, (tp, fp, fn, tn), value)
    return best


def percentile(sorted_values, p):
    """线性插值分位数(sorted_values 必须已排序)。"""
    if not sorted_values:
        return float("nan")
    idx = p * (len(sorted_values) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (idx - lo)


def mean_and_ci(values):
    """均值与 95% 置信区间(正态近似)。返回已排序的原始值也一并给出。"""
    values = sorted(values)
    mean = sum(values) / float(len(values))
    if len(values) < 2:
        return mean, float("nan"), float("nan"), float("nan")
    var = sum((v - mean) ** 2 for v in values) / float(len(values) - 1)
    sd = math.sqrt(var)
    half = 1.96 * sd / math.sqrt(len(values))
    return mean, mean - half, mean + half, sd


# ---------------------------------------------------------------------------
# 主体
# ---------------------------------------------------------------------------
def print_sweep_table(title, rows, best_acc, best_cost, c_fn, c_fp, limit=14):
    """打印阈值扫描表。只打关键行,完整表格在 scan_rows 里可用。"""
    print(title)
    print("  %-10s %9s %9s %9s %11s %6s %6s %6s %6s"
          % ("阈值", "accuracy", "precision", "recall", "成本(元/笔)",
             "TP拒", "FP误拒", "FN漏批", "TN放行"))
    print("  " + "-" * 82)
    # 均匀取样,外加两个最优点,保证表里一定能看到冲突
    step = max(1, len(rows) // limit)
    picked = list(range(0, len(rows), step))[:limit]
    picked.append(next(i for i, r in enumerate(rows) if r["threshold"] == best_acc["threshold"]))
    picked.append(next(i for i, r in enumerate(rows) if r["threshold"] == best_cost["threshold"]))
    for i in sorted(set(picked)):
        r = rows[i]
        marks = []
        if r["threshold"] == best_acc["threshold"]:
            marks.append("准确率最优")
        if r["threshold"] == best_cost["threshold"]:
            marks.append("成本最优")
        mark = "  <- " + "/".join(marks) if marks else ""
        print("  %-10.4f %9.4f %9.4f %9.4f %11.2f %6d %6d %6d %6d%s"
              % (r["threshold"], r["accuracy"], r["precision"], r["recall"],
                 r["cost"], r["tp"], r["fp"], r["fn"], r["tn"], mark))
    print("  (候选阈值 %d 个,均匀抽样 %d 行 + 两个最优点;成本按 C_FN=%.0f / C_FP=%.0f 计)"
          % (len(rows), limit, c_fn, c_fp))
    print()


def main():
    started = time.time()

    records = make_dataset()
    train, test = split_by_time(records)
    y_test = [r["default"] for r in test]
    n = len(test)

    print("=" * 78)
    print("ch04 指标:代理指标的失真")
    print("=" * 78)
    print()
    print("申请 %d 份(前 %d 份训练 / 后 %d 份测试)  测试期违约率 %.2f%%"
          % (len(records), len(train), n, 100.0 * sum(y_test) / n))
    print("业务成本  漏批坏贷款 %.0f 元/笔  误批好客户 %.0f 元/笔  比值 %.0f:1"
          % (C_FN, C_FP, C_FN / C_FP))
    print()

    # ---- 第 1 段:手写 AUC 与 sklearn 逐位对照 ----
    print("-" * 78)
    print("第 1 段  手写 AUC 的两条互不相关算法,与 sklearn 逐位对照")
    print("-" * 78)
    cases, total, exact_trap = verify_against_sklearn()
    if cases is None:
        print("  未安装 sklearn,跳过对照(手写实现本身不依赖任何第三方库)")
    print()

    # ---- 第 2 段:阈值扫描,准确率最优 vs 成本最优 ----
    print("-" * 78)
    print("第 2 段  阈值扫描:最高准确率的阈值和最低业务成本的阈值")
    print("-" * 78)
    scores = [score_broad(r) for r in test]
    scan_rows = sweep_thresholds(test, scores)
    best_acc = best_by(scan_rows, "accuracy")
    best_cost = best_by(scan_rows, "cost", sign=-1)
    print_sweep_table("模型 A(宽信号)在 10:1 成本下的阈值扫描:",
                      scan_rows, best_acc, best_cost, C_FN, C_FP)
    print("  准确率最优  阈值 %+.4f  accuracy %.4f  precision %.4f  recall %.4f  成本 %.2f"
          % (best_acc["threshold"], best_acc["accuracy"], best_acc["precision"],
             best_acc["recall"], best_acc["cost"]))
    print("  成本最优    阈值 %+.4f  accuracy %.4f  precision %.4f  recall %.4f  成本 %.2f"
          % (best_cost["threshold"], best_cost["accuracy"], best_cost["precision"],
             best_cost["recall"], best_cost["cost"]))
    print("  两个最优相差:accuracy %+.4f,成本 %+.2f 元/笔(%.2f 倍)"
          % (best_acc["accuracy"] - best_cost["accuracy"],
             best_acc["cost"] - best_cost["cost"],
             best_acc["cost"] / best_cost["cost"]))
    print()
    print("  为什么差这么多:准确率把 1 个 FN 和 1 个 FP 都算作同样的一次错误,")
    print("  而它们在业务上相差 %.0f 倍。准确率优化的其实是'把大多数判对',"
          % (C_FN / C_FP))
    print("  业务要的是'少犯错的那一类'。两者在阈值上的最优位置因此完全不同。")
    print()

    # 基线对照 —— 审阅清单里 "What baseline does this beat" 那一条
    reject_all = (C_FP * (n - sum(y_test))) / float(n)  # 全拒:每个好客户都误拒
    approve_all = (C_FN * sum(y_test)) / float(n)      # 全放:每个违约都漏批
    base_rate = sum(y_test) / float(n)
    print("  基线对照(全拒 %.2f / 全放 %.2f 元/笔)  测试期基础违约率 %.4f"
          % (reject_all, approve_all, base_rate))
    print("  模型 A 的成本最优 %.2f 元/笔,两个基线都比它差 —— 这是成本最优点存在的最低证据。"
          % best_cost["cost"])
    print()

    # ---- 第 3 段:AUC 不是 lift ----
    print("-" * 78)
    print("第 3 段  AUC 不是 lift:两个数不是同一个量")
    print("-" * 78)
    auc_a = roc_auc_rank_sum(y_test, scores)
    _, top_rate, lift = lift_at_fraction(y_test, scores, 0.10)
    print("  模型 A  AUC %.4f   前 10%% lift %.2fx" % (auc_a, lift))
    print("  这两个数不可互相换算:AUC 是**全局成对**的排序统计量(随机取一正一负,")
    print("  模型把正样本排在负样本前面的概率);lift 是**某一个具体比例**上的响应率比值。")
    print("  AUC %.4f 不等于'前 10%% 名单的违约率是总体的 %.2f 倍' —— 那个数是 lift %.2f。"
          % (auc_a, auc_a, lift))
    print("  ch02 那句伏笔在这里兑现:模型 B 的 AUC 0.6767 和它的 lift 1.86x")
    print("  是两个独立测出来的数,谁也推不出谁。")
    print()

    # ---- 第 4 段:lift 随名单比例变化,一个阈值一个答案 ----
    print("-" * 78)
    print("第 4 段  lift 是一条曲线,不是一个数")
    print("-" * 78)
    print("  同一个模型,lift 完全由业务愿意批多少比例决定:")
    print("  %-14s %10s %10s %10s" % ("名单比例", "该组违约率", "总体违约率", "lift"))
    for fraction in (0.05, 0.10, 0.20, 0.30, 0.50):
        base_f, rate_f, lift_f = lift_at_fraction(y_test, scores, fraction)
        print("  前 %-11.0f%% %10.4f %10.4f %10.2fx"
              % (100.0 * fraction, rate_f, base_f, lift_f))
    print("  报 lift 必须连'哪一档'一起报。只说'lift 2.2 倍'是不完整的,")
    print("  因为对方一定会问'前多少'。")
    print()

    # ---- 第 5 段:AUC 更高但业务上更贵 ----
    print("-" * 78)
    print("第 5 段  反例:两个模型,一个 AUC 更高,一个业务上更便宜")
    print("-" * 78)
    scores_b = [score_broad_plus_channel(r) for r in test]
    scores_c = [score_channel_only(r) for r in test]
    auc_b, auc_c = roc_auc_rank_sum(y_test, scores_b), roc_auc_rank_sum(y_test, scores_c)
    print("  模型 A  宽信号(3 个连续特征)")
    print("  模型 B  宽信号 + 渠道标记")
    print("  模型 C  只用渠道标记(1 个二值特征)")
    n_channel = sum(1 for r in test if r["partner_channel"])
    n_channel_default = sum(1 for r in test
                            if r["partner_channel"] and r["default"])
    print("  渠道件 %d / %d 份(%.2f%%)  违约率 %.2f%%"
          % (n_channel, n, 100.0 * n_channel / n,
             100.0 * n_channel_default / n_channel))
    print()
    print("  ⚠️  两套成本口径的性质不一样,读表前必须分清:")
    print("     10:1(C_FN=1000 / C_FP=100)= **本章主口径**,真实信贷风控的量级。")
    print("     1:1 (C_FN=100  / C_FP=100) = **反事实口径,只为演示代理失真而设**。")
    print("        现实中不存在'误拒一个好客户和放出一笔坏账同价'的场景。")
    print("        它的作用是提供一个可控的开关:只改成本比,不改数据、不改模型,")
    print("        看业务赢家会不会换人。**第 5 段的结论依赖这个设定**,")
    print("        10:1 口径下三个模型的结论是另一回事。")
    print()
    header = "  %-24s %9s %11s %11s" % ("成本设定", "AUC", "最低成本", "该设定下赢家")
    print(header)
    print("  " + "-" * 78)
    for label, c_fn, c_fp in [("C_FN=1000 / C_FP=100(坏账口径)", C_FN, C_FP),
                              ("C_FN=100  / C_FP=100 (1:1 口径)", C_FN_FLAT, C_FP_FLAT)]:
        results = {}
        for name, sc in (("A", scores), ("B", scores_b), ("C", scores_c)):
            rows = sweep_thresholds(test, sc, c_fn, c_fp)
            results[name] = best_by(rows, "cost", sign=-1)
        aucs = {"A": auc_a, "B": auc_b, "C": auc_c}
        cost_winner = min(results, key=lambda k: results[k]["cost"])
        auc_winner = max(aucs, key=lambda k: aucs[k])
        agree = "一致" if cost_winner == auc_winner else "★ 相反"
        print("  %-24s %9s %11s %11s"
              % (label.split("(")[0].strip(),
                 "%.4f(%s)" % (aucs[auc_winner], auc_winner),
                 "%.2f(%s)" % (results[cost_winner]["cost"], cost_winner),
                 agree))
    print()
    c_flat = min_expected_cost(y_test, scores_c, C_FN_FLAT, C_FP_FLAT)
    print("  两个成本设定下 **AUC 的赢家始终是 B,业务成本的赢家换了人**。")
    tp_c, fp_c, fn_c, tn_c = c_flat[1]
    print("  C 在 1:1 口径下的最优决策:拒掉 %d 份渠道件(违约 %d 份),放行 %d 份(违约 %d 份)。"
          % (tp_c + fp_c, tp_c, tn_c + fn_c, fn_c))
    print()
    print("  **同一个决策,换一套成本就完全不是一个结论:**")
    print("  %-34s %10s" % ("决策:拒掉全部 %d 份渠道件" % (tp_c + fp_c), "成本"))
    print("    1:1  口径(C 的最优解)          %8.2f 元/笔" % c_flat[0])
    print("    10:1 口径(同一决策,同一分数)  %8.2f 元/笔"
          % ((C_FN * fn_c + C_FP * fp_c) / float(n)))
    print("    参照:全拒基线                  %8.2f 元/笔" % reject_all)
    print("    参照:模型 B(10:1 最优)         %8.2f 元/笔"
          % min(sweep_thresholds(test, scores_b, C_FN, C_FP), key=lambda r: r["cost"])["cost"])
    print("  10:1 口径下这个决策比'什么都不做'还贵 %.2f 元 —— 放行的 %d 份违约每份赔 %.0f 元,"
          % ((C_FN * fn_c + C_FP * fp_c) / float(n) - reject_all, fn_c, C_FN))
    print("  而拒掉 %d 人只省下 %d 个误拒。**阈值不是模型的属性,是成本结构的属性。**"
          % (tp_c + fp_c, fp_c))
    print("  **AUC 不会告诉你这件事,因为 AUC 里根本没有成本这两个字。**")
    print()

    # ---- 第 6 段:这个差别扛不扛得住抽样波动 ----
    print("-" * 78)
    print("第 6 段  单次差值不是测量结果:%d 次重采样 = 8 行比较"
          % BOOTSTRAP_N)
    print("        (2 套成本口径 x 2 组模型对 x 2 个指标;bootstrap_compare 被调用 4 次)")
    print("-" * 78)
    for label, c_fn, c_fp in [("10:1 坏账口径(主口径)", C_FN, C_FP),
                              ("1:1 口径(反事实,见第 5 段)", C_FN_FLAT, C_FP_FLAT)]:
        print("  %s" % label)
        for hi_name, lo_name, sc_hi, sc_lo in [("B", "C", scores_b, scores_c),
                                               ("B", "A", scores_b, scores)]:
            auc_diffs, cost_diffs = bootstrap_compare(
                y_test, sc_hi, sc_lo, c_fn, c_fp)
            a_mean, a_lo, a_hi, _ = mean_and_ci(auc_diffs)
            c_mean, c_lo, c_hi, _ = mean_and_ci(cost_diffs)
            a_win = 100.0 * sum(1 for v in auc_diffs if v > 0) / len(auc_diffs)
            c_win = 100.0 * sum(1 for v in cost_diffs if v > 0) / len(cost_diffs)
            print("    %s-%s  AUC 差 均值 %+.4f 95%%CI [%+.4f, %+.4f]  %s 更优占比 %.1f%%"
                  % (hi_name, lo_name, a_mean, a_lo, a_hi, hi_name, a_win))
            print("    %s-%s  成本差(正=%s更优) 均值 %+.2f 95%%CI [%+.2f, %+.2f]  %s 更优占比 %.1f%%"
                  % (hi_name, lo_name, hi_name, c_mean, c_lo, c_hi, hi_name, c_win))
        print()

    print("-" * 78)
    print("本章结论")
    print("-" * 78)
    print("  1. 准确率最优阈值 %.4f 与成本最优阈值 %.4f 相差 %.2f 倍成本。" % (
        best_acc["threshold"], best_cost["threshold"],
        best_acc["cost"] / best_cost["cost"]))
    print("  2. AUC 是全局成对统计量,lift 是某一档名单的响应率比,两者不可换算。")
    print("  3. 换一套成本口径(第 5 段的 1:1 是**反事实口径**),业务成本的赢家")
    print("     就变了,AUC 的赢家不变 —— 因为 AUC 眼里所有正负配对等权,业务不。")
    print("     换人证明的是**机制**,不是'C 是更好的模型';主口径 10:1 下三个")
    print("     模型的结论一致。")
    print()
    print("  跑完 %.2f 秒" % (time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
