#!/usr/bin/env python3
"""Split demo: 同一份数据,三种切分,三个完全不同的结论。

纯 Python 标准库实现。无 sklearn / numpy / pandas;固定随机种子,输出可复现。

ch02 讲的是"字段在预测时点拿不到"。本章讲的是另一半:**字段都合规,
但切分边界画错了**,测试集于是替模型泄了底。泄漏的孪生兄弟。

数据集:某工厂的设备故障预测(每天一行,标签 = 当天是否故障)。
  60 台设备 / 4 座工厂 / 每台 20 天 = 1200 行。
  - 设备是 group:同一台设备的多条记录来自同一台机器,共享它的个体故障倾向。
  - 工厂是 group 的 group:同一工厂的设备共享环境应力(ambient),
    这是"设备之间存在共同因子"的来源,也是 group 切分能暴露出真实
    分布偏移的原因。
  - 四座工厂**按交付波次交错进场**(每 6 天各出一台)。如果按工厂整块
    编号,工厂就和时间轴完全共线,时序切分测到的会是"我换了个工厂",
    而不是"时间往前走了"。
  - 记录按 (设备交付日 + 天) 全局有序,time_split 才有意义。

两个被评估的估计器,故意选成两种极端,用来暴露"切分影响多大":
  A 设备先验:测试行属于哪台设备,就用那台设备在训练集里的故障率去
    预测;该设备没在训练集出现过就退回总体故障率。**这是可靠性工程
    里第一个会被做出来的模型**,它完全靠 group 结构吃饭。
  B 特征逻辑回归:手写 sigmoid + 批量梯度下降,只用预测时点拿得到的
    特征(运行时长 / 振动 / 环境温度)。**它记不住实体**,所以它不该
    被 group 泄漏影响。

对照这两种估计器,能看出一件比"随机切分更好"重要得多的事:
切分的影响**取决于模型能不能记住实体**,而这件事你在切分之前
通常不知道。实测:换到按设备留出,设备先验从 0.78 塌到 0.50,而逻辑
回归从 0.80 走到 0.79 几乎没动。**所以"随机切分一定让指标虚高"是
错的** —— 虚高的只有会记实体的那种模型,也就是最容易被业务接受的那种。

四行切分对照(三个函数,group_split 按粒度跑两次):
  random_split          打乱行
  time_split            前 80% 的日子训练,后 20% 测试
  group_split(device)   随机留出 25% 的设备
  group_split(factory)  整座工厂留作测试

跑完约 1 秒。教学代码以可读性优先,不做任何性能优化。

用法:
    /opt/homebrew/bin/python3 docs/tutorial/code/ch03/splits_minimal.py
"""
import math
import random
import sys
import time

SEED = 20261003
N_FACTORIES = 4
DEVICES_PER_FACTORY = 15
N_DEVICES = N_FACTORIES * DEVICES_PER_FACTORY
DAYS_PER_DEVICE = 20
N_ROWS = N_DEVICES * DAYS_PER_DEVICE

TEST_FRACTION = 0.2
EPOCHS = 200
LEARNING_RATE = 0.5

# 四个工厂的环境应力,固定阶梯而不是随机抽样:讲课时需要能指着数字说
# "第 3 号工厂就是比第 0 号热"。工厂之间的差异就是 group 切分要暴露的
# 那个"真实分布偏移"。
FACTORY_AMBIENT = [0.15, 0.40, 0.65, 0.90]

FEATURES = ["hours_z", "vibration_rms", "ambient"]


# ---------------------------------------------------------------------------
# 1. 数据生成:设备级个体倾向 + 工厂级共同因子 + 日级噪声
# ---------------------------------------------------------------------------
def gauss(rng):
    """标准正态,Box-Muller。手写是为了不依赖 random.gauss 的实现细节。"""
    u1 = rng.random()
    u2 = rng.random()
    if u1 < 1e-12:
        u1 = 1e-12
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def sigmoid(z):
    """数值稳定的 sigmoid,避免 math.exp 溢出。"""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def make_dataset(seed=SEED):
    """构造设备故障预测数据集,返回 list[dict],按绝对日全局有序。

    每行字段:
      device     设备号(字符串),group 的最小单位
      factory    工厂号(字符串),group 的一级
      day        该设备自己的第几天(0..29)
      abs_day    全局绝对日 = 交付日 + day,time_split 用它切
      hours_z    累计运行时长的 z 分数(预测时点拿得到)
      vibration_rms 当日振动均方根(预测时点拿得到)
      ambient    工厂环境温度,同工厂所有设备共享(预测时点拿得到)
      failed     标签:未来 7 天内是否故障

    标签由三部分相加:环境应力 + 运行时长 + 设备个体倾向。
    第三项是"这台机器天生就更容易坏",它**只能从这台设备自己的
    故障记录里学到** —— 这正是随机切分会替模型偷看的东西。
    """
    rng = random.Random(seed)
    rows = []

    # 每台设备:交付日、起始运行时长、个体敏感度(工厂级共同因子之外的
    # 第二层结构 —— 同一台机器 30 天里共享同一个敏感度)。
    #
    # 设备编号刻意按 `工厂 = i % 4` 交错:每一"波"交付里四座工厂各出一台。
    # 如果按工厂整块编号,工厂就和时间轴完全共线,time_split 的分布偏移
    # 会退化成"我换了个工厂"的偏移,而不是"时间往前走了"的偏移。
    # 交错之后,任何一个时间点上四座工厂都在场,时序切分测的才是时间。
    devices = []
    for i in range(N_DEVICES):
        factory = i % N_FACTORIES
        install_day = (i // N_FACTORIES) * 6  # 每 6 天交付一波,每波 4 台
        # 起始运行时长 = 交付批次趋势 + 每台设备自己的起始读数。
        # 后一项是刻意加的:没有它, hours_z 就是时间轴的确定性函数,
        # time_split 的均值差会退化成"我切了个时间点"的同义反复,而不是
        # 一个测量结果。
        base_hours = 200.0 + 30.0 * install_day + 120.0 * gauss(rng)
        sensitivity = 0.5 * gauss(rng)
        devices.append(
            {
                "device": "D%02d" % i,
                "factory": "F%d" % factory,
                "ambient": FACTORY_AMBIENT[factory],
                "install_day": install_day,
                "base_hours": base_hours,
                "sensitivity": sensitivity,
            }
        )

    for dev in devices:
        for day in range(DAYS_PER_DEVICE):
            hours = dev["base_hours"] + 18.0 * day
            hours_z = (hours - 900.0) / 700.0
            # 振动 = 工厂环境 + 个体敏感度 + 当日噪声
            vibration = 0.45 * dev["ambient"] + 0.30 * dev["sensitivity"] + 0.22 * gauss(rng)
            vibration = max(0.02, vibration)
            vibration_rms = round(vibration, 4)

            z = (
                -4.20
                + 2.20 * dev["ambient"]
                + 0.95 * hours_z
                + 1.30 * (vibration_rms - 0.30)
                + 1.40 * dev["sensitivity"]
                + 0.90 * gauss(rng)
            )
            rows.append(
                {
                    "device": dev["device"],
                    "factory": dev["factory"],
                    "day": day,
                    "abs_day": dev["install_day"] + day,
                    "hours_z": round(hours_z, 4),
                    "vibration_rms": vibration_rms,
                    "ambient": dev["ambient"],
                    "failed": 1 if rng.random() < sigmoid(z) else 0,
                }
            )

    # 全局按绝对日排序 —— 时序切分必须切的是真实时间轴,不是文件顺序
    rows.sort(key=lambda r: (r["abs_day"], r["device"]))
    return rows


# ---------------------------------------------------------------------------
# 2. 三种切分:签名统一,只接受 list[dict] + 候选行号,返回 (train_idx, test_idx)
# ---------------------------------------------------------------------------
def random_split(records, indices, test_fraction=TEST_FRACTION, seed=SEED):
    """AI 默认给的切分:整行打乱。

    它不问你数据有没有实体结构、有没有时间顺序 —— 而这两个问题在
    真实业务数据里几乎总是有答案。返回打乱后的 (train_idx, test_idx)。
    """
    shuffled = list(indices)
    random.Random(seed).shuffle(shuffled)
    n_test = int(round(len(shuffled) * test_fraction))
    test_idx = sorted(shuffled[:n_test])
    train_idx = sorted(shuffled[n_test:])
    return train_idx, test_idx


def time_split(records, indices, test_fraction=TEST_FRACTION):
    """时序切分:早的_days 训练,晚的日子测试。

    修好的是**方向**:测试期永远在训练期之后,模型看不到未来。
    但只要同一台设备在训练和测试里都出现,实体那条线还是没修 ——
    切完之后再看 overlap 那一列。
    """
    ordered = sorted(indices, key=lambda i: records[i]["abs_day"])
    n_test = int(round(len(ordered) * test_fraction))
    test_idx = sorted(ordered[-n_test:])
    train_idx = sorted(ordered[:-n_test])
    return train_idx, test_idx


def group_split(records, indices, test_fraction=TEST_FRACTION, level="device", seed=SEED):
    """group 切分:整组留出,组内记录永不跨边界。

    level="device"  随机留出 25% 的设备(等价于 sklearn 的 GroupShuffleSplit)
    level="factory" 整座工厂留作测试 —— 新工厂到货,一台老设备都没见过

    两种粒度都能把重叠压到 0,但只有后一种会暴露工厂之间的分布偏移。
    "重叠 = 0" 不等于"切分对了",这一对是本章最容易讲错的地方。
    """
    if level == "device":
        groups = sorted({records[i]["device"] for i in indices})
        random.Random(seed + 1).shuffle(groups)
        n_test_groups = max(1, int(round(len(groups) * test_fraction)))
        held_out = set(groups[:n_test_groups])
        key = "device"
    elif level == "factory":
        groups = sorted({records[i]["factory"] for i in indices})
        # 留出环境应力最高的那座工厂:新厂上线时条件更极端,不是随机挑一座
        groups.sort(key=lambda f: -records[next(i for i in indices if records[i]["factory"] == f)]["ambient"])
        n_test_groups = max(1, int(round(len(groups) * test_fraction)))
        held_out = set(groups[:n_test_groups])
        key = "factory"
    else:
        raise ValueError("level 只能是 'device' 或 'factory',收到 %r" % level)

    train_idx, test_idx = [], []
    for i in indices:
        if records[i][key] in held_out:
            test_idx.append(i)
        else:
            train_idx.append(i)
    return sorted(train_idx), sorted(test_idx)


# ---------------------------------------------------------------------------
# 3. 切分质量的三个量:重叠、时间倒流、分布偏移
# ---------------------------------------------------------------------------
def group_overlap(records, train_idx, test_idx, key="device"):
    """返回 (重叠组数, 测试集组数, 测试集组里有多大比例在训练集出现过)。"""
    train_groups = {records[i][key] for i in train_idx}
    test_groups = {records[i][key] for i in test_idx}
    overlap = train_groups & test_groups
    return len(overlap), len(test_groups), len(overlap) / len(test_groups)


def future_rows(records, train_idx, test_idx):
    """时间倒流行数:测试行里,存在"同一台设备、日子比它更晚"的训练行。

    随机切分下这个数很大 —— 模型可以直接读到测试日的未来状态。
    时序切分下它必须是 0,否则时序切分根本没做对。
    """
    max_train_day = {}
    for i in train_idx:
        r = records[i]
        if r["device"] not in max_train_day or r["day"] > max_train_day[r["device"]]:
            max_train_day[r["device"]] = r["day"]
    count = 0
    for i in test_idx:
        r = records[i]
        if r["device"] in max_train_day and max_train_day[r["device"]] > r["day"]:
            count += 1
    return count


def mean_of(records, indices, key):
    if not indices:
        return float("nan")
    return sum(records[i][key] for i in indices) / len(indices)


def mean_diff(records, train_idx, test_idx, key):
    """训练均值 - 测试均值。0 表示两半同分布,大表示切分切出了偏移。"""
    return mean_of(records, train_idx, key) - mean_of(records, test_idx, key)


# ---------------------------------------------------------------------------
# 4. 两个被评估的估计器
# ---------------------------------------------------------------------------
def device_prior_predict(records, train_idx, test_idx):
    """A 估计器:设备先验。测试行属于哪台设备,就用那台设备训练集里的
    故障率预测;该设备没在训练集出现过就退回总体故障率。

    这不是傻模型 —— 这是可靠性工程师上手第一个会做的东西,而且在
    "预测一台你已经在看的设备的明天"这个场景里,它是对的。
    随机切分会把这个模型喂成一个看起来很厉害的模型。
    """
    pos, tot = {}, {}
    for i in train_idx:
        dev = records[i]["device"]
        tot[dev] = tot.get(dev, 0) + 1
        pos[dev] = pos.get(dev, 0) + records[i]["failed"]
    overall = sum(records[i]["failed"] for i in train_idx) / len(train_idx)

    preds, fallback = [], 0
    for i in test_idx:
        dev = records[i]["device"]
        if dev in tot:
            preds.append(pos[dev] / tot[dev])
        else:
            preds.append(overall)
            fallback += 1
    return preds, fallback


def column_moments(records, indices):
    """训练集的 (均值, 标准差),只从训练集算。"""
    means, stds = [], []
    for key in FEATURES:
        col = [records[i][key] for i in indices]
        m = sum(col) / len(col)
        var = sum((v - m) ** 2 for v in col) / len(col)
        means.append(m)
        stds.append(math.sqrt(var) or 1.0)
    return means, stds


def design(records, indices, means, stds):
    return [
        [(records[i][key] - means[j]) / stds[j] for j, key in enumerate(FEATURES)]
        for i in indices
    ]


def bce_loss(y_true, y_pred):
    total = 0.0
    for t, p in zip(y_true, y_pred):
        p = min(max(p, 1e-12), 1.0 - 1e-12)
        total += -(t * math.log(p) + (1.0 - t) * math.log(1.0 - p))
    return total / len(y_true)


def predict_proba(X, w, b):
    return [sigmoid(sum(xi * wi for xi, wi in zip(row, w)) + b) for row in X]


def train_logistic_regression(X, y, epochs=EPOCHS, lr=LEARNING_RATE):
    """批量梯度下降。返回 (w, b, loss_history)。"""
    n, d = len(X), len(X[0])
    w = [0.0] * d
    b = 0.0
    history = []
    for epoch in range(epochs):
        preds = predict_proba(X, w, b)
        loss = bce_loss(y, preds)
        if epoch in (0, epochs // 2, epochs - 1):
            history.append((epoch + 1, loss))
        gw = [0.0] * d
        gb = 0.0
        for row, t, p in zip(X, y, preds):
            err = p - t
            for j in range(d):
                gw[j] += err * row[j]
            gb += err
        w = [w[j] - lr * gw[j] / n for j in range(d)]
        b = b - lr * gb / n
    return w, b, history


def roc_auc(y_true, y_pred):
    """Mann-Whitney U 形式的 AUC。并列分数按平均秩处理。

    测试集里所有预测分数都相同时返回 0.5 —— group 切分下设备先验
    估计器正是这种情况,这个 0.5 不是 bug,是全部证据。
    """
    pairs = sorted(zip(y_pred, y_true), key=lambda t: t[0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    n_pos = sum(t for _, t in pairs)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = sum(r for r, (_, t) in zip(ranks, pairs) if t == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# 5. 主体:四行切分,两个估计器
# ---------------------------------------------------------------------------
SPLITS = [
    ("random_split", "整行打乱", lambda r, i: random_split(r, i)),
    ("time_split", "前 80% 的日子", lambda r, i: time_split(r, i)),
    ("group_split(device)", "随机留出 25% 设备", lambda r, i: group_split(r, i, level="device")),
    ("group_split(factory)", "整座工厂留出", lambda r, i: group_split(r, i, level="factory")),
]


def main():
    started = time.time()
    records = make_dataset()
    all_idx = list(range(len(records)))
    labels = [r["failed"] for r in records]

    base_rate = sum(labels) / len(labels)
    print("=== 数据:设备故障预测 ===")
    print("  %d 行 / %d 台设备 / %d 座工厂 / 每台 %d 天,总体故障率 %.2f%%"
          % (len(records), N_DEVICES, N_FACTORIES, DAYS_PER_DEVICE, 100.0 * base_rate))
    print("  工厂环境应力 " + ", ".join("F%d=%.2f" % (i, a) for i, a in enumerate(FACTORY_AMBIENT)))
    print()
    print("  各工厂实际故障率(共同因子的直接证据:环境越极端,坏得越多)")
    print("    工厂   环境应力   故障率")
    for f in sorted({r["factory"] for r in records}):
        sub = [r["failed"] for r in records if r["factory"] == f]
        stress = next(r["ambient"] for r in records if r["factory"] == f)
        print("    %-5s %8.2f %8.2f%%" % (f, stress, 100.0 * sum(sub) / len(sub)))
    print()
    print("  标签 = 环境应力 + 运行时长 + 设备个体倾向 + 噪声。工厂级那一项是")
    print("  共同因子(整厂留出才能暴露);设备级那一项只能从这台设备自己的")
    print("  故障记录里学到,是 group 泄漏要偷看的东西。")
    print()

    rows = []
    for name, desc, splitter in SPLITS:
        train_idx, test_idx = splitter(records, all_idx)
        y_train = [labels[i] for i in train_idx]
        y_test = [labels[i] for i in test_idx]

        overlap, n_test_groups, overlap_ratio = group_overlap(records, train_idx, test_idx)
        futures = future_rows(records, train_idx, test_idx)
        d_hours = mean_diff(records, train_idx, test_idx, "hours_z")
        d_ambient = mean_diff(records, train_idx, test_idx, "ambient")

        preds_a, fallback = device_prior_predict(records, train_idx, test_idx)
        auc_a = roc_auc(y_test, preds_a)

        means, stds = column_moments(records, train_idx)
        X_train = design(records, train_idx, means, stds)
        X_test = design(records, test_idx, means, stds)
        w, b, _ = train_logistic_regression(X_train, y_train)
        auc_b = roc_auc(y_test, predict_proba(X_test, w, b))

        rows.append(
            {
                "name": name,
                "desc": desc,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "overlap": overlap,
                "test_groups": n_test_groups,
                "overlap_ratio": overlap_ratio,
                "futures": futures,
                "futures_pct": 100.0 * futures / len(test_idx),
                "d_hours": d_hours,
                "d_ambient": d_ambient,
                "auc_a": auc_a,
                "auc_b": auc_b,
                "fallback": fallback,
            }
        )

    print("=== 切分质量:重叠、时间倒流、分布偏移 ===")
    print("  切分                  训练/测试   组重叠      未来行泄漏    hours_z 均值差  ambient 均值差")
    print("  " + "-" * 96)
    for r in rows:
        print("  %-20s %4d/%4d  %2d/%-2d %5.1f%%  %4d (%5.1f%%)  %+14.4f  %+14.4f"
              % (r["name"], r["n_train"], r["n_test"],
                 r["overlap"], r["test_groups"], 100.0 * r["overlap_ratio"],
                 r["futures"], r["futures_pct"], r["d_hours"], r["d_ambient"]))
    print()
    print("  组重叠     : 同时出现在训练与测试的设备台数 / 测试集设备台数")
    print("  未来行泄漏 : 测试行里存在『同设备、日子更晚』的训练行的行数")
    print("  均值差     : 训练均值 - 测试均值。接近 0 = 两半同分布,明显偏离 = 切出了真实偏移")
    print()

    print("=== 两个估计器在同一份数据、同一组种子下的测试 AUC ===")
    print("  A 设备先验(记实体)   B 特征逻辑回归(只认特征)")
    print("  " + "-" * 60)
    for r in rows:
        print("  %-20s A = %.4f        B = %.4f" % (r["name"], r["auc_a"], r["auc_b"]))
    print()
    for r in rows:
        if r["fallback"]:
            print("  %-20s A 估计器有 %d/%d 条测试行退回总体故障率(该设备没在训练集里)"
                  % (r["name"], r["fallback"], r["n_test"]))
    print()

    rnd, dev, fac = rows[0], rows[2], rows[3]
    print("=== 结论:四个数字,四件不同的事 ===")
    print("  1) 设备先验 %.4f (random) -> %.4f (按设备留出),掉 %.4f。"
          % (rnd["auc_a"], dev["auc_a"], rnd["auc_a"] - dev["auc_a"]))
    print("     每条测试行退回常数,因为它要的那台设备一台都没见过。")
    print("     **这不是 bug,这是真能力被正确地否掉了** —— 上线换新设备时,")
    print("     这个模型确实只能给常数。")
    print("  2) 逻辑回归 %.4f (random) -> %.4f (按设备留出),%+.4f。"
          % (rnd["auc_b"], dev["auc_b"], dev["auc_b"] - rnd["auc_b"]))
    print("     **注意它没掉。** 随机切分对『不记实体』的模型没有系统性乐观,")
    print("     反而因为同类样本被稀释而略吃亏。所以本章不是『随机切分一定")
    print("     让指标虚高』,而是『随机切分对会记实体的模型虚高』。")
    print("  3) hours_z 均值差:random %+.4f / time %+.4f / 按厂 %+.4f。"
          % (rnd["d_hours"], rows[1]["d_hours"], fac["d_hours"]))
    print("     时序切分的偏移最大,这是**老设备磨损到后期**的真实漂移,")
    print("     随机切分几乎测不到(%.4f)。" % rnd["d_hours"])
    print("  4) ambient 均值差:random %+.4f / time %+.4f / 按厂 %+.4f。"
          % (rnd["d_ambient"], rows[1]["d_ambient"], fac["d_ambient"]))
    print("     时序切分测不到工厂差异(交错进场,时间上完全抵消);只有整厂")
    print("     留出才测得到。**两个切分修的是两条不同的线,不能互相替代。**")
    print()
    print("  跑完 %.2f 秒" % (time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
