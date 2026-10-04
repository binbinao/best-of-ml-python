#!/usr/bin/env python3
"""AI-generated engineered version, reviewed and corrected.

生成记录: 2026-10-03,提示词摘要="写一个信贷违约预测的 sklearn 训练脚本,
用 roc_auc_score 评估模型,输出模型性能报告"
审阅发现: 原始产出只有 `print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')`
一行。它在 1:99 类别不平衡的数据上会报出一个漂亮的 0.99,而模型什么都没学到。

    ⚠️  审阅发现(本文件实际状态,2026-10-04)
    ⚠️
    ⚠️  `report_accuracy_first()` 就是那行原始产出的原样保留,一字未改,
    ⚠️  它报告的 accuracy 0.9xx 全是"猜全是好客户"换来的。审阅者发现了、
    ⚠️  记录了,但**没有替换掉它** —— 因为本章正文要逐行对照这个陷阱。
    ⚠️
    ⚠️  说准确一点,免得下结论走偏 —— 这里有三个 accuracy:
    ⚠️
    ⚠️      恒定预测器(全放行,不看任何特征)   0.8561
    ⚠️      原始产出报告的                    0.8578
    ⚠️      成本最优阈值下的                  0.4733
    ⚠️
    ⚠️  **0.8578 那个数是空的**,它比"什么都不看"只高 0.0017。
    ⚠️  它不是 bug —— 它是"用错指标"的正确结果。信贷违约率 14.39%,
    ⚠️  一个恒定输出"好客户"的分类器就有 85.61% 的 accuracy。
    ⚠️  任何以 accuracy 为主指标的脚本,在这个问题上都必然报出一份好看的
    ⚠️  假报告。这份文件保留它,是为了让正文可以指着代码说:
    ⚠️  陷阱不长成 bug 的样子,陷阱长成"运行成功"的样子。
    ⚠️
    ⚠️  正确做法在 cost_optimal_threshold() 里:用业务成本而不是模型指标
    ⚠️  选阈值。函数名就写明了选择依据,不是 `best_by_auc`。
    ⚠️
    ⚠️  ⚠️  另一个更隐蔽的失真:模型 B 的 AUC 比 C 高 0.0984(300 组重采样
    ⚠️  ⚠️  100% 显著),但换到 1:1 成本口径后,C 反而比 B 便宜 3.09 元/笔
    ⚠️  ⚠️  (同样 300 组重采样,100% 显著)。**AUC 里没有成本,所以 AUC 对
    ⚠️  ⚠️  这次换人完全无感。** 见第 4 段,重采样见 metrics_minimal 第 6 段。
    ⚠️
    ⚠️  sklearn 给了 roc_auc_score / precision_recall_curve / 各种 metric,
    ⚠️  但它**没有一个参数是"漏批赔 1000、误批赔 100"**。成本必须你自己
    ⚠️  带进去。库提供度量,不提供业务。

    用法:
        /opt/homebrew/bin/python3 docs/tutorial/code/ch04/metrics_engineered.py
"""
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 数据与成本设定从 minimal 版原样导入,保证两份代码的数字可以直接对照
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from metrics_minimal import (  # noqa: E402
    C_FN, C_FP, C_FN_FLAT, C_FP_FLAT, N_APPLICANTS, SEED,
    make_dataset, min_expected_cost, roc_auc_rank_sum,
    score_broad, score_broad_plus_channel, score_channel_only, split_by_time,
    sweep_thresholds,
)

FEATURES = ["debt_ratio", "income_stability", "credit_history_years"]
COST_OPTIMAL_LABEL = "0.5"  # 只是标签,不代表本章推荐这个阈值


# ---------------------------------------------------------------------------
# ⚠️ 陷阱所在:原始 AI 产出,一字未改
# ---------------------------------------------------------------------------
def report_accuracy_first(y_true, y_pred):
    """⚠️ 陷阱所在:原始 AI 产出,一字未改。

    这是代码示例里最常见的一行评估。它只问"判对了多少",不问"判错在哪边"。

    在类别不平衡的问题上(欺诈率 1%、违约率 14%),一个恒定输出"好客户"的
    分类器就有 85.61% 的 accuracy。这个函数不会报错,不会告警,会打印
    一个漂亮的数字,并让读报告的人以为模型学到了东西。
    """
    acc = accuracy_score(y_true, y_pred)
    print(f"  Model performance report")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Looks good!")
    return acc


def constant_predictor(y_true, value=0):
    """恒定预测器:不看任何特征,永远输出同一个值。

    它存在的意义是给 accuracy 一个**基线**。REVIEW-CHECKLISTS.md 那条
    "What baseline does this beat — a constant, a rule, last week's value?"
    问的就是这个:报告里的每个数,都得先和这个函数比一比。
    """
    return np.full(len(y_true), value, dtype=int)


# ---------------------------------------------------------------------------
# sklearn 的标准评估
# ---------------------------------------------------------------------------
def fit_logreg(X_train, y_train, X_test):
    """标准化放进 Pipeline —— fit 只见过训练集。"""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
    ])
    pipe.fit(X_train, y_train)
    return pipe.predict_proba(X_test)[:, 1], pipe


def to_matrix(records, feature_names=FEATURES):
    return np.array([[r[name] for name in feature_names] for r in records])


def cost_optimal_threshold(y_true, y_score, c_fn=C_FN, c_fp=C_FP):
    """用业务成本选阈值,而不是用 accuracy 或 F1。

    这是本章唯一推荐的做法:阈值是**决策变量**,选它的依据必须是决策的代价。
    扫遍所有候选阈值,取期望成本最低的那个。
    """
    rows = sweep_thresholds_records(y_true, y_score, c_fn, c_fp)
    best = min(rows, key=lambda r: r["cost"])
    return best


def sweep_thresholds_records(y_true, y_score, c_fn=C_FN, c_fp=C_FP):
    """和 minimal 的 sweep_thresholds 同一套逻辑,只是输入换成了 numpy 数组。"""
    y_true = list(y_true)
    y_score = [float(s) for s in y_score]
    records = [{"default": int(t)} for t in y_true]
    return sweep_thresholds(records, y_score, c_fn, c_fp)


def report_confusion(y_true, y_pred, title):
    """打印混淆矩阵 —— 成本从这一格一格里长出来。"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    cost = (C_FN * fn + C_FP * fp) / float(len(y_true))
    print("  %s" % title)
    print("    %-10s %10s %10s" % ("", "实际没违约", "实际违约"))
    print("    %-10s %10d %10d" % ("模型没拒", tn, fn))
    print("    %-10s %10d %10d" % ("模型拒了", fp, tp))
    print("    成本 %.2f 元/笔(C_FN=%.0f x FN=%d + C_FP=%.0f x FP=%d)/%d"
          % (cost, C_FN, fn, C_FP, fp, len(y_true)))
    return cost


def main():
    started = time.time()

    records = make_dataset()
    train, test = split_by_time(records)
    X_train = to_matrix(train)
    X_test = to_matrix(test)
    y_test = np.array([r["default"] for r in test])
    n = len(y_test)

    print("=" * 78)
    print("ch04 AI 工程版:sklearn 指标,以及 accuracy 陷阱")
    print("=" * 78)
    print()
    print("申请 %d 份  测试期 %d 份  违约率 %.2f%%"
          % (N_APPLICANTS, n, 100.0 * y_test.mean()))
    print("特征 %s" % ", ".join(FEATURES))
    print()

    # ---- 1. accuracy 陷阱 ----
    print("-" * 78)
    print("第 1 段  ⚠️ 陷阱:AI 默认报告的 accuracy")
    print("-" * 78)
    scores, pipe = fit_logreg(X_train, np.array([r["default"] for r in train]), X_test)

    cost_best = cost_optimal_threshold(y_test, scores)
    acc_best = max(sweep_thresholds_records(y_test, scores), key=lambda r: r["accuracy"])
    pred_cost = (scores >= cost_best["threshold"]).astype(int)
    pred_acc = (scores >= acc_best["threshold"]).astype(int)

    print("  原始产出会报的数字(模型在 accuracy 最优阈值下):")
    report_accuracy_first(y_test, pred_acc)
    print()
    print("  同一份数据,三个 accuracy:")
    const_acc = accuracy_score(y_test, constant_predictor(y_test, 0))
    print("    恒定预测器(全放行,不看任何特征) accuracy %.4f" % const_acc)
    print("    原始产出报告的 accuracy               %.4f" % accuracy_score(y_test, pred_acc))
    print("    成本最优阈值下的 accuracy              %.4f" % accuracy_score(y_test, pred_cost))
    print()
    print("  **accuracy 最优的阈值,恰好是业务最贵的阈值。** 原因很直白:")
    print("  accuracy 把 1 个 FN 和 1 个 FP 都算作'一次错误',而业务上它们相差 %.0f 倍。"
          % (C_FN / C_FP))
    print("  在 %.2f%% 的违约率下,'把大多数判对'几乎总是等于'几乎全放行',"
          % (100.0 * y_test.mean()))
    print("  于是 accuracy 最优解就滑到了'谁都不拒'那附近。")
    print()

    # ---- 2. 混淆矩阵:成本从哪来 ----
    print("-" * 78)
    print("第 2 段  混淆矩阵:两个成本就是这么长出来的")
    print("-" * 78)
    report_confusion(y_test, pred_cost,
                     "成本最优阈值 %.4f" % cost_best["threshold"])
    print()
    report_confusion(y_test, (scores >= acc_best["threshold"]).astype(int),
                     "accuracy 最优阈值 %.4f" % acc_best["threshold"])
    print()
    print("  上面两张表用的是**同一个模型、同一批分数**,只差阈值。")
    print("  成本相差 %.2f 元/笔(%.2f 倍)。accuracy 一样'好看'的那张表,"
          % (acc_best["cost"] - cost_best["cost"], acc_best["cost"] / cost_best["cost"]))
    print("  业务上是错的。")
    print()

    # ---- 3. sklearn 的 AUC 与 PR 曲线 ----
    print("-" * 78)
    print("第 3 段  roc_auc_score 与 precision_recall_curve")
    print("-" * 78)
    auc_sk = roc_auc_score(y_test, scores)
    auc_mine = roc_auc_rank_sum(list(y_test), [float(s) for s in scores])
    fpr, tpr, roc_thr = roc_curve(y_test, scores)
    precision, recall, pr_thr = precision_recall_curve(y_test, scores)
    ap = average_precision_score(y_test, scores)
    print("  roc_auc_score                %.4f" % auc_sk)
    print("  手写 roc_auc_rank_sum        %.4f  (差 %.2e)" % (auc_mine, abs(auc_sk - auc_mine)))
    print("  ROC 曲线点数 %d,面积与 roc_auc_score 同源" % len(fpr))
    print("  average_precision_score      %.4f  (基线 = 违约率 %.4f)" % (ap, y_test.mean()))
    print()
    print("  ROC 曲线和 PR 曲线在不平衡数据上讲的**不是同一个故事**:")
    print("  ROC 的横轴 FPR 分母是好客户(%.2f%%),分母大,FP 变化被摊薄;"
          % (100.0 * (1 - y_test.mean())))
    print("  PR 的横轴是 precision,只看被拒的那批人。")
    print("  本数据集 AUC %.4f 看着还行,但 base rate 只有 %.4f ——"
          % (auc_sk, y_test.mean()))
    print("  **在 1%% 欺诈率的数据上,应该看 PR 曲线,不看 ROC。** 本节违约率 %.2f%%,"
          % (100.0 * y_test.mean()))
    print("  两者都要看,但它们都不能告诉你阈值该定在哪。")
    print()

    # ---- 4. 三个模型,两种成本口径 ----
    print("-" * 78)
    print("第 4 段  三个模型 × 两种成本口径:minimal 那个反例的 sklearn 版")
    print("-" * 78)
    # 用和 minimal 完全相同的三个分数,这样两章的数字可以逐位对照。
    # **不要**在这里换成训练出来的逻辑回归:minimal 里的 A/B/C 是固定权重,
    # 换掉之后反例就不成立了(训练出来的 B 在两种口径下都赢)。
    models = {
        "A": np.array([score_broad(r) for r in test]),
        "B": np.array([score_broad_plus_channel(r) for r in test]),
        "C": np.array([score_channel_only(r) for r in test]),
    }
    print("  模型 A  宽信号(3 个连续特征)")
    print("  模型 B  宽信号 + 渠道标记")
    print("  模型 C  只用渠道标记(1 个二值特征)")
    print()
    print("  %-8s %10s %14s %14s" % ("模型", "AUC", "10:1 最低成本", "1:1 最低成本"))
    print("  " + "-" * 50)
    winners = {}
    for name in ("A", "B", "C"):
        sc_v = models[name]
        auc_v = roc_auc_score(y_test, sc_v)
        c10 = min_expected_cost(list(y_test), [float(s) for s in sc_v], C_FN, C_FP)[0]
        c11 = min_expected_cost(list(y_test), [float(s) for s in sc_v],
                                C_FN_FLAT, C_FP_FLAT)[0]
        winners[name] = (auc_v, c10, c11)
        print("  %-8s %10.4f %14.2f %14.2f" % (name, auc_v, c10, c11))
    print()
    auc_winner = max(winners, key=lambda k: winners[k][0])
    cost_winner_10 = min(winners, key=lambda k: winners[k][1])
    cost_winner_11 = min(winners, key=lambda k: winners[k][2])
    print("  AUC 赢家     %s (%.4f)" % (auc_winner, winners[auc_winner][0]))
    print("  10:1 成本赢家 %s (%.2f 元/笔)" % (cost_winner_10, winners[cost_winner_10][1]))
    print("  1:1  成本赢家 %s (%.2f 元/笔)" % (cost_winner_11, winners[cost_winner_11][2]))
    print()
    if cost_winner_10 == cost_winner_11 == auc_winner:
        print("  ⚠️ 本次运行三者是同一个赢家 —— 与 minimal 第 5 段不一致,别下结论。")
    else:
        print("  **AUC 的赢家不随成本口径变,业务成本的赢家变了。**")
        print("  roc_auc_score 里没有任何一个参数能表达'漏批赔 %.0f、误拒赔 %.0f',"
              % (C_FN_FLAT, C_FP_FLAT))
        print("  所以它对这次换人**完全无感**。第 1 段那个 accuracy 陷阱是同一个问题的")
        print("  低端版本:不是选错了指标,是根本没把业务代价写进评估。")
    print()

    # ---- 5. 结论 + 重采样由 minimal 提供 ----
    print("-" * 78)
    print("第 5 段  阈值怎么选:三个可落地的判据")
    print("-" * 78)
    print("  1. 阈值是决策变量,选它的依据是决策的代价,不是任何模型指标。")
    print("     sklearn 没有 cost-sensitive_threshold 这个函数,也不该有 ——")
    print("     c_fn / c_fp 是业务数字,得你自己带进去(见 cost_optimal_threshold)。")
    print("  2. 先问'这个指标在不平衡数据上还成立吗'。accuracy 在 14.39% 违约率下")
    print("     已经失效;欺诈率到 1% 时连 F1 都要看 PR 曲线才能判断。")
    print("  3. 报出去的每个数,先和恒定预测器比。本数据集的恒定基线是 %.4f accuracy,"
          % const_acc)
    print("     比它高多少才是模型真正的贡献。")
    print()

    # ---- 与 minimal 的逐位对照 ----
    print("  与 metrics_minimal.py 的逐位对照(同一批分数,两个 AUC 实现):")
    sk_auc_a = roc_auc_score(y_test, models["A"])
    my_auc_a = roc_auc_rank_sum([r["default"] for r in test],
                                [float(s) for s in models["A"]])
    print("    模型 A  sklearn %.17f" % sk_auc_a)
    print("    模型 A  手写     %.17f" % my_auc_a)
    print("    差 %.2e —— 两条算法路径不同(metrics_minimal 第 1 段已逐位对照过)"
          % abs(sk_auc_a - my_auc_a))
    print()
    print("    成本 %.2f 元/笔 (sklearn roc_curve 路径) vs %.2f 元/笔 (手写扫描路径)"
          % (min_expected_cost(list(y_test), [float(s) for s in models["A"]], C_FN, C_FP)[0],
             min(sweep_thresholds_records(y_test, models["A"]), key=lambda r: r["cost"])["cost"]))
    print("    两条路径独立算出同一个成本,说明第 1 段的对照不是同一份代码跑两遍。")
    print()

    print("-" * 78)
    print("跑完 %.2f 秒" % (time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
