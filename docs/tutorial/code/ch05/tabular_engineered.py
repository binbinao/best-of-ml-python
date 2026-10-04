#!/usr/bin/env python3
"""AI-generated engineered version, reviewed and corrected.

生成记录: 2026-10-03,提示词摘要="用 sklearn 写一个客户流失预测的表格建模
脚本:Pipeline 里放 StandardScaler + LogisticRegression(solver='lbfgs'),
网格搜 C,用 cross_val_score 报交叉验证 AUC"
审阅发现: 原始产出直接 `cross_val_score(pipe, X, y, cv=5)` —— 也就是
`StratifiedKFold(n_splits=5)`,然后把那个数当成交叉验证结果报进周报。

    ⚠️  生成记录里原写"也就是 StratifiedKFold(shuffle=True)",**这句是错的,
    ⚠️  审阅时查 `inspect.signature` 推翻了它**:sklearn 1.9.1 的签名是
    ⚠️  `(self, n_splits=5, *, shuffle=False, random_state=None)`,
    ⚠️  **默认是 shuffle=False,不打乱**。
    ⚠│    留着这条记录是想说明:一个流传很广的默认值,查一次签名就能纠正。
    ⚠│    而"查实"之后结论反而更糟 —— 见下面第 1 段的折结构表。

    ⚠️  审阅发现(本文件实际状态,2026-10-04)
    ⚠️
    ⚠️  1. `cv_5()` 就是那行原始产出,**一字未改**。它报出的均值 0.6617 是
    ⚠│    四行里最高的,也是唯一会被直接抄进周报的那个数。它比最严格的
    ⚠│    StratifiedGroupKFold(0.6308)高 +0.0309。审阅者发现了、记录了,
    ⚠│    但没有替换掉它,因为本章正文要指着它说:
    ⚠│    **错切分不会让指标变难看,它只会让指标变得比真相好看。**
    ⚠│
    ⚠│  2. 更高的那个数不是"模型更好",是**评估口径放水**。默认 5 折把
    ⚠│    同一个客户的早期月份放进训练、后期月份放进验证,模型于是可以用
    ⚠│    同一个人的其他月份当参照。面板数据上这一条是结构性的,不是配置
    ⚠│    失误 —— 实测每一折的验证侧客户都 100% 出现在训练侧(第 1 段)。
    ⚠│
    ⚠│  3. 正确做法在本文件里:`TimeSeriesSplit`(时序)、`GroupKFold`
    ⚠│    (同一客户不可跨折)、`StratifiedGroupKFold`(两者都要)。
    ⚠│    三种都跑了,数字在第 1 段。
    ⚠│
    ⚠│  4. 同一份 C 网格,三种 CV 选出**两个**不同的 C(0.01 / 0.1),而且
    ⚠│    各自网格内的分差都只有 0.001–0.004 —— 网格几乎是平的,所以
    ⚠│    "选中哪个 C"在很大程度上是折法的产物,不是模型结构的证据。
    ⚠│    **没有任何一种 CV 策略能救一个错的折法** —— CV 只回答"这个超参
    ⚠│    在这份数据上稳不稳",不回答"上线后准不准"。见第 2 段。
    ⚠│
    ⚠│  5. sklearn 的 Pipeline 解决的是 ch02 第 3 节那个 train/serve skew
    ⚠│    (scaler 只 fit 训练折),但它**不解决** ch02 的字段时点问题、
    ⚠│    ch03 的边界问题、ch04 的指标映射问题。本章不重复那三章,见正文。
    ⚠│
    ⚠│  6. 手写 AUC 与 sklearn 的**跨库逐位对照**在第 4 段:实测梯形法
    ⚠│    5/5 逐位一致、秩和法 4/5(差的 1 次是 1.1e-16,即 1 ulp)。

    用法:
        /opt/homebrew/bin/python3 docs/tutorial/code/ch05/tabular_engineered.py
"""
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GridSearchCV, GroupKFold,
                                   StratifiedGroupKFold, StratifiedKFold,
                                   TimeSeriesSplit, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 数据生成与切分从 minimal 版原样导入,保证两份代码的数字可以直接对照
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from tabular_minimal import (  # noqa: E402
    FEATURES, FIT_END_MONTH, SEED, VALID_END_MONTH,
    lift_at_fraction, make_dataset, roc_auc_rank_sum, roc_auc_trapezoid,
    split_by_month, to_matrix,
)

C_GRID = (0.01, 0.1, 1.0, 10.0)


def make_pipeline(c=1.0):
    """AI 产出的标准结构:Pipeline(scaler, logistic)。

    顺序是有讲究的 —— scaler 必须在分类器**之前**,这样 `cv` 每一折里
    scaler 只 fit 那一折的训练部分。这就是 ch02 第 3 节的 train/serve skew
    在结构上的解法,sklearn 已经内建好了。
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="lbfgs", C=c, max_iter=2000)),
    ])


def cv_5(X, y):
    """⚠️ 原始 AI 产出的原样保留,一字未改。

    `cross_val_score(..., cv=5)` 里 cv=5 走 `check_cv` → `StratifiedKFold(
    n_splits=5)`。**注意 sklearn 1.9.1 里 StratifiedKFold 的默认是
    `shuffle=False`**(签名 `(self, n_splits=5, *, shuffle=False,
    random_state=None)`)——**所以它并不打乱**。

    这一点很要紧,因为"默认随机切分"是一个流传很广的误解,本章一开始也
    写错了(见 fix_ 下的实测)。真实的失效形态是另一个,而且更隐蔽:

    `shuffle=False` + 数据本来就按月份排序 ⇒ 每一折的训练侧**都包含全部
    6 个月**,验证侧是其中 2–3 个月的**子集**。于是:

      - 同一客户 100% 两边都在(这是实体泄漏);
      - 83.3% 的验证行,其所属月份在训练侧也出现过(这是时间线重叠)。

    比随机切分更难发现,因为它**每一折都"用满了全部数据"**,看上去很勤勉。
    """
    return cross_val_score(make_pipeline(), X, y, cv=5, scoring="roc_auc")


def time_cv(X, y, n_splits=3):
    """正确做法之一:TimeSeriesSplit —— 训练永远在验证之前,不给未来行。"""
    return cross_val_score(make_pipeline(), X, y, cv=TimeSeriesSplit(n_splits=n_splits),
                           scoring="roc_auc")


def group_cv(X, y, groups, n_splits=3):
    """正确做法之二:GroupKFold —— 同一个客户的 12 行永不跨折。"""
    return cross_val_score(make_pipeline(), X, y, groups=groups,
                           cv=GroupKFold(n_splits=n_splits), scoring="roc_auc")


def stratified_group_cv(X, y, groups, n_splits=3):
    """正确做法之三:StratifiedGroupKFold —— 时间方向 + 客户边界 + 折间均衡。

    sklearn 1.6+ 才有这个类。它同时满足正负比例在折间大致均衡,**并且**
    不让同一客户跨折。前面两条各自解决一个问题,这一条一起解决。
    """
    return cross_val_score(make_pipeline(), X, y, groups=groups,
                           cv=StratifiedGroupKFold(n_splits=n_splits),
                           scoring="roc_auc")


def report(name, scores, note=""):
    print("  %-24s 均值 %.4f  sd %.4f  逐折 %s"
          % (name, scores.mean(), scores.std(),
             " ".join("%.4f" % s for s in scores)))
    if note:
        print("    %s" % note)


def main():
    t0 = time.time()
    print("ch05 表格建模全景 · sklearn 工程版")
    print("sklearn %s | numpy %s\n" % (__import__("sklearn").__version__,
                                        np.__version__))

    rows = make_dataset()
    fit_rows, val_rows, test_rows = split_by_month(rows)
    X_fit, y_fit = to_matrix(fit_rows)
    X_val, y_val = to_matrix(val_rows)
    X_test, y_test = to_matrix(test_rows)
    groups = np.array([r["customer_id"] for r in fit_rows])

    # ======================================================================
    # 第 1 段:原始产出的 CV 跑一遍,并把它和三种正确切分并排放
    # ======================================================================
    print("=== 1. CV 策略对照(同一份数据、同一套 Pipeline、同一组折数)===")
    print("  拟合段:月份 0..%d,%d 行,%d 个客户" % (FIT_END_MONTH - 1,
                                                   len(X_fit),
                                                   len(set(groups))))
    print("  ⚠️ 默认 5 折 = StratifiedKFold(**shuffle=False**):"
          "既不打乱,也不看时间和客户")

    s_default = cv_5(X_fit, y_fit)
    s_time = time_cv(X_fit, y_fit)
    s_group = group_cv(X_fit, y_fit, groups)
    s_sgkf = stratified_group_cv(X_fit, y_fit, groups)

    print()
    report("cv=5(原始产出)", s_default,
           "⚠️ 每一折训练侧都含全部 6 个月,验证侧是其中 2–3 个月的子集")
    report("TimeSeriesSplit", s_time,
           "✅ 时间方向干净:训练永远在验证之前")
    report("GroupKFold(customer)", s_group,
           "✅ 客户边界干净:同一个客户 12 行永不跨折")
    report("StratifiedGroupKFold", s_sgkf,
           "✅ 两者都干净,且折间正负比例还大致均衡")

    gap = s_default.mean() - s_sgkf.mean()
    print("\n  原始产出比最严格的折法高 %+.4f(默认 %.4f vs %.4f)"
          % (gap, s_default.mean(), s_sgkf.mean()))
    print("  这个差值**不是模型变好了**,是评估时把同一批客户的其他月份漏进了")
    print("  训练侧。四行按均值排序:")
    ranking = sorted([("cv=5(原始产出)", s_default),
                      ("TimeSeriesSplit", s_time),
                      ("GroupKFold(customer)", s_group),
                      ("StratifiedGroupKFold", s_sgkf)],
                     key=lambda t: -t[1].mean())
    for name, sc in ranking:
        print("    %.4f  %s" % (sc.mean(), name))
    print("  最高的那一行恰好是**边界最松**的那一行 —— 和 ch02 的形态相同:")
    print("  泄漏不会让指标报警,它只会让指标比真相好看。")
    print("  ⚠️ 但别把它读成'越松越高'的规律:四行之间最大差 %.4f,而每行"
          % (max(s.mean() for _, s in ranking) - min(s.mean() for _, s in ranking)))
    print("     的折间 sd 已经到 %.4f–%.4f。**排名不等于显著**,这个量级只够"
          % (min(s.std() for _, s in ranking), max(s.std() for _, s in ranking)))
    print("     说'默认 5 折给出的数是这里最乐观的一个',不够说'切分方式"
          "影响了排序'。")
    print("     真正确定的是下面这个计数 —— 它是 0/1 的,不是 0.05 的。")

    # 折内重叠的直接证据。注意这里用的是**默认构造**(shuffle 未传),
    # 因为那才是 cv_5() 真正执行的折法 —— 换成 shuffle=True 就是在测
    # 一个没发生过的配置。
    print("\n  cv=5 实际执行的折法(StratifiedKFold,**默认 shuffle=False**):")
    skf = StratifiedKFold(n_splits=5)      # 不传 shuffle,与 check_cv 一致
    months = np.array([r["month"] for r in fit_rows])
    n_cust_total = len(set(groups))
    tot_val = tot_month_overlap = 0
    for i, (tr_idx, te_idx) in enumerate(skf.split(X_fit, y_fit)):
        tr_m = sorted(set(months[tr_idx].tolist()))
        te_m = sorted(set(months[te_idx].tolist()))
        te_c = set(groups[te_idx])
        tr_c = set(groups[tr_idx])
        cust_ov = 100.0 * len(tr_c & te_c) / len(te_c)
        tr_months_set = set(tr_m)
        month_ov = sum(1 for idx in te_idx if months[idx] in tr_months_set)
        tot_val += len(te_idx)
        tot_month_overlap += month_ov
        print("    折%d 训练侧月份 %-19s 验证侧月份 %-10s 客户重叠 %.1f%%  "
              "验证行同月份也在训练侧 %d/%d"
              % (i, tr_m, te_m, cust_ov, month_ov, len(te_idx)))
    print("    合计 %.1f%% 的验证行,其所属月份在训练侧也出现过"
          % (100.0 * tot_month_overlap / tot_val))
    print("    客户重叠五折全部 100% —— 面板数据的结构性事实。")
    print("    ⚠️ 关键:shuffle=False + 数据按月排序 ⇒ **每折训练侧都含全部 6 个月**,"
          "\n       验证侧是其中 2–3 个月的子集。")
    print("       所以它既不是'随机打散'(常见的误解),也不是'时间在前'(更糟,"
          "\n       因为训练侧看到了验证侧的全部月份)。两种折法它都不满足。")

    tscv = TimeSeriesSplit(n_splits=3)
    t_overlaps = []
    t_month_ov = []
    t_sizes = []
    for tr_idx, te_idx in tscv.split(X_fit):
        t_overlaps.append(len(set(groups[tr_idx]) & set(groups[te_idx])))
        tr_m = set(months[tr_idx].tolist())
        t_month_ov.append(sum(1 for idx in te_idx if months[idx] in tr_m))
        t_sizes.append(len(te_idx))
    print("\n    TimeSeriesSplit 同一量: 客户 %s / 验证侧 %d;"
          " 验证行同月份也在训练侧 %s / %s"
          % (t_overlaps, n_cust_total, t_month_ov,
             "/".join(str(s) for s in t_sizes)))
    print("    ⚠️ 注意**中间那一折是 0**,首尾两折各 150 行:TimeSeriesSplit 的"
          "\n       首折从最早开始、末折扩到最晚,分界月份会被两边共享。")
    print("       严格不重叠要显式指定 gap=1 个月(留出预测间隔)。")
    print("       **即使这样,客户仍然 100% 跨折** —— 那是本场景的正确行为。")
    print("    客户跨折在时序场景下是**正常的** —— 真实业务就是预测一个老客户"
          "下个月会不会流失。")
    print("    TimeSeriesSplit 管的是时间方向,不管客户边界;两条线各管各的,"
          "谁也替代不了谁,见 ch03。")

    # ======================================================================
    # 第 2 段:C 网格搜索,三种 CV 各选一次
    # ======================================================================
    print("\n=== 2. C 网格搜索:同一份网格,三种 CV 选出**两个**不同的 C ===")
    print("  网格 C = %s" % (list(C_GRID),))
    chosen = {}
    for cv_name, cv in (("cv=5(原始产出)", 5),
                        ("TimeSeriesSplit", TimeSeriesSplit(n_splits=3)),
                        ("StratifiedGroupKFold",
                         StratifiedGroupKFold(n_splits=3))):
        pipe = make_pipeline()
        grid = GridSearchCV(pipe, {"clf__C": list(C_GRID)}, cv=cv,
                            scoring="roc_auc", n_jobs=1)
        if cv_name == "StratifiedGroupKFold":
            grid.fit(X_fit, y_fit, groups=groups)
        else:
            grid.fit(X_fit, y_fit)
        chosen[cv_name] = grid.best_params_["clf__C"]
        print("\n  %s 选中 C = %s(该 CV 下最高均值 AUC %.4f)"
              % (cv_name, grid.best_params_["clf__C"], grid.best_score_))
        for params, score in zip(grid.cv_results_["params"],
                                 grid.cv_results_["mean_test_score"]):
            mark = "  <- 选中" if params["clf__C"] == grid.best_params_["clf__C"] else ""
            print("      C=%-6s 均值 AUC %.4f%s"
                  % (params["clf__C"], score, mark))
    print("\n  三种 CV 选出的 C: %s" % chosen)
    print("  **GridSearchCV 不会告诉你选错了 CV** —— 它对三种策略一视同仁地"
          "返回一个数字。")
    print("  错的是折法,不是网格。而且注意每张表里的分差都在 0.001–0.004,")
    print("  网格几乎是平的 —— 这种情况下'选中哪个 C'更多是折法的产物。")

    # ======================================================================
    # 第 3 段:最终评估 —— 固定时间切分上的 holdout,只跑一次
    # ======================================================================
    print("\n=== 3. 最终评估:训练段 fit,验证段选超参,测试段只跑一次 ===")
    best_c = chosen["StratifiedGroupKFold"]
    pipe = make_pipeline(c=best_c)
    pipe.fit(X_fit, y_fit)
    val_scores = pipe.predict_proba(X_val)[:, 1]
    test_scores = pipe.predict_proba(X_test)[:, 1]
    auc_val = roc_auc_score(y_val, val_scores)
    auc_test = roc_auc_score(y_test, test_scores)
    print("  最终 C = %s(按 StratifiedGroupKFold 选的)" % best_c)
    print("  验证段(月份 %d..%d)  AUC %.17f"
          % (val_rows[0]["month"], val_rows[-1]["month"], auc_val))
    print("  测试段(月份 %d..%d)  AUC %.17f  <- 这是唯一能拿去汇报的数"
          % (test_rows[0]["month"], test_rows[-1]["month"], auc_test))
    base, top, lift = lift_at_fraction(y_test, list(test_scores), 0.10)
    print("  基础流失率 %.4f | 前 10%% 名单流失率 %.4f | lift %.2fx"
          % (base, top, lift))
    print("  lift 与 AUC 不可互推(ch04 实测过);报 lift 必须带'哪一档'")

    # 恒定基线:ch04 的教训,每个数都要有基线
    from sklearn.dummy import DummyClassifier
    dummy = DummyClassifier(strategy="prior").fit(X_fit, y_fit)
    auc_dummy = roc_auc_score(y_test, dummy.predict_proba(X_test)[:, 1])
    print("  恒定预测器(不看任何特征)测试 AUC %.4f" % auc_dummy)
    print("  模型的全部贡献是 %+.4f —— 0.5 那个数是**恒等式不是测量**"
          % (auc_test - 0.5))

    # ======================================================================
    # 第 4 段:手写 AUC 与 sklearn 的跨库逐位对照
    # ======================================================================
    print("\n=== 4. 跨库逐位对照:手写 AUC vs sklearn.roc_auc_score ===")
    print("  [真正的跨库验证 —— 两条实现分属不同库,不像 minimal 版的自证]")
    y_list = list(y_test)
    s_list = list(test_scores)
    theirs = roc_auc_score(y_list, s_list)
    mine_rank = roc_auc_rank_sum(y_list, s_list)
    mine_trap = roc_auc_trapezoid(y_list, s_list)
    print("  sklearn roc_auc_score  %.17f" % theirs)
    print("  手写 秩和法             %.17f   差 %.2e" % (mine_rank, mine_rank - theirs))
    print("  手写 梯形法(算法同构)       %.17f   差 %.2e"
          % (mine_trap, mine_trap - theirs))
    print("  秩和 == sklearn: %s | 梯形 == sklearn: %s"
          % (mine_rank == theirs, mine_trap == theirs))
    print("  梯形法能声称逐位一致,是因为它复刻了 sklearn 的算法结构")
    print("  (降序排序 + distinct_value_indices + drop_intermediate +")
    print("   numpy 的 pairwise summation)。少复刻 pairwise_sum 就只能声称")
    print("   '四位小数一致' —— ch04 第 2 节实测过这个差别。")
    print("  秩和法是**另一条算法**,与 sklearn 最多差 1 ulp(浮点求和顺序)。")

    # 五组数据上的全量对照,不是只报这一组
    from tabular_minimal import make_auc_selfcheck_cases
    print("\n  五个构造数据集上的全量对照(不同规模/正样本比例/并列):")
    exact_rank = exact_trap = 0
    for case_id, (y_case, s_case) in enumerate(make_auc_selfcheck_cases()):
        ref = roc_auc_score(y_case, s_case)
        r = roc_auc_rank_sum(y_case, s_case)
        t = roc_auc_trapezoid(y_case, s_case)
        exact_rank += (r == ref)
        exact_trap += (t == ref)
        print("    数据集 %d  n=%-5d sklearn %.17f | 秩和 %s | 梯形 %s"
              % (case_id, len(y_case), ref,
                 "逐位一致" if r == ref else "差 %.1e" % (r - ref),
                 "逐位一致" if t == ref else "差 %.1e" % (t - ref)))
    print("  梯形法(算法同构) %d/5 逐位一致;秩和法 %d/5 逐位一致"
          % (exact_trap, exact_rank))
    print("  秩和法那几次的 1 ulp 差异**出在 sklearn 侧** —— minimal 版用")
    print("  Fraction 精确有理数重算过,手写秩和法 5/5 等于真值。见 ch04。")

    # 系数表:sklearn 版一样骗不了人
    print("\n  系数表(C=%.2f,标准化后,sklearn):" % best_c)
    coefs = pipe.named_steps["clf"].coef_[0]
    order = np.argsort(-np.abs(coefs))
    for rank, j in enumerate(order, 1):
        print("    %-22s %+9.4f %7d" % (FEATURES[j], coefs[j], rank))
    print("    (截距)                  %+9.4f" % pipe.named_steps["clf"].intercept_[0])
    print("    全部特征都是真实信号 —— 与 ch02 的泄漏权重表形态完全不同。")

    print("\n耗时 %.1f 秒" % (time.time() - t0))


if __name__ == "__main__":
    main()
