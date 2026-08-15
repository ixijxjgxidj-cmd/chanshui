# Round 73：T3 源轮次预注册轻量头网格（否决新增头）

日期：2026-08-16

## 合规与协议

实验只使用 R1/R2 训练特征 `features_tta.npz`，在 zzai 独立目录 `/root/projects/t3_head_grid_20260816/` 运行。每个方向先只在源轮次做固定随机种子 5 折选择，再以选定头固定评估目标轮次；目标轮次标签不参与配置选择。没有读取 08、答案包或 holdout。

候选是预注册的 33 个轻量头：Standard/Robust/Normalizer × cosine/euclidean × k=1/3/5/7/9，及 StandardScaler+balanced LogisticRegression (C=.1/1/10)。

## 结果

| 源→目标 | 源端选择 | 源端 CV | 目标准确率 | 目标 balanced acc |
|---|---|---:|---:|---:|
| R1→R2 | `norm_cosine_knn5` | 94.50% / 91.00% | 98.94% | 96.55% |
| R2→R1 | `std_logreg_C10` | 100% / 100% | 74.50% | 36.25% |

R1→R2 未超过 Round 68 标准化 kNN 的 99.47%；R2→R1 略高于普通 kNN 的 70%，但因 R2 不含 3–5 类仍远不足以作为通用方案。

## 文献检索与解释

本轮再检索 32 条候选，筛读与结论直接相关的论文：

1. Ensemble of Averages: Improving Model Selection in Domain Generalization — `10.52202/068431-0601`
2. Source domain selection method for domain generalization — `10.1088/2631-8695/ae1dd9`
3. Towards Optimization and Model Selection for Domain Generalization — `10.1137/1.9781611978032.28`
4. Relevant and invariant feature selection for domain generalization — `10.1109/IGARSS.2014.6947252`
5. Deep Multi-Source Supervised Domain Adaptation with Class Imbalance — `10.21203/rs.3.rs-3160713/v1`
6. Iterative resampling deep decoupling domain adaptation — `10.2139/ssrn.4650965`
7. Class-imbalance suppression in domain adaptation — `10.1007/s13042-026-03165-7`
8. Domain adaptation with label-aligned sampling — `10.1016/j.eswa.2023.122910`
9. Handling Class Imbalance in Black-Box Unsupervised Domain Adaptation — `10.1109/VCIP63160.2024.10849930`
10. Centroida: cross-domain class discrepancy minimization — `10.2139/ssrn.4698435`
11. Uncertainty-Gated Selective Graph Correction — `10.2139/ssrn.6462436`
12. Selective Prediction and Uncertainty-Aware Referral — `10.20944/preprints202607.1763.v1`
13. Uncertainty-Guided Selective Prediction — `10.30693/smj.2026.15.4.9`
14. Boosting Naive Bayes classification using uncertainty-based selective sampling — `10.1016/j.neucom.2004.09.003`
15. Simple Domain Generalization Methods are Strong Baselines — `10.1109/IJCNN60899.2024.10650639`

共同启示是：源域内选出的最优头未必跨域最优，尤其目标域类别支持不完整时；要做稳健选择必须有多个完整源域或显式目标域风险信号。本项目只有 R1 提供 3–5 类训练支持，不能从 R2 中推断这些类别的判别边界。

## 决策

否决“扩大轻量分类头网格”作为主要提分路线。T3 保留 Round 69 联合加权 LogReg（联合 CV 96.92%）和 Round 68 的 R1→R2 标准化 kNN（99.47%）为研究候选；未来训练应优先补充公开五类数据或进行表征级域泛化，而不是继续在现有 R1/R2 特征上搜索分类头。