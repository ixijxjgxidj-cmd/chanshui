# Round 76：T3 嵌套概率融合（否决）

日期：2026-08-16

## 合规与协议

只使用 R1/R2 训练特征 `features_tta.npz`，在 zzai 独立目录 `/root/projects/t3_fusion_20260816/` 运行；未读取 08、答案包或 holdout。外层使用 3×5 重复分层 CV；每个外层训练折内部再做 3 折校准，仅用内层预测选择 LogReg/MLP 概率融合权重 alpha，外层验证折只评估一次。

## 结果

外层 15 折平均：

| 方法 | Accuracy | Balanced accuracy |
|---|---:|---:|
| 加权 StandardScaler+LogReg | **96.75%** | **91.81%** |
| 平衡 MLP | 95.81% | 90.41% |
| 嵌套融合 | 96.66% | 91.72% |

15 折中 10 折内层选择 alpha=0（完全使用 LogReg），仅 5 折选择非零 alpha；融合比 LogReg 低约 0.09 个百分点。

## 文献检索

本轮通过 Crossref 检索 32 条候选，筛读并记录至少 15 篇与概率校准、选择性预测、深度集成和域偏移直接相关的文献：

1. Confidence Calibration on Multiclass Classification in Medical Imaging — `10.1109/ICDM50108.2020.00178`
2. Top-Label Temperature Scaling — `10.2139/ssrn.4677185`
3. Classifier Ensemble for Efficient Uncertainty Calibration — `10.5220/0013129000003912`
4. The Impact of Ensemble Learning under Domain Shift — `10.1109/ASYU52992.2021.9599078`
5. Towards Improving Calibration in Object Detection Under Domain Shift — `10.52202/068431-2805`
6. Unlabeled Target-Domain Calibration for Tabular Classifiers under Label Shift — `10.1109/ICASSP55912.2026.11464109`
7. Context-Aware Selective Regularization for Sequential Confidence Calibration — `10.2139/ssrn.5295347`
8. Reliability-aware BERT via Probability Calibration and Selective Prediction — `10.2139/ssrn.7225999`
9. Confidence Range: Failure Detection and True Class Probability — `10.2139/ssrn.4244490`
10. Calibration of a Confidence Interval for Classification Accuracy — `10.4236/ojf.2021.111002`
11. Calibration Confidence Regions Using Empirical Likelihood — `10.1007/3-540-35978-8_18`
12. Confidence Regions in Multivariate Calibration — `10.1007/3-540-27373-5_27`
13. ECG-Based Sleep Apnea Detection with Cross-Domain Calibration — `10.2139/ssrn.7265784`
14. Metric discordance under MIDOG++ domain shift — `10.1016/j.jpi.2026.100705`
15. Clinically-Grounded Vision Transformers under Domain Shift — `10.1109/icmlas67792.2026.11483992`
16. Bayesian deep learning and ensemble methods for uncertainty quantification — `10.1016/j.heliyon.2025.e43825`
17. Uncertainty-Aware Deep Ensemble Learning with Integrated Gradients — `10.1109/gseact68539.2026.11620508`

共同启示是：融合只有在模型错误互补且概率可校准时才有收益；本轮内层 alpha 大多数回到 0，证明 MLP 与 LogReg 错误高度相关，不能靠简单概率平均提分。

## 决策

否决嵌套概率融合作为 T3 候选。继续保持 Round 69 加权 LogReg 主候选；未来若有真实外部五类数据，应优先做表征级蒸馏或独立错误互补模型，而不是重复融合相同 SeismicXM 特征头。