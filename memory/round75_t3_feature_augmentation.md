# Round 75：T3 训练折内少数类特征扰动增强

日期：2026-08-16

## 合规与协议

实验在 zzai 独立目录 `/root/projects/t3_feature_aug_20260816/` 运行，只使用已有 R1/R2 训练特征 `features_tta.npz`；没有读取 08、答案包或 holdout，也没有把 STEAD 的地震/噪声元数据伪映射成 T3 五类。

每个重复分层 CV 折内，只对训练折的每个已出现类别扩增到 20/30/40 条，加入相对于训练特征全局标准差的 Gaussian 扰动 `.01/.03/.05`；验证折完全保持原始。比较 9 个预注册配置，官方口径按整体 accuracy，另报告 balanced accuracy。

## 结果

| CV seed | 最佳配置 | Accuracy | Balanced accuracy |
|---:|---|---:|---:|
| 20260816 | 每类 20，噪声 .05 | 96.50% | 91.33% |
| 73 | 每类 40，噪声 .03 | 96.61% | 91.39% |
| 同口径无增强加权 LogReg | — | **96.30%** | 91.16% |

两次独立种子均有弱正增益（+0.20、+0.31 个百分点），但最佳增强强度不一致，且增益很小；没有独立 R1→R2 目标域证据证明它能改善跨轮泛化。

## 文献检索与依据

本轮通过 Crossref 检索 32 条候选并筛读至少 15 篇，关键文献/DOI：

1. Deep Learning-Based Rice Grain Classification with Class Imbalance Handling Using Weighted Sampling and Data Augmentation — `10.30871/jaic.v10i3.12930`
2. Feature Selection, SMOTE and Under Sampling on Class Imbalance — `10.1109/UKSIM.2012.116`
3. Heterogeneous Domain Generalization Via Domain Mixup — `10.1109/ICASSP40776.2020.9053273`
4. Mixup-Induced Domain Extrapolation for Domain Generalization — `10.1609/aaai.v38i10.28994`
5. Learning gradient-based mixup with extrapolation toward flatter minima — `10.1016/j.artint.2026.104544`
6. Semantic-Aware Mixup for Domain Generalization — `10.1109/IJCNN54540.2023.10191056`
7. Learning Minority Class prior to Minority Oversampling — `10.1109/IJCNN.2019.8852188`
8. Adaptive learning of minority class prior to minority oversampling — `10.1016/j.patrec.2020.05.020`
9. Hybrid oversampling with Borderline-SMOTE and GANs — `10.1016/j.mlwa.2025.100637`
10. Multi-Class Imbalance in Text Classification: Feature Engineering — `10.3390/informatics7040052`
11. Causality-Preserving Domain Generalization via Adaptive Fourier Mixup — `10.1109/TPAMI.2026.3688520`
12. Domain Adaptation for Defor­estation Detection with Class Imbalance — `10.17771/pucrio.acad.70461`
13. Deep Multi-Source Supervised Domain Adaptation with Class Imbalance — `10.21203/rs.3.rs-3160713/v1`
14. Class-imbalance suppression in domain adaptation — `10.1007/s13042-026-03165-7`
15. Domain adaptation with label-aligned sampling (DALAS) — `10.1016/j.eswa.2023.122910`
16. Handling Class Imbalance in Black-Box Unsupervised Domain Adaptation — `10.1109/VCIP63160.2024.10849930`
17. Centroida: Cross-Domain Class Discrepancy Minimization — `10.2139/ssrn.4698435`
18. Ensemble of Averages: Improving Model Selection in Domain Generalization — `10.52202/068431-0601`

共同启示是：少数类扰动增强只有在增强保持类别语义、且目标域变化与扰动同源时才可靠；本轮特征空间 Gaussian 扰动缺少物理保证，因此只能作为弱候选，不能替代真实波形增强或公开同语义数据。

## 决策

保留该方法为低优先级候选，不覆盖 Round 69 加权 LogReg 或已验证 T3 生产包。下一轮若继续，应改为原始三分量波形上的物理一致增强（增益、时间平移、带限噪声、通道缩放），并用 R1/R2 跨轮留出验证；若没有原始波形缓存则不伪造增强收益。