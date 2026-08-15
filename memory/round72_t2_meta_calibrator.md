# Round 72：T2 成员统计元校准器（否决跨轮迁移）

日期：2026-08-16

## 合规与协议

实验在 zzai 独立目录 `/root/projects/t2_meta_calibrator_20260816/` 运行，只使用 Round 56 的 R1/R2 train-side 7 成员 OOF 预测和训练标签；没有读取 08、答案包或 holdout。轮次内结果使用 5 折训练折拟合，验证折不参与拟合。

## 方法

从 7 成员预测构造均值、中位数、标准差、最小/最大值、25/75 分位数，共 7 个元特征；比较 StandardScaler+Ridge(alpha=1/10) 与 StandardScaler+HuberRegressor。输出截断到 [0,9.9]。

## 结果

| 评估方向 | Ridge | Ridge10 | Huber |
|---|---:|---:|---:|
| R1 内 5 折 score | 156.56 | 156.09 | **158.57** |
| R2 内 5 折 score | 159.79 | 160.17 | **160.23** |
| R1→R2 score | 128.22 | **128.52** | 128.33 |
| R2→R1 score | 115.55 | 115.08 | **118.59** |

轮次内高分不能迁移；相比已冻结 Round 58 的跨域稳健方案，这些元校准器明显失败。

## 文献检索与方法依据

本轮通过 Crossref/API 检索并筛读与域泛化、类别不平衡、开放集识别、协变量偏移、无监督测试时适配和地震识别相关的候选，记录以下 18 篇作为后续方法边界（DOI 可复核）：

1. Simple Domain Generalization Methods are Strong Baselines for Open Domain Generalization — `10.1109/IJCNN60899.2024.10650639`
2. Domain Generalization Using Category Information Independent of Domain Differences — `10.5220/0013300300003905`
3. Adaptive variational sampling-embedded domain generalization under class imbalance — `10.1016/j.ress.2024.110707`
4. A multi-objective optimisation approach to class imbalance learning — `10.1016/j.patcog.2011.01.015`
5. KANBalance: Kolmogorov–Arnold network mitigates class imbalance — `10.1016/j.patcog.2025.112325`
6. Addressing class-imbalance and class-overlap by under-sampling — `10.1016/j.patcog.2023.109721`
7. Open-Set Text Recognition Implementations: Open-set Predictor — `10.1007/978-981-97-0361-6_6`
8. iCausalOSR: invertible causal disentanglement for open-set recognition — `10.31219/osf.io/j5fxz`
9. Open set recognition through Monte Carlo dropout-based uncertainty — `10.1504/IJBIC.2021.119982`
10. Label Shift Adapter for Test-Time Adaptation under Covariate and Label Shifts — `10.1109/ICCV51070.2023.01505`
11. Applications of Covariate Shift Adaptation — `10.7551/mitpress/8494.003.0012`
12. Realistic Evaluation of Test-Time Adaptation: Unsupervised Model Selection — `10.5220/0014320100004084`
13. Robust Online Test-Time Adaptation via a Multilayer Generative-Integrative Framework — `10.36227/techrxiv.176611667.71137127/v1`
14. Test-Time Adaptation for Personal Voice Activity Detection — `10.3390/electronics15143111`
15. Ensemble Calibration and Uncertainty Quantification for risk-based forecasting — `10.1002/essoar.10512511.1`
16. Improving Deep Learning-Based Seismic Phase Picking by Addressing Label Imbalance — `10.21203/rs.3.rs-10439246/v1`
17. PhaseMamba: a Mamba-based deep learning model for seismic phase picking — `10.1109/LGRS.2025.3603915`
18. Domain Adaptation and Domain Generalization with Representation Learning — `10.26686/wgtn.17014700`

共同结论是：域变化下的元校准必须有目标域证据和风险门控；只靠源轮次标签拟合的回归映射会把轮次先验硬编码，正是本轮跨轮失败的原因。

## 决策

否决“轮次内成员统计元校准器”作为跨轮发布方案。正式 T2 仍使用 Round 58 已验证配置；后续若继续，只研究不依赖标签的目标批次风险检测或保守校准，并在训练侧做跨轮留一验证。