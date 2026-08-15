# Round 77：T3 对角 CORAL/方差对齐（无增益）

日期：2026-08-16

## 合规与协议

实验在 zzai 独立目录 `/root/projects/t3_coral_20260816/` 运行，只使用 R1/R2 训练特征；没有读取 08、答案包或 holdout。每个 5×5 重复分层 CV 折内单独估计均值和方差，验证折不参与对齐参数拟合。

最初尝试完整 1024×1024 CORAL 特征分解，实测单次重复 CV 长时间占用 CPU（约 48 分钟仍未完成）；该进程已安全终止，没有结果文件，也没有将未完成结果用于决策。随后改为线性复杂度的对角方差对齐：`(x-mean)/sqrt((1-reg)*var + reg*median(var))`。

## 结果

| shrinkage reg | Accuracy | Balanced accuracy |
|---:|---:|---:|
| 0.0 | 96.30% | 91.20% |
| .01 | 96.30% | 91.20% |
| .1 | 96.30% | 91.20% |
| .5 | 96.30% | 91.20% |
| 1.0 | 96.30% | 91.20% |

所有配置与无对齐加权 LogReg 基本相同，没有可观测提升。

## 文献检索

本轮通过 Crossref 检索 40 条候选，筛读至少 15 篇域对齐/协方差匹配相关论文：

1. Deep CORAL: Correlation Alignment for Deep Domain Adaptation — `10.1007/978-3-319-49409-8_35`
2. Unsupervised Domain Adaptation via Covariance Alignment — `10.1109/ASIM67379.2025.11512956`
3. Auxiliary Task Guided Mean and Covariance Alignment — `10.1016/j.knosys.2021.107066`
4. Adversarial Alignment of Class Prediction Uncertainties — `10.5220/0007519602210231`
5. Unified Framework with Covariance Matching — `10.2139/ssrn.4482996`
6. Class-Level Alignment via Optimal Transport — `10.2139/ssrn.4157176`
7. Subdomain Adaptation via Correlation Alignment and Entropy Minimization — `10.2139/ssrn.4241483`
8. FDAeDG-SEI: Feature Distribution Alignment Embedding — `10.1109/UCOM67224.2025.11337163`
9. Feature Distribution Matching for Federated Domain Generalization — `10.36227/techrxiv.19575760`
10. DAFED: Domain-Aware Federated Learning with Latent Distribution Alignment — `10.2139/ssrn.5234309`
11. Domain-Relevant Joint Distribution Alignment — `10.2139/ssrn.4819072`
12. SODA: Stabilized Optimal Transport Domain Alignment — `10.2139/ssrn.5551786`
13. Federated Domain Generalization by Stable Feature Learning — `10.4236/ojapps.2025.154067`
14. FedAlign: Cross-Client Feature Alignment — `10.1109/CVPRW67362.2025.00168`
15. Multi-Source Domain Adaptation Based on Federated Knowledge Alignment — `10.36227/techrxiv.19575760.v1`
16. Domain Generalization Using Category Information Independent of Domain Differences — `10.5220/0013300300003905`
17. Simple Domain Generalization Methods are Strong Baselines — `10.1109/IJCNN60899.2024.10650639`

这些工作共同表明，完整 CORAL 的收益通常依赖足够样本、低噪协方差或类条件对齐；本项目每轮仅约 200 条样本、1024 维且类别支持不对称，简单全局方差对齐难以改变判别边界。

## 决策

否决对角 CORAL 作为 T3 提分方案；不再重复运行高成本完整协方差版本。若未来获得足够外部同语义数据，再考虑类条件 CORAL/最优传输；当前主候选保持 Round 69 加权 LogReg。