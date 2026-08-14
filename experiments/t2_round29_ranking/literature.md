# 第29轮文献精读：排序学习、序数回归与早期P波震级

检索时间：2026-08-15。OpenAlex 128 条、Crossref 200 条候选，原始记录在 `memory/papers/_raw/round29_ranking_ordinal.json`。人工精读 18 篇如下。

1. Herbrich et al. (2000), Large Margin Rank Boundaries for Ordinal Regression。阈值分解的序数目标；可避免把相邻震级当作同样错误。
2. Frank & Hall (2001), A Simple Approach to Ordinal Classification, DOI:10.1007/3-540-44816-0_13。K−1 二元阈值的稳定序数基线。
3. Niu et al. (2016), Ordinal Regression with Multiple Output CNN, DOI:10.1109/CVPR.2016.492。把连续等级转为有序二元任务；适合震级区间。
4. Cao et al. (2020), Rank Consistent Ordinal Regression, DOI:10.1016/j.patrec.2020.11.008。确保阈值概率单调；避免不一致序数输出。
5. Cheng et al. (2008), Neural Networks for Ordinal Regression, DOI:10.1109/TNN.2007.904712。成对/阈值排序损失的基础。
6. Burges et al. (2005), Learning to Rank using Gradient Descent, DOI:10.1145/1102351.1102363。RankNet pairwise logistic 损失；本轮采用间隔加权的成对比较。
7. Burges (2010), From RankNet to LambdaRank。排序指标对优化目标的影响；验证采用 Spearman/Pearson 排序而非只看 MAE。
8. Xia et al. (2008), Listwise Approach to Learning to Rank, DOI:10.1145/1390334.1390456。列表级损失更接近整体排序，但实现更复杂，作为后续路线。
9. Menon et al. (2013), On the Statistical Consistency of Algorithms for Binary Classification under Class Imbalance。pairwise 子采样的类别平衡注意点。
10. Kuleshov et al. (2018), Accurate Uncertainties for Deep Learning Regression。回归与排序可分开校准；绝对中心不宜反向驱动排序头。
11. Wu & Kanamori (2005), Rapid Assessment of Damage Potential, DOI:10.1111/j.1365-246X.2005.02585.x。早期 P 波震级信息有限且可能饱和；排序优于过度精确点估计。
12. Wu & Kanamori (2008), Development of an Earthquake Early Warning System using Real-Time Strong Motion Signals, DOI:10.1785/0120080032。P波早期指标的实际部署基础。
13. Kuyuk & Allen (2013), A Global Approach to Provide Earthquake Early Warning, DOI:10.1785/0220120142。跨区域早期震级估计及不确定性。
14. Kong et al. (2016), Rapid Earthquake Characterization using a Deep Learning Approach, DOI:10.1785/0220160089。深度网络用于快速源参数估计；强调数据域覆盖。
15. Mousavi et al. (2019), STEAD, DOI:10.1109/ACCESS.2019.2947848。公开训练域和事件级隔离来源。
16. Saabas et al. (2021), Improving Neural Network Prediction of Earthquake Ground Shaking with Transfer Learning, DOI:10.1093/gji/ggab488。跨域迁移时相对关系可能更稳。
17. Arjovsky et al. (2019), Invariant Risk Minimization, arXiv:1907.02893。跨环境不变排序可作为正则目标。
18. Koh et al. (2021), WILDS, DOI:10.1145/3442188.3445922。自然域留出必须评估最差域而非随机平均。

## 实验方案

- 新候选保持现有 CNN 输入、逐道 RMS 与随机增益增广；监督目标由绝对震级 L1 + pairwise logistic rank loss 组成。
- 成对样本只在震级差 ≥0.25 时计入，并分层抽样避免相邻噪声比较主导。
- 与纯 L1 基线在公开真实台站 LOSO 比较 Pearson、Spearman、最差台站相关性；排序为主指标，绝对中心仅作辅助。
- 不合格候选不读取比赛 train；合格才允许一次冻结 train-only 外部报告。
