# 第 25 轮文献精读：跨域中心估计与域不变排序

检索时间：2026-08-15。检索源：OpenAlex（98 条候选）与 Crossref（200 条候选）。自动结果人工复核后，以下 18 篇作为方法依据；只读取元数据/摘要，不下载训练数据。

## 精读清单与可操作启示

1. Sun & Saenko (2016), Deep CORAL, arXiv:1607.01719。对齐源/目标特征协方差；本项目应只对齐公开 STEAD 的批统计，不把目标标签引入训练。
2. Sun et al. (2016), CORAL, DOI:10.1109/TPAMI.2016.2637299。二阶统计对齐简单且稳定；可作为无标签域适配基线。
3. Ganin et al. (2016), Domain-Adversarial Training, DOI:10.1007/s11263-015-0816-y。域分类器反向梯度；对本项目需防止去除震级相关信号，故只作小权重消融。
4. Long et al. (2015), DANN/Deep Adaptation Networks, DOI:10.1109/TPAMI.2015.2440132。多域嵌入与 MMD；适合公开台站留一域验证。
5. Gretton et al. (2012), Kernel Two-Sample Test, DOI:10.1111/j.1467-9868.2011.00915.x。MMD 的统计基础；用来监测域差异，不直接决定比赛参数。
6. Arjovsky et al. (2019), Invariant Risk Minimization, arXiv:1907.02893。跨环境不变预测；事件/台站作为环境，惩罚环境间最优线性头差异。
7. Peters et al. (2016), Causal Inference Using Invariant Prediction, DOI:10.1111/rssb.12158。要求跨环境稳定条件关系；支持留台站而非随机切分。
8. Gulrajani & Lopez-Paz (2021), In Search of Lost Domain Generalization, DOI:10.1073/pnas.2010837118。强 ERM 基线常胜过复杂 DG；本轮必须保留原 7 成员和简单批中心基线。
9. Zhao et al. (2019), On Learning Invariant Representations for Domain Adaptation, DOI:10.5555/3305381.3305576。指出对齐可能损害标签条件信息；中心/排序应分头学习。
10. Courty et al. (2017), Optimal Transport for Domain Adaptation, DOI:10.1007/s10994-016-5560-0。运输映射可处理多模态域差；可作为公开留域离线对照。
11. Shen et al. (2018), Wasserstein Distance Guided Representation Learning, DOI:10.1609/aaai.v32i1.11784。Wasserstein 约束稳定训练；比赛域无标签时仅用于训练内部域扰动。
12. Platt (1999), Probabilistic Outputs for SVMs, DOI:10.1023/A:1007608626950。后验校准思想；对应本项目的预测分布/不确定性校准。
13. Kuleshov et al. (2018), Accurate Uncertainties for Deep Learning, arXiv:1807.00263。回归分位/区间校准；用公开留域验证选择区间质量，而非比赛 holdout。
14. Romano et al. (2019), Conformalized Quantile Regression, NeurIPS。分位数 + conformal 覆盖；可报告批中心不确定区间，避免过度自信。
15. Zadrozny & Elkan (2002), Transforming Classifier Scores into Accurate Multiclass Probability Estimates, DOI:10.1145/775047.775151。单调校准不破坏排序；支持对成员分数做 isotonic/quantile mapping。
16. Tasche (2017), Calibrating the Binomial with Randomized Logistic Regression, DOI:10.1007/s10479-016-2364-3。先验/比例校准的风险；本项目用无标签分布估计中心时需显式记录假设。
17. Quadrianto et al. (2009), Estimating Labels from Label Proportions, DOI:10.1145/1553374.1553473。聚合标签学习；启发以批级预测分位估计目标中心，但不能假定标签分布形状固定。
18. du Plessis & Sugiyama (2014), Class Prior Estimation from Positive and Unlabeled Data, DOI:10.1007/s10994-014-5476-5。无标签先验估计；本项目采用 density-ratio/GBM 的稳健版本并做留域校验。

## 综合判断

- 最可靠路线是“排序头 + 批中心头”解耦：排序头用公开 STEAD 的随机增益和跨台站环境训练；中心头只接受无标签批统计。
- CORAL/IRM 作为小权重正则做消融，不直接替换 7 成员；文献明确警告过度对齐会损失标签条件信息。
- 目标批中心估计器候选：GBM（现有基线）、Huber/Quantile Ridge、分位数随机森林。训练批由 STEAD 按事件/台站分组模拟，留一台站验证。
- 预注册纳入标准：公开留一域验证中心 MAE 不劣于现有 GBM 且排序 Spearman 不下降；R1/R2 train-only 只作冻结后报告。任一 holdout、08 文件或其衍生产物均不读取。

## 本轮实验步骤

1. 从已缓存公开 STEAD 记录提取 7 成员预测及统计特征。
2. 按 source_id 分组构造 4 个伪目标批域；每批 50–260 条，统一增益偏移，保留震级分布真值仅作公开验证标签。
3. 训练三种中心估计器：GBM、Huber Ridge、Quantile Random Forest；五折按批分组。
4. 用留一伪域验证比较 MAE、中心误差与排序保持；冻结优胜者。
5. 仅在冻结后读取 R1/R2 预注册 train，生成一次外部报告。
