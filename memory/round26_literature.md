# 第26轮文献精读：真实台站域泛化与场地项

检索时间：2026-08-15。检索源：OpenAlex 131 条、Crossref 200 条。以下 18 篇经题目、摘要、引用信息与方法相关性人工筛选，用于本轮“真实台站留出”设计。

1. Mousavi et al. (2019), STEAD, DOI:10.1109/ACCESS.2019.2947848。全球多网络波形/元数据；支持按网络和台站作真实域划分。
2. Mousavi et al. (2020), Earthquake Transformer, DOI:10.1038/s41467-020-17591-w。注意力网络跨数据源应用，但需要显式独立测试；本轮保留台站留出。
3. Zhu & Beroza (2019), PhaseNet, DOI:10.1093/gji/ggy423。说明跨台站泛化与训练域覆盖是相位模型关键；不能用随机记录切分替代。
4. Ross et al. (2018), Generalized Seismic Phase Detection, DOI:10.1785/0220180080。跨区域训练有益但对仪器域敏感；启发网络/台站分层报告。
5. Hutton & Boore (1987), The ML scale in southern California, DOI:10.1785/BSSA0770030207。局地震级须有台站校正；绝对振幅项不能假定跨站固定。
6. Boore (2003), Simulation of Ground Motion, DOI:10.1785/0120020195。场地响应、路径与震源效应耦合；以单一振幅反演绝对震级有不可辨识性。
7. Atkinson & Boore (1995), New Ground Motion Relations, DOI:10.1785/BSSA0850010017。不同区域与场地的系统偏置真实存在；应使用域留出而非仅增益增强。
8. Cotton et al. (2006), Site Effects, DOI:10.1785/0120050065。地震台站放大效应频率依赖；本轮批特征加入谱形而非仅幅度。
9. Hanks & Kanamori (1979), A Moment Magnitude Scale, DOI:10.1029/JB084iB05p02348。震级的物理定义与观测链路分离；支持排序/中心双头而非端到端绝对值假设。
10. Wu & Kanamori (2005), Rapid Assessment of Damage Potential, DOI:10.1111/j.1365-246X.2005.02585.x。P波早期信息对大震有饱和和不确定性；需要报告置信度。
11. Satriano et al. (2011), Real-Time Earthquake Location, DOI:10.1111/j.1365-246X.2011.04912.x。实时场景应处理台站异质性和可用台站集合变化。
12. Sun & Saenko (2016), Deep CORAL, DOI:10.1007/978-3-319-49409-8_35。二阶统计对齐可实现无标签域适配；只作为公开训练域正则。
13. Ganin et al. (2016), Domain-Adversarial Training, DOI:10.1007/s11263-015-0816-y。域分类对抗能降低域可分性，但可能影响任务信号；须以真实台站留出检验。
14. Arjovsky et al. (2019), Invariant Risk Minimization, arXiv:1907.02893。跨环境不变关系；真实台站是比事件哈希更可信的环境定义。
15. Gulrajani & Lopez-Paz (2021), In Search of Lost Domain Generalization, DOI:10.1073/pnas.2010837118。强 ERM 常是竞争基线；复杂域泛化不得绕过基线。
16. Kuleshov et al. (2018), Accurate Uncertainties for Deep Learning, arXiv:1807.00263。回归预测需在留域上检验校准，防止不确定性只在随机验证有效。
17. Romano et al. (2019), Conformalized Quantile Regression, arXiv:1905.03222。可为中心估计提供留域覆盖区间，不让模型以不可信的点估计取代回退。
18. Koh et al. (2021), WILDS, DOI:10.1145/3442188.3445922。现实分布漂移必须以自然分组而非 i.i.d. 切分评估；本轮直接采用 station/network 组。

## 可执行结论

- 真实台站留出是本轮唯一模型选择准则；不使用比赛 train/holdout 做成员或超参选择。
- 先在已缓存 STEAD 记录中以 source_id 回接站点元数据；仅保留样本数、震级跨度足够的台站。
- 以“留一台站”和“留一网络”双层指标评估 GBM/ExtraTrees/HGB/Ridge，输入只能为 7 成员预测、预测离散度、SNR 和频谱形状的无标签批统计。
- 只有当新模型在真实台站 LODO 上优于 v2 GBM，且在网络 LODO 无退化时，才冻结候选。随后最多一次 R1/R2 预注册 train-only 报告。
