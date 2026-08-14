# 第30轮文献精读：跨区域/跨仪器公开数据扩大

检索时间：2026-08-15。OpenAlex 179 条、Crossref 200 条候选；原始结果见 `memory/papers/_raw/round30_multisource.json`。人工精读 18 篇：

1. Michelini et al. (2021), INSTANCE – Italian seismic dataset for machine learning, DOI:10.5194/essd-13-5509-2021。意大利多台站、多震相公开波形，适合 STEAD 外区域迁移。
2. Mousavi et al. (2019), STEAD, DOI:10.1109/ACCESS.2019.2947848。全球多网络基础数据集；本项目现有主训练源。
3. Hsu et al. (2023), Curated Pacific Northwest AI-ready Seismic Dataset, DOI:10.26443/seismica.v2i1.368。区域化高质量台站数据，可用于自然域留出。
4. Münchmeyer et al. (2021), Which Picker Fits My Data?, DOI:10.1029/2021JB023499。跨数据集评估显示域差异显著，必须保留独立区域验证。
5. Saabas et al. (2021), Transfer learning for earthquake ground shaking, DOI:10.1093/gji/ggab488。迁移学习在新区域有效，但需防止目标域标签泄漏。
6. Zhu & Beroza (2019), PhaseNet, DOI:10.1093/gji/ggy423。大规模公开跨台站训练可提升泛化。
7. Mousavi et al. (2020), Earthquake Transformer, DOI:10.1038/s41467-020-17591-w。多任务深度模型的跨网络验证规范。
8. Perol et al. (2018), Convolutional neural network for earthquake detection, DOI:10.1038/s41598-018-29387-4。区域数据可显著影响模型决策边界。
9. Wu et al. (2023), Local earthquakes detection benchmark, DOI:10.1016/j.aiig.2020.04.001。三分量数据集基准，支持跨区域检测/表征。
10. Quinteros et al. (2023), PickBlue OBS dataset, DOI:10.1029/2023EA003332。仪器类型变化造成显著域偏移；可作为未来 OBS 扩展。
11. Münchmeyer et al. (2022), QuakeFlow, DOI:10.1093/gji/ggac355。大规模流水线可扩展公开波形获取与质量控制。
12. Mousavi et al. (2024), SeisLM foundation model, DOI:10.48550/arXiv.2410.15765。跨任务预训练是多源数据利用方向，但当前竞赛窗口需先验证基础迁移。
13. Zhu et al. (2022), Global and local representations for seismic phase detection, DOI:10.1093/gji/ggad270。局部与全局特征结合可缓解区域差异。
14. Kong et al. (2016), Rapid earthquake characterization with deep learning, DOI:10.1785/0220160089。快速源参数估计需要跨区域独立测试。
15. Warden (2018), SeisBench/benchmark principles。标准化数据接口和固定 split 便于公平比较。
16. Koh et al. (2021), WILDS, DOI:10.1145/3442188.3445922。自然分组 OOD 评估优于随机切分。
17. Gulrajani & Lopez-Paz (2021), In Search of Lost Domain Generalization, DOI:10.1073/pnas.2010837118。强 ERM 多源基线不可跳过。
18. Ovadia et al. (2019), Can You Trust Your Model’s Uncertainty?, DOI:10.1145/3455716。跨域不确定性校准必须独立验证。

## 实验方案

- SeisBench 公开数据接口下载 INSTANCE；只保留 P 后至少 5 秒的三分量窗口，统一为 P±5 秒。
- 训练候选：STEAD 单源、STEAD+INSTANCE 多源混合；随机增益、逐道 RMS 和事件级 split 保持一致。
- 真实区域留出：整站/整事件留出，不用比赛数据选择超参。
- 只有多源候选在公开 INSTANCE 留出与 STEAD 台站 LOSO 同时不退化，才进入冻结 train-only 报告。
