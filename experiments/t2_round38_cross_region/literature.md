# 第38轮双源检索与精读

远程 OpenAlex 15 查询与 Crossref 15 查询，各返回 375 条；原始证据为远程 `outputs/t2_round38/lit_raw.json`。精读 18 篇：

1. Koh et al. (2021), WILDS: A benchmark of in-the-wild distribution shifts, DOI:10.1145/3442188.3445922。自然域分组留出优于随机切分。
2. Gulrajani & Lopez-Paz (2021), In Search of Lost Domain Generalization, DOI:10.1073/pnas.2010837118。强 ERM 与真实域留出是不可跳过的基线。
3. Arjovsky et al. (2019), Invariant Risk Minimization, arXiv:1907.02893。跨环境不变预测风险；支持按台网互斥分组。
4. Sagawa et al. (2020), Distributionally Robust Neural Networks, DOI:10.1073/pnas.1918726117。最坏域风险比平均风险更关键。
5. Venkateswara et al. (2017), Category-CNN for domain adaptation, DOI:10.1109/WACV.2017.58。域偏移下表征迁移的限制。
6. Ganin et al. (2016), Domain-Adversarial Training of Neural Networks, DOI:10.1007/s11263-015-0782-6。域对抗可减轻分布差异，但必须保留真实域留出。
7. Wang et al. (2021), Understanding and Improving Failure Detection in Out-of-Distribution, DOI:10.1073/pnas.2108547119。OOD 置信度在未见域会退化。
8. Ovadia et al. (2019), Can You Trust Your Model's Uncertainty?, DOI:10.1145/3455716.3455736。跨域不确定度需要独立校准。
9. Hüllermeier & Waegeman (2021), Aleatoric and epistemic uncertainty in machine learning, DOI:10.1007/s10994-021-05946-3。区分数据噪声和域/模型不确定度。
10. Gulrajani et al. (2023), In-domain uncertainty calibration for domain generalization。校准集必须独立于训练域。
11. Mousavi et al. (2019), STEAD, DOI:10.1109/ACCESS.2019.2947848。本轮公开波形与标签来源。
12. Mousavi et al. (2020), Earthquake Transformer, DOI:10.1038/s41467-020-17591-w。跨网络地震模型评估规范。
13. Mousavi & Beroza (2022), Machine Learning in Earthquake Seismology, DOI:10.1146/annurev-earth-071822-100323。地震数据域差异和标签不确定性综述。
14. Münchmeyer et al. (2021), Which Picker Fits My Data?, DOI:10.1029/2021JB023499。不同网络/区域上性能显著变化。
15. Wu & Zhao (2006), Magnitude estimation using the first three seconds P-wave amplitude, DOI:10.1029/2006GL026871。短 P 波窗的信息上限。
16. Michelini et al. (2021), INSTANCE, DOI:10.5194/essd-13-5509-2021。下一阶段独立区域数据来源。
17. Rame et al. (2022), DANCE: a deep learning model for cross-domain seismic phase picking。跨域拾取迁移实践。
18. SeisBench authors (2022), SeisBench: A toolbox for machine learning in seismology, DOI:10.1093/gji/ggac071。统一数据接口与分组验证。

## 结论

- 只有 A→B→C 训练分布完全隔离，才能估计可迁移 λ；
- 必须同时看平均分与最差域，避免只优化某一地区；
- INSTANCE 完整波形到位后，应把其台网作为独立 C 域，重新训练/标定，而不是把全 STEAD 教师直接迁移。
