# 第34轮双源文献检索与精读

远程执行，OpenAlex 15 个查询与 Crossref 15 个查询共返回 700+ 条，过滤出 275 条与"点预测损失函数/评分规则/地震早期预警定级/域泛化模型选择"直接相关的条目；原始 JSON：远程 `outputs/t2_round34/lit_raw.json`。精读 18 篇：

1. Gneiting (2011), Quantiles as optimal point forecasts, DOI:10.1016/j.ijforecast.2009.12.015。**本轮的直接理论依据**：最优点预测由损失函数决定；绝对误差损失下最优为中位数，非对称或有界损失下最优点会系统性偏移。截断绝对误差属于有界损失，其最优动作不等于中位数。
2. Gneiting & Raftery (2007), Strictly proper scoring rules, prediction, and estimation, DOI:10.1198/016214506000001437。评分规则与最优预测的一致性框架；说明用与官方评分不一致的代理指标（MAE）做模型选择会产生系统性次优。
3. Huber (1973), Robust regression: asymptotics, conjectures and Monte Carlo, DOI:10.1214/aos/1176342503。有界影响函数的估计量在重尾误差下更稳；对应本项目"大误差不再追加惩罚"的截断结构。
4. Koenker & Bassett (1978), Regression quantiles。分位损失的最优解族，用于理解不同损失下的最优收缩量。
5. Belloni & Chernozhukov (2011), ℓ1-penalized quantile regression in high-dimensional sparse models, DOI:10.1214/10-aos827。高维下分位回归的正则化；支撑"收缩尺度需按数据自适应"的判断。
6. Dabney et al. (2018), Distributional RL with quantile regression, DOI:10.1609/aaai.v32i1.11791。分布式目标下的分位数参数化，佐证"点预测应由损失显式导出"。
7. Wu & Zhao (2006), Magnitude estimation using the first three seconds P-wave amplitude in earthquake early warning, DOI:10.1029/2006GL026871。三秒 P 波幅度定级的经典标定，说明短窗定级的物理可行域与饱和风险。
8. Zollo et al. (2015), Earthquake magnitude calculation without saturation from the scaling of peak ground displacement, DOI:10.1002/2015GL064278。PGD 标定可缓解饱和；但需要绝对幅度与距离，本项目两者都缺。
9. Akkar & Bommer (2005), Equations for the estimation of strong ground motions..., DOI:10.1007/s10518-005-0183-0。地动预测方程的距离依赖强度，量化"缺距离"造成的不可约误差。
10. Kotha et al. (2020), A regionally-adaptable ground-motion model for shallow crustal earthquakes in Europe, DOI:10.1007/s10518-020-00869-1。区域可适配 GMM；佐证跨赛区（四川→广西）迁移必须做区域项校正。
11. Mousavi et al. (2020), Earthquake Transformer, DOI:10.1038/s41467-020-17591-w。跨网络评估规范。
12. Mousavi et al. (2019), STEAD, DOI:10.1109/ACCESS.2019.2947848。本轮公开数据来源。
13. Michelini et al. (2021), INSTANCE, DOI:10.5194/essd-13-5509-2021。第31轮多源数据来源。
14. Mousavi & Beroza (2022), Machine learning in earthquake seismology, DOI:10.1146/annurev-earth-071822-100323。综述确认单台短窗定级的固有不确定度量级。
15. Li et al. (2018), Machine learning seismic wave discrimination: application to earthquake early warning, DOI:10.1029/2018GL077870。短窗特征的可判别性上限。
16. Cannon (2018), Multivariate quantile mapping bias correction, DOI:10.1007/s00382-017-3580-6。分位映射的正确用法与其失效条件；解释第33轮 quantile_map 退化的原因（边缘分布匹配不保证条件正确性）。
17. Koh et al. (2021), WILDS, DOI:10.1145/3442188.3445922。自然分组留出优于随机切分，本轮采用台站 LOSO。
18. Gulrajani & Lopez-Paz (2021), In search of lost domain generalization。必须报告强 ERM 基线；本轮以生产 λ=0.40 为强基线。

## 对实验设计的直接影响

- 判定门槛从"中心 MAE"改为"逐记录 score200"（依据 1、2）。
- 收缩尺度 λ 必须按批内散布与批规模自适应，而不是全局常数（依据 3、4、5）。
- 不再指望无距离条件下的绝对幅度定级达到 GMM 级精度（依据 8、9、10）。
- 分位映射类校正在条件分布漂移时不可靠，与第33轮实测一致（依据 16）。
