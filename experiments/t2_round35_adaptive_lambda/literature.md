# 第35轮双源文献检索与精读

远程执行 OpenAlex 15 查询 + Crossref 15 查询，各返回 375 条，过滤出 365 条与收缩估计/不确定度/域偏移/集成/地震定级相关条目；原始 JSON：远程 `outputs/t2_round35/lit_raw.json`。精读 17 篇：

1. Stein (1956) / James & Stein (1961), Estimation with quadratic loss。收缩估计的原始结果：在多参数问题中向中心收缩严格优于无偏估计。本轮 `center + λ·z` 正是收缩形式，λ<1 即收缩强度。
2. Efron & Morris (1975), Data analysis using Stein's estimator and its generalizations, DOI:10.1080/01621459.1975.10479864。经验贝叶斯收缩量的可估计形式 `1 − σ²/(σ²+τ²)`，直接对应本轮 `λ* ≈ ρ·σ_y/σ_z`。
3. Rao & Molina, Small Area Estimation。分层收缩在小样本组上的标准做法；解释为何逐台站自由项不可迁移（第27轮）而全局比率可迁移。
4. Ge et al. (2019), Polygenic prediction via Bayesian regression and continuous shrinkage priors, DOI:10.1038/s41467-019-09718-5。连续收缩先验在高维弱信号下的稳健性。
5. Piironen & Vehtari (2017), Sparsity information and regularization in the horseshoe and other shrinkage priors, DOI:10.1214/17-EJS1337SI。收缩强度应由信号-噪声比决定，而非固定常数。
6. Sendur & Selesnick (2002), Bivariate shrinkage with local variance estimation, DOI:10.1109/LSP.2002.806054。**与本轮机制最接近**：收缩系数由局部方差估计驱动，而不是全局常数。
7. Donoho & Johnstone (1998), Minimax estimation via wavelet shrinkage, DOI:10.1214/aos/1024691081。方差自适应阈值的极小极大最优性。
8. Gneiting (2011), Quantiles as optimal point forecasts, DOI:10.1016/j.ijforecast.2009.12.015。点预测最优性由损失决定，支撑第34轮的目标对齐。
9. Gneiting & Raftery (2005), Calibrated probabilistic forecasting using ensemble model output statistics and minimum CRPS estimation, DOI:10.1175/MWR2904.1。用集成散布预测误差尺度的标准框架；本轮用 σ_z 作为可观测散布同源。
10. Raftery et al. (2005), Using Bayesian model averaging to calibrate forecast ensembles, DOI:10.1175/MWR2906.1。集成校准需要独立验证集，本轮用台站 LOSO。
11. Hüllermeier & Waegeman (2021), Aleatoric and epistemic uncertainty in machine learning, DOI:10.1007/s10994-021-05946-3。区分不可约与可约不确定度；σ_y 属前者，决定 λ 的上界。
12. Abdar et al. (2021), A review of uncertainty quantification in deep learning, DOI:10.1016/j.inffus.2021.05.008。集成散布作为不确定度代理的适用条件。
13. Gawlikowski et al. (2023), A survey of uncertainty in deep neural networks, DOI:10.1007/s10462-023-10562-9。域外条件下不确定度校准退化，故 ρ 必须跨台站验证。
14. Wu & Zhao (2006), Magnitude estimation using the first three seconds P-wave amplitude, DOI:10.1029/2006GL026871。短窗定级的物理不确定度量级。
15. Mousavi et al. (2019), STEAD, DOI:10.1109/ACCESS.2019.2947848。本轮公开数据来源。
16. Koh et al. (2021), WILDS, DOI:10.1145/3442188.3445922。自然分组留出协议。
17. Gulrajani & Lopez-Paz (2021), In search of lost domain generalization。强基线必须报告；本轮同时报告 `fixed040` 与 `fixed_public` 两条基线。

## 对实验设计的直接影响

- λ 应由局部方差比驱动（依据 2、5、6、7），而不是全局常数；这正是 `adaptive_ratio` 的形式。
- 优先选择有机制约束的比率式而非黑箱回归 λ（依据 13：域外校准会退化）。
- σ_y 用批级分位模型估计而非用预测方差替代（依据 11：两类不确定度不可混用）。
