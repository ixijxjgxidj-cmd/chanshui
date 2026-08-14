# 第27轮文献精读：分层混合效应与收缩式台站校正

检索时间：2026-08-15。检索源：OpenAlex 109 条、Crossref 200 条（原始记录见 `memory/papers/_raw/round27_hierarchical.json`）。自动相关性排序对本主题噪声较高，故按方法相关性人工筛选并精读下列 18 篇。

1. Abrahamson & Youngs (1992), A Stable Algorithm for Regression Analyses Using the Random Effects Model, DOI:10.1785/BSSA0820010505。地震动回归的随机效应标准解法；台站/事件项应作为随机效应而非逐组自由参数。
2. Joyner & Boore (1993), Methods for Regression Analysis of Strong-Motion Data, DOI:10.1785/BSSA0830020469。两阶段与随机效应比较；说明分层可显著降低组内相关造成的偏差。
3. Al Atik et al. (2010), The Variability of Ground-Motion Prediction Models, DOI:10.1785/gssrl.81.5.794。把总方差分解为事件间、台站间与残差；直接对应本项目“批中心 vs 排序”分解。
4. Rodriguez-Marek et al. (2011), Analysis of Single-Station Standard Deviation, DOI:10.1785/0120100252。单台站方差显著低于总方差；未知台站的系统项是主要不确定性来源。
5. Lin et al. (2011), Repeatable Source, Site, and Path Effects, DOI:10.1785/0120090031。台站项可重复且可估计，但需该台站样本；对未知台站必须退化为收缩到总体均值。
6. Stafford (2014), Crossed and Nested Mixed-Effects Approaches, DOI:10.1785/0120130145。交叉随机效应（事件×台站）实现方式；支持本轮的双层设计。
7. Bates et al. (2015), Fitting Linear Mixed-Effects Models Using lme4, DOI:10.18637/jss.v067.i01。混合效应实现与收敛诊断参考。
8. Gelman & Hill (2007), Data Analysis Using Regression and Multilevel/Hierarchical Models, ISBN 9780521686891。部分池化与收缩的标准处理；未见组的最佳预测是总体均值。
9. Efron & Morris (1975), Data Analysis Using Stein's Estimator, DOI:10.1080/01621459.1975.10479864。收缩估计优于逐组极大似然；直接支持“对小样本台站强收缩”。
10. James & Stein (1961), Estimation with Quadratic Loss。收缩的理论根据；解释为何全局+收缩优于逐台站自由拟合。
11. Robbins (1956), An Empirical Bayes Approach to Statistics。经验贝叶斯框架；用于从公开台站集估计台站项方差。
12. Rao & Molina (2015), Small Area Estimation, DOI:10.1002/9781118735855。小样本域估计；与“每个目标批只有一个台站”高度类似。
13. Morris (1983), Parametric Empirical Bayes Inference, DOI:10.1080/01621459.1983.10477920。方差成分估计与区间；用于给中心估计加不确定度。
14. Huber (1964), Robust Estimation of a Location Parameter, DOI:10.1214/aoms/1177703732。稳健位置估计；批中心应对离群预测稳健。
15. Koenker & Bassett (1978), Regression Quantiles, DOI:10.2307/1913643。分位回归；用于中心与分布形状分离建模。
16. Kuleshov et al. (2018), Accurate Uncertainties for Deep Learning Regression, arXiv:1807.00263。回归校准须在留域上检验。
17. Romano et al. (2019), Conformalized Quantile Regression, arXiv:1905.03222。可给出留域覆盖保证的区间。
18. Koh et al. (2021), WILDS, DOI:10.1145/3442188.3445922。自然分组评估协议；本轮延续真实台站留出。

## 关键结论

- 对**未见台站**，理论最优不是拟合台站项，而是把台站项收缩到公开总体均值；因此“全局模型 + 收缩”应当是强基线，而逐台站自由校正在目标域不可用。
- 方差分解（事件间/台站间/残差）说明：能跨域迁移的是排序与相对结构，绝对中心受台站项支配。这与项目既有证据一致。
- 因此本轮实验对象是：全局 GBM（现状）、加入台站随机效应后再收缩、经验贝叶斯收缩强度选择、稳健 Huber 中心，以及分位数中心。
- 判定仍以真实台站留出为主门槛，并要求最差台站误差不恶化，避免只优化平均值。
