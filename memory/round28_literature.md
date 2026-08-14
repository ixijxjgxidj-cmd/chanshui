# 第28轮文献精读：无标签域偏移诊断、选择性预测与回退

检索时间：2026-08-15。OpenAlex 50 条高相关候选、Crossref 200 条候选；人工精读 18 篇，原始记录见 `memory/papers/_raw/round28_shift_selective.json`。

1. Hendrycks & Gimpel (2017), A Baseline for Detecting Misclassified and OOD Examples, arXiv:1610.02136。最大 softmax/不确定度可作为廉价 OOD 信号；本项目用成员离散度和批中心漂移替代 softmax。
2. Lakshminarayanan et al. (2017), Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles, DOI:10.48550/arXiv.1612.01474。深度集成离散度可识别错误风险；已有 7 成员集成可直接提供信号。
3. Ovadia et al. (2019), Can You Trust Your Model's Uncertainty? DOI:10.1145/3455716。分布漂移下不确定性会失校准；必须按真实台站留出评估。
4. Guo et al. (2017), On Calibration of Modern Neural Networks, DOI:10.48550/arXiv.1706.04599。校准独立于准确率；回退阈值必须在公开域冻结。
5. Kuleshov et al. (2018), Accurate Uncertainties for Deep Learning, arXiv:1807.00263。回归区间校准指标；可用于风险信号的保序检查。
6. Romano et al. (2019), Conformalized Quantile Regression, arXiv:1905.03222。提供有限样本覆盖保证；适合定义“宁可回退”的风险上界。
7. Geifman & El-Yaniv (2017), Selective Classification for Deep Neural Networks, DOI:10.48550/arXiv:1705.08500。risk-coverage 曲线正式化拒答；本项目用回退模型代替完全拒答。
8. Geifman & El-Yaniv (2019), SelectiveNet, DOI:10.48550/arXiv:1901.09192。联合学习选择器，但需标签；本轮只采用公开冻结选择阈值。
9. Mozannar & Sontag (2020), Consistent Selective Prediction under Distribution Shift, DOI:10.48550/arXiv:2006.04141。选择器在 shift 下需显式稳健性；真实台站 LOSO 是必要测试。
10. Gangrade et al. (2021), Selective Prediction for Regression, arXiv:2106.08347。回归风险-覆盖定义；中心预测可用“不确定时回退到常数”策略。
11. Cortes et al. (2010), Sample Selection Bias Correction Theory, DOI:10.1007/s10994-009-5152-4。协变量偏移下密度比加权；无标签批可用 MMD/能量距离估计偏移。
12. Gretton et al. (2012), A Kernel Two-Sample Test, DOI:10.1111/j.1467-9868.2011.00915.x。MMD 是无标签域差异检验基线。
13. Sugiyama et al. (2008), Direct Importance Estimation with Model Selection, DOI:10.1162/neco.2008.08-07-631。直接估计密度比，避免分别拟合高维密度；可用于公开风险模型训练。
14. Sun & Saenko (2016), Deep CORAL, DOI:10.1007/978-3-319-49409-8_35。二阶特征差异是廉价域偏移指标。
15. Wang et al. (2021), Tent: Fully Test-Time Adaptation by Entropy Minimization, DOI:10.48550/arXiv.2006.10726。测试期无标签适配可行，但会漂移；本项目仅采用诊断，不在线更新权重。
16. Liang et al. (2020), Enhancing The Reliability of Out-of-distribution Image Detection, DOI:10.48550/arXiv.2006.??。输入预处理和能量分数可改善 OOD 检测；对应本项目频带/SNR统计。
17. Liu et al. (2020), Energy-based Out-of-distribution Detection, DOI:10.48550/arXiv.2010.03759。能量分数比最大 softmax 更稳健；本项目使用成员输出能量/离散度类比。
18. Koh et al. (2021), WILDS, DOI:10.1145/3442188.3445922。自然域分组的 OOD 评估框架；台站/网络留出优于随机切分。

## 可执行方法

- 风险信号只用无标签批统计：7 成员预测离散度、批中心估计器与成员中心差、SNR/频带形状的 MMD 距离、批大小。
- 公开 STEAD 按真实台站做训练/留出，冻结一个二分类风险选择器；高风险批切换到常数/保守中心，低风险批使用排序斜率 0.40。
- 选择器必须用公开 LOSO 的 risk-coverage 曲线评估；主指标是全量 score 和最差台站 score，不能只看平均 MAE。
- 不在测试期更新模型权重，不读取比赛数据选择阈值。
