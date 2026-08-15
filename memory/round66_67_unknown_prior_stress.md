# Round 66–67：未知先验压力测试与保守部署边界

日期：2026-08-16

## 合规

只使用 R1/R2 train-side OOF 预测和训练标签生成合成未知先验；未读取 holdout，未接触 08。Round 58 的正式 holdout 结果不参与本轮选择。

## Round 66：固定强校准压力测试

在 R1-like、R2-like、low-heavy、high-heavy、bimodal、narrow-mid、uniform 七种合成未知先验上比较 alpha=0/.25/.5/.75/1.0。固定 alpha=.75 并不稳健：

- R2-like：143.15 → 152.42
- narrow-mid：157.40 → 164.91
- low-heavy：144.99 → 112.53
- bimodal：123.83 → 116.09
- R1-like：145.79 → 134.38

简单按预测中位数阈值门控也不能同时保护低震级和双峰先验，最差先验仍在 116–121。**否决固定 alpha=.75 和简单阈值门控。**

## Round 67：文献

检索 17 个主题得到 107 篇候选，精读 risk-controlled test-time adaptation、uncertainty-aware adaptation、conformal prediction under covariate shift、robust calibration、distribution shift detection 和区域地震迁移论文。共同原则是：未知域适配必须有不确定性/风险门控，强迁移只在域偏移证据充分时启用。

## 决策

正式发布配置保持 Round 58：7 成员、多任务预训练、R1/R2 权重 1.75/1.0、alpha=.25、已验证均值 149.29。Round 63 的 alpha=.75 仅作为研究候选，不进入无条件发布。未知 08 盲推理应使用保守 alpha=.25 或更小，并记录批次中心、IQR、成员分歧作为风险指标；不能用 08 结果反向调参。
