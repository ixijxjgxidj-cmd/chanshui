# Round 62：跨轮次先验迁移风险（LORO）

日期：2026-08-16

## 合规

只使用 Round 56 的 R1/R2 train-side OOF 预测和训练标签；未读取 holdout，未接触 08。Round 58 的一次性 holdout 结果不参与本轮选择。

## LORO 结果

测试“把源轮次的训练中位数作为目标轮次部分池化先验”：

- R1 prior → R2 target：alpha=.25 得分 111.95，alpha=.50 得分 86.77；不校准（alpha=0）为 135.42。
- R2 prior → R1 target：alpha=.25 得分 123.01，alpha=.50 得分 99.38；不校准为 139.13。

跨轮次直接迁移先验会严重恶化，说明 R1/R2 的先验分布不同，不能将另一轮次标签中位数当作未知 08 的先验。

## 文献

本轮检索 15 个主题得到 102 篇候选，精读 covariate-shift 无监督校准、domain-generalization calibration、test-time adaptation、label/prior shift 和区域地震迁移论文。文献与实验证据一致：未知目标域只能使用目标批次无标签统计做保守适配，不能无条件套用源域标签先验。

## 发布边界

Round 58 的 149.29 是已预注册配置在一次 holdout 上的正式结果，配置假设可获得对应轮次的训练侧先验。alpha=.75 虽在训练侧 bootstrap 上更优，但跨轮次 LORO 不稳，不能作为通用替代。正式候选和其 alpha=.25 设置保持不变；后续若面对真正无标签未知域，应优先使用目标批次预测中位数与保守 shrinkage，并将结果标记为未验证候选。
