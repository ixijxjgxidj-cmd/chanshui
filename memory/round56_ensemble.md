# Round 56：R1×1.75 七成员集成

日期：2026-08-16

## 合规

只使用 R1/R2 预注册 train 侧数据；未读取 holdout，未接触 08 数据。训练在远程独立目录 `/root/projects/round55_balance_20260816/` 进行。

## 结果

固定 Round 52 多任务预训练 B、R1 loss 权重 1.75/R2=1，7 个独立 seed，每个 seed 5 折 CV：

- 成员平均 raw score：134.46
- 成员平均 FT-center：**137.27**
- R1 FTC：139.13
- R2 FTC：135.42
- 相关系数：R1=.597，R2=.578
- 单成员 score 标准误：.99

Round 55 单成员 R1×1.75 为 FTC=139.64，Round 56 集成反而下降 2.37 分，主要来自 R2 中心偏移；成员方差不是主要瓶颈。七成员集成路线在当前中心估计方式下否决，不读取 holdout。

## 下一步

检索并精读跨域 ensemble calibration、group-wise calibration、hierarchical/partial-pooling calibration 文献。下一轮只改中心校准：在每个 CV fold 中用该 fold 的训练标签估计 round-specific intercept，验证是否能恢复 R2 中心；若诚实 CV 稳定超过 140，才预注册一次 holdout 读取。
