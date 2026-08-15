# Round 65：发布推理边界与入口审计

检查发现仓库旧 `t2_predict.py` 属于 Round 45 的 M1–M7 + 旧中心估计器，不能直接用于 Round 58 发布权重；Round 58 权重是 `NetMTL` 结构，存放在远程 `/root/projects/round58_release_20260816/weights/`。

因此正式发布时必须区分：

- Round 58 已验证配置：7 个 `NetMTL` 权重、R1/R2 权重 1.75/1.0、alpha=.25、R1 slope=.85、R2 slope=.50；这是唯一有 holdout 149.29 证据的配置。
- Round 63 无标签候选：根据当前预测批次中位数映射中心，alpha=.75；只有训练侧证据，不能宣称新的 holdout 分数。

本轮没有读取任何答案文件或 holdout。Round 58 远程权重、配置和 SHA256 保持不变。训练侧拟合的映射系数已保存到 `outputs/t2_round63/mapping.json`，仅供未知域候选实验，不得覆盖正式配置。
