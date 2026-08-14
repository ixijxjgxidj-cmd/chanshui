# T2 第 24 轮：JEPA 自监督表征与集成纳入判定

## 合规边界

- 训练数据仅为公开 STEAD；JEPA 预训练与监督头训练均在远程 `/root/5.6+chanshui1` 完成。
- R1/R2 仅读取预注册 train 子集（各 140 条）作外部 train-only 参照；脚本断言 train 与 holdout 无交集。
- R1/R2 holdout 未在本轮读取；08-an、08-exam 及其衍生物未读取。
- 本轮结果不得用于反向选择冻结 holdout 参数。

## 方法

J1 使用 JEPA 风格遮挡预测：encoder 输出时序 token，context encoder 经遮挡后由 predictor 预测 EMA target encoder 的归一化 latent；随后以公开 STEAD 监督震级头微调。增广包括全局增益、时移和通道扰动，输入采用逐道 RMS 归一化。候选集成将 J1 与已冻结的 7 成员监督集成等权平均。

## 结果

公开 STEAD（M4–6.1）最佳 checkpoint：epoch 14，原始 score 147.46，中心化 score 150.84，r=0.389。

R1/R2 train-only 外部参照：

| 配置 | R1 r | R1 中心化分数 | R2 r | R2 中心化分数 | 最小 r | 两轮分数和 |
|---|---:|---:|---:|---:|---:|---:|
| 冻结 7 成员 | 0.4205 | 138.0481 | 0.3045 | 144.9798 | 0.3045 | 283.0279 |
| 7 成员 + J1 | 0.4182 | 137.9040 | 0.3051 | 145.0086 | 0.3051 | 282.9126 |

预注册纳入标准为「最小 r 不下降且两轮中心化分数和严格提高」。J1 使 R1 相关性下降 0.0023、两轮分数和下降 0.1153，因此判定：**不纳入主生产集成**。J1 保留为公开域负/互补实验资产，不删除其 checkpoint 与脚本。

## 可复现证据

- `J1_result.json`：公开验证 checkpoint 指标。
- `J1_external_trainonly.json`：R1/R2 train-only 外部结果。
- `t2_ens8_decision.json`：预注册集成判定及完整数值。
- `train_t2_jepa.py`、`t2_jepa_eval.py`、`t2_ens8_decide.py`：远程训练、评估与判定脚本副本。

## 下一轮建议

下一轮不读取冻结 holdout，优先研究公开域批级中心估计器的跨域稳健性（留一台站/地区验证、分位数校准和域不变排序），并继续保留 7 成员作为当前主集成。
