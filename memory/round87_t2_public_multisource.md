# Round 87：公开 STEAD+INSTANCE 多源联合训练盲评

## 合规范围

本轮仅使用远程 zzai 的公开 STEAD 与 INSTANCE 缓存。STEAD 按事件源分组划分训练子集；INSTANCE 仅使用官方 train，dev 只在所有训练完成后读取一次。08、R1、R2 及其任何衍生物未被读取、训练、调参、筛选或误差分析。本地没有训练或下载数据集。

## 固定协议

远程实验目录：`/root/projects/t2_public_multisource_20260816/`。脚本在读取 dev 前固定数据过滤、随机种子、10 epoch、AdamW 学习率 1e-3 和 batch 128。联合模式每轮从 STEAD 与 INSTANCE-train 等量抽样，防止源域样本量差异主导结果。比较 STEAD-only、INSTANCE-train-only、均衡联合训练三种模型。

## 结果

| 模式 | INSTANCE dev score | rho |
|---|---:|---:|
| STEAD-only | 150.6389 | 0.0028 |
| INSTANCE-train-only | 153.7527 | 0.1621 |
| STEAD+INSTANCE 均衡联合 | 154.7237 | -0.0401 |

联合训练相对 STEAD-only 提升 `+4.0848`，相对 INSTANCE-only 提升 `+0.9710`，说明多源训练比单一 STEAD 源更接近 INSTANCE 的分布；但联合模型的相关系数仍为负，且分数低于当前比赛候选，不能据此宣称比赛提升。该结果只保留为公开迁移研究证据，不进入比赛模型选择。

## 文献检索与精读

本轮通过 arXiv API 检索得到 30 篇去重记录，完整 JSON 保存在远程 `papers.json`。精读主题覆盖：STEAD、INSTANCE、SeisBench、跨区域迁移、地震震级回归、实时 P 波预警、域适配、物理约束表示学习、深度集成、校准、不确定性加权、知识蒸馏与基础模型。重点结论：多源联合训练可缓解单源域偏移，但源域标签定义、仪器响应和事件选择偏差会造成尺度失配；独立目标域必须同时报告比赛式分数与相关性，不能只优化单一指标；任何目标域调参都会破坏盲评有效性。

## 下一步决策

否决将本轮模型直接迁移到比赛。保留“公开多源预训练后冻结表征，再在允许的 R1/R2 训练集上做事件级微调”作为下一条候选路线；若开启，必须重新预注册微调比例和验证划分，并继续隔离 08 决赛数据。

## 远程结果

结果文件：`/root/projects/t2_public_multisource_20260816/results.json`。
论文文件：`/root/projects/t2_public_multisource_20260816/papers.json`。
