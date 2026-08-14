# T2 第29轮：Pairwise 排序目标重构证伪

## 合规

- 只在远程公开 STEAD 可信台站缓存上训练、验证。
- 本轮未读取 R1/R2 train、holdout、08-an、08-exam 或任何衍生产物；未在本地训练或下载数据。

## 文献与方法

OpenAlex 128 条、Crossref 200 条候选中人工精读 18 篇，详见 `literature.md`。基于 RankNet、序数回归、早期 P 波震级和自然域泛化文献，尝试在原有逐道 RMS + 随机增益增强 CNN 上加入 pairwise logistic ranking loss。

损失为异方差 L1 加 `0.4 × pairwise logistic loss`。只对同批震级差 ≥0.25 的上三角样本对计算排序项，避免相邻震级标签噪声主导。使用真实台站血缘的 8 个公开台站，训练 1,322 条、验证 1,311 条。

## 公开真实台站验证

| 候选 | Pearson r | Spearman rho | MAE |
|---|---:|---:|---:|
| 纯异方差 L1 基线 | **0.5352** | **0.4758** | **0.2697** |
| L1 + pairwise rank loss | 0.4997 | 0.4264 | 0.3123 |

pairwise rank loss 同时降低 Pearson、Spearman 并恶化 MAE，未通过“排序主指标不下降”的预注册条件。

## 判定

不冻结、不纳入 7 成员、不读取比赛 train。该负结果表明，在当前样本规模和 P±5 秒波形信息下，额外 pairwise 项可能放大台站/路径耦合，而非提取可迁移震级排序。后续应优先扩大公开多区域训练数据或改用更强自监督表征，而不是继续扫描 pairwise 权重。
