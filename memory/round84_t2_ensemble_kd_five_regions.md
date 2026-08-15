# Round 84：多教师 disagreement-aware 蒸馏五区域闭环

## 合规范围

仅使用公开 STEAD，远程目录 `/root/projects/t2_public_ensemble_kd_20260816/`。四个比赛包及其任何标签、波形、特征、预测和衍生物均未读取或参与训练、调参、筛选。

## 协议

沿用 Round 83：三个固定种子教师 `1001/2001/3001`，教师仅在 A 域训练事件拟合；学生拟合教师均值特征；KD 权重固定为 `0.25`，并使用 `KD/(1+disagreement)` 门控。hard-only 学生为对照。B 只观察，C 只评估，事件级拆分，未根据 C 域改动参数。

本轮只复核上一轮未覆盖的两个困难设置：CHILE→ALASKA→NZ、OTHER→NZ→CHILE。

## 新增两组结果

| 设置 | hard C | ensemble-KD C | 增量 |
|---|---:|---:|---:|
| CHILE→ALASKA→NZ | 116.404 | 118.551 | +2.147 |
| OTHER→NZ→CHILE | 129.793 | 130.449 | +0.656 |

## 五组闭环汇总

与 Round 83 的前三组逐组合并：

- hard 平均/最差：`132.0031 / 116.4043`
- ensemble-KD 平均/最差：`132.9189 / 118.5508`
- 平均增益：`+0.9158`
- 最差增益：`+2.1465`
- 五组全部为正增益，满足预注册的均值与 worst-case 双门槛。

## 决策

**公开稳健候选通过。** 多教师 disagreement-aware KD 在五组公开区域外推上均改善，保留为后续研究的唯一蒸馏候选。但这不是比赛数据验证，不能据此调整或发布比赛模型；比赛包仍完全盲保留。下一轮若继续，只做独立公开数据源复核或固定教师集成，不扫描超参数。

## 文献检索与精读

本轮 5 组 arXiv 查询得到 96 篇去重记录，保存于远程 `papers_nz.json`。精读候选至少 15 篇：Hinton KD（1503.02531）；Gou KD survey（2006.05525）；Mirzadeh teacher assistant（1902.03393）；Beyer patient teacher（2106.05237）；Bagherzadeh triplet KD（2004.08116）；Lakshminarayanan deep ensembles（1612.01474）；Kendall uncertainty weighting（1705.07115）；Guo calibration（1706.04599）；Mousavi & Beroza magnitude estimation（1911.05975）；Munchmeyer regional transfer（2101.02010）；STEAD（1810.10669）；INSTANCE（2101.06465）；SeisBench（2210.11114）；SeisLM（2410.15765）；地震基础模型综述（2503.24166）；seismic facies domain adaptation（2011.10510）；waveform inversion robustness（1809.10262）。

共同启示：多教师均值降低教师特定区域偏差，disagreement 可作为软目标置信度门控；但必须使用事件级、区域级外域评估并报告最差域；公开候选仍不能替代比赛盲评。
