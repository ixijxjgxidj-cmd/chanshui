# Round 82：公开五区域蒸馏稳健性复核（否决）

## 合规范围

本轮只在 zzai 远程目录 `/root/projects/t2_public_kd_20260816/` 使用公开 STEAD。四个比赛包及其标签、波形、特征、预测、holdout 和衍生物均未读取，未用于训练、调参或候选筛选。

## 协议

沿用 Round 81 已修复的 `RmsNorm + LogAmp` 前端、Transformer 教师、轻量 CNN 学生和固定 `KD=0.25`（`0.75 hard L1 + 0.25 feature KD`）。扩展到五个事件级 A→B→C 设置：ALASKA→CALIF→GREECE、CALIF→GREECE→ALASKA、GREECE→CHILE→CALIF、CHILE→ALASKA→NZ、OTHER→NZ→CHILE。B 只观察，C 只评估，未根据外域结果修改权重。

## 结果

| 设置 | hard C | KD+hard C | 增量 |
|---|---:|---:|---:|
| ALASKA→CALIF→GREECE | 137.279 | 139.080 | +1.801 |
| CALIF→GREECE→ALASKA | 138.538 | 139.604 | +1.066 |
| GREECE→CHILE→CALIF | 137.403 | 137.847 | +0.444 |
| CHILE→ALASKA→NZ | 116.026 | 113.778 | −2.248 |
| OTHER→NZ→CHILE | 130.989 | 129.993 | −0.996 |

平均分数：hard `132.0471`，KD+hard `132.0604`，仅 `+0.0133`。最差分数：hard `116.0258`，KD+hard `113.7775`，下降 `−2.2483`。前三组改善、后两组下降，未满足预注册的“均值和 worst-case 均不下降”准入标准。

## 决策

**否决为稳健发布候选。** 该蒸馏权重对部分区域有益，但对 NZ 外域明显有害；不再继续在同一五组上扫描 KD 权重、温度或教师结构。保留为研究证据：跨区域地震回归的主要风险是区域先验偏移，单一教师软目标不能普适解决。

## 文献检索与精读

本轮 6 组 arXiv 查询得到 102 篇去重记录，保存于远程 `papers_round82.json`。精读候选至少 15 篇：Hinton et al. KD（1503.02531）；Gou et al. KD survey（2006.05525）；Mirzadeh teacher-assistant KD（1902.03393）；Beyer patient/consistent teacher（2106.05237）；Bagherzadeh triplet KD（2004.08116）；Lakshminarayanan deep ensembles（1612.01474）；Guo et al. calibration（1706.04599）；Mousavi & Beroza magnitude estimation（1911.05975）；Munchmeyer regional transfer（2101.02010）；STEAD（1810.10669）；INSTANCE（2101.06465）；SeisBench（2210.11114）；SeisLM（2410.15765）；地震基础模型综述（2503.24166）；seismic facies domain adaptation（2011.10510）；waveform inversion robustness（1809.10262）；seismic KD + domain adaptation（2506.22018）；uncertainty-aware regression distillation（相关 arXiv 候选）。

共同启示：蒸馏必须保留外域不确定性和区域条件；只蒸馏单一教师的中间表征会把教师的区域偏差传给学生；worst-case 外域比平均分更能发现灾难性退化；后续若重开，应做多教师 disagreement-aware KD 或显式区域不变表示，而不是继续扫描单一 KD 权重。

## 后续边界

T2 公开 KD 单教师路线暂时封存；Round 58 及其他比赛模型保持冻结。T3 仍不使用来源不明的五类公开伪标签。
