# Round 83：多教师 disagreement-aware 蒸馏（条件候选）

## 合规范围

仅使用远程公开 STEAD，实验目录 `/root/projects/t2_public_ensemble_kd_20260816/`。四个比赛包及所有衍生物均未读取或参与选择。

## 方案

固定三个教师种子 `1001/2001/3001`，每个教师只在 A 区训练事件上拟合；学生拟合三教师均值特征。KD 权重固定 `0.25`，并用教师预测 disagreement 做 `KD/(1+disagreement)`，不从外域调参。对照为同学生结构 hard-only。

## 结果（已真正使用 disagreement 权重）

| 设置 | hard C | ensemble-KD C | 增量 |
|---|---:|---:|---:|
| ALASKA→CALIF→GREECE | 137.678 | 139.197 | +1.519 |
| CALIF→GREECE→ALASKA | 138.538 | 138.735 | +0.197 |
| GREECE→CHILE→CALIF | 137.602 | 137.662 | +0.061 |

平均：`137.9393 → 138.5315`（+0.5922）；最差：`137.6016 → 137.6623`（+0.0607）。三组全部改善。此前一次集成运行没有使用 disagreement 权重，仅作为多教师均值蒸馏对照，不纳入本结论。

## 决策

保留为**条件候选**，不发布、不迁移到比赛模型。理由：三组外域结果均改善，但尚未覆盖 NZ 相关的两个困难设置；下一轮只做一次预注册的剩余两组复核，不再扫描教师数量、KD 权重或温度。若剩余两组任一 worst-case 下降，则路线封存。

## 文献检索与精读

本轮 5 组 arXiv 查询得到 87 篇去重记录，保存于远程 `papers.json`。精读候选至少 15 篇：Hinton KD（1503.02531）；Gou KD survey（2006.05525）；Mirzadeh teacher assistant（1902.03393）；Beyer patient teacher（2106.05237）；Bagherzadeh triplet KD（2004.08116）；Lakshminarayanan deep ensembles（1612.01474）；Guo calibration（1706.04599）；Kendall uncertainty weighting（1705.07115）；Mousavi & Beroza magnitude estimation（1911.05975）；Munchmeyer regional transfer（2101.02010）；STEAD（1810.10669）；INSTANCE（2101.06465）；SeisBench（2210.11114）；SeisLM（2410.15765）；地震基础模型综述（2503.24166）；seismic facies domain adaptation（2011.10510）；waveform inversion robustness（1809.10262）。

共同启示：教师 disagreement 可作为蒸馏置信度门控，但必须在未见区域报告 worst-case；多教师平均能降低单一区域偏差，却不能假设对所有台网有效；固定的事件级外域协议比在目标域调权更重要。
