# Round 81：公开 STEAD JEPA 教师到学生蒸馏

## 合规范围

所有训练和检索在 zzai 远程机完成，独立目录 `/root/projects/t2_public_kd_20260816/`。只读取公开 STEAD；未读取、训练、调参、融合或筛选 08、R1、R2 及其任何衍生物。

## 预注册方案

三个事件级区域设置：ALASKA→CALIF→GREECE、CALIF→GREECE→ALASKA、GREECE→CHILE→CALIF。A 区域内按 source/event 划分训练与留出；B 只做观察，C 只做外域评估。教师为较大 Transformer，学生为轻量 CNN；KD+hard 固定损失 `0.25 * feature-KD + 0.75 * hard-L1`，不使用外域结果调权。对照为同学生结构的 hard-only。

## 实现审计

第一次运行暴露实现缺陷：教师/学生漏用了已验证的 `RmsNorm + LogAmp` 前端，造成跨区域绝对幅度尺度失配，C 域分数接近 0。该运行被标记为实现失败，不纳入结论。修复前端后重新运行同一协议，才作为有效结果。

## 修复后结果

| 设置 | hard C 分数 | KD+hard C 分数 | KD 增量 | hard rho | KD rho |
|---|---:|---:|---:|---:|---:|
| ALASKA→CALIF→GREECE | 135.885 | 137.358 | +1.473 | 0.3060 | 0.3070 |
| CALIF→GREECE→ALASKA | 138.538 | 139.605 | +1.067 | 0.0691 | 0.0764 |
| GREECE→CHILE→CALIF | 137.604 | 138.051 | +0.447 | 0.0200 | 0.0555 |

平均 C 分数：hard `137.342`，KD+hard `138.338`，增益 `+0.996`；最差 C 分数：hard `135.885`，KD+hard `137.358`，增益 `+1.473`。三个设置全部改善，满足公开外域准入门槛，但增益仍只在公开域验证，不能推断比赛域提升，未迁移到比赛模型。

## 文献检索与精读

本轮 arXiv 5 组查询得到 93 篇去重记录，保存于远程 `papers.json`。精读候选至少 15 篇：Hinton et al. 知识蒸馏（1503.02531）；Beyer et al. Knowledge distillation: A good teacher is patient and consistent（2106.05237）；Mirzadeh et al. Teacher assistant KD（1902.03393）；Bagherzadeh et al. Triplet KD（2004.08116）；Gou et al. Knowledge distillation survey（2006.05525）；Beyer et al. Knowledge distillation: A good teacher is patient and consistent（2106.05237）；Mousavi & Beroza 震级估计（1911.05975）；STEAD（1810.10669）；INSTANCE（2101.06465）；SeisBench（2210.11114）；SeisLM（2410.15765）；地震基础模型综述（2503.24166）；Seismic facies domain adaptation（2011.10510）；波形反演鲁棒性（1809.10262）；Seismic resolution enhancement with KD and domain adaptation（2506.22018）；深度集成不确定性（1612.01474）。

共同启示：回归蒸馏应同时保留 hard 标签与中间表征；教师必须在训练域内生成软目标，不能使用外域标签；跨域实验应报告 worst-case 而非只看均值；前端振幅归一化是地震跨区域迁移的关键实现细节。

## 决策

公开蒸馏路线保留为候选：KD+hard 在三个外域设置上平均和最差均改善。下一轮只能在更多公开区域上复核或做固定教师集成，不得使用比赛数据调节 KD 权重；T3 继续等待可靠公开五类标签。
