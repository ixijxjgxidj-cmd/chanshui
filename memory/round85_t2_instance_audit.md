# Round 85：公开 INSTANCE 独立验证审计（协议失败，不录取）

## 合规范围

本轮只读取公开 STEAD/INSTANCE 缓存，实验路径 `/root/projects/t2_public_instance_kd_20260816/`。运行时合规守卫阻断比赛波形路径；未读取、训练、调参或筛选 08、R1、R2。

## 目标

原计划是使用 STEAD 训练的模型直接评估 INSTANCE dev，检验多教师蒸馏是否只对 STEAD 网络划分有效。INSTANCE 缓存已审计：`X=(21209,3,1000)`，仅有官方 `train/dev`，接口拒绝 test split。

## 协议失败原因

复用旧跨数据集脚本时，自动替换没有覆盖其 `LAMS` 定义，日志仍显示 `lambda=0.00`。这会把跨域预测压成常数，产生虚假的 `score=200`，与上一轮已修订的 `lambda>=0.05` 约束不一致。该运行还只比较了旧 JEPA S-S/S-J，并非本轮多教师 disagreement-aware KD。

发现后立即停止进程；该运行没有用于任何模型选择、蒸馏权重选择或比赛决策。日志中的结果全部作废。

## 决策

本轮不产生有效的 INSTANCE 跨数据集分数。下一次若重开，必须使用独立脚本测试 `LAMS.min()>=0.05`，在运行前编译并用断言阻止零斜率；同时实现真正的多教师 KD，而不是复用旧 JEPA 对照脚本。在未完成这些门控前，不再声称跨数据集迁移成立。

## 文献检索

本轮 4 组 arXiv 查询得到 78 篇去重记录，保存于远程 `papers.json`。精读候选至少 15 篇：STEAD（1810.10669）、INSTANCE（2101.06465）、SeisBench（2210.11114）、Mousavi & Beroza 震级估计（1911.05975）、Munchmeyer 区域迁移（2101.02010）、SeisLM（2410.15765）、地震基础模型综述（2503.24166）、seismic facies domain adaptation（2011.10510）、waveform inversion robustness（1809.10262）、Hinton KD（1503.02531）、Gou KD survey（2006.05525）、Mirzadeh teacher assistant（1902.03393）、Beyer patient teacher（2106.05237）、Lakshminarayanan deep ensembles（1612.01474）、Guo calibration（1706.04599）、Kendall uncertainty weighting（1705.07115）。

主要启示：跨数据集评分必须先锁死评分网格和目标域盲评；任何 `lambda=0` 结果都不可信；公开域正信号不能替代协议正确的独立数据集验证。
