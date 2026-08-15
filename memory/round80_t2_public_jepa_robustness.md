# Round 80：公开 STEAD 跨区域 JEPA 稳健性复核

## 合规与数据

本轮训练仅在 zzai 远程机 `/root/projects/t2_public_jepa_seed_20260816/s1/` 完成。输入是公开 STEAD 的 26,108 条 P 波窗口，按网络区域和 source/event 分组；运行时 `compliance_guard` 明确阻断比赛路径。未读取、训练、调参或筛选任何 08、R1、R2 数据及衍生物。

## 协议

五个公开 A→B→C 区域设置：ALASKA→CALIF→GREECE、CALIF→GREECE→ALASKA、GREECE→CHILE→CALIF、CHILE→ALASKA→NZ、OTHER→NZ→CHILE。A 内再按事件划分 train/holdout；B 只用于校准，C 只评估一次。比较同一 Transformer 编码器的监督 scratch 与 A 域 JEPA 预训练；`PYTHONHASHSEED=17` 固定事件划分；斜率网格从 `0.05` 开始，排除常数预测伪优解。

## 结果

| 指标 | scratch | JEPA | 增量 |
|---|---:|---:|---:|
| 平均 rho（评估域） | 0.320118 | 0.341372 | +0.021254 |
| 平均校准分数 | 159.061223 | 159.105548 | +0.044325 |
| 最差校准分数 | 156.747086 | 157.194730 | +0.447643 |
| 平均 rho 衰减 | 0.805002 | 0.815930 | +0.010928 |

五组中 JEPA 平均与 worst-case 均不低于 scratch，满足本轮公开数据的稳健性门槛。但平均分数增益仅 +0.044，不能证明对比赛域有足够收益，因此不迁移到比赛数据、不改变 Round 58 冻结候选。

## 论文检索与精读（15 篇以上）

远程 arXiv 检索 7 组查询共得到 113 篇去重记录，重点阅读以下工作：

1. Mousavi & Beroza, *A Machine-Learning Approach for Earthquake Magnitude Estimation*（1911.05975）。
2. STEAD, Mousavi et al.（1810.10669）。
3. INSTANCE, Michelini et al.（2101.06465）。
4. SeisBench, Woollam et al.（2210.11114）。
5. SeisLM: A Foundation Model for Seismic Waveforms（2410.15765）。
6. Foundations Models for Seismic Data Processing: An Extensive Review（2503.24166）。
7. Xi-Net: Transformer Based Seismic Waveform Reconstructor（2406.16932）。
8. Var-JEPA: Variational JEPA（2603.20111）。
9. CF-JEPA: Mask-free forward prediction for time series（2606.07031）。
10. JEPA 原始方法（2301.08243）。
11. BYOL 自监督表征（2006.07733）。
12. Seismic facies domain adaptation（2011.10510）。
13. Data-driven seismic waveform inversion robustness（1809.10262）。
14. Seismic resolution enhancement with knowledge distillation and domain adaptation（2506.22018）。
15. Triplet Loss for Knowledge Distillation（2004.08116）。
16. Earthquake Transformer（1909.06396）。
17. PhaseNet（1803.03211）。
18. SeisT 多任务地震表征（2310.01037）。

共同启示是：事件级/区域级隔离比随机窗口划分更重要；自监督表征对跨域排序有帮助，但绝对幅度和区域先验仍需单独建模；蒸馏和域适配必须在独立外域上验证，不能用同一 OOF 选择权重。

## 决策

保留 JEPA 作为公开预训练候选和未来盲迁移的备选 backbone；不以本轮结果重新训练或调参比赛模型。T3 继续冻结，直到找到带可靠五类事件标签的公开数据源。下一轮优先研究公开多源蒸馏的固定权重、外域盲评和不确定性，而不是继续扫描比赛数据。
