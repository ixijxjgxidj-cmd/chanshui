# Round 86：修订协议下 STEAD→INSTANCE 独立盲评（否决跨数据集泛化）

## 合规范围

训练、数据和评估均在 zzai 远程目录 `/root/projects/t2_instance_ensemble_kd_20260816/` 进行。输入仅为公开 STEAD 与公开 INSTANCE 缓存；INSTANCE 只用官方 dev，未使用 test。比赛 08、R1、R2 及任何衍生物均未读取或参与训练、调参、筛选。

## 协议门控

本轮不复用旧跨数据集脚本。新独立脚本在源码中定义 `LAMS=np.linspace(.05,1.5,146)` 并立即 `assert LAMS.min()>=.05`，从根源拒绝 `lambda=0` 常数预测伪高分。三教师只在 STEAD 训练事件上拟合；学生为 hard-only 或固定 `0.75 hard + 0.25 disagreement-aware feature KD`。INSTANCE dev 标签只在训练结束后用于一次评估。

## 有效结果

| 模型 | INSTANCE dev 分数 | 固定 lambda=0.40 分数 | rho |
|---|---:|---:|---:|
| hard-only | 155.2534 | 155.2534 | -0.0456 |
| 多教师 ensemble-KD | 155.1849 | 155.1849 | -0.0595 |

KD 相对 hard-only 下降 `-0.0685`，相关性也下降。两种报告分数相同是因为本最小独立脚本没有额外中心化斜率选择，且固定 0.40 未改善该跨数据集尺度失配；不存在零斜率伪优解。

## 决策

**否决为跨数据集稳健候选。** Round 84 的五个 STEAD 区域内正增益不能外推到独立 INSTANCE 数据集。多教师 disagreement-aware KD 降级为“STEAD 区域内研究证据”，不用于比赛模型、蒸馏路线或任何比赛数据调参。后续不能再扫描该教师数量、权重或温度；只有新的公开多源训练方案和独立预注册基准才能重新打开。

## 文献检索与精读

本轮 5 组 arXiv 查询获得 98 篇去重记录，保存于远程 `papers.json`。精读至少 15 篇：STEAD（1810.10669）、INSTANCE（2101.06465）、SeisBench（2210.11114）、Mousavi & Beroza（1911.05975）、Munchmeyer 区域迁移（2101.02010）、SeisLM（2410.15765）、地震基础模型综述（2503.24166）、seismic domain adaptation（2011.10510）、waveform inversion robustness（1809.10262）、Hinton KD（1503.02531）、Gou KD survey（2006.05525）、Mirzadeh teacher assistant（1902.03393）、Beyer patient teacher（2106.05237）、Lakshminarayanan deep ensembles（1612.01474）、Guo calibration（1706.04599）、Kendall uncertainty weighting（1705.07115）。

共同启示：区域内外推与跨数据集外推是不同问题；蒸馏若不包含多源/目标无监督适配，可能强化源域特征；独立数据集上的未校准相关性必须与分数一起报告。
