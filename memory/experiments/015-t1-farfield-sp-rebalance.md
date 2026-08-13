# 实验 015：远场（大 S-P）重平衡以治理粗大错时

```text
experiment_id: 015-t1-farfield-sp-rebalance
date: 2026-08-13
parent_commit: 5810117
status: preregistered
hypothesis: 官方评分中 P 残差 >1s、S 残差 >2s 直接判 0 分（“粗大错时”）。实测粗大错时窗的 S-P 中位数 26.10s，而全体窗 5.12s，且训练池 73% 为 S-P<10s 的近场窗——即远场欠拟合。按 S-P 分层上采样远场窗（不引入任何新数据源、不动推理后处理），可降低粗大错时率
change_scope: 仅改变训练池中各 S-P 分箱的窗数占比（scripts/mix_pool.py --upsample / prepare_pool.py --sp-min-s）；不改超参、不改阈值、不改成员数、不改后处理
datasets_and_sha256: 沿用实验 014 公开池 ETHZ+CREW（事件散列切分）；远场子池由同一 ETHZ/CREW 缓存按 --sp-min-s 重新抽取，实现后补录 sha256
primary_metric: 公开 dev 留出集（1805 个含 P+S 窗，事件隔离）的粗大错时率 = (gross_P + gross_S + gross_both) / n
secondary_metrics: 按 S-P 分箱的分层均分与满分率；漏检数；官方分均值；R1/R2 四口径仅作不回归检查
penalty_modes: merged_file_floor0, per_phase_floor0, merged_exam, per_phase_exam
admission_threshold: 公开 dev 粗大错时率相对实验 014 臂 A 下降，且近场分箱（S-P<10s）均分不退化；R1/R2 四口径不出现 > 0.005 的系统性下降
safety_checks: 08 全程封存；R1/R2 不进入训练、不用于选择任何超参或成员（AGENTS.md 第 2 条）；训练只在远程 GPU；不提交 checkpoint/池
runtime_environment: GPU 服务器 Tesla P4 8GB，/data/dizheng-sol
result: pending
decision: pending
```

## 1. 触发证据（公开 dev 留出集，未使用任何封存包）

`/data/dizheng-sol/runs/e015_gross_diag.json`，模型为实验 014 臂 A，评估窗 1805（同时含 P 与 S）：

| 类别 | 窗数 | 占比 |
|---|---:|---:|
| 干净（P、S 均在判零阈内） | 1744 | 96.62% |
| 仅 P 粗大错时（>1.0s） | 27 | 1.50% |
| 仅 S 粗大错时（>2.0s） | 28 | 1.55% |
| P、S 同时粗大错时 | 6 | 0.33% |
| P 漏检（无 P 票） | 6 | 0.33% |
| S 漏检（无 S 票） | 5 | 0.28% |
| **粗大错时合计** | **61** | **3.38%** |

### 1.1 关键否证：不是相位混淆

显式检查「P 票落在真 S 附近」与「S 票落在真 P 附近」：

```text
P pick landed near true S : 0
S pick landed near true P : 0
```

**两个方向都恰好为 0**，因此把粗大错时归因为 P/S 相位互换是错的，不要往这个方向做。

### 1.2 真实机制：远场欠拟合

| 群体 | S-P 中位 | S-P p10 | S-P p90 |
|---|---:|---:|---:|
| 粗大错时窗（61） | **26.10s** | 10.68 | 37.30 |
| 全体窗（1805） | 5.12s | 1.71 | 24.71 |

训练池 S-P 分布（`mix_pub_train`，n=18996）：

| S-P 区间 | 窗数 | 占比 |
|---|---:|---:|
| [0,5)s | 8399 | 44.21% |
| [5,10)s | 5537 | 29.15% |
| [10,20)s | 2199 | 11.58% |
| [20,30)s | 1470 | 7.74% |
| [30,40)s | 981 | 5.16% |
| [40,60)s | 410 | 2.16% |

近场（<10s）占 **73.36%**，而粗大错时几乎全部落在 S-P >10s 的远场区。窗长 60.02s，S-P 40s 以上的窗里 S 已接近窗尾，可用上下文更少。**这是分布不平衡问题，不是标签或解码器问题。**

## 2. 唯一变量与三臂

| 臂 | 训练池构成 | 说明 |
|---|---|---|
| A0（对照） | 实验 014 臂 A 的原池 | 已训完，直接复用，不重跑 |
| R1 | 原池 + 远场窗（S-P ≥ 10s）上采样 ×2 | 温和重平衡 |
| R2 | 原池 + 远场窗（S-P ≥ 10s）上采样 ×4 | 激进重平衡，检验是否单调 |

除训练池分箱占比外，超参、seed、起点权重、epoch 全部与实验 014 臂 A 一致。

## 3. 预登记判定顺序

1. 主指标：公开 dev 粗大错时率是否下降。
2. 护栏：近场分箱（S-P<10s）均分不得退化——防止“拆东墙补西墙”。
3. 若 R1/R2 单调，取更优者；若非单调，取粗大错时率最低且近场不退化者。
4. 通过后才做集成级评估，R1/R2 四口径**仅作不回归检查**，不参与选择。

## 4. 与文献的关系

第 9 轮证据矩阵中 B3（Rethinking Heatmap Regression，arXiv 2012.15175）给出「按目标尺度自适应」的先例；A11（Bridging scales，GRL 2020）与 A8（Which picker fits my data?，JGR 2022）都指出训练域与测试域的尺度/距离分布差异是深度拾取器退化的主因。本实验把「尺度」具体化为 S-P 走时差，属于这些结论在本项目的直接落地。