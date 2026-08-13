# 实验 014：合规重建 T1 集成成员（公开数据 + S 软标签 sigma 收紧）

```text
experiment_id: 014-t1-public-domain-compliant-members
date: 2026-08-13
parent_commit: 75c17c6e1cf7b565c1f8d7d446682ddb5478b23c
status: preregistered
hypothesis: 生产第 4/5 成员由 R2 真题窗口（r2train.h5）微调而来，既违反“测试集永不参与训练”，又是四川域拟合物；用纯公开数据（ETHZ + CREW）按事件防泄漏切分重训，并把 S 软标签 sigma 由 0.3s 收紧到 0.2s（对齐官方 S 满分容差与 SeisBench/PhaseNet 不区分相位的 sigma 口径），可在不使用任何封存包训练的前提下，取得跨区域不退化的合规成员
change_scope: 仅替换集成第 4/5 成员的权重来源与软标签 sigma_s；不改推理后处理、不改阈值、不改成员数
datasets_and_sha256: ETHZ(SeisBench, 事件散列切分) train=e463952c5a557ba3551bae9850d7f98415f12047872b0d44175fcdbedd0ea267 dev=049393bacfaab39c713dcd9e5c091e9ad79fbc86ee07bcd3a5c2d863b90ab7d1; CREW chunk 000 见实现后补录
primary_metric: 公开 dev 留出集（ETHZ/CREW，按事件隔离）的 P/S 到时误差分布与满分率
secondary_metrics: R1/R2 四口径均分仅作“不回归检查”；跨区域最坏表现
penalty_modes: merged_file_floor0, per_phase_floor0, merged_exam, per_phase_exam
admission_threshold: 公开 dev 上 S 满分率不降且 P 不退化；R1/R2 四口径不出现 > 0.005 的系统性下降
safety_checks: 08-exam/08-an 全程封存不打开；R1/R2 不进入任何训练或阈值选择；训练只在 GPU 服务器 /data/dizheng-sol；不提交 checkpoint/池/预测
runtime_environment: GPU 服务器 Tesla P4 8GB，/data/dizheng-sol，venv torch 2.6.0+cu124 / seisbench 0.12.3
result: pending
decision: pending
```

## 1. 触发证据（本轮实测，不依赖封存 08）

### 1.1 合规缺陷

`outputs/train/r2train.h5` 实测 1001 个窗口，来自 915 个互不相同的 `source_file`（`T1.A.Q0001.mseed` …），恰为第 2 轮 T1 文件数。由它微调得到：

- `weights/aug/exam_aug6_r2train_sd.pt`（生产成员 4）
- `weights/aug/crew_sp23_r2train_sd.pt`（生产成员 5）

同时 `scripts/train_seismicxm_t2.py` 与 `scripts/train_seismicxm_t3.py` 的部署默认产物为
`Ridge.fit(vstack([X_r1, X_r2]), concat([y_r1, y_r2]))`，即 `t2_seismicxm_r1r2.joblib`、
`t3_seismicxm_r1r2.joblib`，两者都是 `serve_api.py` 的默认模型。

用户目标语句把四个包（08-an、08-exam、第 1 轮、第 2 轮）全部列为“测试数据集永远不参与模型微调与训练”。因此上述三处均为违规，需要用纯公开数据重建。

### 1.2 域先验缺口

USTC 省级权重与 `guangxi` 的平均相对 L2（111 张量）：huanan 0.0167 < guizhou 0.0206 < jiangxi 0.0247 < hunan 0.0250 < hainan 0.0261 << sichuan 0.0769 < shandong 0.1750（最远）。生产成员 3 为 `shandong`，而华南（广西/广东/福建/海南）权重 `huanan` 未进集成。本届由广西地震局举办，去年为四川地震局，去年数据仅供参考，故四川域拟合物的正当性失效。

### 1.3 sigma 口径

SeisBench `ProbabilisticLabeller` 默认 `sigma=10` 采样点（USTC pickers 原生 50Hz → 0.2s）；PhaseNet 参考实现 `sigma = label_width/5 = 6` 点（100Hz → 0.06s）。两者都**不按相位区分 sigma**。本仓 `make_soft_label` 默认 `sigma_p_s=0.2, sigma_s_s=0.3`：P 的 0.2s 与 SeisBench 默认一致，异常的是 S 的 0.3s——它比 P 宽 1.5 倍，且 0.3s 已超过官方 S 满分容差 0.2s。

同时本轮实测带符号残差表明误差是方差而非偏置（最优常数平移仅 +1.7pp，见 `memory/failed-experiments.md`），故应压方差、锐化峰，而非平移标定。

## 2. 唯一变量与两臂

| 臂 | 数据 | sigma_s |
|---|---|---|
| A（对照） | ETHZ + CREW 公开池，事件散列切分 | `--sigma-s 0.2 0.3`（现默认） |
| B（处理） | 同一池、同一切分、同一 seed | `--sigma-s 0.2 0.2` |

除 `--sigma-s` 外所有超参、数据、seed 完全一致，保证单变量。

## 3. 预登记的判定顺序

1. 先看公开 dev 留出集：B 的 S 满分率是否高于 A，P 是否不退化。
2. 再把胜出臂作为成员 4/5 候选，与 `huanan` 一起做集成级评估。
3. R1/R2 只用于“不回归检查”，同时报四口径；**不得**用 R1/R2 选择 sigma、阈值或成员。
4. 真正取舍以“跨区域最坏表现不退化”为准。

## 4. 已完成的集成级合规对照（零训练，R1/R2 仅作不回归检查）

| 配置 | 成员 | R1 merged_file_floor0 | R2 merged_file_floor0 |
|---|---|---:|---:|
| P0 生产冻结（含违规成员） | guangxi,jiangxi,shandong,exam_aug6_r2train,crew_sp23_r2train,geofon_m1,geofon_m3 | 1.786294 | 1.810140 |
| C1 最小合规替换 | guangxi,jiangxi,shandong,huanan,geofon_m2,geofon_m1,geofon_m3 | 1.786522 | 1.806667 |
| C2 华南域先验（去 shandong） | guangxi,huanan,jiangxi,guizhou,hunan,geofon_m1,geofon_m3 | 1.783983 | 1.811840 |

结论：移除两个违规成员后，R1/R2 表现与生产冻结基线在 ±0.004 内，**合规不带来可测代价**；C2 在 R2 上甚至略高于 P0。这三行仅为不回归检查，不作为选型依据。