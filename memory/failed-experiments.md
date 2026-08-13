# 已否证或未录取实验

本文件用于避免重复消耗算力。重新开启任何条目前，必须写明新增证据、不同假设或新数据。

| 实验 | 结果 | 决策 | 证据 |
|---|---|---|---|
| 常数时间平移标定（P/S） | R1/R2 带符号残差 mean/median 均在 ±0.02s 内；最优常数平移后满分率最多 +1.7pp | 不采纳；误差是方差不是偏置 | `outputs/port_verify/_signed_residual_r1r2.json`（第 9 轮） |
| 软标签 S sigma 由 0.3s 收紧到 0.2s | 同池同 seed 双臂，公开 dev 2020 窗配对比较 mean_delta=-0.002195，95%CI=[-0.006730,+0.002341] 跨零，B 胜 141/A 胜 178 | 拒绝；保留默认 `--sigma-s 0.2 0.3` | `memory/experiments/014-*.md` 5.3；`/data/dizheng-sol/runs/e014_paired.json` |
| 加速度积分为速度 | 08 两个 HN 文件的全部积分变体均比原始输入差；最差长记录的主要问题仍是多事件 | 不采纳自动积分 | `af93e6f` |
| 兜底拾取 SNR 闸 | 真实兜底拾取 SNR 中位约 0.03 dB，与噪声不可分 | 不用于决定是否强制补发 | `af93e6f` |
| 极性翻转 TTA | 第 2 轮约 -0.4 总分，耗时约 2 倍 | 不采纳 | `ed77d22` |
| overlap=0.75 | 第 2 轮约 -2.1 总分 | 不采纳 | `ed77d22` |
| overlap=0.9 | 第 2 轮约 +6.5，但 08 约 -2.4，且耗时约 5 倍 | 视为分布特化，不采纳 | `ed77d22` |
| 亚采样抛物线精细化 | 全量第 2 轮 +0.16、第 1 轮 -0.05，总体近似 0 | 保留为无害能力，但不计为提分来源 | `382206c` |
| SeisT-L 零样本 | 最优约 1.5571，低于广西单模型约 1.716 | 不采纳 | `c9a59ff` |
| 拾取级投票集成 | 最优约 1.7013，低于广西单模型 | 不采纳 | `c9a59ff` |
| 广西锚定“只救不删” | 最优约 1.7081，仍低于广西单模型 | 不采纳 | `c9a59ff` |
| uniform model soup（top3/top5/all41） | 最优约等于单体 1.717，低于概率集成 1.723；作为第 4 成员约 1.720 | 生产不采纳；低算力单模型替代可保留 | `exp-soup` / `1853c77` |
| T2 GBR quantile | 08 可到约 0.510，但双向跨轮约 0.75/0.71，显著差于 Ridge 约 0.62/0.66 | 不采纳，判定为分布特化 | `exp-t2-quantile` / `3e010d5` |
| T2 QuantReg | 高维小样本欠拟合，退化接近常数 | 不采纳 | `exp-t2-quantile` / `3e010d5` |
| T2 Huber | 跨分布方向不一致 | 不采纳 | `exp-t2-quantile` / `3e010d5` |
| T2 双模型对冲 | 约 0.551，不及两轮合训 Ridge 约 0.523 | 不采纳 | `exp-t2-quantile` / `3e010d5` |
| T3 训练折中心化 + 等类 NCM/top-m/多原型 + R1 margin/support 门控 | 第 1 轮嵌套 accuracy/balanced accuracy 从 `0.9080/0.8656` 升至候选 `0.9240/0.9084`，但第 1→2 轮从 `187/189` 崩至 `121/189`，门控也仅 `130/189`；raw 成对诊断最高 `182/189` | 开发阶段拒绝，不查看 08、不改生产 | `memory/experiments/002-t3-long-tail-domain-generalization.md`；结果 SHA-256 `8cf922d4...2a34` |
| T2 源包 OOF residual + 预测值/低维 PCA 余弦邻域强收缩 | R1/R2 源包嵌套 OOF 分别改善 `0.01825/0.00846`，并稳定选择 `pca0_k40_s50`；但 R1→R2 MAE `0.621046→0.621532`，R2→R1 `0.660726→0.669618`，两个方向 signed bias 均扩大，gate 覆盖 `90–93%` 也不能识别跨包风险 | 开发阶段拒绝；未读取 08，不改 T2 bundle/API | `memory/experiments/003-t2-cross-package-residual-calibration.md`；结果 SHA-256 `91903410...61c3` |
| T1 长记录 20 秒去重后 FIFO 完整 P/S 事件几何 confidence 删除（tau 0.35/0.40/0.45） | R2 三档最坏归一化增益均为负，最低档 FP `126→87` 却 FN `37→56`；08 tau 0.35 虽四口径总分 `+9.09～+16.09`，仍使 FN `96→116` 并伤害 4/5 个长文件。两包 full/全部 LOFO 均选择 OFF | 开发阶段拒绝；不扩网格、不改生产、不部署 | `memory/experiments/004-t1-long-record-event-confidence.md`；结果 SHA-256 `dc55ae26...5b9d` |
| T1 所有后处理完成后的最终 Pick 列表固定 gap margin mask（0/0.5/1/2/5/10s） | 77 个 R1/R2/噪声变体产生 31 induced、36 lost，其中 13 induced、2 lost 在 gap 10 秒外；0s 只清 8 个，10s 仍留 13 个远程 induced 且误删 37 个稳定 picks。结构、single/batch、重复性和 P95 2.1463ms 均通过 | 开发阶段拒绝；未运行 08 波形，不改生产、不部署 | `memory/experiments/005-t1-gap-mask-robustness.md`；结果 SHA-256 `8f4cd272...dedc` |
| T1 七成员 annotation 平均后、阈值/force-pair 前的 P/S gap guard 置零（0/0.5/1/2/5/10s） | 754,070 个远程 P/S 样点中 295,900 个变化超过 `1e-6`，最大差 0.609977；normal/floor 阈值穿越 745/3,551，remote peaks 为 12/2 与 22/38 induced/lost，raw final 精确复现 13/2 remote induced/lost。0s 为 33 residual/40 lost，10s 为 18 residual/77 lost/37 collateral；结构与 P95 1.6621ms 通过 | 阶段 A 拒绝；77/77 probability 条件失败，未运行 08，不改生产、不部署 | `memory/experiments/006-t1-gap-aware-annotation.md`；结果 SHA-256 `e39c261f...7ada6`；提交 `bbf6e45` |
| T1 现有候选三包四口径稳健重排（`base/cond/fp/g6/g6gate/g7/ov90/prod2`） | 三包 × 四口径共 12 单元；全部候选覆盖完整，但无候选 12 单元全不下降。最佳 `ov90` worst `-6.1500`、mean `-3.0972`、仅 2/12 正向；第 1 轮四口径全降，08 四口径全降 | 拒绝所有候选替换；保持生产 `g7`，不改推理、不部署 | `memory/experiments/010-t1-candidate-robustness-audit.md`；结果 SHA-256 `98353140...0f988`；提交 `79028e0` |
| T1 同架构 PhaseNet 三折 LOPO 蒸馏（`KD-only` / `0.7 KD + 0.3 hard`） | 正式缓存、六次训练和六次推理均成功。`KD-only` 12 单元 worst/mean `-10.4111/-2.9944`；`KD+hard` 虽 8/12 正向、mean `+6.3491`，但 R1 四口径全降，worst `-6.2722`。`KD+hard` 在 R2、08 和 7/7 长记录上涨，仍不满足全单元不退化 | 两个基础学生均拒绝，不部署；不得在相同包上继续扫 alpha/温度/epoch/阈值/成员权重 | `memory/experiments/012-t1-phasenet-lopo-distillation.md`；审计 SHA-256 `ff98f656...8b77d`；提交 `f508480` |
| T1 `package-record-balanced KD+hard` 三折 LOPO | 权重守恒、held-out 屏障和三折训练均通过；R1 默认回退从 `-6.2722` 缩至 `-2.3222`，但 R2/08 反转为 `-5.1944/-8.6889`。12 单元 worst/mean `-8.6889/-3.3602`、仅 2/12 正向；长记录 1/7 上升、合计 `-14.7222` | 拒绝，不部署；同三包不得继续扫包权/记录权/截断/下限/长度分段 | `memory/experiments/013-t1-package-record-balanced-distillation.md`；审计 SHA-256 `8ffe1f5c...5f009`；提交 `4edc4c5` |

## 可重新开启的条件

- 出现真正独立、未参与选型的新分布；
- 官方规则变化直接改变目标函数；
- 有新的校准方法能在预注册的双向跨包验证中同时改善；
- 计算预算或部署约束改变，使“同分但更省资源”的方案具有新价值。
- 对 T3 当前条目，必须出现实质不同的域不变表征/无标签域移检测或新的独立包；仅改中心化系数、top-m、原型数或用第 2/08 包事后选配置不构成重开理由。
- 对 T2 OOF residual 条目，必须出现合法无标签目标批次、可解释包偏移的站点/区域/仪器/距离/绝对幅值元数据、DiTing/目标区标签或新的独立包；仅扩大 PCA 维数、k、shrinkage、gate 分位或根据包均值加常数不构成重开理由。
- 对 T1 长记录事件 confidence 条目，必须出现新的独立长记录包、显式 event/noise 分支、可解释 confidence 校准或多台站/位置/走时证据；仅增加阈值、改变 60 秒窗口、换 min/max/加权分数或按 08 事后挑规则不构成重开理由。
- 对 T1 最终列表 gap mask 条目，仅把 margin 改成别的固定数、按 gap 长度/相位/文件自适应、加 taper/interpolation 或用 08 回选都不构成重开理由。只有把作用点实质前移到 annotation/阈值/force-pair 之前，并证明 gap 10 秒外输出不变，才是不同机制。
- 对 T1 annotation gap mask 条目，前移到输出 probability 后仍已失败；任何固定/自适应局部 guard、NaN 或相同 zero-fill 输出上的 taper/interpolation 都不构成新机制。只有真实 gap/独立 gap 包、可冻结 gap augmentation 训练划分、模型层显式 observation mask，或可验证的滑窗贡献重算机制才允许重开。
- 对 T1 基础蒸馏条目，仅改变 KD/hard 权重、温度、epoch、early stopping、阈值、overlap、教师成员权重或按包选择 checkpoint 都不构成新实验。只有新的文献证据支持、在实现前预注册的域平衡或域不变训练机制，或新的独立包，才允许重开。
- 对 T1 层级等权蒸馏条目，包权/记录权连续插值、clip、floor、按长度分段、保留长记录权重或按结果选择风险都属于同一失败机制的调参。只有真正独立的新包、官方规则变化，或不依赖连续样本权重的新机制才允许重开。
