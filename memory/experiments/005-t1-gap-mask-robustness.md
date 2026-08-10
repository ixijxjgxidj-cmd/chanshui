# 实验 005：T1 零填充缺口屏蔽鲁棒性

- 日期：2026-08-11
- 轮次：05
- 状态：**已预注册；尚未实现候选、尚未运行缺口推理**
- 基准分支：`main`
- 基准提交：`f8d180488176b85a20bec9f6b224dd5d36c4dde1`
- 当前生产推理提交：`53117b9d5912578c93d90e0a53774d66d0286aef`
- 论文证据：待写入 `memory/papers/round-05-t1-gap-mask-robustness.md`

> 本文件在候选函数、缺口注入和任何带缺口模型推理之前冻结样本、注入规则、匹配方法、margin 网格、开发/终检边界及准入条件。实现后只能追加结果，不得根据输出增加样本、margin 或改变成功阈值。

## 1. 当前可证伪问题

读取层会在 MiniSEED 分段合并前记录时间缺口，再以 `merge(method=1, fill_value=0)` 保持时间轴连续，并把各分量缺口并集写入 `Waveform.gaps`。当前 `SeisBenchPicker.pick()` 与 `pick_batch()` 均未消费这些区间。

本轮问题是：

> 对真实无缺口比赛波形注入确定性的单分量或三分量零填充缺口后，一个只删除缺口及固定边界 margin 内拾取的确定性屏蔽器，能否删除全部缺口诱发的新拾取，同时不删除任何在原波形与缺口波形中都稳定存在、且实际到时位于物理缺口之外的参考拾取，不留下远距离 force-pair/dedup 副作用，并保证无缺口、单条和批量路径安全？

主假设：零填充制造的伪触发主要落在缺口内部或很窄的边界邻域；一个小而固定的 margin 可以将其删除，且不会碰到缺口外稳定真实信号。

反假设：零填充会改变更宽上下文内的概率曲线，或者缺口内触发会先参与条件式强制成对、标准去重、长记录 SNR 闸和 20 秒去重；因此最终结果层的区间删除要么留下远处副作用，要么必须扩大到会删除稳定参考拾取的 margin。

本轮是鲁棒性与数量罚安全实验，不是历史包提分实验。历史包没有真实 gap，不能把合成缺口结果换算成官方分数增益。

## 2. 已冻结的代码与数据事实

### 2.1 实现事实

- `src/phasepicker/io/mseed_reader.py::_collect_gaps` 在 merge 前按分量记录分段间缺口。
- `build_waveform` 对各分量缺口取并集，区间使用绝对 epoch 秒。
- merge 以零填充缺口；缺失整分量与时间缺口是两种不同语义。
- `Waveform.gaps` 使用 `compare=False`，默认空列表，已有构造调用兼容。
- 单条路径当前顺序为：模型阈值/条件式 force-pair → `Pick` 转换 → 标准 dedup → 长记录 SNR 闸 → 长记录 20 秒 dedup。
- batch 路径使用相同后处理顺序，但临时台站码把多条波形合并为一个 Stream。
- 当前任何阶段都未根据 `wf.gaps` 屏蔽拾取。

因此，简单地在最终列表上删除区间内 pick 并不自动证明安全：缺口 pick 可能已经抢走 dedup 代表，或让另一相位在远处触发条件式 force-pair。

### 2.2 历史包缺口画像

冻结画像：`outputs/frozen_baseline/baseline_full_profile_prod_20260811.json`

- SHA-256：`2a55e4164db40a8eb87d6aa518fb040f11f7b2996788234f8fc1513bcfa3ac05`
- R1：1400 个文件，`files_with_gaps=0`，`gap_count=0`。
- R2：1305 个文件，`files_with_gaps=0`，`gap_count=0`；另有 1 个 overlap 文件，与 gap 不同。
- 08：1189 个文件，`files_with_gaps=0`，`gap_count=0`。
- 三包合计 3894 个文件，没有可用于直接估计 gap 规则官方分数的真实样本。

这冻结了两项解释边界：

1. 任何候选在三包无缺口输入上都必须逐位不变；
2. 本轮若通过，收益只能表述为未来输入鲁棒性和数量罚风险收敛，不能声称历史官方均分提高。

### 2.3 输入包身份

实验只读取以下包；Git 跟踪输出只记录 basename 与 SHA-256，不记录机器绝对路径：

| 包 | basename | SHA-256 |
|---|---|---|
| R1 试题与答案归档 | `第1轮比赛试题与答案_1765249163770..zip` | `beb93b7544718c3b05be9cd5f4f3cbf78f7be8a32c125a696adeb87c5d3a524e` |
| R2 试题与答案归档 | `第2轮比赛试题与答案_1765249253663..zip` | `d5ffc69223ab75815618e7647e4212d39e4c0e756a91c1e6ee04cb55c29f54e6` |
| 08 试题归档 | `08-exam.zip` | `560145f74f2d8861bb344a7396c15793a54f0ce3f4ad056dfa115cd7f756bb3b` |

本轮不读取答案标签做 margin 选择；原始无缺口生产 pick 是合成破坏前的参考。答案包仅保留给后续无缺口全包回归，不用于合成 gap 的伪“官方得分”。

## 3. 严格作用域与候选定义

首轮只研究一个候选机制：**最终 Pick 列表的确定性区间子集过滤**。

候选纯函数语义：

```text
mask_gap_picks(picks, gaps, margin_s)
```

固定规则：

1. 先规范化 gap：丢弃非有限端点和 `end <= start` 的区间；合法区间按起点排序并合并重叠/相接段。
2. 每段扩为闭区间 `[start - margin_s, end + margin_s]`，扩展后再次取并集。
3. `Pick.time_utc` 落在任一扩展闭区间内时删除；边界相等也删除。
4. 其余 Pick 保持原对象、原顺序、原 phase/time/confidence/station/sample_index，不重排、不复制、不改写。
5. `gaps=[]` 时必须直接返回输入列表对象本身；`margin_s=0` 仍删除物理 gap 闭区间内的 pick。
6. 非有限 pick 时间不擅自删除；原样保留并记录诊断，因为静默处理未知时间可能掩盖上游错误。

候选不得：

- 改波形、插值、加噪、重建缺失数据或改变零填充；
- 改 P/S 阈值、force-pair、SNR、标准 dedup、20 秒 dedup、成员数或 overlap；
- 根据 confidence、相位、文件类别、采样率或包身份改变 margin；
- 新增模型调用、TTA、训练或权重；
- 把缺失整分量误标成覆盖整条记录的时间 gap；
- 根据 08 结果回选 margin。

如果最终层过滤因远距离副作用失败，本轮直接拒绝；不得在同一轮事后改成 annotation 级屏蔽、插值、taper 或概率重算。那些属于新机制，必须另行预注册。

## 4. 冻结生产配置

所有真实波形推理使用当前生产 T1 配置：

```text
pretrained = diting
members = guangxi, jiangxi, shandong,
          weights/aug/exam_aug6_r2train_sd.pt,
          weights/aug/crew_sp23_r2train_sd.pt,
          weights/geofon/geofon_m1_last_sd.pt,
          weights/geofon/geofon_m3_last_sd.pt
p_threshold = 0.20
s_threshold = 0.15
standard dedup = P 1s / S 3s
force_pair_short_s = 300
force_pair_floor = 0.03
force_pair_mode = conditional
long_snr_db = -1
long_snr_min_s = 300
long_dedup_s = 20
ensemble_long_members = 5
overlap = 0.5
tta_polarity_flip = false
subsample_refine = true
```

设备可为 CPU 或 CUDA，但同一阶段的原始/注入推理必须使用同一设备、dtype、batch size 和软件环境。设备与库版本写入 ignored JSON。

## 5. 固定真实波形样本

文件选择在任何带 gap 推理前按包和题型字母冻结，不根据答案、当前错误、confidence 或运行结果换样本。

### 5.1 开发集：R1 + R2

R1 全长短文件：

- `T1.A.Q0001.mseed`
- `T1.B.Q0001.mseed`
- `T1.C.Q0001.mseed`
- `T1.D.Q0001.mseed`

R2：

- 长记录来源：`T1.A.Q0001.mseed`
- 短文件：`T1.A.Q0501.mseed`
- 短文件：`T1.B.Q0326.mseed`
- 短文件：`T1.C.Q0021.mseed`
- 短文件：`T1.D.Q0051.mseed`

### 5.2 一次性终检：08

- 长记录来源：`T1.A.Q0001.mseed`
- 短文件：`T1.A.Q0943.mseed`
- 短文件：`T1.B.Q0501.mseed`
- 短文件：`T1.C.Q0501.mseed`
- 短文件：`T1.D.Q0131.mseed`

### 5.3 长记录裁剪

为在保持 `>300s` 生产门控语义的同时限制算力，两个长文件不跑全部 3600/4000 秒变体，而是冻结为 360 秒片段：

1. 先对完整原波形跑一次生产 picker，得到无缺口基准 picks。
2. 选择相对时间位于 `[180s, duration-180s]` 的最早 pick 作为 anchor；若不存在，使用波形中点。
3. 片段起点为 `clamp(anchor-180s, 0, duration-360s)`，长度严格 360 秒。
4. 从原数组按采样点裁剪，`starttime_utc` 同步平移；不得重采样。
5. 对裁剪后的原始 360 秒片段重新推理，只有这次输出作为后续注入的 reference。

裁剪算法只读取无缺口原始输出，不读取带 gap 输出；片段起点、anchor 和输入数组 SHA-256 写入 ignored JSON。

## 6. 固定合成噪声样本

为检查零填充边缘是否能在无事件背景制造拾取，另生成两条确定性三分量高斯噪声：

- `noise-short`：60 秒，100 Hz；
- `noise-long`：360 秒，100 Hz。

固定 `numpy.random.default_rng(20260811)`，每分量独立标准正态，存为 `float32`，不落盘。原始噪声可能因模型本身产生 pick，因此安全比较仍以各自无缺口 reference 为准，而不是先验假定必须为空。

## 7. 固定缺口注入

### 7.1 数组和元数据语义

- 注入区间使用半开采样区间 `[i0, i1)`，对应绝对时间 `[start, end)`；区间内指定分量样本精确设为 `0.0f`。
- `Waveform.gaps` 写入对应绝对 `(start, end)`；候选过滤按第 3 节的闭区间保守处理。
- 未注入分量逐位不变；数组形状、采样率、起点和站名不变。
- 注入后计算数组 SHA-256，重复运行必须一致。

### 7.2 anchor

每条最终实验波形先跑无缺口 reference：

- 在相对时间 `[15s, duration-15s]` 内选最早 pick；
- 若没有 pick，使用 `duration/2`；
- anchor 一经生成即冻结，任何 gapped 输出不得改变它。

### 7.3 七种变体

每条真实与噪声波形按下表生成七个变体；靠近端点时只做确定性 clamp，使区间完整落在 `[5s, duration-5s]` 内：

| id | 分量 | gap 规则 |
|---|---|---|
| `mid-0p5-all` | Z/N/E | 0.5 秒，中心为 `duration/2` |
| `mid-2-all` | Z/N/E | 2 秒，中心为 `duration/2` |
| `mid-10-all` | Z/N/E | 10 秒，中心为 `duration/2` |
| `anchor-center-2-all` | Z/N/E | 2 秒，中心为 anchor |
| `anchor-edge-2-all` | Z/N/E | 2 秒，右边界为 `anchor-0.5s` |
| `double-2-10-all` | Z/N/E | 2 秒中心在 `duration/3`，10 秒中心在 `2*duration/3` |
| `anchor-center-2-one` | 单分量 | 2 秒，中心为 anchor |

单分量由 `SHA-256(file_id_or_noise_id)` 首字节 `% 3` 确定，`0/1/2 -> Z/N/E`；不得人工挑分量。

## 8. 固定 margin 网格

只允许：

```text
OFF
0.0s
0.5s
1.0s
2.0s
5.0s
10.0s
```

- `OFF` 是 raw gapped picks，不执行过滤。
- active margin 仅做第 3 节区间子集过滤。
- 不得根据结果新增 `0.1/0.25/3/15/30s`，不得按 gap 长度、采样率、短/长文件或相位自适应。

10 秒是本轮允许的最大删除邻域，同时也是“远距离副作用”的分析边界。若仍需更宽区域才能消除新拾取，本机制判失败，而不是继续扩大。

## 9. 原始与缺口输出的固定匹配

所有比较按 P/S 分开，保持原列表索引。允许匹配容差：

```text
P: abs(dt) <= 0.10s
S: abs(dt) <= 0.20s
```

对同相位所有容差内候选边按以下键升序，再贪心一对一认领：

```text
(abs(dt), reference_index, gapped_index)
```

定义：

- `matched_stable`：reference 与 raw gapped 成功匹配。
- `induced_new`：raw gapped 中未匹配到 reference 的 pick。
- `lost_reference`：reference 中未匹配到 raw gapped 的 pick。
- `remote_induced_new`：到所有物理 gap 的距离都严格大于 10 秒的 `induced_new`。
- `remote_lost_reference`：到所有物理 gap 的距离都严格大于 10 秒的 `lost_reference`。
- `stable_outside_physical_gap`：`matched_stable` 中 reference 与 raw gapped 两个到时均不在任何未扩展物理 gap 内。
- `collateral_deleted`：active margin 删除了 `stable_outside_physical_gap` 对应的 raw gapped pick。
- `residual_induced_new`：active margin 后仍保留的 `induced_new`。

匹配只用于诊断，不改变候选输出。

## 10. 开发选择与防终检泄漏

### 10.1 结构哨兵

先只运行以下 R1/R2 哨兵：

- R1 `T1.A.Q0001.mseed`
- R2 `T1.A.Q0501.mseed`
- R2 360 秒长记录片段
- `noise-short`
- `noise-long`

若出现输入哈希不稳定、原始无缺口重复推理不确定、候选不是 raw 子集、对象被改写、gap 元数据与零区间不一致或基准模型加载不完整，立即停止并修复实验实现；这类失败不允许继续解释 margin。

### 10.2 开发集 margin eligibility

在全部 R1/R2 真实样本和两条噪声上，一个 active margin 只有同时满足下列条件才合格：

1. 所有无缺口调用返回输入 Pick 列表对象本身，逐对象与逐字段不变。
2. 每个 gapped 输出都是 raw gapped picks 的稳定子序列；不新增、重排或改写 pick。
3. 每个候选输出在扩展 gap 闭区间内的 pick 数为 0。
4. 全部变体 `residual_induced_new=0`。
5. 全部变体 `remote_induced_new=0` 且 `remote_lost_reference=0`；任一出现都说明 force-pair、dedup 或概率上下文产生了最终层屏蔽无法安全修复的远距离副作用。
6. 全部变体 `collateral_deleted=0`：不得用删除原始和缺口波形中都稳定存在、且实际位于物理 gap 外的参考 pick 来换取伪触发减少。
7. 单分量 gap 与三分量 gap 均通过，不得只在更容易的全零区间上通过。
8. 短记录与 360 秒长路径均通过。

排序：在合格 active margin 中选择数值最小者。若没有 active margin 合格，选择 `OFF`，设置 `development_pass=false`。

### 10.3 08 一次性终检

只有开发集选出 active margin 后才运行 08。终检代码只报告冻结 margin 与 OFF，不输出其它 margin 的 08 指标，避免事后挑选。

08 五个真实样本必须逐项满足第 10.2 节 1–8 条。任一失败即 `holdout_pass=false`；不得改 margin 后重跑并称为通过。

## 11. 单条、batch、确定性与性能

### 11.1 单条和 batch

- 对所有无缺口 reference 和全部开发变体，候选纯函数分别应用于 `pick()` 与 `pick_batch()` 输出。
- raw 单条/batch 使用第 9 节相位容差比较，pick 数和匹配数必须一致；候选不得扩大已有差异。
- 对同一份 raw Pick 列表，单条封装和 batch 封装的过滤结果必须逐对象、逐顺序、逐字段一致。
- 多波形 batch 中每条只允许使用自己的 `Waveform.gaps`，不得串台站屏蔽。

### 11.2 重复性

- 同一输入重复两次，注入数组哈希、gap 列表、anchor、raw pick 序列、选定 margin 和候选序列必须一致；runtime 字段除外。
- JSON 按键排序写出；不得包含本地绝对路径、答案内容或凭据。

### 11.3 性能

单独构造 10,000 个 Pick 与 100 个互不相交 gap，预热后运行 500 次：

- P95 小于 `5 ms`；
- 结果必须是稳定子序列；
- 不增加模型调用、权重、成员数或持久内存；
- 算法不得为每个 pick 线性扫描全部 gap，目标复杂度为规范化 `O(G log G)`、过滤 `O(P+G)` 或等价上界。

## 12. 开发通过与生产资格

只有以下全部为真，才设置 `development_pass=true`：

```text
input_identity_pass
no_gap_identity_pass
subset_and_order_pass
development_margin_selected != OFF
development_all_variants_pass
holdout08_pass
single_batch_pass
determinism_pass
performance_pass
```

即使 `development_pass=true`，也只代表最终层 gap mask 有资格进入生产实现评审，不代表自动上线。还必须追加：

1. 在 `PickerConfig` 增加默认关闭、明确可回滚的开关与固定 margin；
2. 同时接入 `pick()` 和 `pick_batch()`，并证明多台站 gap 不串扰；
3. 三套历史 T1 无 gap 全包预测 SHA-256 逐位不变；
4. 全量测试、纯噪声、4000 秒长记录、非 100 Hz、缺分量、畸形上传和 API 回归通过；
5. 只有正向结果才更新发布清单、部署和重启服务。

若任何必要条件失败：

- 不改 `PickerConfig`、默认生产路径、API 或部署参数；
- 不扩大 margin 网格；
- 不在本轮改成插值、taper、annotation mask 或概率重算；
- 保留脚本、测试、逐变体失败证据和论文矩阵；
- 更新 `memory/failed-experiments.md` 与 `memory/CURRENT_STATE.md`；
- 提交负结果，但不同步服务器、不重启服务。

## 13. 结果输出与可复现性

计划实现：

- `scripts/experiment_t1_gap_mask.py`
- `tests/test_t1_gap_mask.py`
- ignored 输出：`outputs/experiments/round05_t1_gap_mask.json`

输出至少记录：

- Git HEAD、平台、Python/NumPy/PyTorch/ObsPy/SeisBench 版本与设备；
- 三个输入包 basename、SHA-256 与 identity pass；
- 全部固定样本、裁剪相对起点、数组 SHA-256、采样率、时长、anchor；
- 七种注入的分量、样点区间、绝对 gap 与数组哈希；
- reference/raw/candidate pick 的 phase、相对时间和 confidence；
- 第 9 节全部匹配分类；
- 每个开发 margin 的 eligibility 与失败理由；
- 冻结 margin 的 08 一次性结果；
- 单条/batch、重复性和性能结果；
- 所有 pass/fail 与最终 `development_pass`。

固定随机种子：`20260811`。真实波形注入没有随机分支；种子只用于两条噪声。

## 14. 已知局限

- 历史包没有真实 gap；合成置零能复刻数值缺失，但不能覆盖所有遥测中断、时钟跳变、压缩损坏或非零缺失编码。
- 无缺口生产 pick 不是地震学真值，只是“不应因局部数据缺失而在远处改变”的保守参考。
- 14 个真实文件与两个噪声样本是机制反证集，不足以证明所有 gap 分布。
- 终结果层屏蔽天然无法恢复已被 force-pair、概率曲线或 dedup 改变的远处 pick；本轮用硬失败门槛暴露该限制，不以聚合平均掩盖。
- 单分量时间 gap 与整分量永久缺失不同；本轮不会把 `missing_components="pad"` 当成时间 gap。

## 15. 为什么不是历史失败方向重复

本轮不改变固定长去重窗，不做短文件 P/S 重选，不做极性翻转、overlap 扫描、加速度积分或固定时间平移。它只验证读取层已经保存、但推理层尚未消费的输入完整性元数据，目标是阻止零填充数据制造数量罚，而不是从现有无 gap 历史包继续调分。

## 16. 失败后的下一步

若最终层 mask 因远距离副作用失败，下一步优先考虑一个新的、独立预注册问题：能否在模型阈值拾取和条件式 force-pair 之前消费 gap 元数据，或在训练/增强阶段使用 gap-aware 标签屏蔽。只有论文证据和本轮逐变体结果能明确定位失败阶段时才开启；不得在本轮顺手实现。

## 17. 实现与运行

本节及其后内容是在上述预注册冻结后追加；没有修改样本、注入规则、margin 网格、匹配容差或成功条件。

新增隔离文件：

- `scripts/experiment_t1_gap_mask.py`
- `tests/test_t1_gap_mask.py`

生产文件未修改。候选仍只是实验脚本中的纯函数：

```text
mask_gap_picks(picks, gaps, margin_s)
```

实现要点：

- 合法 gap 排序、扩展并合并；有序 picks 走 `O(P+G)` sweep，非有序输入走 `O(P log G)` 二分，不逐 pick 线性扫描全部 gap。
- `gaps=[]` 直接返回原列表对象；有 gap 时只返回原 `Pick` 对象组成的稳定子序列。
- 非有限 gap 端点被丢弃，非有限 pick 时间保留并进入诊断。
- 七种注入均逐样点检查：目标区精确 `0.0f`、非目标区逐位不变、未选分量逐位不变、shape/rate/start/station 不变、重复注入 SHA-256 相同。
- 单条路径运行一次；同一组 reference + variants 用 `pick_batch()` 重复运行两次，分别验证单条/批量容差匹配和批量重复逐字段完全一致。
- 08 输出裁剪器只允许保留 `OFF` 与开发冻结 margin；开发没有 active margin 时不会扫描或推理 08 波形。

定向测试：

```text
python -m pytest -q tests/test_t1_gap_mask.py
20 passed
```

第一次模型命令在任何波形推理前因 Windows 默认 GBK 无法读取 SeisBench UTF-8 权重元数据而退出。部署脚本本来就固定 `PYTHONUTF8=1`；按同一环境重启后完成实验。可复现命令只记录脱敏占位符：

```text
PYTHONUTF8=1 python scripts/experiment_t1_gap_mask.py \
  --round1 <R1_ZIP> --round2 <R2_ZIP> --final08 <08_EXAM_ZIP> \
  --device cpu
```

运行环境：

- Python `3.13.3`
- PyTorch `2.13.0+cpu`
- NumPy `2.5.2`
- ObsPy `1.5.0`
- SeisBench `0.12.3`
- 实际设备：CPU；生产配置线程数 `2`

ignored 原始结果：

- 路径：`outputs/experiments/round05_t1_gap_mask.json`
- 大小：`2,162,874` bytes
- SHA-256：`8f4cd2720b59d301fcf71cad8dfceb355b4d4266cd276ec375ab58e114f8dedc`
- JSON 已扫描，不含机器绝对路径、服务器、账号、密钥、代理、令牌或 Gitee 信息。

## 18. 结构哨兵与实现可信度

结构哨兵通过：

```text
sentinel_pass = true
ensemble_member_count = 7
input_package_identity_pass = true
production_weight_identity_pass = true
```

哨兵样本严格为预注册的 5 条：R1 A、R2 A0501、R2 360 秒长片段、60 秒噪声和 360 秒噪声。哨兵通过后才运行其余开发样本。

全部开发结构检查：

- 11 条最终开发波形、77 个注入变体；
- 77/77 注入数组身份检查通过；
- 所有 reference 和 raw variants 的单条/批量数量及容差匹配通过；
- 所有批量重复输出逐 phase/time/confidence/station/sample_index 完全一致；
- 所有 margin 的候选均为 raw 的原对象稳定子序列，且未改写对象字段；
- 所有无 gap 调用均返回输入列表对象本身；
- 10,000 picks、100 个互不相交物理 gap、500 次：P95 `2.1463 ms`，低于 `5 ms`。

因此后续失败可解释为预注册机制失败，而不是输入、模型、随机性、单条/批量分叉或纯函数实现错误。

## 19. 开发集原始扰动结果

R1/R2 真实波形与两条噪声合计 77 个变体，在**执行任何 mask 前**相对无 gap reference 得到：

```text
induced_new = 31，分布于 28 个变体
lost_reference = 36，分布于 33 个变体
remote_induced_new (>10s) = 13，分布于 13 个变体
remote_lost_reference (>10s) = 2，分布于 2 个变体
```

按包分组：

| 包 | 变体 | induced | lost | remote induced | remote lost |
|---|---:|---:|---:|---:|---:|
| R1 | 28 | 15 | 15 | 6 | 1 |
| R2 | 35 | 16 | 21 | 7 | 1 |
| 固定高斯噪声 | 14 | 0 | 0 | 0 | 0 |

按注入类型分组：

| 变体 | 样本数 | induced | lost | remote induced | remote lost |
|---|---:|---:|---:|---:|---:|
| `anchor-center-2-all` | 11 | 9 | 9 | 3 | 1 |
| `anchor-center-2-one` | 11 | 5 | 5 | 2 | 1 |
| `anchor-edge-2-all` | 11 | 3 | 3 | 3 | 0 |
| `double-2-10-all` | 11 | 6 | 7 | 1 | 0 |
| `mid-0p5-all` | 11 | 0 | 1 | 0 | 0 |
| `mid-10-all` | 11 | 6 | 8 | 3 | 0 |
| `mid-2-all` | 11 | 2 | 3 | 1 | 0 |

两项安全边界值得单独记录：

1. `noise-short` 与 `noise-long` 的无 gap reference 都是 0 picks，14 个注入变体也全部保持 0 picks。当前生产模型没有在这两条固定高斯噪声上因零区间制造误报。
2. R2 的 360 秒长片段 reference 有 13 picks；7 个变体各有 12 或 13 picks，`margin=0` 下均无 residual induced、remote effect 或 collateral deletion。失败主要集中在短真实记录，而不是长路径爆炸。

## 20. 固定 margin 结果

下表只统计 single 路径，避免把已证明一致的 batch 路径重复计数；准入判断实际同时检查 single 与 batch。

| margin | residual induced | 有 residual 的变体 | collateral deleted | 有 collateral 的变体 | 合格 |
|---:|---:|---:|---:|---:|---|
| `0.0s` | 23 | 20 | 0 | 0 | 否 |
| `0.5s` | 19 | 17 | 5 | 5 | 否 |
| `1.0s` | 17 | 15 | 8 | 8 | 否 |
| `2.0s` | 15 | 15 | 10 | 10 | 否 |
| `5.0s` | 15 | 15 | 20 | 18 | 否 |
| `10.0s` | 13 | 13 | 37 | 31 | 否 |

解释：

- 31 个 induced 中只有 8 个实际落在物理 gap 闭区间内，因此 `0.0s` 只能清掉 8 个。
- 扩大到 `10s` 后，所有 gap 邻域内 induced 都被清掉，但 13 个 remote induced 仍然全部留下；它们不可能由本轮允许的最终区间 mask 修复。
- margin 越大，稳定真实结构的误删快速上升：`0.5s` 已误删 5 个稳定匹配，`10s` 误删 37 个。
- 2 个 remote lost reference 同样无法由删除型后处理恢复，并使所有 margin 直接失败。

代表性远程变化：

- R1 C 的 midpoint 10 秒 gap 为 `[33.03, 43.04]s`，却新增 S `20.283s`；最近 reference S 在 `45.631s`，相差 `25.348s`。
- R1 D 的 anchor-centered 2 秒 gap 为 `[36.83, 38.83]s`，却新增 S `56.978s`；最近 reference S 在 `88.740s`。
- R2 C 的 anchor-centered 2 秒 gap 为 `[60.75, 62.75]s`，新增 S `73.103s`，已经在 gap 右侧超过 10 秒。
- R2 D 的 anchor-centered 2 秒 gap 为 `[29.80, 31.80]s`，新增 P `48.217s`，并出现一个 10 秒外 reference 丢失。

另有一例 R1 B 单分量 gap 把 S 从 `38.816s` 改到 `38.565s`，差 `0.251s`，超过预注册 S 匹配容差 `0.20s`。这类变化即使数量不变，也会降低官方 S 到时得分，不能当作等价输出。

## 21. 决策

```text
development_margin_selected = OFF
development_pass = false
holdout08 = not run
production_eligible = false
decision = rejected
```

结论：**拒绝最终 Pick 列表固定 margin mask。**

拒绝依据不是“没有历史分数提升”，而是机制违反了预注册安全条件：零填充会改变 gap 10 秒外的概率/触发结果；删除最终列表中的邻域 pick 既清不掉远程诱发，也恢复不了远程丢失。扩大 margin 只会用更多稳定拾取的 collateral deletion 换取局部诱发减少。

严格执行后果：

- 没有运行 08 波形推理，只计算了预注册输入包 SHA-256；ignored JSON 中 `holdout08.records=null`。
- 没有修改 `PickerConfig`、默认推理路径、API、发布 manifest 或部署脚本。
- 没有同步服务器、重启服务或改变线上生产提交。
- 不扩大 margin、不改为自适应、不在本轮切换到 interpolation/taper/annotation mask。

本实验不能否证 gap-aware 推理本身；它只否证“所有后处理完成后再按固定时间区间删除最终 picks”这一小机制。

## 22. 下一轮最高价值问题

逐变体结果表明，下一步若继续研究 gap，作用点必须前移：在正常阈值触发、条件式 force-pair、SNR 和 dedup 之前，让 gap 区域不产生或不参与 annotation/候选峰。新一轮必须独立预注册以下边界：

- probability/annotation mask 是置 noise、置 NaN、置低概率还是跳过窗口；
- gap 边界是否需要固定 guard，且不得根据 08 回选；
- 是否能保持 gap 10 秒外 probability 与最终 picks 不变；
- 无 gap、纯噪声、短/长、单分量和 batch 多台站必须逐位安全；
- 若仍有远程变化，必须拒绝，不能继续用更宽最终 margin 掩盖。
