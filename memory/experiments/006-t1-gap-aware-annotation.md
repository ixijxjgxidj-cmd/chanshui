# 实验 006：T1 gap-aware annotation 概率层最小反证

- 日期：2026-08-11
- 轮次：06
- 状态：**已预注册；尚未实现候选、尚未运行新的 annotation/带 gap 模型推理**
- 基准分支：`main`
- 基准提交：`6d0f50baf33f37175c832e2391b714e083396b07`
- 当前生产推理提交：`53117b9d5912578c93d90e0a53774d66d0286aef`
- 论文证据：`memory/papers/round-06-t1-gap-aware-annotation.md`

> 本文件在第 6 轮候选函数、概率导出和任何新的带 gap 模型调用前冻结。实现后只能追加结果；不得根据输出改变样本、注入、guard、数值容差、匹配容差、开发/终检边界或成功条件。

## 1. 失败现象与可证伪问题

第 5 轮在 R1/R2/固定噪声 77 个合成缺口变体中观察到：

```text
induced_new = 31
lost_reference = 36
remote_induced_new (>10s) = 13
remote_lost_reference (>10s) = 2
```

最终列表 `margin=0` 只删除 8/31 个 induced；`margin=10s` 仍留下全部 13 个远程 induced，同时误删 37 个稳定 reference picks。生产流程显示，最终列表之前已经完成 ensemble probability、正常阈值、条件式 force-pair、亚采样精化和 dedup，因此最终删除无法恢复上游变化。

本轮问题是：

> 在七成员概率平均完成后、正常阈值和条件式 force-pair 之前，根据 `Waveform.gaps` 把 P/S annotation 的物理缺口及固定 guard 区间精确置零，能否阻止 gap 区间候选参与正常阈值、低阈值补相和后续 dedup，同时保持物理 gap 10 秒外 P/S probability、正常/低阈值峰与最终 picks 不变？

主假设：第 5 轮大部分远程最终变化是由 gap 内峰先触发 force-pair 或抢占 dedup 代表造成；在峰形成前置零可消除这些下游副作用。

反假设：零填充已经在 PhaseNet 卷积/滑窗聚合阶段改变 gap 10 秒外概率；annotation mask 只改变局部数组，无法恢复远程概率、阈值穿越、峰位或 lost reference。若反假设成立，本轮直接拒绝，不扩大 guard、不切换重建。

预计改善项是未来 gap 输入的数量罚、P/S 时差稳定性和纯噪声安全，不是三套历史无 gap 包的官方均分。历史三包没有真实 gap，本轮不得虚构历史分数增益。

## 2. 代码位置与冻结生产顺序

当前七成员路径为：

```text
每个成员 model.annotate
→ 按成员/可选 TTA 对齐并平均 P/S/noise annotation
→ P=0.20 / S=0.15 正常阈值 picks_from_annotations
→ 某相位零触发且另一相位有正常触发时，以 0.03 floor 重挑该相位最高峰
→ 三点抛物线 refine
→ 转内部 Pick
→ P 1s / S 3s 标准 dedup
→ >300s 的 -1 dB SNR 闸
→ >300s 的 P/S 20s dedup
```

本轮候选插入点固定为“七成员 annotation 平均后、任何 `picks_from_annotations` 前”。生产 `picker.py`、`PickerConfig`、API、manifest 和部署脚本在开发结论前不得修改。

## 3. 冻结数据、样本与输入身份

### 3.1 输入包

| 包 | basename | SHA-256 | 本轮用途 |
|---|---|---|---|
| R1 | `第1轮比赛试题与答案_1765249163770..zip` | `beb93b7544718c3b05be9cd5f4f3cbf78f7be8a32c125a696adeb87c5d3a524e` | 开发 |
| R2 | `第2轮比赛试题与答案_1765249253663..zip` | `d5ffc69223ab75815618e7647e4212d39e4c0e756a91c1e6ee04cb55c29f54e6` | 开发 |
| 08 exam | `08-exam.zip` | `560145f74f2d8861bb344a7396c15793a54f0ce3f4ad056dfa115cd7f756bb3b` | 仅开发通过后一次终检 |

只记录 basename 与哈希，不记录比赛数据机器绝对路径。R1/R2 答案不参与本轮选型；无 gap 生产概率和 picks 是“局部破坏不应改变远处输出”的 reference。

### 3.2 开发与终检样本

完全复用第 5 轮在任何带 gap 推理前冻结的样本，不根据本轮概率结果换样本。

R1：

- `T1.A.Q0001.mseed`
- `T1.B.Q0001.mseed`
- `T1.C.Q0001.mseed`
- `T1.D.Q0001.mseed`

R2：

- `T1.A.Q0001.mseed` 的固定 360 秒裁剪
- `T1.A.Q0501.mseed`
- `T1.B.Q0326.mseed`
- `T1.C.Q0021.mseed`
- `T1.D.Q0051.mseed`

固定噪声：

- `noise-short`：60 秒、100 Hz、三分量独立标准高斯；
- `noise-long`：360 秒、100 Hz、三分量独立标准高斯；
- `numpy.random.default_rng(20260811)`。

08 终检仅在开发选出 active guard 后运行：

- `T1.A.Q0001.mseed` 的固定 360 秒裁剪；
- `T1.A.Q0943.mseed`；
- `T1.B.Q0501.mseed`；
- `T1.C.Q0501.mseed`；
- `T1.D.Q0131.mseed`。

长记录裁剪、reference anchor 和注入区间完全复用实验 005 的冻结算法；实现优先复用实验 005 的纯数据准备函数，避免同名规则漂移。

### 3.3 七种固定注入

每条开发波形仍生成以下 7 个变体：

| id | 分量 | 规则 |
|---|---|---|
| `mid-0p5-all` | Z/N/E | 中点 0.5 秒 |
| `mid-2-all` | Z/N/E | 中点 2 秒 |
| `mid-10-all` | Z/N/E | 中点 10 秒 |
| `anchor-center-2-all` | Z/N/E | 2 秒、中心为冻结 anchor |
| `anchor-edge-2-all` | Z/N/E | 2 秒、右边界为 `anchor-0.5s` |
| `double-2-10-all` | Z/N/E | 1/3 处 2 秒 + 2/3 处 10 秒 |
| `anchor-center-2-one` | SHA-256 决定单分量 | 2 秒、中心为 anchor |

半开样点区间内目标分量精确写 `0.0f`，`Waveform.gaps` 保存对应 UTC 区间。77/77 数组身份、未注入区域逐位不变和重复哈希必须重新通过。

## 4. 冻结生产模型与环境

使用 `deploy/production_release_manifest.json` 的 T1 配置和七成员顺序：

```text
guangxi
jiangxi
shandong
weights/aug/exam_aug6_r2train_sd.pt
weights/aug/crew_sp23_r2train_sd.pt
weights/geofon/geofon_m1_last_sd.pt
weights/geofon/geofon_m3_last_sd.pt
```

参数：

```text
P threshold = 0.20
S threshold = 0.15
force-pair floor = 0.03
force-pair = conditional, <=300s
standard dedup = P 1s / S 3s
long SNR = -1 dB, >300s
long dedup = P/S 20s, >300s
long ensemble members = 5
overlap = 0.5
polarity TTA = false
subsample refine = true
```

输入包与全部生产权重必须按 release manifest 校验 SHA-256 和大小；成员数必须为 7。运行固定 `PYTHONUTF8=1`，同一阶段 reference/gapped 使用同一设备、dtype、线程和库版本。

## 5. 原始 ensemble annotation 导出

实验脚本必须独立复刻 `ProbEnsemblePicker._classify_refined` 的 annotation 部分：

1. 每个成员对同一 ObsPy Stream 调用 `annotate(batch_size=256, overlap=0.5)`；
2. trace 必须按 `id/starttime/length` 一一对齐；
3. 短记录平均 7 成员，`>300s` 台站只平均前 5 成员；
4. 使用 `float64` 累加、均值后保留与生产相同的 annotation 数组语义；
5. 不在导出函数内挑峰、mask 或修改输入 Stream。

对结构哨兵，使用导出的 raw annotation 重新执行正常阈值、conditional floor、refine、内部 Pick 转换和全部生产后处理；结果必须与 `picker.pick()` 的 phase/count/time/confidence 在固定容差内一致。若不能复刻生产结果，停止实验，不能解释概率差。

## 6. 概率对齐与远程定义

P/S annotation 按 `trace.id` 对齐；要求 reference 与 gapped trace 的 starttime、delta、长度完全相同。只分析原始真实波形时间范围：

```text
[waveform.starttime_utc,
 waveform.starttime_utc + waveform.n_samples / waveform.sampling_rate)
```

短波形为满足模型窗长而复制补齐的尾部不进入 probability 安全统计。

对每个 annotation 样点时间 `t`：

- 到 gap 的距离为 0：位于物理 gap 闭区间；
- remote：到所有物理 gap 的最小距离严格大于 `10.0s`；
- local：其余真实数据范围内样点。

每相位、每变体记录：

- `remote_max_abs_delta = max |G(t)-R(t)|`；
- `remote_mean_abs_delta`；
- `remote_count_gt_1e6 = count(|G-R| > 1e-6)`；
- 正常阈值侧变化：`(R>=threshold) != (G>=threshold)`；
- floor 侧变化：`(R>=0.03) != (G>=0.03)`；
- raw reference/gapped 在正常阈值和 `0.03` 下的峰集合、峰位、峰值与匹配结果。

`1e-6` 是预注册的 float probability 数值等价容差，不根据结果放宽。最大差仍记录原值。

## 7. 唯一候选与固定 guard

候选纯函数：

```text
mask_phase_annotations(annotations, gaps, guard_s)
```

固定语义：

1. 丢弃非有限端点和 `end <= start` 的 gap；排序并合并重叠/相接区间；
2. 每段扩展为闭区间 `[start-guard_s, end+guard_s]`，再次取并集；
3. 对 channel 名以 `_P` 或 `_S` 结尾的 trace，按每条 trace 自己的 starttime/delta 计算落入闭区间的样点并精确置 `0.0`；
4. mask 外 P/S 数组逐位不变；其它 annotation 通道逐位不变；全部 trace metadata 不变；raw annotations 不得被原位修改；
5. `gaps=[]` 时直接返回输入 Stream 对象本身；有合法 gap 时返回独立副本；
6. 非有限 annotation 值不在 mask 外擅自修复；只记录诊断。

固定网格：

```text
OFF
0.0s
0.5s
1.0s
2.0s
5.0s
10.0s
```

`OFF` 为 raw gapped annotation。不得新增 `0.1/0.25/3/15/30s`，不得按 gap 长度、相位、文件、包或结果自适应。

禁止项：

- 不把 P/S 置 NaN；
- 不改 noise annotation；
- 不改原始波形、滤波、taper、插值、重建或 overlap；
- 不改阈值、force-pair、SNR、dedup、成员数或 TTA；
- 不新增模型调用、训练或权重；
- 不根据 08 选择 guard。

## 8. 从 annotation 到候选最终 picks

对 reference、raw gapped 与每个 active guard，都使用同一条冻结路径：

1. P/S 正常阈值调用生产模型的 `picks_from_annotations`；
2. 对每个正常峰调用生产三点抛物线 refine；
3. 若短文件某相位正常零触发且另一相位有正常触发，用同一 annotations、`0.03` floor 重挑并只留最高峰，再 refine；
4. 转成内部 `Pick`；
5. 应用末端护栏、标准 dedup、长 SNR 和长 20 秒 dedup。

active guard 必须在**正常阈值和 floor 两次挑峰前**完成。不得先挑峰后再删除，这会退化为第 5 轮已否证机制。

## 9. 匹配与诊断

最终 Pick 与第 5 轮一致，按相位使用：

```text
P: abs(dt) <= 0.10s
S: abs(dt) <= 0.20s
```

容差内候选按 `(abs(dt), reference_index, other_index)` 贪心一对一匹配。定义：

- `induced_new`：gapped/candidate 中无法匹配 reference 的最终 pick；
- `lost_reference`：reference 中无法匹配的最终 pick；
- `remote_induced_new` / `remote_lost_reference`：到物理 gap 严格大于 10 秒；
- `stable_outside_physical_gap`：reference 与 raw gapped 成功匹配且两者都不在物理 gap 内；
- `collateral_changed`：active candidate 删除或移动上述稳定 pick，使其不再按官方满分容差匹配；
- `residual_induced_new`：active candidate 后仍保留的 raw induced。

annotation 正常/floor 峰也按 P/S 相同时间容差匹配；另记录 peak value 差，不用 confidence 重新配对。

## 10. 两阶段开发判定

### 10.1 阶段 A：概率层必要条件

对全部 R1/R2/噪声 77 个变体，必须同时满足：

```text
remote_count_gt_1e6 == 0
remote_normal_threshold_crossing_count == 0
remote_floor_crossing_count == 0
remote_normal_peak_induced == 0
remote_normal_peak_lost == 0
remote_floor_peak_induced == 0
remote_floor_peak_lost == 0
remote_final_induced_new == 0
remote_final_lost_reference == 0
```

理由：active mask 在 gap±guard 外逐位等于 raw gapped annotation；若 remote probability/峰/最终输出已经改变，局部 mask 逻辑上无法恢复。任一失败时：

- 所有 active guard 标记 `ineligible_remote_probability`；
- 仍可离线计算 guard 的局部清除/误伤诊断，但不得宣称候选通过；
- 不运行 08；
- 不扩大 guard、不改插值/重建/训练。

### 10.2 阶段 B：active guard 资格

只有阶段 A 全绿，一个 guard 才可能合格，并且还须满足：

1. mask 结构、raw 非修改、metadata 和 mask 外逐位一致全部通过；
2. 正常阈值和 floor 在扩展 mask 内均无 P/S 峰；
3. 全部最终 `residual_induced_new=0`；
4. `lost_reference=0`、`collateral_changed=0`；
5. 单分量、三分量、短记录、360 秒长记录和两条噪声全部通过；
6. no-gap 调用返回原 Stream 对象，最终 picks 与 reference 逐字段不变；
7. 重复运行完全一致；
8. 性能通过。

在所有合格 guard 中选数值最小者。没有合格 guard 时选择 `OFF`，设置 `development_pass=false`。

## 11. 08 一次性终检

只有 `development_pass=true` 且已冻结最小 active guard 后才读取/推理 08 五条波形。输出只允许包含 `OFF` 与冻结 guard，不暴露其它 guard 指标。

08 必须重复满足阶段 A 与阶段 B 的全部条件；任一失败即 `holdout_pass=false`，不得改 guard 后重跑并称为通过。若开发失败，ignored JSON 固定写：

```text
holdout08.records = null
```

## 12. 确定性、结构与性能

结构哨兵固定为：

- R1 `T1.A.Q0001.mseed`；
- R2 `T1.A.Q0501.mseed`；
- R2 360 秒长片段；
- `noise-short`；
- `noise-long`。

先验证：输入/权重哈希、7 成员、注入身份、annotation 对齐、从 annotation 复刻生产 picks、raw 非修改和重复性。任何结构失败先修脚本，不得继续解释科学结果。

性能构造：两条各 400,000 样点的 float32 P/S annotation、100 个互不相交 gap，预热后 200 次；候选 P95 必须 `<10ms`，不增加模型调用或持久权重。性能只作为实现资格，不能覆盖科学失败。

## 13. 开发通过与生产资格

只有以下全部为真才设置 `development_pass=true`：

```text
package_identity_pass
weight_identity_pass
ensemble_member_count == 7
injection_identity_pass
annotation_alignment_pass
annotation_to_production_pick_pass
remote_probability_pass
remote_peak_pass
remote_final_pick_pass
selected_guard != OFF
all_development_variants_pass
determinism_pass
performance_pass
holdout08_pass
```

即使通过，也只代表进入生产实现评审，不自动上线。生产评审还需默认关闭开关、single/batch 多台站隔离、三套无 gap 全包预测哈希不变、全量测试、纯噪声/4000 秒/非 100Hz/缺分量/API 回归和可回滚部署。

若失败：

- 不改生产 `PickerConfig`、推理路径、API、manifest 或部署；
- 不同步服务器、不重启服务；
- 保留实验脚本、测试、论文矩阵和逐变体概率证据；
- 更新 `CURRENT_STATE`、decision log、failed experiments 和 scoreboard；
- 提交并推送负结果。

## 14. 计划产物与可复现性

计划新增：

- `scripts/experiment_t1_gap_annotation.py`
- `tests/test_t1_gap_annotation.py`
- ignored 输出：`outputs/experiments/round06_t1_gap_annotation.json`

JSON 至少记录：

- Git HEAD、平台、Python/NumPy/PyTorch/ObsPy/SeisBench 版本与设备；
- 输入包和权重 basename/相对路径、大小与 SHA-256；
- 样本、裁剪、anchor、数组哈希和七种注入；
- 每条 P/S annotation 的 id、start/delta/length/hash；
- remote probability、threshold/floor crossing、正常/floor 峰和最终 picks；
- 每个 guard 的结构、局部清除、residual、lost/collateral；
- 开发选择、08 锁状态、重复性与性能；
- 全部 pass/fail 和最终决定。

输出不得包含本地绝对路径、比赛答案内容、服务器/账号、私钥、代理、令牌或 Gitee 信息。固定随机种子 `20260811`。

## 15. 为什么不是历史失败方向重复

- 不重扫 overlap、极性、长去重窗、短文件 P/S 重选或固定时间平移；
- 不做加速度积分、taper、插值或重建；
- 不重复第 5 轮最终列表删除：本轮在任何阈值和 force-pair 前改变候选可见的 P/S annotation；
- 新增的决定性证据是逐样点 remote probability、正常阈值与 floor 峰，而不是只观察最终 Pick。

## 16. 预登记失败后的下一步

若 remote probability 已改变，本轮结论只是否证“局部 annotation 置零足以恢复远处不变”，不否证训练型 gap awareness。下一步不得原样扩大 guard；只有以下新证据之一出现时才重开 gap：

- 真实 gap 或可冻结的 gap augmentation 训练/验证数据；
- 能让 convolution 本身消费 observation mask 的架构/蒸馏机制；
- 新的独立 gap 包；
- 可靠的窗口贡献追踪，能在不伤害远处真实峰的前提下重算 annotation。

在这些条件出现前，最高价值工作转向 watchdog 捕获隔离与最终回滚/前滚发布演练，而不是继续在同一零填充输出上叠加后处理。
