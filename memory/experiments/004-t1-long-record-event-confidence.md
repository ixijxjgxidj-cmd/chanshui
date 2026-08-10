# 实验 004：T1 长记录 P/S 事件级联合置信度过滤

- 日期：2026-08-11
- 轮次：04
- 状态：**预注册已冻结；尚未实现、尚未运行阈值评分**
- 基准分支：`main`
- 基准提交：`fb4345e4d3cfbeda55a8f8da74d719b1499220d1`
- 生产代码提交仍为：`53117b9d5912578c93d90e0a53774d66d0286aef`
- 论文证据：`memory/papers/round-04-t1-long-record-event-confidence.md`

> 本文件在实验代码与阈值评分之前冻结输入、配对规则、阈值网格、源包选择、双向验证和成功/失败条件。运行后只能追加实现、结果与结论，不得根据 R2、08 或逐文件结果回改本节参数。

## 1. 本轮可证伪问题

> 在冻结的长记录 5 成员推理、`-1 dB` SNR 闸和 20 秒同相位去重之后，能否只删除 FIFO P→S 配对中联合置信度很低的完整事件，使 R2 与 08 的长记录官方分数在四种数量罚口径下都提高，同时不新增任何真实 P/S 漏检、不降低时差分、不伤害任何长文件，并保证全部 `≤300s` 文件逐位不变？

主假设：长记录剩余 FP 中存在大量由两个低置信相位组成的完整假事件；几何联合置信度能在跨包条件下识别其中一部分，而不触碰匹配到真值的事件。

反假设：PhaseNet confidence 未跨包校准，假事件和弱真实事件高度重叠；任何有实际删除量的阈值都会在至少一个包或文件中增加 FN、降低时差分或造成分布特化。

## 2. 冻结输入与身份核对

### 2.1 长记录拾取缓存

- 文件：`outputs/port_verify/_long_picks_cache.json`
- SHA-256：`b7e20333fbe97480017e8c8b5167be6f92a95c1969b5c7e91bb6e0319cc38699`
- 语义：7 个长文件在前 5 个生产成员、`-1 dB` SNR 闸之后、20 秒同相位去重之前的相位、相对时间与 confidence。
- 禁止：重新推理后用不同模型、阈值或缓存替换本文件再与当前冻结基线比较。

### 2.2 冻结生产基线

- 文件：`outputs/frozen_baseline/baseline_full_profile_prod_20260811.json`
- SHA-256：`2a55e4164db40a8eb87d6aa518fb040f11f7b2996788234f8fc1513bcfa3ac05`
- scorer 数量罚口径：
  - `merged_file_floor0`
  - `per_phase_floor0`
  - `merged_exam`
  - `per_phase_exam`

实验必须先在缓存上仅应用生产 20 秒 dedup，并断言 7 文件的预测数量恰为：

| 包 | 文件 | P/S |
|---|---|---:|
| R2 | `T1.A.Q0001.mseed` | `64/68` |
| R2 | `T1.A.Q0002.mseed` | `64/67` |
| 08 | `T1.A.Q0001.mseed` | `47/58` |
| 08 | `T1.A.Q0002.mseed` | `63/72` |
| 08 | `T1.A.Q0003.mseed` | `46/55` |
| 08 | `T1.A.Q0004.mseed` | `61/73` |
| 08 | `T1.A.Q0005.mseed` | `57/69` |

还必须复刻冻结逐文件 P/S 时差分、FP/FN、文件级分数与数量罚，浮点容差 `1e-9`。任一不一致都设置 `baseline_reproduction_pass=false` 并停止；不得继续跑阈值后再解释差异。

### 2.3 历史短文件审计缓存

- 文件：`outputs/port_verify/_reselect_cands.json`
- SHA-256：`5696f5ea2216deb8339db257243930d794a65438abb92dd53b465e80a5b34372`
- 用途：只证明短文件联合重选已做过且跨 R2/08 近零；本轮实验不得读取该缓存做参数选择。

## 3. 严格作用域

候选函数签名必须显式接收 `duration_s`，并满足：

```text
duration_s <= 300.0  -> 原列表逐对象顺序、时间、phase、confidence 完全不变
duration_s > 300.0   -> 先生产 20 秒 dedup，再执行本轮事件过滤
```

固定不变的生产因素：

- T1 成员及顺序；
- 长记录只用前 5 个成员；
- P/S 模型阈值；
- `long_snr_db=-1`；
- `long_dedup_s=20`；
- 波形预处理、overlap、概率集成和峰值细化；
- scorer 与四种数量罚实现。

本轮不得：

- 扫描新的同相位 dedup 窗；
- 重新打开 overlap、极性翻转 TTA、拾取级投票或救援；
- 删除所有 orphan、强制每个 P/S 成对或给长记录设置事件数量上限；
- 使用原始波形、振幅、目标包统计或目标包标签拟合校准器；
- 根据结果改变 P/S 间隔、配对顺序、事件分数或阈值集合。

## 4. 固定事件构建规则

### 4.1 生产 20 秒去重

从缓存重建 `Pick`，使用现行 `deduplicate`：

```text
merge_window_s[P] = 20.0
merge_window_s[S] = 20.0
```

此步是基线复刻，不属于本轮参数网格。

### 4.2 FIFO 非交叉 P→S 配对

将去重后的 P、S 分别按 `(time_utc, original_index)` 升序排序，使用确定性的双指针 FIFO 匹配：

1. 对当前最早未配对 P，跳过所有 `S < P + 0.2s` 的 S；这些 S 标记为 orphan 并保留。
2. 若最早可用 S 满足 `S <= P + 60.0s`，将该 P 与该 S 配成一对，并同时推进两个指针。
3. 若最早可用 S 晚于 `P + 60.0s`，当前 P 标记为 orphan 并保留，只推进 P。
4. 列表结束后剩余 P 或 S 全部为 orphan 并保留。

固定窗口：

```text
sp_min_s = 0.2
sp_max_s = 60.0
```

这不是同相位去重窗，不参与扫描。选择依据是：S 必须晚于 P；60 秒是前置诊断使用的宽松上界，目的是减少错误删除而不是证明事件距离。近同时重叠事件若无法被 FIFO 安全分离，应留下 orphan 或高分对，而不是引入更复杂的事后配对器。

### 4.3 事件联合分数

每对事件的固定分数：

```text
event_confidence = sqrt(max(P_conf, 0) * max(S_conf, 0))
```

- confidence 必须是有限数；若任一 confidence 为 NaN/Inf，该对保留并记录 `nonfinite_confidence_keep=true`。
- 不加入 min/max confidence veto、时间差权重、相位特定校准、文件内分位数或批次归一化。
- 不使用目标包整体 confidence 分布。

## 5. 冻结阈值网格

只允许四个状态：

```text
OFF
tau = 0.35
tau = 0.40
tau = 0.45
```

对于 active `tau`：

```text
event_confidence < tau  -> 同时删除该 P 和 S
event_confidence >= tau -> 两者都保留
```

严格使用 `<`，等于阈值时保留。所有 orphan 保留。

网格冻结后不得增加 `0.30/0.50`、连续优化 tau、按 P/S 分开阈值、按文件自适应阈值或根据目标包改阈值。

## 6. 评分与诊断定义

每个文件至少输出：

- 真值 P/S 数；
- baseline/candidate P/S 数；
- baseline/candidate FP、FN，另分 P、S；
- P/S 匹配残差与时差分；
- `merged_file_floor0`、`per_phase_floor0` 文件分数和数量罚；
- 配对数、orphan P/S 数、删除事件数；
- 每个配对的 P/S 时间、confidence、几何分数和 keep/drop 决策。

每个包输出四种完整包分数。`merged_exam` 与 `per_phase_exam` 必须把冻结短文件的原预测/真值计数与候选长文件计数合并后重新计算，不能只在 7 文件子集上冒充全卷口径。

定义：

```text
mode_delta = candidate_package_score - baseline_package_score
normalized_mode_gain = mode_delta / long_truth_phase_count
worst_normalized_gain = min(normalized_mode_gain over four modes)
```

其中：

- R2 `long_truth_phase_count = 174`
- 08 `long_truth_phase_count = 443`

## 7. 源包阈值选择

R2、08 分别独立选阈值；选择某包阈值时不得读取另一个包的候选分数或标签。

### 7.1 active 阈值的源包安全资格

一个 active tau 只有同时满足以下条件才进入排序：

1. 每个源长文件的 P FN、S FN 均不得高于 baseline；
2. 每个源长文件的 P time score、S time score 均不得低于 baseline（容差 `1e-9`）；
3. 每个源长文件的 `merged_file_floor0` 与 `per_phase_floor0` 总分均不得低于 baseline；
4. 源包四种完整包分数均严格高于 baseline；
5. 源包总 false positives 严格减少；
6. `worst_normalized_gain >= 0.01`。

任何一项失败即为 source-ineligible，不允许用某一数量罚口径的大涨覆盖 FN 或时差损失。

### 7.2 排序与并列

在 source-eligible 的 active tau 中：

1. 最大化 `worst_normalized_gain`；
2. 差值不超过 `1e-4` 时选择更低 tau；
3. 若没有 active tau 合格，源包选择 `OFF`。

### 7.3 leave-one-long-file-out 稳定性

对每个源包逐个移除一个长文件，按完全相同规则重选 tau：

- 所有 leave-one-file-out 子集都必须选择 active tau；
- full-source 与所有子集所选 tau 的最大值减最小值不得超过 `0.05`；
- 任一子集选择 OFF 或跨越两个网格档，记为 `source_selection_stable=false`。

R2 只有两个长文件，因此该检查等价于分别在单文件上验证阈值不会完全依赖另一个文件；它不能证明统计充分，但能阻止单文件主导的明显不稳定配置进入跨包阶段。

## 8. 双向跨包验证

源包选择冻结后，执行且仅执行：

1. 用 R2 选出的 tau 原样评估 08；
2. 用 08 选出的 tau 原样评估 R2。

目标包标签只用于一次性评分，不得改变 tau、FIFO、S-P 窗、几何分数或 eligibility 条件。

每个方向必须同时满足目标包条件：

1. 四种完整包 `mode_delta > 0`；
2. `worst_normalized_gain >= 0.01`；
3. 每个目标长文件 P FN、S FN 不增加；
4. 每个目标长文件 P/S time score 不下降；
5. 每个目标长文件两种文件级总分不下降；
6. 目标包 false positives 严格减少。

任一方向失败，`bidirectional_pass=false`。

## 9. 唯一共同生产候选

只有两个源包都选择 active tau 且选择稳定时，定义：

```text
tau_common = min(tau_selected_from_R2, tau_selected_from_08)
```

选择较低值是预先冻结的保守规则，不能根据其跨包总分改成较高阈值或第三个阈值。

共同候选必须分别在完整 R2 和 08 上满足第 8 节全部目标条件。还必须满足：

- 7 个长文件没有任一文件在两种文件级口径下降；
- 合计删除事件数大于 0；
- 合计 FP 严格减少；
- 合计 FN 不增加；
- 四口径方向完全一致。

只有源选择、双向跨包和共同候选全部通过，才设置 `development_pass=true`。

## 10. 短文件与安全性硬门槛

即使 7 个长文件分数通过，仍必须满足：

1. 候选函数对 `duration_s <= 300.0` 返回逐元素完全相同的 Pick 列表；
2. 300 秒边界固定为短路径，`300.000001s` 才进入长路径；
3. 纯噪声/任意输入下候选输出必须是 baseline picks 的子集，绝不新增拾取；
4. 空列表、只有 P、只有 S、乱序输入、重复时间和非有限 confidence 不崩溃；
5. 同一输入重复运行结果和 JSON 除 runtime 外完全一致；
6. 后处理单文件 P95 小于 `10ms`，不增加模型调用、成员数、TTA、主要内存或权重体积。

若后续进入生产集成，还需在统一冻结全包入口验证：

- R1 1000 个 T1 文件预测哈希逐位不变；
- R2 913 个短 T1 文件逐位不变；
- 08 779 个短 T1 文件逐位不变；
- 多台站、非 100Hz、缺分量、畸形上传和纯噪声 API 回归通过。

## 11. 结果输出与可复现性

计划实现：

- `scripts/experiment_t1_long_event_confidence.py`
- `tests/test_t1_long_event_confidence.py`
- 默认输出：`outputs/experiments/round04_t1_long_event_confidence.json`（受 `.gitignore` 保护）

固定随机种子：无需随机算法；仍记录 `seed=20260811` 作为运行元数据。

JSON 至少记录：

- Git HEAD、平台、Python/NumPy 版本；
- 两个输入文件的 SHA-256 与 baseline reproduction；
- 固定 FIFO、S-P 窗、事件分数和阈值网格；
- 每个包、每个 tau、每个文件的全部分数/计数/配对/删除决策；
- source eligibility、排序、leave-one-file-out 稳定性；
- R2→08、08→R2 结果；
- `tau_common` 与共同候选判定；
- 短路径/子集安全测试摘要；
- runtime；
- 所有 pass/fail 原因。

逐拾取历史答案和本地数据绝对路径不得写入 Git 跟踪文件；输出 JSON 留在 ignored 目录。

## 12. 已知非盲边界

- R2 与 08 标签都已在历史项目中被查看；
- 08 五个长文件参与过 20 秒去重选择；
- 本轮前置 AUC 和 confidence 中位数诊断也使用了两包长文件标签；
- 因此本实验不能称为真正盲测或独立终检。

缓解措施只有：粗网格、参数完全预注册、每包只用自身选阈值、leave-one-file-out 稳定性、双向跨包、共同保守阈值和逐文件无伤害门槛。即使通过，仍需把“缺少独立长记录包”写入生产风险。

## 13. 为什么不是历史重复

本轮不重复：

- 固定 `long_dedup_s` 窗扫描；
- overlap `0.75/0.9`；
- 极性翻转 TTA；
- 短文件唯一 P/S 对重选；
- 简单删除 orphan；
- 拾取级投票或救援。

实质新机制是：在冻结 20 秒去重之后，只对多事件长记录构建多个 FIFO P/S 事件，使用两相位联合分数删除完整低可信事件，并要求全部短记录逐位不变。

## 14. 采用与失败规则

### 若 `development_pass=false`

- 不改 `PickerConfig`、生产 dedup、API 或部署参数；
- 不扩大 tau 网格、不改变 60 秒、不添加事后 confidence 组合；
- 把配置、结果和失败原因追加到本文件及 `memory/failed-experiments.md`；
- 提交论文、实验脚本、测试和负结果；
- 不同步服务器、不重启当前稳定服务。

只有出现新独立长记录包、显式 event/noise 分支、可解释 confidence 校准或多台站输入，才值得重开。

### 若 `development_pass=true`

通过只代表有资格进入生产实现，不代表自动部署。后续必须：

- 在 `PickerConfig` 增加默认关闭、可回滚的长记录事件过滤开关；
- 接入现有 20 秒去重之后；
- 完成全量 256+ 测试、统一三包复评、4000 秒长记录、纯噪声、300 请求内存和 API 回归；
- 记录关闭开关与回滚提交；
- 只有正向结果才提交生产变更并部署服务器。

## 15. 失败后的下一步

若失败，保持生产 20 秒去重。下一轮优先级转向不依赖继续阈值扫描的事项：旧文档漂移与 release manifest、缺口屏蔽的合成鲁棒性、回滚再前滚演练，或等待 DiTing 原始数据后训练带显式事件/噪声分支的轻量学生。

## 16. 实现与运行

- 实现日期：2026-08-11
- 实验脚本：`scripts/experiment_t1_long_event_confidence.py`
- 防回归测试：`tests/test_t1_long_event_confidence.py`
- 结果文件：`outputs/experiments/round04_t1_long_event_confidence.json`（ignored）
- 结果 SHA-256：`dc55ae264e8ff830eafdd809e7d12b0c4bd3dc2608a2d427cc7cd753e17f5b9d`
- 结果大小：`2213081` 字节
- 随机种子记录：`20260811`；算法本身无随机分支
- 实际运行命令（本地绝对路径省略）：

```text
python scripts/experiment_t1_long_event_confidence.py \
  --round2-answers <第2轮官方答案zip> \
  --final08-answers <08-an.zip>
```

实现严格包含：

- 固定网格 `OFF/0.35/0.40/0.45`，拒绝未登记阈值；
- `duration_s <= 300.0` 原 Pick 对象与顺序不变；
- 长路径先走生产 20 秒同相位去重；
- FIFO、非交叉、仅时间决定的 `0.2–60s` P→S 配对；
- 几何联合 confidence、非有限 confidence 保留、严格 `< tau` 删除；
- 四种完整包口径、逐文件 P/S FP/FN/时差分、LOFO、双向和共同候选判定；
- active 阈值前的 OFF 硬闸；
- 输出只记录答案包 basename 与哈希，不记录本地绝对路径。

冻结基线来自正式 `T1.an`，写出器会先将相对秒格式化为两位小数再回读评分。首次 OFF 自检准确捕获到“缓存亚采样时间直接评分”产生的毫秒级伪差异；实现随后复刻 `.2f` 提交边界，才允许继续。这个修正只恢复冻结评分坐标，没有改变已预注册的去重、配对、confidence、阈值或准入规则。

定向测试结果：`16 passed`。覆盖固定网格、FIFO 非交叉与确定性、orphan 保留、完整低/高分对、阈值等号边界、NaN/Inf、300 秒边界、候选子集、两位小数提交边界、全部 source 合取门槛、tie 选择低 tau、LOFO、双向、共同最小 tau 和基线失败不运行 active 阈值。

## 17. OFF 基线复现与安全性

`baseline_reproduction_pass=true`：

- 两个冻结输入文件及两个答案包哈希全部匹配；
- 7 个长文件 20 秒去重后 P/S 数量逐项匹配预注册表；
- 逐文件 P/S 真值数、预测数、FP/FN、残差、时差分、两种文件级分数和数量罚在 `1e-9` 内一致；
- R2 与 08 四种完整包总分及卷级数量罚在 `1e-9` 内一致；
- R2 长记录真值相位总数 `174`、08 为 `443`，均匹配冻结分母。

安全性结果：

- `duration_s=300.0` 原对象与顺序不变；
- `300.000001` 才进入长路径；
- 所有候选输出均为 baseline picks 子集；
- 重复运行的选择与 JSON 诊断确定；
- 700 次后处理计时均值约 `0.1678 ms`，P95 `0.2048 ms`，显著低于 `10 ms` 门槛；
- 不增加模型调用、成员、权重或主要内存。

## 18. 固定阈值结果

表中分数是完整包总分差，FP/FN 只统计预注册长文件；`worst` 为四口径最坏归一化增益。

### 18.1 R2 源包

| tau | 删除事件 | FP | FN | merged_file | per_phase_file | merged_exam | per_phase_exam | worst |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.35 | 29 | 126→87 | 37→56 | +0.9056 | +0.9056 | -18.0944 | -18.0944 | -0.103991 |
| 0.40 | 39 | 126→77 | 37→66 | -11.0444 | -11.0444 | -28.0444 | -28.0444 | -0.161175 |
| 0.45 | 54 | 126→65 | 37→84 | -29.6389 | -29.6389 | -45.6389 | -45.6389 | -0.262292 |

R2 的三个 active tau 全部不合格，选择 `OFF`。最低阈值 `0.35` 虽在两个按文件截零口径各增加 `0.9056`，但卷级两口径各下降 `18.0944`，FN 增加 `19`。逐文件看：

- `Q0001` 删除 11 个事件，文件级分数 `+8.0`，但 P/S 时差分合计下降 `3.0`、FN 增加 `3`；
- `Q0002` 删除 18 个事件，文件级分数 `-7.0944`，P/S 时差分合计下降 `15.0944`、FN 增加 `16`。

这说明文件级数量罚的局部奖励会掩盖真实相位被一起删除；四口径与逐文件硬门槛都不可省略。

### 18.2 08 源包

| tau | 删除事件 | FP | FN | merged_file | per_phase_file | merged_exam | per_phase_exam | worst |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.35 | 44 | 254→186 | 96→116 | +16.0944 | +14.0944 | +9.0944 | +9.5944 | +0.020529 |
| 0.40 | 62 | 254→161 | 96→127 | +10.8722 | +8.8722 | -1.6278 | -1.1278 | -0.003674 |
| 0.45 | 101 | 254→135 | 96→179 | -45.4278 | -44.4278 | -53.4278 | -52.9278 | -0.120604 |

08 的 `0.35` 是最容易产生错误乐观结论的候选：四种整包分数全部上升，最坏归一化增益也达到 `0.020529`，但它仍增加 `20` 个 FN，并伤害 4/5 个长文件的真实相位：

- `Q0001` 没有新增 FN，文件分约 `+1.0/+0.5`；
- `Q0002` S FN `+1`、S 时差分 `-1.0`；
- `Q0003` FN `+7`、P/S 时差分合计 `-6.9056`、文件分 `-2.9056/-2.4056`；
- `Q0004` FN `+3`、P/S 时差分合计 `-3.0`，虽文件分 `+9.0`；
- `Q0005` FN `+9`、P/S 时差分合计 `-9.0`，虽文件分 `+3.0/+1.0`。

因此 `0.35` 也必须 source-ineligible，08 选择 `OFF`。`0.40/0.45` 的 FN 和最坏口径进一步恶化。

### 18.3 稳定性、双向与共同候选

- R2 full-source 选择 `OFF`；分别移除两个长文件的 LOFO 也都选择 `OFF`。
- 08 full-source 选择 `OFF`；五个 LOFO 子集全部选择 `OFF`。
- 因两个源包均没有 active tau，按预注册不执行可用的 active 双向配置，也不定义 `tau_common`。
- `source_selection_pass=false`、`bidirectional_pass=false`、`development_pass=false`。

## 19. 结论与生产决定

结论：**拒绝**固定 FIFO 事件几何 confidence 删除，不进入生产。

原因不是它完全无法减少 FP，而是 confidence 对弱真实完整事件的排序仍不可靠：低阈值能显著删除 FP，但会同步删除真实 P/S；08 上甚至出现“四种整包分数都上涨、逐文件真实相位却受损”的评分聚合陷阱。当前单台站输入、无位置/速度模型/多台站一致性时，简单完整对阈值不能提供预注册要求的安全选择性。

执行纪律：

- 不扩大 tau 网格；
- 不调整 `0.2–60s`、几何分数或加入事后 veto；
- 不改 `PickerConfig`、生产 dedup、API 或部署参数；
- 不同步实验逻辑到服务器，不重启当前稳定服务；
- 只有新的独立长记录包、显式 event/noise 分支、可解释的 confidence 校准或多台站/走时信息出现时才重开。

下一项最高价值工作转向不依赖继续阈值扫描的 release manifest/旧文档漂移修复与部署可复现性；随后做缺口屏蔽的合成鲁棒性反证。等待 DiTing 原始数据不作为当前阻塞。
