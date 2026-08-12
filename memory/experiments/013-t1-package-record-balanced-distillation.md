# 实验 013：T1 包—记录层级等权 KD+hard

```text
experiment_id: 013-t1-package-record-balanced-distillation
date: 2026-08-12
parent_commit: fcfac3d6fd8b596892c98f10eafbbd0157f6c39d
hypothesis: 基础 KD+hard 的跨包回退部分来自按窗口 ERM 的层级贡献偏差；在不删除窗口的前提下让两个训练包等权、包内记录等权，可改善未见包稳健性
change_scope: 仅改变 KD+hard 每个训练窗口对 dense CE 的标量权重
datasets_and_sha256: 沿用实验 012 的 round1/round2/final08 三包与答案哈希；实现提交后重新生成绑定新 git_head 的教师缓存
prediction_or_feature_sha256: r1=4af59150902db83f05cd9ee50cf480c4f25edbaaa007d88f29efab808a729910;r2=27f2ea2aaac4f825a8e4519eff46842a3cb5b7dfea5e63f6ef457e3528a365bd;f08=5b2b0b6537534e33b9aad24acd007295a31bbe673388cc24890f37cbb8c6448f
primary_metric: 相对生产 g7 的三包 × 四数量罚 12 单元最差完整包总分差
secondary_metrics: 12 单元均值/正向数；逐包 FP/FN；P/S 时差分；受损文件数；7 条长记录差值；训练权重审计
penalty_modes: merged_file_floor0, per_phase_floor0, merged_exam, per_phase_exam
admission_threshold: 12 单元全部 >= -1e-9 且至少一个 > +1e-9；覆盖必须完整
safety_checks: held-out 波形和答案训练阶段均不打开；固定 10 epoch；不部署；不提交缓存/checkpoint/预测/日志/私有 JSON
runtime_environment: 训练服务器 CPU 离线实验；服务器不承担最终部署
result: completed; rejected_robustness
decision: keep production g7; do not deploy balanced students
commit: implementation 4edc4c55a3bd9a189f7ac888cc066a529a06e7f5
```

## 1. 触发证据

实验 012 的 `KD+hard` 在 12 单元中 8 个上升，均值 `+6.3490740741`，但 R1 四单元全部为 `-6.2722222222`，因此已拒绝。它在 R2、08 和 7/7 长记录上改善，说明硬标签有信号，但 R1 普通记录仍发生系统回退。

匿名诊断没有支持“长记录本身导致失败”：最高时长、RMS 和峰度分箱反而改善。更直接的结构偏差是按窗口 ERM 让长记录和窗口较多的包占据更大风险质量：

| held-out | 训练窗 | 训练包内长记录窗占比 |
|---|---:|---:|
| round1 | 4,698 | 19.22% |
| round2 | 4,221 | 15.75% |
| final08 | 3,939 | 6.04% |

该诊断只形成事前假设，不用于按记录或按包选择结果。

## 2. 唯一变量

基础 `window-erm` 的每窗权重均为 `1`。新风险定义为：

```text
weight(window) = total_windows
                 / (number_of_packages
                    * records_in_package
                    * windows_in_record)
```

训练保留原随机窗口顺序和 batch size。每个 batch 对 KD 与 hard 分别计算逐样本、逐时间点平均 CE，乘对应窗口权重后取 batch mean：

```text
kd_loss   = mean_i(weight_i * CE_i(teacher, student))
hard_loss = mean_i(weight_i * CE_i(hard, student))
loss      = 0.7 * kd_loss + 0.3 * hard_loss
```

不做 weighted-mean 分母归一化，因为全训练集平均权重已严格为 `1`；按固定随机排列遍历所有窗口时，epoch 目标正是预注册的层级等权风险。

## 3. 固定配置

除 `risk` 外全部沿用实验 012：

```text
student          = PhaseNet.from_pretrained("diting")
loss             = kd-hard
risk             = package-record-balanced
teacher          = frozen g7 averaged N/P/S float16
sampling_rate    = 50 Hz
window_samples   = 3001
central_loss     = [250, 2751)
stride           = 1500 samples / 30 s
batch_size       = 2
epochs           = 10
optimizer        = AdamW(lr=1e-4, weight_decay=0)
grad_clip        = 1.0
kd_weight        = 0.7
hard_weight      = 0.3
p_sigma_s        = 0.2
s_sigma_s        = 0.3
BatchNorm        = frozen/eval
Dropout          = eval
seed             = 20260812
```

三折：

```text
hold round1: train round2 + final08
hold round2: train round1 + final08
hold final08: train round1 + round2
```

## 4. 实现审计要求

每份 `training.json` 必须记录且不包含文件 ID：

- `risk` 名称和公式；
- 窗口权重最小值、最大值、平均值；
- 每个训练包的记录数、窗口数和总权重；
- 每包期望总权重及偏差；
- 每包内记录总权重的期望值和最大绝对偏差；
- 平均权重为 1、包总权重相等、包内记录总权重相等的布尔检查。

单元测试必须证明：

1. `window-erm` 返回全 1；
2. 新风险的平均权重为 1；
3. 包总权重相等；
4. 每包内记录总权重相等；
5. 新风险只允许与 `kd-hard` 组合。

## 5. 运行与准入

实现提交并推送 GitHub 后，服务器只用 `git fetch` + `git merge --ff-only` 快进。教师 manifest 绑定代码 `git_head`，因此必须在新提交上重新生成完整 2,699 文件缓存；不得修改旧 manifest 绕过检查。

三折训练结束后才逐折打开 held-out 波形做推理，随后使用与实验 012 相同的 `g7`、答案、完整生产后处理和 `audit_t1_candidate_robustness.py` 执行正式审计。候选只有在 12 单元全部不下降且至少一个严格上升时才可进入生产讨论；否则拒绝并保持 `g7`。

## 6. 明确禁止

- 不训练新的 `KD-only` 或 `window-erm KD+hard`；它们已有实验 012 权威结果；
- 不改 alpha、温度、epoch、学习率、batch、stride、teacher 成员、阈值、overlap、cap、force-pair、SNR 或去重；
- 不使用 held-out 包做早停、权重设计、样本筛选或 checkpoint 选择；
- 不按包选择不同模型；
- 不提交缓存、checkpoint、预测、日志、答案内容或私有审计 JSON；
- 不在训练服务器部署 API、systemd、cron 或生产服务。

## 7. 正式运行证据

服务器仓库只通过 `git fetch` + `git merge --ff-only` 快进到实现提交 `4edc4c5`。训练服务器没有执行部署、服务重启、systemd、cron、API 或生产捕获操作。

### 7.1 教师缓存

实现提交改变后，按 manifest 的 `git_head` 屏障重新生成全部教师缓存：

```text
complete                    = true
files                       = 2699
package_counts              = round1 1000 / round2 915 / final08 784
probability_bytes           = 65136648
elapsed_seconds             = 474.7178692870
manifest_sha256             = 0a15cabc28839673e173816915d3993bb9cd4e19bd49ceab0ac78a978c0d2015
manifest_mode               = 0600
manifest_git_head           = 4edc4c55a3bd9a189f7ac888cc066a529a06e7f5
```

新旧 2,699 条记录的键、`.npy` SHA-256、shape、active-member 数和缓存配置逐条完全相同。新缓存只用于恢复代码—数据证据绑定；教师本身没有变化。

### 7.2 三折训练与权重守恒

| held-out | 记录 | 窗口 | 权重范围 | 平均权重 | 训练秒 | 最终 loss | checkpoint SHA-256 | training JSON SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| R1 | 1,699 | 4,698 | `0.0215732–2.9961734` | `0.9999999751` | `227.2687` | `0.05345885` | `6a35f6d1cc285d793150066711b8511ca2e2f4ea3c79fcd46a555d9ba66f37c0` | `788c7de69703c32bb4c931d273df2492e732ebf478e1343c723e285b6acd79d1` |
| R2 | 1,784 | 4,221 | `0.0202403–2.6919644` | `1.0000000231` | `204.0188` | `0.05048987` | `4ddcd904e29318a38f0cbac648685b048dce69ea50bd26a5810bc0f6d77ece91` | `9c24132746b90f7c5fe92fa6ab21d411a2b550180fb09f033c02ca6a48ef96e8` |
| 08 | 1,915 | 3,939 | `0.0180879–2.1524589` | `0.9999999644` | `189.0010` | `0.04997666` | `73efe525363be9c20791a8bcd9ef020455f515f7be98a4b7741aec7af7e04e76` | `d6c041fa21cec58345af2c37a2064e6ba0b2e75b68646ff69e87dc77b846ef18` |

三折的以下检查全部为 `true`：

```text
mean_weight_is_one
package_total_weights_equal
record_total_weights_equal_within_package
```

三折 `held_out_opened=false`、`held_out_answer_opened=false`。checkpoint 与训练 JSON 均为 `0600`，实算哈希与 evidence 内记录一致。

### 7.3 Held-out 推理

三次推理均使用冻结生产后处理：P/S `0.2/0.15`、overlap `0.5`、短文件 cap `1P/1S`、长记录 SNR `-1 dB`、conditional force-pair floor `0.03`、长记录 20 秒去重。

| 包 | 文件 | 秒 | 文件/秒 | P/S 数 | prediction SHA-256 | log SHA-256 |
|---|---:|---:|---:|---:|---|---|
| R1 | 1,000 | 4.4 | 229.75 | 995 / 995 | `4af59150902db83f05cd9ee50cf480c4f25edbaaa007d88f29efab808a729910` | `558021b38ac8a6ebe02b12f0dcea8ed570bfa37fd89ac2635655808a67c6c0a9` |
| R2 | 915 | 4.4 | 206.81 | 1040 / 1053 | `27f2ea2aaac4f825a8e4519eff46842a3cb5b7dfea5e63f6ef457e3528a365bd` | `5aae75f7042b7c1b61b41aacdd2a5d002ab91711f9fd0d4b9f69e52124e7e4ae` |
| 08 | 784 | 4.0 | 195.35 | 1071 / 1114 | `5b2b0b6537534e33b9aad24acd007295a31bbe673388cc24890f37cbb8c6448f` | `3cbcd416211ccbaf84e9d759389bedb95086a25b0a6f1f4e63f71b31e0dee461` |

三份预测覆盖完整、官方格式回读自检通过，文件与日志均为 `0600`。

## 8. 权威 12 单元审计

```text
audit basename = t1_balanced_lopo_audit_4edc4c5.json
audit SHA-256  = 8ffe1f5cd419cdcaaad4066cc3e3036569fb01bd324ba6c7e26d7a807ad5f009
audit mode     = 0600
audit git_head = 4edc4c55a3bd9a189f7ac888cc066a529a06e7f5
coverage       = complete on all 2699 files
robust_pass    = false
```

候选相对 `g7` 的完整包总分差：

| 包 | merged-file | per-phase-file | merged-exam | per-phase-exam |
|---|---:|---:|---:|---:|
| R1 | `-2.3222` | `-1.8222` | `-1.8222` | `-1.8222` |
| R2 | `-5.1944` | `-4.6944` | `+4.3056` | `+1.8056` |
| 08 | `-8.6889` | `-6.6889` | `-6.6889` | `-6.6889` |

聚合结果：

```text
positive_cells          = 2 / 12
worst_delta             = -8.688888888890
mean_delta              = -3.360185185186
all_cells_non_decreasing = false
robust_pass             = false
```

R2 两个卷级口径上升，但两个按文件口径下降；R1 四口径均下降，08 四口径均下降。数量罚口径不确定性不会改变拒绝结论。

## 9. FP/FN、相位与逐文件诊断

默认 `merged_file_floor0`：

| 包 | FP 变化 | FN 变化 | P 时差分变化 | S 时差分变化 | 受损文件 | 最差文件差 |
|---|---:|---:|---:|---:|---:|---:|
| R1 | `0` | `+3` | `-0.1667` | `-1.6556` | 157 | `-1.4778` |
| R2 | `+12` | `-1` | `+0.0111` | `+4.2944` | 212 | `-4.9611` |
| 08 | `+19` | `-7` | `+3.7111` | `+2.6000` | 179 | `-3.8889` |

层级等权把基础 `KD+hard` 的 R1 默认口径回退从 `-6.2722` 缩小到 `-2.3222`，说明窗口贡献结构确实影响 R1；但它同时把 R2 默认口径从 `+7.9222` 变为 `-5.1944`，把 08 从 `+17.2722` 变为 `-8.6889`。R2/08 的相位时差和 FN 仍有改善，却被 `+12/+19` FP 与按文件数量罚抵消。

## 10. 七条长记录

匿名诊断 JSON：

```text
basename = t1_balanced_lopo_long_records_4edc4c5.json
SHA-256  = 1c68c98c8c7c551ea0fe99fddb6bf850d58a801535bc53753f634ed8de08fff0
mode     = 0600
```

| 匿名记录 | 包 | 时长 | 总分差 | FP 变化 | FN 变化 |
|---|---|---:|---:|---:|---:|
| long-01 | 08 | 4000.01s | `-1.8556` | `+4` | `+1` |
| long-02 | 08 | 4000.01s | `-3.8889` | `+9` | `-1` |
| long-03 | 08 | 4000.01s | `-2.5056` | `+7` | `-2` |
| long-04 | 08 | 4000.01s | `-0.9778` | `+4` | `-2` |
| long-05 | 08 | 4000.01s | `-0.8722` | `+1` | `+1` |
| long-06 | R2 | 3600.00s | `-4.9611` | `+8` | `+3` |
| long-07 | R2 | 3600.00s | `+0.3389` | `+6` | `-7` |

只有 1/7 上升，七条合计 `-14.7222`。新风险将长记录的每窗权重压到全局最小附近，这与长记录 FP 控制退化的观察一致，但该诊断不能单独证明因果，也不能据此事后设计权重下限。

## 11. 决策

```text
keep production = g7
deploy students = false
retune same risk = forbidden
```

包—记录完全等权不是安全的统一训练目标。它缓解了 R1 回退，却牺牲 R2/08 和长记录，说明当前任务既不能让多窗记录按窗口数无限主导，也不能把每条记录的总训练质量强行压成相同。

不得继续在相同三包上扫描包权、记录权、权重截断、权重下限、长度分段、KD/hard 比例、epoch、阈值或后处理。只有真正独立的新包、官方规则实质变化，或能在实现前由新证据定义且不属于连续权重调节的机制，才允许重开。生产、T2/T3 和 gap 方向均保持冻结。

## 12. 隐私与产物边界

- 缓存、checkpoint、训练 JSON、预测、日志、答案和私有审计 JSON 不进入 Git；
- 仓库只记录 basename、SHA-256、权限和聚合指标；
- 不记录服务器、账号、SSH、代理或机器绝对数据路径；
- 本轮未修改或重启任何生产服务。
