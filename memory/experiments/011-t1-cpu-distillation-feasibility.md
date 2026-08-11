# T1 实验 011：七成员教师—学生 PhaseNet CPU 可行性基准

- 日期：2026-08-12
- 状态：**已完成；通过可行性门槛，进入正式三折 LOPO 蒸馏预注册**
- `experiment_id`：`011-t1-cpu-distillation-feasibility`
- `parent_commit`：`a846c26118195f8c63cb8d26f0cfe5ebac3bd7bc`
- 性质：训练实验的容量、吞吐、缓存和时间预算基准；不是候选分数实验，不产生可部署模型
- 服务器边界：服务器只用于训练、特征生成和实验计算，不作为最终部署服务器；本轮没有修改或重启任何服务

## 1. 假设与预登记门槛

假设：若冻结七成员教师的全量 N/P/S 概率能在 CPU 上于 12 小时内缓存，且单个 LOPO 折的同架构 PhaseNet 训练能在 24 小时内完成，则不应先降级到 LogisticRegression/小 MLP 等概率特征学生，而应直接进入正式 PhaseNet 教师—学生蒸馏。

固定门槛：

```text
teacher_cache_estimated_hours <= 12
max_single_fold_10epoch_hours <= 24
```

本轮只测可行性，不用历史答案挑超参数，不生成正式 held-out 预测，也不比较 T1 分数。

## 2. 输入身份与数据边界

三套历史包身份：

| 包 | SHA-256 | T1 文件 |
|---|---|---:|
| 第 1 轮 | `beb93b7544718c3b05be9cd5f4f3cbf78f7be8a32c125a696adeb87c5d3a524e` | 1,000 |
| 第 2 轮 | `d5ffc69223ab75815618e7647e4212d39e4c0e756a91c1e6ee04cb55c29f54e6` | 915 |
| 08 试题 | `560145f74f2d8861bb344a7396c15793a54f0ce3f4ad056dfa115cd7f756bb3b` | 784 |

这些包长期参与项目分析，只能称历史回归集，不能称盲测。脚本直接从 ZIP 流式读取 T1，不把原始比赛数据或机器绝对路径写入仓库、JSON 或本记录。

## 3. 实现与修复

新增：

```text
scripts/benchmark_t1_distillation_cpu.py
tests/test_benchmark_t1_distillation_cpu.py
```

脚本完成以下工作：

1. 盘点三套 ZIP 中全部 T1 波形的文件数、时长、采样点、台站数和长记录；
2. 加载冻结生产七成员，短文件使用 7 成员，`>300s` 长记录按生产逻辑只测前 5 成员；
3. 对每包固定分位短文件与全部/受限长文件测量 annotation 和概率平均吞吐；
4. 自动从实际 annotation 标定概率采样率和边缘裁剪时长；
5. 用 `PhaseNet.from_pretrained("diting")` 测量 CPU forward、backward、optimizer step 与 RSS；
6. 投影全量教师缓存时间、缓存字节和三折 10 epoch 训练时间；
7. 只输出 JSON，不保存 checkpoint，不修改生产代码或配置；POSIX 上结果文件权限设为 `0600`。

首次服务器运行在学生输入处报三分量变成两分量。根因是 `scipy.signal.resample_poly` 默认沿 axis 0 重采样，把分量轴也缩放。提交 `71a34c7` 固定 `axis=-1`，并增加“重采样后保持三分量”测试。

随后完整性审计发现缓存体积曾按输入 `100 Hz` 估算，而教师 annotation 实际为 `50 Hz`，约高估两倍。提交 `a846c26` 改为从代表 annotation 自动标定：

```text
probability_sampling_rate = 50.0 Hz
annotation_trim_seconds   = 9.99 s
```

此前两份中间运行不作为权威结果；只有修复后绑定 `a846c26` 的 JSON 可引用。

## 4. 权威证据身份

服务器权威结果：

```text
git_head = a846c26118195f8c63cb8d26f0cfe5ebac3bd7bc
basename = t1_distillation_cpu_a846c26.json
sha256   = 7a6a46473d05d5f78fcd2440db8c1643f46142ba839c25782ceaece22b6e43be
mode     = 0600
bytes    = 12755
```

JSON 位于服务器私有训练工作目录，不在仓库中；本记录只保留 basename、哈希、权限和聚合指标。服务器运行后仓库保持干净，未生成模型权重，未修改服务。

## 5. 全量输入盘点

| 包 | 文件 | 短文件 | 长文件 | 总波形秒 |
|---|---:|---:|---:|---:|
| 第 1 轮 | 1,000 | 1,000 | 0 | 66,034.05 |
| 第 2 轮 | 915 | 913 | 2 | 79,910.35 |
| 08 | 784 | 779 | 5 | 86,261.65 |
| 合计 | 2,699 | 2,692 | 7 | 232,206.05 |

附加事实：

```text
stations       = 2699
input_samples  = 23220605
sampling_rate  = 100 Hz
inventory_time = about 2.67 s
read_failures  = 0
```

## 6. 教师实测与全量投影

冻结教师顺序：

```text
guangxi
jiangxi
shandong
weights/aug/exam_aug6_r2train_sd.pt
weights/aug/crew_sp23_r2train_sd.pt
weights/geofon/geofon_m1_last_sd.pt
weights/geofon/geofon_m3_last_sd.pt
```

代表集合为每包两个短文件、第 2 轮两个约 3,600 秒长文件、08 一个约 4,000 秒长文件，共 9 个。七成员短文件典型总 annotation 约 `0.045–0.063 s/file`；五成员 3,600 秒长文件约 `0.29–0.30 s/file`，五成员约 4,000 秒长文件约 `0.37 s/file`。

概率完整性审计：

- 每个样本都有 N/P/S 三条概率曲线；
- 概率网格为 `50 Hz`；
- 3,600 秒文件约 `179,500` 个概率点/通道；
- 4,000 秒文件约 `199,501` 个概率点/通道；
- 测量包含真实 annotation，不是空结果或固定启动开销。

标定与投影：

```text
annotation_seconds_per_member_window = 0.0030488335704874424
probability_sampling_rate            = 50.0
annotation_trim_seconds              = 9.99
teacher_cache_estimated_seconds      = 132.1285057
teacher_cache_estimated_hours        = 0.0367023627
```

即全量教师概率缓存预计约 `2 分 12 秒`。这是由代表文件实测窗口成本投影的预算，不是“已经生成完整缓存”的声明。

## 7. 概率缓存体积

只保存 active 成员平均后的 N/P/S：

```text
averaged_float16_bytes = 65137386
averaged_float32_bytes = 130274772
```

约为 `62.1 MiB / 124.2 MiB`。若保存每个 active member 的 float16 概率，需要 `439683618` 字节；若所有文件都强制保存七成员 float16，保守上界 `455961702` 字节。

正式蒸馏只需要平均后的 float16 N/P/S；不保存七条成员曲线，既减少数据体积，也避免后续事后学习成员权重。

## 8. 学生 CPU 训练步实测

学生配置：

```text
model          = SeisBench PhaseNet
pretrained     = diting
labels         = [N, P, S]
examples       = 6
batch_size     = 2
warmup_steps   = 2
measured_steps = 12
optimizer      = AdamW-compatible benchmark path
learning_rate  = 1e-4
```

实测：

```text
mean_step_seconds   = 0.0084710991
median_step_seconds = 0.0084321185
steps_per_second    = 118.0484
examples_per_second = 236.0969
first_loss          = 0.3029853404
last_loss           = 0.2160541862
peak_rss_bytes      = 702156800
```

loss 下降、反向传播和参数更新均真实发生，不是空训练。峰值 RSS 约 `670 MiB`，低于服务器内存预算。

## 9. 三折 LOPO 训练投影

按 `60.02s` 学生窗、`30s` stride：

```text
round1_windows  = 1731
round2_windows  = 2208
final08_windows = 2490
```

固定 `10 epochs`、batch 2：

| held-out 包 | 训练窗 | 每 epoch steps | 10 epoch 估算 | 小时 |
|---|---:|---:|---:|---:|
| 08 | 3,939 | 1,970 | 166.88 s | 0.04636 |
| 第 1 轮 | 4,698 | 2,349 | 198.99 s | 0.05527 |
| 第 2 轮 | 4,221 | 2,111 | 178.82 s | 0.04967 |

最慢单折约 `3 分 19 秒`，即：

```text
max_lopo_10epoch_hours = 0.0552739215
```

## 10. 门槛判断

| 门槛 | 实测/估算 | 结果 |
|---|---:|---|
| 教师全量缓存 `<=12h` | `0.036702h` | 通过 |
| 最慢单折 10 epoch `<=24h` | `0.055274h` | 通过 |
| 内存可承受 | 峰值约 `670 MiB` | 通过 |
| 概率输出完整 | N/P/S、50 Hz、长文件长度正确 | 通过 |
| 不改变生产 | 无 checkpoint、无服务/配置变更 | 通过 |

权威 JSON 决策：

```text
phase_net_distillation_feasible = true
recommended_route = preregister_phase_net_distillation
```

因此不切换到 LogisticRegression/小 MLP 等轻量概率特征蒸馏。

## 11. 正式 LOPO 预注册

下一实验固定：

```text
student          = PhaseNet(diting)
sampling_rate    = 50 Hz
window_samples   = 3001
window_stride    = 30 s
loss_region      = model-default central [250, 2751) samples
final_window     = boundary-aligned when needed
batch_size       = 2
epochs           = 10
optimizer        = AdamW
learning_rate    = 1e-4
weight_decay     = 0
grad_clip        = 1.0
BatchNorm        = frozen/eval
Dropout          = eval
teacher_target   = frozen g7 averaged N/P/S float16
teacher_long     = first 5 members for >300s
```

两个候选：

```text
KD-only = CE(teacher_probability, student_probability)

KD+hard = 0.7 * CE(teacher_probability, student_probability)
        + 0.3 * CE(hard_gaussian_label, student_probability)
```

硬标签固定 P `sigma=0.2s`、S `sigma=0.3s`。三折分别留出第 1 轮、第 2 轮和 08；held-out 包不参与训练、早停、epoch 选择或候选选择。固定 10 epochs 后才运行 held-out 推理，并用生产完整后处理评分。

正式实现必须读取 DiTing PhaseNet metadata 中固定的 `blinding=[250,250]`，把教师从 `+5.0s` 开始的概率曲线与学生输出中央 2,501 点按时间轴对齐；首尾 250 点不进入 loss。这是模型定义的有效输出区，不允许按历史评分另行调节。30 秒为名义 stride，只有最后一窗在记录长度不能整除时边界对齐。

最终准入仍为三包 × 四数量罚 12 单元全部不下降且至少一项严格提升；还要报告逐文件回归、FP/FN、P/S 时差分与七个长文件。若两个候选都失败，不围绕相同历史包继续调温度、alpha、epoch、成员权重或后处理阈值。

## 12. 结论与限制

本轮结论只回答“CPU 上是否值得正式做 PhaseNet 蒸馏”：答案是明确可行，且时间、内存和缓存均有两个数量级以上余量。它不证明学生能超过或等于七成员教师，也没有产生任何可部署权重。

下一步是在训练服务器生成真实平均概率缓存并完成三折 OOP 训练/评分。服务器只作为实验计算节点；最终部署、systemd、cron、服务重启、捕获权限与生产发布均不属于本轮或下一训练轮范围。
