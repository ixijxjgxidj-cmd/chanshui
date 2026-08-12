# T1 实验 012：PhaseNet 三折 LOPO 教师—学生蒸馏

- 日期：2026-08-12
- 状态：**已完成；两个基础候选均未通过生产准入**
- `experiment_id`：`012-t1-phasenet-lopo-distillation`
- 代码提交：`f508480b537b968ed67d3210beb5782d6589fc01`
- 上游预注册：`memory/papers/round-07-t1-teacher-student-distillation.md`
- 可行性基准：`memory/experiments/011-t1-cpu-distillation-feasibility.md`
- 服务器边界：只做缓存、训练、推理和离线审计；没有部署、重启或修改任何生产服务

## 1. 结论

冻结七成员教师的同架构蒸馏在 CPU 上工程可行，但本轮两个预注册候选都不能替换生产 `g7`：

- `KD-only`：12 单元最差总分差 `-10.411111111112`，均值 `-2.994444444445`，仅 08 四个单元上升；
- `KD+hard`：12 单元均值 `+6.349074074074`，8/12 单元上升，但第 1 轮四个单元均下降，最差 `-6.272222222223`；
- 两者覆盖均完整，但都不满足“12 单元全部不下降且至少一个严格上升”，因此 `robust_pass=false`；
- 生产继续使用 `g7`，不部署学生 checkpoint，不调整生产阈值、成员权重或后处理。

`KD+hard` 在第 2 轮、08 和全部 7 个长记录上都显示出真实正向信号，但这不能抵消第 1 轮整包回退。该结构只用于形成下一轮新假设，不能用于事后挑包录取。

## 2. 预注册与防泄漏边界

本轮在打开 held-out 评分前固定：

```text
student          = PhaseNet(diting)
teacher          = frozen g7 averaged N/P/S
sampling_rate    = 50 Hz
input_samples    = 3001
loss_region      = central 2501 samples, blinding=[250,250]
nominal_stride   = 30 s
batch_size       = 2
epochs           = 10, fixed
optimizer        = AdamW
learning_rate    = 1e-4
weight_decay     = 0
grad_clip        = 1
BatchNorm        = frozen/eval
Dropout          = eval
seed             = 20260812
```

候选固定为：

```text
KD-only = CE(teacher_probability, student_probability)

KD+hard = 0.7 * CE(teacher_probability, student_probability)
        + 0.3 * CE(hard_gaussian_label, student_probability)
```

硬标签固定 P `sigma=0.2s`、S `sigma=0.3s`。三折分别整包留出第 1 轮、第 2 轮和 08；held-out 包不参与训练、早停、epoch 选择或候选选择。`KD-only` 不读取任何答案包；`KD+hard` 只读取两个训练包的答案，训练 JSON 均记录：

```text
held_out_opened=false
held_out_answer_opened=false
```

评分准入固定为三包 × 四数量罚 12 单元全部不下降，且至少一个单元严格上升。不得按包选择不同学生，也不得在结果出现后改 alpha、温度、epoch、阈值、成员权重或后处理。

## 3. 正式教师概率缓存

缓存覆盖全部 2,699 个 T1 文件：

| 包 | 文件 | active members |
|---|---:|---:|
| 第 1 轮 | 1,000 | 7 |
| 第 2 轮 | 915 | 913×7，2×5 |
| 08 | 784 | 779×7，5×5 |

固定格式：

```text
dtype                    = float16
channel order            = [N, P, S]
probability sampling     = 50 Hz
probability start offset = +5.0 s
member curves saved      = false
checkpoint saved         = false
```

权威缓存证据：

```text
manifest basename         = manifest.json
manifest SHA-256          = 645f8d04bc1a167f016594b94bbf5ed11495d8ebb51c0adac1bc7cb858555f5b
manifest mode             = 0600
cache directory mode      = 0700
raw probability bytes     = 65,136,648
elapsed seconds           = 466.721230233
max probability-sum error = 0.0003662109375
```

实验 011 的 `132.1285s` 是只根据代表 annotation 窗成本做的预算投影。实际完整物化用时约 `466.72s`，因为还包含 ZIP/ObsPy 读取、逐模型调用、每文件写入、哈希和 `fsync`。实际值约为投影的 3.53 倍，但仍只有约 7.78 分钟，远低于 12 小时门槛。预算体积 `65,137,386` 字节与实际概率字节只差 738 字节。

## 4. 六次固定训练

| 候选 / held-out | 窗 | 首轮 loss | 末轮 loss | 训练秒 | 峰值 RSS | checkpoint SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| KD-only / R1 | 4,698 | 0.0615467 | 0.0584317 | 210.75 | 753,049,600 | `72a4022bbb4d9caa2a3ee4d9a9dde97801ab070e04f947b6419f23354c41b93c` |
| KD+hard / R1 | 4,698 | 0.0547597 | 0.0518964 | 225.32 | 753,934,336 | `03229392de6a89b592687737fd57c0b5d8f4b6c337c5964927bfaf95868ec267` |
| KD-only / R2 | 4,221 | 0.0596627 | 0.0566334 | 196.37 | 675,573,760 | `9c66cb307363c76ffc74c8debe1b052ae1b72a02dea6755f8ac911a1e9bc0768` |
| KD+hard / R2 | 4,221 | 0.0534888 | 0.0507096 | 198.97 | 676,954,112 | `bfc6d8215ce1699972f96b27ff685fdecbc790483f65cbc142cb409cb9730730` |
| KD-only / 08 | 3,939 | 0.0591829 | 0.0563995 | 180.83 | 734,408,704 | `f1db3adfedae99f99096ee19d5ffe24c94f2048710eaac3b4b0ebbc137247fad` |
| KD+hard / 08 | 3,939 | 0.0533516 | 0.0507792 | 187.94 | 735,547,392 | `124bbe45fe97fb66b349ae442255a151277f982ee4cdcca583eafa7a885e1cda` |

六份 checkpoint 约 `1.115 MiB`，均为 `0600`；训练 JSON 也为 `0600`。所有运行完成固定 10 epochs，没有 held-out early stopping，loss 连续下降，峰值 RSS 约 `644–719 MiB`。

训练 JSON SHA-256：

| 运行 | SHA-256 |
|---|---|
| KD-only / R1 | `5da9592980dfbd26162e7691d443d7b77c2b686568cb2f603ffc1a1da974559c` |
| KD+hard / R1 | `76cdc7b1e39f0a2ef87435f8bb81d592ac783105a2ad77d70e55a84a3d6678a1` |
| KD-only / R2 | `24e30d168ba84b44972a73e6ed1a9495d17e4214dc6d3fdd0dceecf57d09a98a` |
| KD+hard / R2 | `34e3c7bf07b1e366ff62e672b61c526f248b9b0d879cd43c07e663622b128562` |
| KD-only / 08 | `8f5a843acfff837fe4bf6f65cea1b218a63a78299d9f5bddaaa2ed7e31d8efba` |
| KD+hard / 08 | `6e3ec44c1fcbd08097021aec1c458c6b5a61a2c693c165a9b97c19f74c04d617` |

## 5. Held-out 推理

六次推理都使用完整生产后处理：

```text
P threshold       = 0.2
S threshold       = 0.15
overlap           = 0.5
short cap         = <=300s, max 1P/1S
long SNR          = -1 dB
force pair        = conditional, floor 0.03
long dedup        = 20 s
```

| 候选 / 包 | 文件 | 秒 | 文件/秒 | prediction SHA-256 | log SHA-256 |
|---|---:|---:|---:|---|---|
| KD-only / R1 | 1,000 | 4.4 | 229.28 | `b9e31acfee05cded4e7029b78e198ce85c32fc2217baa408386a106bf8e9e748` | `53a498369237bc8001fe4e49525eec54de4ab62fac8de9176d51c4668cd75c7c` |
| KD+hard / R1 | 1,000 | 4.4 | 228.44 | `7ef0329884e69af32c61453da8cd3a20399a8002243e945fca967d439b207e22` | `b04487caa6e947fbb63b8638d545af950eee3db0b38e3e61ca80460d19d633e9` |
| KD-only / R2 | 915 | 4.3 | 212.53 | `5e276b3f972cc937a060ba83b7d23bebb7f6118dfcae2dea2731accda57903ba` | `1de0e74614964045ab74695a71187fe4e1831dd9e3a5567f0268a57de74d0b2e` |
| KD+hard / R2 | 915 | 4.3 | 212.34 | `9aa442a0c89e6acda8cc58519e6e7f2d9845be7a2300c636d7c3b302670064bd` | `5a7dda9dfde42b53b4815b41042243a9ba0dd26da0fea55cc28e4deb30304219` |
| KD-only / 08 | 784 | 4.0 | 195.29 | `f19f11172ea80b9f5fc202f79778bc6332197635e796132259fa1acefad9a48f` | `2efd40aa042e0d2804cf720fae3f9bcececafb26280440c892e26bb72bd783f6` |
| KD+hard / 08 | 784 | 4.0 | 196.00 | `034d4ad9d15b5c76a115bf86fad9dbf22629e972628e52b393b99a55cfd19ed4` | `3344b2e14bfa81859ae9e2a863b82e9b6c7994316ac05806c172860ac6468bfa` |

所有输出覆盖完整，行数、P/S 计数和官方格式自检通过。推理日志中的即时评分发生在运行过程；正式结论统一使用下一节审计器重新读取写出预测后的结果，避免把日志中的小数差异与权威审计混用。

## 6. 权威 12 单元审计

```text
audit basename = t1_distillation_lopo_audit_f508480.json
audit SHA-256  = ff98f656505e213f55c70a951ad4abdc8a07631b00d360d014cf35806d28b77d
audit mode     = 0600
audit git_head = f508480b537b968ed67d3210beb5782d6589fc01
exit code      = 2 (no robust candidate; expected experiment result)
```

下表为候选相对 `g7` 的**完整包总分差**，不是均分差：

| 候选 | 包 | merged-file | per-phase-file | merged-exam | per-phase-exam |
|---|---|---:|---:|---:|---:|
| KD-only | R1 | -10.4111 | -10.4111 | -10.4111 | -10.4111 |
| KD-only | R2 | -1.9222 | -1.9222 | -0.9222 | -0.9222 |
| KD-only | 08 | +2.1000 | +3.1000 | +3.1000 | +3.1000 |
| KD+hard | R1 | -6.2722 | -6.2722 | -6.2722 | -6.2722 |
| KD+hard | R2 | +7.9222 | +7.9222 | +6.9222 | +6.9222 |
| KD+hard | 08 | +17.2722 | +17.7722 | +18.2722 | +18.2722 |

聚合准入：

| 候选 | 正向单元 | 最差差值 | 12 单元均值 | 全部不下降 | 录取 |
|---|---:|---:|---:|---|---|
| KD-only | 4/12 | -10.4111 | -2.9944 | 否 | 否 |
| KD+hard | 8/12 | -6.2722 | +6.3491 | 否 | 否 |

## 7. FP/FN、相位和逐文件诊断

默认 `merged_file_floor0` 下：

| 候选 / 包 | FP 变化 | FN 变化 | P 时差分变化 | S 时差分变化 | 受损文件 | 最差文件差 |
|---|---:|---:|---:|---:|---:|---:|
| KD-only / R1 | +6 | +9 | -6.8111 | -3.6000 | 172 | -1.4778 |
| KD+hard / R1 | +4 | +7 | -4.4778 | -1.7944 | 175 | -1.4778 |
| KD-only / R2 | +1 | +1 | +0.8778 | -1.8000 | 232 | -3.1167 |
| KD+hard / R2 | -12 | -4 | +3.7444 | +3.1778 | 214 | -2.0000 |
| KD-only / 08 | -9 | -7 | +0.2556 | +1.8444 | 202 | -1.8167 |
| KD+hard / 08 | -26 | -16 | +6.8222 | +6.4500 | 174 | -1.0056 |

`KD+hard` 在 R2/08 同时减少 FP/FN 并提高 P/S 时差分，说明正向结果不是只靠数量罚口径产生；但 R1 的 FP、FN、P、S 全部恶化，因此仍是明确的跨包回归。

## 8. 七个长记录

匿名长记录诊断：

```text
basename = t1_distillation_lopo_long_records_f508480.json
SHA-256  = cbc0134ff73730dffdb17c8bbb6408c6bf2e32ebd8818144eef5e25cad9cadda
mode     = 0600
```

| 匿名记录 | 包 | 时长 | KD-only 总分差 | KD+hard 总分差 |
|---|---|---:|---:|---:|
| long-01 | 08 | 4000.01s | +1.6167 | +1.6111 |
| long-02 | 08 | 4000.01s | +1.0167 | +2.6222 |
| long-03 | 08 | 4000.01s | -0.0278 | +1.5389 |
| long-04 | 08 | 4000.01s | +4.5222 | +6.0222 |
| long-05 | 08 | 4000.01s | +0.1833 | +1.4333 |
| long-06 | R2 | 3600.00s | -3.1167 | +1.3833 |
| long-07 | R2 | 3600.00s | +4.8556 | +7.1278 |

`KD-only` 为 5 升 2 降，七条合计约 `+9.05`；`KD+hard` 为 7/7 上升，合计约 `+21.74`。因此本轮失败不是长记录能力不足：`KD+hard` 的主要阻塞是第 1 轮普通短文件域回退。长记录结果仍然只是历史回归诊断，不能单独成为生产录取规则。

## 9. 决策与下一假设

生产决策：

```text
keep production = g7
deploy student  = false
retune same run = forbidden
```

不得在三套相同历史包上继续扫：

- KD/hard 权重；
- 温度；
- epoch 或 early stopping；
- P/S 阈值、overlap、force-pair、SNR 或 dedup；
- 教师成员权重；
- 按包选择不同 checkpoint。

可用于下一轮研究的可证伪观察是：硬标签显著改善 R2、08 和全部长记录，却在 R1 的普通短文件上同时增加 FP/FN 并降低 P/S 时差分。下一候选必须是**实质不同的训练机制**，例如预注册的域平衡采样、包条件不变性或只使用训练包信息的稳健约束；开始实现前必须完成新的 AnySearch MCP + Browser MCP 原始论文轮次，并重新冻结假设与准入门槛。

## 10. 隐私与产物边界

- 缓存、manifest、checkpoint、训练 JSON、预测、日志、答案包和审计 JSON 均保留在私有实验目录，不进入 Git；
- 仓库只记录 basename、SHA-256、权限和聚合指标；
- memory 不记录服务器地址、账号、代理、SSH 路径或比赛数据机器绝对路径；
- 本轮没有修改 picker、API、模型发布清单、systemd、cron、捕获或生产资产。
