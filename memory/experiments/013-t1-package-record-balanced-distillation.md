# 实验 013：T1 包—记录层级等权 KD+hard

```text
experiment_id: 013-t1-package-record-balanced-distillation
date: 2026-08-12
parent_commit: fcfac3d6fd8b596892c98f10eafbbd0157f6c39d
hypothesis: 基础 KD+hard 的跨包回退部分来自按窗口 ERM 的层级贡献偏差；在不删除窗口的前提下让两个训练包等权、包内记录等权，可改善未见包稳健性
change_scope: 仅改变 KD+hard 每个训练窗口对 dense CE 的标量权重
datasets_and_sha256: 沿用实验 012 的 round1/round2/final08 三包与答案哈希；实现提交后重新生成绑定新 git_head 的教师缓存
prediction_or_feature_sha256: 待正式三折 held-out 推理后填写
primary_metric: 相对生产 g7 的三包 × 四数量罚 12 单元最差完整包总分差
secondary_metrics: 12 单元均值/正向数；逐包 FP/FN；P/S 时差分；受损文件数；7 条长记录差值；训练权重审计
penalty_modes: merged_file_floor0, per_phase_floor0, merged_exam, per_phase_exam
admission_threshold: 12 单元全部 >= -1e-9 且至少一个 > +1e-9；覆盖必须完整
safety_checks: held-out 波形和答案训练阶段均不打开；固定 10 epoch；不部署；不提交缓存/checkpoint/预测/日志/私有 JSON
runtime_environment: 训练服务器 CPU 离线实验；服务器不承担最终部署
result: preregistered
decision: pending
commit: pending
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
