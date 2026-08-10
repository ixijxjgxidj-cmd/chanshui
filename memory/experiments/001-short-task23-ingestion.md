# 001：T2/T3 短窗摄入边界

- `experiment_id`: `task23-short-window-ingestion-20260811`
- `date`: 2026-08-11
- `parent_commit`: `47fb72c655896913c656c0765acde57bd334c3a4`
- `research`: `memory/papers/round-01-short-window-ingestion.md`
- `hypothesis`: 不依赖 picks 的 T2/T3 固定窗模型应接收任意正长度波形；T1 与依赖 picks 的模型继续保留 5 秒下限。
- `datasets_and_sha256`: 第 1 轮 `beb93b7544718c3b05be9cd5f4f3cbf78f7be8a32c125a696adeb87c5d3a524e`
- `model_sha256`: T3 bundle `75f5ed29d7995651f53798bbdbdfe06110257cdad11b376b1169971090e8cdbb`
- `prediction_sha256`: `c63ad7b7e3e1667e277232d46e3d96ff27857e67e1c3426d41bee0b8a1138dfa`
- `primary_metric`: 第 1 轮 T3 accuracy 与第 5 类 recall，空响应按错误计。
- `safety_checks`: T1 短窗仍在 picker 前拒绝；依赖 picks 的估计器仍使用默认下限；合成 2 秒三分量 MSEED 测试；官方 6 个短文件端到端复验。

## 预注册门槛

1. 六个 1.5–3.5 秒官方 T3 文件全部从空响应恢复为合法 1–5 类整数。
2. 新响应必须与完整离线 SeismicXM 路径的预测一致。
3. T1 对同一个 2 秒合成 MSEED 仍返回空结果，且 picker 不被调用。
4. 现有 API、摄入和推理测试全部通过。

## 基线事实

波形画像发现第 1 轮 T3 有 6 个 `<5s` 文件：

| 文件 | 时长 | 真值 | 旧摄入结果 |
|---|---:|---:|---|
| `T3.D.Q0001.mseed` | 1.5 s | 5 | `too_short`，0 waveform |
| `T3.D.Q0002.mseed` | 1.7 s | 5 | `too_short`，0 waveform |
| `T3.D.Q0003.mseed` | 1.8 s | 5 | `too_short`，0 waveform |
| `T3.D.Q0004.mseed` | 2.0 s | 5 | `too_short`，0 waveform |
| `T3.D.Q0005.mseed` | 2.0 s | 5 | `too_short`，0 waveform |
| `T3.D.Q0006.mseed` | 3.5 s | 5 | `too_short`，0 waveform |

旧线上路径会在分类器前返回 `{}`。离线生产模型对第 1 轮完整 200 条预测为 `188/200=94.0%`，第 5 类 `10/10`；上述六条离线均正确预测为 5。若把旧 API 的六个空响应计错，则有效表现为：

- accuracy：`(188-6)/200 = 91.0%`
- 第 5 类 recall：`(10-6)/10 = 40.0%`

## 改动

- `mseed_reader.build_waveform` 与 `load_waveforms` 新增关键字参数 `min_duration_s`，默认仍为 5 秒。
- `/pick` 继续调用默认路径。
- `/magnitude` 和 `/classify` 仅当估计器 `needs_picks=False` 时传 `min_duration_s=0.0`；依赖 picks 的估计器继续默认 5 秒。
- 补充合成短 MSEED 的任务隔离测试。

## 结果

- 六个官方短文件经修复后的 `Engine.process_mseed_bytes_classify` 全部返回合法结果 `{"AAA": 5}`，与真值一致：`6/6`。
- 第 1 轮 T3 有效 accuracy 恢复为 `188/200=94.0%`，相对旧 API `+3.0` 个百分点。
- 第 5 类 recall 从旧 API 的 `40.0%` 恢复为 `100.0%`，增加 60 个百分点。
- 合成 2 秒 MSEED：T1 仍返回 `{}` 且 picker 未调用；T3 返回类别 5。
- 定向测试：`36 passed`。
- 全量回归：`243 passed, 7108 warnings`。

## 决策

录取。它修复的是任务间摄入边界串扰，不改变模型权重或 T1 生产行为；直接恢复官方历史包中被旧 API 静默丢弃的合法稀有类样本。

- `status`: accepted_pending_deploy
- `commit`: 本轮短窗修复提交（见包含本文件的 Git 记录）
