# 000：四包冻结基线（预注册）

- `experiment_id`: `baseline-freeze-20260811`
- `date`: 2026-08-11
- `parent_commit`: `47fb72c655896913c656c0765acde57bd334c3a4`
- `hypothesis`: 现有带哈希预测可在统一代码下复现 release manifest 的 T1 默认口径结果，并揭示四种数量罚口径、逐文件误差和数据质量差异。
- `change_scope`: 只新增只读画像/评估工具与 memory 事实层，不改变模型或生产预测。
- `primary_metric`: T1 四口径均分与总分；T2 MAE；T3 accuracy 和每类召回。
- `secondary_metrics`: 包/预测哈希、覆盖率、残差分位数、FP/FN、时长/采样率/通道/台站/缺口分布、运行时、吞吐与峰值内存。
- `admission_threshold`: 工具对已有 parser/scorer 复用；同一输入重复运行 JSON 核心结果一致；预测哈希必须匹配 release manifest；任何不一致先标红而不是覆盖旧值。
- `safety_checks`: 只读官方 zip；不解压原始数据进仓库；输出到被忽略的 `outputs/`，仅将摘要写入 `memory/scoreboard.csv`。
- `result`: T1 默认口径逐位复现 release manifest；四口径结果已入 `memory/scoreboard.csv`。08 T2 `MAE=0.523197455`、平均有符号误差 `+0.099781905`；08 T3 `183/205=0.8926829268`，真值只含标签 1–4。三包共 3894 个 MSEED 全部画像成功，采样率均为 100 Hz、无 gap/缺分量；第 2 轮 T3 有 1 个重叠文件；第 1 轮 T3 有 6 个 `<5s` 文件；T1 长记录为第 2 轮 2 个、08 超长记录 5 个。
- `output_sha256`: 完整画像 JSON `2a55e4164db40a8eb87d6aa518fb040f11f7b2996788234f8fc1513bcfa3ac05`（文件位于 ignored `outputs/`）。
- `commit`: 本轮冻结基线提交（见包含本文件的 Git 记录）
- `status`: completed
