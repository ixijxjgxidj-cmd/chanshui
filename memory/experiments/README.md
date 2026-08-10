# 实验记录目录

每轮实验使用独立文件，并在运行前完成预注册。最低字段：

```text
experiment_id:
date:
parent_commit:
hypothesis:
change_scope:
datasets_and_sha256:
prediction_or_feature_sha256:
primary_metric:
secondary_metrics:
penalty_modes:
admission_threshold:
safety_checks:
runtime_environment:
result:
decision:
commit:
```

要求：

- 历史包是回归集，不伪装成新盲测；
- 不只报均值，必须保存逐文件残差、FP/FN、数量罚与最坏样本；
- T2 报 MAE、偏差、分位误差和跨包方向；
- T3 报 accuracy、每类召回、混淆矩阵和标签覆盖；
- T1 同时报告四种数量罚口径；
- 负结果同样进入 `failed-experiments.md`，避免重复试验。
