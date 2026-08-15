# Round 58 发布候选复现产物

预注册配置已在远程独立目录 `/root/projects/round58_release_20260816/` 重新训练 7 个全量成员，仅使用 R1/R2 train：

- 多任务 B 预训练骨干
- R1/R2 loss 权重 1.75/1.0
- 40 epoch，lr=2e-4，wd=1e-3
- gain augmentation `10^U(-1.5,1.5)`
- 7 个成员 seed=9000+113*i
- 发布推理校准：alpha=.25，R1 slope=.85，R2 slope=.50

权重文件留在远程项目目录，配置与 SHA256 已提交到仓库。最终一次 holdout 评估结果见 `outputs/t2_round58_holdout/round58_holdout.json`：均值 149.29（R1 144.42，R2 154.16）。holdout 已封存，不再用于后续选择。
