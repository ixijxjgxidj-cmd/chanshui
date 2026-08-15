# 最终交付审计报告

日期：2026-08-16

## 结论摘要

本仓库包含两条独立工程线，不能混用分数：

1. **T1：官方 2026 公网 HTTP 震相拾取 API。** 这是仓库 README 和 release manifest 定义的主交付。当前生产候选为 `t1_g7_release_20260811`，包含 7 个有序成员、SHA-256、API 服务和部署校验。
2. **T2：单台站震级研究线。** Round 58 的预注册配置在一次 holdout 验证上得到均值 149.29；该结果不能替代 T1 官方总分，只能作为研究候选和专家评审材料。

## T1 交付证据

- `python -m pytest -q tests`：**376 passed, 1 skipped**。
- `python scripts/verify_release_manifest.py --require-external`：**PASS production-g7-seismicxm-20260811**。
- T1 历史冻结回放（manifest）：round1 `1.786294`，round2 `1.810140`，final08 `2.010084`。
- 生产入口：`scripts/serve_api.py`；响应契约为台站键下的 `P`/`S` UTC 到时数组。
- 生产配置、成员顺序、权重哈希和外部 SeismicXM 资产哈希见 `experiments/t1_g7_release_20260811.json`。

## T2 研究证据

Round 58 预注册配置：多任务预训练（震级、距离、深度）+ R1/R2 loss 权重 `1.75/1.0` + 7 成员 + alpha `.25` 部分池化中心校准。

- R1：144.42
- R2：154.16
- 均值：**149.29**
- 相对 T2 冻结 42c 参考均值 140.76：`+8.53`

该 holdout 已按预注册配置读取一次；之后所有 Round 59–67 分析只使用训练侧 OOF 或合成先验，未再次读取 holdout。

## 数据合规

- `08-an.zip`、`08-exam.zip` 及其衍生物没有进入任何新训练、微调、调参、早停、候选筛选或错误诊断。
- R1/R2 只在预注册事件级 train/holdout 划分下使用；训练侧实验没有读取 holdout 标签。
- 本地没有执行训练或数据集下载；训练和公开数据缓存均在远程服务器。

## 已知边界

- 旧 `t2_predict.py` 属于 Round 45 旧模型，不能用于 Round 58 `NetMTL` 发布权重；两条推理管线必须分开。
- Round 63–67 的无标签强校准在未知先验压力测试中并不普适，已否决固定 alpha `.75` 和简单阈值门控。
- 正式发布不得把 T2 149.29 写成 T1 官方总分，也不得把历史 final08 回放写成盲测成绩。

## 发布前检查清单

- [x] T1 API 测试通过。
- [x] T1 release manifest 与外部权重校验通过。
- [x] T2 Round 58 权重 SHA-256 全部通过。
- [x] GitHub 已持久化配置、哈希、实验记录和文献库。
- [x] 08 数据隔离规则保持有效。
- [x] T1/T2 分数口径已明确分离。
