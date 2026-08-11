# 发布可靠性阶段 008：最终冻结复评与公网代表样本

- 日期：2026-08-11
- 状态：**已完成；生产模型与参数保持冻结**
- 基准分支：`main`
- 被复评提交：`5d014b4cb7ed1bba363cd3888fb693ecd93bd617`
- 记录提交：待本文件提交并推送后回写
- 性质：冻结前最终证据复核，不是新模型实验，不宣称新的历史包分数增益

## 1. 目标与成功条件

目标是关闭两个冻结前缺口：

1. 在当前代码与相同冻结预测身份下重新解析全部历史输入，确认 T1/T2/T3 数值、覆盖率、逐文件结果和波形画像没有漂移；
2. 从非服务器机器经公网使用真实三分量代表样本复验 `/health`、`/pick`、`/magnitude`、`/classify`，并在测试后恢复空生产捕获。

预登记通过条件：

- 四个官方归档的 SHA-256 与已冻结身份一致；
- 三个 T1 预测和 08 T2/T3 预测哈希不变；
- 全部任务结果与旧冻结 JSON 逐字段一致；
- 3,894 个 MSEED 全部画像成功，结构统计不漂移；
- 公网四端点均 HTTP 200，模式合法，真实样本 T1 至少返回 1P/1S，T2/T3 非空；
- 公网请求不得返回 probe accepted；
- 测试捕获必须按唯一身份精确治理，最终生产 manifest 与波形数都为 0。

## 2. 输入身份与运行命令

输入继续使用已冻结的第 1 轮、第 2 轮、08 试题和 08 答案包；四个 SHA-256 分别为：

- `beb93b7544718c3b05be9cd5f4f3cbf78f7be8a32c125a696adeb87c5d3a524e`
- `d5ffc69223ab75815618e7647e4212d39e4c0e756a91c1e6ee04cb55c29f54e6`
- `560145f74f2d8861bb344a7396c15793a54f0ce3f4ad056dfa115cd7f756bb3b`
- `a7978b61915313509a64d7b91e5609f86a55a7a3ce4b393c22780be9c7088b95`

运行入口：

```text
python scripts/freeze_baseline.py \
  --dataset round1=<R1_ZIP> \
  --dataset round2=<R2_ZIP> \
  --dataset final08=<08_EXAM_ZIP>::<08_ANSWER_ZIP> \
  --t1-pred round1=outputs/port_verify/r1_g7.an \
  --t1-pred round2=outputs/port_verify/r2_g7.an \
  --t1-pred final08=outputs/port_verify/f08_g7.an \
  --t2-pred final08=outputs/frozen_baseline/final08_seismicxm_prod/T2.pred.an \
  --t3-pred final08=outputs/frozen_baseline/final08_seismicxm_prod/T3.pred.an \
  --profile-waveforms --workers 8 \
  --output outputs/frozen_baseline/final_unified_20260811_5d014b4.json
```

输出为 ignored 文件，SHA-256：

```text
6b5520fbb2b45fcb7a733dc7638681389a16c11bd16ed235b5ea97b363fb204a
```

运行耗时约 `8.01s`，进程峰值 RSS 约 `965,726,208` 字节。该耗时主要是解析、评分和画像，不包含重新推理；预测文件身份由 SHA-256 锁定。

## 3. 统一复评结果

| 任务 | 数据包 | 指标 | 最终复评 |
|---|---|---|---:|
| T1 | 第 1 轮 | `merged_file_floor0` 均分 | 1.7862944444 |
| T1 | 第 2 轮 | `merged_file_floor0` 均分 | 1.8101396478 |
| T1 | 08 | `merged_file_floor0` 均分 | 2.0100836168 |
| T2 | 08 | MAE | 0.5231974550 |
| T3 | 08 | accuracy | 183/205 = 0.8926829268 |

确定性比较结果：

- 第 1 轮、第二轮和 08 的 T1 完整任务对象与旧冻结输出逐字段相同；
- 08 T2/T3 完整任务对象逐字段相同；
- 三包波形画像只有总运行时与异常样本的读取延迟字段不同；去除这两个运行时字段后，结构画像逐字段相同；
- 第 1 轮 `1,400`、第 2 轮 `1,305`、08 `1,189` 个文件全部画像成功，失败、gap 与缺分量均为 `0`。

因此没有代码漂移、预测身份漂移或评分漂移；旧冻结事实仍成立。

## 4. 公网四端点复验

先使用唯一三分量合成事件样本验证接口安全边界：四端点均 HTTP 200，JSON 模式合法，T2/T3 非空，公网响应均无 accepted；T1 返回空 P/S，因此该样本只计接口安全证据，不冒充真实域代表样本。三条业务捕获按文件身份核验后精确归档，生产捕获恢复为空。

随后从已核验的历史归档中只读抽取一条 51.5 秒、100 Hz、三分量真实样本，不记录其文件名或机器路径。公网观测：

| 端点 | HTTP | 结果约束 | 本机观测往返 |
|---|---:|---|---:|
| `/health` | 200 | `status=ok` | 224.8 ms |
| `/pick` | 200 | 单台站、1P/1S、UTC 列表有序且格式合法 | 226.8 ms |
| `/magnitude` | 200 | 单台站浮点值 | 214.9 ms |
| `/classify` | 200 | 单台站整数类别 | 201.6 ms |

三个业务响应都没有 `X-PhasePicker-Probe: accepted`，证明公网普通请求没有被误当作认证回环探针。

## 5. 捕获治理与服务器终态

- 真实样本产生的三份测试波形先逐一核验原始名、字节数和内容哈希；
- 核验后从生产 manifest 原子移除，并删除服务器测试副本；原始官方归档仍在本地只读保留，可恢复；
- 服务器只保留不含波形内容的审计元数据；
- 本地两个临时样本均已删除；
- 最终生产 manifest 行数 `0`、捕获波形数 `0`；
- 服务 `active`、`enabled`，unit probe 参数数量 `1`，token 模式 `0600`，root managed cron 数量 `1`，最近一次 watchdog 为 OK。

## 6. 结论

最终冻结复评通过。当前七成员 T1、SeismicXM T2/T3、生产后处理参数与部署配置继续保持不变。下一最高价值事项是为服务器配置仓库级只读 GitHub 发布路径，替代后续手工 bundle；不得向服务器部署个人写权限凭据。
