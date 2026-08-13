# 当前状态（2026-08-12）

本文件是项目事实入口。只记录已经核验的事实、明确标注的历史声明和待验证问题；旧文档与本文件冲突时，以原始证据和本文件中的证据等级为准。

## 一句话结论

项目已有可复现的三任务冻结生产版本：T1 七成员概率集成、T2/T3 SeismicXM 特征模型、统一冻结基线和历史发布证据链均已完成。第 2–6 轮候选、已有 T1 候选四口径重排、第 7 轮基础 PhaseNet 蒸馏和第 8 轮包—记录层级等权蒸馏均未通过生产准入。层级等权候选在三折权重守恒、held-out 屏障和完整推理上全部通过，但 12 单元 worst/mean 为 `-8.6889/-3.3602`，仅 2/12 正向；它把 R1 默认回退缩至 `-2.3222`，却使 R2/08 变为 `-5.1944/-8.6889`，7 条长记录只有 1 条上涨。生产继续冻结 `g7`。没有真正独立的新包或官方规则变化时，应停止围绕相同三包继续扫描蒸馏权重、层级权重、截断或长度分段；当前服务器只承担训练、特征生成和实验计算，不作为最终部署服务器。公开数据可在线流式读取、按需缓存或完整下载，不要求预先离线化。

## 合规缺陷（2026-08-13 实测，最高优先级）

用户目标语句把四个包（`08-an.zip`、`08-exam.zip`、第 1 轮、第 2 轮）全部列为「测试数据集永远不参与模型微调与训练」。按此口径，以下三处生产默认**违规**：

| 资产 | 角色 | 违规证据 |
|---|---|---|
| `weights/aug/exam_aug6_r2train_sd.pt` | T1 集成成员 4 | 由 `outputs/train/r2train.h5` 微调；该池实测 1001 窗、来自 915 个互不相同的 `source_file`（`T1.A.Q0001.mseed` …），恰为第 2 轮 T1 文件数 |
| `weights/aug/crew_sp23_r2train_sd.pt` | T1 集成成员 5 | 同上；`deploy/WEEK_TRAINING_RUNBOOK.md` 记为「R2 真题锚点池」，同时用作 `--holdout` 与混池训练输入 |
| `weights/official_r1_to_r2/t2_seismicxm_r1r2.joblib`、`t3_seismicxm_r1r2.joblib` | `serve_api.py` 的 T2/T3 默认模型 | `scripts/train_seismicxm_t2.py:71`、`train_seismicxm_t3.py:74` 为 `fit(vstack([X_r1, X_r2]), concat([y_r1, y_r2]))` |

`AGENTS.md` 第 2 条写有「第 1、2 轮比赛数据……可按预注册协议参与实验」，与用户目标语句直接冲突。`git log -- AGENTS.md` 显示该句由本项目自身提交（`d1d7d52`→`75c17c6`，2026-08-12/13）加入，非用户撰写。**以用户指令为准**：R1/R2 自 2026-08-13 起为训练禁用，只可用于「不回归检查」。既有冻结资产保留不删除，但不得进入任何新发布清单。

赛区变更（本届广西地震局，去年四川地震局，去年数据仅供参考）使上述资产同时存在**第二重失效**：它们是四川域拟合物。

### 合规替换的实测代价：可忽略

同一冻结生产参数下（`cap_short_s=300 cap_max_p=1 cap_max_s=1 long_snr_db=-1 force_pair_mode=conditional long_dedup_s=20 ensemble_long_members=5`），R1/R2 仅作不回归检查：

| 配置 | 成员 | R1 | R2 |
|---|---|---:|---:|
| P0 生产冻结（含违规成员） | guangxi,jiangxi,shandong,exam_aug6_r2train,crew_sp23_r2train,geofon_m1,geofon_m3 | 1.786294 | 1.810140 |
| C1 最小合规替换 | guangxi,jiangxi,shandong,huanan,geofon_m2,geofon_m1,geofon_m3 | 1.786522 | 1.806667 |
| C2 华南域先验（去 shandong） | guangxi,huanan,jiangxi,guizhou,hunan,geofon_m1,geofon_m3 | 1.783983 | 1.811840 |

移除两个违规成员后 R1/R2 变化在 ±0.004 内；C2 在 R2 上高于 P0。**合规不带来可测代价。**
## 已核验的生产状态

- Git 主分支：`main`。
- 当前运行发布代码提交：`baa6f77a71519c9e496dc0c27c1f2659de130b7e`；T1/T2/T3 模型和参数仍为已验收七成员/SeismicXM 生产版本，本次新增内容只在发布可靠性层。服务器仓库允许继续快进仅文档提交而不重启服务。
- 最终复评时本地、GitHub `origin/main` 与服务器仓库的共同基线为 `5d014b4cb7ed1bba363cd3888fb693ecd93bd617`；服务器随后通过仓库级只读 GitHub SSH 发布链持续以 `fetch` + `merge --ff-only` 快进当前 `main`。T1 候选稳健性权威审计绑定 `79028e0cf78d7eedae5b86a8849e237b03babe9d`，运行服务没有因脚本、测试或文档提交重启。
- 第 4 轮结果 `90ff84f`、发布硬化 `ee2337a`、第 5 轮 gap mask 否证 `d4a311d`、第 6 轮 annotation 否证 `bbf6e45`、watchdog 隔离 `c5153be` 与跨用户日志修复 `baa6f77` 均已推送 GitHub。负实验没有进入生产推理路径。
- watchdog 服务器双向验收、cron 与真实回滚/再前滚记录提交 `b5f2afb214dde406f6b62d68e1ea2f35e19619e1` 已成功推送到 GitHub `origin/main`；后续状态提交不改变生产推理。
- 最终统一复评与公网代表样本记录提交 `141d514c4dafb1b3523325cc56337a2a610072f5` 已成功推送到 GitHub `origin/main`；它只增加冻结证据与 memory，不改变生产推理。
- 唯一远端：`https://github.com/ixijxjgxidj-cmd/chanshui.git`。
- 当前训练服务器的职责已收窄为训练、特征生成和实验计算，不承担最终生产部署；公开数据允许在线流式读取、按需缓存或完整下载。后续 Git 快进仅为运行实验代码，不触发服务、systemd、cron、捕获或发布操作。
- 上次核验时服务器工作区干净；新的专用 SSH 密钥已通过 `BatchMode` 直连验证，备用代理未使用。凭据本身不进入仓库或 memory。
- `phasepick-api` 是系统级 systemd unit，`active` 且 `enabled`，但进程以普通用户运行。
- 公网 `/health`、`/pick`、`/magnitude`、`/classify` 已再次用 51.5 秒真实三分量代表样本验收为 HTTP 200；T1 返回 1P/1S，T2/T3 均返回单台站合法非空结果，公网请求没有 accepted 探针标记。
- 新提交重启后，2 秒三分量线上烟测得到 `/classify={"SMK2":5}`、`/magnitude={"SMK2":4.81}`、`/pick={}`，确认 T2/T3 短窗修复已生效且 T1 保护未回归。
- 300 请求稳态验收：均值约 42.17 ms，P95 约 44.91 ms，最大约 56.35 ms，RSS 增长约 4 MiB。
- 发布链最终审计时生产捕获目录与 manifest 均为空；随后观察到 1 条非 probe 的公网 `/pick` 畸形上传（1 个 217 字节项，来源不是 loopback，文件仍完整）。服务器生产环境中的 ObsPy 无法将它解析为 MiniSEED，API 安全返回空对象；该记录保留为接口鲁棒性审计证据，不作为训练样本，也不误删为探针。
- 最新只读运维复核：公网 `/health` 为 HTTP 200 且 `status=ok`，systemd 没有 failed unit，服务器仓库所在文件系统约有 230 GiB 可用空间。服务器候选稳健性审计后完整回归 `352 passed in 4.34s`；临时测试依赖未写入生产 `.venv`，服务仍 active。
- 服务器已经配置只属于本仓库的只读 GitHub deploy key；私钥只保留在服务器且权限为 `0600`。仓库本地 SSH 命令固定 `IdentitiesOnly=yes` 与严格主机校验，GitHub 主机公钥来自官方 metadata API。`ls-remote`、`fetch`、`merge --ff-only` 均已通过，写入 dry-run 被 GitHub 拒绝，服务器远端只保留 GitHub。
- watchdog 捕获隔离已部署并完成双向验收：只有“直接数值型 loopback 对端 + 0600 文件中的随机 token”同时满足才跳过采集；代理头不参与授权，服务还必须返回 accepted。Linux 专用测试 `26 passed`；认证回环、手动 watchdog 和真实 cron 的捕获增量均为 `0`，公网伪造 token/代理头仍恰好产生 `1` 条捕获并已精确清理。
- `deploy/production_release_manifest.json` 已成为生产资产、参数和 fallback 策略的机器可校验清单；`scripts/verify_release_manifest.py --require-external` 已对本地 14 个跟踪资产和外置 SeismicXM encoder 全绿。
- `deploy/deploy_api.sh` 会在写 unit/重启服务之前运行发布校验。默认 `ALLOW_MODEL_FALLBACK=0`；缺少默认 SeismicXM encoder 时中止，只有显式设为 `1` 才允许 baseline 降级，已存在但哈希错误的 encoder 无条件拒绝。
- root cron 初验发现固定 `/tmp` 日志会因跨用户 sticky-dir 保护误判失败；managed cron 立即撤下，提交 `baa6f77` 改为每次 `mktemp` 私有日志。修复后 root 手动与真实定时触发均 OK、服务 PID 不变、捕获增量 `0`。
- 已执行真实回滚/再前滚：回滚前停用 cron，旧 unit 不含 probe 参数但 token 仍安全保留；旧版四端点 200，三条回滚捕获精确归档。再前滚复用原 token，恢复 probe unit 和唯一 root cron，四端点、手动与定时探活再次全绿。最终服务 active、enabled，服务器工作树干净，生产捕获为空。
- 仓库敏感扫描已不再发现服务器地址、账号、代理、密钥路径或比赛数据机器绝对路径。两份归档 checkpoint 清洗时逐一验证 174 个张量完全相同；两份 baseline joblib 清洗后固定探针预测一致。

## 当前生产模型

T1 成员顺序：

1. `guangxi`
2. `jiangxi`
3. `shandong`
4. `weights/aug/exam_aug6_r2train_sd.pt`
5. `weights/aug/crew_sp23_r2train_sd.pt`
6. `weights/geofon/geofon_m1_last_sd.pt`
7. `weights/geofon/geofon_m3_last_sd.pt`

T1 生产参数：

```text
cap_short_s=300
cap_max_p=1
cap_max_s=1
long_snr_db=-1
force_pair_mode=conditional
long_dedup_s=20
ensemble_long_members=5
```

T2/T3 均使用 SeismicXM middle 编码器与已发布 joblib bundle。外部编码器大小为 `207709060` 字节，SHA-256 为 `671d02d677c25c3d075963889602299ec71f52c724470f2fa85bb28035fe1528`。

## 官方历史包事实

| 数据包 | SHA-256 | T1 | T2 | T3 | 已知异常 |
|---|---|---:|---:|---:|---|
| 第 1 轮 | `beb93b7544718c3b05be9cd5f4f3cbf78f7be8a32c125a696adeb87c5d3a524e` | 1000/1000，全部 1P+1S | 200 | 200 | T3 含标签 1–5 |
| 第 2 轮 | `d5ffc69223ab75815618e7647e4212d39e4c0e756a91c1e6ee04cb55c29f54e6` | 915/915，2 个多事件文件 | 200 | 输入 190、答案 219、交集 189 | 30 个答案无输入，1 个输入无答案 |
| 08-exam | `560145f74f2d8861bb344a7396c15793a54f0ce3f4ad056dfa115cd7f756bb3b` | 784 | 200 | 205 | 与答案包配套使用 |
| 08-an | `a7978b61915313509a64d7b91e5609f86a55a7a3ce4b393c22780be9c7088b95` | 784 | 200 | 205 | T3 实际只出现标签 1–4 |

08 T3 实际类别分布为 `{1: 140, 2: 40, 3: 20, 4: 5}`。仓库历史文档中的“08 五类盲测 89.3%”不能原样继续引用；目前只能表述为“205 条、标签 1–4 的 08 包上 183/205”。

## 当前冻结基线与证据等级

统一冻结输出为 `outputs/frozen_baseline/baseline_full_profile_prod_20260811.json`，SHA-256 为 `2a55e4164db40a8eb87d6aa518fb040f11f7b2996788234f8fc1513bcfa3ac05`。三套历史包共 3,894 个 MSEED 均完成画像；全部为 100 Hz、无缺口、无缺分量。已知边界包括第 1 轮 T3 六个 1.5–3.5 秒文件、第 2 轮 T3 一个重叠段文件、第 2 轮 T1 两个多事件长记录，以及 08 T1 五个约 4,000 秒文件。

冻结前最终复评输出为 ignored `outputs/frozen_baseline/final_unified_20260811_5d014b4.json`，SHA-256 为 `6b5520fbb2b45fcb7a733dc7638681389a16c11bd16ed235b5ea97b363fb204a`。它在提交 `5d014b4` 上重新解析全部 3,894 个 MSEED；T1/T2/T3 完整任务对象与旧冻结输出逐字段一致，波形画像去除运行时和单文件读取延迟后也逐字段一致。最终数值仍为 T1 `1.7862944444 / 1.8101396478 / 2.0100836168`、08 T2 MAE `0.5231974550`、08 T3 `183/205=0.8926829268`。

T1 冻结结果同时报告四种数量罚解释：

| 数据包 | merged_file | per_phase_file | merged_exam | per_phase_exam |
|---|---:|---:|---:|---:|
| 第 1 轮 | 1.7862944444 | 1.7862944444 | 1.7862944444 | 1.7862944444 |
| 第 2 轮 | 1.8101396478 | 1.8095931998 | 1.8549483910 | 1.8549483910 |
| 08 | 2.0100836168 | 2.0075325964 | 2.0617417800 | 2.0611040249 |

08 T2 冻结 MAE 为 `0.523197455`，平均有符号误差为 `+0.099781905`。08 T3 为 `183/205=0.8926829268`；每类召回为 1 类 `0.992857`、2 类 `0.725`、3 类 `0.700`、4 类 `0.200`，答案中没有第 5 类。

这些是统一脚本、数据包哈希和预测哈希约束下的冻结回归结果，但仍不是真正盲测：历史包长期参与选型，且官方材料未唯一确定数量罚按文件/全卷及 P/S 合并/分开计算。所有新候选必须继续在同一冻结脚本下报告四种口径、跨包验证和逐类/逐文件结果；`geofon_m3` 精确训练配方与 `geofon_m1` 真正留出证据仍不完整。

## 已确认的实现风险

### 1. T2/T3 短窗曾被 T1 下限误删（已修复并发布）

统一摄入器原先对所有任务拒绝 `<5s` 波形。第 1 轮官方 T3 实际有 6 个 1.5–3.5 秒的合法第 5 类样本；生产 SeismicXM 离线对六条均分类正确，但旧 API 在模型前返回空表。提交 `53117b9` 已将 5 秒下限保留给 T1/依赖 picks 的模型，对 `needs_picks=False` 的 T2/T3 固定窗模型允许短窗。六个官方文件端到端恢复为类别 5；第 1 轮有效 T3 accuracy 从旧 API 的 91.0% 恢复为 94.0%，第 5 类 recall 从 40% 恢复为 100%。服务器 2 秒隔离烟测和公网健康检查均已通过。

### 2. 第 5/6 轮固定 gap 屏蔽均已否证，生产仍未接通 gap 规则

历史三包 3,894 个已画像文件没有真实 gap，因此本轮只验证未来输入鲁棒性，不能声称历史官方分数提升。读取层仍会记录 `Waveform.gaps` 并零填充，但生产 `picker.py` 没有消费该元数据；这次没有因实验失败而改默认路径。

第 5 轮同时用 AnySearch 与 Playwright 核验 17 篇与零填充边界、prediction inconsistency、gap augmentation、缺失数据重建和下游偏差直接相关的研究，并预注册最终 `Pick` 列表固定 margin mask。七成员、输入包/权重身份、注入逐位一致、无 gap 对象身份、single/batch、重复性和 P95 `2.1463 ms` 均通过。

但 R1/R2/固定噪声共 77 个变体产生 `31` 个 induced 和 `36` 个 lost，其中 `13` 个 induced、`2` 个 lost 位于物理 gap 10 秒之外。`margin=0` 只清掉 8/31 个 induced；`margin=10s` 仍留下全部 13 个远程 induced，并误删 37 个稳定 reference picks。没有 active margin，08 波形推理被锁住，`development_pass=false`。

稳定结论：零填充会改变更宽上下文内的概率或条件式 force-pair 结果，所有后处理完成后的删除层无法修复。不得扩大 margin、按文件/相位自适应或把 taper/interpolation 当作同轮补救。若继续研究 gap，作用点必须前移到 annotation/正常阈值/force-pair 之前，并另行预注册；完整证据见 `memory/experiments/005-t1-gap-mask-robustness.md`，结果提交为 `d4a311d`，已推送 GitHub；未部署。

第 6 轮已按上述重开边界把作用点前移到七成员 annotation 平均之后、任何正常阈值和 conditional force-pair 之前。17 篇新的 mask-aware convolution、缺失时序与 blackout imputation 论文形成硬假设：若 gap 10 秒外 raw probability 已改变，局部 annotation veto 逻辑上不能恢复。

实际 77 个冻结变体中，754,070 个远程 P/S 样点有 295,900 个变化超过 `1e-6`，最大绝对差 `0.609977`；出现 745 次正常阈值穿越、3,551 次 `0.03` floor 穿越、normal remote peak `12 induced / 2 lost`、floor remote peak `22 induced / 38 lost`。raw final 精确复现第 5 轮的 `31 induced / 36 lost / 13 remote induced / 2 remote lost`，且第 5/6 轮逐变体数组、区间、reference/raw picks 全部一致。

六档 annotation guard 均不可录取：0 秒为 `33 residual / 40 lost / 0 collateral`，10 秒为 `18 residual / 77 lost / 37 collateral`。结构、五条 annotation→生产 Pick 零差复刻、66 次 no-gap 身份、重复性和 P95 `1.6621 ms` 均通过；拒绝原因是阶段 A 科学必要条件失败。开发选择 `OFF`，08 保持 `records=null`，未改生产、未部署。完整证据见 `memory/experiments/006-t1-gap-aware-annotation.md`，ignored JSON SHA-256 为 `e39c261f61a81d1895c26fc4767d9a3a8ccefc85c8de921d30fa315d6437ada6`。

稳定结论升级为：现有冻结 PhaseNet 对 zero-fill 不具备显式 missingness 语义；不得继续扩大固定 guard、按结果自适应，或在同一零填充输出上叠加 NaN/taper/interpolation。只有真实 gap/独立 gap 包、可冻结 gap augmentation 训练划分、模型层 observation mask，或可验证的滑窗贡献重算机制出现时才重开。

### 3. T3 中心化/等类轻量头在跨包验证中崩落（已拒绝）

第 2 轮同时使用 AnySearch 与 Playwright 核验 15 篇新的原始实验论文，并预注册冻结 SeismicXM、只比较 26 个轻量头。第 1 轮 5×5 嵌套验证中，候选将 accuracy/balanced accuracy 从 `0.9080/0.865636` 提高到 `0.9240/0.908364`，但从第 1 轮训练到第 2 轮仅 `121/189`；R1 margin/support 门控覆盖第 2 轮 91.5%，也只有 `130/189`，远低于生产基线 `187/189` 和准入线 `186/189`。

25 个外层折中 23 个选择中心化家族。事后只用于解释的成对诊断显示，对应 hybrid 去掉中心化可从 `121/189` 恢复至 `178/189`，top-m 可从 `133/189` 恢复至 `182/189`，但仍均未达门槛。结论是训练包均值中心化放大域移，而当前等类局部/原型头本身也不够稳。按预注册没有提取或查看 08 T3 特征，没有构建 bundle、改 API 或部署。完整证据见 `memory/experiments/002-t3-long-tail-domain-generalization.md`。

### 4. T2 源包 OOF residual 校准包内上涨、跨包反向（已拒绝）

第 3 轮核验 16 篇新的震级估计、分布偏移和不确定性原始研究，并预注册 12 个只使用源包 OOF residual 的低维局部校准配置。R1/R2 都独立选择 `pca0_k40_s50`；重复嵌套 OOF MAE 分别从 `0.293562→0.275316`、`0.222057→0.213599`，配置稳定性通过。

但双向跨包中，R1→R2 从 `0.621046` 恶化到 `0.621532`，signed bias 从 `-0.589283` 变为 `-0.589620`；R2→R1 从 `0.660726` 恶化到 `0.669618`，signed bias 从 `+0.539461` 变为 `+0.552312`。候选在两个方向的平均修正方向都没有提供目标包所需的全局截距纠偏，源距离闸覆盖 `90–93%` 且不能识别危险样本。`development_pass=false` 后没有计算或打开 08 T2 缓存。完整证据见 `memory/experiments/003-t2-cross-package-residual-calibration.md`。

稳定结论：当前 T2 最大限制更像绝对幅值/位置/距离信息缺失、训练分布覆盖不足或真实条件/标签漂移，而不是 Ridge 后面缺少 source-only 小校准头。不得继续扩大相同 PCA/kNN residual 网格，也不得用包均值差或 08 偏差硬加常数。

### 5. 第 4 轮长记录事件级联合置信度过滤已否证

冻结残差显示，R2 两个长文件占该包 T1 损失约 `23.5%`，08 五个约 4000 秒文件占约 `39.4%`；7 文件在生产 20 秒去重后仍有 R2 `126/37`、08 `254/96` 的 FP/FN。前置只读诊断中，完整假 P/S 事件多于孤立假相位，事件几何联合 confidence 区分 MM/FF 的 AUC 为 `0.7616`，但历史短文件联合重选在 R2/08 仅 `+0.0/+0.1`，不能重开短文件规则。

本轮已同时使用 AnySearch 与 Playwright 核验 16 篇新的原始实验研究。成熟 PhaseLink/GaMMA/PyOcto/GENIE 等方法依赖多台站、位置和速度模型，当前单站输入不能直接使用；EQTransformer/CRED/PhaseNet+ 等支持“完整事件证据优于独立相位峰值”。实验严格只处理 `>300s`，固定生产 20 秒去重、FIFO `0.2–60s` P→S 配对、几何联合分数和 tau `{0.35,0.40,0.45}`。

OFF 路径已逐文件和四种完整包口径复现到 `1e-9`。R2 tau 0.35/0.40/0.45 分别删除 `29/39/54` 个事件，FP `126→87/77/65`，但 FN `37→56/66/84`，最坏归一化增益 `-0.103991/-0.161175/-0.262292`。08 tau 0.35 删除 44 个事件，FP `254→186`，四种整包总分均上升 `+9.0944～+16.0944`，但 FN `96→116`，并使 4/5 个文件新增 FN 或时差分下降；更高阈值更差。R2、08 full-source 和全部 7 个 LOFO 子集均选择 OFF，`development_pass=false`。

稳定结论：简单事件联合 confidence 能减少假事件，但无法安全区分弱真实完整事件；整包数量罚上涨不能覆盖逐文件真实相位损失。不得扩阈值、改变 60 秒窗口或事后换 confidence 组合，不改生产、不部署。结果与逐文件证据见 `memory/experiments/004-t1-long-record-event-confidence.md`；ignored 结果 SHA-256 为 `dc55ae264e8ff830eafdd809e7d12b0c4bd3dc2608a2d427cc7cd753e17f5b9d`。

### 6. 数量罚口径不确定

代码实现四种读法：`merged_file_floor0`、`per_phase_floor0`、`merged_exam`、`per_phase_exam`。现有 PPT 只给出“误差 5% 内不扣，每超一个扣 0.5”，未说明按文件/全卷、P/S 合并/分开。默认值只是历史兼容，不是已证明的官方真值。

### 7. 历史 T1 预测已排除贪心匹配敏感性，官方未来规则仍未公开

提交 `4bf7d5158f5b9f408043e6dd4a65d7518bcdb5db` 增加独立 Hungarian 审计器，同时比较“最大官方时差分”和“最大匹配数后最小总残差”两种精确目标。第 1 轮 1,000 个、第 2 轮 915 个、08 784 个文件的冻结生产预测在 P/S 上逐文件差异全部为 `0`，包括第 2 轮 2 个和 08 5 个密集多事件文件；匹配数也全部一致。

审计器用合成密集反例检出精确匹配比贪心高 `0.1111111111`，证明检查不是恒等比较，也说明贪心并非数学上普遍等价。因此当前三包冻结分数不受此实现近似影响，但官方若给出新密集分布或明确匹配算法，仍必须重新审计；详见 `memory/experiments/009-t1-matching-sensitivity-audit.md`。

### 8. 所有已有 T1 候选均未通过三包四口径稳健替换

提交 `79028e0cf78d7eedae5b86a8849e237b03babe9d` 增加统一候选审计器，复用 `score_file/exam_total_score` 对 `base/cond/fp/g6/g6gate/g7/ov90/prod2` 的三包预测执行四种数量罚重排。八组预测在第 1 轮 1,000 个、第 2 轮 915 个、08 784 个文件上覆盖均完整；`g7` 四口径均分逐位复现冻结计分板。

除 `g7` 外 7 个候选均未满足“12 单元全部不下降且至少一个严格上升”。最佳 `ov90` 的最差完整包总分差为 `-6.150000000001`，12 单元均值为 `-3.097222222223`，只有第 2 轮两种卷级口径各约 `+0.8611`；第 1 轮和 08 四口径全部下降，默认口径三包合计 467 个文件受损。服务器权威 JSON SHA-256 为 `98353140ad1f0958d2caab60aa92c4ec8e1bf229b0085cead2cb838da140f988`。

稳定结论：继续冻结生产 `g7`，不得从这些同一批历史输出中事后挑包、挑口径或重排后声称提升。只有新独立数据、官方规则实质变化，或完全不同且预注册后 12 单元全不下降的新候选才能重开；详见 `memory/experiments/010-t1-candidate-robustness-audit.md`。

### 9. PhaseNet 三折蒸馏已完成，基础候选未录取

提交 `a846c26118195f8c63cb8d26f0cfe5ebac3bd7bc` 的 CPU 基准完整盘点三套包 2,699 个 T1 文件、232,206.05 秒波形和 7 个长文件；代表 annotation 确认 N/P/S 三通道、50 Hz 概率网格和正确长记录长度。缓存预算为 `132.1285s`、`65,137,386` 字节；正式缓存实际为 `466.7212s`、`65,136,648` 概率字节。实际时间包含 ZIP/ObsPy 读取、逐模型调用、写盘、哈希和 `fsync`，约为预算 3.53 倍但仍远低于 12 小时门槛。manifest SHA-256 为 `645f8d04bc1a167f016594b94bbf5ed11495d8ebb51c0adac1bc7cb858555f5b`。

同架构 `PhaseNet(diting)` 六次固定训练全部完成：每折 3,939–4,698 个窗，训练约 181–225 秒，峰值 RSS 约 676–754 MB，loss 均下降；held-out 在训练和 early stopping 中保持关闭。六份单学生预测覆盖完整，吞吐约 195–229 文件/秒，并运行完整生产后处理。

绑定 `f508480b537b968ed67d3210beb5782d6589fc01` 的权威审计 JSON SHA-256 为 `ff98f656505e213f55c70a951ad4abdc8a07631b00d360d014cf35806d28b77d`。`KD-only` 仅 08 四单元上升，12 单元最差/均值为 `-10.4111/-2.9944`；`KD+hard` 在 R2、08 的 8 个单元及全部 7 个长记录上上升，12 单元均值 `+6.3491`，但 R1 四单元全部下降，最差 `-6.2722`。两者均 `robust_pass=false`，不部署、不改生产。不得在相同包上继续扫 alpha、温度、epoch、成员权重或后处理；完整证据见 `memory/experiments/012-t1-phasenet-lopo-distillation.md`。

### 10. 包—记录层级等权 KD+hard 已完成，未录取

第 8 轮用 AnySearch MCP 与本机 Playwright 核验 15 篇此前未计数的域泛化、群组稳健和跨尺度地震拾取原始实验论文，并在实现前只冻结一个变量：两个训练包等权、每包内记录等权、记录内窗口等权；模型、KD/hard 比例、10 epoch、阈值和后处理均不变。

绑定 `4edc4c55a3bd9a189f7ac888cc066a529a06e7f5` 的新教师 manifest SHA-256 为 `0a15cabc28839673e173816915d3993bb9cd4e19bd49ceab0ac78a978c0d2015`，覆盖 2,699 文件、耗时 `474.7179s`，与旧缓存逐条 SHA-256/shape/active-member 数完全相同。三折权重均值约为 1，两个训练包总权重相等，每包内记录总权重相等；held-out 波形和答案训练阶段均未打开。

权威审计 JSON SHA-256 为 `8ffe1f5cd419cdcaaad4066cc3e3036569fb01bd324ba6c7e26d7a807ad5f009`。候选 12 单元 worst/mean 为 `-8.6889/-3.3602`，仅 R2 两个卷级口径上升。默认口径 R1/R2/08 分别为 `-2.3222/-5.1944/-8.6889`；R2/08 新增 `12/19` FP。七条长记录仅 1/7 上升，合计 `-14.7222`。候选拒绝，不部署；完整证据见 `memory/experiments/013-t1-package-record-balanced-distillation.md`。

稳定结论：按窗口 ERM 会让多窗记录贡献更大，但完全包—记录等权又过度削弱长记录和复杂事件结构。不得在相同三包上继续扫描包权/记录权插值、clip、floor、长度分段、长记录保权或按结果选风险。这些都属于同一失败机制的连续调参，不构成新研究。

### 11. 训练谱系不完整

`geofon_m3` 缺精确训练配方；`geofon_m1` 无真实留出。它们可作为已冻结发布资产继续使用，但在补齐谱系前不能宣称完全可重训。

## 下一步顺序

1. 优先获得真正独立、未参与选型的新 T1 包，或等待官方明确数量罚/匹配规则；在此之前不继续围绕同一三包训练蒸馏权重变体。
2. 若出现新独立包，先原样盲评冻结 `g7`、基础 `KD+hard` 和层级等权候选，用它判断历史三包结论是否外推；不得先用新包调参。
3. 若没有新数据但必须继续研究，只允许不属于连续样本权重调节、且由新原始证据在实现前严格预注册的机制；仍需三包 × 四口径全不下降。
4. 服务器继续只做训练实验；数据可在线读取或缓存，但不部署、不修改 systemd/cron/API，不把缓存、checkpoint、预测、日志或私有审计 JSON 提交仓库。
5. T2/T3 与 gap 方向继续冻结，除非出现各自已记录的重开条件。
