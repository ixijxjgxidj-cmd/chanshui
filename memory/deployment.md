# 部署事实与发布检查

更新时间：2026-08-11。本文不记录私钥、口令、代理、访问令牌或其他凭据。

## 当前状态

- 当前运行发布代码提交：`baa6f77a71519c9e496dc0c27c1f2659de130b7e`。该提交只修复 watchdog 跨用户临时日志；T1/T2/T3 推理模型与参数仍是已验收的七成员/SeismicXM 版本。
- 服务器仓库通过核验 SHA-256 的增量 Git bundle 严格 fast-forward；分支保持 `main`、工作树干净，未依赖服务器直接读取私有 GitHub。仅文档快进不需要重启服务。
- 服务：系统级 `phasepick-api.service`，已启用并正在运行。
- 运行身份：非 root 普通用户。
- 工作目录：服务器仓库目录；启动命令显式包含七成员、后处理参数及 T2/T3 模型选择。
- 本地回环与公网健康检查均返回 `{"status":"ok"}`。
- 公网真实样本：T1 返回 1P/1S；T2 返回震级；T3 返回类别；三个业务端点均为 HTTP 200。
- watchdog 使用“直接数值型 loopback + 0600 随机 token”双重认证，root crontab 中唯一 5 分钟条目已安装；手动与真实定时触发均为 OK，且不污染生产捕获。

## 2026-08-11 发布记录

- 服务器无法直接读取私有 GitHub 仓库，因此使用已核验 SHA-256 的增量 Git bundle，将 `47fb72c` 安全快进到 `53117b9`；未重写历史，也未覆盖服务器本地改动。
- 重启后 `phasepick-api.service` 为 `active`、`enabled`，仓库 HEAD 精确等于目标提交，工作树为空。
- 启动日志确认：T1 七成员顺序正确，短记录使用七成员、长记录限制为前五成员；T2/T3 SeismicXM 均已就绪；生产后处理参数完整打印；预热约 62 ms。
- 服务冷启动约 2 秒完成。第一次立即探活发生在端口监听前，随后回环与独立外网探活持续成功，不属于启动故障。
- 2 秒三分量 MSEED 线上隔离烟测：`/classify → {"SMK2": 5}`、`/magnitude → {"SMK2": 4.81}`、`/pick → {}`，全部 HTTP 200。这直接证明 T2/T3 已运行新短窗路径，同时 T1 的 5 秒保护仍生效。
- 烟测样本 SHA-256：`6c05420f02bba3a183a2f7b171703df5c588d041d63c31c37b28988d1a23150e`。三次测试捕获已移入服务器上的独立测试归档，生产 `captured/` 复核为空。
- 远端临时部署 bundle 已删除；该文件可由本地忽略副本重新传输，不影响回滚或审计。
- 部署验收文档提交 `c08bda5` 随后以增量 bundle 严格 fast-forward 到服务器；它不含 Python 变更，因此没有再次重启已验收服务。
- 新的专用 SSH 密钥已用非交互 `BatchMode` 直连核验成功；备用代理未使用。本文不记录地址、账号、密钥路径或代理凭据。
- 第 3 轮 T2 residual 校准在双向开发阶段失败，未读取 08、未产生生产 bundle；因此没有向服务器同步实验代码、没有重启服务，当前已验收运行代码和回滚入口保持不变。
- 本次发布当时登记的直接回滚目标为此前已验收的 `47fb72c655896913c656c0765acde57bd334c3a4`；该条仅描述当时状态。后续已完成真实回滚与再前滚验收，冻结前缺口已由本文后续记录关闭。
- 发布硬化实现提交为 `ee2337a`；阶段内再次用专用密钥非交互直连核验服务为 `active`。本轮只增加发布清单、部署前校验、文档/元数据脱敏和测试，没有生产推理变化，因此未同步服务器、未重启服务。

## 2026-08-11 watchdog、cron 与回滚验收

- 隔离实现提交 `c5153be` 部署前在 Linux 上通过 `26` 项专用边界测试、Python 编译、
  两个 shell 脚本语法和 `tracked=14 / external=1` 发布清单校验。
- 部署后 token 目录 `0700`、文件 `0600`，所有者与服务用户一致；token 内容未出现在
  systemd unit、进程参数或 journal。四端点均 HTTP 200，认证的三个业务探针都返回
  accepted，捕获波形与 manifest 增量为 `0`。
- 公网使用伪造 token 和伪造回环代理头时，响应没有 accepted，并恰好新增 `1` 条捕获；
  唯一测试记录按文件名、字节数和哈希精确归档后，生产捕获恢复为空。
- 第一次 root cron 模拟发现固定 `/tmp` 日志的跨用户覆盖会被 Linux sticky-dir 保护拒绝，
  导致成功探活被误判并触发一次不必要重启。cron 立即撤下；提交 `baa6f77` 改为每次
  `mktemp` 私有日志并在 EXIT 清理。修复后 Linux `26 passed`，root 手动与真实定时触发
  均 OK、无 FAIL、PID 不变、捕获增量 `0`。
- 已执行真实回滚：先停用 cron，切到部署前提交并恢复旧 unit；服务 active、四端点 200、
  token 仍以 `0600` 保留且旧 unit 不使用。三个回滚请求形成的捕获均精确归档并清空。
  再前滚后复用原 token、恢复 probe unit 与唯一 root cron；四端点、手动 watchdog 和真实
  定时触发再次全绿。
- 上述服务器双向验收、cron 与回滚/再前滚记录提交
  `b5f2afb214dde406f6b62d68e1ea2f35e19619e1` 已成功推送到 GitHub `origin/main`。
- 服务器 `.runtime` 中保留旧/新 unit、提交、root crontab及合成验收样本的回滚审计副本；
  这些运行时文件不进入 Git。最终服务 active、enabled，生产捕获为空。

## 性能验收

- 300 请求连续压测通过。
- 平均延迟约 42.17 ms，P95 约 44.91 ms，最大约 56.35 ms。
- 压测前后 RSS 增长约 4 MiB，未见持续泄漏趋势。
- 服务启动完成后常驻内存约 0.8–0.9 GiB，systemd 配有高水位与最大内存限制。
- 本次重启后 `systemctl status` 观测约 814.7 MiB，峰值约 876.2 MiB，仍处于既定限制内。

## 发布资产

- T1 七个成员的 SHA-256 见 `experiments/t1_g7_release_20260811.json`，已逐一核验。
- T2/T3 joblib bundle 已核验。
- SeismicXM 外部编码器 SHA-256：`671d02d677c25c3d075963889602299ec71f52c724470f2fa85bb28035fe1528`。
- `deploy/production_release_manifest.json` 当前记录 14 个 Git 跟踪资产和 1 个外置资产，并静态核对 `serve_api.py`、`deploy_api.sh` 与 manifest 的默认值。
- 生产启动前必须执行 `python scripts/verify_release_manifest.py --require-external`；不得只检查文件名或大小。
- 两份归档 checkpoint 与两份 baseline fallback joblib 已移除机器绝对路径元数据；checkpoint 逐一验证 174 个张量完全相同，joblib 固定探针预测一致。默认 SeismicXM 生产 bundle 未修改。

## 已知运维缺口

1. 服务器暂不能直接拉取私有 GitHub 仓库。后续配置只读 deploy key；不得把个人写权限密钥部署到服务器。
2. 比赛平台各任务 URL 仍需由用户按最终窗口核对/登记；本次没有修改平台配置。
3. 最终冻结前仍应做一次统一历史包复评与公网代表样本复验，但当前 watchdog、发布闸门和回滚路径均已在目标主机实测。

## 每次发布的最低检查

1. 本地工作区干净；提交已推送 GitHub。
2. 全量测试通过，部署脚本语法检查通过；运行 `verify_release_manifest.py --require-external` 全绿。
3. 服务器仓库提交与目标提交一致，且无未跟踪业务文件污染。
4. release manifest 中全部 14 个跟踪资产和外置 SeismicXM encoder 的大小、SHA-256 全部一致。
5. systemd unit 运行账户非 root；`active`、`enabled`。
6. 本机回环 `/health` 与公网 `/health` 成功。
7. 用隔离测试样本验证 `/pick`、`/magnitude`、`/classify`，并确认正式捕获目录未混入探针。
8. 执行短压测，记录 P50/P95/max、吞吐和 RSS 前后值。
9. 验证回滚路径，再宣布发布完成。
