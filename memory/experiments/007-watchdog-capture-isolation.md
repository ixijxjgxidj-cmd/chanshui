# 发布可靠性阶段 007：watchdog 回环探针捕获隔离

- 日期：2026-08-11
- 状态：**已实现、提交、服务器双向验收、安装 cron，并完成真实回滚/再前滚演练**
- 基准分支：`main`
- 基准提交：`70a3ecf`
- 隔离实现提交：`c5153be`
- 跨用户临时日志修复：`baa6f77`
- 性质：发布可靠性与数据治理，不是 T1/T2/T3 模型提分实验；不宣称官方分数增益

## 1. 当前问题

生产服务默认启用 `--capture-dir`，把真实评测请求的原始波形和响应落盘。`/health` 不经过推理引擎，无法发现全局推理锁被挂死请求占住的“半死”状态，因此评测日 watchdog 必须定期发送真实 `/pick` 请求。

当前 `deploy/watchdog.sh` 每 5 分钟发送一次真实波形，但没有安全的 skip-capture 协议；安装 cron 后一天会制造约 288 条探针记录。按文件名或 `client_ip=127.0.0.1` 事后过滤容易把探针误当比赛数据，也不能在写盘前证明请求身份。当前服务器因此没有安装 watchdog cron。

## 2. 威胁模型与禁止做法

公网请求可以自行设置任意 HTTP header，包括：

- `X-Forwarded-For: 127.0.0.1`；
- `X-Real-IP: 127.0.0.1`；
- 公开约定的“probe/skip capture”标记。

这些 header 只能用于捕获元数据，不能作为授权依据。禁止：

1. 仅凭固定 header 名或固定公开值跳过捕获；
2. 仅凭 `X-Forwarded-For` / `X-Real-IP` 判断回环；
3. 仅凭文件名、station、User-Agent 或 URL 参数判断 watchdog；
4. 把随机令牌直接写入仓库、systemd unit、启动命令、日志或 cron 文本；
5. 认证失败时静默继续并让 watchdog 误以为没有采集。

## 3. 冻结协议

### 3.1 双重授权

请求只有同时满足以下两项，才是 trusted probe：

1. ASGI `request.client.host` 的直接 TCP 对端由 `ipaddress.ip_address` 判定为 loopback；
2. 请求头 `X-PhasePicker-Probe-Token` 与服务启动时从令牌文件读取的值通过 `secrets.compare_digest`。

`client_ip_of()` 仍可从代理头恢复评测方 IP并写入 manifest，但它与 trusted probe 授权完全分离。IPv4/IPv6 回环均允许；主机名、私网 IP、代理头中的回环值均不允许。

即使公网知道 header 名称，或伪造全部转发头，也缺少直接 loopback 对端。即使令牌意外泄漏但 TCP 对端非 loopback，也不得跳过捕获。若未来使用本机反向代理，边缘代理还应显式删除外部同名 header；双条件中的令牌继续提供第二道边界。

### 3.2 令牌文件

- 默认路径：仓库运行时目录 `.runtime/watchdog_probe_token`；整个 `.runtime/` 必须 Git ignore。
- 部署脚本在文件不存在时用 `secrets.token_urlsafe(48)` 生成；已存在时保持不变。
- 目录权限 `0700`，文件权限 `0600`；systemd 路径归属运行服务的普通用户。
- API 只接受长度至少 32、至多 512 字符的非空令牌；文件缺失、过短、过长或读取失败时拒绝启动，不降级为不安全模式。
- 日志只打印“回环探针隔离已启用”，不得打印令牌或文件内容。

### 3.3 请求与确认

- `/pick`、`/magnitude`、`/classify` 及其别名共用同一 trusted probe 判定。
- trusted probe 正常执行完整模型推理，但不挂 `capture_save` 后台任务。
- trusted probe 响应增加 `X-PhasePicker-Probe: accepted`；普通请求响应体和既有 header 保持不变。
- `scripts/check_api.py --probe-token-file <path>` 从文件读取令牌、发送认证 header，并要求每个响应都带精确的 accepted header；否则自检返回非零。
- watchdog 默认读取同一令牌文件。文件缺失时直接失败，不发送会污染捕获的降级请求。

## 4. 部署脚本冻结行为

`deploy/deploy_api.sh` 必须：

1. 继续在任何服务重启前运行 release manifest 校验；
2. 创建/复用令牌文件，不打印令牌；
3. 为 systemd 用户设置正确目录与文件权限；
4. 通过 `--probe-token-file` 只传文件路径，不把令牌放进进程参数；
5. 部署后的合成 `/pick` 自检使用 probe 模式并要求 accepted，从而不污染 `captured/`；
6. 不改变任何生产模型、阈值、成员顺序、fallback 或捕获默认开关。

## 5. 预登记验收条件

### 5.1 单元与端到端边界

必须全部通过：

1. 正确令牌 + IPv4 loopback：完整推理成功、accepted header 存在、没有任何捕获文件；
2. 正确令牌 + IPv6 loopback：同上；
3. loopback + 缺失/错误令牌：正常捕获，无 accepted header；
4. 非 loopback + 正确令牌：正常捕获，无 accepted header；
5. 非 loopback + 正确令牌 + 伪造 XFF/X-Real-IP 回环：仍正常捕获；
6. API 未配置 probe token 时，任何 header 都不能跳过捕获；
7. 普通请求的 `client_ip_of()` 代理头记录行为不变；
8. `check_api` probe 模式只有在 accepted header 存在时返回成功；
9. token 文件缺失、过短、过长均拒绝加载，合法文件不泄漏内容；
10. capture 关闭时 API 行为保持不变。

### 5.2 发布验证

- 定向测试与全量测试全部通过；
- `bash -n deploy/deploy_api.sh deploy/watchdog.sh` 通过；
- release manifest 校验通过；
- Git diff、CSV、敏感信息和机器绝对路径扫描通过；
- 仓库不出现 token 文件或 token 内容；
- 本机进程参数只出现 token 文件路径，不出现 token 值。

## 6. 失败与回滚

任一边界测试失败，或 watchdog 不能确认 accepted header，则不安装 cron、不部署。本轮只允许回滚这些运维文件，不触碰模型配置。

正向结果部署后，先手动运行 watchdog，并在运行前后核对生产捕获 manifest/文件数完全不变；再从公网发送同名 header 的正常请求，确认仍新增捕获。只有这两个方向都通过，才允许安装 5 分钟 cron。

回滚方式：恢复部署前 `serve_api.py` 与 systemd 启动命令并停用 cron；捕获默认仍保持开启。任何已生成 token 只属于服务器运行时秘密，不进入 Git 历史。

## 7. 本地实现与验证结果

已实现：

- 新增 `src/phasepicker/probe_auth.py`，集中定义 token 文件校验、数值型 loopback 判定与恒定时间认证；
- `serve_api.py` 对 `/pick`、`/magnitude`、`/classify` 及别名使用同一 trusted probe 边界；
- `check_api.py --probe-token-file` 发送认证请求并强制要求 accepted header；
- `deploy_api.sh` 在 release 校验后、服务重启前只在固定 `.runtime` 目录生成或复用 token，并拒绝符号链接；
- 部署合成烟测使用 probe 模式，不再污染生产捕获；
- `watchdog.sh` 对 token 文件和数值型 loopback URL fail closed；
- `.runtime/` 已加入 Git ignore。

本地定向验证：

```text
25 passed, 1 skipped, 1 warning  # 专用安全边界测试
342 passed, 1 skipped, 7108 warnings  # 全量回归
python compile = pass
bash -n deploy/deploy_api.sh deploy/watchdog.sh = pass
release manifest = PASS, tracked=14, external=1
```

边界覆盖包括 IPv4/IPv6 loopback、缺失/错误 token、非 loopback 即使持有正确 token、非 loopback 同时伪造 XFF/X-Real-IP、服务未配置 token、三个业务端点、token 文件缺失/过短/过长/非法字符，以及 check_api 未获 accepted 时非零退出。所有测试通过；公网元数据 header 没有进入授权判断。

## 8. 服务器双向验收

服务器先以增量 Git bundle 严格 fast-forward 到隔离实现提交；重启前闸门全部通过：

```text
Linux 专用边界测试 = 26 passed, 1 warning
python compile = pass
bash -n = pass
release manifest = PASS, tracked=14, external=1
```

正式部署后已核验：

- systemd 服务为 `active`、`enabled`，仍以非 root 普通用户运行；
- token 是普通文件而非符号链接，目录 `0700`、文件 `0600`，所有者与服务用户一致；
- unit 只含 `--probe-token-file` 路径，token 值不在 unit、进程参数或 journal 中；
- 启动日志确认 T1/T2/T3 均正常加载，无模型 fallback；
- `/health`、`/pick`、`/magnitude`、`/classify` 均为 HTTP 200；三个业务端点的认证回环请求均返回 accepted；
- 部署烟测、三业务端点认证探针和手动 watchdog 的捕获波形增量、manifest 增量均为 `0`。

公网反向验证使用唯一合成 MiniSEED，并同时伪造 probe token、`X-Forwarded-For` 与
`X-Real-IP` 回环值：请求仍为 HTTP 200，但响应没有 accepted；服务器恰好新增 `1`
条 manifest 和 `1` 个波形，文件名、字节数与哈希逐项匹配。该唯一记录随后被精确移入
`.runtime` 测试归档，manifest 原子重写只移除对应行，生产捕获恢复为空。

## 9. root cron 暴露的问题与修复

第一次用 root 的最小 cron 环境手动运行时，推理本身成功，但旧脚本试图覆盖普通用户先前
创建的固定 `/tmp/phasepick_watchdog_last.log`。Linux sticky-dir 的
`fs.protected_regular` 边界使重定向失败，watchdog 因而误判失败并触发了一次不必要重启。
managed cron 随即撤下，服务恢复 active，捕获仍为空。

修复提交 `baa6f77` 将固定日志改为每次运行由 `mktemp` 创建的私有临时文件，并用 EXIT
trap 清理；临时文件创建失败时 fail closed。新增回归断言禁止重新引入固定 `/tmp` 路径。
修复后：

- 本地专用测试 `25 passed, 1 skipped`，全量 `342 passed, 1 skipped`；
- Linux 专用测试 `26 passed`；
- 在旧冲突文件仍存在时，root watchdog 返回 OK、服务 PID 不变、捕获增量 `0`；
- 唯一 root cron 安装后，手动 cron 环境与真实 5 分钟定时触发均产生 OK、无 FAIL，
  服务 PID 不变，捕获和 manifest 增量均为 `0`。

## 10. 真实回滚与再前滚

回滚前保存当前 unit、提交与 root crontab，并先撤下 managed cron。使用
`git switch --detach` 切到部署前提交，未使用 `reset --hard`；恢复旧 unit 后：

- 服务 `active`、`enabled`，unit 不含 probe 参数；
- token 文件仍安全保留且模式为 `0600`，但 cron 为 `0`，不会由旧 watchdog 误用；
- 启动无 fallback，四端点均 HTTP 200；旧版三个业务请求均不返回 accepted；
- 三个唯一回滚测试请求按预期形成 `3` 条 manifest 与 `3` 个波形，完整核验后精确归档，
  生产捕获再次恢复为空。

随后切回 `main` 并重跑正式部署脚本。原 token 被复用，unit 恢复 probe 参数；三业务端点
再次全部 accepted 且捕获增量 `0`。root crontab 从备份恢复为唯一 managed 条目，手动和
真实定时触发均为 OK、无 PID 变化。最终服务器停在 `baa6f77`，工作树干净，服务 active，
生产捕获为空，watchdog cron 数量为 `1`。

## 11. 结论

本阶段通过生产准入并已部署。它不改变 T1/T2/T3 推理结果或历史官方分数，只提高探活、
捕获数据治理和回滚可靠性。稳定规则为：watchdog 跳过采集必须同时满足直接数值型
loopback 与随机 token；代理头永不授权；cron 必须以 root 唯一安装；临时日志必须每次
私有创建；任何回滚必须先撤下 cron，再恢复旧 unit，前滚验收后才重新启用。
