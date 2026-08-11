# 发布可靠性阶段 007：watchdog 回环探针捕获隔离

- 日期：2026-08-11
- 状态：**实现与本地边界验证完成；待提交及服务器双向验收**
- 基准分支：`main`
- 基准提交：`70a3ecf`
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

仍未完成：GitHub 提交，以及服务器上“本机认证探针不新增捕获 / 公网伪造 header 仍新增捕获”的双向验收和 cron 安装。
