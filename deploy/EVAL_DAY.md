# 评测日 Runbook（U13）

开放窗口约 1 天，任何小时级的服务半死都是该时段样本全丢。下文用 `$REPO` 表示
目标服务器上的仓库根目录；先按实际部署设置 `export REPO=<仓库路径>`。服务为
systemd 单元 `phasepick-api`，端口 8000。

速查：

```bash
journalctl -u phasepick-api -f                        # 看日志
systemctl restart phasepick-api                       # 重启
systemctl show phasepick-api -p MemoryCurrent         # 当前内存
sudo bash $REPO/deploy/deploy_api.sh                  # 一键重部（幂等）
bash $REPO/deploy/watchdog.sh                         # 手动跑一次真实请求探活
```

> **2026-08-11 当前实际部署：Debian 13 云主机 + systemd，服务以非 root 普通用户
> 运行；服务器本机的 T1/T2/T3 与 300 请求 soak 已全绿。
> 公网地址按实际机器配置为 `http://<公网地址>:8000`，云安全组 TCP 8000 已放行，且已从非服务器
> 机器完成 `/health`、`/pick`、`/magnitude`、`/classify` 四端点 HTTP 200 复测。
> 当前剩余的是比赛平台 URL 登记；watchdog cron 仍待按 §2 的采集污染说明处理后安装。**

---

## 0. 远程操作一律先开 tmux（平台原话：避免网络波动或 SSH 关闭导致中断）

**先分清什么会断：**

- **API 服务本身不会断。** `deploy_api.sh` 用 `nohup` 起 supervisor 循环，已脱离控制终端、
  忽略 SIGHUP，SSH 关掉照跑（2026-08-01 实测：2315 次真题请求打的就是无 SSH 连接的进程，全绿）。
- **会断的是你手敲的前台长命令**：`deploy_api.sh` 本身（首次/重租后装依赖几分钟）、
  soak 测试（~5 分钟）、微调训练（小时级）、§5 变体演练。SSH 一抖就 SIGHUP 死。
  其中 **deploy 断在半路最难受**：依赖装一半、端口占着、权重下一半，重跑前还得先清残留。
- **tmux 救不了容器本身被停/释放**——那会连 tmux 会话带守护循环一起没，
  只能重跑部署 + 到平台改登记地址（见 §3 第 4 级）。

```bash
tmux new -s seis        # 首次开会话
tmux a -t seis          # 断线后重连（回到原样，命令还在跑）
tmux ls                 # 看有哪些会话
# 会话内：Ctrl-b d 脱离（命令继续跑）， Ctrl-b [ 进翻页模式看长输出，q 退出
```

断线后**先 `tmux a -t seis` 看上一条跑完没有，再决定要不要重跑**——尤其是 deploy，
盲目重跑会撞上还占着 8000 端口的旧实例。

---

## 1. T-1 检查单（评测前一天，全部打勾）

- [ ] **外网复测**：从**非服务器**机器（家里/公司电脑）用去年真题打公网地址，全绿：
  `PYTHONUTF8=1 python scripts/check_api.py --url http://<公网IP>:8000/pick --input "<第1轮真题zip>" --limit 20`
- [ ] **安全组**：云控制台确认 TCP 8000 对 0.0.0.0/0 放行（截图留档）。
- [ ] **磁盘余量**：`df -h` 根分区 > 5GB；不够先 `journalctl --vacuum-size=200M`。
- [ ] **权重指纹**：`md5sum $REPO/weights/*.tar.gz` 与本地记录一致；
  `systemctl cat phasepick-api | grep ExecStart` 抄下全行（基座/权重/阈值为证，出问题时对照）。
- [ ] **发布清单**：`PYTHONUTF8=1 $REPO/.venv/bin/python $REPO/scripts/verify_release_manifest.py --require-external`
  必须无 error/warning；默认禁止缺少 SeismicXM encoder 时静默降级。
- [ ] **基线延迟记录**：把上面外网复测输出的 均值/中位/最大 延迟与 P/S 总数抄进值班笔记，
  评测中延迟翻倍/拾取数异常时对照。
- [ ] **内存 soak 复测**（泄漏补丁的 Linux 侧验证，本地探针数字是 Windows 的）——
  **~5 分钟，务必在 tmux 里跑（见 §0）**：
  ```bash
  systemctl show phasepick-api -p MemoryCurrent                    # 前
  PYTHONUTF8=1 $REPO/.venv/bin/python $REPO/scripts/check_api.py \
    --url http://127.0.0.1:8000/pick --input "<第1轮真题zip>" --limit 300
  systemctl show phasepick-api -p MemoryCurrent                    # 后
  ```
  300 条后增长 < ~200MB 视为通过；若接近 GB 级说明泄漏补丁在该环境失效，
  靠 MemoryHigh/Max 护栏也能撑，但要把 watchdog cron 间隔缩到 2 分钟并升配内存。
- [ ] **变体演练**：多台站与 50Hz 样本各打一遍 HTTP，响应格式正常（命令见 §5）。
- [ ] **探活 cron 已装**且最近一次输出 OK（见 §2）。
- [ ] **平台登记地址**再次核对是 `http://<公网IP>:8000/pick`（不是 /health、不是内网 IP）。

---

## 2. 探活与自动恢复（cron，每 5 分钟）

`/health` 是静态返回、不经过推理引擎；推理互斥锁被一次挂死请求占住时进程不退出、
systemd 不拉起、/health 依旧绿。**判活必须用真实波形打 /pick**，这正是
`deploy/watchdog.sh` 做的事（失败自动 `systemctl restart` + webhook 告警）。

> **采集隔离：**部署脚本会在 `.runtime/watchdog_probe_token` 生成 0600 随机令牌。
> watchdog 只向数值型 loopback URL 发送该令牌；API 只有在“直接 TCP 对端是
> `127.0.0.0/8` 或 `::1`”且令牌恒定时间匹配时才跳过采集，并回
> `X-PhasePicker-Probe: accepted`。watchdog 收不到该确认会 fail closed，不会静默制造
> 探针样本。`X-Forwarded-For` / `X-Real-IP` 只用于 manifest 元数据，永不参与授权；
> 公网伪造同名 header 仍会正常进入 `captured/`。

```bash
# 1) 放一条去年真题 mseed 当探测样例（不放也行，watchdog 会自动生成合成波形兜底）
mkdir -p $REPO/probe_sample && scp <本地某条真题.mseed> <SSH用户>@<公网地址>:$REPO/probe_sample/

# 2) 手动跑一次确认 OK
bash $REPO/deploy/watchdog.sh

# 可选但推荐：运行前后比较捕获文件数，应完全不变
find "$REPO/captured" -type f 2>/dev/null | wc -l

# 3) 装 cron（WEBHOOK_URL 填钉钉/企微机器人地址，留空=只写日志不告警）
crontab -e
# 先在 crontab 顶部设置：REPO=<仓库路径>
# */5 * * * * WEBHOOK_URL='' bash "$REPO/deploy/watchdog.sh" >> "$REPO/watchdog.log" 2>&1
```

评测中每小时瞄一眼 `tail $REPO/watchdog.log`：连续 FAIL→restart 循环 = 转 §4 处置。

---

## 3. Fallback 阶梯（从上往下试，每步 3~10 分钟，做完必须复跑 §2 手动探活 + 外网复测）

| 级 | 症状 | 动作 |
|---|---|---|
| 1 | 服务半死/挂死/偶发失败 | `systemctl restart phasepick-api`（watchdog 会自动做） |
| 2 | diting 权重异常（加载失败/拾取明显异常） | 切 stead 基座：`sudo PRETRAINED=stead bash $REPO/deploy/deploy_api.sh`（去年 R1 前200条 1.496，比 diting 1.899 低但可用） |
| 3 | 基座全不可用而本地微调件在 | `sudo PRETRAINED=stead WEIGHTS=$REPO/weights/phasenet_iquique_full_best.pt bash $REPO/deploy/deploy_api.sh`（stead+Iquique 全量微调，R1 前200条 1.678；注意微调基座是 stead，勿配 diting） |
| 4 | 整机挂/网络不可达 | 备用机重部（预算 ~15 分钟：`git clone` → `sudo bash deploy/deploy_api.sh` → 安全组放行）→ 平台**改登记地址**并通知组委确认生效 |

P2/P3 产出后的正常切换（非故障）：带环境变量重跑部署脚本，例如
`sudo WEIGHTS=$REPO/weights/best.pt P_THRESHOLD=0.2 S_THRESHOLD=0.15 bash $REPO/deploy/deploy_api.sh`
（阈值留空 = 用 `src/phasepicker/defaults.py` 全局默认，当前 P=0.2 / S=0.15）。

---

## 4. 常见故障 → 处置表

| 症状 | 判定 | 处置 |
|---|---|---|
| 健康门 180s 超时 / curl /health 不通 | `journalctl -u phasepick-api -n 50` 看崩溃栈 | 权重加载崩→阶梯2；依赖崩→重跑 deploy 脚本看 [2/6] 是否红 |
| journalctl 反复 "bind: address already in use" | 旧 nohup 实例占端口：`ss -ltnp \| grep :8000` | `kill $(cat $REPO/serve_api.supervisor.pid) $(cat $REPO/serve_api.pid)`；没有 PID 文件就按 ss 给的 pid 精确 kill（**不要 pkill -f**） |
| /health 绿但 /pick 全超时 | 推理锁挂死（watchdog 会探到） | `systemctl restart phasepick-api` |
| 服务被频繁 OOM 杀（journalctl 见 oom-kill / MemoryMax） | `watchdog.log` restart 频率 >2次/小时 | 泄漏护栏在兜底但太频繁：升配内存重部，或把 watchdog 改 2 分钟一跑硬扛 |
| UnicodeDecodeError / GBK/ASCII 编码栈 | systemd 干净环境编码问题 | 本包单元已带 `PYTHONUTF8=1`，若见此错说明跑的是旧单元：重跑 deploy 脚本覆盖 |
| 磁盘满 | `df -h` | `journalctl --vacuum-size=200M`；`rm $REPO/serve_api.log*`（nohup 路径日志是追加的，会长大） |
| 官方发了多台站/非100Hz 文件，响应看着异常 | 代码本身支持这两条路径（已演练§5） | 保存该样本文件与响应 JSON 留档复盘；只要 200+JSON 格式对就不动服务 |
| 平台侧显示 4xx/不可用 | 外网 `check_api` 复测 | 外网绿→查平台登记地址（/pick、公网 IP、端口）；外网红→安全组/防火墙 `firewall-cmd --list-ports` |

---

## 5. 变体演练命令（T-1 做一遍）

去年 1915 条真题全部是单台站 100Hz，但代码支持多台站与非 100Hz——评测日前先从
HTTP 层各演练一次，别让第一次真实流量当小白鼠。在服务器上：

```bash
PY=$REPO/.venv/bin/python

# 50Hz 变体：把一条真题重采样到 50Hz 再打
PYTHONUTF8=1 $PY - <<'EOF'
import os
import obspy
st = obspy.read(os.path.join(os.environ["REPO"], "probe_sample", "*.mseed"))
st.resample(50.0)
st.write("/tmp/variant_50hz.mseed", format="MSEED")
EOF
PYTHONUTF8=1 $PY $REPO/scripts/check_api.py --url http://127.0.0.1:8000/pick --input /tmp/variant_50hz.mseed

# 多台站单文件：两个台站六道合成
PYTHONUTF8=1 $PY - <<'EOF'
import numpy as np, obspy
rng = np.random.default_rng(11)
st = obspy.Stream()
for sta in ("AAA", "BBB"):
    for comp in "ENZ":
        tr = obspy.Trace(rng.standard_normal(9000).astype("float32"))
        tr.stats.sampling_rate = 100.0
        tr.stats.station, tr.stats.network, tr.stats.channel = sta, "XB", "BH"+comp
        st.append(tr)
st.write("/tmp/variant_multista.mseed", format="MSEED")
EOF
PYTHONUTF8=1 $PY $REPO/scripts/check_api.py --url http://127.0.0.1:8000/pick --input /tmp/variant_multista.mseed
```

两条都应返回 200 + 合法 JSON（多台站响应应含两个台站键）。任何异常记录样本与
响应，评测日照 §4 最后一行处置。
