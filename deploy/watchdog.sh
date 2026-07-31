#!/usr/bin/env bash
# =============================================================================
# 评测日探活（deploy/EVAL_DAY.md §2）：用真实波形打 /pick 判活，失败自动恢复+告警。
#
# 为什么不用 /health：/health 是静态返回不经过推理引擎；推理全局互斥锁被一次
# 挂死的请求占住时，进程不退出（systemd 不会拉起）、/health 依旧绿，只有真实
# /pick 请求能暴露"半死"状态。
#
# cron 安装（root）：
#   */5 * * * * WEBHOOK_URL='' bash /root/dizheng/deploy/watchdog.sh >> /root/dizheng/watchdog.log 2>&1
#
# 可配环境变量：
#   URL=http://127.0.0.1:8000/pick   # 探测地址
#   SAMPLE_DIR=<目录>                # 探测用 mseed；默认 <repo>/probe_sample，
#                                    # 建议 scp 一条去年真题 mseed 进去；目录为空则自动生成合成波形兜底
#   SERVICE=phasepick-api            # systemd 服务名
#   WEBHOOK_URL=                     # 钉钉/企微机器人 webhook，留空只写日志
#   TIMEOUT=90                       # 单次探测超时秒数
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
URL="${URL:-http://127.0.0.1:8000/pick}"
SERVICE="${SERVICE:-phasepick-api}"
SAMPLE_DIR="${SAMPLE_DIR:-$REPO_ROOT/probe_sample}"
TIMEOUT="${TIMEOUT:-90}"
PY="$REPO_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY=python3

mkdir -p "$SAMPLE_DIR"
if ! ls "$SAMPLE_DIR"/*.mseed >/dev/null 2>&1; then
  echo "$(date '+%F %T') 探测样例缺失，生成合成波形兜底（建议换成一条去年真题 mseed）"
  PYTHONUTF8=1 "$PY" - <<PYEOF
import numpy as np, obspy
rng = np.random.default_rng(7)
st = obspy.Stream()
for comp in "ENZ":
    tr = obspy.Trace(rng.standard_normal(9000).astype("float32"))
    tr.stats.sampling_rate = 100.0
    tr.stats.station, tr.stats.network, tr.stats.channel = "TST", "XB", "BH"+comp
    st.append(tr)
st.write("$SAMPLE_DIR/probe.mseed", format="MSEED")
PYEOF
fi

if PYTHONUTF8=1 "$PY" "$REPO_ROOT/scripts/check_api.py" \
    --url "$URL" --input "$SAMPLE_DIR" --limit 1 --timeout "$TIMEOUT" \
    > /tmp/phasepick_watchdog_last.log 2>&1; then
  echo "$(date '+%F %T') OK"
  exit 0
fi

echo "$(date '+%F %T') FAIL —— /pick 真实请求探测失败，开始自动恢复。最后输出："
tail -5 /tmp/phasepick_watchdog_last.log || true

if command -v systemctl >/dev/null 2>&1 && systemctl restart "$SERVICE" 2>/dev/null; then
  echo "$(date '+%F %T') 已 systemctl restart $SERVICE"
elif [ -f "$REPO_ROOT/serve_api.pid" ]; then
  # nohup 兜底部署：-9 杀掉可能挂死的 api 进程，守护循环 3s 内自动拉起
  kill -9 "$(cat "$REPO_ROOT/serve_api.pid")" 2>/dev/null || true
  echo "$(date '+%F %T') 已按 PID 文件 kill -9，等待守护循环重启"
else
  echo "$(date '+%F %T') !! 无 systemd 服务也无 PID 文件，需要人工介入"
fi

if [ -n "${WEBHOOK_URL:-}" ]; then
  # 钉钉/企微自定义机器人通用 text 格式；换飞书等自行改 payload
  curl -sf -m 5 -H 'Content-Type: application/json' \
    -d "{\"msgtype\":\"text\",\"text\":{\"content\":\"[phasepick-api] $(date '+%F %T') 探活失败，已触发自动重启，请上机确认\"}}" \
    "$WEBHOOK_URL" >/dev/null 2>&1 || echo "$(date '+%F %T') webhook 推送失败"
fi
exit 1
