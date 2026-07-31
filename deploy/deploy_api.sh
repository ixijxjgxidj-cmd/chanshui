#!/usr/bin/env bash
# =============================================================================
# 云主机一键部署：震相拾取 API（P0 保底上线，2~4 vCPU 的阿里云/华为云 CPU 机即可）
#
# 用法（服务器上）：
#   git clone https://gitee.com/hulk-cheng/dizheng.git && cd dizheng
#   sudo bash deploy/deploy_api.sh
#
# 可用环境变量覆盖：
#   PORT=8000                  # 服务端口
#   PRETRAINED=diting          # SeisBench 基座（A/B 实测 diting > stead，见 README）
#   WEIGHTS=/path/xx.pt        # 换微调权重时指定（P2 产出 best.pt 后热切用）
#
# 完成后：
#   curl http://127.0.0.1:${PORT}/health           → {"status":"ok"}
#   平台登记地址： http://<公网IP>:${PORT}/pick    （云安全组放行 TCP ${PORT}）
#   看日志： journalctl -u phasepick-api -f
#   换权重： 改 /etc/systemd/system/phasepick-api.service 后
#            systemctl daemon-reload && systemctl restart phasepick-api
# =============================================================================
set -euo pipefail

PORT="${PORT:-8000}"
PRETRAINED="${PRETRAINED:-diting}"
WEIGHTS="${WEIGHTS:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$REPO_ROOT/.venv"
CACHE="$REPO_ROOT/.seisbench_cache"
PY="$VENV/bin/python"

echo "==================== [1/6] 系统依赖 ===================="
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -y >/dev/null 2>&1 || true
  apt-get install -y python3-venv python3-pip curl >/dev/null 2>&1 || true
elif command -v yum >/dev/null 2>&1; then
  yum install -y python3 python3-pip curl >/dev/null 2>&1 || true
fi
python3 --version

echo "==================== [2/6] 虚拟环境与 Python 依赖 ===================="
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$PY" -m pip install -q --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# torch 优先装 CPU-only 轮子（~200MB）；官方 cpu 源不通时回退清华 PyPI
# （清华的是带 CUDA 的大包 ~2GB+，慢但国内稳）。
if ! "$PY" -c "import torch" 2>/dev/null; then
  echo "安装 torch（先试官方 CPU 轮子，失败回退清华源）..."
  "$PY" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu \
    || "$PY" -m pip install -q torch -i https://pypi.tuna.tsinghua.edu.cn/simple
fi
"$PY" -m pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple \
  "seisbench>=0.11" obspy fastapi uvicorn python-multipart requests
"$PY" - <<'PYEOF'
import torch, seisbench, obspy, fastapi, uvicorn, python_multipart
print("torch", torch.__version__, "| seisbench", seisbench.__version__, "| 依赖齐全")
PYEOF

echo "==================== [3/6] 恢复预训练权重（免跨境下载） ===================="
mkdir -p "$CACHE"
for f in "$REPO_ROOT"/weights/phasenet_*_weights.tar.gz; do
  [ -f "$f" ] && tar xzf "$f" -C "$CACHE" && echo "已恢复: $(basename "$f")"
done
SEISBENCH_CACHE_ROOT="$CACHE" "$PY" - <<PYEOF
import seisbench.models as sbm
m = sbm.PhaseNet.from_pretrained("$PRETRAINED")
print("PhaseNet('$PRETRAINED') 加载成功: 采样率", m.sampling_rate, "Hz, 输入", m.in_samples, "点")
PYEOF

echo "==================== [4/6] 安装 systemd 服务 ===================="
WEIGHTS_ARG=""
[ -n "$WEIGHTS" ] && WEIGHTS_ARG=" --weights $WEIGHTS"
START_CMD="$PY $REPO_ROOT/scripts/serve_api.py --pretrained $PRETRAINED --device cpu --host 0.0.0.0 --port $PORT$WEIGHTS_ARG"

if command -v systemctl >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
  cat > /etc/systemd/system/phasepick-api.service <<UNIT
[Unit]
Description=PhasePick official API (zhenzhibei)
After=network.target

[Service]
Type=simple
WorkingDirectory=$REPO_ROOT
Environment=SEISBENCH_CACHE_ROOT=$CACHE
Environment=PYTHONUNBUFFERED=1
ExecStart=$START_CMD
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT
  systemctl daemon-reload
  systemctl enable phasepick-api >/dev/null 2>&1
  systemctl restart phasepick-api
  echo "systemd 服务已启动（开机自启 + 崩溃自拉起）"
else
  echo "无 systemd/非 root：用 nohup 兜底启动"
  pkill -f "serve_api.py" 2>/dev/null || true
  SEISBENCH_CACHE_ROOT="$CACHE" nohup $START_CMD > "$REPO_ROOT/serve_api.log" 2>&1 &
  echo "日志: tail -f $REPO_ROOT/serve_api.log"
fi

echo "==================== [5/6] 等待服务就绪 ===================="
for i in $(seq 1 60); do
  if curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "健康检查通过（第 ${i} 秒）"; break
  fi
  [ "$i" = "60" ] && { echo "!! 服务 60s 未就绪，查日志: journalctl -u phasepick-api -n 50"; exit 1; }
  sleep 1
done

echo "==================== [6/6] 官方格式自检（合成波形） ===================="
SMOKE_DIR="$(mktemp -d)"
"$PY" - <<PYEOF
import numpy as np, obspy
rng = np.random.default_rng(7)
st = obspy.Stream()
for comp in "ENZ":
    tr = obspy.Trace(rng.standard_normal(9000).astype("float32"))
    tr.stats.sampling_rate = 100.0
    tr.stats.station, tr.stats.network, tr.stats.channel = "TST", "XB", "BH"+comp
    st.append(tr)
st.write("$SMOKE_DIR/smoke.mseed", format="MSEED")
PYEOF
"$PY" "$REPO_ROOT/scripts/check_api.py" \
  --url "http://127.0.0.1:$PORT/pick" --input "$SMOKE_DIR"
rm -rf "$SMOKE_DIR"

PUB_IP="$(curl -sf -m 3 https://myip.ipip.net 2>/dev/null || curl -sf -m 3 ifconfig.me 2>/dev/null || echo '<公网IP>')"
echo ""
echo "======================================================================"
echo " 部署完成。下一步："
echo "   1. 云控制台安全组放行 TCP $PORT"
echo "   2. 外网复测: python scripts/check_api.py --url http://<公网IP>:$PORT/pick --input <mseed>"
echo "   3. 平台登记: http://<公网IP>:$PORT/pick"
echo "   本机出口:   $PUB_IP"
echo "======================================================================"
