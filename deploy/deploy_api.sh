#!/usr/bin/env bash
# =============================================================================
# 云主机一键部署：震相拾取 API（P0 保底上线，2~4 vCPU 的阿里云/华为云 CPU 机即可）
#
# 用法（服务器上）：
#   git clone https://github.com/ixijxjgxidj-cmd/dizheng-gpt5.6-sol.git dizheng && cd dizheng
#   sudo bash deploy/deploy_api.sh
#
# 可用环境变量覆盖：
#   PORT=8000                  # 服务端口
#   PRETRAINED=diting          # SeisBench 基座（A/B 实测 diting > stead，见 README）
#   WEIGHTS=<逗号分隔成员>     # 默认=2026-08-11 七成员生产顺序；可覆盖做降级/试验
#   ENSEMBLE_LONG_MEMBERS=5    # >300s 只用前5成员；七成员生产配置不可漏
#   CAP_SHORT_S=300 CAP_MAX_P=1 CAP_MAX_S=1
#   LONG_SNR_DB=-1.0 LONG_SNR_MIN_S=300
#   FORCE_PAIR_SHORT_S=300 FORCE_PAIR_MODE=conditional FORCE_PAIR_FLOOR=0.03
#   LONG_DEDUP_S=20
#   MAG_MODEL=seismicxm CLS_MODEL=seismicxm
#   ALLOW_MODEL_FALLBACK=0    # 缺 SeismicXM 时默认中止；仅应急显式设 1
#   P_THRESHOLD= / S_THRESHOLD=            # 拾取阈值；留空 = 用代码内全局默认
#   P_MERGE_WINDOW= / S_MERGE_WINDOW=      # 去重合并窗(秒)；留空 = 用代码内全局默认
#                              #（全局默认见 src/phasepicker/defaults.py，P3 网格产出后从这里接入）
#   PYTHON_BIN=python3         # 老镜像装了 python3.12 时指定，如 PYTHON_BIN=python3.12
#   THREADS=2                  # CPU 推理线程数（含 OMP_NUM_THREADS）。实测 2 线程最优，
#                              #  满核反而更慢（seisbench issue #68/#202 同结论），一般别改
#   DEVICE=cpu                 # cpu / cuda。GPU 机传 DEVICE=cuda：改装 CUDA 版 torch
#                              #  并以 --device cuda 启动（fp16 仍默认关，须同分验证后才开）
#   CAPTURE_DIR=<仓库>/captured # 评测请求采集目录（原始波形+响应落盘，复赛间微调的
#                              #  数据来源）。传 CAPTURE_DIR=off 关闭
#   SERVICE_USER=<用户名>       # systemd 运行用户；sudo 执行时默认取 SUDO_USER（不以 root 跑）
#   SERVICE_GROUP=<组名>        # 默认取 SERVICE_USER 的主组
#   SERVICE_HOME=<目录>         # 默认取 passwd 中的用户 home，并写入 systemd 的 HOME
#
# 依赖版本锁定在 deploy/requirements.lock（本地彩排验证过的精确版本）；
# 要求 python >= 3.12（Ubuntu 24.04 自带；老镜像先装 python3.12 再 PYTHON_BIN 指定）。
#
# 完成后：
#   curl http://127.0.0.1:${PORT}/health           → {"status":"ok"}
#   平台登记地址： http://<公网IP>:${PORT}/pick    （云安全组放行 TCP ${PORT}）
#   看日志： journalctl -u phasepick-api -f
#   换权重/换阈值： 带上面的环境变量重跑本脚本即可（幂等，只重写单元并重启），
#                   不要手改 /etc/systemd/system/phasepick-api.service
#   评测日 runbook（checklist/探活/故障处置/fallback）： deploy/EVAL_DAY.md
# =============================================================================
set -euo pipefail

PORT="${PORT:-8000}"
PRETRAINED="${PRETRAINED:-diting}"
PRODUCTION_WEIGHTS="guangxi,jiangxi,shandong,weights/aug/exam_aug6_r2train_sd.pt,weights/aug/crew_sp23_r2train_sd.pt,weights/geofon/geofon_m1_last_sd.pt,weights/geofon/geofon_m3_last_sd.pt"
WEIGHTS="${WEIGHTS:-$PRODUCTION_WEIGHTS}"
CAP_SHORT_S="${CAP_SHORT_S:-300}"
CAP_MAX_P="${CAP_MAX_P:-1}"
CAP_MAX_S="${CAP_MAX_S:-1}"
LONG_SNR_DB="${LONG_SNR_DB:--1.0}"
LONG_SNR_MIN_S="${LONG_SNR_MIN_S:-300}"
FORCE_PAIR_SHORT_S="${FORCE_PAIR_SHORT_S:-300}"
FORCE_PAIR_MODE="${FORCE_PAIR_MODE:-conditional}"
FORCE_PAIR_FLOOR="${FORCE_PAIR_FLOOR:-0.03}"
LONG_DEDUP_S="${LONG_DEDUP_S:-20}"
ENSEMBLE_LONG_MEMBERS="${ENSEMBLE_LONG_MEMBERS:-5}"
MAG_MODEL="${MAG_MODEL:-seismicxm}"
CLS_MODEL="${CLS_MODEL:-seismicxm}"
ALLOW_MODEL_FALLBACK="${ALLOW_MODEL_FALLBACK:-0}"
P_THRESHOLD="${P_THRESHOLD:-}"
S_THRESHOLD="${S_THRESHOLD:-}"
P_MERGE_WINDOW="${P_MERGE_WINDOW:-}"
S_MERGE_WINDOW="${S_MERGE_WINDOW:-}"
PYBIN="${PYTHON_BIN:-python3}"
# CPU 推理线程数默认 2：PhaseNet 这类小模型 CPU 满核多线程反而更慢
# （seisbench issue #68/#202；本仓实测 2 线程最优，28 分钟长文件 classify 0.05~0.07s，
#   速度非瓶颈）。--threads 管 torch 线程，OMP_NUM_THREADS 管底层 BLAS/OpenMP。
THREADS="${THREADS:-2}"
export OMP_NUM_THREADS="$THREADS"
DEVICE="${DEVICE:-cpu}"
CAPTURE_DIR="${CAPTURE_DIR:-}"
SERVICE_USER="${SERVICE_USER:-}"
SERVICE_GROUP="${SERVICE_GROUP:-}"
SERVICE_HOME="${SERVICE_HOME:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROBE_TOKEN_DIR="$REPO_ROOT/.runtime"
PROBE_TOKEN_FILE="$PROBE_TOKEN_DIR/watchdog_probe_token"
VENV="$REPO_ROOT/.venv"
CACHE="$REPO_ROOT/.seisbench_cache"
PY="$VENV/bin/python"
SUP_PID_FILE="$REPO_ROOT/serve_api.supervisor.pid"
API_PID_FILE="$REPO_ROOT/serve_api.pid"

echo "==================== [1/6] 系统依赖 ===================="
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -y >/dev/null 2>&1 || true
  apt-get install -y python3-venv python3-pip curl >/dev/null 2>&1 || true
elif command -v yum >/dev/null 2>&1; then
  yum install -y python3 python3-pip curl >/dev/null 2>&1 || true
fi
"$PYBIN" --version
# 锁定的 numpy 2.4.6 / scipy 1.18.0 只发 >=3.12 的轮子，老 python 会在装依赖时才莫名失败，这里提前拦住
if ! "$PYBIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'; then
  echo "!! $PYBIN 版本过旧。deploy/requirements.lock 锁定的依赖需要 python >= 3.12（本地彩排 3.13）"
  echo "   - 推荐 Ubuntu 24.04+ 镜像（自带 3.12）"
  echo "   - 或安装 python3.12 后： sudo PYTHON_BIN=python3.12 bash deploy/deploy_api.sh"
  exit 1
fi

echo "==================== [2/6] 虚拟环境与 Python 依赖（按 lock 锁定版本） ===================="
[ -d "$VENV" ] || "$PYBIN" -m venv "$VENV"
"$PY" -m pip install -q --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# torch 钉死 2.12.1（本地彩排版本）。CPU 机优先官方 CPU-only 轮子（~200MB），
# 失败回退清华 PyPI；GPU 机（DEVICE=cuda）直接装清华 PyPI 的完整轮子（带 CUDA，
# ~2GB+，国内快），装完自检 cuda 可用性——不可用只告警并继续（服务端 --device cuda
# 会自动退化 cpu，见 picker._resolve_device，绝不因此拒绝上线）。
if [ "$DEVICE" = "cuda" ]; then
  if ! "$PY" -c "import torch; assert torch.__version__.split('+')[0] == '2.12.1' and torch.cuda.is_available()" 2>/dev/null; then
    echo "安装 torch==2.12.1（CUDA 版，清华源）..."
    "$PY" -m pip install -q "torch==2.12.1" -i https://pypi.tuna.tsinghua.edu.cn/simple
  fi
  "$PY" - <<'PYEOF'
import torch
if torch.cuda.is_available():
    print("CUDA 可用:", torch.cuda.get_device_name(0))
else:
    print("!! DEVICE=cuda 但 torch.cuda 不可用（驱动缺失?）——服务会自动退化 CPU 运行")
    print("!! GPU 机需已装 NVIDIA 驱动: nvidia-smi 先能跑通")
PYEOF
elif ! "$PY" -c "import torch; assert torch.__version__.split('+')[0] == '2.12.1'" 2>/dev/null; then
  echo "安装 torch==2.12.1（先试官方 CPU 轮子，失败回退清华源）..."
  "$PY" -m pip install -q "torch==2.12.1" --index-url https://download.pytorch.org/whl/cpu \
    || "$PY" -m pip install -q "torch==2.12.1" -i https://pypi.tuna.tsinghua.edu.cn/simple
fi
"$PY" -m pip install -q -r "$SCRIPT_DIR/requirements.lock" -i https://pypi.tuna.tsinghua.edu.cn/simple
"$PY" -m pip check   # 依赖关系不自洽就在部署时大声失败，别带病进评测
"$PY" - <<'PYEOF'
import torch, seisbench, obspy, numpy, scipy, fastapi, starlette, uvicorn, python_multipart
print("torch", torch.__version__, "| seisbench", seisbench.__version__,
      "| numpy", numpy.__version__, "| obspy", obspy.__version__,
      "| 依赖齐全（版本按 deploy/requirements.lock 锁定）")
PYEOF

echo "==================== [3/6] 恢复预训练权重（免跨境下载） ===================="
mkdir -p "$CACHE"
for f in "$REPO_ROOT"/weights/phasenet_*_weights.tar.gz; do
  [ -f "$f" ] && tar xzf "$f" -C "$CACHE" && echo "已恢复: $(basename "$f")"
done
SEISBENCH_CACHE_ROOT="$CACHE" PYTHONUTF8=1 "$PY" - <<PYEOF
import seisbench.models as sbm
m = sbm.PhaseNet.from_pretrained("$PRETRAINED")
print("PhaseNet('$PRETRAINED') 加载成功: 采样率", m.sampling_rate, "Hz, 输入", m.in_samples, "点")
PYEOF

# T2/T3 共用的 SeismicXM encoder 约 208MB（超过 GitHub 单文件限制，靠 scp/发布资产）。
# T3 当前为 TTA+余弦 kNN：r2 两类 98.9%；08 包答案实际只出现标签 1–4，
# 183/205=89.27%。encoder 缺失时默认中止部署，不静默回退。
SXM="$REPO_ROOT/weights/seismicxm/seismicxm.middle.pt"
SXM_SHA256="671d02d677c25c3d075963889602299ec71f52c724470f2fa85bb28035fe1528"
if [ -f "$SXM" ]; then
  ACTUAL_SXM_SHA256="$(sha256sum "$SXM" | awk '{print $1}')"
  if [ "$ACTUAL_SXM_SHA256" != "$SXM_SHA256" ]; then
    echo "!! SeismicXM 编码器 SHA-256 不匹配，拒绝加载损坏/错误权重"
    echo "   期望: $SXM_SHA256"
    echo "   实际: $ACTUAL_SXM_SHA256"
    exit 1
  fi
  echo "SeismicXM 编码器就绪且校验通过（T2 MAE 0.621；T3 r2 98.9% / 08标签1–4为183/205）—— $(du -h "$SXM" | cut -f1)"
else
  echo "T2/T3: !! 未找到 $SXM。默认生产部署将在发布校验阶段中止。"
  echo "       要用当前 SeismicXM 版，从本地传 encoder 权重后重跑本脚本："
  echo "         scp -P <ssh端口> weights/seismicxm/seismicxm.middle.pt <用户>@<主机>:$REPO_ROOT/weights/seismicxm/"
  echo "       仅应急显式降级时设置 ALLOW_MODEL_FALLBACK=1，或直接选择 baseline 模型。"
fi

# 在写入或重启服务前统一核对生产参数、Git 跟踪资产和外置编码器。默认
# SeismicXM 配置必须有正确的 encoder；只有显式应急开关才允许缺失时继续。
# 即使允许 fallback，已存在但大小或哈希错误的 encoder 仍会被校验器拒绝。
VERIFY_ARGS=()
if { [ "$MAG_MODEL" = "seismicxm" ] || [ "$CLS_MODEL" = "seismicxm" ]; } \
   && [ "$ALLOW_MODEL_FALLBACK" != "1" ]; then
  VERIFY_ARGS+=(--require-external)
fi
if [ "$ALLOW_MODEL_FALLBACK" = "1" ]; then
  echo "!! ALLOW_MODEL_FALLBACK=1：允许缺失 SeismicXM encoder 时显式降级到 baseline"
fi
PYTHONUTF8=1 "$PY" "$REPO_ROOT/scripts/verify_release_manifest.py" \
  --manifest "$REPO_ROOT/deploy/production_release_manifest.json" \
  "${VERIFY_ARGS[@]}"

echo "==================== [4/6] 安装服务 ===================="
# watchdog skip-capture 使用运行时随机令牌；只把文件路径放进进程参数，令牌内容
# 不进仓库、unit、命令行或日志。已存在的令牌保持不变，避免每次幂等部署失配。
mkdir -p "$PROBE_TOKEN_DIR"
chmod 700 "$PROBE_TOKEN_DIR"
if [ -L "$PROBE_TOKEN_FILE" ]; then
  echo "!! watchdog probe token 不允许使用符号链接"
  exit 1
fi
if [ ! -e "$PROBE_TOKEN_FILE" ]; then
  (umask 077; "$PY" -c 'import secrets; print(secrets.token_urlsafe(48))' > "$PROBE_TOKEN_FILE")
fi
chmod 600 "$PROBE_TOKEN_FILE"
PYTHONPATH="$REPO_ROOT/src" "$PY" - "$PROBE_TOKEN_FILE" <<'PYEOF'
import sys
from phasepicker.probe_auth import load_probe_token_file

load_probe_token_file(sys.argv[1], require_private=True)
print("watchdog 回环探针令牌文件已就绪（内容不输出）")
PYEOF

EXTRA_ARGS=" --weights $WEIGHTS"
EXTRA_ARGS="$EXTRA_ARGS --cap-short-s $CAP_SHORT_S --cap-max-p $CAP_MAX_P --cap-max-s $CAP_MAX_S"
EXTRA_ARGS="$EXTRA_ARGS --long-snr-db $LONG_SNR_DB --long-snr-min-s $LONG_SNR_MIN_S"
EXTRA_ARGS="$EXTRA_ARGS --force-pair-short-s $FORCE_PAIR_SHORT_S --force-pair-mode $FORCE_PAIR_MODE --force-pair-floor $FORCE_PAIR_FLOOR"
EXTRA_ARGS="$EXTRA_ARGS --long-dedup-s $LONG_DEDUP_S --ensemble-long-members $ENSEMBLE_LONG_MEMBERS"
EXTRA_ARGS="$EXTRA_ARGS --mag-model $MAG_MODEL --cls-model $CLS_MODEL"
EXTRA_ARGS="$EXTRA_ARGS --probe-token-file $PROBE_TOKEN_FILE"
[ -n "$P_THRESHOLD" ] && EXTRA_ARGS="$EXTRA_ARGS --p-threshold $P_THRESHOLD"
[ -n "$S_THRESHOLD" ] && EXTRA_ARGS="$EXTRA_ARGS --s-threshold $S_THRESHOLD"
[ -n "$P_MERGE_WINDOW" ] && EXTRA_ARGS="$EXTRA_ARGS --p-merge-window $P_MERGE_WINDOW"
[ -n "$S_MERGE_WINDOW" ] && EXTRA_ARGS="$EXTRA_ARGS --s-merge-window $S_MERGE_WINDOW"
# 采集默认开（比赛数据 = 复赛微调材料）；CAPTURE_DIR=off 显式关闭
if [ "$CAPTURE_DIR" != "off" ]; then
  [ -n "$CAPTURE_DIR" ] || CAPTURE_DIR="$REPO_ROOT/captured"
  mkdir -p "$CAPTURE_DIR"
  EXTRA_ARGS="$EXTRA_ARGS --capture-dir $CAPTURE_DIR"
fi
START_CMD="$PY $REPO_ROOT/scripts/serve_api.py --pretrained $PRETRAINED --device $DEVICE --host 0.0.0.0 --port $PORT --threads $THREADS$EXTRA_ARGS"
echo "生产成员顺序: $WEIGHTS"
echo "长记录成员数: $ENSEMBLE_LONG_MEMBERS | cap: ${CAP_SHORT_S}s P${CAP_MAX_P}/S${CAP_MAX_S} | SNR: ${LONG_SNR_DB}dB | force-pair: $FORCE_PAIR_MODE | dedup: ${LONG_DEDUP_S}s"
echo "启动命令: $START_CMD"

# 无论走哪条分支，先按 PID 文件把上一轮 nohup 兜底进程清干净：
# 旧实例残留占端口会让 systemd 每 3s 拉起-绑定失败地空转到健康门超时，报错还看不出根因。
for pf in "$SUP_PID_FILE" "$API_PID_FILE"; do
  if [ -f "$pf" ]; then
    kill "$(cat "$pf")" 2>/dev/null || true
    rm -f "$pf"
  fi
done

if command -v systemctl >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
  # sudo 部署时，安装动作需要 root，但常驻推理服务不应继承 root 权限。优先使用
  # sudo 的原始调用者；若脚本由 root 直接运行，则尝试仓库所有者，最后才回退 root。
  if [ -z "$SERVICE_USER" ]; then
    SERVICE_USER="${SUDO_USER:-}"
  fi
  if [ -z "$SERVICE_USER" ] || [ "$SERVICE_USER" = "root" ]; then
    REPO_OWNER="$(stat -c '%U' "$REPO_ROOT" 2>/dev/null || true)"
    if [ -n "$REPO_OWNER" ] && [ "$REPO_OWNER" != "root" ]; then
      SERVICE_USER="$REPO_OWNER"
    else
      SERVICE_USER="root"
    fi
  fi
  if ! id "$SERVICE_USER" >/dev/null 2>&1; then
    echo "!! systemd 运行用户不存在: $SERVICE_USER（可用 SERVICE_USER=<用户名> 覆盖）"
    exit 1
  fi
  [ -n "$SERVICE_GROUP" ] || SERVICE_GROUP="$(id -gn "$SERVICE_USER")"
  if ! getent group "$SERVICE_GROUP" >/dev/null 2>&1; then
    echo "!! systemd 运行组不存在: $SERVICE_GROUP（可用 SERVICE_GROUP=<组名> 覆盖）"
    exit 1
  fi
  if [ -z "$SERVICE_HOME" ]; then
    SERVICE_HOME="$(getent passwd "$SERVICE_USER" | cut -d: -f6)"
  fi
  [ -n "$SERVICE_HOME" ] || SERVICE_HOME="$REPO_ROOT"
  if [ ! -d "$SERVICE_HOME" ]; then
    echo "!! systemd 运行用户 HOME 不存在: $SERVICE_HOME（可用 SERVICE_HOME=<目录> 覆盖）"
    exit 1
  fi

  # 服务需要写 SeisBench 缓存和请求采集目录；其余仓库内容保持只读即可。
  chown -R "$SERVICE_USER:$SERVICE_GROUP" "$CACHE"
  if [ "$CAPTURE_DIR" != "off" ]; then
    chown -R "$SERVICE_USER:$SERVICE_GROUP" "$CAPTURE_DIR"
  fi
  chown "$SERVICE_USER:$SERVICE_GROUP" "$PROBE_TOKEN_DIR"
  chown "$SERVICE_USER:$SERVICE_GROUP" "$PROBE_TOKEN_FILE"
  chmod 700 "$PROBE_TOKEN_DIR"
  chmod 600 "$PROBE_TOKEN_FILE"

  cat > /etc/systemd/system/phasepick-api.service <<UNIT
[Unit]
Description=PhasePick official API (zhenzhibei)
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_GROUP
WorkingDirectory=$REPO_ROOT
Environment=HOME=$SERVICE_HOME
Environment=SEISBENCH_CACHE_ROOT=$CACHE
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONUTF8=1
Environment=OMP_NUM_THREADS=$THREADS
ExecStart=$START_CMD
Restart=always
RestartSec=3
TimeoutStopSec=5
MemoryHigh=60%
MemoryMax=75%

[Install]
WantedBy=multi-user.target
UNIT
  # PYTHONUTF8=1: systemd 干净环境无 locale，最小镜像下 Python 编码回落 ASCII，
  #   读 diting 中文元数据 JSON 会崩（交互 shell 自检发现不了，只有服务里炸）。
  # OMP_NUM_THREADS: 与 --threads 一致压到 $THREADS——CPU 满核多线程反而慢（见文件头 THREADS 注释）。
  # MemoryHigh/Max + TimeoutStopSec: 内存异常时快速干净地重启，而不是整机 OOM 抖动。
  systemctl daemon-reload
  systemctl enable phasepick-api >/dev/null 2>&1
  systemctl restart phasepick-api
  echo "systemd 服务已启动（用户 $SERVICE_USER:$SERVICE_GROUP；开机自启 + 崩溃自拉起 + 内存护栏 60%/75%）"
else
  echo "无 systemd/非 root：用 nohup 守护循环兜底（评测日强烈建议用 root+systemd 路径）"
  SUP_SH="$REPO_ROOT/serve_api_supervisor.sh"
  cat > "$SUP_SH" <<SUPEOF
#!/usr/bin/env bash
# 由 deploy_api.sh 生成：nohup 兜底守护循环。崩溃 3s 自动重启；
# 停止只按 PID 文件精确 kill（先杀 supervisor 再杀 api），绝不 pkill -f。
echo \$\$ > "$SUP_PID_FILE"
while true; do
  SEISBENCH_CACHE_ROOT="$CACHE" PYTHONUTF8=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$THREADS \\
    $START_CMD >> "$REPO_ROOT/serve_api.log" 2>&1 &
  echo \$! > "$API_PID_FILE"
  wait \$! || true
  echo "[supervisor] \$(date '+%F %T') serve_api 退出，3s 后自动重启" >> "$REPO_ROOT/serve_api.log"
  sleep 3
done
SUPEOF
  chmod +x "$SUP_SH"
  nohup "$SUP_SH" >/dev/null 2>&1 &
  echo "守护循环已启动（supervisor PID 见 $SUP_PID_FILE）"
  echo "日志: tail -f $REPO_ROOT/serve_api.log"
  echo "停止: kill \$(cat $SUP_PID_FILE) && kill \$(cat $API_PID_FILE)"
fi

echo "==================== [5/6] 等待服务就绪（最长 180s，慢云机首启加载权重较久） ===================="
for i in $(seq 1 180); do
  if curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "健康检查通过（第 ${i} 秒）"; break
  fi
  if [ "$i" = "180" ]; then
    echo "!! 服务 180s 未就绪。排查："
    echo "   systemd 路径: journalctl -u phasepick-api -n 50"
    echo "   nohup 路径:   tail -50 $REPO_ROOT/serve_api.log"
    echo "   端口被旧进程占用: ss -ltnp | grep :$PORT"
    exit 1
  fi
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
PYTHONUTF8=1 "$PY" "$REPO_ROOT/scripts/check_api.py" \
  --url "http://127.0.0.1:$PORT/pick" --input "$SMOKE_DIR" \
  --probe-token-file "$PROBE_TOKEN_FILE"
rm -rf "$SMOKE_DIR"

PUB_IP="$(curl -sf -m 3 https://myip.ipip.net 2>/dev/null || curl -sf -m 3 ifconfig.me 2>/dev/null || echo '<公网IP>')"
echo ""
echo "======================================================================"
echo " 部署完成。下一步："
echo "   1. 云控制台安全组放行 TCP $PORT"
echo "   2. 外网复测: python scripts/check_api.py --url http://<公网IP>:$PORT/pick --input <mseed>"
echo "   3. 平台登记: http://<公网IP>:$PORT/pick"
echo "   4. 装评测日探活 cron 并过一遍 checklist: 见 deploy/EVAL_DAY.md"
echo "   本机出口:   $PUB_IP"
echo "======================================================================"
