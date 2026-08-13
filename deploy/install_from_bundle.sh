#!/usr/bin/env bash
# =============================================================================
# 从中转站分发的离线依赖包部署 API（GPU 机执行，零公网下载）
#
# 前提：/data/bundle/ 下已有 wheels.tar（54 个锁定版本依赖 + torch cu126）
#       与 seismicxm.middle.pt（默认必需；T2 MAE 0.621；T3 r2 两类 98.9%，
#       08 包标签实际为 1–4，183/205=89.27%）
#
# 用法（GPU 机）：
#   bash /data/bundle/install_from_bundle.sh
#
# 做什么：解包 wheels → 建 venv → 全离线装（--no-index）→ 放权重 → 起服务 → 自检
# =============================================================================
set -euo pipefail

BUNDLE="${BUNDLE:-/data/bundle}"
REPO="${REPO:-/data/dizheng}"
PORT="${PORT:-8000}"

echo "==================== [1/6] 检查离线包 ===================="
[ -f "$BUNDLE/wheels.tar" ] || { echo "!! 缺 $BUNDLE/wheels.tar"; exit 1; }
if [ ! -d "$BUNDLE/wheels" ]; then
  tar xf "$BUNDLE/wheels.tar" -C "$BUNDLE"
fi
N=$(ls "$BUNDLE"/wheels/*.whl 2>/dev/null | wc -l)
echo "wheels: $N 个"
ls "$BUNDLE"/wheels/torch-*cu126*.whl >/dev/null 2>&1 \
  && echo "torch: cu126 版本已就位（匹配 CUDA 12.x 驱动）" \
  || echo "!! 未见 cu126 torch，GPU 可能不可用"

echo "==================== [2/6] 仓库代码 ===================="
if [ -d "$REPO/.git" ]; then
  cd "$REPO" && git pull --ff-only 2>&1 | tail -1
else
  git clone https://github.com/ixijxjgxidj-cmd/chanshui.git "$REPO" 2>&1 | tail -1
fi
cd "$REPO"

echo "==================== [3/6] venv + 离线安装 ===================="
PY_BIN="${PYTHON_BIN:-python3}"
[ -d "$REPO/.venv" ] || "$PY_BIN" -m venv "$REPO/.venv"
PY="$REPO/.venv/bin/python"
# --no-index：完全不碰公网，只用本地 wheels
"$PY" -m pip install -q --no-index --find-links "$BUNDLE/wheels" \
  --upgrade pip setuptools 2>/dev/null || true
"$PY" -m pip install --no-index --find-links "$BUNDLE/wheels" \
  -r deploy/requirements.lock 2>&1 | tail -3
# torch 单独钉 cu126 变体（requirements.lock 里写的是 2.12.1，需覆盖成 +cu126）
"$PY" -m pip install --no-index --find-links "$BUNDLE/wheels" \
  --force-reinstall --no-deps "$BUNDLE"/wheels/torch-*cu126*.whl 2>&1 | tail -2
"$PY" - <<'PYEOF'
import torch, seisbench, obspy, fastapi, sklearn
print("torch", torch.__version__, "| cuda 可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("seisbench", seisbench.__version__, "| obspy", obspy.__version__, "| 依赖齐全")
PYEOF

echo "==================== [4/6] 权重 ===================="
mkdir -p "$REPO/weights/seismicxm" "$REPO/.seisbench_cache"
if [ -f "$BUNDLE/seismicxm.middle.pt" ] && [ ! -f "$REPO/weights/seismicxm/seismicxm.middle.pt" ]; then
  cp "$BUNDLE/seismicxm.middle.pt" "$REPO/weights/seismicxm/"
  echo "SeismicXM 编码器已放置（部署脚本将核对 SHA-256）"
fi
for f in "$REPO"/weights/phasenet_*_weights.tar.gz; do
  [ -f "$f" ] && tar xzf "$f" -C "$REPO/.seisbench_cache" && echo "恢复: $(basename "$f")"
done

echo "==================== [5/6] 起服务 ===================="
DEV=cuda
"$PY" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || DEV=cpu
echo "推理设备: $DEV"
DEVICE="$DEV" PORT="$PORT" bash "$REPO/deploy/deploy_api.sh"

echo "==================== [6/6] 完成 ===================="
echo "健康: curl http://127.0.0.1:$PORT/health"
echo "日志: tail -f $REPO/serve_api.log"
