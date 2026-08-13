#!/bin/bash
# 实验 016 三臂训练（zzai DCU）。单变量 = 训练池远场占比。
# 注意：不要用 `set -u`——/opt/dtk-26.04/env.sh 引用未定义变量，
# 在 set -u 下会让整个 shell 退出（本轮实测踩到，症状是任务静默不启动）。
set -e
cd /root/5.6+chanshui1
source env.sh
PY=./venv/bin/python

for ARM in A0_near F2_third F1_half; do
  OUT=runs/e016_${ARM}
  mkdir -p "$OUT"
  echo "=== ARM $ARM start $(date -u +%H:%M:%S) ==="
  $PY repo/scripts/finetune_phasenet.py \
      --data pool/mix_${ARM}.hdf5 \
      --out "$OUT" \
      --holdout pool/dev_mixed.hdf5 --holdout-max 600 \
      --epochs 8 --batch 32 --lr 3e-5 --sr 50 --win 3001 --seed 42 \
      --checkpoint-selection best --pretrained diting \
      --init-weights repo/weights/ustc_pickers/guangxi_sd.pt \
      > logs/e016_${ARM}.log 2>&1
  echo "=== ARM $ARM rc=$? $(date -u +%H:%M:%S) ==="
  tail -4 logs/e016_${ARM}.log
done
echo ALL_ARMS_DONE