#!/bin/bash
# 实验 016 续跑：更高远场占比（F3=100%、F4=80.1%）。
# 注意：不要用 `set -u`——/opt/dtk-26.04/env.sh 引用未定义变量，set -u 下会杀掉整个 shell。
set -e
cd /root/5.6+chanshui1
source env.sh
PY=./venv/bin/python

for ARM in F4_far80 F3_far75; do
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
  tail -3 logs/e016_${ARM}.log
done
echo E016B_ARMS_DONE