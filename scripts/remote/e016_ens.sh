#!/bin/bash
# 集成级对照：c3 基线 vs 用 e016 远场臂替换公开成员。
# 单变量 = 第 4/5 成员（公开 ETHZ+CREW 权重）是否换成远场微调权重。
# 注意：不要用 set -u（DTK env.sh 会杀 shell）。
set -e
cd /root/5.6+chanshui1
source env.sh
PY=./venv/bin/python
POOL=$1
TAG=$2
LIMIT=${3:-0}

PUB_A=repo/weights/pub/pub_ethzcrew_a_sd.pt
PUB_B=repo/weights/pub/pub_ethzcrew_b_sd.pt
G1=repo/weights/geofon/geofon_m1_last_sd.pt
G3=repo/weights/geofon/geofon_m3_last_sd.pt
F1=runs/e016_F1_half/best.pt
F4=runs/e016_F4_far80/best.pt

run () {
  local NAME=$1; local MEM=$2
  local OUT=runs/ens_${NAME}_${TAG}.json
  if [ -f "$OUT" ]; then echo "HAVE $OUT"; return; fi
  $PY e016_ens_eval.py --members "$MEM" --pool "$POOL" --out "$OUT" \
      --label "${NAME}_${TAG}" --limit "$LIMIT" > logs/ens_${NAME}_${TAG}.log 2>&1
  echo "$NAME rc=$? -> $OUT"
}

# c3 基线（全合规七成员）
run c3      "guangxi,huanan,jiangxi,$PUB_A,$PUB_B,$G1,$G3"
# 候选 D1：把两个公开 ETHZ+CREW 成员换成远场臂 F1 / F4（保持 7 成员不变）
run d1_f1f4 "guangxi,huanan,jiangxi,$F1,$F4,$G1,$G3"
# 候选 D2：只换一个（更保守，检验边际贡献）
run d2_f1   "guangxi,huanan,jiangxi,$F1,$PUB_B,$G1,$G3"
echo ENS_DONE_$TAG