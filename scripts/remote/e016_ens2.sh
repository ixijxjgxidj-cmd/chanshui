#!/bin/bash
# 集成级第二组：加法 vs 替换。
# 上一组已证伪"替换公开成员"（c3->d1 显著退化 -0.0166）。这里检验 F1 是否作为
# 【第 8 个成员】提供互补信息 —— 若集成的价值在多样性而非单成员强度，加法应优于替换。
set -e
cd /root/5.6+chanshui1
source env.sh
PY=./venv/bin/python
POOL=$1
TAG=$2

PUB_A=repo/weights/pub/pub_ethzcrew_a_sd.pt
PUB_B=repo/weights/pub/pub_ethzcrew_b_sd.pt
G1=repo/weights/geofon/geofon_m1_last_sd.pt
G3=repo/weights/geofon/geofon_m3_last_sd.pt
F1=runs/e016_F1_half/best.pt
F4=runs/e016_F4_far80/best.pt
F3=runs/e016_F3_far75/best.pt
A0=runs/e016_A0_near/best.pt

run () {
  local NAME=$1; local MEM=$2
  local OUT=runs/ens_${NAME}_${TAG}.json
  if [ -f "$OUT" ]; then echo "HAVE $OUT"; return; fi
  $PY e016_ens_eval.py --members "$MEM" --pool "$POOL" --out "$OUT" \
      --label "${NAME}_${TAG}" > logs/ens_${NAME}_${TAG}.log 2>&1
  echo "$NAME rc=$? -> $OUT"
}

# 加法：c3 七成员 + F1 = 8 成员
run e1_add_f1   "guangxi,huanan,jiangxi,$PUB_A,$PUB_B,$G1,$G3,$F1"
# 加法：c3 + F1 + F4 = 9 成员（远场多样性拉满）
run e2_add_f1f4 "guangxi,huanan,jiangxi,$PUB_A,$PUB_B,$G1,$G3,$F1,$F4"
# 加法：c3 + F3(100%远场，与近场成员最互补) = 8 成员
run e3_add_f3   "guangxi,huanan,jiangxi,$PUB_A,$PUB_B,$G1,$G3,$F3"
echo ENS2_DONE_$TAG