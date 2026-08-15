#!/bin/bash
# 幂等单实例启动器：用 flock 保证同一实验只有一个进程，避免重复进程争抢 GPU。
# 用法: bash launch.sh <tag> <env-assignments...> -- <python-script>
set -u
cd /root/5.6+chanshui1
TAG="$1"; shift
LOCK="/root/5.6+chanshui1/.lock_${TAG}"
LOG="/root/5.6+chanshui1/${TAG}.log"
ENVS=()
while [ "$1" != "--" ]; do ENVS+=("$1"); shift; done
shift
SCRIPT="$1"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "ALREADY_RUNNING $TAG"
  exit 0
fi
flock -u 9
setsid bash -c "exec 9>'$LOCK'; flock -n 9 || exit 0; exec env ${ENVS[*]} python -u '$SCRIPT'" > "$LOG" 2>&1 < /dev/null &
sleep 6
n=$(ps -ef | grep -c "[p]ython -u $SCRIPT")
echo "STARTED $TAG procs=$n log=$LOG"