#!/bin/bash
# 环境与远端数据集体积探测（只读）
set -u
cd /root/5.6+chanshui1
source env.sh
python - <<'PY'
import torch, seisbench, sys
print("torch", torch.__version__, "hip", getattr(torch.version,"hip",None))
print("avail", torch.cuda.is_available(), "n", torch.cuda.device_count())
if torch.cuda.is_available():
    print("dev0", torch.cuda.get_device_name(0))
    print("mem_GB", round(torch.cuda.get_device_properties(0).total_memory/2**30,1))
print("seisbench", seisbench.__version__)
PY
echo "=== REMOTE SIZES ==="
R=https://hifis-storage.desy.de/Helmholtz/HelmholtzAI/SeisBench/datasets
for p in ethz/waveforms.hdf5 ethz/metadata.csv geofon/waveforms.hdf5 geofon/metadata.csv crew/waveforms000.hdf5 crew/metadata000.csv lendb/waveforms.hdf5 instance/metadata.csv ; do
  sz=$(curl -sI --max-time 40 "$R/$p" | awk 'BEGIN{IGNORECASE=1}/^content-length:/{gsub(/\r/,"");print $2}')
  echo "$p -> ${sz:-NA}"
done