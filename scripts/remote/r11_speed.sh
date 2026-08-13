#!/bin/bash
set -u
R=https://hifis-storage.desy.de/Helmholtz/HelmholtzAI/SeisBench/datasets
echo "=== sizes via Content-Range ==="
for p in ethz/waveforms.hdf5 ethz/metadata.csv geofon/waveforms.hdf5 geofon/metadata.csv \
         crew/metadata000.csv crew/waveforms000.hdf5 \
         lendb/waveforms.hdf5 iquique/waveforms.hdf5 txed/waveforms.hdf5 ; do
  cr=$(curl -s --max-time 45 -r 0-0 -D - -o /dev/null "$R/$p" | awk 'BEGIN{IGNORECASE=1}/^content-range:/{gsub(/\r/,"");print $2}')
  echo "$p -> ${cr:-NA}"
done
echo "=== speed: single stream 20MB from ethz waveforms ==="
/usr/bin/time -f "single 20MB in %e s" curl -s --max-time 120 -r 0-20971519 -o /dev/null "$R/ethz/waveforms.hdf5" 2>&1 | tail -1
echo "=== speed: 6 parallel streams x 10MB ==="
start=$(date +%s.%N)
for i in 0 1 2 3 4 5; do
  lo=$((i*10485760)); hi=$((lo+10485759))
  curl -s --max-time 180 -r ${lo}-${hi} -o /dev/null "$R/ethz/waveforms.hdf5" &
done
wait
end=$(date +%s.%N)
echo "6x10MB=60MB in $(echo "$end - $start" | bc) s"