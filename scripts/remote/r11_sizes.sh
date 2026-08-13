#!/bin/bash
set -u
R=https://hifis-storage.desy.de/Helmholtz/HelmholtzAI/SeisBench/datasets
for p in ethz/waveforms.hdf5 ethz/metadata.csv geofon/waveforms.hdf5 geofon/metadata.csv \
         crew/chunks crew/metadata000.csv crew/waveforms000.hdf5 \
         lendb/waveforms.hdf5 lendb/metadata.csv \
         instance/metadata.csv iquique/waveforms.hdf5 iquique/metadata.csv \
         txed/waveforms.hdf5 txed/metadata.csv ; do
  h=$(curl -sI --max-time 45 "$R/$p")
  sz=$(printf '%s' "$h" | awk 'BEGIN{IGNORECASE=1}/^content-length:/{gsub(/\r/,"");print $2}')
  code=$(printf '%s' "$h" | head -1 | awk '{print $2}')
  echo "$p  http=$code  bytes=${sz:-NA}"
done