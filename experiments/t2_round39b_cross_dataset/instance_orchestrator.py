"""INSTANCE 完整性编排器：确保只有在真正完整的 HDF5 上才做审计与建缓存。

修正的缺陷：原 verify_instance_after_extract.py 只要求 HDF5 文件超过 100 GB 即开始抽检。
但解压是流式写入的（bzip2 -dc > file），因此该条件可能在解压进行中为真，
导致对被截断文件抽检"通过"。本编排器改为三重条件：

  1) 压缩包字节数精确等于官方目标；
  2) `bzip2 -t` 对压缩包完整性校验通过（CRC）；
  3) 解压进程已退出，且 HDF5 大小在连续多次采样中保持不变。

只有全部满足才写 ready_for_cache=true 并触发建缓存。全过程只读公开 INSTANCE。
"""
import hashlib
import json
import os
import subprocess
import time

ROOT = '/root/5.6+chanshui1/public_round31'
BZ2 = f'{ROOT}/Instance_events_gm.hdf5.bz2'
H = f'{ROOT}/Instance_events_gm.hdf5'
META = '/root/5.6+chanshui1/public_round30/instance_metadata.csv.bz2'
SPLIT = f'{ROOT}/instance_split.csv'
AUDIT = f'{ROOT}/instance_hdf5_audit.json'
STATE = f'{ROOT}/orchestrator_state.json'
TARGET = 161809684189
POLL = 120

def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)

def size(p):
    return os.path.getsize(p) if os.path.exists(p) else 0

def running(pattern):
    r = subprocess.run(['bash', '-lc', f"ps -ef | grep -F '{pattern}' | grep -v grep | wc -l"],
                       capture_output=True, text=True)
    try:
        return int(r.stdout.strip()) > 0
    except Exception:
        return False

def save(**kw):
    st = {}
    if os.path.exists(STATE):
        try:
            st = json.load(open(STATE))
        except Exception:
            st = {}
    st.update(kw)
    st['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    json.dump(st, open(STATE, 'w'), indent=1)
    return st

# 阶段 1：等待压缩包精确达到目标字节
while size(BZ2) < TARGET:
    save(stage='downloading', bz2_bytes=size(BZ2), fraction=round(size(BZ2) / TARGET, 6))
    log('downloading', size(BZ2), '%.4f' % (size(BZ2) / TARGET))
    time.sleep(POLL)

if size(BZ2) != TARGET:
    save(stage='error', reason=f'bz2 size {size(BZ2)} != {TARGET}')
    raise SystemExit(f'压缩包字节数异常: {size(BZ2)} != {TARGET}')
log('download complete', size(BZ2))
save(stage='download_complete', bz2_bytes=size(BZ2))

# 阶段 2：CRC 完整性校验（bzip2 -t 会读完整个流）
log('bzip2 -t 开始')
t0 = time.time()
r = subprocess.run(['bzip2', '-t', BZ2], capture_output=True, text=True)
ok = (r.returncode == 0)
save(stage='integrity_checked', bzip2_test_rc=r.returncode,
     bzip2_test_secs=round(time.time() - t0, 1), bzip2_test_err=r.stderr[-500:])
log('bzip2 -t rc=', r.returncode, 'secs=%.0f' % (time.time() - t0))
if not ok:
    raise SystemExit('压缩包 CRC 校验失败，拒绝解压与使用: ' + r.stderr[-300:])

# 阶段 3：等待解压完成（进程退出 + 大小稳定）
log('等待解压完成')
stable = 0
last = -1
while True:
    dec_running = running('bzip2 -dc') or running('Instance_events_gm.hdf5.bz2; sync')
    cur = size(H)
    if cur == last and cur > 0 and not dec_running:
        stable += 1
    else:
        stable = 0
    save(stage='extracting', hdf5_bytes=cur, decompressor_running=bool(dec_running), stable_polls=stable)
    log('extract hdf5=%d running=%s stable=%d' % (cur, dec_running, stable))
    if stable >= 3:
        break
    last = cur
    time.sleep(POLL)
log('解压稳定完成', size(H))
save(stage='extract_complete', hdf5_bytes=size(H))

# 阶段 4：真实审计（大样本键/形状/有限值 + 分量与元数据一致性）
import numpy as np
import pandas as pd
import h5py

df = pd.read_csv(META, compression='bz2',
                 usecols=['trace_name', 'source_id', 'source_type', 'source_magnitude',
                          'source_magnitude_type', 'station_network_code', 'station_code',
                          'trace_P_arrival_sample', 'trace_dt_s', 'trace_npts',
                          'trace_E_min_counts', 'trace_N_min_counts', 'trace_Z_min_counts'],
                 low_memory=False)
sp = pd.read_csv(SPLIT)
mp = dict(zip(sp.source_id.astype(str), sp.split.astype(str)))
samp = df.sample(min(400, len(df)), random_state=20260826)
res = dict(hdf5_size=size(H), bz2_size=size(BZ2), metadata_rows=int(len(df)),
           sample_n=int(len(samp)), rows=[])
with h5py.File(H, 'r') as f:
    res['top_keys'] = list(f.keys())
    res['data_group_exists'] = 'data' in f
    g = f['data']
    res['hdf5_n_datasets'] = int(len(g))
    miss = shape_bad = nonfinite = 0
    for _, r in samp.iterrows():
        k = str(r.trace_name)
        if k not in g:
            miss += 1
            continue
        a = g[k][()]
        if tuple(a.shape) != (3, 12000):
            shape_bad += 1
            continue
        if not np.isfinite(a).all():
            nonfinite += 1
            continue
        if len(res['rows']) < 12:
            res['rows'].append(dict(trace_name=k, source_id=str(r.source_id),
                                    split=mp.get(str(r.source_id)), shape=list(a.shape),
                                    dtype=str(a.dtype),
                                    p_sample=(None if pd.isna(r.trace_P_arrival_sample)
                                              else int(r.trace_P_arrival_sample)),
                                    ch_min_counts=[None if pd.isna(r.trace_E_min_counts) else float(r.trace_E_min_counts),
                                                   None if pd.isna(r.trace_N_min_counts) else float(r.trace_N_min_counts),
                                                   None if pd.isna(r.trace_Z_min_counts) else float(r.trace_Z_min_counts)],
                                    ch_actual_min=[float(a[0].min()), float(a[1].min()), float(a[2].min())]))
    res['missing_key'] = miss
    res['shape_mismatch'] = shape_bad
    res['nonfinite'] = nonfinite

# 分量顺序核验：元数据 trace_{E,N,Z}_min_counts 应与 HDF5 通道 0/1/2 最小值一致
match = 0
checked = 0
for row in res['rows']:
    if None in row['ch_min_counts']:
        continue
    checked += 1
    if all(abs(row['ch_min_counts'][i] - row['ch_actual_min'][i]) <= max(1.0, 1e-3 * abs(row['ch_min_counts'][i]))
           for i in range(3)):
        match += 1
res['component_order_checked'] = checked
res['component_order_matches_ENZ'] = match
res['component_order_ok'] = bool(checked > 0 and match == checked)
res['ready_for_cache'] = bool(res['data_group_exists'] and res['missing_key'] == 0 and
                             res['shape_mismatch'] == 0 and res['nonfinite'] == 0 and
                             res['component_order_ok'])
json.dump(res, open(AUDIT, 'w'), indent=2)
save(stage='audit_done', ready_for_cache=res['ready_for_cache'],
     missing_key=res['missing_key'], shape_mismatch=res['shape_mismatch'],
     nonfinite=res['nonfinite'], component_order_ok=res['component_order_ok'])
log('AUDIT ready=%s miss=%d shape=%d nonfinite=%d comp_ok=%s' %
    (res['ready_for_cache'], res['missing_key'], res['shape_mismatch'], res['nonfinite'],
     res['component_order_ok']))
if not res['ready_for_cache']:
    raise SystemExit('审计未通过，不建缓存: ' + json.dumps({k: res[k] for k in
                     ['missing_key', 'shape_mismatch', 'nonfinite', 'component_order_ok']}))

# 阶段 5：建缓存
log('开始建缓存')
save(stage='building_cache')
rc = subprocess.run(['bash', '-lc',
                     f'cd {ROOT} && python -u build_instance_cache.py > build_instance_cache.log 2>&1']).returncode
save(stage='cache_done' if rc == 0 else 'cache_failed', build_rc=rc)
log('build_instance_cache rc=', rc)